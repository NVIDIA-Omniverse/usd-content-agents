# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-neutral contracts for bounded black-box optimization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

OPTIMIZER_AUTO = "auto"
OPTIMIZER_BOTORCH = "botorch"
OPTIMIZER_RANDOM = "random"
OPTIMIZER_CMA_ES = "cma-es"
SUPPORTED_OPTIMIZERS: tuple[str, ...] = (
    OPTIMIZER_AUTO,
    OPTIMIZER_BOTORCH,
    OPTIMIZER_RANDOM,
    OPTIMIZER_CMA_ES,
)

DEFAULT_JUDGE_PROGRAMMATIC_WEIGHT = 0.60
DEFAULT_JUDGE_VLM_WEIGHT = 0.40


def combine_judge_scores(
    programmatic_score: float,
    vlm_score: float,
    *,
    programmatic_weight: float = DEFAULT_JUDGE_PROGRAMMATIC_WEIGHT,
    vlm_weight: float = DEFAULT_JUDGE_VLM_WEIGHT,
) -> float:
    """Return one normalized programmatic/VLM judge score."""

    scores = {
        "programmatic_score": float(programmatic_score),
        "vlm_score": float(vlm_score),
    }
    for name, value in scores.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    weights = {
        "programmatic_weight": float(programmatic_weight),
        "vlm_weight": float(vlm_weight),
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    scale = max(weights.values())
    if scale <= 0.0:
        raise ValueError("judge score weights must have a positive sum")
    normalized_programmatic_weight = weights["programmatic_weight"] / scale
    normalized_vlm_weight = weights["vlm_weight"] / scale
    total_weight = normalized_programmatic_weight + normalized_vlm_weight
    combined = (
        normalized_programmatic_weight * scores["programmatic_score"]
        + normalized_vlm_weight * scores["vlm_score"]
    ) / total_weight
    return max(0.0, min(1.0, combined))


class BoundedParameter(Protocol):
    """One optimizer dimension expressed as a closed numeric interval."""

    @property
    def name(self) -> str: ...

    @property
    def min_value(self) -> float: ...

    @property
    def max_value(self) -> float: ...

    @property
    def integer(self) -> bool: ...


class BoundedSearchSpace(Protocol):
    """Minimum search-space interface required by the optimizers."""

    @property
    def params(self) -> Sequence[BoundedParameter]: ...


@dataclass(frozen=True)
class TunableParam:
    """A named numeric parameter with closed lower and upper bounds."""

    name: str
    min_value: float
    max_value: float
    integer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tunable parameter name must not be empty")
        if not all(
            math.isfinite(float(value)) for value in (self.min_value, self.max_value)
        ):
            raise ValueError(f"Parameter {self.name!r} bounds must be finite")
        if self.min_value > self.max_value:
            raise ValueError(
                f"Parameter {self.name!r} has min_value > max_value "
                f"({self.min_value} > {self.max_value})"
            )
        if self.integer and (
            not float(self.min_value).is_integer()
            or not float(self.max_value).is_integer()
        ):
            raise ValueError(
                f"Integer parameter {self.name!r} must have integer bounds"
            )

    def clip(self, value: float) -> float:
        """Clip a value into the allowed range."""

        clipped = max(self.min_value, min(self.max_value, float(value)))
        return float(round(clipped)) if self.integer else clipped


@dataclass(frozen=True)
class TuningObjective:
    """One scalar result used to compare completed trials."""

    name: str
    unit: str
    direction: Literal["minimize", "maximize"] = "minimize"
    failure_penalty: float = 1.0e12

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tuning objective name must not be empty")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("tuning objective unit must not be empty")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("objective direction must be minimize or maximize")
        if not math.isfinite(self.failure_penalty) or self.failure_penalty <= 0:
            raise ValueError("objective failure_penalty must be positive")

    def optimizer_score(self, value: float) -> float:
        """Convert the result into the optimizer's lower-is-better score."""

        return float(value) if self.direction == "minimize" else -float(value)


@dataclass(frozen=True)
class OptimizerSettings:
    """Optimizer budget and common per-candidate seed schedule."""

    name: str = "auto"
    max_trials: int = 30
    seed: int = 42
    replicas: int = 1
    replica_seed: int | None = None

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_OPTIMIZERS:
            from .registry import is_optimizer_registered

            if is_optimizer_registered(self.name):
                self._validate_budget()
                return
            raise ValueError(f"unsupported optimizer: {self.name!r}")
        self._validate_budget()

    def _validate_budget(self) -> None:
        if (
            isinstance(self.max_trials, bool)
            or not isinstance(self.max_trials, int)
            or self.max_trials <= 0
        ):
            raise ValueError("optimizer max_trials must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("optimizer seed must be non-negative")
        if (
            isinstance(self.replicas, bool)
            or not isinstance(self.replicas, int)
            or self.replicas <= 0
        ):
            raise ValueError("optimizer replicas must be positive")
        if self.replica_seed is not None and (
            isinstance(self.replica_seed, bool)
            or not isinstance(self.replica_seed, int)
            or self.replica_seed < 0
        ):
            raise ValueError("optimizer replica_seed must be non-negative")

    @property
    def replica_seeds(self) -> tuple[int, ...]:
        """Return common seeds reused for every optimizer candidate."""

        base_seed = self.seed if self.replica_seed is None else self.replica_seed
        return tuple(base_seed + index for index in range(self.replicas))


@dataclass
class ReplicaRecord:
    """One seeded execution contributing to an optimizer trial."""

    seed: int
    objective_value: float | None
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    trial_dir: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible record shape."""

        payload = {
            "seed": self.seed,
            "objective_value": self.objective_value,
            "success": self.success,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "trial_dir": self.trial_dir,
            "error": self.error,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.artifact_metadata:
            payload["artifact_metadata"] = self.artifact_metadata
        return payload


@dataclass
class TrialRecord:
    """One parameter candidate and the score returned by its task."""

    trial_index: int
    params: dict[str, float]
    score: float
    backend_metrics: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    failed: bool = False
    error: str | None = None
    objective_value: float | None = None
    replicas: list[ReplicaRecord] = field(default_factory=list)

    @property
    def optimizer_score(self) -> float:
        """Return the lower-is-better score consumed by the optimizer."""

        return self.score

    @property
    def success(self) -> bool:
        """Whether every execution represented by this trial succeeded."""

        return not self.failed and all(replica.success for replica in self.replicas)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible record shape."""

        return {
            "trial_index": self.trial_index,
            "params": self.params,
            "score": self.score,
            "objective_value": self.objective_value,
            "replicas": [replica.to_dict() for replica in self.replicas],
            "backend_metrics": self.backend_metrics,
            "duration_seconds": self.duration_seconds,
            "failed": self.failed,
            "error": self.error,
        }


__all__ = [
    "BoundedParameter",
    "BoundedSearchSpace",
    "DEFAULT_JUDGE_PROGRAMMATIC_WEIGHT",
    "DEFAULT_JUDGE_VLM_WEIGHT",
    "OPTIMIZER_AUTO",
    "OPTIMIZER_BOTORCH",
    "OPTIMIZER_CMA_ES",
    "OPTIMIZER_RANDOM",
    "OptimizerSettings",
    "ReplicaRecord",
    "SUPPORTED_OPTIMIZERS",
    "TrialRecord",
    "TunableParam",
    "TuningObjective",
    "combine_judge_scores",
]
