# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics compatibility wrappers around the shared optimizers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from world_understanding.optimization import optimizers as _shared
from world_understanding.optimization.errors import OptimizerUnavailableError

from .errors import BoTorchUnavailableError, TuningError
from .search_space import BoundedParameter, BoundedSearchSpace
from .types import Scenario

OPTIMIZER_AUTO = _shared.OPTIMIZER_AUTO
OPTIMIZER_BOTORCH = _shared.OPTIMIZER_BOTORCH
OPTIMIZER_RANDOM = _shared.OPTIMIZER_RANDOM
OPTIMIZER_CMA_ES = _shared.OPTIMIZER_CMA_ES
SUPPORTED_OPTIMIZERS = _shared.SUPPORTED_OPTIMIZERS
EvaluateFn = _shared.EvaluateFn

FrictionPair = tuple[
    int,
    BoundedParameter,
    int,
    BoundedParameter,
]
FrictionInequality = tuple[NDArray[np.int64], NDArray[np.float64], float]


def _params_from_vector(
    search_space: BoundedSearchSpace,
    vector: np.ndarray,
) -> dict[str, float]:
    """Decode each Physics parameter independently from the unit cube."""

    return _shared.params_from_vector(search_space, vector)


def _vector_from_params(
    search_space: BoundedSearchSpace,
    params: dict[str, float],
) -> NDArray[np.float64]:
    """Encode each Physics parameter independently into the unit cube."""

    return _shared.vector_from_params(search_space, params)


def _friction_pair(
    search_space: BoundedSearchSpace,
) -> FrictionPair | None:
    """Return indexed built-in static/dynamic friction parameters when tuned."""
    if not isinstance(search_space, Scenario):
        return None
    indexed = {
        param.name: (index, param) for index, param in enumerate(search_space.params)
    }
    static = indexed.get("static_friction")
    dynamic = indexed.get("dynamic_friction")
    if static is None or dynamic is None:
        return None
    static_index, static_param = static
    dynamic_index, dynamic_param = dynamic
    if dynamic_param.min_value > static_param.max_value:
        raise ValueError(
            "dynamic_friction minimum must not exceed static_friction maximum"
        )
    return static_index, static_param, dynamic_index, dynamic_param


def _single_point_friction_features(
    search_space: BoundedSearchSpace,
) -> dict[int, float] | None:
    """Return fixed unit coordinates when only one friction pair is feasible."""
    pair = _friction_pair(search_space)
    if pair is None:
        return None
    static_index, static_param, dynamic_index, dynamic_param = pair
    if static_param.max_value != dynamic_param.min_value:
        return None

    static_range = static_param.max_value - static_param.min_value
    return {
        static_index: 0.0 if static_range == 0.0 else 1.0,
        dynamic_index: 0.0,
    }


def _fixed_unit_features(search_space: BoundedSearchSpace) -> dict[int, float]:
    """Return optimizer coordinates that have no physical freedom."""
    fixed = {
        index: 0.0
        for index, param in enumerate(search_space.params)
        if param.min_value == param.max_value
    }
    friction_features = _single_point_friction_features(search_space)
    if friction_features is not None:
        fixed.update(friction_features)
    return fixed


def _friction_inequality_spec(
    search_space: BoundedSearchSpace,
) -> FrictionInequality | None:
    """Express ``static_friction >= dynamic_friction`` in unit-cube space."""
    pair = _friction_pair(search_space)
    if pair is None:
        return None

    static_index, static_param, dynamic_index, dynamic_param = pair
    static_range = static_param.max_value - static_param.min_value
    dynamic_range = dynamic_param.max_value - dynamic_param.min_value

    indices: list[int] = []
    coefficients: list[float] = []
    if static_range != 0.0:
        indices.append(static_index)
        coefficients.append(static_range)
    if dynamic_range != 0.0:
        indices.append(dynamic_index)
        coefficients.append(-dynamic_range)
    if not indices:
        return None

    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(coefficients, dtype=np.float64),
        dynamic_param.min_value - static_param.min_value,
    )


def _is_friction_feasible(
    search_space: BoundedSearchSpace,
    vector: np.ndarray,
) -> bool:
    """Return whether a candidate satisfies the Physics friction invariant."""
    if _friction_pair(search_space) is None:
        return True
    params = _params_from_vector(search_space, vector)
    return params["dynamic_friction"] <= params["static_friction"]


def _friction_tolerance(pair: FrictionPair) -> float:
    """Return a physical-unit tolerance for numerical optimizer output."""
    _, static_param, _, dynamic_param = pair
    scale = max(
        1.0,
        abs(static_param.min_value),
        abs(static_param.max_value),
        abs(dynamic_param.min_value),
        abs(dynamic_param.max_value),
        static_param.max_value - static_param.min_value,
        dynamic_param.max_value - dynamic_param.min_value,
    )
    return float(np.sqrt(np.finfo(np.float64).eps) * scale)


def _repair_numerical_friction_overshoot(
    search_space: BoundedSearchSpace,
    vector: np.ndarray,
) -> NDArray[np.float64] | None:
    """Project solver-scale friction overshoot into the feasible half-space."""
    candidate = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)
    pair = _friction_pair(search_space)
    if pair is None or _is_friction_feasible(search_space, candidate):
        return candidate

    params = _params_from_vector(search_space, candidate)
    violation = params["dynamic_friction"] - params["static_friction"]
    tolerance = _friction_tolerance(pair)
    if violation > tolerance:
        return None

    constraint = _friction_inequality_spec(search_space)
    if constraint is None:  # pragma: no cover - fixed valid pairs return above
        return None
    indices, coefficients, rhs = constraint
    lhs = float(np.dot(coefficients, candidate[indices]))
    norm_squared = float(np.dot(coefficients, coefficients))
    if norm_squared == 0.0:  # pragma: no cover - nonempty coefficients are nonzero
        return None

    # Project to the boundary, then move a few unit-cube ULPs farther along the
    # normalized constraint normal so independent decoding is strictly feasible.
    norm = float(np.sqrt(norm_squared))
    distance = max(0.0, rhs - lhs) / norm
    unit_margin = 8.0 * np.finfo(np.float64).eps
    repaired = candidate.copy()
    repaired[indices] += (distance + unit_margin) * coefficients / norm
    repaired = np.clip(repaired, 0.0, 1.0)
    return repaired if _is_friction_feasible(search_space, repaired) else None


def _sample_feasible_vector(
    search_space: BoundedSearchSpace,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw directly from the feasible region without changing vector decoding."""
    vector = np.asarray(rng.random(len(search_space.params)), dtype=np.float64)
    pair = _friction_pair(search_space)
    if pair is None:
        return vector

    static_index, static_param, dynamic_index, dynamic_param = pair
    static_unit = vector[static_index]
    dynamic_unit = vector[dynamic_index]

    static_lower = max(static_param.min_value, dynamic_param.min_value)
    static_value = static_lower + static_unit * (static_param.max_value - static_lower)
    dynamic_upper = min(dynamic_param.max_value, static_value)
    dynamic_value = dynamic_param.min_value + dynamic_unit * (
        dynamic_upper - dynamic_param.min_value
    )

    static_range = static_param.max_value - static_param.min_value
    dynamic_range = dynamic_param.max_value - dynamic_param.min_value
    vector[static_index] = (
        0.0
        if static_range == 0.0
        else (static_value - static_param.min_value) / static_range
    )
    vector[dynamic_index] = (
        0.0
        if dynamic_range == 0.0
        else (dynamic_value - dynamic_param.min_value) / dynamic_range
    )
    if not _is_friction_feasible(search_space, vector):  # pragma: no cover - invariant
        raise RuntimeError("Direct friction sampler produced an infeasible candidate")
    return vector


def _finite_failed_trial_penalty(observed_scores: list[float]) -> float:
    """Retain the Physics compatibility helper for existing callers and tests."""
    return _shared.finite_failed_trial_penalty(observed_scores)


def is_botorch_available() -> bool:
    """Return whether the optional Physics tuning dependencies are installed."""

    return _shared.is_botorch_available()


def resolve_optimizer(name: str) -> str:
    """Resolve ``auto`` while preserving Physics Agent's install-hint error."""

    if name == OPTIMIZER_AUTO:
        if not is_botorch_available():
            raise BoTorchUnavailableError()
        return OPTIMIZER_BOTORCH
    if name == OPTIMIZER_BOTORCH and not is_botorch_available():
        raise BoTorchUnavailableError()
    try:
        return _shared.resolve_optimizer(name)
    except OptimizerUnavailableError as error:
        raise TuningError(str(error)) from error


def get_supported_optimizer_names() -> tuple[str, ...]:
    """Return built-in and installed optimizer extension names."""

    return _shared.get_supported_optimizer_names()


def run_random_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run shared random search with Physics feasibility sampling."""

    _shared.run_random_optimizer(
        search_space,
        evaluate,
        max_trials=max_trials,
        seed=seed,
        cancel_check=cancel_check,
        candidate_decoder=_params_from_vector,
        candidate_sampler=_sample_feasible_vector,
        candidate_feasibility=_is_friction_feasible,
        fixed_features=_fixed_unit_features(search_space),
    )


def run_cma_es_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run shared CMA-ES with Physics feasibility handling."""

    _shared.run_cma_es_optimizer(
        search_space,
        evaluate,
        max_trials=max_trials,
        seed=seed,
        cancel_check=cancel_check,
        candidate_decoder=_params_from_vector,
        candidate_sampler=_sample_feasible_vector,
        candidate_feasibility=_is_friction_feasible,
        fixed_features=_single_point_friction_features(search_space),
    )


def run_botorch_optimizer(
    search_space: BoundedSearchSpace,
    evaluate: EvaluateFn,
    *,
    max_trials: int,
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Run shared BoTorch with Physics constraints and install-hint errors."""

    constraint = (
        None
        if _single_point_friction_features(search_space) is not None
        else _friction_inequality_spec(search_space)
    )
    inequalities = None if constraint is None else (constraint,)
    try:
        _shared.run_botorch_optimizer(
            search_space,
            evaluate,
            max_trials=max_trials,
            seed=seed,
            cancel_check=cancel_check,
            candidate_decoder=_params_from_vector,
            candidate_sampler=_sample_feasible_vector,
            candidate_feasibility=_is_friction_feasible,
            candidate_repair=_repair_numerical_friction_overshoot,
            inequality_constraints=inequalities,
            fixed_features=_fixed_unit_features(search_space),
        )
    except OptimizerUnavailableError as error:
        raise BoTorchUnavailableError() from error


def get_runner(name: str) -> Callable[..., None]:
    """Return the optimizer entry-point keyed by ``name``.

    Caller must pass a *resolved* optimizer name (i.e. ``auto`` already mapped).
    """
    if name == OPTIMIZER_BOTORCH:
        return run_botorch_optimizer
    if name == OPTIMIZER_RANDOM:
        return run_random_optimizer
    if name == OPTIMIZER_CMA_ES:
        return run_cma_es_optimizer
    return _shared.get_runner(name)


__all__ = [
    "OPTIMIZER_AUTO",
    "OPTIMIZER_BOTORCH",
    "OPTIMIZER_RANDOM",
    "OPTIMIZER_CMA_ES",
    "SUPPORTED_OPTIMIZERS",
    "EvaluateFn",
    "resolve_optimizer",
    "is_botorch_available",
    "get_runner",
    "get_supported_optimizer_names",
    "run_random_optimizer",
    "run_cma_es_optimizer",
    "run_botorch_optimizer",
]
