# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contracts for trusted local external-runtime tuning."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from physics_agent.tuning.types import (
    OptimizationOutput,
    OptimizerSettings,
    TunableParam,
    TuningObjective,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
BASE_EXTERNAL_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "ISAACLAB_PATH",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "PYTHONPATH",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
)


@dataclass(frozen=True)
class ExternalRuntime:
    """Pre-provisioned local Python runtime and trial adapter."""

    python: Path
    script: Path
    cwd: Path
    timeout_s: float = 600.0
    python_args: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    pass_env: tuple[str, ...] = ()
    fingerprint_paths: tuple[Path, ...] = ()
    trial: dict[str, Any] = field(default_factory=dict)
    max_result_bytes: int = 1 * 1024 * 1024
    max_log_bytes: int = 16 * 1024 * 1024
    max_artifact_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.python.is_file():
            raise ValueError(f"external runtime python does not exist: {self.python}")
        if not self.script.is_file():
            raise ValueError(f"external runtime script does not exist: {self.script}")
        if not self.cwd.is_dir():
            raise ValueError(f"external runtime cwd does not exist: {self.cwd}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("external runtime timeout_s must be positive")
        if not self.fingerprint_paths:
            raise ValueError("external runtime fingerprint_paths must not be empty")
        for label, values in (
            ("python_args", self.python_args),
            ("extra_args", self.extra_args),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"external runtime {label} must contain strings")
        for path in self.fingerprint_paths:
            if not path.exists():
                raise ValueError(f"external fingerprint path does not exist: {path}")
        for name in self.pass_env:
            if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"invalid pass_env name: {name!r}")
        try:
            json.dumps(self.trial, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "external runtime trial must contain only finite JSON values"
            ) from exc
        for name, value in (
            ("max_result_bytes", self.max_result_bytes),
            ("max_log_bytes", self.max_log_bytes),
            ("max_artifact_bytes", self.max_artifact_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"external runtime {name} must be a positive integer")


@dataclass(frozen=True)
class QualificationSettings:
    """Nominal parameters required for runtime qualification."""

    nominal_params: dict[str, float]
    seed: int = 1000
    parameter_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if not self.nominal_params:
            raise ValueError("qualification nominal_params must not be empty")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("qualification seed must be non-negative")
        if not math.isfinite(self.parameter_tolerance) or self.parameter_tolerance < 0:
            raise ValueError("parameter_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class QualifiedParameter:
    """Parameter identity and numeric type covered by qualification."""

    name: str
    integer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("qualified parameter name must not be empty")
        if not isinstance(self.integer, bool):
            raise ValueError(
                f"qualified parameter {self.name!r} integer must be a boolean"
            )


@dataclass(frozen=True)
class EvidenceSettings:
    """Qualification PNG-frame and evaluated-rollout recording requirements."""

    artifact_name: str = "frames"
    renderer: str = "isaac_sim_kit_rtx"
    media_type: str = "application/json"
    width: int = 960
    height: int = 720
    fps: float = 30.0
    min_frames: int = 2
    require_motion: bool = True
    min_frame_stddev: float = 1.0
    min_motion_score: float = 0.25
    camera_position: tuple[float, float, float] = (1.5, 1.5, 1.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.2)
    recording_artifact_name: str = "recording_usd"
    playback_renderer: Literal["remote", "ovrtx"] = "ovrtx"
    max_duration_seconds: float = 10.0
    num_sensor_updates: int = 32
    render_mode: str = "rt2"

    def __post_init__(self) -> None:
        if not _ARTIFACT_NAME.fullmatch(self.artifact_name):
            raise ValueError("evidence artifact_name is invalid")
        if not self.renderer.strip():
            raise ValueError("evidence renderer must not be empty")
        if self.media_type != "application/json":
            raise ValueError("evidence media_type must be application/json")
        for name, integer_value in (("width", self.width), ("height", self.height)):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"evidence {name} must be a positive integer")
        if not math.isfinite(self.fps) or self.fps <= 0 or self.fps > 60:
            raise ValueError("evidence fps must be positive and <= 60")
        if (
            isinstance(self.min_frames, bool)
            or not isinstance(self.min_frames, int)
            or self.min_frames < 2
        ):
            raise ValueError("evidence min_frames must be an integer >= 2")
        for name, metric_value in (
            ("min_frame_stddev", self.min_frame_stddev),
            ("min_motion_score", self.min_motion_score),
        ):
            if not math.isfinite(metric_value) or metric_value < 0:
                raise ValueError(f"evidence {name} must be finite and non-negative")
        for name, vector in (
            ("camera_position", self.camera_position),
            ("camera_target", self.camera_target),
        ):
            if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
                raise ValueError(f"evidence {name} must contain three finite numbers")
        if self.camera_position == self.camera_target:
            raise ValueError("evidence camera_position and camera_target must differ")
        if not _ARTIFACT_NAME.fullmatch(self.recording_artifact_name):
            raise ValueError("evidence recording_artifact_name is invalid")
        if self.recording_artifact_name == self.artifact_name:
            raise ValueError("frame manifest and recording artifact names must differ")
        if self.playback_renderer not in {"remote", "ovrtx"}:
            raise ValueError("evidence playback_renderer must be remote or ovrtx")
        if (
            not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("evidence max_duration_seconds must be positive")
        if (
            isinstance(self.num_sensor_updates, bool)
            or not isinstance(self.num_sensor_updates, int)
            or self.num_sensor_updates <= 0
        ):
            raise ValueError("evidence num_sensor_updates must be a positive integer")
        if not isinstance(self.render_mode, str) or not self.render_mode.strip():
            raise ValueError("evidence render_mode must not be empty")

    def request(self) -> dict[str, Any]:
        """Return the adapter-facing qualification PNG-frame request."""

        return {
            "required": True,
            "artifact_name": self.artifact_name,
            "renderer": self.renderer,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "min_frames": self.min_frames,
            "require_motion": self.require_motion,
            "camera": {
                "position": list(self.camera_position),
                "target": list(self.camera_target),
            },
        }

    def recording_request(self) -> dict[str, Any]:
        """Return the adapter-facing evaluated-rollout recording request."""

        return {
            "required": True,
            "artifact_name": self.recording_artifact_name,
            "media_type": "model/vnd.usd",
            "fps": self.fps,
            "camera": {
                "position": list(self.camera_position),
                "target": list(self.camera_target),
            },
        }


@dataclass(frozen=True)
class ExternalTuneSpec:
    """Complete, immutable definition of one external optimization run."""

    task: str
    runtime: ExternalRuntime
    parameter_catalog: tuple[QualifiedParameter, ...]
    params: tuple[TunableParam, ...]
    objective: TuningObjective
    optimizer: OptimizerSettings
    qualification: QualificationSettings
    evidence: EvidenceSettings | None = None
    publish_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("external tuning task must not be empty")
        if not self.parameter_catalog:
            raise ValueError("external parameter catalog must not be empty")
        catalog_names = [parameter.name for parameter in self.parameter_catalog]
        if len(catalog_names) != len(set(catalog_names)):
            raise ValueError("external parameter catalog names must be unique")
        if not self.params:
            raise ValueError("external active search params must not be empty")
        names = [param.name for param in self.params]
        if len(names) != len(set(names)):
            raise ValueError("external tuning parameter names must be unique")
        catalog = {parameter.name: parameter for parameter in self.parameter_catalog}
        unknown_active = sorted(set(names) - set(catalog))
        if unknown_active:
            raise ValueError(
                f"active search has unqualified parameter(s): {unknown_active}"
            )
        for param in self.params:
            if param.integer != catalog[param.name].integer:
                raise ValueError(
                    f"active parameter {param.name!r} numeric type changed"
                )
        nominal_names = set(self.qualification.nominal_params)
        if nominal_names != set(catalog_names):
            raise ValueError(
                "qualification nominal_params must cover the parameter catalog exactly"
            )
        for parameter in self.parameter_catalog:
            value = float(self.qualification.nominal_params[parameter.name])
            if not math.isfinite(value):
                raise ValueError(f"nominal parameter {parameter.name!r} must be finite")
            if parameter.integer and not value.is_integer():
                raise ValueError(
                    f"nominal parameter {parameter.name!r} must be integer"
                )
        for param in self.params:
            if param.min_value >= param.max_value:
                raise ValueError(
                    f"external parameter {param.name!r} bounds must satisfy min < max"
                )
        if self.evidence is None:
            raise ValueError(
                "external tuning requires qualification PNG-frame and rollout "
                "recording settings"
            )
        if len(self.publish_artifacts) != len(set(self.publish_artifacts)):
            raise ValueError("external publish_artifacts must be unique")
        for name in self.publish_artifacts:
            if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
                raise ValueError(f"external publish artifact name is invalid: {name!r}")
        reserved = {
            self.evidence.artifact_name,
            self.evidence.recording_artifact_name,
        }
        overlap = sorted(set(self.publish_artifacts) & reserved)
        if overlap:
            raise ValueError(
                "external publish_artifacts must not duplicate evidence artifacts: "
                f"{overlap}"
            )


@dataclass(kw_only=True)
class ExternalTuneInput:
    """Input for :func:`run_external_tune`."""

    config: Path | dict[str, Any] | ExternalTuneSpec
    output_dir: Path
    approval_digest: str | None = None
    qualification_dir: Path | None = None
    fixed_params: dict[str, float] | None = None
    render_winning_trial: bool = False
    approval_source: str = "local"
    cancel_event: Any = None
    event_listener: Any = None
    verbose: bool = False


@dataclass
class ExternalTuneOutput(OptimizationOutput):
    """Result of qualification or a completed external optimization run."""

    status: str = "failed"
    qualification_digest: str | None = None
    qualification_path: Path | None = None
    rendered_frames: list[Path] = field(default_factory=list)
    render_error: str | None = None
    selected_evidence: dict[str, Any] = field(default_factory=dict)
    published_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None


__all__ = [
    "BASE_EXTERNAL_ENVIRONMENT_NAMES",
    "EvidenceSettings",
    "ExternalRuntime",
    "ExternalTuneInput",
    "ExternalTuneOutput",
    "ExternalTuneSpec",
    "QualifiedParameter",
    "QualificationSettings",
]
