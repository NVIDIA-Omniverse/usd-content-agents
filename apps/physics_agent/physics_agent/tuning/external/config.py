# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed YAML/JSON loader for external tuning specifications."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from physics_agent.tuning.errors import TuningConfigError
from physics_agent.tuning.types import OptimizerSettings, TunableParam, TuningObjective

from .types import (
    EvidenceSettings,
    ExternalRuntime,
    ExternalTuneSpec,
    QualificationSettings,
    QualifiedParameter,
)


class ExternalTuneConfigError(TuningConfigError):
    """The external tuning configuration is ambiguous or invalid."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalTuneConfigError(f"{label} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ExternalTuneConfigError(
            f"{label} has unknown field(s): {sorted(unknown)}"
        )


def _path(
    value: Any,
    label: str,
    base_dir: Path,
    *,
    resolve_symlinks: bool = True,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ExternalTuneConfigError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve() if resolve_symlinks else candidate.absolute()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExternalTuneConfigError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ExternalTuneConfigError(f"{label} must be a finite number")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalTuneConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExternalTuneConfigError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ExternalTuneConfigError(f"{label} must be a boolean")
    return value


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ExternalTuneConfigError(f"{label} must contain three numbers")
    return cast(
        tuple[float, float, float],
        tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value)),
    )


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ExternalTuneConfigError(f"{label} must be a list of non-empty strings")
    return tuple(value)


def _load_raw(source: Path | Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    if isinstance(source, Path):
        path = source.expanduser().resolve()
        if not path.is_file():
            raise ExternalTuneConfigError(
                f"external tuning config does not exist: {path}"
            )
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ExternalTuneConfigError(
                f"cannot read external tuning config {path}: {exc}"
            ) from exc
        return _object(raw, str(path)), path.parent
    return _object(copy.deepcopy(source), "external tuning config"), Path.cwd()


def load_external_tune_spec(
    source: Path | Mapping[str, Any] | ExternalTuneSpec,
) -> ExternalTuneSpec:
    """Load and validate a complete external tuning specification."""

    if isinstance(source, ExternalTuneSpec):
        return source
    raw, base_dir = _load_raw(source)
    _reject_unknown(
        raw,
        {
            "schema_version",
            "task",
            "runtime",
            "parameters",
            "objective",
            "optimizer",
            "qualification",
            "evidence",
            "publish_artifacts",
        },
        "external tuning config",
    )
    if raw.get("schema_version", 1) != 1:
        raise ExternalTuneConfigError("schema_version must be 1")
    task = raw.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ExternalTuneConfigError("task must be a non-empty string")

    runtime_raw = _object(raw.get("runtime"), "runtime")
    _reject_unknown(
        runtime_raw,
        {
            "python",
            "script",
            "cwd",
            "timeout_s",
            "python_args",
            "extra_args",
            "pass_env",
            "fingerprint_paths",
            "trial",
            "max_result_bytes",
            "max_log_bytes",
            "max_artifact_bytes",
        },
        "runtime",
    )
    fingerprint_raw = runtime_raw.get("fingerprint_paths")
    if not isinstance(fingerprint_raw, list) or not fingerprint_raw:
        raise ExternalTuneConfigError(
            "runtime.fingerprint_paths must be a non-empty list"
        )
    fingerprint_paths = tuple(
        _path(value, f"runtime.fingerprint_paths[{index}]", base_dir)
        for index, value in enumerate(fingerprint_raw)
    )
    trial = runtime_raw.get("trial", {})
    if not isinstance(trial, Mapping):
        raise ExternalTuneConfigError("runtime.trial must be an object")
    trial_config = copy.deepcopy(dict(trial))
    try:
        json.dumps(trial_config, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExternalTuneConfigError(
            "runtime.trial must contain only finite JSON-serializable values"
        ) from exc
    try:
        runtime = ExternalRuntime(
            # Resolving a venv's bin/python symlink changes sys.prefix and
            # silently escapes the operator-selected environment.
            python=_path(
                runtime_raw.get("python"),
                "runtime.python",
                base_dir,
                resolve_symlinks=False,
            ),
            script=_path(runtime_raw.get("script"), "runtime.script", base_dir),
            cwd=_path(runtime_raw.get("cwd", "."), "runtime.cwd", base_dir),
            timeout_s=_finite(runtime_raw.get("timeout_s", 600), "runtime.timeout_s"),
            python_args=_strings(
                runtime_raw.get("python_args", []), "runtime.python_args"
            ),
            extra_args=_strings(
                runtime_raw.get("extra_args", []), "runtime.extra_args"
            ),
            pass_env=_strings(runtime_raw.get("pass_env", []), "runtime.pass_env"),
            fingerprint_paths=fingerprint_paths,
            trial=trial_config,
            max_result_bytes=_integer(
                runtime_raw.get("max_result_bytes", 1 * 1024 * 1024),
                "runtime.max_result_bytes",
                minimum=1,
            ),
            max_log_bytes=_integer(
                runtime_raw.get("max_log_bytes", 16 * 1024 * 1024),
                "runtime.max_log_bytes",
                minimum=1,
            ),
            max_artifact_bytes=_integer(
                runtime_raw.get("max_artifact_bytes", 256 * 1024 * 1024),
                "runtime.max_artifact_bytes",
                minimum=1,
            ),
        )
    except ValueError as exc:
        raise ExternalTuneConfigError(str(exc)) from exc

    parameters_raw = _object(raw.get("parameters"), "parameters")
    if not parameters_raw:
        raise ExternalTuneConfigError("parameters must not be empty")
    parameters: list[TunableParam] = []
    parameter_catalog: list[QualifiedParameter] = []
    for name, value in parameters_raw.items():
        if not isinstance(name, str) or not name:
            raise ExternalTuneConfigError("parameter names must be non-empty strings")
        parameter_raw = _object(value, f"parameters.{name}")
        _reject_unknown(parameter_raw, {"min", "max", "integer"}, f"parameters.{name}")
        integer = parameter_raw.get("integer", False)
        if not isinstance(integer, bool):
            raise ExternalTuneConfigError(
                f"parameters.{name}.integer must be a boolean"
            )
        try:
            parameter_catalog.append(QualifiedParameter(name=name, integer=integer))
            parameters.append(
                TunableParam(
                    name=name,
                    min_value=_finite(
                        parameter_raw.get("min"), f"parameters.{name}.min"
                    ),
                    max_value=_finite(
                        parameter_raw.get("max"), f"parameters.{name}.max"
                    ),
                    integer=integer,
                )
            )
        except ValueError as exc:
            raise ExternalTuneConfigError(str(exc)) from exc

    objective_raw = _object(raw.get("objective"), "objective")
    _reject_unknown(
        objective_raw,
        {"name", "unit", "direction", "failure_penalty"},
        "objective",
    )
    direction = _string(
        objective_raw.get("direction", "minimize"), "objective.direction"
    )
    if direction not in {"minimize", "maximize"}:
        raise ExternalTuneConfigError(
            "objective.direction must be minimize or maximize"
        )
    try:
        objective = TuningObjective(
            name=_string(objective_raw.get("name"), "objective.name"),
            unit=_string(objective_raw.get("unit"), "objective.unit"),
            direction=cast(Literal["minimize", "maximize"], direction),
            failure_penalty=_finite(
                objective_raw.get("failure_penalty", 1.0e12),
                "objective.failure_penalty",
            ),
        )
    except ValueError as exc:
        raise ExternalTuneConfigError(str(exc)) from exc

    optimizer_raw = _object(raw.get("optimizer", {}), "optimizer")
    _reject_unknown(
        optimizer_raw,
        {"name", "max_trials", "seed", "replicas", "replica_seed"},
        "optimizer",
    )
    try:
        optimizer = OptimizerSettings(
            name=_string(optimizer_raw.get("name", "auto"), "optimizer.name"),
            max_trials=_integer(
                optimizer_raw.get("max_trials", 30),
                "optimizer.max_trials",
                minimum=1,
            ),
            seed=_integer(optimizer_raw.get("seed", 42), "optimizer.seed"),
            replicas=_integer(
                optimizer_raw.get("replicas", 1),
                "optimizer.replicas",
                minimum=1,
            ),
            replica_seed=(
                _integer(optimizer_raw.get("replica_seed"), "optimizer.replica_seed")
                if optimizer_raw.get("replica_seed") is not None
                else None
            ),
        )
    except ValueError as exc:
        raise ExternalTuneConfigError(str(exc)) from exc

    qualification_raw = _object(raw.get("qualification"), "qualification")
    _reject_unknown(
        qualification_raw,
        {
            "nominal_params",
            "seed",
            "parameter_tolerance",
        },
        "qualification",
    )
    nominal_raw = _object(
        qualification_raw.get("nominal_params"), "qualification.nominal_params"
    )
    nominal: dict[str, float] = {}
    for name, value in nominal_raw.items():
        if not isinstance(name, str) or not name:
            raise ExternalTuneConfigError(
                "qualification.nominal_params keys must be non-empty strings"
            )
        nominal[name] = _finite(value, f"qualification.nominal_params.{name}")
    try:
        qualification = QualificationSettings(
            nominal_params=nominal,
            seed=_integer(qualification_raw.get("seed", 1000), "qualification.seed"),
            parameter_tolerance=_finite(
                qualification_raw.get("parameter_tolerance", 1.0e-9),
                "qualification.parameter_tolerance",
            ),
        )
        evidence: EvidenceSettings | None = None
        if raw.get("evidence") is not None:
            evidence_raw = _object(raw.get("evidence"), "evidence")
            _reject_unknown(
                evidence_raw,
                {
                    "artifact_name",
                    "renderer",
                    "media_type",
                    "width",
                    "height",
                    "fps",
                    "min_frames",
                    "require_motion",
                    "min_frame_stddev",
                    "min_motion_score",
                    "camera",
                    "recording_artifact_name",
                    "playback_renderer",
                    "max_duration_seconds",
                    "num_sensor_updates",
                    "render_mode",
                },
                "evidence",
            )
            camera_raw = _object(evidence_raw.get("camera", {}), "evidence.camera")
            _reject_unknown(camera_raw, {"position", "target"}, "evidence.camera")
            evidence = EvidenceSettings(
                artifact_name=_string(
                    evidence_raw.get("artifact_name", "frames"),
                    "evidence.artifact_name",
                ),
                renderer=_string(
                    evidence_raw.get("renderer", "isaac_sim_kit_rtx"),
                    "evidence.renderer",
                ),
                media_type=_string(
                    evidence_raw.get("media_type", "application/json"),
                    "evidence.media_type",
                ),
                width=_integer(
                    evidence_raw.get("width", 960), "evidence.width", minimum=1
                ),
                height=_integer(
                    evidence_raw.get("height", 720), "evidence.height", minimum=1
                ),
                fps=_finite(evidence_raw.get("fps", 30.0), "evidence.fps"),
                min_frames=_integer(
                    evidence_raw.get("min_frames", 2),
                    "evidence.min_frames",
                    minimum=2,
                ),
                require_motion=_boolean(
                    evidence_raw.get("require_motion", True),
                    "evidence.require_motion",
                ),
                min_frame_stddev=_finite(
                    evidence_raw.get("min_frame_stddev", 1.0),
                    "evidence.min_frame_stddev",
                ),
                min_motion_score=_finite(
                    evidence_raw.get("min_motion_score", 0.25),
                    "evidence.min_motion_score",
                ),
                camera_position=_vector3(
                    camera_raw.get("position", [1.5, 1.5, 1.0]),
                    "evidence.camera.position",
                ),
                camera_target=_vector3(
                    camera_raw.get("target", [0.0, 0.0, 0.2]),
                    "evidence.camera.target",
                ),
                recording_artifact_name=_string(
                    evidence_raw.get("recording_artifact_name", "recording_usd"),
                    "evidence.recording_artifact_name",
                ),
                playback_renderer=cast(
                    Literal["remote", "ovrtx"],
                    _string(
                        evidence_raw.get("playback_renderer", "ovrtx"),
                        "evidence.playback_renderer",
                    ),
                ),
                max_duration_seconds=_finite(
                    evidence_raw.get("max_duration_seconds", 10.0),
                    "evidence.max_duration_seconds",
                ),
                num_sensor_updates=_integer(
                    evidence_raw.get("num_sensor_updates", 32),
                    "evidence.num_sensor_updates",
                    minimum=1,
                ),
                render_mode=_string(
                    evidence_raw.get("render_mode", "rt2"),
                    "evidence.render_mode",
                ),
            )
        return ExternalTuneSpec(
            task=task.strip(),
            runtime=runtime,
            parameter_catalog=tuple(parameter_catalog),
            params=tuple(parameters),
            objective=objective,
            optimizer=optimizer,
            qualification=qualification,
            evidence=evidence,
            publish_artifacts=_strings(
                raw.get("publish_artifacts", []), "publish_artifacts"
            ),
        )
    except ValueError as exc:
        raise ExternalTuneConfigError(str(exc)) from exc


__all__ = ["ExternalTuneConfigError", "load_external_tune_spec"]
