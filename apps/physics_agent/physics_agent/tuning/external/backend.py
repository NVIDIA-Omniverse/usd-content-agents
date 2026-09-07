# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess adapter for trusted local customer simulation runtimes."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from physics_agent.tuning.errors import TuningCancelledError, TuningError
from physics_agent.tuning.types import ReplicaRecord

from .evidence import (
    EvidenceValidationError,
    inspect_file_artifact,
    inspect_frame_evidence,
    inspect_usd_recording,
)
from .types import BASE_EXTERNAL_ENVIRONMENT_NAMES, ExternalTuneSpec


class ExternalTrialError(TuningError):
    """An external runtime failed transport or result validation."""


def _is_cancelled(cancel_event: Any) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _artifact_path(value: Any, *, name: str, root: Path) -> Path:
    declared = value.get("path") if isinstance(value, dict) else value
    if not isinstance(declared, str) or not declared:
        raise ExternalTrialError(f"artifact {name!r} must declare a path")
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExternalTrialError(f"artifact {name!r} does not exist: {path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ExternalTrialError(
            f"artifact {name!r} escapes the trial artifacts directory"
        ) from None
    if not resolved.is_file():
        raise ExternalTrialError(f"artifact {name!r} is not a regular file")
    return resolved


class ExternalTrialExecutor:
    """Execute and validate one subprocess for each candidate replica."""

    def __init__(
        self,
        spec: ExternalTuneSpec,
        output_dir: Path,
        *,
        cancel_event: Any = None,
    ) -> None:
        self.spec = spec
        self.output_dir = output_dir.resolve()
        self.cancel_event = cancel_event

    def _environment(self) -> dict[str, str]:
        allowed = BASE_EXTERNAL_ENVIRONMENT_NAMES | set(self.spec.runtime.pass_env)
        environment = {
            name: value for name, value in os.environ.items() if name in allowed
        }
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def _run_process(
        self,
        argv: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        started = time.monotonic()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(self.spec.runtime.cwd),
                    env=self._environment(),
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ExternalTrialError(
                    f"cannot start external trial process: {exc}"
                ) from exc
            try:
                while process.poll() is None:
                    if _is_cancelled(self.cancel_event):
                        _terminate_group(process)
                        raise TuningCancelledError("external trial cancelled")
                    if time.monotonic() - started > self.spec.runtime.timeout_s:
                        _terminate_group(process)
                        raise ExternalTrialError(
                            "external trial timed out after "
                            f"{self.spec.runtime.timeout_s:g}s"
                        )
                    stdout.flush()
                    stderr.flush()
                    log_bytes = stdout_path.stat().st_size + stderr_path.stat().st_size
                    if log_bytes > self.spec.runtime.max_log_bytes:
                        _terminate_group(process)
                        raise ExternalTrialError(
                            "external trial exceeded its combined log byte limit"
                        )
                    time.sleep(0.05)
                stdout.flush()
                stderr.flush()
                log_bytes = stdout_path.stat().st_size + stderr_path.stat().st_size
                if log_bytes > self.spec.runtime.max_log_bytes:
                    raise ExternalTrialError(
                        "external trial exceeded its combined log byte limit"
                    )
                return int(process.returncode or 0)
            except BaseException:
                _terminate_group(process)
                raise

    def _trial_dir(self, purpose: str, index: int, seed: int) -> Path:
        base = self.output_dir / purpose / f"trial_{index:04d}_seed_{seed}"
        if not base.exists():
            return base
        attempt = 2
        while True:
            candidate = base.with_name(f"{base.name}_attempt_{attempt}")
            if not candidate.exists():
                return candidate
            attempt += 1

    def run(
        self,
        params: dict[str, float],
        seed: int,
        *,
        purpose: str,
        index: int,
    ) -> ReplicaRecord:
        """Execute and validate one adapter request."""

        expected_names = {parameter.name for parameter in self.spec.parameter_catalog}
        if set(params) != expected_names:
            raise ExternalTrialError(
                "external trial params must cover configured parameters exactly"
            )
        active_bounds = {parameter.name: parameter for parameter in self.spec.params}
        catalog = {
            parameter.name: parameter for parameter in self.spec.parameter_catalog
        }
        for name, parameter in catalog.items():
            value = params[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or (parameter.integer and not float(value).is_integer())
            ):
                raise ExternalTrialError(
                    f"external trial parameter {name!r} is invalid"
                )
            bounds = active_bounds.get(name)
            if (
                purpose == "optimization"
                and bounds is not None
                and (float(value) < bounds.min_value or float(value) > bounds.max_value)
            ):
                raise ExternalTrialError(
                    f"external trial parameter {name!r} is outside active bounds"
                )

        trial_dir = self._trial_dir(purpose, index, seed)
        artifacts_dir = trial_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=False)
        request_path = trial_dir / "request.json"
        result_path = trial_dir / "result.json"
        stdout_path = trial_dir / "stdout.log"
        stderr_path = trial_dir / "stderr.log"
        request = {
            "schema_version": 1,
            "purpose": purpose,
            "task": self.spec.task,
            "params": {name: float(value) for name, value in params.items()},
            "seed": int(seed),
            "output_dir": str(trial_dir),
            "artifacts_dir": str(artifacts_dir),
            "trial": self.spec.runtime.trial,
        }
        request["objective"] = {
            "name": self.spec.objective.name,
            "unit": self.spec.objective.unit,
            "direction": self.spec.objective.direction,
        }
        if self.spec.evidence is not None and purpose == "qualification":
            request["evidence"] = self.spec.evidence.request()
        if self.spec.evidence is not None and purpose == "optimization":
            request["recording"] = self.spec.evidence.recording_request()
            if self.spec.publish_artifacts:
                request["publish_artifacts"] = list(self.spec.publish_artifacts)
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        argv = [
            str(self.spec.runtime.python),
            *self.spec.runtime.python_args,
            str(self.spec.runtime.script),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
            *self.spec.runtime.extra_args,
        ]
        returncode = self._run_process(
            argv, stdout_path=stdout_path, stderr_path=stderr_path
        )
        if returncode != 0:
            raise ExternalTrialError(f"external trial exited with status {returncode}")
        if not result_path.is_file():
            raise ExternalTrialError("external trial did not write result.json")
        if result_path.stat().st_size > self.spec.runtime.max_result_bytes:
            raise ExternalTrialError("external trial result exceeded its byte limit")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalTrialError(
                f"external trial result is invalid JSON: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise ExternalTrialError("external trial result must be an object")
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ExternalTrialError(
                "external trial result must contain only finite JSON values"
            ) from exc
        if result.get("status") != "ok":
            raise ExternalTrialError(
                f"external trial reported status {result.get('status')!r}"
            )
        success = result.get("success")
        if not isinstance(success, bool):
            raise ExternalTrialError("external trial success must be a boolean")
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise ExternalTrialError("external trial metadata must be an object")
        applied = metadata.get("applied_params")
        if not isinstance(applied, dict) or set(applied) != set(params):
            raise ExternalTrialError(
                "external trial metadata.applied_params must cover requested params exactly"
            )
        tolerance = self.spec.qualification.parameter_tolerance
        for name, requested in params.items():
            actual = applied.get(name)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int | float)
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    float(requested),
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
            ):
                raise ExternalTrialError(
                    f"external trial did not apply requested parameter {name!r}"
                )
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ExternalTrialError("external trial metrics must be an object")
        if success:
            objective = result.get("objective")
            if not isinstance(objective, dict):
                raise ExternalTrialError("external trial objective must be an object")
            expected_objective = {
                "name": self.spec.objective.name,
                "unit": self.spec.objective.unit,
                "direction": self.spec.objective.direction,
            }
            identity = {key: objective.get(key) for key in expected_objective}
            if identity != expected_objective:
                raise ExternalTrialError(
                    "external trial objective identity changed: "
                    f"expected {expected_objective}, got {identity}"
                )
            objective_value = objective.get("value")
            if (
                isinstance(objective_value, bool)
                or not isinstance(objective_value, int | float)
                or not math.isfinite(float(objective_value))
            ):
                raise ExternalTrialError(
                    "external trial objective.value must be finite"
                )
            resolved_objective = float(objective_value)
        else:
            resolved_objective = None
        artifacts_raw = result.get("artifacts", {})
        if not isinstance(artifacts_raw, dict):
            raise ExternalTrialError("external trial artifacts must be an object")
        declared_artifacts = {
            name: _artifact_path(raw, name=name, root=artifacts_dir.resolve())
            for name, raw in artifacts_raw.items()
        }
        artifact_files = [
            path.resolve() for path in artifacts_dir.rglob("*") if path.is_file()
        ]
        for path in artifact_files:
            try:
                path.relative_to(artifacts_dir.resolve())
            except ValueError:
                raise ExternalTrialError(
                    "external trial artifact resolves outside artifacts_dir"
                ) from None
        artifact_bytes = sum(path.stat().st_size for path in artifact_files)
        if artifact_bytes > self.spec.runtime.max_artifact_bytes:
            raise ExternalTrialError(
                "external trial artifacts exceeded their byte limit"
            )
        relative_artifacts = {
            name: str(path.relative_to(self.output_dir))
            for name, path in declared_artifacts.items()
        }
        artifact_metadata: dict[str, dict[str, Any]] = {}
        if self.spec.evidence is not None and purpose == "qualification":
            evidence_name = self.spec.evidence.artifact_name
            evidence_path = declared_artifacts.get(evidence_name)
            if evidence_path is None:
                raise ExternalTrialError(
                    f"external trial must declare evidence artifact {evidence_name!r}"
                )
            try:
                artifact_metadata[evidence_name] = inspect_frame_evidence(
                    evidence_path,
                    self.spec.evidence,
                    metadata,
                    relative_path=relative_artifacts[evidence_name],
                )
            except EvidenceValidationError as exc:
                raise ExternalTrialError(str(exc)) from exc
        if self.spec.evidence is not None and success and purpose == "optimization":
            recording_name = self.spec.evidence.recording_artifact_name
            recording_path = declared_artifacts.get(recording_name)
            if recording_path is None:
                raise ExternalTrialError(
                    "successful optimization trial must declare rollout recording "
                    f"artifact {recording_name!r}"
                )
            try:
                artifact_metadata[recording_name] = inspect_usd_recording(
                    recording_path,
                    self.spec.evidence,
                    relative_path=relative_artifacts[recording_name],
                )
            except EvidenceValidationError as exc:
                raise ExternalTrialError(str(exc)) from exc
            for name in self.spec.publish_artifacts:
                artifact_path = declared_artifacts.get(name)
                if artifact_path is None:
                    raise ExternalTrialError(
                        "successful optimization trial must declare publish artifact "
                        f"{name!r}"
                    )
                artifact_metadata[name] = inspect_file_artifact(
                    artifact_path,
                    relative_path=relative_artifacts[name],
                )
        return ReplicaRecord(
            seed=seed,
            objective_value=resolved_objective,
            success=success,
            metrics=dict(metrics),
            artifacts=relative_artifacts,
            metadata=dict(metadata),
            artifact_metadata=artifact_metadata,
            trial_dir=str(trial_dir.relative_to(self.output_dir)),
        )


__all__ = [
    "ExternalTrialExecutor",
    "ExternalTrialError",
]
