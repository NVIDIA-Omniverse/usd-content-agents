# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification-gated optimizer runner for trusted local customer runtimes."""

from __future__ import annotations

import asyncio
import getpass
import json
import logging
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from world_understanding.optimization import TuneWorkflow

from physics_agent.tuning.errors import TuningCancelledError
from physics_agent.tuning.optimizers import get_runner, resolve_optimizer
from physics_agent.tuning.types import ReplicaRecord, TrialRecord

from .artifacts import (
    APPROVAL,
    ARTIFACT_SCHEMA_VERSION,
    BEST_PARAMS,
    BEST_RECORDING,
    HISTORY,
    RESULTS,
    RUN_SPEC,
    external_trial_payload,
    file_descriptor,
    write_run_spec,
)
from .backend import ExternalTrialError, ExternalTrialExecutor
from .config import load_external_tune_spec
from .evidence import (
    inspect_file_artifact,
    inspect_frame_evidence,
    inspect_usd_recording,
)
from .fingerprint import build_runtime_fingerprint, canonical_digest
from .types import ExternalTuneInput, ExternalTuneOutput, ExternalTuneSpec

logger = logging.getLogger(__name__)


class _QualificationValidationError(ValueError):
    """Approved qualification failed canonical runtime validation."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        qualification_digest: str | None = None,
        qualification_path: Path | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.qualification_digest = qualification_digest
        self.qualification_path = qualification_path
        self.artifacts = artifacts or {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(listener: Any, event_type: str, data: dict[str, Any]) -> None:
    if listener is None:
        return
    try:
        listener.event(event_type, data)
    except Exception:  # pragma: no cover - listeners must not break optimization
        logger.debug("event listener failed for %s", event_type, exc_info=True)


def _cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    if not callable(is_set):
        raise TypeError(
            "cancel_event must be an Event-like object with an is_set() method"
        )
    return bool(is_set())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_history(path: Path, history: list[TrialRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(external_trial_payload(row), sort_keys=True, allow_nan=False)
            + "\n"
            for row in history
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _failed_replica(seed: int, exc: Exception) -> ReplicaRecord:
    return ReplicaRecord(
        seed=seed,
        objective_value=None,
        success=False,
        error=str(exc),
    )


def _fixed_params(
    spec: ExternalTuneSpec, provided: dict[str, float] | None
) -> dict[str, float]:
    """Validate the full parameter vector used to pin inactive dimensions."""

    values = dict(
        provided if provided is not None else spec.qualification.nominal_params
    )
    catalog = {parameter.name: parameter for parameter in spec.parameter_catalog}
    if set(values) != set(catalog):
        raise ValueError(
            "fixed_params must cover the qualified parameter catalog exactly"
        )
    normalized: dict[str, float] = {}
    for name, parameter in catalog.items():
        raw = values[name]
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"fixed parameter {name!r} must be numeric")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"fixed parameter {name!r} must be finite")
        if parameter.integer and not value.is_integer():
            raise ValueError(f"fixed parameter {name!r} must be integer")
        normalized[name] = value
    return normalized


def _validate_run_directory_layout(
    spec: ExternalTuneSpec,
    *,
    output_dir: Path,
    qualification_dir: Path,
) -> None:
    """Reject generated outputs that would mutate a fingerprinted directory."""

    for fingerprint_path in spec.runtime.fingerprint_paths:
        if not fingerprint_path.is_dir():
            continue
        fingerprint_root = fingerprint_path.resolve()
        for label, run_dir in (
            ("output_dir", output_dir),
            ("qualification_dir", qualification_dir),
        ):
            try:
                run_dir.relative_to(fingerprint_root)
            except ValueError:
                continue
            raise ValueError(
                f"{label} must not be inside runtime fingerprint directory "
                f"{fingerprint_root}"
            )


def _publish_evidence(
    record: ReplicaRecord,
    spec: ExternalTuneSpec,
    output_dir: Path,
    *,
    purpose: str,
) -> Path:
    """Publish validated adapter PNG frames to stable public artifact paths."""

    if spec.evidence is None:
        raise ValueError("cannot publish evidence without evidence settings")
    artifact_name = spec.evidence.artifact_name
    relative_source = record.artifacts.get(artifact_name)
    descriptor = record.artifact_metadata.get(artifact_name)
    if relative_source is None or descriptor is None:
        raise ExternalTrialError(
            f"external {purpose} did not return validated PNG-frame evidence"
        )
    source = (output_dir / relative_source).resolve()
    try:
        source.relative_to(output_dir)
    except ValueError:
        raise ExternalTrialError(
            "validated evidence escaped the output directory"
        ) from None
    try:
        current = inspect_frame_evidence(
            source,
            spec.evidence,
            record.metadata,
            relative_path=relative_source,
        )
    except Exception as exc:  # noqa: BLE001 - normalize evidence failures
        raise ExternalTrialError(
            f"external {purpose} frame evidence is no longer valid: {exc}"
        ) from exc
    if current != descriptor:
        raise ExternalTrialError(
            f"external {purpose} frame evidence changed after validation"
        )

    destination = output_dir / f"{purpose}_frames.json"
    destination_frames = output_dir / f"{purpose}_frames"
    temporary_manifest = output_dir / f".{purpose}_frames.tmp.json"
    temporary_frames = output_dir / f".{purpose}_frames.tmp"
    shutil.rmtree(temporary_frames, ignore_errors=True)
    temporary_manifest.unlink(missing_ok=True)
    temporary_frames.mkdir(parents=True)
    manifest_frames: list[dict[str, Any]] = []
    try:
        for index, frame_descriptor in enumerate(current["frames"]):
            relative_frame = frame_descriptor.get("path")
            if not isinstance(relative_frame, str):
                raise ExternalTrialError("validated frame descriptor omitted its path")
            frame_source = (output_dir / relative_frame).resolve()
            try:
                frame_source.relative_to(output_dir)
            except ValueError:
                raise ExternalTrialError(
                    "validated frame evidence escaped the output directory"
                ) from None
            frame_name = f"frame_{index:04d}.png"
            frame_destination = temporary_frames / frame_name
            shutil.copy2(frame_source, frame_destination)
            copied = inspect_file_artifact(
                frame_destination,
                relative_path=f"{purpose}_frames/{frame_name}",
            )
            if copied["sha256"] != frame_descriptor.get("sha256") or copied[
                "size_bytes"
            ] != frame_descriptor.get("size_bytes"):
                raise ExternalTrialError(
                    "frame evidence changed while publishing the qualification"
                )
            manifest_frames.append(
                {
                    "path": f"{purpose}_frames/{frame_name}",
                    "timestamp_seconds": frame_descriptor["timestamp_seconds"],
                }
            )
        _write_json(
            temporary_manifest,
            {
                "schema_version": "physics-agent.qualification-frames.v1",
                "renderer": spec.evidence.renderer,
                "width": spec.evidence.width,
                "height": spec.evidence.height,
                "fps": spec.evidence.fps,
                "frames": manifest_frames,
            },
        )
        shutil.rmtree(destination_frames, ignore_errors=True)
        destination.unlink(missing_ok=True)
        temporary_frames.replace(destination_frames)
        temporary_manifest.replace(destination)
        relative_destination = str(destination.relative_to(output_dir))
        published = inspect_frame_evidence(
            destination,
            spec.evidence,
            record.metadata,
            relative_path=relative_destination,
        )
    except Exception:
        shutil.rmtree(temporary_frames, ignore_errors=True)
        temporary_manifest.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        shutil.rmtree(destination_frames, ignore_errors=True)
        raise
    relative_destination = str(destination.relative_to(output_dir))
    record.artifacts[artifact_name] = relative_destination
    record.artifact_metadata[artifact_name] = published
    return destination


def _discover_camera_paths(stage_path: Path) -> list[str] | None:
    """Return camera prims from a recorded rollout, if it authored any."""

    try:
        from pxr import Usd, UsdGeom
    except ImportError:  # pragma: no cover - Physics Agent dependency invariant
        return None
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        return None
    paths = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]
    return paths or None


def _resolve_best_recording(
    record: TrialRecord,
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and revalidate one recorded rollout from the selected candidate."""

    settings = spec.evidence
    if settings is None:  # pragma: no cover - ExternalTuneSpec rejects this
        raise ExternalTrialError("external tuning has no recording settings")
    if not record.replicas:
        raise ExternalTrialError("winning candidate has no replica recording")
    replica = record.replicas[0]
    recording_name = settings.recording_artifact_name
    relative_recording = replica.artifacts.get(recording_name)
    descriptor = replica.artifact_metadata.get(recording_name)
    if relative_recording is None or descriptor is None:
        raise ExternalTrialError("winning candidate did not persist recording_usd")
    recording_path = (output_dir / relative_recording).resolve()
    try:
        recording_path.relative_to(output_dir)
    except ValueError:
        raise ExternalTrialError(
            "winning candidate recording escaped the output directory"
        ) from None

    try:
        current_descriptor = inspect_usd_recording(
            recording_path,
            settings,
            relative_path=relative_recording,
        )
    except Exception as exc:  # noqa: BLE001 - normalize validation failures
        raise ExternalTrialError(
            f"winning candidate recording is no longer valid: {exc}"
        ) from exc
    if current_descriptor != descriptor:
        raise ExternalTrialError("winning candidate recording changed after execution")
    return recording_path, descriptor


def _publish_best_recording(
    source: Path,
    descriptor: dict[str, Any],
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Copy the scored recording to a stable root path and revalidate it."""

    settings = spec.evidence
    if settings is None:  # pragma: no cover - ExternalTuneSpec rejects this
        raise ExternalTrialError("external tuning has no recording settings")
    destination = output_dir / BEST_RECORDING
    temporary = output_dir / ".best_recording.tmp.usd"
    destination.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temporary)
        try:
            published = inspect_usd_recording(
                temporary,
                settings,
                relative_path=BEST_RECORDING,
            )
        except Exception as exc:  # noqa: BLE001 - normalize validation failures
            raise ExternalTrialError(
                f"winning candidate recording could not be published: {exc}"
            ) from exc
        expected = dict(descriptor)
        expected["path"] = BEST_RECORDING
        if published != expected:
            raise ExternalTrialError(
                "winning candidate recording changed while publishing"
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination, published


def _promote_winner_outputs(
    record: TrialRecord,
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Copy allowlisted adapter outputs from the rendered winner replica."""

    if not spec.publish_artifacts:
        return {}, {}
    if not record.replicas:
        raise ExternalTrialError("winning candidate has no replica artifacts")
    replica = record.replicas[0]
    promoted_paths: dict[str, Path] = {}
    promoted_descriptors: dict[str, dict[str, Any]] = {}
    output_root = output_dir.resolve()
    destination_root = output_root / "outputs"
    temporary_root = output_root / ".outputs.tmp"
    shutil.rmtree(temporary_root, ignore_errors=True)
    try:
        for name in spec.publish_artifacts:
            relative_source = replica.artifacts.get(name)
            descriptor = replica.artifact_metadata.get(name)
            if relative_source is None or descriptor is None:
                raise ExternalTrialError(
                    f"winning candidate did not persist publish artifact {name!r}"
                )
            source = (output_root / relative_source).resolve()
            try:
                source.relative_to(output_root)
            except ValueError:
                raise ExternalTrialError(
                    f"winning publish artifact {name!r} escaped the output directory"
                ) from None
            current = inspect_file_artifact(source, relative_path=relative_source)
            if current != descriptor:
                raise ExternalTrialError(
                    f"winning publish artifact {name!r} changed after execution"
                )
            temporary_destination = temporary_root / name / source.name
            temporary_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, temporary_destination)
            relative_destination = f"outputs/{name}/{source.name}"
            published = file_descriptor(
                temporary_destination,
                relative_path=relative_destination,
            )
            if published["sha256"] != descriptor["sha256"]:
                raise ExternalTrialError(
                    f"winning publish artifact {name!r} changed while publishing"
                )
            promoted_paths[name] = destination_root / name / source.name
            promoted_descriptors[name] = published
        shutil.rmtree(destination_root, ignore_errors=True)
        temporary_root.replace(destination_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return promoted_paths, promoted_descriptors


def _clear_run_artifacts(output_dir: Path) -> None:
    """Remove stale canonical outputs while retaining raw trial evidence."""

    for name in (APPROVAL, RUN_SPEC, HISTORY, BEST_PARAMS, RESULTS, BEST_RECORDING):
        (output_dir / name).unlink(missing_ok=True)
    (output_dir / ".best_recording.tmp.usd").unlink(missing_ok=True)
    shutil.rmtree(output_dir / "render", ignore_errors=True)
    shutil.rmtree(output_dir / "outputs", ignore_errors=True)
    shutil.rmtree(output_dir / ".outputs.tmp", ignore_errors=True)


def _render_best_recording(
    *,
    recording_path: Path,
    descriptor: dict[str, Any],
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> list[Path]:
    """Render the winner directly to PNG evidence from its recorded rollout."""

    settings = spec.evidence
    if settings is None:  # pragma: no cover - ExternalTuneSpec rejects this
        raise ExternalTrialError("external tuning has no recording settings")
    relative_recording = str(recording_path.relative_to(output_dir))

    def validate_recording_unchanged() -> None:
        try:
            current_descriptor = inspect_usd_recording(
                recording_path,
                settings,
                relative_path=relative_recording,
            )
        except Exception as exc:  # noqa: BLE001 - normalize validation failures
            raise ExternalTrialError(
                f"winning candidate recording is no longer valid: {exc}"
            ) from exc
        if current_descriptor != descriptor:
            raise ExternalTrialError(
                "winning candidate recording changed after execution"
            )

    render_dir = output_dir / "render"
    shutil.rmtree(render_dir, ignore_errors=True)
    try:
        validate_recording_unchanged()
        from world_understanding.functions.graphics import render_time_sampled_usd

        cameras = _discover_camera_paths(recording_path)
        frames = render_time_sampled_usd(
            recording_path,
            render_dir,
            renderer=settings.playback_renderer,
            cameras=cameras[:1] if cameras else None,
            fps=descriptor["fps"],
            image_width=settings.width,
            image_height=settings.height,
            max_duration_seconds=settings.max_duration_seconds,
            num_sensor_updates=settings.num_sensor_updates,
            render_mode=settings.render_mode,
        )
        validate_recording_unchanged()
        if not frames:
            raise RuntimeError("renderer produced no frames")
        normalized_frames: list[Path] = []
        for frame in frames:
            resolved = Path(frame).resolve()
            try:
                resolved.relative_to(render_dir.resolve())
            except ValueError:
                raise RuntimeError(
                    "renderer returned a frame outside render directory"
                ) from None
            if not resolved.is_file() or resolved.suffix.lower() != ".png":
                raise RuntimeError("renderer did not return a PNG frame")
            normalized_frames.append(resolved)
    except Exception:
        shutil.rmtree(render_dir, ignore_errors=True)
        raise
    return normalized_frames


def _stored_evidence_is_valid(
    qualification: dict[str, Any],
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> bool:
    """Revalidate the exact qualification PNG sequence referenced by the report."""

    if spec.evidence is None:
        return True
    record = qualification.get("record")
    if not isinstance(record, dict):
        return False
    artifacts = record.get("artifacts")
    artifact_metadata = record.get("artifact_metadata")
    metadata = record.get("metadata")
    if not isinstance(artifacts, dict):
        return False
    if not isinstance(artifact_metadata, dict):
        return False
    if not isinstance(metadata, dict):
        return False
    artifact_name = spec.evidence.artifact_name
    relative_path = artifacts.get(artifact_name)
    stored_descriptor = artifact_metadata.get(artifact_name)
    if not isinstance(relative_path, str) or not isinstance(stored_descriptor, dict):
        return False
    path = (output_dir / relative_path).resolve()
    try:
        path.relative_to(output_dir)
        current = inspect_frame_evidence(
            path,
            spec.evidence,
            metadata,
            relative_path=relative_path,
        )
    except (OSError, ValueError):
        return False
    return current == stored_descriptor


def _qualification_artifacts(
    qualification_path: Path,
    qualification: dict[str, Any],
    spec: ExternalTuneSpec,
    output_dir: Path,
) -> dict[str, Path]:
    artifacts = {"qualification": qualification_path}
    if spec.evidence is None:
        return artifacts
    record = qualification.get("record")
    if not isinstance(record, dict):
        return artifacts
    record_artifacts = record.get("artifacts")
    if not isinstance(record_artifacts, dict):
        return artifacts
    relative_path = record_artifacts.get(spec.evidence.artifact_name)
    if isinstance(relative_path, str):
        path = output_dir / relative_path
        if path.is_file():
            artifacts["qualification_frames"] = path
    return artifacts


def _load_qualification(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_valid_qualification(path: Path) -> dict[str, Any] | None:
    existing = _load_qualification(path)
    if existing is None:
        return None
    stored_payload = {
        key: value for key, value in existing.items() if key != "qualification_digest"
    }
    if existing.get("qualification_digest") != canonical_digest(stored_payload):
        return None
    return existing


def _validate_approved_qualification(
    *,
    spec: ExternalTuneSpec,
    qualification_dir: Path,
    approval_digest: str,
    fingerprint: dict[str, Any],
) -> tuple[dict[str, Any], Path, str, dict[str, Path]]:
    """Validate approval, evidence, and runtime identity without mutating a run."""

    qualification_path = qualification_dir / "qualification.json"
    qualification = _load_valid_qualification(qualification_path)
    if qualification is None:
        raise _QualificationValidationError(
            "qualification_missing",
            "approved optimization requires an existing valid qualification",
        )
    digest = str(qualification["qualification_digest"])
    if not _stored_evidence_is_valid(qualification, spec, qualification_dir):
        raise _QualificationValidationError(
            "qualification_evidence_changed",
            "qualification frame evidence changed after review",
            qualification_digest=digest,
            qualification_path=qualification_path,
            artifacts={"qualification": qualification_path},
        )
    artifacts = _qualification_artifacts(
        qualification_path,
        qualification,
        spec,
        qualification_dir,
    )
    if not qualification.get("eligible"):
        raise _QualificationValidationError(
            "qualification_failed",
            "external runtime qualification failed",
            qualification_digest=digest,
            qualification_path=qualification_path,
            artifacts=artifacts,
        )
    if approval_digest != digest:
        raise _QualificationValidationError(
            "approval_rejected",
            "approval digest does not match the current qualification",
            qualification_digest=digest,
            qualification_path=qualification_path,
            artifacts=artifacts,
        )
    stored_fingerprint = qualification.get("input_fingerprint")
    if (
        not isinstance(stored_fingerprint, dict)
        or stored_fingerprint.get("digest") != fingerprint["digest"]
        or build_runtime_fingerprint(spec)["digest"] != fingerprint["digest"]
    ):
        raise _QualificationValidationError(
            "fingerprint_changed",
            "external runtime inputs changed after qualification",
            qualification_digest=digest,
            qualification_path=qualification_path,
            artifacts=artifacts,
        )
    return qualification, qualification_path, digest, artifacts


def _run_qualification(
    spec: ExternalTuneSpec,
    executor: ExternalTrialExecutor,
    output_dir: Path,
    fingerprint: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = output_dir / "qualification.json"
    existing = _load_valid_qualification(path)
    if existing is not None:
        if existing.get("input_fingerprint", {}).get("digest") == fingerprint[
            "digest"
        ] and _stored_evidence_is_valid(existing, spec, output_dir):
            return existing, path

    if spec.evidence is not None:
        (output_dir / "qualification_frames.json").unlink(missing_ok=True)
        shutil.rmtree(output_dir / "qualification_frames", ignore_errors=True)
    try:
        record = executor.run(
            dict(spec.qualification.nominal_params),
            spec.qualification.seed,
            purpose="qualification",
            index=0,
        )
    except TuningCancelledError:
        raise
    except ExternalTrialError as exc:
        record = _failed_replica(spec.qualification.seed, exc)
    if record.success and spec.evidence is not None:
        try:
            _publish_evidence(
                record,
                spec,
                output_dir,
                purpose="qualification",
            )
        except ExternalTrialError as exc:
            record = _failed_replica(spec.qualification.seed, exc)
    payload = {
        "schema_version": 1,
        "task": spec.task,
        "created_at": _now(),
        "eligible": bool(record.success),
        "input_fingerprint": fingerprint,
        "nominal_params": spec.qualification.nominal_params,
        "record": record.to_dict(),
    }
    payload["qualification_digest"] = canonical_digest(payload)
    _write_json(path, payload)
    return payload, path


def _result_payload(
    output: ExternalTuneOutput, spec: ExternalTuneSpec
) -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for name, path in output.artifacts.items():
        try:
            artifacts[name] = str(path.relative_to(output.output_dir))
        except ValueError:
            artifacts[name] = str(path)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": "external_runtime",
        "task": spec.task,
        "success": output.success,
        "status": output.status,
        "error": output.error,
        "optimizer_used": output.optimizer_used,
        "optimizer_loss": (
            output.best_score if math.isfinite(output.best_score) else None
        ),
        "n_trials": output.n_trials,
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
            "best_value": output.best_objective,
        },
        "active_parameters": [parameter.name for parameter in spec.params],
        "best_params": output.best_params,
        "qualification_digest": output.qualification_digest,
        "started_at": output.started_at,
        "completed_at": output.completed_at,
        "history": [external_trial_payload(row) for row in output.history],
        "rendered_frames": [
            str(path.relative_to(output.output_dir)) for path in output.rendered_frames
        ],
        "render_error": output.render_error,
        "selected_evidence": output.selected_evidence,
        "published_outputs": output.published_outputs,
        "artifacts": artifacts,
        "cancelled": output.cancelled,
    }


def _persist_result(
    output: ExternalTuneOutput, spec: ExternalTuneSpec, output_dir: Path
) -> ExternalTuneOutput:
    output.output_dir = output_dir
    if output.started_at is None:
        output.started_at = _now()
    if output.completed_at is None:
        output.completed_at = _now()
    results_path = output_dir / RESULTS
    _write_json(results_path, _result_payload(output, spec))
    output.artifacts["results"] = results_path
    return output


def _persist_unhandled_result(output: ExternalTuneOutput) -> ExternalTuneOutput:
    """Best-effort terminal record when configuration loading itself failed."""

    if output.output_dir is None:
        return output
    output_dir = Path(output.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output.output_dir = output_dir
    output.started_at = output.started_at or _now()
    output.completed_at = output.completed_at or _now()
    results_path = output_dir / RESULTS
    _write_json(
        results_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "mode": "external_runtime",
            "success": output.success,
            "status": output.status,
            "error": output.error,
            "cancelled": output.cancelled,
            "started_at": output.started_at,
            "completed_at": output.completed_at,
            "n_trials": output.n_trials,
            "artifacts": {},
        },
    )
    output.artifacts["results"] = results_path
    return output


def _run_external_tune(params: ExternalTuneInput) -> ExternalTuneOutput:
    started_at = _now()
    output_dir = params.output_dir.expanduser().resolve()
    spec = load_external_tune_spec(params.config)
    qualification_dir = (
        params.qualification_dir.expanduser().resolve()
        if params.qualification_dir is not None
        else output_dir
    )
    try:
        _validate_run_directory_layout(
            spec,
            output_dir=output_dir,
            qualification_dir=qualification_dir,
        )
    except ValueError as exc:
        return ExternalTuneOutput(
            success=False,
            error=str(exc),
            status="invalid_output_layout",
            output_dir=output_dir,
            started_at=started_at,
            completed_at=_now(),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    qualification_dir.mkdir(parents=True, exist_ok=True)

    def finish(output: ExternalTuneOutput) -> ExternalTuneOutput:
        output.started_at = started_at
        output.completed_at = _now()
        return _persist_result(output, spec, output_dir)

    fingerprint = build_runtime_fingerprint(spec)
    executor = ExternalTrialExecutor(spec, output_dir, cancel_event=params.cancel_event)
    qualification_path = qualification_dir / "qualification.json"
    qualification: dict[str, Any]
    if params.approval_digest is None:
        _clear_run_artifacts(output_dir)
        _emit(
            params.event_listener,
            "external_tune.qualification.started",
            {"task": spec.task},
        )
        qualification_executor = (
            executor
            if qualification_dir == output_dir
            else ExternalTrialExecutor(
                spec,
                qualification_dir,
                cancel_event=params.cancel_event,
            )
        )
        qualification, qualification_path = _run_qualification(
            spec, qualification_executor, qualification_dir, fingerprint
        )
        digest = str(qualification["qualification_digest"])
        qualification_artifacts = _qualification_artifacts(
            qualification_path,
            qualification,
            spec,
            qualification_dir,
        )
        if not qualification.get("eligible"):
            return finish(
                ExternalTuneOutput(
                    success=False,
                    error="external runtime qualification failed",
                    status="qualification_failed",
                    output_dir=output_dir,
                    qualification_digest=digest,
                    qualification_path=qualification_path,
                    artifacts=qualification_artifacts,
                )
            )
        _emit(
            params.event_listener,
            "external_tune.qualification.awaiting_approval",
            {"task": spec.task, "qualification_digest": digest},
        )
        output = ExternalTuneOutput(
            success=True,
            status="awaiting_approval",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            artifacts=qualification_artifacts,
        )
        output.started_at = started_at
        output.completed_at = _now()
        return output
    try:
        (
            qualification,
            qualification_path,
            digest,
            qualification_artifacts,
        ) = _validate_approved_qualification(
            spec=spec,
            qualification_dir=qualification_dir,
            approval_digest=params.approval_digest,
            fingerprint=fingerprint,
        )
    except _QualificationValidationError as exc:
        return finish(
            ExternalTuneOutput(
                success=False,
                error=str(exc),
                status=exc.status,
                output_dir=output_dir,
                qualification_digest=exc.qualification_digest,
                qualification_path=exc.qualification_path,
                artifacts=dict(exc.artifacts),
            )
        )
    _clear_run_artifacts(output_dir)
    approval_path = qualification_dir / APPROVAL
    _write_json(
        approval_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "qualification_digest": digest,
            "approval_source": params.approval_source,
            "recorded_by_process_user": getpass.getuser(),
            "approved_at": _now(),
        },
    )
    approved_artifacts = {**qualification_artifacts, "approval": approval_path}

    history_path = output_dir / HISTORY
    try:
        pinned_params = _fixed_params(spec, params.fixed_params)
    except ValueError as exc:
        return finish(
            ExternalTuneOutput(
                success=False,
                error=str(exc),
                status="invalid_fixed_params",
                output_dir=output_dir,
                qualification_digest=digest,
                qualification_path=qualification_path,
                artifacts=approved_artifacts,
            )
        )
    try:
        tune_workflow = TuneWorkflow(
            spec.optimizer,
            resolve_optimizer=resolve_optimizer,
            get_optimizer_runner=get_runner,
        )
        optimizer_name = tune_workflow.optimizer_used
    except Exception as exc:  # noqa: BLE001 - durable optimizer setup failure
        run_spec_path = write_run_spec(
            output_dir,
            spec,
            qualification_digest=digest,
            fixed_params=pinned_params,
            optimizer_used=None,
        )
        output = ExternalTuneOutput(
            success=False,
            error=str(exc),
            status="optimizer_unavailable",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=spec.optimizer.name,
            artifacts={**approved_artifacts, "run_spec": run_spec_path},
        )
        return finish(output)
    run_spec_path = write_run_spec(
        output_dir,
        spec,
        qualification_digest=digest,
        fixed_params=pinned_params,
        optimizer_used=optimizer_name,
    )
    approved_artifacts["run_spec"] = run_spec_path

    def evaluate_full(
        candidate: dict[str, float],
        *,
        purpose: str,
        candidate_index: int,
    ) -> TrialRecord:
        if _cancelled(params.cancel_event):
            raise TuningCancelledError("external tuning cancelled")
        trial_started = time.monotonic()
        replicas: list[ReplicaRecord] = []
        for replica_index, seed in enumerate(spec.optimizer.replica_seeds):
            execution_index = candidate_index * spec.optimizer.replicas + replica_index
            try:
                replica = executor.run(
                    candidate,
                    seed,
                    purpose=purpose,
                    index=execution_index,
                )
            except TuningCancelledError:
                raise
            except ExternalTrialError as exc:
                replica = _failed_replica(seed, exc)
            replicas.append(replica)
        values = [
            float(replica.objective_value)
            for replica in replicas
            if replica.objective_value is not None
        ]
        objective_value = fmean(values) if values else None
        feasible = bool(replicas) and all(replica.success for replica in replicas)
        score = (
            spec.objective.optimizer_score(objective_value)
            if feasible and objective_value is not None
            else spec.objective.failure_penalty
        )
        replica_errors = [
            replica.error for replica in replicas if replica.error is not None
        ]
        return TrialRecord(
            trial_index=candidate_index,
            params={name: float(value) for name, value in candidate.items()},
            score=float(score),
            duration_seconds=time.monotonic() - trial_started,
            objective_value=objective_value,
            failed=not feasible,
            error="; ".join(replica_errors) if replica_errors else None,
            replicas=replicas,
        )

    trial_run_history: list[TrialRecord] = []

    def record_trial(record: TrialRecord) -> None:
        trial_run_history.append(record)
        _write_history(history_path, trial_run_history)
        _emit(
            params.event_listener,
            "external_tune.trial.completed",
            external_trial_payload(record),
        )

    def evaluate_trial(
        candidate: dict[str, float],
        candidate_index: int,
        _trial_seed: int,
    ) -> TrialRecord:
        full_candidate = dict(pinned_params)
        full_candidate.update(candidate)
        return evaluate_full(
            full_candidate,
            purpose="optimization",
            candidate_index=candidate_index,
        )

    _emit(
        params.event_listener,
        "external_tune.optimization.started",
        {"task": spec.task, "optimizer": optimizer_name},
    )
    try:
        tune_run = tune_workflow.run(
            search_space=spec,
            evaluate_trial=evaluate_trial,
            cancel_check=lambda: _cancelled(params.cancel_event),
            on_trial=record_trial,
            cancellation_exceptions=(TuningCancelledError,),
        )
    except Exception as exc:  # noqa: BLE001 - durable optimizer failure boundary
        logger.exception("External optimizer failed")
        history = trial_run_history
        output = ExternalTuneOutput(
            success=False,
            error=str(exc),
            status="optimization_failed",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            n_trials=len(history),
            history=history,
            artifacts={
                **approved_artifacts,
                **({"history": history_path} if history_path.is_file() else {}),
            },
        )
        return finish(output)
    history = tune_run.history
    if tune_run.cancelled or _cancelled(params.cancel_event):
        output = ExternalTuneOutput(
            success=False,
            error="external tuning cancelled",
            status="cancelled",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            n_trials=len(history),
            history=history,
            cancelled=True,
            artifacts={
                **approved_artifacts,
                **({"history": history_path} if history_path.is_file() else {}),
            },
        )
        return finish(output)
    best = tune_run.best()
    if best is None:
        output = ExternalTuneOutput(
            success=False,
            error="external optimization produced no feasible candidate",
            status="optimization_failed",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            n_trials=len(history),
            history=history,
            artifacts={
                **approved_artifacts,
                **({"history": history_path} if history_path.is_file() else {}),
            },
        )
        return finish(output)
    if build_runtime_fingerprint(spec)["digest"] != fingerprint["digest"]:
        output = ExternalTuneOutput(
            success=False,
            error="external runtime inputs changed during optimization",
            status="fingerprint_changed",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            n_trials=len(history),
            history=history,
            artifacts={**approved_artifacts, "history": history_path},
        )
        return finish(output)

    best_params_path = output_dir / BEST_PARAMS
    _write_json(
        best_params_path,
        {
            "best_score": best.optimizer_score,
            "params": best.params,
        },
    )
    completed_artifacts = {
        **approved_artifacts,
        "history": history_path,
        "best_params": best_params_path,
    }
    try:
        scored_recording, scored_recording_descriptor = _resolve_best_recording(
            best, spec, output_dir
        )
        best_recording, published_recording_descriptor = _publish_best_recording(
            scored_recording,
            scored_recording_descriptor,
            spec,
            output_dir,
        )
        promoted_paths, published_outputs = _promote_winner_outputs(
            best,
            spec,
            output_dir,
        )
    except (ExternalTrialError, OSError) as exc:
        (output_dir / BEST_RECORDING).unlink(missing_ok=True)
        shutil.rmtree(output_dir / "outputs", ignore_errors=True)
        output = ExternalTuneOutput(
            success=False,
            error=str(exc),
            status="recording_failed",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            best_params=best.params,
            best_score=best.optimizer_score,
            best_objective=best.objective_value,
            n_trials=len(history),
            history=history,
            artifacts=completed_artifacts,
        )
        return finish(output)
    completed_artifacts["best_recording"] = best_recording
    completed_artifacts.update(
        {f"output.{name}": path for name, path in promoted_paths.items()}
    )
    selected_replica = best.replicas[0]
    selected_evidence = {
        "policy": "first_replica",
        "trial_index": best.trial_index,
        "replica_index": 0,
        "seed": selected_replica.seed,
        "replica_objective": selected_replica.objective_value,
        "aggregate_objective": best.objective_value,
        "scored_recording": scored_recording_descriptor,
        "published_recording": published_recording_descriptor,
    }
    rendered_frames: list[Path] = []
    render_error: str | None = None

    def cancelled_after_optimization() -> ExternalTuneOutput:
        return ExternalTuneOutput(
            success=False,
            error="external tuning cancelled after optimization",
            status="cancelled",
            output_dir=output_dir,
            qualification_digest=digest,
            qualification_path=qualification_path,
            optimizer_used=optimizer_name,
            best_params=best.params,
            best_score=best.optimizer_score,
            best_objective=best.objective_value,
            n_trials=len(history),
            history=history,
            cancelled=True,
            rendered_frames=rendered_frames,
            render_error=render_error,
            selected_evidence=selected_evidence,
            published_outputs=published_outputs,
            artifacts=completed_artifacts,
        )

    if _cancelled(params.cancel_event):
        return finish(cancelled_after_optimization())
    if params.render_winning_trial:
        _emit(
            params.event_listener,
            "external_tune.best_recording.render.started",
            {"task": spec.task, "trial_index": best.trial_index},
        )
        try:
            rendered_frames = _render_best_recording(
                recording_path=scored_recording,
                descriptor=scored_recording_descriptor,
                spec=spec,
                output_dir=output_dir,
            )
        except ExternalTrialError as exc:
            if _cancelled(params.cancel_event):
                return finish(cancelled_after_optimization())
            output = ExternalTuneOutput(
                success=False,
                error=str(exc),
                status="render_failed",
                output_dir=output_dir,
                qualification_digest=digest,
                qualification_path=qualification_path,
                optimizer_used=optimizer_name,
                best_params=best.params,
                best_score=best.optimizer_score,
                best_objective=best.objective_value,
                n_trials=len(history),
                history=history,
                selected_evidence=selected_evidence,
                published_outputs=published_outputs,
                artifacts=completed_artifacts,
            )
            return finish(output)
        except Exception as exc:  # noqa: BLE001 - standard render is non-fatal
            render_error = type(exc).__name__
            logger.warning(
                "External winner render failed (non-fatal): %s: %s",
                render_error,
                exc,
            )
            _emit(
                params.event_listener,
                "external_tune.best_recording.render.failed",
                {"task": spec.task, "error": render_error},
            )
        else:
            _emit(
                params.event_listener,
                "external_tune.best_recording.render.completed",
                {
                    "task": spec.task,
                    "trial_index": best.trial_index,
                    "frame_count": len(rendered_frames),
                    "render_dir": str(output_dir / "render"),
                },
            )
    if _cancelled(params.cancel_event):
        return finish(cancelled_after_optimization())
    output = ExternalTuneOutput(
        success=True,
        status="completed",
        output_dir=output_dir,
        qualification_digest=digest,
        qualification_path=qualification_path,
        optimizer_used=optimizer_name,
        best_params=best.params,
        best_score=best.optimizer_score,
        best_objective=best.objective_value,
        n_trials=len(history),
        history=history,
        rendered_frames=rendered_frames,
        render_error=render_error,
        selected_evidence=selected_evidence,
        published_outputs=published_outputs,
        artifacts=completed_artifacts,
    )
    finish(output)
    _emit(
        params.event_listener,
        "external_tune.completed",
        {"task": spec.task, "success": output.success, "status": output.status},
    )
    return output


def run_external_tune(params: ExternalTuneInput) -> ExternalTuneOutput:
    """Qualify or optimize one trusted local external runtime."""

    try:
        return _run_external_tune(params)
    except TuningCancelledError as exc:
        return _persist_unhandled_result(
            ExternalTuneOutput(
                success=False,
                error=str(exc),
                status="cancelled",
                output_dir=params.output_dir.expanduser().resolve(),
                cancelled=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - public API result boundary
        logger.exception("External tuning failed")
        return _persist_unhandled_result(
            ExternalTuneOutput(
                success=False,
                error=str(exc),
                status="failed",
                output_dir=params.output_dir.expanduser().resolve(),
            )
        )


async def arun_external_tune(params: ExternalTuneInput) -> ExternalTuneOutput:
    """Run :func:`run_external_tune` without blocking an async caller."""

    return await asyncio.to_thread(run_external_tune, params)


__all__ = ["arun_external_tune", "run_external_tune"]
