# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution using MAA Python async API.

Calls arun_pipeline directly - no wrappers or thread pools needed!
"""

import asyncio
import json
import logging
import threading
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from material_agent.api import (
    PipelineInput,
    ScenePipelineInput,
    ScenePipelineOutput,
    arun_pipeline,
    arun_scene_pipeline,
)
from world_understanding.agentic.config import clone_config_containers
from world_understanding.telemetry import get_current_span, traced
from world_understanding.telemetry.attributes import MAAttributes
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)
from world_understanding.utils.result_projection import project_result_metadata

from ..artifact_lineage import (
    ARTIFACT_CANONICAL_KEYS,
    ARTIFACT_PRODUCER_STEPS,
    emitted_artifacts_for_completed_steps,
    revalidate_artifacts_for_completed_steps,
)
from ..coverage import (
    CoveragePolicy,
    build_material_coverage,
    build_not_evaluated_material_coverage,
    coverage_is_release_ready,
    normalize_coverage_policy,
)
from ..events.listener import FastAPIEventListener
from ..events.sanitization import bounded_step_completion_data
from ..events.telemetry_listener import TelemetryEventListener
from ..json_utils import to_json_safe
from ..runtime.bus import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..session.manager import RegenerationClaim, SessionManager

logger = logging.getLogger(__name__)

_REGENERATION_LEASE_SECONDS = 300.0
_REGENERATION_HEARTBEAT_SECONDS = 60.0

_STEP_DISPLAY_NAMES = {
    "optimize_usd": "Optimizing USD Scene",
    "build_dataset_usd": "Rendering USD Scene",
    "build_dataset_prepare_dataset": "Preparing Dataset",
    "cluster_prims": "Clustering Prims",
    "expand_cluster_predictions": "Expanding Cluster Predictions",
    "prepare_dataset": "Preparing Dataset",
    "predict": "Running VLM Predictions",
    "restore_usd": "Restoring Prediction Paths",
    "apply": "Applying Materials",
    "render": "Rendering Final Output",
}


def _current_run_completed_steps(
    config_dict: dict[str, Any],
    completed_steps: list[str] | None,
) -> list[str]:
    """Exclude cached upstream completions from this run's publication set."""
    configured_steps = config_dict.get("steps")
    active_steps = (
        set(configured_steps) if isinstance(configured_steps, dict) else set()
    )
    return [step for step in (completed_steps or []) if step in active_steps]


def _artifact_file_signature(path: Path) -> tuple[int, int, int] | None:
    """Return a cheap identity for detecting files written by this execution."""
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _capture_promotable_file_signatures(
    session_dir: Path,
) -> dict[str, tuple[int, int, int]]:
    """Snapshot files that promotion may otherwise mistake for current outputs."""
    candidates = {key for keys in ARTIFACT_CANONICAL_KEYS.values() for key in keys}
    candidates.add("cache/.pipeline_state.json")
    for relative_dir in (
        "cache/dataset",
        "cache/generated_material_library",
        "cache/preview",
        "generated_material_library",
        "output/generated_material_library",
        "preview",
    ):
        directory = session_dir / relative_dir
        if directory.exists():
            candidates.update(
                path.relative_to(session_dir).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )
    signatures: dict[str, tuple[int, int, int]] = {}
    for key in candidates:
        signature = _artifact_file_signature(session_dir / key)
        if signature is not None:
            signatures[key] = signature
    return signatures


async def _promote_current_run_artifacts(
    session_manager: SessionManager,
    session_id: str,
    session_dir: Path,
    completed_steps: list[str],
    step_results: dict[str, Any] | None,
    *,
    regeneration_claim: RegenerationClaim | None = None,
    artifact_map: dict[str, str] | None = None,
    baseline_signatures: dict[str, tuple[int, int, int]] | None = None,
) -> set[str]:
    """Publish current artifacts canonically or under a claimed immutable prefix."""
    emitted = emitted_artifacts_for_completed_steps(completed_steps, step_results)
    completed = set(completed_steps)

    def produced_this_run(key: str, path: Path) -> bool:
        signature = _artifact_file_signature(path)
        return signature is not None and (
            baseline_signatures is None or baseline_signatures.get(key) != signature
        )

    # Some Material Agent step results report counts/bindings but omit their
    # output path. A completed producer plus a canonical file is still strong
    # publication evidence; otherwise successful legacy and claimed runs lose
    # downloadable artifacts despite having written them.
    for artifact, producer_steps in ARTIFACT_PRODUCER_STEPS.items():
        if artifact == "prediction_report" or not (completed & producer_steps):
            continue
        canonical_file_exists = any(
            produced_this_run(key, session_dir / key)
            for key in ARTIFACT_CANONICAL_KEYS.get(artifact, ())
        )
        preview_file_exists = artifact == "previews" and any(
            produced_this_run(path.relative_to(session_dir).as_posix(), path)
            for preview_dir in (
                session_dir / "cache" / "preview",
                session_dir / "preview",
            )
            if preview_dir.exists()
            for path in preview_dir.glob("*.png")
        )
        if canonical_file_exists or preview_file_exists:
            emitted.add(artifact)
    promoted: set[str] = set()

    async def publish(key: str, path: Path) -> None:
        target_key = (
            f"{regeneration_claim.artifact_prefix}/{key}"
            if regeneration_claim is not None
            else key
        )
        await session_manager.put_file_to_store(
            session_id,
            target_key,
            str(path),
        )
        if artifact_map is not None:
            artifact_map[key] = target_key

    for artifact in emitted:
        artifact_promoted = False
        for key in ARTIFACT_CANONICAL_KEYS.get(artifact, ()):
            path = session_dir / key
            # Explicit step output identifies the artifact group, but it must
            # not waive freshness for every canonical alias left on disk by a
            # prior generation.
            if not produced_this_run(key, path):
                continue
            await publish(key, path)
            artifact_promoted = True
        if artifact_promoted:
            promoted.add(artifact)

    auxiliary_keys: set[str] = set()
    if "build_dataset_usd" in completed_steps:
        preview_dir = session_dir / "cache" / "preview"
        if preview_dir.exists():
            preview_keys = {
                str(path.relative_to(session_dir))
                for path in preview_dir.glob("*.png")
                if path.is_file()
            }
            auxiliary_keys.update(preview_keys)
        dataset_usd_dir = session_dir / "cache" / "dataset" / "usd"
        if dataset_usd_dir.exists():
            auxiliary_keys.update(
                path.relative_to(session_dir).as_posix()
                for path in dataset_usd_dir.rglob("*")
                if path.is_file()
            )
    if "build_dataset_prepare_dataset" in completed_steps:
        dataset_dir = session_dir / "cache" / "dataset"
        if dataset_dir.exists():
            auxiliary_keys.update(
                path.relative_to(session_dir).as_posix()
                for path in dataset_dir.rglob("*")
                if path.is_file()
            )
    if "generate_material_library" in completed_steps:
        for generated_dir in (
            session_dir / "cache" / "generated_material_library",
            session_dir / "generated_material_library",
            session_dir / "output" / "generated_material_library",
        ):
            if generated_dir.exists():
                auxiliary_keys.update(
                    path.relative_to(session_dir).as_posix()
                    for path in generated_dir.rglob("*")
                    if path.is_file()
                )
    checkpoint_path = session_dir / "cache" / ".pipeline_state.json"
    if checkpoint_path.is_file():
        auxiliary_keys.add("cache/.pipeline_state.json")

    published_auxiliary_keys: set[str] = set()
    for key in sorted(auxiliary_keys):
        path = session_dir / key
        if not produced_this_run(key, path):
            continue
        await publish(key, path)
        published_auxiliary_keys.add(key)
    if "previews" in emitted and any(
        key.startswith(("cache/preview/", "preview/"))
        for key in published_auxiliary_keys
    ):
        promoted.add("previews")
    return promoted


async def _carry_forward_regeneration_artifacts(
    session_manager: SessionManager,
    session_id: str,
    claim: RegenerationClaim,
    metadata: dict[str, Any],
    artifact_validity: dict[str, bool],
    artifact_map: dict[str, str],
    configured_steps: set[str],
) -> dict[str, bool]:
    """Copy still-valid prior artifacts into the claim's immutable generation."""
    publication = metadata.get("published_artifacts")
    prior_map = (
        publication.get("artifacts", {}) if isinstance(publication, dict) else {}
    )
    logical_keys = {
        key for key in prior_map if isinstance(key, str) and key not in artifact_map
    }
    if not isinstance(publication, dict):
        prefixes = (
            "cache/dataset/",
            "cache/generated_material_library/",
            "generated_material_library/",
            "output/generated_material_library/",
            "cache/optimized/",
            "cache/preview/",
            "preview/",
        )
        for prefix in prefixes:
            logical_keys.update(
                await session_manager.store.list_keys(session_id, prefix=prefix)
            )
        logical_keys.add("cache/.pipeline_state.json")
        for keys in ARTIFACT_CANONICAL_KEYS.values():
            logical_keys.update(keys)

    artifact_by_key = {
        key: artifact
        for artifact, keys in ARTIFACT_CANONICAL_KEYS.items()
        for key in keys
    }
    for logical_key in sorted(logical_keys):
        if logical_key in artifact_map:
            continue
        if "build_dataset_usd" in configured_steps and logical_key.startswith(
            ("cache/dataset/", "cache/preview/", "preview/")
        ):
            continue
        if (
            "build_dataset_prepare_dataset" in configured_steps
            and logical_key.startswith("cache/dataset/")
        ):
            continue
        if "generate_material_library" in configured_steps and logical_key.startswith(
            (
                "cache/generated_material_library/",
                "generated_material_library/",
                "output/generated_material_library/",
            )
        ):
            continue
        artifact = artifact_by_key.get(logical_key)
        if artifact is not None and not artifact_validity.get(artifact, False):
            continue
        if logical_key == "cache/predictions/prediction_report.html":
            continue
        if logical_key == "cache/.pipeline_state.json":
            # A current checkpoint is promoted before carry-forward. Never
            # republish a prior generation's checkpoint for a new execution.
            continue
        if logical_key.startswith(("cache/preview/", "preview/")) and not (
            artifact_validity.get("previews", False)
        ):
            continue
        source_key = session_manager.resolve_published_artifact_key(
            metadata,
            logical_key,
            legacy_key=logical_key,
        )
        if source_key is None:
            continue
        data = await session_manager.read_from_store(session_id, source_key)
        if data is None:
            continue
        target_key = f"{claim.artifact_prefix}/{logical_key}"
        await session_manager.put_bytes_to_store(
            session_id,
            target_key,
            data,
        )
        artifact_map[logical_key] = target_key

    verified_validity = dict(artifact_validity)
    for artifact, canonical_keys in ARTIFACT_CANONICAL_KEYS.items():
        if verified_validity.get(artifact) and not any(
            key in artifact_map for key in canonical_keys
        ):
            verified_validity[artifact] = False
    if verified_validity.get("previews") and not any(
        key.startswith(("cache/preview/", "preview/")) for key in artifact_map
    ):
        verified_validity["previews"] = False
    return verified_validity


async def _emit_persisted_pipeline_failure(
    session_id: str,
    error: str,
    *,
    step: str,
    regeneration_claim: RegenerationClaim | None,
) -> None:
    """Emit the authoritative failure marker after metadata is persisted."""
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is None:
        return

    try:
        await event_bus.emit_for_owner(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.FAILED,
                message=error,
                extra={"pipeline_failed": True},
            ),
            regeneration_claim=regeneration_claim,
        )
    except Exception:  # pragma: no cover - diagnostics must not mask root failure
        log_durable_failure(
            logger,
            "material_pipeline_failure_event_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )


async def _emit_persisted_pipeline_cancellation(
    session_id: str,
    *,
    step: str,
    regeneration_claim: RegenerationClaim | None,
) -> None:
    """Emit the authoritative cancellation marker after metadata is persisted."""
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is None:
        return

    try:
        await event_bus.emit_for_owner(
            ProgressEvent(
                session_id=session_id,
                step=step,
                state=StepState.CANCELLED,
                percent=100,
                message="Pipeline cancelled",
                extra={"pipeline_cancelled": True},
            ),
            regeneration_claim=regeneration_claim,
        )
    except Exception:  # pragma: no cover - diagnostics must not mask cancellation
        log_durable_failure(
            logger,
            "pipeline_cancellation_event_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=True,
        )


async def _poll_standard_pipeline_cancellation(
    session_manager: SessionManager,
    session_id: str,
    owner_task: asyncio.Task[Any],
) -> None:
    """Cancel a standard worker when another instance writes its durable marker."""
    while True:
        try:
            if await session_manager.is_cancelled(session_id):
                owner_task.cancel()
                return
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_cancellation_poll_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
        await asyncio.sleep(0.5)


async def _monitor_regeneration_claim(
    session_manager: SessionManager,
    session_id: str,
    claim: RegenerationClaim,
    owner_task: asyncio.Task[Any],
) -> None:
    """Renew a claim and stop its worker on cancellation, expiry, or takeover."""
    while True:
        try:
            if await session_manager.is_regeneration_cancel_requested(
                session_id,
                claim,
            ):
                owner_task.cancel()
                return
            renewed = await session_manager.renew_regeneration_claim(
                session_id,
                claim,
                lease_seconds=_REGENERATION_LEASE_SECONDS,
            )
            if not renewed:
                metadata = await session_manager.get_session_metadata(session_id)
                current = (
                    RegenerationClaim.from_metadata(metadata)
                    if metadata is not None
                    else None
                )
                raw_claim = (
                    metadata.get("regeneration_claim") if metadata is not None else None
                )
                if (
                    current is not None
                    and metadata is not None
                    and isinstance(raw_claim, Mapping)
                    and current.generation == claim.generation
                    and current.token == claim.token
                    and raw_claim.get("active") is not True
                    and not raw_claim.get("aborted_at")
                    and metadata.get("status")
                    in {
                        "completed",
                        "succeeded",
                        "failed",
                        "cancelled",
                        "canceled",
                    }
                ):
                    finalized_at_value = raw_claim.get("finalized_at")
                    try:
                        finalized_at = datetime.fromisoformat(
                            str(finalized_at_value).replace("Z", "+00:00")
                        )
                    except ValueError:
                        finalized_at = None
                    if (
                        finalized_at is not None
                        and current.lease_expires_at > finalized_at
                    ):
                        return
                owner_task.cancel()
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            log_durable_failure(
                logger,
                "regeneration_claim_monitor_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
            owner_task.cancel()
            return
        await asyncio.sleep(_REGENERATION_HEARTBEAT_SECONDS)


async def _update_pipeline_session(
    session_manager: SessionManager,
    session_id: str,
    updates: dict[str, Any],
    *,
    regeneration_claim: RegenerationClaim | None,
    remove_fields: tuple[str, ...] = (),
) -> bool:
    """Persist a nonterminal update without letting a stale worker write."""
    if regeneration_claim is None:
        await session_manager.update_session(
            session_id,
            updates,
            remove_fields=remove_fields,
        )
        return True
    return await session_manager.update_session_for_claim(
        session_id,
        regeneration_claim,
        updates,
        remove_fields=remove_fields,
    )


async def _finalize_pipeline_session(
    session_manager: SessionManager,
    session_id: str,
    updates: dict[str, Any],
    *,
    regeneration_claim: RegenerationClaim | None,
    artifact_map: dict[str, str] | None = None,
    remove_fields: tuple[str, ...] = (),
) -> bool:
    """Persist terminal metadata and atomically publish immutable artifacts."""
    if regeneration_claim is None:
        return await session_manager.finalize_standard_pipeline(
            session_id,
            updates,
            remove_fields=remove_fields,
        )
    return await session_manager.finalize_regeneration_claim(
        session_id,
        regeneration_claim,
        updates=updates,
        artifact_map=artifact_map,
        remove_fields=remove_fields,
    )


async def _mark_standard_terminal_events_quiesced(
    session_manager: SessionManager,
    session_id: str,
    *,
    regeneration_claim: RegenerationClaim | None,
    expected_status: str,
) -> None:
    """Open regeneration admission after the standard terminal emit phase."""
    if regeneration_claim is not None:
        return
    if not await session_manager.mark_terminal_events_quiesced(
        session_id,
        expected_status=expected_status,
    ):
        raise RuntimeError(
            f"Failed to quiesce terminal events for {session_id} ({expected_status})"
        )


@traced("maa.pipeline.execution")
async def execute_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
    coverage_policy: CoveragePolicy = "allow_partial",
    regeneration_claim: RegenerationClaim | None = None,
) -> None:
    """Execute pipeline workflow using MAA Python async API.

    Handles cancellation and errors by updating session metadata so
    sessions never get stuck in "running" state.

    Args:
        session_id: Session identifier
        config_dict: Complete unified pipeline config dict (built in router)
        session_manager: SessionManager instance
        user_email: User email address for telemetry
    """
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution always has a task
        raise RuntimeError("Pipeline execution requires an active asyncio task")
    cancel_poll_task = asyncio.create_task(
        _monitor_regeneration_claim(
            session_manager,
            session_id,
            regeneration_claim,
            owner_task,
        )
        if regeneration_claim is not None
        else _poll_standard_pipeline_cancellation(
            session_manager,
            session_id,
            owner_task,
        )
    )
    try:
        await _execute_pipeline_inner(
            session_id,
            config_dict,
            session_manager,
            user_email,
            coverage_policy=normalize_coverage_policy(coverage_policy),
            regeneration_claim=regeneration_claim,
        )
    except asyncio.CancelledError:
        logger.info(f"Pipeline cancelled for {session_id[:8]}")
        persisted = await _finalize_pipeline_session(
            session_manager,
            session_id,
            {
                "status": "cancelled",
                "cancelled_at": datetime.now(UTC).isoformat(),
            },
            regeneration_claim=regeneration_claim,
        )
        if persisted:
            await _emit_persisted_pipeline_cancellation(
                session_id,
                step="pipeline",
                regeneration_claim=regeneration_claim,
            )
            await _mark_standard_terminal_events_quiesced(
                session_manager,
                session_id,
                regeneration_claim=regeneration_claim,
                expected_status="cancelled",
            )
        raise  # Re-raise so JobRegistry cleanup runs
    except Exception:
        diagnostic = durable_diagnostic(
            "material_pipeline_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        persisted = await _finalize_pipeline_session(
            session_manager,
            session_id,
            {
                "status": "failed",
                "error": diagnostic.code,
                "error_diagnostic": diagnostic.to_dict(),
                "failed_at": datetime.now(UTC).isoformat(),
            },
            regeneration_claim=regeneration_claim,
        )
        if persisted:
            await _emit_persisted_pipeline_failure(
                session_id,
                diagnostic.code,
                step="pipeline",
                regeneration_claim=regeneration_claim,
            )
            await _mark_standard_terminal_events_quiesced(
                session_manager,
                session_id,
                regeneration_claim=regeneration_claim,
                expected_status="failed",
            )
        raise
    finally:
        cancel_poll_task.cancel()
        try:
            await cancel_poll_task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - monitor is defensive internally
            logger.exception(
                "Pipeline cancellation monitor failed during cleanup for %s",
                session_id[:8],
            )


@traced("maa.scene_pipeline.execution")
async def execute_scene_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
    scene_options: dict[str, Any] | None = None,
    coverage_policy: CoveragePolicy = "allow_partial",
) -> None:
    """Execute the large-scene material pipeline via the public Python API."""
    try:
        normalized_policy = normalize_coverage_policy(coverage_policy)
        if normalized_policy == "strict":
            raise ValueError(
                "coverage_policy=strict currently requires the single-asset "
                "pipeline because large-scene prim-level coverage is not qualified"
            )
        await _execute_scene_pipeline_inner(
            session_id,
            config_dict,
            session_manager,
            user_email,
            scene_options or {},
            coverage_policy=normalized_policy,
        )
    except asyncio.CancelledError:
        logger.info("Scene pipeline cancelled for %s", session_id[:8])
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "cancelled_at": datetime.now(UTC).isoformat(),
            },
        )
        await _emit_persisted_pipeline_cancellation(
            session_id,
            step="scene_pipeline",
            regeneration_claim=None,
        )
        raise
    except Exception:
        diagnostic = durable_diagnostic(
            "material_scene_pipeline_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        await session_manager.update_session(
            session_id,
            {
                "status": "failed",
                "error": diagnostic.code,
                "error_diagnostic": diagnostic.to_dict(),
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
        await _emit_persisted_pipeline_failure(
            session_id,
            diagnostic.code,
            step="scene_pipeline",
            regeneration_claim=None,
        )
        raise


async def _execute_scene_pipeline_inner(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
    scene_options: dict[str, Any] | None = None,
    coverage_policy: CoveragePolicy = "allow_partial",
) -> None:
    """Inner large-scene execution logic."""
    logger.info("Scene pipeline execution started for %s...", session_id[:8])
    scene_options = scene_options or {}
    session_dir = session_manager.get_session_dir(session_id)

    scene_config = clone_config_containers(config_dict)
    if not isinstance(scene_config, dict):  # pragma: no cover - input contract
        raise TypeError("Scene pipeline configuration must be a mapping")
    project_config = scene_config.setdefault("project", {})
    if not isinstance(project_config, dict):
        project_config = {}
        scene_config["project"] = project_config
    project_config["working_dir"] = str(session_dir / "scene")

    output_usd_path = session_dir / "output" / "scene_with_materials.usd"
    output_usd_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract session-specific material icons from config for progress events.
    session_material_icons: dict[str, str] = {}
    materials_entries = scene_config.get("materials", {}).get("entries", [])
    for entry in materials_entries:
        if isinstance(entry, dict) and "name" in entry and "icon" in entry:
            session_material_icons[str(entry["name"])] = str(entry["icon"])

    inner_listener = FastAPIEventListener(
        session_id,
        session_dir,
        session_material_icons=session_material_icons,
        regeneration_claim=None,
    )
    telemetry_listener = TelemetryEventListener(inner_listener)

    metadata = await session_manager.get_session_metadata(session_id)
    asset_info = metadata.get("asset", {}) if metadata else {}
    predict_step = scene_config.get("steps", {}).get("predict", {})
    vlm_model = (
        predict_step.get("vlm", {}).get("model", "")
        if isinstance(predict_step, dict)
        else ""
    )

    span = get_current_span()
    if span:
        span.set_attribute(MAAttributes.PIPELINE_SESSION_ID, session_id)
        span.set_attribute(MAAttributes.PIPELINE_USER_EMAIL, user_email)
        span.set_attribute(MAAttributes.LANGFUSE_USER_ID, user_email)
        span.set_attribute(MAAttributes.LANGFUSE_SESSION_ID, session_id)
        span.set_attribute("maa.pipeline_type", "large_scene")
        if vlm_model:
            span.set_attribute(MAAttributes.PIPELINE_VLM_MODEL, vlm_model)
            span.set_attribute(MAAttributes.LANGFUSE_META_VLM_MODEL, vlm_model)
        if asset_info:
            if asset_info.get("filename"):
                span.set_attribute(MAAttributes.ASSET_FILENAME, asset_info["filename"])
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILENAME,
                    asset_info["filename"],
                )
            if asset_info.get("file_size_bytes"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_SIZE_BYTES,
                    asset_info["file_size_bytes"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_SIZE,
                    asset_info["file_size_bytes"],
                )
            if asset_info.get("file_extension"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_EXTENSION,
                    asset_info["file_extension"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_EXT,
                    asset_info["file_extension"],
                )

    await session_manager.update_session(
        session_id,
        {
            "status": "running",
            "pipeline_type": "large_scene",
        },
    )

    local_cancel_event = threading.Event()

    def is_scene_cancelled() -> bool:
        if local_cancel_event.is_set():
            return True
        if (session_dir / ".cancel").exists():
            local_cancel_event.set()
            return True
        return False

    async def poll_scene_cancellation() -> None:
        while not local_cancel_event.is_set():
            if (session_dir / ".cancel").exists():
                local_cancel_event.set()
                return
            try:
                if await session_manager.is_cancelled(session_id):
                    local_cancel_event.set()
                    return
            except Exception:
                log_durable_failure(
                    logger,
                    "scene_cancellation_poll_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
            await asyncio.sleep(0.5)

    cancel_poll_task = asyncio.create_task(poll_scene_cancellation())
    try:
        result = await arun_scene_pipeline(
            ScenePipelineInput(
                config=scene_config,
                config_base_dir=session_dir,
                assets=list(scene_options.get("assets") or []),
                skip_steps=list(scene_options.get("skip_steps") or []),
                only_steps=list(scene_options.get("only_steps") or []),
                from_step=scene_options.get("from_step"),
                skip_existing=bool(scene_options.get("skip_existing", False)),
                max_workers=int(scene_options.get("max_workers", 1)),
                resume=bool(scene_options.get("resume", False)),
                clean=bool(scene_options.get("clean", False)),
                no_render=bool(scene_options.get("no_render", False)),
                clear_materials=bool(scene_options.get("clear_materials", False)),
                output_usd_path=output_usd_path,
                validate_output=bool(scene_options.get("validate_output", True)),
                fail_on_validation_error=bool(
                    scene_options.get("fail_on_validation_error", False)
                ),
                simulate=bool(scene_options.get("simulate", False)),
                simulate_mock_analyze=bool(
                    scene_options.get("simulate_mock_analyze", False)
                ),
                predict_max_workers=scene_options.get("predict_max_workers"),
                cancel_checker=is_scene_cancelled,
                event_listener=telemetry_listener,
                verbose=bool(scene_options.get("verbose", False)),
            )
        )
    except asyncio.CancelledError:
        local_cancel_event.set()
        raise
    finally:
        cancel_poll_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_poll_task

    stats = _extract_scene_stats(result)
    logger.info("Scene pipeline stats for %s: %s", session_id[:8], stats)
    coverage = build_not_evaluated_material_coverage(
        policy=normalize_coverage_policy(coverage_policy),
        warning=(
            "Large-scene prim-level prediction and binding evidence is not yet "
            "qualified; scene validation and asset totals are not substitutes "
            "for material coverage."
        ),
    )
    validation_report_path = _write_scene_validation_report(session_dir, result)
    scene_predictions_path = _write_scene_predictions_index(session_dir, result)

    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    step_timings_dict = {}
    for timing in telemetry_listener.get_step_timings():
        duration_ns = timing["completed_at_ns"] - timing["started_at_ns"]
        step_timings_dict[timing["name"]] = duration_ns / 1_000_000_000

    await _mirror_scene_outputs(
        session_manager,
        session_id,
        result,
        validation_report_path=validation_report_path,
        scene_predictions_path=scene_predictions_path,
    )

    validation_failure = (
        not result.success
        and result.validation_passed is False
        and result.validation_report is not None
    )
    if validation_failure:
        diagnostic = durable_diagnostic(
            "material_scene_validation_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        await session_manager.update_session(
            session_id,
            {
                "status": "failed",
                "pipeline_type": "large_scene",
                "error": diagnostic.code,
                "error_diagnostic": diagnostic.to_dict(),
                "failed_step": "scene_validate",
                "results": stats,
                "coverage": coverage,
                "partial_results": {"stats": stats, "coverage": coverage},
                "duration_seconds": duration_seconds,
                "step_timings": step_timings_dict,
                "scene": {
                    "working_dir": result.working_dir,
                    "manifest_path": result.manifest_path,
                    "output_usd_path": result.output_usd_path,
                    "rendered_images": result.rendered_images,
                    "validation_passed": result.validation_passed,
                    "validation_report_path": (
                        str(validation_report_path) if validation_report_path else ""
                    ),
                    "scene_predictions_path": (
                        str(scene_predictions_path) if scene_predictions_path else ""
                    ),
                    "warnings": result.warnings,
                },
                "failed_at": datetime.now(UTC).isoformat(),
            },
        )
        event_bus = get_event_bus()
        if event_bus.get_snapshot(session_id) is not None:
            try:
                await event_bus.emit_for_owner(
                    ProgressEvent(
                        session_id=session_id,
                        step="scene_validate",
                        state=StepState.FAILED,
                        percent=100,
                        message=diagnostic.code,
                        extra={
                            "pipeline_failed": True,
                            "coverage": coverage,
                            **stats,
                        },
                    ),
                    regeneration_claim=None,
                )
            except Exception:
                log_durable_failure(
                    logger,
                    "material_scene_validation_event_failed",
                    phase=FailurePhase.PIPELINE_EXECUTION,
                    retryable=False,
                )
        if span:
            span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
        _emit_step_spans(
            session_id=session_id,
            step_timings=telemetry_listener.get_step_timings(),
        )
        logger.info("Scene pipeline validation failed for %s", session_id[:8])
        return

    if not result.success:
        diagnostic = durable_diagnostic(
            "material_scene_pipeline_result_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        _emit_step_spans(
            session_id=session_id,
            step_timings=telemetry_listener.get_step_timings(),
        )
        if span:
            span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
        raise RuntimeError(diagnostic.code)

    await session_manager.update_session(
        session_id,
        {
            "status": "completed",
            "pipeline_type": "large_scene",
            "results": stats,
            "coverage": coverage,
            "duration_seconds": duration_seconds,
            "step_timings": step_timings_dict,
            "scene": {
                "working_dir": result.working_dir,
                "manifest_path": result.manifest_path,
                "output_usd_path": result.output_usd_path,
                "rendered_images": result.rendered_images,
                "validation_passed": result.validation_passed,
                "validation_report_path": (
                    str(validation_report_path) if validation_report_path else ""
                ),
                "scene_predictions_path": (
                    str(scene_predictions_path) if scene_predictions_path else ""
                ),
                "warnings": result.warnings,
            },
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is not None:
        try:
            await event_bus.emit_for_owner(
                ProgressEvent(
                    session_id=session_id,
                    step="scene_pipeline",
                    state=StepState.COMPLETED,
                    percent=100,
                    message="Large-scene pipeline completed successfully",
                    extra={
                        "pipeline_completed": True,
                        "coverage": coverage,
                        **stats,
                    },
                ),
                regeneration_claim=None,
            )
        except Exception:
            logger.exception(
                "Failed to emit scene completion event for session %s; "
                "metadata remains completed",
                session_id,
            )
    if span:
        span.set_attribute(MAAttributes.PIPELINE_STATUS, "completed")
        span.set_attribute(MAAttributes.PIPELINE_DURATION_SECONDS, duration_seconds)
        span.set_attribute(
            MAAttributes.PIPELINE_PRIM_COUNT,
            stats.get("original_prim_count", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PRIMS_PROCESSED,
            stats.get("prims_processed", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_IMAGES_GENERATED,
            stats.get("images_generated", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PREDICTIONS_MADE,
            stats.get("predictions_made", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_MATERIALS_APPLIED,
            stats.get("materials_applied", 0),
        )

    _emit_step_spans(
        session_id=session_id,
        step_timings=telemetry_listener.get_step_timings(),
    )
    logger.info("Scene pipeline execution completed for %s", session_id[:8])


async def _execute_pipeline_inner(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    user_email: str = "",
    coverage_policy: CoveragePolicy = "allow_partial",
    regeneration_claim: RegenerationClaim | None = None,
) -> None:
    """Inner pipeline execution logic.

    Args:
        session_id: Session identifier
        config_dict: Complete unified pipeline config dict (built in router)
        session_manager: SessionManager instance
        user_email: User email address for telemetry
    """
    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    # Get session directory for thumbnail creation
    session_dir = session_manager.get_session_dir(session_id)

    # Extract session-specific material icons from config
    # Materials are stored under config["materials"]["entries"]
    session_material_icons: dict[str, str] = {}
    materials_entries = config_dict.get("materials", {}).get("entries", [])
    for entry in materials_entries:
        if "name" in entry and "icon" in entry:
            session_material_icons[entry["name"]] = entry["icon"]

    logger.info(
        f"Loaded {len(session_material_icons)} material icons from config "
        f"(materials_entries count: {len(materials_entries)})"
    )
    if session_material_icons:
        # Log first 3 material icon mappings for debugging
        sample = list(session_material_icons.items())[:3]
        logger.info(f"Sample material icons: {sample}")

    # Create event listener with telemetry wrapper for per-step timing
    inner_listener = FastAPIEventListener(
        session_id,
        session_dir,
        session_material_icons=session_material_icons,
        regeneration_claim=regeneration_claim,
    )
    telemetry_listener = TelemetryEventListener(inner_listener)

    # Read asset metadata from session for telemetry
    metadata = await session_manager.get_session_metadata(session_id)
    asset_info = metadata.get("asset", {}) if metadata else {}

    vlm_model = (
        config_dict.get("steps", {}).get("predict", {}).get("vlm", {}).get("model", "")
    )

    # Set pipeline attributes on the root span from @traced decorator
    span = get_current_span()
    if span:
        span.set_attribute(MAAttributes.PIPELINE_SESSION_ID, session_id)
        # User email is sent only to our self-hosted Langfuse instance for
        # internal per-user pipeline tracing. No external data collection.
        span.set_attribute(MAAttributes.PIPELINE_USER_EMAIL, user_email)
        # Langfuse-specific: maps to native user/session fields for dashboard filtering
        span.set_attribute(MAAttributes.LANGFUSE_USER_ID, user_email)
        span.set_attribute(MAAttributes.LANGFUSE_SESSION_ID, session_id)
        if vlm_model:
            span.set_attribute(MAAttributes.PIPELINE_VLM_MODEL, vlm_model)
            span.set_attribute(MAAttributes.LANGFUSE_META_VLM_MODEL, vlm_model)
        for key, value in _cluster_telemetry_attributes(config_dict).items():
            span.set_attribute(key, value)
        if asset_info:
            if asset_info.get("filename"):
                span.set_attribute(MAAttributes.ASSET_FILENAME, asset_info["filename"])
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILENAME,
                    asset_info["filename"],
                )
            if asset_info.get("file_size_bytes"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_SIZE_BYTES,
                    asset_info["file_size_bytes"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_SIZE,
                    asset_info["file_size_bytes"],
                )
            if asset_info.get("file_extension"):
                span.set_attribute(
                    MAAttributes.ASSET_FILE_EXTENSION,
                    asset_info["file_extension"],
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_META_ASSET_FILE_EXT,
                    asset_info["file_extension"],
                )

    # Call async API directly - no wrapper or thread pool needed!
    baseline_signatures = _capture_promotable_file_signatures(session_dir)
    result = await arun_pipeline(
        PipelineInput(
            config=config_dict,
            event_listener=telemetry_listener,
            verbose=False,
        )
    )
    executed_completed_steps = _current_run_completed_steps(
        config_dict,
        result.completed_steps,
    )
    published_artifacts: dict[str, str] | None = (
        {} if regeneration_claim is not None else None
    )
    promoted_artifacts = await _promote_current_run_artifacts(
        session_manager,
        session_id,
        session_dir,
        executed_completed_steps,
        result.step_results,
        regeneration_claim=regeneration_claim,
        artifact_map=published_artifacts,
        baseline_signatures=baseline_signatures,
    )
    latest_metadata = await session_manager.get_session_metadata(session_id)
    artifact_validity = revalidate_artifacts_for_completed_steps(
        latest_metadata,
        executed_completed_steps,
        result.step_results,
        verified_artifacts=promoted_artifacts,
    )
    for promoted_artifact in promoted_artifacts:
        artifact_validity[promoted_artifact] = True
    if regeneration_claim is not None:
        assert published_artifacts is not None
        artifact_validity = await _carry_forward_regeneration_artifacts(
            session_manager,
            session_id,
            regeneration_claim,
            latest_metadata or {},
            artifact_validity,
            published_artifacts,
            set(config_dict.get("steps", {})),
        )

    if not result.success:
        diagnostic = durable_diagnostic(
            "material_pipeline_result_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        failure_snapshot = get_event_bus().get_snapshot(session_id)
        failure_updates: dict[str, Any] = {
            "artifact_validity": artifact_validity,
        }
        if artifact_validity["previews"] and failure_snapshot:
            failure_updates["preview_images"] = list(
                failure_snapshot.get("preview_images") or []
            )
        if regeneration_claim is None:
            persisted = await _update_pipeline_session(
                session_manager,
                session_id,
                failure_updates,
                regeneration_claim=None,
            )
        else:
            persisted = await _finalize_pipeline_session(
                session_manager,
                session_id,
                {
                    **failure_updates,
                    "status": "failed",
                    "error": diagnostic.code,
                    "error_diagnostic": diagnostic.to_dict(),
                    "failed_at": datetime.now(UTC).isoformat(),
                },
                regeneration_claim=regeneration_claim,
                artifact_map=published_artifacts,
            )
        if not persisted:
            raise asyncio.CancelledError("Regeneration claim was superseded")
        # Emit step spans before raising so they appear as children
        _emit_step_spans(
            session_id=session_id,
            step_timings=telemetry_listener.get_step_timings(),
            step_results=result.step_results,
        )
        # Set failure status on root span
        if span:
            span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
        if regeneration_claim is not None:
            # The regeneration claim was finalized above. Emit its sole
            # authoritative failure marker now; raising would make the outer
            # wrapper attempt a second finalization, see an inactive claim,
            # and skip the terminal SSE event.
            await _emit_persisted_pipeline_failure(
                session_id,
                diagnostic.code,
                step="pipeline",
                regeneration_claim=regeneration_claim,
            )
            return
        raise RuntimeError(diagnostic.code)

    # Extract stats from step results and save to session metadata
    # Debug: log available data
    logger.info("Pipeline completed %d step(s)", len(result.completed_steps or []))
    logger.info("Pipeline reported %d step result(s)", len(result.step_results or {}))
    if result.raw_result:
        logger.info("Pipeline raw_result contains %d field(s)", len(result.raw_result))

    stats = _extract_stats_from_result(result, session_dir)
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")
    coverage = build_material_coverage(
        result,
        session_dir,
        policy=normalize_coverage_policy(coverage_policy),
    )
    logger.info("Pipeline material coverage for %s: %s", session_id[:8], coverage)

    # Calculate duration
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    # Build step_timings dict for persistence
    step_timings_dict = {}
    for t in telemetry_listener.get_step_timings():
        duration_ns = t["completed_at_ns"] - t["started_at_ns"]
        step_timings_dict[t["name"]] = duration_ns / 1_000_000_000

    event_bus = get_event_bus()
    snapshot = event_bus.get_snapshot(session_id)
    overall_progress = dict(
        latest_metadata.get("overall_progress", {}) if latest_metadata else {}
    )
    completed_step_count = len(result.completed_steps)
    overall_progress["current_step"] = completed_step_count
    overall_progress["total_steps"] = max(
        int(overall_progress.get("total_steps", 0) or 0),
        completed_step_count,
    )
    overall_progress["percent"] = 100
    completed_steps = []
    if snapshot and snapshot.get("completed_steps"):
        completed_steps = snapshot["completed_steps"]
    elif latest_metadata and latest_metadata.get("completed_steps"):
        completed_steps = latest_metadata["completed_steps"]
    completed_steps = _merge_completed_steps_from_result(
        completed_steps,
        result,
        step_timings_dict,
    )
    preview_images = list(
        (snapshot or {}).get("preview_images")
        or (latest_metadata or {}).get("preview_images")
        or []
    )

    if coverage["policy"] == "strict" and not coverage_is_release_ready(coverage):
        error_message = (
            "Strict material coverage qualification failed: "
            f"readiness={coverage['readiness_grade']}, "
            f"predictions={coverage['usable_prediction_count']} usable + "
            f"{coverage['fallback_count']} fallback / {coverage['target_count']} "
            f"targets, bindings={coverage['bound_count']} / "
            f"{coverage['target_count']}."
        )
        persisted = await _finalize_pipeline_session(
            session_manager,
            session_id,
            {
                "status": "failed",
                "error": error_message,
                "failed_step": "coverage_validation",
                "results": stats,
                "coverage": coverage,
                "partial_results": {"stats": stats, "coverage": coverage},
                "duration_seconds": duration_seconds,
                "step_timings": step_timings_dict,
                "current_step": None,
                "overall_progress": overall_progress,
                "failed_at": datetime.now(UTC).isoformat(),
                "artifact_validity": artifact_validity,
                "preview_images": preview_images,
                **(
                    {"completed_steps": to_json_safe(completed_steps)}
                    if completed_steps
                    else {}
                ),
            },
            regeneration_claim=regeneration_claim,
            artifact_map=published_artifacts,
        )
        if not persisted:
            raise asyncio.CancelledError("Regeneration claim was superseded")
        if event_bus.get_snapshot(session_id) is not None:
            try:
                await event_bus.emit_for_owner(
                    ProgressEvent(
                        session_id=session_id,
                        step="coverage_validation",
                        state=StepState.FAILED,
                        percent=100,
                        message=error_message,
                        extra={
                            "pipeline_failed": True,
                            "coverage": coverage,
                        },
                    ),
                    regeneration_claim=regeneration_claim,
                )
            except Exception:
                logger.exception(
                    "Failed to emit coverage failure event for session %s; "
                    "metadata remains failed",
                    session_id,
                )
        await _mark_standard_terminal_events_quiesced(
            session_manager,
            session_id,
            regeneration_claim=regeneration_claim,
            expected_status="failed",
        )
        if span:
            span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
        _emit_step_spans(
            session_id=session_id,
            step_timings=telemetry_listener.get_step_timings(),
            step_results=result.step_results,
        )
        logger.warning("Pipeline coverage qualification failed for %s", session_id[:8])
        return

    # Save results to session metadata (include status to ensure atomicity
    # with results - prevents race where EventBus sets status="completed"
    # before stats are persisted)
    persisted = await _finalize_pipeline_session(
        session_manager,
        session_id,
        {
            "status": "completed",
            "results": stats,
            "coverage": coverage,
            "duration_seconds": duration_seconds,
            "step_timings": step_timings_dict,
            "current_step": None,
            "overall_progress": overall_progress,
            "completed_at": datetime.now(UTC).isoformat(),
            "artifact_validity": artifact_validity,
            "preview_images": preview_images,
            **(
                {"completed_steps": to_json_safe(completed_steps)}
                if completed_steps
                else {}
            ),
        },
        regeneration_claim=regeneration_claim,
        artifact_map=published_artifacts,
    )
    if not persisted:
        raise asyncio.CancelledError("Regeneration claim was superseded")

    # Force the in-memory progress snapshot to terminal state after metadata is
    # persisted. Otherwise /status can prefer a stale running EventBus snapshot
    # over the completed session metadata.
    if event_bus.get_snapshot(session_id) is not None:
        try:
            await event_bus.emit_for_owner(
                ProgressEvent(
                    session_id=session_id,
                    step="pipeline",
                    state=StepState.COMPLETED,
                    percent=100,
                    message="Pipeline completed successfully",
                    extra={
                        "pipeline_completed": True,
                        "coverage": coverage,
                        **stats,
                    },
                ),
                regeneration_claim=regeneration_claim,
            )
        except Exception:
            logger.exception(
                "Failed to emit completion event for session %s; "
                "metadata remains completed",
                session_id,
            )
    await _mark_standard_terminal_events_quiesced(
        session_manager,
        session_id,
        regeneration_claim=regeneration_claim,
        expected_status="completed",
    )

    # Set completion attributes on root span
    if span:
        span.set_attribute(MAAttributes.PIPELINE_STATUS, "completed")
        span.set_attribute(MAAttributes.PIPELINE_DURATION_SECONDS, duration_seconds)
        span.set_attribute(
            MAAttributes.PIPELINE_PRIM_COUNT,
            stats.get("original_prim_count", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PRIMS_PROCESSED,
            stats.get("prims_processed", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_IMAGES_GENERATED,
            stats.get("images_generated", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_PREDICTIONS_MADE,
            stats.get("predictions_made", 0),
        )
        span.set_attribute(
            MAAttributes.PIPELINE_MATERIALS_APPLIED,
            stats.get("materials_applied", 0),
        )
        for key, value in _cluster_telemetry_attributes(config_dict, stats).items():
            span.set_attribute(key, value)

    # Emit per-step child spans from TelemetryEventListener
    _emit_step_spans(
        session_id=session_id,
        step_timings=telemetry_listener.get_step_timings(),
        step_results=result.step_results,
    )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")


def _merge_completed_steps_from_result(
    completed_steps: list[dict[str, Any]],
    result: Any,
    step_timings: dict[str, float],
) -> list[dict[str, Any]]:
    """Merge authoritative pipeline output steps into a status step list."""
    merged = to_json_safe(completed_steps)
    if not isinstance(merged, list):
        merged = []
    seen = {step.get("name") for step in merged if isinstance(step, dict)}
    now = datetime.now(UTC).isoformat()
    step_results = result.step_results or {}

    for step_name in result.completed_steps or []:
        if step_name in seen:
            continue
        duration = int(step_timings.get(step_name, 0))
        outputs = step_results.get(step_name, {})
        safe_outputs = project_result_metadata(outputs)
        merged.append(
            {
                "name": step_name,
                "display_name": _STEP_DISPLAY_NAMES.get(step_name, step_name),
                "started_at": now,
                "completed_at": now,
                "duration_seconds": duration,
                "stats": bounded_step_completion_data(
                    {
                        "step_name": step_name,
                        "outputs": to_json_safe(safe_outputs),
                    }
                ),
            }
        )
        seen.add(step_name)

    return merged


def _emit_step_spans(
    session_id: str,
    step_timings: list[dict] | None = None,
    step_results: dict[str, Any] | None = None,
) -> None:
    """Emit OTel child spans for each pipeline step.

    These appear as children of the current active span (the root
    ``maa.pipeline.execution`` span created by the ``@traced`` decorator
    on ``execute_pipeline_async``).
    """
    if not step_timings:
        return

    try:
        from world_understanding.telemetry import get_tracer

        tracer = get_tracer(__name__)
        for timing in step_timings:
            step_name = timing["name"]
            step_status = timing["status"]
            duration_ns = timing["completed_at_ns"] - timing["started_at_ns"]
            duration_secs = duration_ns / 1_000_000_000

            with tracer.start_as_current_span(
                f"maa.pipeline.step.{step_name}"
            ) as step_span:
                step_span.set_attribute(MAAttributes.PIPELINE_STEP_NAME, step_name)
                step_span.set_attribute(MAAttributes.PIPELINE_STEP_STATUS, step_status)
                step_span.set_attribute(
                    MAAttributes.PIPELINE_STEP_DURATION_SECONDS, duration_secs
                )
                step_span.set_attribute(MAAttributes.PIPELINE_SESSION_ID, session_id)
                if timing.get("error"):
                    step_span.set_attribute(
                        MAAttributes.PIPELINE_STEP_ERROR, timing["error"]
                    )
                for key, value in _cluster_telemetry_attributes(
                    {},
                    (step_results or {}).get(step_name, {}),
                    step_name=step_name,
                ).items():
                    step_span.set_attribute(key, value)

    except Exception as e:
        logger.warning(f"Failed to emit step telemetry: {e}")


def _extract_scene_stats(result: ScenePipelineOutput) -> dict[str, Any]:
    """Extract service-facing statistics from a scene pipeline result."""
    raw_result = result.raw_result or {}
    sub_assets_detected = int(raw_result.get("sub_assets") or 0)
    payload_groups_detected = int(raw_result.get("payload_groups") or 0)
    assets_completed = result.completed_assets + result.completed_payloads
    assets_failed = result.failed_assets + result.failed_payloads
    failed_items = _extract_failed_scene_items(result.manifest_path)
    asset_image_count = _extract_scene_asset_image_count(result.manifest_path)
    scene_render_count = len(result.rendered_images)
    validation_report = result.validation_report or {}
    validation_errors = (
        len(validation_report.get("errors", []))
        if isinstance(validation_report, dict)
        else 0
    )
    validation_warnings = (
        len(validation_report.get("warnings", []))
        if isinstance(validation_report, dict)
        else 0
    )

    return {
        "pipeline_type": "large_scene",
        # Keep the legacy stats keys populated so existing clients can render
        # completed scene jobs without branching on a new response model.
        "original_prim_count": sub_assets_detected,
        "prims_processed": assets_completed,
        "images_generated": asset_image_count + scene_render_count,
        "predictions_made": assets_completed,
        "materials_applied": assets_completed,
        "scene_sub_assets_detected": sub_assets_detected,
        "scene_sub_assets_completed": result.completed_assets,
        "scene_sub_assets_failed": result.failed_assets,
        "scene_payload_groups_detected": payload_groups_detected,
        "scene_payload_groups_completed": result.completed_payloads,
        "scene_payload_groups_failed": result.failed_payloads,
        "scene_assets_completed": assets_completed,
        "scene_assets_failed": assets_failed,
        "scene_failed_items": failed_items,
        "scene_asset_image_count": asset_image_count,
        "scene_render_count": scene_render_count,
        "scene_validation_passed": result.validation_passed,
        "scene_validation_errors": validation_errors,
        "scene_validation_warnings": validation_warnings,
        "scene_warnings": len(result.warnings),
    }


def _extract_failed_scene_items(manifest_path_raw: str) -> list[dict[str, Any]]:
    if not manifest_path_raw:
        return []

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.exists():
        return []

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load scene manifest for failed item stats: %s", exc)
        return []

    if not isinstance(manifest_data, dict):
        return []

    failed_items: list[dict[str, Any]] = []
    for item in manifest_data.get("sub_assets", []):
        if isinstance(item, dict) and item.get("status") == "failed":
            failed_items.append(
                {
                    "source_type": "sub_asset",
                    "source_id": item.get("id"),
                    "source_name": item.get("name"),
                    "source_prim_path": item.get("prim_path"),
                }
            )

    for item in manifest_data.get("payload_groups", []):
        if isinstance(item, dict) and item.get("status") == "failed":
            failed_items.append(
                {
                    "source_type": "payload_group",
                    "source_id": item.get("id"),
                    "source_name": item.get("group_name"),
                    "source_payload_file": item.get("payload_file"),
                }
            )

    return failed_items


def _extract_scene_asset_image_count(manifest_path_raw: str) -> int:
    """Count per-asset render images recorded by scene child pipelines."""
    if not manifest_path_raw:
        return 0

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.exists():
        return 0

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load scene manifest for image stats: %s", exc)
        return 0

    if not isinstance(manifest_data, dict):
        return 0

    total = 0
    for section_name in ("sub_assets", "payload_groups"):
        for item in manifest_data.get(section_name, []):
            if isinstance(item, dict):
                total += _extract_scene_item_image_count(item)
    return total


def _extract_scene_item_image_count(item: dict[str, Any]) -> int:
    working_dir_raw = item.get("working_dir")
    if not working_dir_raw:
        return 0

    working_dir = Path(str(working_dir_raw))
    state_path = working_dir / ".pipeline_state.json"
    if state_path.exists():
        try:
            with open(state_path, encoding="utf-8") as f:
                state_data = json.load(f)
            step_outputs = state_data.get("step_outputs", {})
            if isinstance(step_outputs, dict):
                usd_outputs = step_outputs.get("build_dataset_usd", {})
                if isinstance(usd_outputs, dict):
                    num_images = usd_outputs.get("num_images")
                    if isinstance(num_images, int) and num_images >= 0:
                        return num_images
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load scene pipeline state %s: %s", state_path, exc
            )

    renders_dir = working_dir / "dataset" / "usd" / "renders"
    if not renders_dir.exists():
        return 0
    try:
        return len(list(renders_dir.glob("**/*.png")))
    except OSError as exc:
        logger.warning(
            "Failed to count scene render images in %s: %s", renders_dir, exc
        )
        return 0


def _write_scene_validation_report(
    session_dir: Path,
    result: ScenePipelineOutput,
) -> Path | None:
    """Persist structured validation output for artifact serving."""
    if result.validation_report is None:
        return None

    report_path = session_dir / "scene" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.validation_report, f, indent=2)
        f.write("\n")
    return report_path


def _write_scene_predictions_index(
    session_dir: Path,
    result: ScenePipelineOutput,
) -> Path | None:
    """Collate per-asset scene predictions into one service artifact."""
    if not result.manifest_path:
        return None

    manifest_path = Path(result.manifest_path)
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load scene manifest for predictions index: %s", exc)
        return None

    if not isinstance(manifest_data, dict):
        return None

    output_path = session_dir / "scene" / "predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for item in manifest_data.get("sub_assets", []):
            if isinstance(item, dict):
                written += _write_scene_prediction_records(
                    out,
                    item,
                    source_type="sub_asset",
                )
        for item in manifest_data.get("payload_groups", []):
            if isinstance(item, dict):
                written += _write_scene_prediction_records(
                    out,
                    item,
                    source_type="payload_group",
                )

    if written == 0:
        output_path.unlink(missing_ok=True)
        return None

    return output_path


def _write_scene_prediction_records(
    output_file: Any,
    source: dict[str, Any],
    *,
    source_type: str,
) -> int:
    predictions_path_raw = source.get("predictions_path")
    if not predictions_path_raw:
        return 0

    predictions_path = Path(str(predictions_path_raw))
    if not predictions_path.exists():
        return 0

    source_record = {
        "source_type": source_type,
        "source_id": source.get("id"),
        "source_name": source.get("name") or source.get("group_name"),
        "source_prim_path": source.get("prim_path"),
        "source_payload_file": source.get("payload_file"),
        "source_predictions_key": f"{source_type}:{source.get('id') or 'unknown'}",
    }
    written = 0
    try:
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid scene prediction in %s", line[:80])
                    continue
                json.dump({**source_record, "prediction": prediction}, output_file)
                output_file.write("\n")
                written += 1
    except OSError as exc:
        logger.warning("Failed to read scene predictions %s: %s", predictions_path, exc)
        return 0

    return written


async def _mirror_scene_outputs(
    session_manager: SessionManager,
    session_id: str,
    result: ScenePipelineOutput,
    validation_report_path: Path | None = None,
    scene_predictions_path: Path | None = None,
) -> None:
    """Mirror scene outputs to the configured session store when available."""
    await _mirror_scene_artifact(
        session_manager,
        session_id,
        result.output_usd_path,
        "output/scene_with_materials.usd",
        content_type="application/octet-stream",
    )
    if result.output_usd_path:
        composed_flat_path = (
            Path(result.output_usd_path).parent / "composed_scene_flat.usd"
        )
        await _mirror_scene_artifact(
            session_manager,
            session_id,
            str(composed_flat_path),
            "output/scene_with_materials_flat.usd",
            content_type="application/octet-stream",
        )
    await _mirror_scene_artifact(
        session_manager,
        session_id,
        result.manifest_path,
        "scene/manifest.json",
        content_type="application/json",
    )
    if validation_report_path is not None:
        await _mirror_scene_artifact(
            session_manager,
            session_id,
            str(validation_report_path),
            "scene/validation_report.json",
            content_type="application/json",
        )
    if scene_predictions_path is not None:
        await _mirror_scene_artifact(
            session_manager,
            session_id,
            str(scene_predictions_path),
            "scene/predictions.jsonl",
            content_type="application/x-ndjson",
        )
    for image in result.rendered_images:
        image_path = Path(image)
        await _mirror_scene_artifact(
            session_manager,
            session_id,
            image,
            f"output/{image_path.name}",
            content_type="image/png",
        )
    if result.rendered_images:
        await _mirror_scene_artifact(
            session_manager,
            session_id,
            result.rendered_images[0],
            "output/scene_with_materials.png",
            content_type="image/png",
        )


async def _mirror_scene_artifact(
    session_manager: SessionManager,
    session_id: str,
    file_path: str,
    key: str,
    content_type: str | None = None,
) -> None:
    if not file_path:
        return
    path = Path(file_path)
    if not path.exists():
        return
    try:
        await session_manager.put_file_to_store(
            session_id,
            key,
            str(path),
            content_type=content_type,
        )
    except Exception:
        log_durable_failure(
            logger,
            "scene_artifact_sync_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )


def _cluster_telemetry_attributes(
    config_dict: dict[str, Any],
    stats: dict[str, Any] | None = None,
    *,
    step_name: str | None = None,
) -> dict[str, bool | int | float | str]:
    """Build sanitized telemetry attributes for prim clustering."""
    stats = stats or {}
    steps = config_dict.get("steps", {}) if config_dict else {}
    cluster_config = steps.get("cluster_prims", {}) if isinstance(steps, dict) else {}
    enabled = bool(cluster_config) or bool(stats.get("cluster_prims_ran", False))

    attrs: dict[str, bool | int | float | str] = {}
    if step_name is None:
        attrs[MAAttributes.CLUSTERING_ENABLED] = enabled

    if cluster_config:
        attrs[MAAttributes.CLUSTER_EMBEDDING_BACKEND] = str(
            cluster_config.get("embedding_service", "")
        )
        attrs[MAAttributes.CLUSTER_EMBEDDING_MODEL] = str(
            cluster_config.get("embedding_model", "")
        )

    metric_keys = {
        MAAttributes.CLUSTER_TOTAL_PRIMS: "cluster_total_prims",
        MAAttributes.CLUSTER_COUNT: "cluster_count",
        MAAttributes.CLUSTER_REPRESENTATIVE_COUNT: "cluster_representative_count",
        MAAttributes.CLUSTER_REDUCTION_PERCENT: "cluster_reduction_percent",
        MAAttributes.CLUSTER_MULTI_MEMBER_COUNT: "cluster_multi_member_count",
        MAAttributes.CLUSTER_SINGLETON_COUNT: "cluster_singleton_count",
        MAAttributes.CLUSTER_MAX_SIZE: "cluster_max_size",
        MAAttributes.CLUSTER_CAPPED_COUNT: "cluster_capped_count",
    }
    for attr_name, stat_name in metric_keys.items():
        value = stats.get(stat_name)
        if value is not None:
            attrs[attr_name] = value

    return attrs


def _extract_stats_from_result(
    result: Any,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract statistics from pipeline result.

    Args:
        result: PipelineOutput from arun_pipeline
        session_dir: Optional session directory for file-based fallback

    Returns:
        Dictionary with extracted stats
    """
    stats: dict[str, Any] = {
        "original_prim_count": 0,
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
        "materials_applied": 0,
        "cluster_prims_ran": False,
        "cluster_total_prims": 0,
        "cluster_count": 0,
        "cluster_representative_count": 0,
        "cluster_reduction_percent": 0.0,
        "cluster_multi_member_count": 0,
        "cluster_singleton_count": 0,
        "cluster_max_size": None,
        "cluster_capped_count": 0,
    }

    # Use step_results for basic info
    step_results = result.step_results or {}

    # Extract predictions_count from predict step (this field IS extracted)
    if "predict" in step_results:
        stats["predictions_made"] = step_results["predict"].get("predictions_count", 0)
    elif "benchmark" in step_results:
        stats["predictions_made"] = step_results["benchmark"].get(
            "predictions_count", 0
        )

    # Extract materials_applied count from apply step
    if "apply" in step_results:
        materials_applied = step_results["apply"].get("materials_applied", {})
        # materials_applied is a dict mapping material names to prim paths
        stats["materials_applied"] = (
            len(materials_applied) if isinstance(materials_applied, dict) else 0
        )

    # For num_prims and num_images, we need to look at raw_result
    # which contains the full workflow context
    raw_result = result.raw_result or {}

    # First, try to get the ORIGINAL prim count (before optimization)
    # This is stored in pipeline_results by the optimize_usd step
    original_prim_count = 0

    # Check pipeline_results (where step outputs are stored)
    pipeline_results = raw_result.get("pipeline_results", {})
    logger.debug("pipeline_results contains %d step(s)", len(pipeline_results))

    if "cluster_prims" in pipeline_results:
        cluster_outputs = pipeline_results["cluster_prims"]
        stats["cluster_prims_ran"] = bool(
            cluster_outputs.get("cluster_prims_ran", False)
        )
        for key in (
            "cluster_total_prims",
            "cluster_count",
            "cluster_representative_count",
            "cluster_multi_member_count",
            "cluster_singleton_count",
            "cluster_max_size",
            "cluster_capped_count",
        ):
            stats[key] = cluster_outputs.get(key, stats[key])
        stats["cluster_reduction_percent"] = cluster_outputs.get(
            "cluster_reduction_percent", 0.0
        )

    if "optimize_usd" in pipeline_results:
        optimize_outputs = pipeline_results["optimize_usd"]
        original_prim_count = optimize_outputs.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from optimize_outputs: {original_prim_count}"
        )
        # Also check optimization_metadata within the step outputs
        if original_prim_count == 0 and "optimization_metadata" in optimize_outputs:
            opt_metadata = optimize_outputs["optimization_metadata"]
            if opt_metadata:
                original_prim_count = opt_metadata.get("original_prim_count", 0)
                logger.debug(
                    f"original_prim_count from optimization_metadata: {original_prim_count}"
                )

    # Fallback: check optimization_metadata at top level
    if original_prim_count == 0 and "optimization_metadata" in raw_result:
        opt_metadata = raw_result["optimization_metadata"]
        original_prim_count = opt_metadata.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from top-level optimization_metadata: {original_prim_count}"
        )

    # Fallback: check context directly
    if original_prim_count == 0:
        original_prim_count = raw_result.get("original_prim_count", 0)
        logger.debug(
            f"original_prim_count from raw_result directly: {original_prim_count}"
        )

    logger.info(f"Final original_prim_count: {original_prim_count}")

    # Store original prim count (before optimization)
    stats["original_prim_count"] = original_prim_count

    # Try to get stats from pipeline_results (where step outputs are stored)
    # Check build_dataset_usd step outputs
    if "build_dataset_usd" in pipeline_results:
        usd_result = pipeline_results["build_dataset_usd"]
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)
        logger.debug(
            f"From build_dataset_usd: prims={stats['prims_processed']}, images={stats['images_generated']}"
        )

    # Legacy fallback: check for build_dataset_usd_result directly in raw_result
    if stats["prims_processed"] == 0 and "build_dataset_usd_result" in raw_result:
        usd_result = raw_result["build_dataset_usd_result"]
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)

    # Check build_dataset_prepare_dataset step outputs
    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset" in pipeline_results
    ):
        prepare_result = pipeline_results["build_dataset_prepare_dataset"]
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    # Legacy fallback: check for build_dataset_prepare_dataset_result
    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset_result" in raw_result
    ):
        prepare_result = raw_result["build_dataset_prepare_dataset_result"]
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    # If still no prims count, try to get from dataset loading
    if stats["prims_processed"] == 0:
        # Count from completed_steps or dataset
        dataset_info = raw_result.get("dataset_info", {})
        stats["prims_processed"] = dataset_info.get("num_entries", 0)

    # Fallback: count from actual files in session directory
    if session_dir and (
        stats["prims_processed"] == 0
        or stats["images_generated"] == 0
        or stats["predictions_made"] == 0
        or (stats["cluster_prims_ran"] and stats["cluster_count"] == 0)
    ):
        stats = _count_stats_from_files(session_dir, stats)

    # If optimize_usd didn't run, original_prim_count equals prims_processed
    # (no optimization means original == processed)
    if stats["original_prim_count"] == 0 and stats["prims_processed"] > 0:
        stats["original_prim_count"] = stats["prims_processed"]
        logger.debug(
            f"Set original_prim_count to prims_processed (no optimization): {stats['original_prim_count']}"
        )

    return stats


def _count_stats_from_files(
    session_dir: Path | None,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Count stats from actual files in session directory.

    Args:
        session_dir: Path to session directory
        stats: Current stats dict to update

    Returns:
        Updated stats dict
    """
    if session_dir is None:
        return stats
    session_path = Path(session_dir)

    # Count entries from dataset.jsonl
    if stats["prims_processed"] == 0:
        dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["prims_processed"] = len(lines)
                    logger.info(
                        f"Counted {stats['prims_processed']} entries from dataset.jsonl"
                    )
            except Exception as e:
                logger.warning(f"Failed to count dataset entries: {e}")

    # Count images from dataset directory
    if stats["images_generated"] == 0:
        dataset_dir = session_path / "cache" / "dataset"
        if dataset_dir.exists():
            try:
                image_count = len(list(dataset_dir.glob("**/*.png")))
                stats["images_generated"] = image_count
                logger.info(f"Counted {image_count} images in dataset directory")
            except Exception as e:
                logger.warning(f"Failed to count images: {e}")

    # Count predictions from predictions.jsonl
    if stats["predictions_made"] == 0:
        predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["predictions_made"] = len(lines)
                    logger.info(
                        f"Counted {stats['predictions_made']} predictions from predictions.jsonl"
                    )
            except Exception as e:
                logger.warning(f"Failed to count predictions: {e}")

    cluster_map_file = session_path / "cache" / "clusters" / "cluster_map.jsonl"
    if stats.get("cluster_prims_ran") and stats.get("cluster_count", 0) == 0:
        if cluster_map_file.exists():
            try:
                import json

                cluster_rows = []
                with open(cluster_map_file) as f:
                    cluster_rows = [json.loads(line) for line in f if line.strip()]
                cluster_ids = {row["cluster_id"] for row in cluster_rows}
                representatives = [
                    row for row in cluster_rows if row.get("is_representative")
                ]
                multi_member = {
                    row["cluster_id"]
                    for row in cluster_rows
                    if int(row.get("cluster_size", 1)) > 1
                }
                total = len(cluster_rows)
                cluster_count = len(cluster_ids)
                stats["cluster_total_prims"] = total
                stats["cluster_count"] = cluster_count
                stats["cluster_representative_count"] = len(representatives)
                stats["cluster_multi_member_count"] = len(multi_member)
                stats["cluster_singleton_count"] = max(
                    0, cluster_count - len(multi_member)
                )
                stats["cluster_reduction_percent"] = (
                    round(100.0 * (1 - cluster_count / total), 3) if total else 0.0
                )
                logger.info("Counted %s clusters from cluster_map.jsonl", cluster_count)
            except Exception as e:
                logger.warning(f"Failed to count cluster map: {e}")

    return stats
