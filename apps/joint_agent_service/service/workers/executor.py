# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution using Joint Agent Python async API.

Calls arun_pipeline directly - no wrappers or thread pools needed!
"""

import asyncio
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import zipfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from joint_agent.api import PipelineInput, arun_pipeline
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_directory_at,
    remove_confined_tree,
    validated_artifact_relative_key,
)
from world_understanding.utils.durable_diagnostics import (
    DurableDiagnostic,
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)
from world_understanding.utils.result_projection import project_result_metadata

from ..events.listener import FastAPIEventListener
from ..progress import step_display_name
from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..session.cache_publications import (
    CACHE_NAMESPACES,
    CACHE_PUBLICATIONS_FIELD,
    cache_publication_path,
    cache_publication_prefix,
    parse_cache_publications,
)
from ..session.manager import (
    JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD,
    JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS,
    JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD,
    JOINT_RIGGER_PUBLICATION_PREFIX,
    SessionManager,
)

logger = logging.getLogger(__name__)


class PipelineRunSupersededError(RuntimeError):
    """Raised when an executor no longer owns the accepted session run."""


_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}
_RUN_CANCELLATION_POLL_SECONDS = 1.0
_DATASET_PRODUCER_STEPS = {
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
}
_PREDICTION_PRODUCER_STEPS = {
    "predict",
    "consistency_pass",
    "infer_articulation_candidates",
}


def _pipeline_failure_diagnostic(failed_step: str) -> DurableDiagnostic:
    """Project one executor failure into a bounded durable diagnostic."""

    if failed_step == "artifact_sync":
        return durable_diagnostic(
            "joint_pipeline_artifact_sync_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )
    if failed_step == "artifact_publication":
        return durable_diagnostic(
            "joint_pipeline_artifact_publication_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=False,
        )
    if failed_step == "pipeline_completion":
        return durable_diagnostic(
            "joint_pipeline_completion_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
    return durable_diagnostic(
        "joint_pipeline_execution_failed",
        phase=FailurePhase.PIPELINE_EXECUTION,
        retryable=False,
    )


def _produced_cache_namespaces(
    only_steps: list[str] | None,
) -> tuple[str, ...]:
    """Return cache namespaces this execution can actually change."""
    if not only_steps:
        return CACHE_NAMESPACES

    requested = set(only_steps)
    namespaces: list[str] = []
    if requested.intersection(_PREDICTION_PRODUCER_STEPS):
        namespaces.append("predictions")
    if requested.intersection(_DATASET_PRODUCER_STEPS):
        namespaces.append("dataset")
    return tuple(namespaces)


def _completed_cache_namespaces(
    only_steps: list[str] | None,
    completed_steps: set[str],
) -> tuple[str, ...]:
    """Return namespaces whose requested producer steps all completed."""

    requested_steps = set(
        only_steps
        or [
            *_DATASET_PRODUCER_STEPS,
            *_PREDICTION_PRODUCER_STEPS,
        ]
    )
    namespaces: list[str] = []
    requested_predictions = requested_steps.intersection(_PREDICTION_PRODUCER_STEPS)
    if requested_predictions and requested_predictions.issubset(completed_steps):
        namespaces.append("predictions")
    requested_dataset = requested_steps.intersection(_DATASET_PRODUCER_STEPS)
    if requested_dataset and requested_dataset.issubset(completed_steps):
        namespaces.append("dataset")
    return tuple(namespaces)


def _confined_cache_cleanup_target(
    path: Path,
    cache_root: Path,
) -> tuple[Path, Path, str]:
    """Return one lexical cache child and its validated relative key."""

    confined_root = Path(os.path.abspath(cache_root))
    target = Path(os.path.abspath(path))
    try:
        relative = target.relative_to(confined_root)
    except ValueError:
        raise ArtifactPathError(
            "Cache cleanup target is outside the cache root"
        ) from None
    if not relative.parts:
        raise ArtifactPathError("Cache cleanup cannot remove the cache root")
    try:
        relative_key = validated_artifact_relative_key(relative.as_posix())
    except ValueError:
        raise ArtifactPathError("Cache cleanup target is not canonical") from None
    return confined_root, target, relative_key


def _unlink_confined_cache_entry(cache_root: Path, relative_key: str) -> None:
    """Unlink one non-directory cache entry through held directory descriptors."""

    parts = tuple(relative_key.split("/"))

    def unlink_from(parent_descriptor: int) -> None:
        try:
            metadata = os.stat(
                parts[-1],
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if stat.S_ISDIR(metadata.st_mode):
            raise ArtifactPathError("Cache cleanup target changed type")
        os.unlink(parts[-1], dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)

    try:
        with open_confined_directory(cache_root) as root_descriptor:
            if len(parts) == 1:
                unlink_from(root_descriptor)
            else:
                with open_confined_directory_at(
                    root_descriptor,
                    "/".join(parts[:-1]),
                ) as parent_descriptor:
                    unlink_from(parent_descriptor)
    except FileNotFoundError:
        return
    except ArtifactPathError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise ArtifactPathError("Cache cleanup could not be confined") from None


def _remove_cache_path(path: Path, cache_root: Path) -> None:
    """Remove one cache child without following a swapped path component."""

    confined_root, target, relative_key = _confined_cache_cleanup_target(
        path,
        cache_root,
    )
    try:
        remove_confined_tree(target, confined_root)
        return
    except ValueError:
        # The confined recursive helper deliberately accepts directories only.
        # Preserve the historical file/symlink behavior via a held parent dirfd.
        pass
    except (OSError, RuntimeError):
        raise ArtifactPathError("Cache cleanup could not be confined") from None
    _unlink_confined_cache_entry(confined_root, relative_key)


def _prepare_cache_namespaces_for_run(
    session_dir: Path,
    only_steps: list[str] | None,
) -> None:
    """Remove outputs that this run must not inherit from an older generation."""

    cache_root = session_dir / "cache"
    _remove_cache_path(cache_root / ".pipeline_state.json", cache_root)
    requested = set(
        only_steps
        or [
            *_DATASET_PRODUCER_STEPS,
            *_PREDICTION_PRODUCER_STEPS,
        ]
    )
    dataset_dir = cache_root / "dataset"
    predictions_dir = cache_root / "predictions"

    if "build_dataset_usd" in requested:
        _remove_cache_path(dataset_dir, cache_root)
    elif "build_dataset_prepare_dataset" in requested:
        _remove_cache_path(dataset_dir / "dataset.jsonl", cache_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if "predict" in requested:
        _remove_cache_path(predictions_dir, cache_root)
    elif requested.intersection({"consistency_pass", "infer_articulation_candidates"}):
        for filename in (
            "articulation_candidates.json",
            "articulation_candidates.html",
            "articulation_candidate_adjudications.json",
        ):
            _remove_cache_path(predictions_dir / filename, cache_root)
        if "consistency_pass" in requested:
            _remove_cache_path(
                predictions_dir / "predictions.stats.json",
                cache_root,
            )
    predictions_dir.mkdir(parents=True, exist_ok=True)


def _pipeline_checkpoint_progress(session_dir: Path) -> tuple[set[str], str | None]:
    """Recover completed and failed steps from this run's pipeline checkpoint."""

    checkpoint_path = session_dir / "cache" / ".pipeline_state.json"
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set(), None
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Could not read pipeline checkpoint: %s", checkpoint_path)
        return set(), None
    if not isinstance(checkpoint, dict):
        logger.warning("Ignoring invalid pipeline checkpoint: %s", checkpoint_path)
        return set(), None

    completed_value = checkpoint.get("completed_steps")
    completed_steps = (
        {step for step in completed_value if isinstance(step, str)}
        if isinstance(completed_value, list)
        else set()
    )
    failed_value = checkpoint.get("failed_steps")
    failed_steps = (
        [step for step in failed_value if isinstance(step, str)]
        if isinstance(failed_value, list)
        else []
    )
    return completed_steps, failed_steps[-1] if failed_steps else None


def _validate_cache_snapshot_tree(path: Path) -> bool:
    """Reject links and special files; return whether a regular file exists."""

    if not path.exists():
        return False
    root_metadata = path.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise RuntimeError(f"Cache namespace is not a safe directory: {path}")
    has_files = False
    for entry in path.rglob("*"):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"Refusing to publish symlinked cache artifact: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Unsupported cache artifact type: {entry}")
        has_files = True
    return has_files


def _materialize_cache_publications(
    session_dir: Path,
    run_id: str,
    namespaces: tuple[str, ...],
) -> dict[str, str]:
    """Move run-owned mutable caches into generation-specific publications."""

    published: dict[str, str] = {}
    for namespace in namespaces:
        source_dir = session_dir / "cache" / namespace
        publication_dir = cache_publication_path(session_dir, run_id, namespace)
        if not publication_dir.exists():
            if not _validate_cache_snapshot_tree(source_dir):
                _remove_cache_path(source_dir, session_dir / "cache")
                continue
            publication_dir.parent.mkdir(parents=True, exist_ok=True)
            source_dir.rename(publication_dir)
        if _validate_cache_snapshot_tree(publication_dir):
            published[namespace] = run_id
    return published


async def _publish_cache_publications(
    session_manager,
    session_id: str,
    run_id: str,
    only_steps: list[str] | None,
    namespaces: tuple[str, ...] | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Upload immutable cache snapshots and return their validated bindings."""

    if namespaces is None:
        namespaces = _produced_cache_namespaces(only_steps)
    session_dir = session_manager.get_session_dir(session_id)
    published = await asyncio.to_thread(
        _materialize_cache_publications,
        session_dir,
        run_id,
        namespaces,
    )
    for namespace in published:
        await session_manager.sync_to_store(
            session_id,
            prefix=cache_publication_prefix(run_id, namespace),
            overwrite=False,
        )
    return set(namespaces), published


async def _merged_cache_publications(
    session_manager,
    session_id: str,
    produced: set[str],
    published: dict[str, str],
) -> dict[str, str]:
    metadata = await session_manager.get_session_metadata(session_id)
    existing = parse_cache_publications(metadata or {}) or {}
    for namespace in produced:
        existing.pop(namespace, None)
    existing.update(published)
    return existing


async def execute_pipeline_async(
    session_id: str,
    run_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    only_steps: list[str] | None = None,
) -> None:
    """Execute a pipeline without exporting its implementation exception graph."""
    failure_message: str | None = None
    superseded = False
    cancelled = False
    try:
        await _execute_pipeline_async_impl(
            session_id,
            run_id,
            config_dict,
            session_manager,
            only_steps,
        )
        return
    except asyncio.CancelledError:
        cancelled = True
    except PipelineRunSupersededError:
        superseded = True
    except RuntimeError as error:
        # The implementation emits only code-owned RuntimeError messages. Its
        # traceback still contains runtime config and service collaborators, so
        # reconstruct the public exception after discarding that entire graph.
        failure_message = str(error)

    del config_dict, only_steps, run_id, session_id, session_manager
    if cancelled:
        raise asyncio.CancelledError()
    if superseded:
        raise PipelineRunSupersededError("Pipeline run no longer owns the session")
    assert failure_message is not None
    raise RuntimeError(failure_message)


async def _execute_pipeline_async_impl(
    session_id: str,
    run_id: str,
    config_dict: dict[str, Any],
    session_manager: SessionManager,
    only_steps: list[str] | None = None,
) -> None:
    """Execute pipeline workflow using Joint Agent Python async API."""
    logger.info(f"Pipeline execution started for {session_id[:8]}...")
    failure_step = "pipeline"
    failure_message: str | None = None
    stats: dict[str, Any] | None = None
    result: Any = None
    initial_metadata: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    terminal_committed = False
    completed_steps: set[str] = set()
    upgrade_legacy_cache = False
    pipeline_cancel_event = threading.Event()
    try:
        await _require_current_run(session_manager, session_id, run_id)
        initial_metadata = await session_manager.get_session_metadata(session_id)
        upgrade_legacy_cache = parse_cache_publications(initial_metadata or {}) is None
        session_dir = session_manager.get_session_dir(session_id)
        await _await_inflight_work(
            asyncio.to_thread(
                _prepare_cache_namespaces_for_run,
                session_dir,
                only_steps,
            )
        )
        await _await_inflight_work(
            asyncio.to_thread(_clear_previous_joint_rigger_artifacts, session_dir)
        )

        listener = FastAPIEventListener(
            session_id,
            session_dir,
        )

        result = await _await_inflight_work(
            arun_pipeline(
                PipelineInput(
                    config=config_dict,
                    event_listener=listener,
                    only_steps=only_steps or [],
                    verbose=False,
                    cancel_checker=pipeline_cancel_event.is_set,
                )
            ),
            on_cancel=pipeline_cancel_event.set,
        )

        completed_steps = {
            step
            for step in (getattr(result, "completed_steps", None) or [])
            if isinstance(step, str)
        }
        checkpoint_completed, checkpoint_failed = await _await_inflight_work(
            asyncio.to_thread(_pipeline_checkpoint_progress, session_dir)
        )
        completed_steps.update(checkpoint_completed)
        if not result.success:
            if checkpoint_failed is not None:
                failure_step = checkpoint_failed
            raise RuntimeError("Pipeline failed")

        logger.info(
            "Pipeline completed %d step(s)",
            len(result.completed_steps or []),
        )
        if result.step_results:
            logger.info(
                "Pipeline reported %d step result(s)",
                len(result.step_results),
            )

        stats = project_result_metadata(_extract_stats_from_result(result, session_dir))
        logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")

        metadata = await session_manager.get_session_metadata(session_id)
        duration_seconds = 0
        if metadata and metadata.get("created_at"):
            created_at = datetime.fromisoformat(metadata["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

        await _require_current_run(session_manager, session_id, run_id)
        failure_step = "artifact_sync"
        synced = 0
        sync_failed = False
        try:
            produced_cache_namespaces, published_cache = await _await_inflight_work(
                _publish_cache_publications(
                    session_manager,
                    session_id,
                    run_id,
                    only_steps,
                    namespaces=(CACHE_NAMESPACES if upgrade_legacy_cache else None),
                )
            )
        except Exception:
            sync_failed = True
            log_durable_failure(
                logger,
                "joint_pipeline_cache_publication_failed",
                phase=FailurePhase.SYNC_UPLOAD,
                retryable=True,
            )
        if sync_failed:
            raise RuntimeError("Failed to sync result artifacts")
        cache_publications = await _merged_cache_publications(
            session_manager,
            session_id,
            produced_cache_namespaces,
            published_cache,
        )
        failure_step = "artifact_publication"
        artifact_prefixes: list[str] = []
        await _await_inflight_work(
            asyncio.to_thread(_materialize_joint_rigger_download_usdz, session_dir)
        )
        joint_rigger_source_keys = _joint_rigger_artifact_keys(session_dir)
        joint_rigger_artifact_keys: dict[str, str] = {}
        publication_id: str | None = None
        if joint_rigger_source_keys:
            publication_id = uuid4().hex
            joint_rigger_artifact_keys = await _await_inflight_work(
                asyncio.to_thread(
                    _publish_joint_rigger_artifacts,
                    session_dir,
                    joint_rigger_source_keys,
                    publication_id,
                )
            )
        stats[JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD] = joint_rigger_artifact_keys
        if publication_id is not None:
            stats[JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD] = publication_id
        stats["joint_rigger_artifacts"] = {
            artifact_type: artifact_type in joint_rigger_artifact_keys
            for artifact_type in JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS
        }
        if publication_id is not None:
            artifact_prefixes.append(
                f"{JOINT_RIGGER_PUBLICATION_PREFIX}/{publication_id}/"
            )

        failure_step = "artifact_sync"
        for prefix in artifact_prefixes:
            try:
                n = await _await_inflight_work(
                    session_manager.sync_to_store(
                        session_id,
                        prefix=prefix,
                        overwrite=True,
                    )
                )
                synced += n
            except Exception:
                sync_failed = True
                log_durable_failure(
                    logger,
                    "joint_pipeline_artifact_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
                )
        if sync_failed:
            raise RuntimeError("Failed to sync result artifacts")
        if synced > 0:
            logger.info(
                f"Synced {synced} artifact file(s) to store for {session_id[:8]}"
            )

        await _require_current_run(session_manager, session_id, run_id)
        failure_step = "pipeline_completion"
        accepted = await _await_inflight_work(
            session_manager.terminalize_and_release_run(
                session_id,
                run_id,
                {
                    "status": "completed",
                    "completed_steps": _completed_step_records(result, metadata),
                    "results": stats,
                    CACHE_PUBLICATIONS_FIELD: cache_publications,
                    "duration_seconds": duration_seconds,
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
        )
        if not accepted:
            raise PipelineRunSupersededError(
                f"Pipeline run {run_id} no longer owns session {session_id}"
            )
        terminal_committed = True

        event_bus = get_event_bus()
        if event_bus.get_snapshot(session_id) is not None:
            try:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="pipeline",
                        state=StepState.COMPLETED,
                        percent=100,
                        message="Pipeline artifacts synced and ready",
                        extra={"pipeline_ready": True, "pipeline_completed": True},
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log_durable_failure(
                    logger,
                    "joint_pipeline_completion_event_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )

        logger.info(f"Pipeline execution completed for {session_id[:8]}")
    except asyncio.CancelledError:
        if not terminal_committed:
            cancelled_cache_publications = (
                await _persist_failure_prerequisites_without_masking(
                    session_manager,
                    session_id,
                    run_id,
                    only_steps,
                    completed_steps,
                    upgrade_legacy_cache,
                )
            )
            await _record_run_failure(
                session_manager,
                session_id,
                run_id,
                failed_step="cancelled",
                error_message="Pipeline run was cancelled",
                status="cancelled",
                partial_results=stats,
                cache_publications=cancelled_cache_publications,
            )
        raise
    except PipelineRunSupersededError:
        raise
    except Exception:
        failure_message = _pipeline_failure_message(failure_step)
        diagnostic = _pipeline_failure_diagnostic(failure_step)
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase(diagnostic.phase),
            retryable=diagnostic.retryable,
        )
        if not terminal_committed:
            failure_cache_publications = (
                await _persist_failure_prerequisites_without_masking(
                    session_manager,
                    session_id,
                    run_id,
                    only_steps,
                    completed_steps,
                    upgrade_legacy_cache,
                )
            )
            try:
                await _record_run_failure(
                    session_manager,
                    session_id,
                    run_id,
                    failed_step=failure_step,
                    diagnostic=diagnostic,
                    partial_results=stats,
                    cache_publications=failure_cache_publications,
                )
            except Exception:
                log_durable_failure(
                    logger,
                    "joint_pipeline_failure_record_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )

    if failure_message is not None:
        # The runtime pipeline must receive the original config, but a rejected
        # operation and its result must not remain reachable from the public
        # replacement exception's traceback frames.
        del config_dict
        del result
        del stats
        del initial_metadata
        del metadata
        raise RuntimeError(failure_message)


def _pipeline_failure_message(failure_step: str) -> str:
    """Return a value-free public failure category for one pipeline stage."""

    if failure_step == "artifact_sync":
        return "Failed to sync result artifacts"
    if failure_step == "artifact_publication":
        return "Failed to publish result artifacts"
    if failure_step == "pipeline_completion":
        return "Failed to finalize pipeline run"
    return "Pipeline failed"


async def finalize_pipeline_run(
    session_manager,
    session_id: str,
    run_id: str,
) -> None:
    """Make pre-start cancellation durable, then release the owned run claim."""

    metadata = await session_manager.get_session_metadata(session_id)
    if metadata and metadata.get("status") not in _TERMINAL_RUN_STATUSES:
        error_message = "Pipeline run was cancelled before completion"
        accepted = await _await_inflight_work(
            session_manager.terminalize_and_release_run(
                session_id,
                run_id,
                {
                    "status": "cancelled",
                    "error": error_message,
                    "failed_step": "cancelled",
                },
            )
        )
        if accepted:
            await _emit_pipeline_failed(
                session_id,
                "cancelled",
                error_message,
                status="cancelled",
            )
    else:
        await _await_inflight_work(session_manager.release_run(session_id, run_id))


async def _await_inflight_work[T](
    awaitable: Awaitable[T],
    *,
    on_cancel: Callable[[], None] | None = None,
) -> T:
    """Delay outer cancellation until already-started work has actually stopped."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        if task.done():
            raise
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not task.cancelled():
            try:
                task.result()
            except BaseException:
                pass
        raise


async def _require_current_run(session_manager, session_id: str, run_id: str) -> None:
    if not await session_manager.is_run_current(session_id, run_id):
        raise PipelineRunSupersededError(
            f"Pipeline run {run_id} no longer owns session {session_id}"
        )


async def _persist_failure_prerequisites_without_masking(
    session_manager,
    session_id: str,
    run_id: str,
    only_steps: list[str] | None,
    completed_steps: set[str],
    upgrade_legacy_cache: bool,
) -> dict[str, str] | None:
    """Best-effort durable cache sync before recording a primary terminal error."""

    async def sync_owned_prerequisites() -> dict[str, str] | None:
        if not await session_manager.is_run_current(session_id, run_id):
            logger.info(
                "Skipping failure artifact sync for superseded run %s/%s",
                session_id[:8],
                run_id[:8],
            )
            return None

        checkpoint_completed, _ = await asyncio.to_thread(
            _pipeline_checkpoint_progress,
            session_manager.get_session_dir(session_id),
        )
        effective_completed_steps = completed_steps.union(checkpoint_completed)
        safe_namespaces = set(
            _completed_cache_namespaces(only_steps, effective_completed_steps)
        )
        if upgrade_legacy_cache:
            safe_namespaces.update(
                set(CACHE_NAMESPACES).difference(_produced_cache_namespaces(only_steps))
            )
        ordered_namespaces = tuple(
            namespace for namespace in CACHE_NAMESPACES if namespace in safe_namespaces
        )
        produced, published = await _publish_cache_publications(
            session_manager,
            session_id,
            run_id,
            only_steps,
            namespaces=ordered_namespaces,
        )
        if not await session_manager.is_run_current(session_id, run_id):
            logger.info(
                "Run %s/%s was superseded during immutable cache publication",
                session_id[:8],
                run_id[:8],
            )
            return None
        return await _merged_cache_publications(
            session_manager,
            session_id,
            produced,
            published,
        )

    try:
        return await _await_inflight_work(sync_owned_prerequisites())
    except asyncio.CancelledError:
        # _await_inflight_work has already drained the sync task. Preserve the
        # original pipeline cancellation as the terminal cause.
        logger.debug(
            "Additional cancellation arrived while preserving failure artifacts for %s",
            session_id[:8],
        )
        return None
    except Exception:  # pragma: no cover - inner operations are guarded
        log_durable_failure(
            logger,
            "joint_failure_prerequisite_publication_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )
        return None


async def _record_run_failure(
    session_manager,
    session_id: str,
    run_id: str,
    *,
    failed_step: str,
    error_message: str | None = None,
    diagnostic: DurableDiagnostic | None = None,
    status: str = "failed",
    partial_results: dict[str, Any] | None = None,
    cache_publications: dict[str, str] | None = None,
) -> None:
    if diagnostic is not None:
        durable_error = diagnostic.code
    elif error_message is not None:
        durable_error = error_message
    else:
        raise ValueError("A failure message or diagnostic is required")
    updates: dict[str, Any] = {
        "status": status,
        "error": durable_error,
        "failed_step": failed_step,
    }
    if diagnostic is not None:
        updates["error_diagnostic"] = diagnostic.to_dict()
    if partial_results is not None:
        updates["partial_results"] = partial_results
    if cache_publications is not None:
        updates[CACHE_PUBLICATIONS_FIELD] = cache_publications
    accepted = await _await_inflight_work(
        session_manager.terminalize_and_release_run(
            session_id,
            run_id,
            updates,
        )
    )
    if accepted:
        await _emit_pipeline_failed(
            session_id,
            failed_step,
            durable_error,
            status=status,
            diagnostic=diagnostic,
        )


async def maintain_run_claim(
    session_manager,
    session_id: str,
    run_id: str,
) -> None:
    """Renew one run claim for the registry's full queued and active lifecycle."""

    loop = asyncio.get_running_loop()
    heartbeat_seconds = session_manager.run_claim_heartbeat_seconds
    next_renewal = loop.time() + heartbeat_seconds
    while True:
        await asyncio.sleep(
            min(
                _RUN_CANCELLATION_POLL_SECONDS,
                max(0.0, next_renewal - loop.time()),
            )
        )
        try:
            cancellation_accepted = await session_manager.is_cancellation_accepted(
                session_id,
                run_id,
            )
        except Exception:
            log_durable_failure(
                logger,
                "joint_pipeline_cancellation_poll_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
        else:
            if cancellation_accepted:
                raise asyncio.CancelledError(
                    f"Pipeline run {run_id} was cancelled for session {session_id}"
                )
        if loop.time() < next_renewal:
            continue
        if not await session_manager.renew_run(session_id, run_id):
            raise PipelineRunSupersededError(
                f"Pipeline run {run_id} lost its lease for session {session_id}"
            )
        next_renewal = loop.time() + heartbeat_seconds


def _clear_previous_joint_rigger_artifacts(session_dir: Path) -> None:
    """Remove stale Joint Rigger outputs before a rerun reuses the session dir."""
    cache_root = session_dir / "cache"
    _remove_cache_path(cache_root / "joint_rigger", cache_root)


async def _emit_pipeline_failed(
    session_id: str,
    failed_step: str,
    error_message: str,
    *,
    status: str = "failed",
    diagnostic: DurableDiagnostic | None = None,
) -> None:
    """Keep same-instance status/SSE snapshots aligned with terminal metadata."""
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is None:
        return
    cancelled = status == "cancelled"
    extra: dict[str, Any] = {
        "cancelled_step" if cancelled else "failed_step": failed_step
    }
    if diagnostic is not None:
        extra["error_diagnostic"] = diagnostic.to_dict()
    try:
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=failed_step,
                state=StepState.CANCELLED if cancelled else StepState.FAILED,
                message=error_message,
                extra=extra,
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "joint_pipeline_failure_event_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )


def _completed_step_records(result, metadata: dict[str, Any] | None) -> list[dict]:
    """Build durable completed-step records from the pipeline result."""
    existing_steps = []
    if metadata:
        existing_steps = [
            step
            for step in metadata.get("completed_steps", [])
            if isinstance(step, dict) and step.get("name")
        ]
    records_by_name = {str(step["name"]): dict(step) for step in existing_steps}

    step_results = result.step_results or {}
    completed_at = datetime.now(UTC).isoformat()
    for step_name in result.completed_steps or []:
        step_name = str(step_name)
        if step_name in records_by_name:
            continue
        step_stats = step_results.get(step_name, {})
        if not isinstance(step_stats, dict):
            step_stats = {}
        safe_step_stats = project_result_metadata(step_stats)
        records_by_name[step_name] = {
            "name": step_name,
            "display_name": step_display_name(step_name),
            "started_at": completed_at,
            "completed_at": completed_at,
            "duration_seconds": 0,
            "stats": safe_step_stats,
        }
    return list(records_by_name.values())


def _materialize_joint_rigger_download_usdz(session_dir: Path) -> None:
    """Package a supported raw Joint output into the one-file service contract."""

    joint_rigger_dir = session_dir / "cache" / "joint_rigger"
    package_path = joint_rigger_dir / "rigged.usdz"
    if package_path.exists() or package_path.is_symlink():
        if package_path.is_file() and not package_path.is_symlink():
            return
        raise RuntimeError(
            f"Joint Rigger package target is not a safe file: {package_path}"
        )

    raw_output = joint_rigger_dir / "rigged.usd"
    if not raw_output.exists() and not raw_output.is_symlink():
        return
    if not raw_output.is_file() or raw_output.is_symlink():
        raise RuntimeError(f"Joint Rigger raw output is not a safe file: {raw_output}")

    from joint_agent.functions.joint_rigger_adapter import _package_usdz_for_handoff
    from pxr import Usd, UsdUtils
    from world_understanding.functions.physics.joint_rigger.reference import (
        local_usd_dependency_paths,
    )
    from world_understanding.utils.usd.package import find_usdz_root_layer

    # Fail before packaging when the source closure is unresolved or unstable.
    closure_validation_failed = False
    stage = None
    unresolved = None
    try:
        stage = Usd.Stage.Open(str(raw_output))
        if not stage:
            raise RuntimeError(f"Could not open Joint Rigger raw USD: {raw_output}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(raw_output))
        if unresolved:
            raise RuntimeError(
                "Joint Rigger raw USD has unresolved dependencies: "
                + ", ".join(sorted(str(path) for path in unresolved))
            )
    except Exception:
        closure_validation_failed = True
    if closure_validation_failed:
        # Reject the unsafe exception graph after its handler has exited and
        # remove raw path/parser objects from the replacement traceback frame.
        del stage
        del unresolved
        del raw_output
        del package_path
        del joint_rigger_dir
        del session_dir
        raise RuntimeError("Could not validate Joint Rigger raw USD closure")
    with tempfile.NamedTemporaryFile(
        prefix=".rigged-service-",
        suffix=".usdz",
        dir=joint_rigger_dir,
        delete=False,
    ) as temp_file:
        temp_package = Path(temp_file.name)
    temp_package.unlink(missing_ok=True)
    try:
        _package_usdz_for_handoff(raw_output, temp_package)
        if not zipfile.is_zipfile(temp_package):
            raise RuntimeError(
                f"Joint Rigger packaging wrote a non-ZIP artifact: {temp_package}"
            )
        find_usdz_root_layer(temp_package)
        package_closure = local_usd_dependency_paths(temp_package)
        expected_closure = (temp_package.resolve(),)
        if package_closure != expected_closure:
            raise RuntimeError(
                "Joint Rigger download package is not self-contained: "
                + ", ".join(str(path) for path in package_closure)
            )
        temp_package.replace(package_path)
    finally:
        temp_package.unlink(missing_ok=True)


def _joint_rigger_artifact_keys(session_dir: Path) -> dict[str, str]:
    """Return exact current-run artifact keys captured before completion."""

    artifact_keys: dict[str, str] = {}
    for artifact_type, candidates in JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS.items():
        for relative_path in candidates:
            artifact_path = session_dir / relative_path
            if artifact_path.is_file() and not artifact_path.is_symlink():
                artifact_keys[artifact_type] = relative_path
                break
    return artifact_keys


def _publish_joint_rigger_artifacts(
    session_dir: Path,
    source_keys: dict[str, str],
    publication_id: str,
) -> dict[str, str]:
    """Publish the full Joint output tree while preserving relative locators."""

    source_dir = session_dir / "cache" / "joint_rigger"
    source_metadata = source_dir.lstat()
    if not stat.S_ISDIR(source_metadata.st_mode) or stat.S_ISLNK(
        source_metadata.st_mode
    ):
        raise RuntimeError(
            f"Joint Rigger output root is not a safe directory: {source_dir}"
        )
    publication_dir = session_dir / JOINT_RIGGER_PUBLICATION_PREFIX / publication_id
    publication_dir.mkdir(parents=True, exist_ok=False)
    try:
        for source_path in sorted(source_dir.rglob("*")):
            relative_path = source_path.relative_to(source_dir)
            published_path = publication_dir / relative_path
            metadata = source_path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(
                    f"Refusing to publish symlinked Joint Rigger artifact: {source_path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                published_path.mkdir()
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"Unsupported Joint Rigger artifact type: {source_path}"
                )
            published_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, published_path)
            except OSError:
                shutil.copy2(source_path, published_path)

        published_relative_paths: dict[str, Path] = {}
        for artifact_type, source_key in source_keys.items():
            source_relative_path = Path(source_key).relative_to(
                Path("cache") / "joint_rigger"
            )
            if (
                artifact_type == "joint_rigger_output"
                and source_relative_path.suffix.lower() != ".usdz"
            ):
                raise RuntimeError(
                    "Joint Rigger output must be packaged as self-contained USDZ "
                    "before immutable publication"
                )
            published_relative_paths[artifact_type] = source_relative_path

        published_keys: dict[str, str] = {}
        for artifact_type, relative_path in published_relative_paths.items():
            published_key = (
                f"{JOINT_RIGGER_PUBLICATION_PREFIX}/{publication_id}/"
                f"{relative_path.as_posix()}"
            )
            if not (session_dir / published_key).is_file():
                raise RuntimeError(
                    f"Joint Rigger publication is missing {artifact_type}: {published_key}"
                )
            published_keys[artifact_type] = published_key
        return published_keys
    except BaseException:
        shutil.rmtree(publication_dir, ignore_errors=True)
        raise


def _extract_stats_from_result(result, session_dir=None) -> dict[str, Any]:
    """Extract statistics from pipeline result."""
    stats: dict[str, Any] = {
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
        "articulation_candidates": 0,
    }

    step_results = result.step_results or {}
    has_articulation_candidate_count = False

    if "predict" in step_results:
        stats["predictions_made"] = step_results["predict"].get("predictions_count", 0)
    if "infer_articulation_candidates" in step_results:
        stats["articulation_candidates"] = step_results[
            "infer_articulation_candidates"
        ].get("articulation_candidate_count", 0)
        has_articulation_candidate_count = True
    if "apply_joint_rigger" in step_results:
        apply_result = step_results["apply_joint_rigger"]
        stats["joint_rigger_authored_joints"] = int(
            apply_result.get("authored_joint_count") or 0
        )
        stats["joint_rigger_status"] = str(
            apply_result.get("joint_rigger_status") or "unknown"
        )
        stats["joint_rigger_skipped"] = bool(
            apply_result.get("apply_joint_rigger_skipped")
        )

    raw_result = result.raw_result or {}

    if "build_dataset_usd_result" in raw_result:
        usd_result = raw_result["build_dataset_usd_result"]
        stats["prims_processed"] = usd_result.get("num_prims", 0)
        stats["images_generated"] = usd_result.get("num_images", 0)

    if (
        stats["prims_processed"] == 0
        and "build_dataset_prepare_dataset_result" in raw_result
    ):
        prepare_result = raw_result["build_dataset_prepare_dataset_result"]
        stats["prims_processed"] = prepare_result.get("num_entries", 0)

    if stats["prims_processed"] == 0:
        dataset_info = raw_result.get("dataset_info", {})
        stats["prims_processed"] = dataset_info.get("num_entries", 0)

    if session_dir and (
        stats["prims_processed"] == 0
        or stats["predictions_made"] == 0
        or not has_articulation_candidate_count
    ):
        stats = _count_stats_from_files(
            session_dir,
            stats,
            count_articulation_candidates=not has_articulation_candidate_count,
        )

    return stats


def _count_stats_from_files(
    session_dir: str | Path,
    stats: dict[str, Any],
    *,
    count_articulation_candidates: bool = True,
) -> dict[str, Any]:
    """Count stats from actual files in session directory."""
    session_path = Path(session_dir)

    if stats["prims_processed"] == 0:
        dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["prims_processed"] = len(lines)
            except Exception:
                logger.warning("Failed to count dataset entries")

    if stats["images_generated"] == 0:
        dataset_dir = session_path / "cache" / "dataset"
        if dataset_dir.exists():
            try:
                image_count = len(list(dataset_dir.glob("**/*.png")))
                stats["images_generated"] = image_count
            except Exception:
                logger.warning("Failed to count images")

    if stats["predictions_made"] == 0:
        predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["predictions_made"] = len(lines)
            except Exception:
                logger.warning("Failed to count predictions")

    if count_articulation_candidates:
        candidates_file = (
            session_path / "cache" / "predictions" / "articulation_candidates.json"
        )
        if candidates_file.exists():
            try:
                import json

                with open(candidates_file) as f:
                    payload = json.load(f)
                stats["articulation_candidates"] = int(
                    payload.get("summary", {}).get("candidate_count", 0)
                )
            except Exception:
                logger.warning("Failed to count articulation candidates")

    return stats
