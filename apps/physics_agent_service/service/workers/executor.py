# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution using Physics Agent Python async API.

Calls arun_pipeline directly - no wrappers or thread pools needed!
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from physics_agent.api import PipelineInput, arun_pipeline
from world_understanding.utils.durable_diagnostics import (
    DurableDiagnostic,
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)
from world_understanding.utils.model_auth import MODEL_AUTHENTICATION_FAILURE_MESSAGE

from ..events.listener import FastAPIEventListener
from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..session.manager import SessionManager
from ..utils import derive_completed_step_names

logger = logging.getLogger(__name__)

_TERMINAL_PERSIST_ATTEMPTS = 3
_TERMINAL_PERSIST_RETRY_DELAY_SECONDS = 0.05
_CANCELLATION_POLL_INTERVAL_SECONDS = 1.0


async def execute_pipeline_async(
    session_id: str,
    config_dict: dict,
    session_manager: SessionManager,
    only_steps: list[str] | None = None,
) -> None:
    """Execute one pipeline and terminalize cancellation from any lifecycle phase."""
    try:
        await _execute_pipeline_lifecycle(
            session_id,
            config_dict,
            session_manager,
            only_steps=only_steps,
        )
    except asyncio.CancelledError:
        try:
            await terminalize_pipeline_cancellation(
                session_manager,
                session_id,
            )
        except Exception:  # noqa: BLE001
            log_durable_failure(
                logger,
                "physics_pipeline_cancellation_record_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
        raise


async def _execute_pipeline_lifecycle(
    session_id: str,
    config_dict: dict,
    session_manager: SessionManager,
    only_steps: list[str] | None = None,
) -> None:
    """Run the complete pipeline lifecycle, including cooperative cancellation."""
    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    session_dir = session_manager.get_session_dir(session_id)

    listener = FastAPIEventListener(
        session_id,
        session_dir,
        suppress_failure_events=True,
    )
    cancel_event = Event()
    cancel_watcher = asyncio.create_task(
        _watch_for_cancel(session_manager, session_id, cancel_event)
    )
    lifecycle_returned = False
    try:
        await _execute_pipeline_with_cancel_signal(
            session_id,
            config_dict,
            session_manager,
            session_dir,
            listener,
            cancel_event,
            only_steps=only_steps,
        )
        lifecycle_returned = True
    finally:
        cancel_watcher.cancel()
        try:
            await cancel_watcher
        except asyncio.CancelledError:
            pass
        if lifecycle_returned and await _cancellation_requested(
            session_manager,
            session_id,
            cancel_event,
        ):
            metadata = await session_manager.get_session_metadata(session_id) or {}
            if metadata.get("status") != "cancelled":
                await _mark_cancelled(
                    session_manager,
                    session_id,
                    listener.canonical_current_step or "pipeline",
                )


async def _execute_pipeline_with_cancel_signal(
    session_id: str,
    config_dict: dict,
    session_manager: SessionManager,
    session_dir: Path,
    listener: FastAPIEventListener,
    cancel_event: Event,
    *,
    only_steps: list[str] | None,
) -> None:
    """Execute and commit a terminal state while the cancel bridge remains live."""

    execution_failure: DurableDiagnostic | None = None
    failed_completed_steps: list[str] = []
    failed_partial_results: dict[str, Any] | None = None
    result: Any = None
    pipeline_task = asyncio.create_task(
        arun_pipeline(
            PipelineInput(
                config=config_dict,
                event_listener=listener,
                only_steps=only_steps or [],
                verbose=False,
                cancel_event=cancel_event,
            )
        )
    )
    try:
        result = await asyncio.shield(pipeline_task)
    except asyncio.CancelledError:
        # Task.arun delegates synchronous pipeline steps to asyncio.to_thread.
        # Cancelling the await cannot stop that worker thread, so ask it to
        # stop between steps and do not publish cancellation until it is quiet.
        cancel_event.set()
        await _wait_for_pipeline_quiescence(pipeline_task)
        raise
    except Exception:  # noqa: BLE001
        execution_failure = _pipeline_failure_diagnostic()
    else:
        try:
            if not result.success:
                execution_failure = _pipeline_failure_diagnostic(result.error)
                failed_completed_steps = list(result.completed_steps or [])
                failed_partial_results = dict(result.step_results or {})
        except Exception:  # noqa: BLE001
            execution_failure = _pipeline_failure_diagnostic()

    if bool(getattr(result, "cancelled", False)) or await _cancellation_requested(
        session_manager,
        session_id,
        cancel_event,
    ):
        await _mark_cancelled(
            session_manager,
            session_id,
            listener.canonical_current_step or "pipeline",
            completed_steps=list(getattr(result, "completed_steps", []) or []),
            partial_results=dict(getattr(result, "step_results", {}) or {}),
        )
        return

    if execution_failure is not None:
        log_durable_failure(
            logger,
            execution_failure.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        terminal_status = await _mark_failed(
            session_manager,
            session_id,
            execution_failure,
            listener.canonical_current_step or "pipeline",
            completed_steps=failed_completed_steps,
            partial_results=failed_partial_results,
        )
        if terminal_status != "failed":
            return
        raise RuntimeError(execution_failure.code)

    logger.info("Pipeline completed %d step(s)", len(result.completed_steps or []))
    if result.step_results:
        logger.info("Pipeline reported %d step result(s)", len(result.step_results))

    stats = _extract_stats_from_result(result, session_dir)
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")

    metadata = await session_manager.get_session_metadata(session_id)
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    if await _cancellation_requested(session_manager, session_id, cancel_event):
        await _mark_cancelled(
            session_manager,
            session_id,
            listener.canonical_current_step or "pipeline",
            completed_steps=list(result.completed_steps or []),
            partial_results=dict(result.step_results or {}),
        )
        return

    terminal_status = await session_manager.claim_pipeline_terminal_state(
        session_id,
        "completed",
    )
    if terminal_status == "cancelled":
        await _mark_cancelled(
            session_manager,
            session_id,
            listener.canonical_current_step or "pipeline",
            completed_steps=list(result.completed_steps or []),
            partial_results=dict(result.step_results or {}),
        )
        return
    if terminal_status != "completed":
        return

    await _persist_claimed_terminal_metadata(
        session_manager,
        session_id,
        {
            "status": "completed",
            "results": stats,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now(UTC).isoformat(),
            "can_cancel": False,
        },
        failure_code="physics_pipeline_completion_metadata_failed",
    )

    # Sync key artifacts to store (uploads to S3 if configured).
    # Only sync the result files — skip rendered images (can be thousands of PNGs)
    # which are too large to upload reliably and not needed cross-instance.
    synced = 0
    for prefix in (
        "cache/predictions/",
        "cache/dataset/dataset.jsonl",
        "cache/physics/",
    ):
        try:
            n = await session_manager.sync_to_store(session_id, prefix=prefix)
            synced += n
        except Exception as e:
            logger.warning(
                f"Failed to sync {prefix} to store for {session_id[:8]}: {e}"
            )
    if synced > 0:
        logger.info(f"Synced {synced} artifact file(s) to store for {session_id[:8]}")

    # Signal SSE clients that artifacts are now in the store and the pipeline is fully done.
    # This fires AFTER update_session + sync_to_store so clients get "done" only when
    # status and artifacts are already available in S3.
    # Guard: only emit if this instance built up a snapshot (i.e., was the executing instance).
    # Avoids creating a stale empty snapshot on cross-instance calls or in tests.
    event_bus = get_event_bus()
    if event_bus.get_snapshot(session_id) is not None:
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="pipeline",
                state=StepState.COMPLETED,
                percent=100,
                message="Pipeline artifacts synced and ready",
                extra={"pipeline_ready": True},
            )
        )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")


async def _watch_for_cancel(
    session_manager: SessionManager,
    session_id: str,
    cancel_event: Event,
) -> None:
    """Bridge the durable cross-instance cancel marker to the worker signal."""
    try:
        while not cancel_event.is_set():
            if await _cancellation_requested(
                session_manager,
                session_id,
                cancel_event,
            ):
                return
            await asyncio.sleep(_CANCELLATION_POLL_INTERVAL_SECONDS)
    except asyncio.CancelledError:
        return


async def _cancellation_requested(
    session_manager: SessionManager,
    session_id: str,
    cancel_event: Event,
) -> bool:
    """Observe durable cancellation and publish it on the executing instance."""
    if cancel_event.is_set():
        return True
    try:
        if not await session_manager.is_cancelled(session_id):
            return False
    except Exception:  # noqa: BLE001
        log_durable_failure(
            logger,
            "physics_pipeline_cancellation_poll_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        return False

    cancel_event.set()
    await get_event_bus().mark_cancelling(session_id)
    return True


async def _wait_for_pipeline_quiescence(
    pipeline_task: asyncio.Task[Any],
) -> None:
    """Wait until thread-backed pipeline work exits after cancellation."""
    while not pipeline_task.done():
        try:
            await asyncio.shield(pipeline_task)
        except asyncio.CancelledError:
            # A repeated cancellation request must not release the registry
            # slot while the underlying worker can still mutate artifacts.
            continue
        except Exception:  # noqa: BLE001
            break

    if pipeline_task.cancelled():
        return
    try:
        pipeline_task.result()
    except Exception:  # noqa: BLE001
        log_durable_failure(
            logger,
            "physics_pipeline_cancellation_quiescence_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )


def _active_pipeline_step(session_id: str) -> str:
    """Return the canonical active EventBus step for cancellation metadata."""
    snapshot = get_event_bus().get_snapshot(session_id) or {}
    current_step = snapshot.get("current_step")
    if isinstance(current_step, dict) and isinstance(current_step.get("name"), str):
        return current_step["name"]
    return "pipeline"


def _pipeline_failure_diagnostic(error: str | None = None) -> DurableDiagnostic:
    code = (
        "physics_pipeline_model_authentication_failed"
        if error == MODEL_AUTHENTICATION_FAILURE_MESSAGE
        else "physics_pipeline_execution_failed"
    )
    return durable_diagnostic(
        code,
        phase=FailurePhase.PIPELINE_EXECUTION,
        retryable=False,
    )


async def _mark_failed(
    session_manager: SessionManager,
    session_id: str,
    diagnostic: DurableDiagnostic,
    failed_step: str,
    *,
    completed_steps: list[str] | None = None,
    partial_results: dict[str, Any] | None = None,
) -> str:
    """Persist and publish a terminal pipeline failure."""
    terminal_status = await session_manager.claim_pipeline_terminal_state(
        session_id,
        "failed",
    )
    if terminal_status == "cancelled":
        await _mark_cancelled(
            session_manager,
            session_id,
            failed_step,
            completed_steps=completed_steps,
            partial_results=partial_results,
        )
        return terminal_status
    if terminal_status != "failed":
        return terminal_status

    event_bus = get_event_bus()
    snapshot = event_bus.get_snapshot(session_id) or {}
    persisted_completed_steps = snapshot.get("completed_steps", [])
    if not isinstance(persisted_completed_steps, list):
        persisted_completed_steps = []
    completed_step_names = derive_completed_step_names(
        completed_steps or None,
        persisted_completed_steps,
    )
    updates: dict[str, Any] = {
        "status": "failed",
        "error": diagnostic.code,
        "error_diagnostic": diagnostic.to_dict(),
        "failed_step": failed_step,
        "completed_at": datetime.now(UTC).isoformat(),
        "can_cancel": False,
        "completed_steps": list(persisted_completed_steps),
        "completed_step_names": completed_step_names,
        "partial_results": (
            dict(partial_results) if partial_results is not None else None
        ),
    }

    await _persist_terminal_metadata(
        session_manager,
        session_id,
        updates,
        failure_code="physics_pipeline_failure_metadata_failed",
    )

    try:
        if event_bus.get_snapshot(session_id) is not None:
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=failed_step,
                    state=StepState.FAILED,
                    message=diagnostic.code,
                    extra={
                        "error": diagnostic.code,
                        "error_diagnostic": diagnostic.to_dict(),
                    },
                )
            )
    except Exception:  # noqa: BLE001
        log_durable_failure(
            logger,
            "physics_pipeline_failure_event_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
    return terminal_status


async def _mark_cancelled(
    session_manager: SessionManager,
    session_id: str,
    cancelled_step: str,
    *,
    completed_steps: list[str] | None = None,
    partial_results: dict[str, Any] | None = None,
) -> bool:
    """Persist and publish terminal cancellation before propagating it."""
    terminal_status = await session_manager.claim_pipeline_terminal_state(
        session_id,
        "cancelled",
    )
    if terminal_status != "cancelled":
        return False

    event_bus = get_event_bus()
    snapshot = event_bus.get_snapshot(session_id) or {}
    persisted_completed_steps = snapshot.get("completed_steps", [])
    if not isinstance(persisted_completed_steps, list):
        persisted_completed_steps = []
    completed_step_names = derive_completed_step_names(
        completed_steps or None,
        persisted_completed_steps,
    )
    await _persist_terminal_metadata(
        session_manager,
        session_id,
        {
            "status": "cancelled",
            "cancelled_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "can_cancel": False,
            "error": None,
            "error_diagnostic": None,
            "failed_step": None,
            "completed_steps": list(persisted_completed_steps),
            "completed_step_names": completed_step_names,
            "partial_results": (
                dict(partial_results) if partial_results is not None else None
            ),
        },
        failure_code="physics_pipeline_cancellation_metadata_failed",
    )

    try:
        if event_bus.get_snapshot(session_id) is not None:
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=cancelled_step,
                    state=StepState.CANCELLED,
                    message="Pipeline cancelled",
                )
            )
    except Exception:  # noqa: BLE001
        log_durable_failure(
            logger,
            "physics_pipeline_cancellation_event_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
    return True


async def terminalize_pipeline_cancellation(
    session_manager: SessionManager,
    session_id: str,
) -> None:
    """Persist and publish cancellation for running or pre-start jobs."""
    await _mark_cancelled(
        session_manager,
        session_id,
        _active_pipeline_step(session_id),
    )


async def _persist_terminal_metadata(
    session_manager: SessionManager,
    session_id: str,
    updates: dict[str, Any],
    *,
    failure_code: str,
) -> None:
    """Retry terminal metadata writes and propagate the final store error."""
    for attempt in range(1, _TERMINAL_PERSIST_ATTEMPTS + 1):
        try:
            await session_manager.update_session(session_id, updates)
            return
        except Exception:  # noqa: BLE001
            log_durable_failure(
                logger,
                failure_code,
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=attempt < _TERMINAL_PERSIST_ATTEMPTS,
            )
            if attempt == _TERMINAL_PERSIST_ATTEMPTS:
                raise RuntimeError(failure_code) from None
            await asyncio.sleep(_TERMINAL_PERSIST_RETRY_DELAY_SECONDS * attempt)


async def _persist_claimed_terminal_metadata(
    session_manager: SessionManager,
    session_id: str,
    updates: dict[str, Any],
    *,
    failure_code: str,
) -> None:
    """Finish a claimed terminal commit before propagating task cancellation."""
    cancellation_seen = False
    cancelled_persistence_attempts = 0
    persistence_task = asyncio.create_task(
        _persist_terminal_metadata(
            session_manager,
            session_id,
            updates,
            failure_code=failure_code,
        )
    )
    while True:
        try:
            await asyncio.shield(persistence_task)
            break
        except asyncio.CancelledError:
            cancellation_seen = True
            if persistence_task.done() and persistence_task.cancelled():
                # A storage adapter should not normally raise CancelledError on
                # its own, but retry rather than leave an accepted claim with
                # non-terminal metadata.
                cancelled_persistence_attempts += 1
                if cancelled_persistence_attempts >= _TERMINAL_PERSIST_ATTEMPTS:
                    raise RuntimeError(failure_code) from None
                persistence_task = asyncio.create_task(
                    _persist_terminal_metadata(
                        session_manager,
                        session_id,
                        updates,
                        failure_code=failure_code,
                    )
                )

    if cancellation_seen:
        raise asyncio.CancelledError


def _extract_stats_from_result(result, session_dir=None) -> dict:
    """Extract statistics from pipeline result."""
    stats = {
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
    }

    step_results = result.step_results or {}

    if "predict" in step_results:
        stats["predictions_made"] = step_results["predict"].get("predictions_count", 0)

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
        stats["prims_processed"] == 0 or stats["predictions_made"] == 0
    ):
        stats = _count_stats_from_files(session_dir, stats)

    return stats


def _count_stats_from_files(session_dir, stats: dict) -> dict:
    """Count stats from actual files in session directory."""
    from pathlib import Path

    session_path = Path(session_dir)

    if stats["prims_processed"] == 0:
        dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                with open(dataset_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["prims_processed"] = len(lines)
            except Exception as e:
                logger.warning(f"Failed to count dataset entries: {e}")

    if stats["images_generated"] == 0:
        dataset_dir = session_path / "cache" / "dataset"
        if dataset_dir.exists():
            try:
                image_count = len(list(dataset_dir.glob("**/*.png")))
                stats["images_generated"] = image_count
            except Exception as e:
                logger.warning(f"Failed to count images: {e}")

    if stats["predictions_made"] == 0:
        predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    lines = [line for line in f if line.strip()]
                    stats["predictions_made"] = len(lines)
            except Exception as e:
                logger.warning(f"Failed to count predictions: {e}")

    return stats
