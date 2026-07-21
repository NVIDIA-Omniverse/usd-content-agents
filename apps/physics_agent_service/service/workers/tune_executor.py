# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune execution wrapper — drives :func:`physics_agent.tuning.arun_tune`.

Mirrors :mod:`workers.executor` (pipeline) so the JobRegistry / EventBus /
SessionManager wiring is identical. The big difference: we install a
threading.Event that is polled on every trial — when the user POSTs
``/tune/{id}/cancel`` it sets the event and the optimizer exits cleanly
between trials.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from physics_agent.tuning import (
    BoTorchUnavailableError,
    OvPhysXUnavailableError,
    TuneInput,
    TuningCancelledError,
    arun_tune,
)
from world_understanding.agentic.events import EventListener
from world_understanding.utils.durable_diagnostics import (
    DurableDiagnostic,
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)

from ..artifact_contract import collect_public_artifact_manifest
from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


def _finite_best_score(value: object) -> float | None:
    """Round 15 (doyubkim blocker #2): coerce ``best_score`` to a JSON-safe
    finite float or ``None`` before persisting it to session metadata.

    The runner emits ``float("inf")`` for the cancelled-before-first-trial
    path (see ``physics_agent.tuning.runner._handle_zero_trial_cancel``);
    a backend overflow during a normal trial can also stamp ``inf``. Both
    routes write through ``update_session({...,"results":{"best_score":...}})``,
    which is later returned by ``GET /tune/{id}/status``. Starlette's JSON
    encoder rejects ``inf`` / ``-inf`` / ``nan`` outright and raises
    ``ValueError: Out of range float values are not JSON compliant``,
    turning a clean cancel into a 500 at every status poll. Sanitising at
    the WRITE site means every later read — ``/status``, ``/results``,
    artifact sync, refine-loop re-entry — sees a finite-only number or
    ``None`` instead.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _tune_results_metadata(result: Any) -> dict[str, Any]:
    """Return the session ``results`` payload for completed or partial tunes."""
    return {
        "best_params": dict(getattr(result, "best_params", {}) or {}),
        "best_score": _finite_best_score(getattr(result, "best_score", None)),
        "n_trials": int(getattr(result, "n_trials", 0) or 0),
        "optimizer_used": str(getattr(result, "optimizer_used", "") or ""),
        "engine_used": str(getattr(result, "engine_used", "") or ""),
    }


def _has_partial_tune_results(result: Any) -> bool:
    """Return True when a failed tune still has useful result artifacts."""
    if int(getattr(result, "n_trials", 0) or 0) > 0:
        return True
    if getattr(result, "best_params", None):
        return True
    return bool(getattr(result, "artifacts", None))


async def _publish_tune_artifacts(
    session_manager: Any, session_id: str, session_dir: Path
) -> tuple[list[str], DurableDiagnostic | None]:
    manifest = collect_public_artifact_manifest(session_dir, "tune")
    try:
        await session_manager.sync_to_store(session_id, prefix="tune/")
    except Exception:
        diagnostic = durable_diagnostic(
            "physics_tune_artifact_sync_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )
        return manifest, diagnostic
    return manifest, None


class _TuneEventListener(EventListener):
    """Adapter from the tuning runner's events → FastAPI ProgressEvent bus.

    Intentionally minimal — tuning has a much smaller event vocabulary than
    the full physics pipeline (started / trial.completed / completed /
    failed / warning) so we don't need the multi-step bookkeeping of the
    pipeline listener.
    """

    def __init__(self, session_id: str, max_trials: int):
        self.session_id = session_id
        self.max_trials = max(max_trials, 1)
        self.bus = get_event_bus()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        self.best_score: float | None = None
        self.best_params: dict[str, float] | None = None
        self.n_trials = 0

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.info(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug(f"[tune {self.session_id[:8]}] {message}", *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        log_durable_failure(
            logger,
            "physics_tune_runner_reported_warning",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=True,
        )

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        log_durable_failure(
            logger,
            "physics_tune_runner_reported_failure",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        if event_type == "tune.started":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                percent=0,
                message="Tuning started",
                extra=data,
            )
        elif event_type == "tune.trial.completed":
            self.n_trials += 1
            score = float(data.get("score", float("inf")))
            failed = bool(data.get("failed", False))
            if not failed and (self.best_score is None or score < self.best_score):
                self.best_score = score
                self.best_params = dict(data.get("params") or {})
            percent = int(min(100, 100 * self.n_trials / self.max_trials))
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                current=self.n_trials,
                total=self.max_trials,
                percent=percent,
                message=(
                    f"Trial {self.n_trials}/{self.max_trials}: "
                    f"score={score:.4g}{' (failed)' if failed else ''}"
                ),
                extra={
                    "trial_index": data.get("trial_index"),
                    "score": score,
                    "params": data.get("params"),
                    "failed": failed,
                    "best_score": self.best_score,
                    "best_params": self.best_params,
                },
            )
        elif event_type == "tune.completed":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                percent=100,
                message="Tuning completed; publishing artifacts",
                extra=dict(data),
            )
        elif event_type == "tune.cancelled":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                message="Tuning cancelled; publishing partial artifacts",
                extra=dict(data),
            )
        elif event_type == "tune.failed":
            diagnostic = durable_diagnostic(
                "physics_tune_runner_event_failed",
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            )
            ev = ProgressEvent(
                session_id=self.session_id,
                step="tune",
                state=StepState.RUNNING,
                message="Tuning failed; publishing partial artifacts",
                extra={
                    "error": diagnostic.code,
                    "error_diagnostic": diagnostic.to_dict(),
                },
            )
        else:
            return

        self._emit_threadsafe(ev)

    def _emit_threadsafe(self, ev: ProgressEvent) -> None:
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(self.bus.emit(ev)))


async def _emit_terminal_bus_event(
    session_id: str,
    state: StepState,
    message: str,
    *,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    percent: int | None = None,
) -> None:
    """Emit a terminal CANCELLED/FAILED bus frame for early-exit branches.

    Round 11 thread #3: the BoTorch/OvPhysX-unavailable, asyncio.Cancelled,
    TuningCancelledError, and generic-Exception handlers in
    ``execute_tune_async`` exit before the post-finally code that emits the
    terminal event. Same-instance ``stream_tune_events`` clients only
    short-circuit on FAILED/CANCELLED/tune_ready, so without an explicit
    emit here SSE consumers hang until the 30s timeout fallback notices the
    durable metadata flip.
    """
    bus = get_event_bus()
    if bus.get_snapshot(session_id) is None:
        return
    event_extra: dict[str, Any] = dict(extra or {})
    if error is not None:
        event_extra["error"] = error
    try:
        await bus.emit(
            ProgressEvent(
                session_id=session_id,
                step="tune",
                state=state,
                percent=percent,
                message=message,
                extra=event_extra,
            )
        )
    except Exception:
        log_durable_failure(
            logger,
            "physics_tune_terminal_event_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )


async def _watch_for_cancel(
    session_manager: Any,
    session_id: str,
    cancel_event: threading.Event,
    poll_interval: float = 0.25,
) -> None:
    """Watch the SessionManager for an out-of-process cancel signal.

    The /tune/{id}/cancel endpoint writes a ``.cancel`` marker via
    ``request_cancellation``; that signal is visible cross-instance, but the
    optimizer running inside ``to_thread`` only checks the ``cancel_event``
    we hand it. This task polls and bridges the two.
    """
    try:
        while not cancel_event.is_set():
            try:
                if await session_manager.is_cancelled(session_id):
                    cancel_event.set()
                    return
            except Exception:  # pragma: no cover
                log_durable_failure(
                    logger,
                    "physics_tune_cancellation_poll_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        return


async def execute_tune_async(
    session_id: str,
    session_manager: Any,
    scenario_path: Path | None,
    physics_usd: Path,
    *,
    user_prompt: str | None = None,
    engine: str,
    optimizer: str,
    max_trials: int,
    seed: int,
    enable_judge: bool = True,
    judge_max_iterations: int = 3,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    reference_images: list[Path] | None = None,
    reference_videos: list[Path] | None = None,
    reference_descriptions: list[str] | None = None,
    reference_video_descriptions: list[str] | None = None,
    reference_video_frames: int = 8,
    judge_reference_frames: int = 8,
    judge_generated_frames: int = 16,
) -> None:
    """Run one tuning session end-to-end and persist results."""
    logger.info(f"Tune execution started for {session_id[:8]}...")

    session_dir = session_manager.get_session_dir(session_id)
    output_dir = session_dir / "tune"
    output_dir.mkdir(parents=True, exist_ok=True)

    listener = _TuneEventListener(session_id, max_trials=max_trials)
    cancel_event = threading.Event()

    cancel_watcher = asyncio.create_task(
        _watch_for_cancel(session_manager, session_id, cancel_event)
    )

    await session_manager.update_session(session_id, {"status": "running"})

    result = None
    execution_failure: tuple[type[Exception], DurableDiagnostic] | None = None
    try:
        try:
            result = await arun_tune(
                TuneInput(
                    scenario=scenario_path,
                    user_prompt=user_prompt,
                    physics_usd=physics_usd,
                    output_dir=output_dir,
                    reference_images=reference_images,
                    reference_videos=reference_videos,
                    reference_descriptions=reference_descriptions,
                    reference_video_descriptions=reference_video_descriptions,
                    reference_video_frames=reference_video_frames,
                    judge_reference_frames=judge_reference_frames,
                    judge_generated_frames=judge_generated_frames,
                    engine=engine,
                    optimizer=optimizer,
                    max_trials=max_trials,
                    seed=seed,
                    enable_judge=enable_judge,
                    judge_max_iterations=judge_max_iterations,
                    judge_max_tokens=judge_max_tokens,
                    judge_temperature=judge_temperature,
                    cancel_event=cancel_event,
                    event_listener=listener,
                )
            )
        except BoTorchUnavailableError:
            diagnostic = durable_diagnostic(
                "physics_tune_botorch_unavailable",
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
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id,
                StepState.FAILED,
                diagnostic.code,
                error=diagnostic.code,
                extra={"error_diagnostic": diagnostic.to_dict()},
            )
            execution_failure = (BoTorchUnavailableError, diagnostic)
        except OvPhysXUnavailableError:
            diagnostic = durable_diagnostic(
                "physics_tune_ovphysx_unavailable",
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
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id,
                StepState.FAILED,
                diagnostic.code,
                error=diagnostic.code,
                extra={"error_diagnostic": diagnostic.to_dict()},
            )
            execution_failure = (OvPhysXUnavailableError, diagnostic)
        except asyncio.CancelledError:
            # Outer task cancellation (session delete, server shutdown).
            # Set the cooperative cancel signal so the worker thread's
            # optimizer can exit between trials before the asyncio side
            # tears down — without this the thread would keep running and
            # could write into a freshly-deleted session directory.
            cancel_event.set()
            await session_manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.CANCELLED, "Tune cancelled"
            )
            raise
        except TuningCancelledError:
            # Cooperative cancellation surfaced by the runner's LLM-call
            # wrapper (interpreter or judge phase), or by the optimizer
            # detecting the cancel marker. Persist as 'cancelled' rather
            # than the generic 'failed' branch below — codex round 5.
            await session_manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id, StepState.CANCELLED, "Tune cancelled"
            )
            return
        except Exception:
            # Any unexpected failure inside the runner (USD parse, optimizer
            # bug, backend RuntimeError, …). Without this catch the exception
            # propagates into JobRegistry, whose cleanup does NOT update
            # session metadata — leaving the session stuck in 'running'.
            diagnostic = durable_diagnostic(
                "physics_tune_execution_failed",
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
                    "failed_step": "tune",
                    "completed_at": datetime.now(UTC).isoformat(),
                },
            )
            await _emit_terminal_bus_event(
                session_id,
                StepState.FAILED,
                diagnostic.code,
                error=diagnostic.code,
                extra={"error_diagnostic": diagnostic.to_dict()},
            )
            execution_failure = (RuntimeError, diagnostic)
    finally:
        cancel_watcher.cancel()
        try:
            await cancel_watcher
        except asyncio.CancelledError:
            pass

    if execution_failure is not None:
        exception_type, diagnostic = execution_failure
        # Raise outside the handler so no raw backend exception remains as
        # context on the background task.
        raise exception_type(diagnostic.code)

    metadata = await session_manager.get_session_metadata(session_id) or {}
    duration = 0
    created_at_str = metadata.get("created_at")
    if created_at_str:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        duration = int((datetime.now(UTC) - created_at).total_seconds())

    try:
        artifact_manifest, artifact_sync_diagnostic = await _publish_tune_artifacts(
            session_manager, session_id, session_dir
        )
    except asyncio.CancelledError:
        cancel_event.set()
        artifact_manifest = collect_public_artifact_manifest(session_dir, "tune")
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": duration,
                "can_cancel": False,
                "artifact_manifest": artifact_manifest,
                "results": _tune_results_metadata(result),
            },
        )
        await _emit_terminal_bus_event(
            session_id,
            StepState.CANCELLED,
            "Tune cancelled during artifact publication",
        )
        raise
    # Check after publication because a cancellation can arrive while a large
    # artifact set is uploading to the shared store.
    late_cancel = await session_manager.is_cancelled(session_id) or result.cancelled

    if late_cancel:
        updates: dict[str, Any] = {
            "status": "cancelled",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "can_cancel": False,
            "artifact_manifest": artifact_manifest,
            "results": _tune_results_metadata(result),
        }
        if artifact_sync_diagnostic is not None:
            updates["artifact_sync_error"] = artifact_sync_diagnostic.code
            updates["artifact_sync_diagnostic"] = artifact_sync_diagnostic.to_dict()
        await session_manager.update_session(
            session_id,
            updates,
        )
        percent = (
            min(100, int(100 * result.n_trials / max(max_trials, 1)))
            if max_trials
            else 0
        )
        await _emit_terminal_bus_event(
            session_id,
            StepState.CANCELLED,
            "Tune cancelled",
            percent=percent,
            extra={
                "best_score": _finite_best_score(result.best_score),
                "best_params": result.best_params,
                "n_trials": result.n_trials,
            },
        )
        return

    if not result.success:
        result_diagnostic = durable_diagnostic(
            "physics_tune_result_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        log_durable_failure(
            logger,
            result_diagnostic.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        partial_results = (
            _tune_results_metadata(result)
            if _has_partial_tune_results(result)
            else None
        )
        updates = {
            "status": "failed",
            "error": result_diagnostic.code,
            "error_diagnostic": result_diagnostic.to_dict(),
            "failed_step": "tune",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "can_cancel": False,
            "artifact_manifest": artifact_manifest,
        }
        if artifact_sync_diagnostic is not None:
            updates["artifact_sync_error"] = artifact_sync_diagnostic.code
            updates["artifact_sync_diagnostic"] = artifact_sync_diagnostic.to_dict()
        if partial_results is not None:
            # A media-backed judge/evidence failure can happen after the
            # optimizer wrote useful tune artifacts. Keep the terminal status
            # failed, but persist the same discoverable metadata shape as
            # completed/cancelled runs so REST clients can fetch artifacts.
            updates["results"] = partial_results
            updates["partial_results"] = partial_results
        await session_manager.update_session(session_id, updates)
        await _emit_terminal_bus_event(
            session_id,
            StepState.FAILED,
            result_diagnostic.code,
            error=result_diagnostic.code,
            extra={"error_diagnostic": result_diagnostic.to_dict()},
        )
        return

    results = _tune_results_metadata(result)
    status = "completed"
    updates = {
        "status": status,
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "can_cancel": False,
        "results": results,
        "artifact_manifest": artifact_manifest,
    }
    if artifact_sync_diagnostic is not None:
        status = "failed"
        updates.update(
            {
                "status": status,
                "error": artifact_sync_diagnostic.code,
                "error_diagnostic": artifact_sync_diagnostic.to_dict(),
                "failed_step": "artifact_sync",
                "partial_results": results,
                "artifact_sync_error": artifact_sync_diagnostic.code,
                "artifact_sync_diagnostic": artifact_sync_diagnostic.to_dict(),
            }
        )
    await session_manager.update_session(
        session_id,
        updates,
    )
    if status == "failed":
        await _emit_terminal_bus_event(
            session_id,
            StepState.FAILED,
            artifact_sync_diagnostic.code,
            error=artifact_sync_diagnostic.code,
            extra={"error_diagnostic": artifact_sync_diagnostic.to_dict()},
        )
        return
    await _emit_terminal_bus_event(
        session_id,
        StepState.COMPLETED,
        "Tune artifacts synced and ready",
        percent=100,
        extra={"tune_ready": True},
    )
    logger.info(
        "Tune execution completed for %s with %d artifact(s)",
        session_id[:8],
        len(artifact_manifest),
    )
