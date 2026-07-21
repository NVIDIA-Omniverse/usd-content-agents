# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refine execution wrapper for the Physics Agent service."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from physics_agent.api import RefineInput, arun_refine
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


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iteration_to_metadata(iteration: Any) -> dict[str, Any]:
    metadata = _json_safe_metadata(asdict(iteration))
    if metadata.get("error") is not None:
        diagnostic = durable_diagnostic(
            "physics_refine_iteration_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        metadata["error"] = diagnostic.code
        metadata["error_diagnostic"] = diagnostic.to_dict()
    if metadata.get("recording_error") is not None:
        diagnostic = durable_diagnostic(
            "physics_refine_iteration_recording_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        metadata["recording_error"] = diagnostic.code
        metadata["recording_error_diagnostic"] = diagnostic.to_dict()
    return metadata


def _json_safe_metadata(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata(item) for item in value]
    return value


def _load_final_best_params(final_dir: Path | None) -> dict[str, float] | None:
    if final_dir is None:
        return None
    best_params_path = final_dir / "best_params.json"
    try:
        payload = json.loads(best_params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_params = payload.get("params")
    if not isinstance(raw_params, dict):
        if "best_score" in payload:
            return None
        raw_params = payload

    params: dict[str, float] = {}
    for key, value in raw_params.items():
        if not isinstance(key, str):  # pragma: no cover - JSON object keys are strings
            continue
        finite_value = _finite_or_none(value)
        if finite_value is not None:
            params[key] = finite_value
    return params or None


def _refine_results_metadata(result: Any) -> dict[str, Any]:
    iterations = [_iteration_to_metadata(item) for item in result.iterations]
    final_best_params = _load_final_best_params(
        Path(result.final_dir) if result.final_dir else None
    )
    if final_best_params is not None and iterations:
        final_iteration = int(result.final_iteration or 0)
        target_iteration = None
        for iteration in iterations:
            try:
                if int(iteration.get("iteration") or 0) == final_iteration:
                    target_iteration = iteration
                    break
            except (TypeError, ValueError):
                continue
        if target_iteration is None:
            target_iteration = iterations[-1]
        target_iteration["best_params"] = final_best_params

    return {
        "termination_reason": str(result.termination_reason),
        "iteration_count": int(result.iteration_count),
        "final_iteration": int(result.final_iteration),
        "final_judge_score": _finite_or_none(result.final_judge_score),
        "final_best_params": final_best_params,
        "iterations": iterations,
        "final_dir": str(result.final_dir) if result.final_dir else None,
        "output_dir": str(result.output_dir) if result.output_dir else None,
    }


async def _publish_refine_artifacts(
    session_manager: Any, session_id: str, session_dir: Path
) -> tuple[list[str], DurableDiagnostic | None]:
    manifest = collect_public_artifact_manifest(session_dir, "refine")
    try:
        await session_manager.sync_to_store(session_id, prefix="refine/")
    except Exception:
        diagnostic = durable_diagnostic(
            "physics_refine_artifact_sync_failed",
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


def _build_refine_models(
    *,
    judge_max_tokens: int | None,
    judge_temperature: float | None,
) -> tuple[Any, Any]:
    """Build server-side models for scenario-refine and judge calls."""

    backend = os.getenv("PA_REFINE_BACKEND") or os.getenv("PA_VLM_BACKEND", "gemini")
    model = os.getenv("PA_REFINE_MODEL") or os.getenv(
        "PA_VLM_MODEL", "gemini-3-pro-preview"
    )

    try:
        from physics_agent.api.defaults import (
            DEFAULT_JUDGE_MAX_TOKENS,
            DEFAULT_JUDGE_TEMPERATURE,
            DEFAULT_VLM_REASONING_EFFORT,
        )
        from physics_agent.tuning.visual_evidence import (
            backend_supports_reasoning_effort,
        )
        from world_understanding.agentic.config import get_api_key_for_model_config
        from world_understanding.functions.models.backends.registry import (
            chat_backend_requires_api_key,
            list_chat_backends,
            list_vlm_backends,
            vlm_backend_requires_api_key,
        )
        from world_understanding.functions.models.chat_models import create_chat_model
        from world_understanding.functions.models.vision_language_models import (
            create_vlm,
        )
        from world_understanding.utils.credentials import (
            API_KEY_ENV_VAR_MAP,
            get_env_api_key_for_backend,
        )
    except Exception as exc:
        raise RuntimeError(
            "Refine model dependencies are unavailable in the service image."
        ) from exc

    if backend not in list_chat_backends() or backend not in list_vlm_backends():
        raise RuntimeError(
            f"Refine backend {backend!r} is not registered as both chat and VLM."
        )

    api_key = get_env_api_key_for_backend(backend)
    if not api_key and (
        chat_backend_requires_api_key(backend) or vlm_backend_requires_api_key(backend)
    ):
        env_vars = API_KEY_ENV_VAR_MAP.get(backend, ())
        env_hint = ", ".join(env_vars) if env_vars else "a configured credential"
        raise RuntimeError(
            f"API key for refine backend {backend!r} is required for /refine; "
            f"set {env_hint}."
        )

    chat_kwargs: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "temperature": 0.0,
    }
    if api_key:
        chat_kwargs["api_key"] = api_key
    chat_model = create_chat_model(**chat_kwargs)
    vlm_config: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "temperature": (
            judge_temperature
            if judge_temperature is not None
            else DEFAULT_JUDGE_TEMPERATURE
        ),
        "max_tokens": (
            judge_max_tokens
            if judge_max_tokens is not None
            else DEFAULT_JUDGE_MAX_TOKENS
        ),
        "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
    }
    if api_key:
        vlm_config["api_key"] = api_key
    resolved_vlm_api_key = get_api_key_for_model_config(backend, vlm_config, "vlm")
    if resolved_vlm_api_key:
        vlm_config["api_key"] = resolved_vlm_api_key
    if not backend_supports_reasoning_effort(backend):
        vlm_config.pop("reasoning_effort", None)
    vlm_model = create_vlm(**vlm_config)
    return chat_model, vlm_model


class _RefineEventListener(EventListener):
    """Adapter from refine/tune events to service progress events."""

    def __init__(self, session_id: str, max_iterations: int, max_trials: int):
        self.session_id = session_id
        self.max_iterations = max(max_iterations, 1)
        self.max_trials = max(max_trials, 1)
        self.iteration = 0
        self.n_trials = 0
        self.best_score: float | None = None
        self.best_params: dict[str, float] | None = None
        self.judge_score: float | None = None
        self.bus = get_event_bus()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.info("[refine %s] " + message, self.session_id[:8], *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        logger.debug("[refine %s] " + message, self.session_id[:8], *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        log_durable_failure(
            logger,
            "physics_refine_runner_reported_warning",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=True,
        )

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        log_durable_failure(
            logger,
            "physics_refine_runner_reported_failure",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        if event_type == "refine.iteration.started":
            self.iteration = int(data.get("iteration") or self.iteration or 1)
            self.n_trials = 0
            self.best_score = None
            self.best_params = None
            self.judge_score = None
            message = f"Refine iteration {self.iteration}/{self.max_iterations}"
        elif event_type == "tune.trial.completed":
            self.n_trials += 1
            score = _finite_or_none(data.get("score"))
            failed = bool(data.get("failed", False))
            if (
                score is not None
                and not failed
                and (self.best_score is None or score < self.best_score)
            ):
                self.best_score = score
                self.best_params = dict(data.get("params") or {})
            message = (
                f"Iteration {self.iteration}, trial {self.n_trials}/{self.max_trials}"
            )
        elif event_type == "refine.iteration.tune_completed":
            self.iteration = int(data.get("iteration") or self.iteration)
            self.n_trials = int(data.get("n_trials") or self.n_trials)
            self.best_score = _finite_or_none(data.get("best_score"))
            self.best_params = dict(data.get("best_params") or {})
            message = f"Iteration {self.iteration} tune completed"
        elif event_type == "refine.iteration.judged":
            self.iteration = int(data.get("iteration") or self.iteration)
            self.judge_score = _finite_or_none(data.get("judge_score"))
            decision = data.get("decision", "unknown")
            message = f"Iteration {self.iteration} judge decision: {decision}"
        elif event_type == "refine.completed":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="refine",
                state=StepState.RUNNING,
                percent=100,
                message="Refine loop completed; syncing artifacts",
                extra=dict(data),
            )
            self._emit_threadsafe(ev)
            return
        elif event_type == "tune.cancelled":
            ev = ProgressEvent(
                session_id=self.session_id,
                step="refine",
                state=StepState.RUNNING,
                message="Refine cancelled; publishing partial artifacts",
                extra=dict(data),
            )
            self._emit_threadsafe(ev)
            return
        elif event_type == "tune.failed":
            diagnostic = durable_diagnostic(
                "physics_refine_runner_event_failed",
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            )
            ev = ProgressEvent(
                session_id=self.session_id,
                step="refine",
                state=StepState.RUNNING,
                message="Refine failed; publishing partial artifacts",
                extra={
                    "error": diagnostic.code,
                    "error_diagnostic": diagnostic.to_dict(),
                },
            )
            self._emit_threadsafe(ev)
            return
        else:
            return

        percent = int(
            min(
                99,
                max(
                    0,
                    100
                    * (
                        max(self.iteration - 1, 0)
                        + min(self.n_trials / self.max_trials, 1.0)
                    )
                    / self.max_iterations,
                ),
            )
        )
        ev = ProgressEvent(
            session_id=self.session_id,
            step="refine",
            state=StepState.RUNNING,
            current=self.n_trials,
            total=self.max_trials,
            percent=percent,
            message=message,
            extra={
                "event_type": event_type,
                "iteration": self.iteration,
                "max_iterations": self.max_iterations,
                "n_trials": self.n_trials,
                "max_trials": self.max_trials,
                "best_score": self.best_score,
                "best_params": self.best_params,
                "judge_score": self.judge_score,
                **dict(data),
            },
        )
        self._emit_threadsafe(ev)

    def _emit_threadsafe(self, ev: ProgressEvent) -> None:
        if self.loop is None or self.loop.is_closed():
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        if self.loop.is_closed():
            return

        def _schedule() -> None:
            try:
                asyncio.create_task(self.bus.emit(ev))
            except RuntimeError:
                log_durable_failure(
                    logger,
                    "physics_refine_progress_event_schedule_failed",
                    phase=FailurePhase.LOCAL_PUBLICATION,
                    retryable=True,
                )

        try:
            self.loop.call_soon_threadsafe(_schedule)
        except RuntimeError:
            log_durable_failure(
                logger,
                "physics_refine_progress_event_schedule_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )


async def _watch_for_cancel(
    session_manager: Any,
    session_id: str,
    cancel_event: threading.Event,
    poll_interval: float = 0.25,
) -> None:
    try:
        while not cancel_event.is_set():
            try:
                if await session_manager.is_cancelled(session_id):
                    cancel_event.set()
                    return
            except Exception:
                log_durable_failure(
                    logger,
                    "physics_refine_cancellation_poll_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        return


async def _emit_terminal_bus_event(
    session_id: str,
    state: StepState,
    message: str,
    *,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
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
                step="refine",
                state=state,
                message=message,
                extra=event_extra,
            )
        )
    except Exception:
        log_durable_failure(
            logger,
            "physics_refine_terminal_event_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )


async def execute_refine_async(
    session_id: str,
    session_manager: Any,
    scenario_path: Path,
    physics_usd: Path,
    *,
    user_prompt: str,
    engine: str,
    optimizer: str,
    max_trials: int,
    seed: int,
    max_iterations: int,
    score_threshold: float,
    judge_max_tokens: int | None = None,
    judge_temperature: float | None = None,
    reference_images: list[Path] | None = None,
    reference_videos: list[Path] | None = None,
    reference_descriptions: list[str] | None = None,
    reference_video_descriptions: list[str] | None = None,
    reference_video_frames: int = 8,
    judge_reference_frames: int = 8,
    judge_generated_frames: int = 16,
    visual_evidence_enabled: bool = True,
    llm_timeout_seconds: float = 180.0,
) -> None:
    """Run an iterative refine session and persist service metadata."""

    session_dir = session_manager.get_session_dir(session_id)
    output_dir = session_dir / "refine"
    output_dir.mkdir(parents=True, exist_ok=True)

    listener = _RefineEventListener(
        session_id,
        max_iterations=max_iterations,
        max_trials=max_trials,
    )
    cancel_event = threading.Event()
    cancel_watcher = asyncio.create_task(
        _watch_for_cancel(session_manager, session_id, cancel_event)
    )

    await session_manager.update_session(session_id, {"status": "running"})
    result = None
    execution_failure: DurableDiagnostic | None = None
    try:
        chat_model, vlm_model = _build_refine_models(
            judge_max_tokens=judge_max_tokens,
            judge_temperature=judge_temperature,
        )
        result = await arun_refine(
            RefineInput(
                scenario=scenario_path,
                physics_usd=physics_usd,
                user_prompt=user_prompt,
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
                max_iterations=max_iterations,
                score_threshold=score_threshold,
                judge_max_tokens=judge_max_tokens,
                judge_temperature=judge_temperature,
                chat_model=chat_model,
                vlm_model=vlm_model,
                force_record_video="off",
                render_winning_trial=False,
                visual_evidence_enabled=visual_evidence_enabled,
                llm_timeout_seconds=llm_timeout_seconds,
                cancel_event=cancel_event,
                event_listener=listener,
            )
        )
    except asyncio.CancelledError:
        cancel_event.set()
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "completed_at": datetime.now(UTC).isoformat(),
                "can_cancel": False,
            },
        )
        await _emit_terminal_bus_event(
            session_id, StepState.CANCELLED, "Refine cancelled"
        )
        raise
    except Exception:
        diagnostic = durable_diagnostic(
            "physics_refine_execution_failed",
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
                "failed_step": "refine",
                "completed_at": datetime.now(UTC).isoformat(),
                "can_cancel": False,
            },
        )
        await _emit_terminal_bus_event(
            session_id,
            StepState.FAILED,
            diagnostic.code,
            error=diagnostic.code,
            extra={"error_diagnostic": diagnostic.to_dict()},
        )
        execution_failure = diagnostic
    finally:
        cancel_watcher.cancel()
        try:
            await cancel_watcher
        except asyncio.CancelledError:
            pass

    if execution_failure is not None:
        # Raise outside the exception handler so the raw exception cannot be
        # retained as context on the background task's durable failure.
        raise RuntimeError(execution_failure.code)

    metadata = await session_manager.get_session_metadata(session_id) or {}
    duration = 0
    created_at_str = metadata.get("created_at")
    if created_at_str:
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        duration = int((datetime.now(UTC) - created_at).total_seconds())

    raw_recording_path = getattr(result, "final_recording_usd", None)
    recording_path = (
        Path(raw_recording_path) if raw_recording_path is not None else None
    )
    raw_recording_error = getattr(result, "final_recording_error", None)
    results = _refine_results_metadata(result)
    if recording_path is not None:
        results["final_recording_usd"] = str(recording_path)
    if raw_recording_error is not None:
        recording_diagnostic = durable_diagnostic(
            "physics_refine_final_recording_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        log_durable_failure(
            logger,
            recording_diagnostic.code,
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        results["final_recording_error"] = recording_diagnostic.code
        results["final_recording_error_diagnostic"] = recording_diagnostic.to_dict()
    try:
        artifact_manifest, artifact_sync_diagnostic = await _publish_refine_artifacts(
            session_manager, session_id, session_dir
        )
    except asyncio.CancelledError:
        cancel_event.set()
        artifact_manifest = collect_public_artifact_manifest(session_dir, "refine")
        await session_manager.update_session(
            session_id,
            {
                "status": "cancelled",
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": duration,
                "can_cancel": False,
                "artifact_manifest": artifact_manifest,
                "results": results,
            },
        )
        await _emit_terminal_bus_event(
            session_id,
            StepState.CANCELLED,
            "Refine cancelled during artifact publication",
        )
        raise
    late_cancel = await session_manager.is_cancelled(session_id)
    if late_cancel or result.termination_reason == "cancelled":
        status = "cancelled"
    elif result.success:
        status = "completed"
    else:
        status = "failed"

    updates: dict[str, Any] = {
        "status": status,
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_seconds": duration,
        "can_cancel": False,
        "results": results,
        "artifact_manifest": artifact_manifest,
    }
    terminal_diagnostic: DurableDiagnostic | None = None
    if status == "failed":
        terminal_diagnostic = durable_diagnostic(
            "physics_refine_result_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        updates["error"] = terminal_diagnostic.code
        updates["error_diagnostic"] = terminal_diagnostic.to_dict()
        updates["failed_step"] = "refine"
        updates["partial_results"] = results
    if artifact_sync_diagnostic is not None:
        updates["artifact_sync_error"] = artifact_sync_diagnostic.code
        updates["artifact_sync_diagnostic"] = artifact_sync_diagnostic.to_dict()
        if status == "completed":
            status = "failed"
            terminal_diagnostic = artifact_sync_diagnostic
            updates.update(
                {
                    "status": status,
                    "error": terminal_diagnostic.code,
                    "error_diagnostic": terminal_diagnostic.to_dict(),
                    "failed_step": "artifact_sync",
                    "partial_results": results,
                }
            )

    if status == "failed" and terminal_diagnostic is None:
        terminal_diagnostic = durable_diagnostic(
            "physics_refine_terminal_state_invalid",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        updates.update(
            {
                "error": terminal_diagnostic.code,
                "error_diagnostic": terminal_diagnostic.to_dict(),
                "failed_step": "refine",
                "partial_results": results,
            }
        )

    await session_manager.update_session(session_id, updates)

    if status == "completed":
        await _emit_terminal_bus_event(
            session_id,
            StepState.COMPLETED,
            "Refine artifacts synced and ready",
            extra={"refine_ready": True},
        )
    elif status == "cancelled":
        await _emit_terminal_bus_event(
            session_id,
            StepState.CANCELLED,
            "Refine cancelled",
        )
    else:
        if terminal_diagnostic is None:
            raise RuntimeError("physics_refine_terminal_state_invalid")
        await _emit_terminal_bus_event(
            session_id,
            StepState.FAILED,
            terminal_diagnostic.code,
            error=terminal_diagnostic.code,
            extra={"error_diagnostic": terminal_diagnostic.to_dict()},
        )

    logger.info(
        "Refine execution finished for %s with status=%s", session_id[:8], status
    )


__all__ = [
    "execute_refine_async",
]
