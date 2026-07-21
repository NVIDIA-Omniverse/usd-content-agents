# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict-only executor for the /predict route.

Drives prediction in one of two modes:

* **Mode A (dataset_only):** A readable ``dataset_path`` was supplied, or a
  session-cached ``dataset.jsonl`` exists and its referenced images resolve.
  Only the ``predict`` step runs.
* **Mode B (full_predict):** No runnable session dataset is available and an
  input USD exists. This includes a cached JSONL with missing image artifacts.
  The minimum upstream steps run before prediction:
  ``optimize_usd`` (optional) → ``identify_asset`` → ``build_dataset_usd``
  → ``build_dataset_prepare_dataset`` → ``predict``.

Mode is picked **automatically** at job start based on what exists on disk.
The detected mode is recorded in session metadata under ``predict_mode`` so
``/predict/{id}/results`` can surface it without re-detecting.

This module deliberately avoids importing anything from ``physics_agent.tuning``
or BoTorch — predict must remain independent of tuning concerns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from physics_agent.api import (
    PipelineInput,
    PredictInput,
    arun_pipeline,
    arun_predict,
)
from world_understanding.utils.durable_diagnostics import (
    DurableDiagnostic,
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)

from ..events.listener import FastAPIEventListener
from ..runtime import get_event_bus
from ..runtime.events import ProgressEvent, StepState

logger = logging.getLogger(__name__)


class _PredictResultError(RuntimeError):
    """Stop post-processing after a runner returned a failed result."""

    def __init__(self, diagnostic: DurableDiagnostic) -> None:
        self.diagnostic = diagnostic
        super().__init__(diagnostic.code)


def _extract_image_paths(entry: dict[str, Any]) -> list[str]:
    """Pull image-file path strings out of a dataset.jsonl entry.

    Handles both schema flavours that ``physics_agent`` writes:

    * v0.2 (``prepare_dataset.py``): ``{"media": {"images": [{"path": "..."}]}}``
    * legacy / test stub: ``{"images": {"prim_only": "..."}}``,
      ``{"images": ["a.png", "b.png"]}``, ``{"images": "a.png"}``
    """
    paths: list[str] = []
    media = entry.get("media")
    if isinstance(media, dict):
        media_images = media.get("images")
        if isinstance(media_images, list):
            for img in media_images:
                if isinstance(img, dict):
                    p = img.get("path")
                    if isinstance(p, str):
                        paths.append(p)
                elif isinstance(img, str):
                    paths.append(img)
    images_field = entry.get("images")
    if isinstance(images_field, dict):
        for v in images_field.values():
            if isinstance(v, str):
                paths.append(v)
    elif isinstance(images_field, list | tuple):
        for v in images_field:
            if isinstance(v, str):
                paths.append(v)
            elif isinstance(v, dict):
                p = v.get("path")
                if isinstance(p, str):
                    paths.append(p)
    elif isinstance(images_field, str):
        paths.append(images_field)
    return paths


def _dataset_jsonl_has_resolvable_images(jsonl_path: Path) -> bool:
    """Return True iff at least one entry's image files are accessible.

    A dataset.jsonl is "runnable" only when its image references resolve
    relative to the JSONL's directory (or via absolute paths). When the
    JSONL was staged from an external ``dataset_path`` or synced from a
    different pod, the images live next to the *original* file — staging
    only copies the JSONL itself. Returning False here lets the executor
    fall back to Mode B (rebuild from USD) instead of running predict
    against missing image paths.

    The check inspects up to the first 20 image-bearing entries (skipping
    blanks, unparseable lines, and entries with no image paths) and
    considers the JSONL "resolvable" the moment one entry's images are
    all present. Empty / unparseable / image-less JSONL returns False so
    an obviously broken file cannot win Mode A.
    """
    max_entries = 20
    entries_seen = 0
    try:
        base_dir = jsonl_path.parent
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if entries_seen >= max_entries:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(entry, dict):
                    continue
                paths = _extract_image_paths(entry)
                if not paths:
                    continue
                entries_seen += 1
                # Dedupe before existence checks: hybrid v0.2/legacy entries
                # can list the same image under both fields, and we only need
                # one stat per unique path.
                resolved: list[Path] = []
                seen_strs: set[str] = set()
                for raw in paths:
                    p = Path(raw)
                    if not p.is_absolute():
                        p = base_dir / p
                    key = str(p)
                    if key in seen_strs:
                        continue
                    seen_strs.add(key)
                    resolved.append(p)
                if all(p.exists() for p in resolved):
                    return True
    except OSError:
        return False
    return False


# Steps the /predict route runs in Mode B (full_predict). apply_physics is
# intentionally excluded — /predict is for prediction-only workflows. Use
# /pipeline if you need the full classify/apply flow. restore_usd IS
# included so that runs with optimize_usd=true map deinstanced/split prim
# paths back to the original USD before predictions are exposed via
# /artifacts; without it /predict/{id}/predictions would reference prim
# paths that don't exist in the user's uploaded scene.
PREDICT_PIPELINE_STEPS: list[str] = [
    "optimize_usd",
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "restore_usd",
]


def detect_predict_mode(
    *,
    session_dir: Path,
    dataset_path: Path | None,
) -> tuple[str, Path | None]:
    """Pick Mode A vs Mode B based on what's on disk.

    Returns ``(mode, resolved_dataset_path)``:

    * ``mode``: ``"dataset_only"`` (Mode A) or ``"full_predict"`` (Mode B).
    * ``resolved_dataset_path``: the dataset.jsonl that should drive predict
      in Mode A; ``None`` when running Mode B.

    Resolution order:

    1. Explicit ``dataset_path`` argument takes precedence — if it points at a
       readable file, run Mode A from that.
    2. The session's own ``cache/dataset/dataset.jsonl``, if it exists AND
       its referenced images are resolvable next to it. The router stages
       only the JSONL (not the images) when ``dataset_path`` came from an
       external location, and ``sync_from_store`` pulls only the JSONL prefix
       on cross-pod failover. Treating a JSONL whose images are missing as
       Mode A would make ``PredictConfigTask`` use the cache directory as
       ``image_base_dir`` and the predict step would fail with file-not-found
       errors — falling back to Mode B (rebuild from USD) is correct in that
       case.
    3. Otherwise Mode B (the executor will build the dataset from USD).
    """
    if dataset_path is not None:
        candidate = Path(dataset_path)
        if candidate.exists() and candidate.is_file():
            return "dataset_only", candidate
        # Fall through — handler should have validated, but if it didn't,
        # don't silently pretend the dataset exists.

    session_dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    if session_dataset.exists() and session_dataset.is_file():
        if _dataset_jsonl_has_resolvable_images(session_dataset):
            return "dataset_only", session_dataset
        logger.info(
            "Session dataset.jsonl exists but its images do not resolve next "
            "to it; falling back to Mode B (rebuild from USD) for "
            f"{session_dir.name[:8]}..."
        )

    return "full_predict", None


async def execute_predict_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager,
    *,
    dataset_path: Path | None = None,
) -> None:
    """Execute predict-only workflow for a /predict session.

    Parameters
    ----------
    session_id:
        Session identifier (UUID4).
    config_dict:
        Pipeline config dict (typically built via
        :func:`physics_agent.api.build_default_pipeline_config` plus any
        overrides). For Mode A we route this through ``run_predict``; for Mode
        B we route through ``run_pipeline`` with ``only_steps`` clamped to
        :data:`PREDICT_PIPELINE_STEPS`.
    session_manager:
        Session manager owning persistence + locking.
    dataset_path:
        Optional explicit dataset.jsonl path. When provided and readable,
        forces Mode A.
    """
    logger.info(f"/predict execution started for {session_id[:8]}...")

    session_dir = session_manager.get_session_dir(session_id)

    # Pull cache/ from the shared store before mode detection so a session
    # whose runnable dataset was prepared on a different instance is still
    # detected as Mode A. A synced JSONL whose images do not resolve remains
    # Mode B so the executor can rebuild from USD.
    if dataset_path is None:
        try:
            await session_manager.sync_from_store(session_id, prefix="cache/dataset/")
        except Exception:  # noqa: BLE001
            log_durable_failure(
                logger,
                "physics_predict_dataset_sync_failed",
                phase=FailurePhase.SYNC_UPLOAD,
                retryable=True,
            )

    mode, resolved_dataset = detect_predict_mode(
        session_dir=session_dir, dataset_path=dataset_path
    )
    logger.info(
        f"/predict mode for {session_id[:8]}: {mode}"
        + (f" (dataset={resolved_dataset})" if resolved_dataset else "")
    )

    # Persist mode + intended steps up front so /predict/{id}/status and
    # /predict/{id}/results expose this without depending on whether the
    # pipeline got far enough to populate step_results.
    if mode == "dataset_only":
        steps_run = ["predict"]
    else:
        steps_run = [
            s
            for s in PREDICT_PIPELINE_STEPS
            if config_dict.get("steps", {}).get(s, {}).get("enabled", False)
        ]
        # `predict` is always part of the run in Mode B, even if a caller
        # passed a config with predict disabled — the route is /predict.
        if "predict" not in steps_run:
            steps_run.append("predict")

    await session_manager.update_session(
        session_id,
        {
            "status": "running",
            "predict_mode": mode,
            "predict_steps_run": steps_run,
        },
    )

    listener = FastAPIEventListener(session_id, session_dir)
    event_bus = get_event_bus()

    # The body is wrapped in a single try/except so that an
    # asyncio.CancelledError (raised by JobRegistry.cancel) reliably lands on
    # a terminal "cancelled" status in session metadata + a CANCELLED event
    # on the bus. Without this wrapper, /predict/{id}/cancel would leave
    # status stuck on "cancelling".
    execution_failure: DurableDiagnostic | None = None
    try:
        if mode == "dataset_only":
            # Seed an EventBus snapshot for Mode A so /predict/{id}/events
            # can stream. arun_predict doesn't accept an event_listener, so
            # without this seeding the bus has no record of the session and
            # SSE returns 503. Mirrors the listener's "step.started"
            # semantics via a single RUNNING event so the SSE handler can
            # lifecycle-equivalently match /pipeline.
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step="predict",
                    state=StepState.RUNNING,
                    percent=0,
                    message="Predict (dataset_only) starting",
                )
            )

            # Mode A: only run predict. We pass the dict-form config through
            # to the predict workflow factory; dataset_override forces it to
            # read from the resolved jsonl.
            params = PredictInput(
                config=config_dict,
                dataset_override=resolved_dataset,
                output_dir_override=Path(session_dir) / "cache" / "predictions",
                verbose=False,
            )
            result = await arun_predict(params)

            if not result.success:
                raise _PredictResultError(
                    durable_diagnostic(
                        "physics_predict_result_failed",
                        phase=FailurePhase.PIPELINE_EXECUTION,
                        retryable=False,
                    )
                )

            stats = {
                "predictions_made": result.predictions_count,
                "failed_count": result.failed_count,
                "predictions_path": (
                    str(result.predictions_path) if result.predictions_path else None
                ),
                "token_stats": result.token_stats or {},
            }
        else:
            # Mode B: run upstream prep + predict via the pipeline workflow.
            # only_steps keeps apply_physics off (the pipeline default
            # config would otherwise enable it).
            only_steps = [
                s
                for s in PREDICT_PIPELINE_STEPS
                if config_dict.get("steps", {}).get(s, {}).get("enabled", False)
            ]
            if "predict" not in only_steps:
                only_steps.append("predict")

            result = await arun_pipeline(
                PipelineInput(
                    config=config_dict,
                    event_listener=listener,
                    only_steps=only_steps,
                    verbose=False,
                )
            )

            if not result.success:
                raise _PredictResultError(
                    durable_diagnostic(
                        "physics_predict_result_failed",
                        phase=FailurePhase.PIPELINE_EXECUTION,
                        retryable=False,
                    )
                )

            stats = _extract_stats_from_pipeline_result(result, session_dir)

        # Pipeline duration
        metadata = await session_manager.get_session_metadata(session_id)
        duration_seconds = 0
        if metadata and metadata.get("created_at"):
            created_at = datetime.fromisoformat(metadata["created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

        # Persist a terminal overall_progress snapshot so /predict/{id}/status
        # reads from the store (other pods, post-eventbus-reset, or the bus
        # was never seeded for Mode A) report 100% rather than predict's
        # mid-pipeline 90% or Mode A's 0%. Matches what the in-memory
        # snapshot ends up with after the pipeline_completed event.
        await session_manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": stats,
                "duration_seconds": duration_seconds,
                "completed_at": datetime.now(UTC).isoformat(),
                "overall_progress": {
                    "percent": 100,
                    "current_step": 1 if mode == "dataset_only" else 4,
                    "total_steps": 1 if mode == "dataset_only" else 4,
                    "message": "Predict completed",
                },
            },
        )

        # Sync prediction artifacts to store; we deliberately omit the
        # physics/ prefix because /predict never runs apply_physics.
        synced = 0
        for prefix in (
            "cache/predictions/",
            "cache/dataset/dataset.jsonl",
        ):
            try:
                n = await session_manager.sync_to_store(session_id, prefix=prefix)
                synced += n
            except Exception:
                log_durable_failure(
                    logger,
                    "physics_predict_artifact_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
                )
        if synced > 0:
            logger.info(
                f"Synced {synced} predict artifact file(s) to store for "
                f"{session_id[:8]}"
            )

        # Mark the EventBus snapshot terminal. The bus's COMPLETED handler
        # is if/elif: when ``current_step.name == event.step`` it appends
        # the step result and clears current_step but does NOT flip status
        # (predict caps at 90%); the ``pipeline_completed`` branch is only
        # the elif arm. So a single event with both step="predict" and
        # extra.pipeline_completed=True flows through the first branch and
        # the elif never fires. We emit two events: the first closes the
        # predict step (clears current_step), the second carries
        # pipeline_completed and pipeline_ready so the elif can flip the
        # snapshot to "completed" and the SSE close logic still triggers.
        if event_bus.get_snapshot(session_id) is not None:
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step="predict",
                    state=StepState.COMPLETED,
                    percent=100,
                    message="Predict step finished",
                    extra={"predict_mode": mode},
                )
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step="predict",
                    state=StepState.COMPLETED,
                    percent=100,
                    message="Predict artifacts synced and ready",
                    extra={
                        "pipeline_ready": True,
                        "pipeline_completed": True,
                        "predict_mode": mode,
                    },
                )
            )

        logger.info(f"/predict execution completed for {session_id[:8]}")

    except asyncio.CancelledError:
        # JobRegistry.cancel() raised CancelledError into us. Persist the
        # terminal "cancelled" state and emit a CANCELLED event so SSE
        # subscribers + /status see a definitive end state, then re-raise so
        # JobRegistry's task cleanup proceeds normally.
        try:
            await session_manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "cancelled_at": datetime.now(UTC).isoformat(),
                    "can_cancel": False,
                },
            )
            if event_bus.get_snapshot(session_id) is not None:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="predict",
                        state=StepState.CANCELLED,
                        message="Predict cancelled",
                    )
                )
        except Exception:  # noqa: BLE001
            log_durable_failure(
                logger,
                "physics_predict_cancellation_record_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
        raise

    except _PredictResultError as failure:
        diagnostic = failure.diagnostic
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        await _mark_failed(session_manager, session_id, diagnostic, "predict")
        execution_failure = diagnostic

    except Exception:  # noqa: BLE001
        # Catch-all: if arun_predict / arun_pipeline / stats extraction /
        # artifact sync raises an unexpected error, the explicit
        # result.success==False paths above don't cover it, so without this
        # the session would stay "running" forever. _mark_failed records
        # metadata and emits the FAILED event so /predict/{id}/status and
        # SSE both observe a definitive terminal state.
        diagnostic = durable_diagnostic(
            "physics_predict_execution_failed",
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.PIPELINE_EXECUTION,
            retryable=False,
        )
        await _mark_failed(session_manager, session_id, diagnostic, "predict")
        execution_failure = diagnostic

    if execution_failure is not None:
        # Raise after leaving the handler so the original exception is not
        # retained as ``__context__`` on a background-task failure.
        raise RuntimeError(execution_failure.code)


async def _mark_failed(
    session_manager,
    session_id: str,
    diagnostic: DurableDiagnostic,
    failed_step: str,
) -> None:
    """Record a failure in session metadata and on the EventBus snapshot.

    Emitting the FAILED event is required for any path that seeded a
    snapshot upstream (Mode A seeds RUNNING before arun_predict), because
    /predict/{id}/status prefers the snapshot over store-backed metadata
    on the executing instance — without the event the snapshot stays
    "running" forever.
    """
    try:
        await session_manager.update_session(
            session_id,
            {
                "status": "failed",
                "error": diagnostic.code,
                "error_diagnostic": diagnostic.to_dict(),
                "failed_step": failed_step,
            },
        )
    except Exception:
        log_durable_failure(
            logger,
            "physics_predict_failure_metadata_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )

    try:
        from ..runtime import get_event_bus

        bus = get_event_bus()
        if bus.get_snapshot(session_id) is not None:
            await bus.emit(
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
            "physics_predict_failure_event_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )


def _extract_stats_from_pipeline_result(result, session_dir: Path) -> dict[str, Any]:
    """Pull predict-relevant stats from a pipeline run.

    Mirrors `pipeline.executor._extract_stats_from_result` but only for the
    fields /predict surfaces.
    """
    stats: dict[str, Any] = {
        "prims_processed": 0,
        "images_generated": 0,
        "predictions_made": 0,
        "failed_count": 0,
        "predictions_path": None,
        "token_stats": {},
    }

    step_results = result.step_results or {}

    if "predict" in step_results:
        predict_out = step_results["predict"] or {}
        stats["predictions_made"] = predict_out.get("predictions_count", 0)
        stats["failed_count"] = predict_out.get("failed_count", 0)
        path = predict_out.get("predictions_path")
        if path:
            stats["predictions_path"] = str(path)
        token_stats = predict_out.get("token_stats")
        if token_stats:
            stats["token_stats"] = token_stats

    # When the run includes restore_usd (optimize_usd=true Mode B), the
    # restored predictions live at restored_predictions_path and use the
    # original scene's prim paths. Mirror it onto cache/predictions/
    # predictions.jsonl so /artifacts/{id}/predictions and the recorded
    # predictions_path expose the original-scene paths instead of the
    # optimized/deinstanced ones the predict step emitted.
    if "restore_usd" in step_results:
        restore_out = step_results["restore_usd"] or {}
        restored_path = restore_out.get("restored_predictions_path")
        if restored_path:
            try:
                target = session_dir / "cache" / "predictions" / "predictions.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not Path(restored_path).resolve().samefile(target):
                    shutil.copyfile(restored_path, target)
                stats["predictions_path"] = str(target)
            except Exception:  # noqa: BLE001
                log_durable_failure(
                    logger,
                    "physics_predict_stats_collection_failed",
                    phase=FailurePhase.PIPELINE_EXECUTION,
                    retryable=False,
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

    # File-based fallbacks — keeps results sane when the workflow returns
    # bare counts (e.g. when run_predict is invoked from Mode A).
    session_path = Path(session_dir)
    predictions_file = session_path / "cache" / "predictions" / "predictions.jsonl"
    if stats["predictions_made"] == 0 and predictions_file.exists():
        try:
            with open(predictions_file) as f:
                stats["predictions_made"] = sum(1 for line in f if line.strip())
        except Exception:
            log_durable_failure(
                logger,
                "physics_predict_stats_collection_failed",
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            )

    if stats["predictions_path"] is None and predictions_file.exists():
        stats["predictions_path"] = str(predictions_file)

    dataset_file = session_path / "cache" / "dataset" / "dataset.jsonl"
    if stats["prims_processed"] == 0 and dataset_file.exists():
        try:
            with open(dataset_file) as f:
                stats["prims_processed"] = sum(1 for line in f if line.strip())
        except Exception:
            log_durable_failure(
                logger,
                "physics_predict_stats_collection_failed",
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            )

    return stats
