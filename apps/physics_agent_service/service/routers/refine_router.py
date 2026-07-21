# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refine API endpoints for iterative Physics Agent tuning."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    DEFAULT_REFERENCE_VIDEO_FRAMES,
    validate_visual_frame_count,
)
from sse_starlette import EventSourceResponse
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import HeldFileResponse

from ..artifact_contract import (
    REFINE_ARTIFACT_SPECS,
    artifact_name_from_key,
    available_artifact_keys,
)
from ..config import config
from ..config_persistence import avalidate_durable_request_content
from ..models.responses import (
    S3_INPUT_ERROR_RESPONSES,
    PipelineError,
    RefineResults,
    RefineStatus,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..runtime.events import StepState
from ..session.manager import SessionManager
from .tune_router import (
    _copy_from_source_session,
    _copy_reference_uploads,
    _download_s3_to_session,
    _find_input_physics,
    _nonempty_uploads,
    _parse_reference_descriptions,
    _scenario_param_names_from_mapping,
    _stream_copy,
    _validate_and_authorize_s3_usd_uri,
    _validate_engine_name_for_request,
    _validate_engine_supports_param_names_for_request,
    _validate_ovphysx_runtime_for_request,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/refine", tags=["refine"])

session_manager: SessionManager | None = None

_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
_VALID_REFERENCE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VALID_REFERENCE_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".avi",
    ".mkv",
}
_MAX_REFINE_TRIALS = 1000
_MAX_REFINE_ITERATIONS = 12
_MAX_SCENARIO_YAML_BYTES = 64 * 1024
_MAX_USER_PROMPT_BYTES = 16 * 1024
_MAX_REFERENCE_UPLOADS = 16


def get_session_manager() -> SessionManager:
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    global session_manager
    session_manager = manager


def _validate_optimizer_name_for_request(optimizer: str) -> None:
    from physics_agent.tuning.optimizers import SUPPORTED_OPTIMIZERS

    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown optimizer {optimizer!r}. "
                f"Supported: {sorted(SUPPORTED_OPTIMIZERS)}"
            ),
        )


def _coerce_finite_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first_present(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


async def _refine_download_urls(
    manager: SessionManager,
    session_id: str,
    metadata: dict,
) -> dict[str, str]:
    available = await available_artifact_keys(manager, session_id, metadata, "refine")
    urls = {
        spec.logical_name: (
            f"/refine/{session_id}/artifacts/"
            f"{artifact_name_from_key(spec.key, 'refine')}"
        )
        for spec in REFINE_ARTIFACT_SPECS
        if spec.key in available
    }
    legacy_tuned_usd = "refine/final/tuned_physics.usda"
    if "final_tuned_usd" not in urls and legacy_tuned_usd in available:
        urls["final_tuned_usd"] = (
            f"/refine/{session_id}/artifacts/final/tuned_physics.usda"
        )

    return urls


def _validate_source_session_id(source_session_id: str | None) -> None:
    if source_session_id is None or not source_session_id.strip():
        return
    from ..session.manager import _SESSION_ID_PATTERN

    if not _SESSION_ID_PATTERN.fullmatch(source_session_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "source_session_id must be a UUID4-shaped string; "
                f"got {source_session_id!r}"
            ),
        )


def _validate_route_session_id(session_id: str) -> None:
    from ..session.manager import _SESSION_ID_PATTERN

    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=400,
            detail=f"session_id must be a UUID4-shaped string; got {session_id!r}",
        )


def _validate_scenario_yaml_for_refine(scenario_yaml_text: str, engine: str) -> None:
    try:
        scenario_data = yaml.safe_load(scenario_yaml_text)
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Invalid scenario YAML") from None
    if not isinstance(scenario_data, dict):
        raise HTTPException(
            status_code=400, detail="scenario_yaml must parse to a mapping"
        )

    try:
        from physics_agent.tuning.errors import TuningError
        from physics_agent.tuning.runner import _validate_engine_supports_scenario
        from physics_agent.tuning.scenario import load_scenario

        parsed = load_scenario(scenario_data)
        _validate_engine_supports_scenario(engine, parsed.name)
        _validate_engine_supports_param_names_for_request(
            engine,
            {param.name for param in parsed.params}
            or _scenario_param_names_from_mapping(scenario_data),
        )
    except (ValueError, TuningError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid scenario: {exc}") from exc


def _metadata_elapsed_seconds(metadata: dict[str, object]) -> int:
    created_at = datetime.fromisoformat(str(metadata["created_at"]))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return int((datetime.now(UTC) - created_at).total_seconds())


@router.post(
    "",
    response_model=SessionCreated,
    status_code=202,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def create_refine(
    physics_usd: UploadFile | None = File(
        None,
        description="Physics-authored USD (output of apply_physics) to refine",
    ),
    s3_uri: str | None = Form(None, description="S3 URI to a physics-authored USD"),
    source_session_id: str | None = Form(
        None,
        description="Pipeline session ID whose apply_physics output_usd should be used",
    ),
    reference_images: list[UploadFile] = File(
        default=[],
        description="Optional reference images for the visual/VLM judge",
    ),
    reference_videos: list[UploadFile] = File(
        default=[],
        description="Optional reference videos for the visual/VLM judge",
    ),
    reference_descriptions: str = Form(
        default="",
        description="Optional JSON array of descriptions parallel to reference_images",
    ),
    reference_video_descriptions: str = Form(
        default="",
        description="Optional JSON array of descriptions parallel to reference_videos",
    ),
    reference_video_frames: int = Form(
        default=DEFAULT_REFERENCE_VIDEO_FRAMES,
        description="Frames to extract from each reference video for visual judging",
    ),
    judge_reference_frames: int = Form(
        default=DEFAULT_JUDGE_REFERENCE_FRAMES,
        description="Max reference images/video frames to send to the VLM judge",
    ),
    judge_generated_frames: int = Form(
        default=DEFAULT_JUDGE_GENERATED_FRAMES,
        description="Max generated render frames to send to the VLM judge",
    ),
    scenario_yaml: str = Form(
        default="",
        description="Initial refine scenario YAML body",
    ),
    user_prompt: str = Form(
        default="",
        description="Natural-language desired behavior for judge/refine loop",
    ),
    optimizer: str = Form(
        default="botorch", description="botorch, auto, random, cma-es"
    ),
    engine: str = Form(default="ovphysx", description="ovphysx, newton, or fake"),
    max_trials: int = Form(
        default=30, description="Optimizer trial budget per iteration"
    ),
    max_iterations: int = Form(default=5, description="Refine iteration cap"),
    score_threshold: float = Form(default=0.9, description="Judge approval threshold"),
    seed: int = Form(default=42, description="Seed for optimizer + backend"),
    judge_max_tokens: int | None = Form(
        default=None,
        description="Optional max output tokens for judge responses",
    ),
    judge_temperature: float | None = Form(
        default=None,
        description="Optional temperature for judge calls",
    ),
    visual_evidence_enabled: bool = Form(
        default=True,
        description="Send generated/reference media to the VLM judge",
    ),
    llm_timeout_seconds: float = Form(
        default=180.0,
        description="Wall-clock deadline for each judge/refine LLM call",
    ),
) -> SessionCreated:
    """Create an iterative refine session and queue it for background execution."""

    s3_uri_text = (s3_uri or "").strip()
    source_session_id_text = (source_session_id or "").strip()
    sources_set = sum(
        1 for source in (physics_usd, s3_uri_text, source_session_id_text) if source
    )
    if sources_set != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly one of physics_usd, s3_uri, or source_session_id "
                "must be provided"
            ),
        )

    if not (1 <= max_trials <= _MAX_REFINE_TRIALS):
        raise HTTPException(
            status_code=400,
            detail=f"max_trials must be between 1 and {_MAX_REFINE_TRIALS}, got {max_trials}.",
        )
    if not (1 <= max_iterations <= _MAX_REFINE_ITERATIONS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_iterations must be between 1 and {_MAX_REFINE_ITERATIONS}, "
                f"got {max_iterations}."
            ),
        )
    if not math.isfinite(score_threshold) or not (0.0 <= score_threshold <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"score_threshold must be finite and between 0 and 1, got {score_threshold}.",
        )
    if judge_max_tokens is not None and judge_max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail=f"judge_max_tokens must be >= 1, got {judge_max_tokens}.",
        )
    if judge_temperature is not None and (
        not math.isfinite(judge_temperature) or judge_temperature < 0.0
    ):
        raise HTTPException(
            status_code=400,
            detail=f"judge_temperature must be finite and >= 0, got {judge_temperature}.",
        )
    if not math.isfinite(llm_timeout_seconds):
        raise HTTPException(
            status_code=400,
            detail=f"llm_timeout_seconds must be finite, got {llm_timeout_seconds}.",
        )
    try:
        reference_video_frames = validate_visual_frame_count(
            "reference_video_frames",
            reference_video_frames,
        )
        judge_reference_frames = validate_visual_frame_count(
            "judge_reference_frames",
            judge_reference_frames,
        )
        judge_generated_frames = validate_visual_frame_count(
            "judge_generated_frames",
            judge_generated_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _validate_engine_name_for_request(engine)
    _validate_ovphysx_runtime_for_request(engine)
    _validate_optimizer_name_for_request(optimizer)

    scenario_yaml_text = scenario_yaml or ""
    user_prompt_text = (user_prompt or "").strip()
    if len(scenario_yaml_text.encode("utf-8")) > _MAX_SCENARIO_YAML_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"scenario_yaml exceeds {_MAX_SCENARIO_YAML_BYTES // 1024} KB size limit",
        )
    if len(user_prompt_text.encode("utf-8")) > _MAX_USER_PROMPT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"user_prompt exceeds {_MAX_USER_PROMPT_BYTES // 1024} KB size limit",
        )
    if not scenario_yaml_text.strip():
        raise HTTPException(status_code=400, detail="scenario_yaml must be supplied")
    if not user_prompt_text:
        raise HTTPException(status_code=400, detail="user_prompt must be supplied")
    canonical_yaml = await avalidate_durable_request_content(
        {
            "user_prompt": user_prompt_text,
            "reference_descriptions": reference_descriptions,
            "reference_video_descriptions": reference_video_descriptions,
            "s3_uri": s3_uri_text or None,
        },
        yaml_documents={"scenario_yaml": scenario_yaml_text},
        context="physics refine durable request content",
    )
    scenario_yaml_text = canonical_yaml["scenario_yaml"]
    _validate_scenario_yaml_for_refine(scenario_yaml_text, engine)
    _validate_source_session_id(source_session_id_text)

    reference_image_uploads = _nonempty_uploads(reference_images)
    reference_video_uploads = _nonempty_uploads(reference_videos)
    if (
        len(reference_image_uploads) + len(reference_video_uploads)
        > _MAX_REFERENCE_UPLOADS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Too many reference media files. Max total: {_MAX_REFERENCE_UPLOADS}",
        )

    parsed_reference_descriptions = _parse_reference_descriptions(
        reference_descriptions,
        "reference_descriptions",
    )
    parsed_reference_video_descriptions = _parse_reference_descriptions(
        reference_video_descriptions,
        "reference_video_descriptions",
    )
    if parsed_reference_descriptions is not None and len(
        parsed_reference_descriptions
    ) != len(reference_image_uploads):
        raise HTTPException(
            status_code=400,
            detail=(
                "reference_descriptions must have one item per reference image "
                f"({len(reference_image_uploads)} expected)"
            ),
        )
    if parsed_reference_video_descriptions is not None and len(
        parsed_reference_video_descriptions
    ) != len(reference_video_uploads):
        raise HTTPException(
            status_code=400,
            detail=(
                "reference_video_descriptions must have one item per reference video "
                f"({len(reference_video_uploads)} expected)"
            ),
        )

    if s3_uri_text:
        _validate_and_authorize_s3_usd_uri(s3_uri_text)

    manager = get_session_manager()
    session_id = str(uuid.uuid4())
    session_dir = await manager.create_session(session_id)

    try:
        if physics_usd:
            ext = Path(physics_usd.filename or "").suffix.lower() or ".usd"
            if ext not in _VALID_USD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid USD file type: {ext}. "
                        f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}"
                    ),
                )
            usd_path = session_dir / "input" / f"physics{ext}"
            total = await _stream_copy(
                physics_usd,
                usd_path,
                max_bytes=config.max_upload_size_mb * 1024 * 1024,
                too_large_detail=f"File too large. Max: {config.max_upload_size_mb}MB",
            )
            if total <= 0:
                raise HTTPException(status_code=400, detail="physics_usd is empty")
        elif s3_uri_text:
            await asyncio.to_thread(_download_s3_to_session, s3_uri_text, session_dir)
        else:
            await _copy_from_source_session(
                manager, source_session_id_text, session_dir
            )
    except HTTPException:
        await manager.delete_session(session_id)
        raise
    except Exception:
        await manager.delete_session(session_id)
        log_durable_failure(
            logger,
            "refine_input_provision_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to provision input physics USD",
        ) from None

    input_physics = _find_input_physics(session_dir)
    if not input_physics:
        await manager.delete_session(session_id)
        raise HTTPException(
            status_code=400, detail="Failed to provision input physics USD"
        )

    try:
        max_reference_batch_bytes = config.max_upload_size_mb * 1024 * 1024
        reference_image_paths, reference_batch_bytes = await _copy_reference_uploads(
            uploads=reference_image_uploads,
            session_dir=session_dir,
            subdir="reference_images",
            file_prefix="reference_image",
            valid_extensions=_VALID_REFERENCE_IMAGE_EXTENSIONS,
            label="reference image",
            max_batch_bytes=max_reference_batch_bytes,
        )
        reference_video_paths, _ = await _copy_reference_uploads(
            uploads=reference_video_uploads,
            session_dir=session_dir,
            subdir="reference_videos",
            file_prefix="reference_video",
            valid_extensions=_VALID_REFERENCE_VIDEO_EXTENSIONS,
            label="reference video",
            current_batch_bytes=reference_batch_bytes,
            max_batch_bytes=max_reference_batch_bytes,
        )
    except HTTPException:
        await manager.delete_session(session_id)
        raise
    except Exception as exc:
        await manager.delete_session(session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to copy reference media: {type(exc).__name__}",
        ) from exc

    scenario_path = session_dir / "input" / "scenario.yaml"
    scenario_path.write_text(scenario_yaml_text, encoding="utf-8")
    user_prompt_path = session_dir / "input" / "user_prompt.txt"
    user_prompt_path.write_text(user_prompt_text, encoding="utf-8")

    await manager.update_session(
        session_id,
        {
            "status": "pending",
            "kind": "refine",
            "can_cancel": True,
            "config": {
                "kind": "refine",
                "engine": engine,
                "optimizer": optimizer,
                "max_trials": max_trials,
                "max_iterations": max_iterations,
                "score_threshold": score_threshold,
                "seed": seed,
                "physics_usd": str(input_physics),
                "scenario_path": str(scenario_path),
                "user_prompt": user_prompt_text,
                "user_prompt_path": str(user_prompt_path),
                "reference_images": [str(p) for p in reference_image_paths],
                "reference_videos": [str(p) for p in reference_video_paths],
                "reference_descriptions": parsed_reference_descriptions,
                "reference_video_descriptions": parsed_reference_video_descriptions,
                "reference_video_frames": reference_video_frames,
                "judge_reference_frames": judge_reference_frames,
                "judge_generated_frames": judge_generated_frames,
                "judge_max_tokens": judge_max_tokens,
                "judge_temperature": judge_temperature,
                "visual_evidence_enabled": visual_evidence_enabled,
                "llm_timeout_seconds": llm_timeout_seconds,
                "source_session_id": source_session_id_text or None,
                "s3_uri": s3_uri_text or None,
            },
        },
    )
    await get_event_bus().seed_pending_session(session_id)

    from ..workers.refine_executor import execute_refine_async

    await get_job_registry().register(
        session_id,
        execute_refine_async(
            session_id=session_id,
            session_manager=manager,
            scenario_path=scenario_path,
            user_prompt=user_prompt_text,
            physics_usd=input_physics,
            reference_images=reference_image_paths,
            reference_videos=reference_video_paths,
            reference_descriptions=parsed_reference_descriptions,
            reference_video_descriptions=parsed_reference_video_descriptions,
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
            visual_evidence_enabled=visual_evidence_enabled,
            llm_timeout_seconds=llm_timeout_seconds,
        ),
    )

    logger.info("Refine queued for session %s", session_id)
    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Refine queued for execution",
        estimated_duration_minutes=15,
    )


def _ensure_refine_session(metadata: dict[str, object], session_id: str) -> None:
    session_config = metadata.get("config") or {}
    config_kind = (
        session_config.get("kind") if isinstance(session_config, dict) else None
    )
    metadata_kind = metadata.get("kind")
    if metadata_kind == "refine" or config_kind == "refine":
        return
    other_kind = metadata_kind or config_kind or "unknown"
    raise HTTPException(
        status_code=409,
        detail=f"Session {session_id} is not a refine session (kind={other_kind!r}).",
    )


@router.get("/{session_id}/status", response_model=RefineStatus)
async def get_refine_status(session_id: str) -> RefineStatus:
    _validate_route_session_id(session_id)
    event_bus = get_event_bus()
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    _ensure_refine_session(metadata, session_id)

    config_meta = metadata.get("config") or {}
    results = metadata.get("results") or {}
    snapshot = event_bus.get_snapshot(session_id) or {}
    current_step = snapshot.get("current_step") or {}
    progress = current_step.get("progress") or {}
    extra = current_step.get("extra") or {}

    latest_iteration = {}
    iterations = results.get("iterations") or []
    if iterations:
        latest_iteration = iterations[-1]

    return RefineStatus(
        session_id=session_id,
        status=metadata["status"],
        iteration=int(
            extra.get("iteration")
            or latest_iteration.get("iteration")
            or results.get("final_iteration")
            or 0
        ),
        max_iterations=int(
            extra.get("max_iterations") or config_meta.get("max_iterations") or 0
        ),
        n_trials=int(extra.get("n_trials") or progress.get("current") or 0),
        max_trials=int(extra.get("max_trials") or config_meta.get("max_trials") or 0),
        best_score=_coerce_finite_score(
            _first_present(extra.get("best_score"), latest_iteration.get("best_score"))
        ),
        best_params=_first_present(
            extra.get("best_params"),
            latest_iteration.get("best_params"),
            results.get("final_best_params"),
        ),
        judge_score=_coerce_finite_score(
            _first_present(
                extra.get("judge_score"), latest_iteration.get("judge_score")
            )
        ),
        termination_reason=results.get("termination_reason"),
        elapsed_seconds=_metadata_elapsed_seconds(metadata),
        can_cancel=metadata.get("status") in ("pending", "running"),
        error_message=metadata.get("error")
        if metadata.get("status") == "failed"
        else None,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
    )


@router.get("/{session_id}/results", response_model=RefineResults | PipelineError)
async def get_refine_results(session_id: str):
    _validate_route_session_id(session_id)
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    _ensure_refine_session(metadata, session_id)

    status = metadata["status"]
    if status in ("completed", "cancelled"):
        results = metadata.get("results") or {}
        return RefineResults(
            session_id=session_id,
            status=status,
            termination_reason=results.get("termination_reason", status),
            iteration_count=int(results.get("iteration_count") or 0),
            final_iteration=int(results.get("final_iteration") or 0),
            final_judge_score=_coerce_finite_score(results.get("final_judge_score")),
            iterations=results.get("iterations") or [],
            download_urls=await _refine_download_urls(manager, session_id, metadata),
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )
    if status == "failed":
        results = metadata.get("results") or {}
        if results:
            return RefineResults(
                session_id=session_id,
                status=status,
                termination_reason=results.get("termination_reason", "error"),
                iteration_count=int(results.get("iteration_count") or 0),
                final_iteration=int(results.get("final_iteration") or 0),
                final_judge_score=_coerce_finite_score(
                    results.get("final_judge_score")
                ),
                iterations=results.get("iterations") or [],
                download_urls=await _refine_download_urls(
                    manager, session_id, metadata
                ),
                duration_seconds=metadata.get("duration_seconds", 0),
                completed_at=metadata.get("completed_at", ""),
                error_message=metadata.get("error", "Unknown error"),
            )
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step="refine",
            completed_steps=[],
            partial_results=metadata.get("partial_results"),
        )
    raise HTTPException(status_code=202, detail=f"Refine still {status}")


@router.post("/{session_id}/cancel")
async def cancel_refine(session_id: str):
    _validate_route_session_id(session_id)
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    _ensure_refine_session(metadata, session_id)

    if metadata["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel refine with status: {metadata['status']}",
        )
    await manager.request_cancellation(session_id)
    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Refine cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_refine_events(session_id: str):
    _validate_route_session_id(session_id)
    event_bus = get_event_bus()
    manager = get_session_manager()
    snapshot = event_bus.get_snapshot(session_id)
    metadata = await manager.get_session_metadata(session_id)
    if snapshot is None and metadata is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if metadata is not None:
        _ensure_refine_session(metadata, session_id)

    terminal_states = ("completed", "failed", "cancelled")
    if snapshot is None:
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail="Refine is running on a different instance; use polling instead",
            )

    async def event_generator():  # pragma: no cover - SSE transport loop
        queue = event_bus.get_queue(session_id)
        if snapshot is not None and snapshot.get("status") in terminal_states:
            yield {
                "event": "done",
                "data": json.dumps(
                    {"session_id": session_id, "final_state": snapshot["status"]}
                ),
            }
            return
        metadata = await manager.get_session_metadata(session_id)
        if metadata and metadata.get("status") in terminal_states:
            yield {
                "event": "done",
                "data": json.dumps(
                    {"session_id": session_id, "final_state": metadata["status"]}
                ),
            }
            return
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "progress", "data": event.model_dump_json()}
                    if event.state in (StepState.FAILED, StepState.CANCELLED):
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {"session_id": session_id, "final_state": event.state}
                            ),
                        }
                        break
                    if event.extra and event.extra.get("refine_ready"):
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {"session_id": session_id, "final_state": event.state}
                            ),
                        }
                        break
                except TimeoutError:
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        yield {
                            "event": "done",
                            "data": json.dumps(
                                {
                                    "session_id": session_id,
                                    "final_state": metadata["status"],
                                }
                            ),
                        }
                        break
                    yield {"event": "ping", "data": "keepalive"}
        except asyncio.CancelledError:
            logger.debug("SSE stream cancelled for refine %s", session_id[:8])
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.get("/{session_id}/artifacts/{name:path}")
async def download_refine_artifact(session_id: str, name: str):
    _validate_route_session_id(session_id)
    manager = get_session_manager()
    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    _ensure_refine_session(metadata, session_id)

    allowed: dict[str, tuple[str, str]] = {
        artifact_name_from_key(spec.key, "refine"): (
            spec.media_type,
            spec.download_name,
        )
        for spec in REFINE_ARTIFACT_SPECS
    }
    allowed.update(
        {
            # Backward-compatible alias for sessions/clients created before the
            # canonical tuned artifact switched to .usd.
            "final/tuned_physics.usda": (
                "application/octet-stream",
                "tuned_physics.usda",
            ),
        }
    )
    if name not in allowed:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {name}")

    available = await available_artifact_keys(manager, session_id, metadata, "refine")
    media_type, filename = allowed[name]
    legacy_aliases = {"final/tuned_physics.usda": "final/tuned_physics.usd"}
    requested_key = f"refine/{name}"
    if requested_key not in available and name in legacy_aliases:
        requested_key = f"refine/{legacy_aliases[name]}"
    if requested_key not in available:
        raise HTTPException(status_code=404, detail=f"Artifact not available: {name}")

    selected_key = requested_key
    artifact = await manager.open_local_artifact_key(session_id, selected_key)
    if artifact is None:
        await manager.sync_from_store(session_id, prefix="refine/")
        artifact = await manager.open_local_artifact_key(session_id, selected_key)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact not available: {name}")
    if selected_key != f"refine/{name}":
        filename = Path(selected_key).name
    return HeldFileResponse(artifact, media_type=media_type, filename=filename)
