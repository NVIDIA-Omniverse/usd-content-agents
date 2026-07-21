# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.api.defaults import (
    DEFAULT_RENDER_BACKEND,
    build_default_pipeline_config,
)
from sse_starlette import EventSourceResponse
from world_understanding.functions.graphics.rendering_backend_factory import (
    RENDERING_BACKEND_NAMES,
    validate_rendering_backend_name,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.s3_utils import (
    S3BucketNotAllowedError,
    authorize_s3_uri_for_extensions,
    download_file_from_s3,
)

from ..config import config
from ..config_persistence import (
    build_and_validate_pipeline_config,
    build_and_write_pipeline_config,
)
from ..models.requests import RegenerateRequest
from ..models.responses import (
    S3_INPUT_ERROR_RESPONSES,
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..session.manager import SessionManager
from ..utils import derive_completed_step_names
from ..workers.executor import (
    execute_pipeline_async,
    terminalize_pipeline_cancellation,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


def _apply_render_request_limit(pipeline_config: dict) -> None:
    """Clamp remote render worker settings to the process-wide render cap."""
    from world_understanding.functions.graphics.render_remote_async import (
        get_global_remote_render_limit,
    )

    global_limit = get_global_remote_render_limit()
    if global_limit is None:
        return

    step_config = pipeline_config.get("steps", {}).get("build_dataset_usd")
    if not isinstance(step_config, dict):
        return

    existing_workers = step_config.get("num_workers", global_limit)
    try:
        worker_limit = min(int(existing_workers), global_limit)
    except (TypeError, ValueError):
        worker_limit = global_limit

    existing_requests = step_config.get("max_concurrent_requests", global_limit)
    try:
        request_limit = min(int(existing_requests), global_limit)
    except (TypeError, ValueError):
        request_limit = global_limit

    step_config["num_workers"] = max(1, worker_limit)
    step_config["max_concurrent_requests"] = max(1, request_limit)


async def _stream_copy(
    upload: UploadFile, dest: Path, chunk_size: int = 2 * 1024 * 1024
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    with dest.open("wb") as f:
        while True:
            data = await upload.read(chunk_size)
            if not data:
                break
            f.write(data)
            total_bytes += len(data)

    return total_bytes


_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


def _validate_and_authorize_s3_usd_uri(s3_uri: str) -> str:
    """Validate and authorize a client S3 USD URI without performing I/O."""
    try:
        return authorize_s3_uri_for_extensions(
            s3_uri,
            config.s3_allowed_buckets,
            allowed_extensions=_VALID_USD_EXTENSIONS,
        )
    except S3BucketNotAllowedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _download_s3_to_session(s3_uri: str, session_dir: Path) -> Path:
    """Reauthorize and download a client-supplied S3 USD into session input."""
    ext = _validate_and_authorize_s3_usd_uri(s3_uri)

    local_path = session_dir / "input" / f"scene{ext}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file_from_s3(s3_uri, local_path)
    except FileNotFoundError:
        log_durable_failure(
            logger,
            "pipeline_s3_object_not_found",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(status_code=404, detail="S3 object not found") from None
    except PermissionError:
        log_durable_failure(
            logger,
            "pipeline_s3_access_denied",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=403, detail="Access denied to S3 object"
        ) from None
    except Exception:
        log_durable_failure(
            logger,
            "pipeline_s3_download_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=502, detail="Failed to download from S3"
        ) from None

    size_mb = local_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"S3 file too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )

    return local_path


@router.post(
    "/upload-usd",
    response_model=SessionCreated,
    status_code=201,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def upload_usd_immediate(
    usd_file: UploadFile = File(
        None, description="USD file to upload (provide this OR s3_uri)"
    ),
    s3_uri: str = Form(
        None,
        description="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    ),
) -> SessionCreated:
    """Upload a USD file and create a session for later pipeline execution."""
    if not usd_file and not s3_uri:
        raise HTTPException(
            status_code=400,
            detail="Either usd_file or s3_uri must be provided",
        )
    if usd_file and s3_uri:
        raise HTTPException(
            status_code=400,
            detail="Provide either usd_file or s3_uri, not both",
        )

    if s3_uri:
        _validate_and_authorize_s3_usd_uri(s3_uri)

    manager = get_session_manager()
    session_id = str(uuid.uuid4())

    if s3_uri:
        session_dir = await manager.create_session(session_id)
        failure_phase = FailurePhase.LOCAL_PUBLICATION
        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
            # Push input to the shared store BEFORE marking the session
            # ready. In multi-instance deploys the durable session
            # metadata is replicated, but the input artifacts are only
            # visible to other replicas via this sync. If we advertised
            # status=ready first and the sync then failed, a follow-up
            # POST /pipeline routed to another instance would 400 with
            # "Input USD not found for session" despite /sessions
            # showing ready.
            failure_phase = FailurePhase.SYNC_UPLOAD
            await manager.sync_to_store(session_id)

            # Persist the upload outcome so GET /sessions and /sessions/{id}
            # reflect "USD ready, pipeline not started yet" instead of the
            # default "pending / config:{}" left by create_session, which is
            # indistinguishable from a placeholder session.
            # Record the S3 object key basename rather than local_path.name
            # -- _download_s3_to_session normalizes the file to scene.<ext>
            # so local_path.name would lose the user-facing filename the
            # operator probably recognizes from the bucket.
            s3_basename = s3_uri.rstrip("/").rsplit("/", 1)[-1] or local_path.name
            failure_phase = FailurePhase.PERSISTENCE_VERIFICATION
            await manager.update_session(
                session_id,
                {
                    "status": "ready",
                    "config": {
                        "has_usd_upload": True,
                        "usd_path": str(local_path),
                        "s3_uri": s3_uri,
                        "original_filename": s3_basename,
                        "size_mb": round(size_mb, 2),
                    },
                },
            )

            return SessionCreated(
                session_id=session_id,
                status="ready",
                message=f"USD downloaded from S3 successfully ({size_mb:.1f}MB)",
                estimated_duration_minutes=0,
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_s3_ingest_failed",
                phase=failure_phase,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to download USD from S3"
            ) from None

    # File upload path
    if usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in _VALID_USD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid USD file type: {ext}. Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
            )

    session_dir = await manager.create_session(session_id)

    original_ext = (
        Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
    )
    usd_path = session_dir / "input" / f"scene{original_ext}"

    failure_phase = FailurePhase.LOCAL_PUBLICATION
    try:
        total_bytes = await _stream_copy(usd_file, usd_path)
        size_mb = total_bytes / (1024 * 1024)

        if size_mb > config.max_upload_size_mb:
            usd_path.unlink(missing_ok=True)
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
            )

        logger.info(
            f"USD uploaded for session {session_id[:8]}: {size_mb:.2f}MB ({original_ext})"
        )

        # Sync input to the shared store before advertising ready (see
        # s3 branch above for rationale).
        failure_phase = FailurePhase.SYNC_UPLOAD
        await manager.sync_to_store(session_id)

        # Persist the upload outcome (see s3 branch above for rationale).
        failure_phase = FailurePhase.PERSISTENCE_VERIFICATION
        await manager.update_session(
            session_id,
            {
                "status": "ready",
                "config": {
                    "has_usd_upload": True,
                    "usd_path": str(usd_path),
                    "s3_uri": None,
                    "original_filename": (
                        usd_file.filename if usd_file.filename else None
                    ),
                    "size_mb": round(size_mb, 2),
                },
            },
        )

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded successfully",
            estimated_duration_minutes=0,
        )

    except HTTPException:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "pipeline_usd_upload_failed",
            phase=failure_phase,
            retryable=True,
        )
        await manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail="Failed to upload USD") from None


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


@router.post(
    "",
    response_model=SessionCreated,
    status_code=202,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def create_pipeline(
    usd_file: UploadFile = File(
        None,
        description=(
            "USD file to process. Lowest-priority source; used only when neither "
            "session_id nor s3_uri is provided."
        ),
    ),
    session_id: str = Form(
        None,
        description=(
            "Existing session ID (from /upload-usd endpoint). Highest-priority "
            "source when multiple source fields are supplied."
        ),
    ),
    s3_uri: str = Form(
        None,
        description=(
            "S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz). Used when "
            "session_id is absent, ahead of usd_file."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional)",
    ),
    render_backend: str = Form(
        default="",
        description=(
            "Rendering backend: 'remote' (default, HTTP render service; the "
            "bundled compose points this at the OVRTX sidecar), 'warp' "
            "(local CUDA), 'ovrtx' (local Vulkan subprocess), or 'mock' "
            "(deterministic CPU-only test images)"
        ),
        json_schema_extra={"enum": [*RENDERING_BACKEND_NAMES, ""]},
    ),
    optimize_usd: bool = Form(
        default=False,
        description="Enable USD optimization step (default: false). "
        "When enabled, runs Scene Optimizer before rendering/prediction "
        "and restore_usd afterward to map results back to original paths.",
    ),
    enable_deinstance: bool = Form(
        default=True,
        description="Enable deinstance operation when optimize_usd is true "
        "(default: true). Required for instanced USD assets "
        "(e.g. robot arms with shared prototypes). FastAPI accepts common "
        "boolean form values such as true/false, 1/0, yes/no, and on/off.",
    ),
    enable_split: bool = Form(
        default=False,
        description="Enable split meshes operation when optimize_usd is true "
        "(default: false).",
    ),
    enable_deduplicate: bool = Form(
        default=False,
        description="Enable deduplicate operation when optimize_usd is true "
        "(default: false).",
    ),
) -> SessionCreated:
    """Create and execute a physics agent pipeline.

    At least one input source is required. When more than one is supplied, the
    selected source is deterministic: ``session_id`` first, then ``s3_uri``,
    then ``usd_file``. Lower-priority source fields are ignored.
    """
    try:
        render_backend_text = validate_rendering_backend_name(
            render_backend.strip()
            if render_backend and render_backend.strip()
            else DEFAULT_RENDER_BACKEND
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    user_prompt_text = user_prompt.strip() if user_prompt else None
    session_created_here = False

    # Validate request-only flags before allocating a session.
    if optimize_usd and not any([enable_deinstance, enable_split, enable_deduplicate]):
        raise HTTPException(
            status_code=400,
            detail="At least one optimization operation must be enabled when "
            "optimize_usd is true (enable_deinstance, enable_split, or "
            "enable_deduplicate).",
        )

    selected_s3_uri = s3_uri if s3_uri and not session_id else None
    selected_usd_file = (
        usd_file
        if usd_file is not None and not session_id and not selected_s3_uri
        else None
    )
    if selected_s3_uri:
        _validate_and_authorize_s3_usd_uri(selected_s3_uri)

    manager = get_session_manager()

    if session_id:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_dir = manager.get_session_dir(session_id)

    elif selected_s3_uri:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True

        try:
            local_path = _download_s3_to_session(selected_s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_s3_ingest_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to download USD from S3"
            ) from None

    elif selected_usd_file:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True

        try:
            if selected_usd_file.filename:
                ext = Path(selected_usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid USD file type: {ext}. Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
                    )

            original_ext = (
                Path(selected_usd_file.filename).suffix.lower()
                if selected_usd_file.filename
                else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
            total_bytes = await _stream_copy(selected_usd_file, usd_path)
            size_mb = total_bytes / (1024 * 1024)

            if size_mb > config.max_upload_size_mb:
                usd_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
                )

            logger.info(
                f"USD uploaded for session {session_id[:8]}: {size_mb:.2f}MB ({original_ext})"
            )

        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_usd_local_publication_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to save USD file"
            ) from None

    else:
        raise HTTPException(
            status_code=400,
            detail="One of usd_file, session_id, or s3_uri must be provided",
        )

    if session_id is None:  # defensive guard before derived path/store mutation
        raise HTTPException(
            status_code=500,
            detail="Pipeline session initialization failed",
        )
    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        # May be on a different instance — pull input/ from store and retry
        pulled = await manager.sync_from_store(session_id, prefix="input/")
        if pulled > 0:
            logger.info(
                f"Pulled {pulled} input file(s) from store for session {session_id[:8]}"
            )
        input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path:
        raise HTTPException(status_code=400, detail="Input USD not found for session")

    config_path = session_dir / "input" / "config.yaml"

    def prepare_pipeline_config() -> dict[str, Any]:
        prepared: dict[str, Any] = build_default_pipeline_config(
            session_id=session_id,
            usd_path=str(input_usd_path),
            working_dir=str(session_dir / "cache"),
            user_prompt=user_prompt_text,
            render_backend=render_backend_text,
            optimize_usd=optimize_usd,
            enable_deinstance=enable_deinstance,
            enable_split=enable_split,
            enable_deduplicate=enable_deduplicate,
        )
        _apply_render_request_limit(prepared)
        return prepared

    job_registry = get_job_registry()
    try:
        reservation = await job_registry.reserve(session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    # Hold the registry slot across config persistence and state reset. A
    # concurrent request for the same reusable session must lose before it can
    # clear cancellation, overwrite config.yaml, or erase the live snapshot.
    async with reservation:
        pipeline_config = await build_and_write_pipeline_config(
            config_factory=prepare_pipeline_config,
            config_path=config_path,
            session_manager=manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )

        # Reset status to "pending" and write the pipeline config before
        # queueing the job. Two non-obvious requirements:
        #
        # 1. Status reset: /pipeline/upload-usd persists status="ready" for
        #    upload-only sessions, so without an explicit reset here a
        #    subsequent POST /pipeline against an upload-only session would
        #    leave the persisted state at "ready" until the executor starts.
        #    GET /pipeline/{id}/status would lie about a queued job and
        #    POST /pipeline/{id}/cancel would 400 (cancel only allows
        #    pending/running).
        #
        # 2. Config merge: /pipeline/upload-usd writes upload-only metadata
        #    (original_filename, size_mb, has_usd_upload) into config that
        #    operators rely on for /sessions visibility. Replacing config
        #    wholesale here would erase those fields the moment the user
        #    starts the pipeline. Spread existing config first, then layer
        #    the pipeline-specific keys on top, and OR has_usd_upload across
        #    both writers (start-from-session-id sets selected_usd_file=None, which
        #    would otherwise flip the flag back to False).
        #
        # The single update_session call is atomic so the status reset and
        # the merged-config write land together.
        existing = await manager.get_session_metadata(session_id) or {}
        existing_config = existing.get("config") or {}
        if not session_created_here:
            await manager.clear_cancellation(session_id)
            await manager.clear_pipeline_terminal_claim(session_id)
        await manager.update_session(
            session_id,
            {
                "status": "pending",
                "current_step": None,
                "can_cancel": True,
                "error": None,
                "error_diagnostic": None,
                "failed_step": None,
                "completed_at": None,
                "cancelled_at": None,
                "completed_steps": [],
                "completed_step_names": [],
                "partial_results": None,
                "results": {},
                "duration_seconds": 0,
                "config": {
                    **existing_config,
                    "project_name": pipeline_config.get("project", {}).get("name", ""),
                    "usd_path": str(input_usd_path),
                    "has_usd_upload": existing_config.get("has_usd_upload", False)
                    or (
                        selected_usd_file is not None
                        and selected_usd_file.filename is not None
                    ),
                    "s3_uri": selected_s3_uri or existing_config.get("s3_uri"),
                    "user_prompt": user_prompt_text,
                    "optimize_usd": optimize_usd,
                    "enable_deinstance": enable_deinstance,
                    "enable_split": enable_split,
                    "enable_deduplicate": enable_deduplicate,
                },
            },
        )

        event_bus = get_event_bus()
        event_bus.cleanup_session(session_id)
        await event_bus.seed_pending_session(
            session_id,
            created_at=existing.get("created_at"),
        )

        await reservation.start(
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
            )
        )

    logger.info(f"Pipeline registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Pipeline queued for execution",
        estimated_duration_minutes=15,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to store-based SessionManager for completed/cross-instance sessions.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    # Try in-memory state first (active sessions on this instance)
    snapshot = event_bus.get_snapshot(session_id)
    if (
        snapshot
        and snapshot.get("status") in {"pending", "running", "cancelling"}
        and not get_job_registry().is_running(session_id)
    ):
        snapshot = None

    if snapshot:
        metadata = snapshot
        preview_images = snapshot.get("preview_images", [])
    else:
        # Fall back to store (works cross-instance)
        metadata = await manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        preview_images = metadata.get("preview_images", [])

    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    created_at = datetime.fromisoformat(metadata["created_at"])
    now = datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    elapsed_seconds = int((now - created_at).total_seconds())
    can_cancel = metadata.get("status") in ["pending", "running"]

    return PipelineStatus(
        session_id=session_id,
        status=metadata["status"],
        current_step=metadata.get("current_step"),
        completed_steps=metadata.get("completed_steps", []),
        overall_progress=metadata.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=can_cancel,
        elapsed_seconds=elapsed_seconds,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
    )


@router.get("/{session_id}/results", response_model=PipelineResults | PipelineError)
async def get_pipeline_results(session_id: str):
    """Get terminal pipeline execution results."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status in {"completed", "cancelled"}:
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            download_urls={
                "predictions": f"/artifacts/{session_id}/predictions",
                "report": f"/artifacts/{session_id}/report",
                "dataset": f"/artifacts/{session_id}/dataset",
                "output_usd": f"/artifacts/{session_id}/output-usd",
            },
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )

    elif status == "failed":
        completed_step_names = derive_completed_step_names(
            metadata.get("completed_step_names"),
            metadata.get("completed_steps"),
        )
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step=metadata.get("failed_step", "unknown"),
            completed_steps=completed_step_names,
            partial_results=metadata.get("partial_results"),
        )

    else:
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post("/{session_id}/cancel")
async def cancel_pipeline(session_id: str):
    """Cancel a running pipeline.

    Works cross-instance: writes a cancel signal to the store (S3)
    so the executing instance can detect it. Also tries local cancellation.
    """
    job_registry = get_job_registry()
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel pipeline with status: {metadata['status']}",
        )

    # Atomically claim cancellation against completion/failure before exposing
    # it locally. The metadata read above is only advisory across replicas.
    if not await manager.request_pipeline_cancellation(session_id):
        latest = await manager.get_session_metadata(session_id) or metadata
        raise HTTPException(
            status_code=409,
            detail=(
                "Pipeline reached terminal state before cancellation was accepted: "
                f"{latest.get('status', 'unknown')}"
            ),
        )
    await get_event_bus().mark_cancelling(session_id)

    # Also try local cancellation (fast path if this is the executing instance)
    if job_registry.is_running(session_id):
        cancelled = await job_registry.cancel(session_id)
        if cancelled and not job_registry.is_running(session_id):
            # A job cancelled while waiting for the registry semaphore never
            # enters execute_pipeline_async, so its cancellation handler cannot
            # terminalize the session. Running jobs do so before cancel()
            # returns; only fill the pre-start gap when state is still active.
            post_cancel = await manager.get_session_metadata(session_id)
            if post_cancel and post_cancel.get("status") in (
                "cancelling",
                "pending",
                "running",
            ):
                await terminalize_pipeline_cancellation(manager, session_id)

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Pipeline cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str):
    """Stream real-time progress events via Server-Sent Events (SSE).

    Only works when connected to the instance running the pipeline.
    For cross-instance progress, use GET /pipeline/{session_id}/status (polling).
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")

    # If the pipeline is not running on this instance, SSE can't stream live events.
    # Return immediately so the client falls back to polling.
    if snapshot is None:
        metadata = await manager.get_session_metadata(session_id)
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail="Pipeline is running on a different instance; use polling instead",
            )

    async def event_generator():  # pragma: no cover - SSE transport loop
        queue = event_bus.get_queue(session_id)

        # Check if already terminal (late connect to same instance after completion).
        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
            }
            return

        # If cross-instance and already terminal, send done and close.
        if snapshot is None:
            metadata = await manager.get_session_metadata(session_id)
            if metadata and metadata.get("status") in terminal_states:
                final_state = metadata["status"]
                yield {
                    "event": "done",
                    "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
                }
                return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    event_data = event.model_dump_json()

                    yield {
                        "event": "progress",
                        "data": event_data,
                    }

                    should_close = False

                    if event.state in ["failed", "cancelled"]:
                        should_close = True
                    elif event.extra and event.extra.get("pipeline_ready"):
                        # Executor fired this after update_session + sync_to_store —
                        # status and artifacts are now available in S3.
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    # Check store on each timeout in case pipeline completed on another instance
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        final_state = metadata["status"]
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
                        }
                        break
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)


@router.post("/{session_id}/regenerate", response_model=SessionCreated, status_code=202)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> SessionCreated:
    """Regenerate specific pipeline steps from cached data."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot regenerate while pipeline is {metadata['status']}",
        )

    session_dir = manager.get_session_dir(session_id)
    config_path = session_dir / "input" / "config.yaml"

    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original config not found for session",
        )

    only_steps = [s.value for s in request.steps]

    def prepare_regeneration_config() -> dict[str, Any]:
        with open(config_path) as config_file:
            prepared_config = yaml.safe_load(config_file)
        _apply_render_request_limit(prepared_config)

        if request.user_prompt is not None:
            steps_section = prepared_config.get("steps", {})
            prepare_dataset = steps_section.get(
                "build_dataset_prepare_dataset",
                {},
            )
            prompts = prepare_dataset.get("prompts", {})
            prompts["user"] = request.user_prompt
            prepare_dataset["prompts"] = prompts
            steps_section["build_dataset_prepare_dataset"] = prepare_dataset
            prepared_config["steps"] = steps_section
        return prepared_config

    pipeline_config = await build_and_validate_pipeline_config(
        config_factory=prepare_regeneration_config,
        session_manager=manager,
        session_id=session_id,
        session_created_here=False,
    )

    job_registry = get_job_registry()

    # Atomically claim the slot BEFORE writing session state. Mirrors the
    # /predict rerun fix: a losing concurrent /regenerate must not flip
    # status to "pending" or clear current_step before getting 409.
    try:
        reservation = await job_registry.reserve(session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    async with reservation:
        await manager.clear_cancellation(session_id)
        await manager.clear_pipeline_terminal_claim(session_id)
        await manager.update_session(
            session_id,
            {
                "status": "pending",
                "current_step": None,
                "can_cancel": True,
                "error": None,
                "error_diagnostic": None,
                "failed_step": None,
                "completed_at": None,
                "completed_steps": [],
                "completed_step_names": [],
                "partial_results": None,
                "results": {},
                "duration_seconds": 0,
                "cancelled_at": None,
            },
        )
        event_bus = get_event_bus()
        event_bus.cleanup_session(session_id)
        await event_bus.seed_pending_session(
            session_id,
            created_at=metadata.get("created_at"),
        )

        await reservation.start(
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                only_steps=only_steps,
            ),
        )

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str):
    """Get the persisted event log for a session."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Try store first (works cross-instance)
    try:
        events = await manager.store.get_event_log(session_id)
        return {"events": events, "total": len(events)}
    except Exception:
        log_durable_failure(
            logger,
            "event_log_store_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )

    # Fall back to local file
    log_file = manager.get_session_dir(session_id) / "event_log.jsonl"
    if not log_file.exists():
        return {"events": []}

    events = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return {"events": events, "total": len(events)}
    except Exception:
        log_durable_failure(
            logger,
            "event_log_local_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to load event log",
        ) from None
