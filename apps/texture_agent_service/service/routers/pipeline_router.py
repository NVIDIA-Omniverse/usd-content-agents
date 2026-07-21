# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from filelock import Timeout
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from texture_agent.config.rendering_backends import (
    DEFAULT_TEXTURE_RENDERING_BACKEND,
)
from texture_agent.functions.cached_apply import is_valid_cached_texture_png
from texture_agent.planning import (
    TEXTURE_UV_AWARE_DEFAULT_CAP,
    TextureDiscoveryMode,
    TexturePlan,
    TextureUnitMode,
    validate_texture_plan_payload,
)
from texture_agent.tasks.plan_textures import backend_default_texture_cap
from world_understanding.utils.credentials import (
    InlineSecretError,
    ensure_no_inline_secrets,
)
from world_understanding.utils.s3_utils import (
    S3BucketNotAllowedError,
    authorize_s3_uri_for_extensions,
    download_file_from_s3,
)

from ..config import config
from ..models.requests import (
    MaterialTextures,
    RegenerateRequest,
    TextureDetailPolicy,
    TexturePipelineStep,
)
from ..models.responses import (
    S3_INPUT_ERROR_RESPONSES,
    PipelineError,
    PipelineResults,
    PipelineStatus,
    SessionCreated,
    TexturePlanStatus,
)
from ..runtime import ProgressEvent, StepState, get_event_bus, get_job_registry
from ..sanitization import sanitize_message, sanitize_payload, sanitize_step_stats
from ..session.manager import SessionManager
from ..storage import METADATA_KEY
from ..workers.executor import _artifact_download_urls, execute_pipeline_async
from .common import JSON_RESPONSE

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


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    chunk_size: int = 2 * 1024 * 1024,
    max_bytes: int = 0,
) -> int:
    """Stream upload file to disk in chunks to avoid memory spikes.

    Args:
        upload: FastAPI upload file.
        dest: Destination path.
        chunk_size: Read chunk size in bytes.
        max_bytes: Maximum allowed bytes (0 = unlimited).

    Raises:
        HTTPException: If the file exceeds max_bytes during streaming.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    try:
        with dest.open("wb") as f:
            while True:
                data = await upload.read(chunk_size)
                if not data:
                    break
                total_bytes += len(data)
                if max_bytes and total_bytes > max_bytes:
                    size_mb = total_bytes / (1024 * 1024)
                    limit_mb = max_bytes / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: >{size_mb:.1f}MB. Max: {limit_mb:.0f}MB"
                        ),
                    )
                f.write(data)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise

    return total_bytes


_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
_VALID_REFERENCE_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
_SIMPLE_TEXTURE_BACKENDS = {"simple", "simple_image_gen"}
_APPLY_CACHE_KEY_MODE_MARKER_KEY = "cache/apply_cache_key_mode.json"
_APPLY_CACHE_KEY_MODE_MARKER_SCHEMA = "texture-agent-apply-cache-key-mode.v1"
_APPLY_CACHE_KEY_MODE_LEGACY = "legacy_display"
_APPLY_CACHE_KEY_MODE_PLAN = "plan_unit_id"
_INVALID_PIPELINE_CONFIG_DETAIL = "Pipeline configuration is invalid"


def _normalized_backend_name(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def _is_simple_texture_backend(value: str | None) -> bool:
    return _normalized_backend_name(value) in _SIMPLE_TEXTURE_BACKENDS


def _canonical_texture_backend(value: str | None) -> str | None:
    normalized = _normalized_backend_name(value)
    if not normalized:
        return None
    if normalized in _SIMPLE_TEXTURE_BACKENDS:
        return "simple_image_gen"
    return normalized


def _write_pipeline_config(
    config_path: Path,
    pipeline_config: dict[str, Any],
) -> None:
    """Persist one config only after the complete resolved value is proven safe."""
    ensure_no_inline_secrets(
        pipeline_config,
        context="texture pipeline configuration",
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(pipeline_config, config_file, default_flow_style=False)


def _strip_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_texture_route(
    *,
    texture_backend: str | None,
    texture_endpoint: str | None,
    backend_engine: str | None,
    uv_scope: str | None,
    uv_rebake_source_albedo: bool | None,
    uv_rebake_size: int | None,
) -> dict[str, Any]:
    """Resolve request/backend defaults into one concrete pipeline route."""
    requested_backend = _canonical_texture_backend(texture_backend)
    configured_backend = (
        _canonical_texture_backend(config.texture_backend) or "simple_image_gen"
    )
    resolved_backend = requested_backend or configured_backend

    resolved_endpoint = _strip_or_none(texture_endpoint) or _strip_or_none(
        config.texture_endpoint
    )
    resolved_engine = _strip_or_none(backend_engine) or _strip_or_none(
        config.backend_engine
    )
    resolved_workers = config.texture_workers
    resolved_job_timeout_sec = config.texture_job_timeout_sec
    resolved_uv_scope = _strip_or_none(uv_scope) or config.uv_scope
    resolved_uv_rebake_source_albedo = (
        config.uv_rebake_source_albedo
        if uv_rebake_source_albedo is None
        else uv_rebake_source_albedo
    )
    resolved_uv_rebake_size = (
        config.uv_rebake_size if uv_rebake_size is None else uv_rebake_size
    )

    if _is_simple_texture_backend(resolved_backend):
        simple_endpoint = _strip_or_none(texture_endpoint) or _strip_or_none(
            config.simple_texture_endpoint
        )
        if simple_endpoint:
            resolved_backend = "service"
            resolved_endpoint = simple_endpoint
            resolved_engine = (
                _strip_or_none(backend_engine)
                or _strip_or_none(config.simple_backend_engine)
                or "simple_image_gen"
            )
            resolved_workers = config.simple_texture_workers or config.texture_workers
            resolved_job_timeout_sec = (
                config.simple_texture_job_timeout_sec or config.texture_job_timeout_sec
            )
            if uv_scope is None:
                resolved_uv_scope = config.simple_uv_scope
            if uv_rebake_source_albedo is None:
                resolved_uv_rebake_source_albedo = config.simple_uv_rebake_source_albedo
            if uv_rebake_size is None:
                resolved_uv_rebake_size = config.simple_uv_rebake_size

    return {
        "backend": resolved_backend,
        "endpoint": resolved_endpoint,
        "engine": resolved_engine,
        "workers": resolved_workers,
        "job_timeout_sec": resolved_job_timeout_sec,
        "uv_scope": resolved_uv_scope,
        "uv_rebake_source_albedo": resolved_uv_rebake_source_albedo,
        "uv_rebake_size": resolved_uv_rebake_size,
    }


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
        raise HTTPException(status_code=404, detail=f"S3 object not found: {s3_uri}")
    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Access denied to S3 object: {s3_uri}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download from S3: {e}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"S3 file too large: {size_mb:.1f}MB. Max: {config.max_upload_size_mb}MB",
        )

    return local_path


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


async def _load_texture_plan(
    manager: SessionManager,
    session_id: str,
) -> TexturePlan | None:
    """Load the validated plan artifact locally or from the shared store."""
    if manager.uses_shared_store():
        try:
            await asyncio.to_thread(
                manager.sync_from_store,
                session_id,
                "cache/texture_plan.json",
            )
        except FileNotFoundError:
            return None
    plan_path = manager.get_session_dir(session_id) / "cache" / "texture_plan.json"
    if not plan_path.is_file():
        return None
    try:
        return validate_texture_plan_payload(plan_path.read_bytes())
    except (OSError, ValidationError, ValueError):
        logger.exception("Invalid texture plan artifact for %s", session_id[:8])
        return None


def _texture_plan_status(
    session_id: str,
    plan: TexturePlan | None,
) -> TexturePlanStatus | None:
    if plan is None:
        return None
    return TexturePlanStatus(
        schema_version=plan.schema_version,
        decision_state=plan.decision.state,
        execution_allowed=plan.decision.execution_allowed,
        counts=plan.counts.model_dump(),
        limits=plan.limits.model_dump(),
        plan_url=f"/pipeline/{session_id}/plan",
    )


def _input_usd_path_from_metadata(
    session_dir: Path,
    metadata: dict[str, Any] | None,
) -> Path | None:
    """Infer the canonical input path for a shared session without hydrating it."""
    config_payload = (metadata or {}).get("config") or {}
    if not isinstance(config_payload, dict):
        return None
    extension = config_payload.get("input_extension")
    if not isinstance(extension, str) or extension not in _VALID_USD_EXTENSIONS:
        return None
    return session_dir / "input" / f"scene{extension}"


def _material_textures_validation_detail(
    error: ValidationError,
    root_loc: list[str],
) -> list[dict[str, Any]]:
    """Translate pydantic locations to the API field that carried the JSON."""
    detail: list[dict[str, Any]] = []
    for item in error.errors():
        translated = dict(item)
        raw_loc = translated.get("loc", ())
        loc = list(raw_loc) if isinstance(raw_loc, list | tuple) else [raw_loc]
        if loc and loc[0] == "root":
            loc = loc[1:]
        translated["loc"] = [*root_loc, *loc]
        ctx = translated.get("ctx")
        if isinstance(ctx, dict):
            translated["ctx"] = {key: str(value) for key, value in ctx.items()}
        detail.append(translated)
    return detail


def _validate_material_textures(
    decoded: Any,
    root_loc: list[str],
) -> dict[str, Any]:
    """Validate material override payloads before accepting a pipeline job."""
    try:
        return MaterialTextures(root=decoded).as_config()
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=_material_textures_validation_detail(e, root_loc),
        )


def _decode_json_form_field(
    raw: str,
    *,
    field_name: str,
    expected_type: type,
) -> Any:
    """Decode a JSON form field with FastAPI-style 422 errors."""
    if not raw or not raw.strip():
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "json_invalid",
                    "loc": ["form", field_name],
                    "msg": f"JSON decode error: {e}",
                }
            ],
        )
    if not isinstance(decoded, expected_type):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": f"{expected_type.__name__}_type",
                    "loc": ["form", field_name],
                    "msg": f"Input should be a JSON {expected_type.__name__}",
                }
            ],
        )
    return decoded


def _normalize_uri_list(decoded: Any, *, field_name: str) -> list[str] | None:
    """Validate and trim a decoded JSON URI list form field."""
    if decoded is None:
        return None
    normalized: list[str] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, str):
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "string_type",
                        "loc": ["form", field_name, index],
                        "msg": "Input should be a string",
                    }
                ],
            )
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized or None


def _require_projection_endpoint(
    *,
    texture_backend: str | None,
    texture_endpoint: str | None,
) -> None:
    """Reject service-backend requests that cannot reach a backend endpoint."""
    route = _resolve_texture_route(
        texture_backend=texture_backend,
        texture_endpoint=texture_endpoint,
        backend_engine=None,
        uv_scope=None,
        uv_rebake_source_albedo=None,
        uv_rebake_size=None,
    )
    if route["backend"] == "service" and not route["endpoint"]:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "missing",
                    "loc": ["form", "texture_endpoint"],
                    "msg": (
                        "texture_endpoint is required when the texture backend "
                        "is 'service'"
                    ),
                }
            ],
        )


async def _reserve_worker_slot(manager: SessionManager, session_id: str) -> Any:
    """Reserve the cross-process worker lock before acknowledging a job."""
    try:
        worker_lock = await asyncio.to_thread(
            manager.acquire_worker_lock, session_id, 0
        )
    except Timeout:
        raise HTTPException(
            status_code=409,
            detail=(
                "Session is still draining worker writes. Wait for the worker "
                "to stop before starting it."
            ),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    if await asyncio.to_thread(manager.is_worker_stalled, session_id):
        await asyncio.to_thread(manager.release_worker_lock, worker_lock, session_id)
        raise HTTPException(
            status_code=409,
            detail=(
                "Session is still draining worker writes. Wait for the worker "
                "to stop before starting it."
            ),
        )

    return worker_lock


def _release_worker_slot_callback(
    manager: SessionManager,
    session_id: str,
    worker_lock: Any,
) -> Callable[[], None]:
    """Build a typed registry callback that releases an accepted-job lock."""

    def _release() -> None:
        manager.release_worker_lock(worker_lock, session_id)

    return _release


def _active_snapshot_status(session_id: str) -> PipelineStatus | None:
    snapshot = get_event_bus().get_snapshot(session_id)
    if (
        not snapshot
        or snapshot.get("status") not in {"pending", "running", "cancelling"}
        or not get_job_registry().is_running(session_id)
    ):
        return None

    preview_images = snapshot.get("preview_images", [])
    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    created_at_text = snapshot.get("created_at")
    elapsed_seconds = int(snapshot.get("elapsed_seconds", 0) or 0)
    if isinstance(created_at_text, str):
        try:
            created_at = datetime.fromisoformat(created_at_text)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            elapsed_seconds = int((datetime.now(UTC) - created_at).total_seconds())
        except ValueError:
            pass

    storage_root = config.session_storage_path
    completed_steps = sanitize_payload(
        snapshot.get("completed_steps", []), storage_root
    )
    if not isinstance(completed_steps, list):
        completed_steps = []

    return PipelineStatus(
        session_id=session_id,
        status=snapshot["status"],
        current_step=snapshot.get("current_step"),
        completed_steps=completed_steps,
        overall_progress=snapshot.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=snapshot.get("status") in ("pending", "running"),
        elapsed_seconds=elapsed_seconds,
        created_at=snapshot.get("created_at", ""),
        updated_at=snapshot.get("updated_at", snapshot.get("created_at", "")),
        error=sanitize_message(snapshot.get("error"), storage_root),
        failed_step=snapshot.get("failed_step"),
        failed_step_stats=sanitize_step_stats(
            snapshot.get("failed_step_stats"), storage_root
        ),
    )


async def _save_reference_image_upload(
    upload: UploadFile | None,
    session_dir: Path,
) -> str | None:
    """Persist an optional reference image upload and return a file URI."""
    if upload is None:
        return None
    if upload.filename:
        ext = Path(upload.filename).suffix.lower()
        if ext not in _VALID_REFERENCE_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid reference image file type: {ext}. "
                    f"Allowed: {', '.join(sorted(_VALID_REFERENCE_IMAGE_EXTENSIONS))}"
                ),
            )
    else:
        ext = ".png"

    dest = session_dir / "input" / "reference_images" / f"reference_image{ext}"
    max_bytes = config.max_upload_size_mb * 1024 * 1024
    await _stream_copy(upload, dest, max_bytes=max_bytes)
    return dest.resolve().as_uri()


def _heartbeat_worker_slot_callback(
    manager: SessionManager,
    session_id: str,
    worker_lock: Any,
) -> Callable[[], Any]:
    """Build a callback that keeps an accepted queued job reservation fresh."""
    owner_token = getattr(worker_lock, "_wu_shared_reservation_token", None)

    def _heartbeat() -> Any:
        if owner_token is None:
            return None
        return asyncio.to_thread(
            manager.heartbeat_worker,
            session_id,
            owner_token=owner_token,
        )

    return _heartbeat


def _cancel_never_started_callback(
    manager: SessionManager,
    session_id: str,
) -> Callable[[], None]:
    """Mark a queued job cancelled if it never reaches the executor body."""

    def _cancel() -> None:
        try:
            manager.update_session(
                session_id,
                {
                    "status": "cancelled",
                    "can_cancel": False,
                },
            )
        except FileNotFoundError:
            return
        except Exception:  # pragma: no cover - pre-start cancellation write failure
            logger.exception(  # pragma: no cover
                "Failed to persist pre-start cancellation for %s", session_id[:8]
            )
            return  # pragma: no cover

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        emit_task = loop.create_task(
            get_event_bus().emit(
                ProgressEvent(
                    session_id=session_id,
                    step="pipeline",
                    state=StepState.CANCELLED,
                    message="Pipeline cancelled before startup",
                )
            )
        )

        def _log_emit_failure(task: asyncio.Task) -> None:
            try:
                task.result()
            except Exception:
                logger.exception(
                    "Failed to emit pre-start cancellation for %s", session_id[:8]
                )

        emit_task.add_done_callback(_log_emit_failure)

    return _cancel


_RUN_SCOPED_METADATA_FIELDS = (
    "error",
    "failed_step",
    "failed_step_stats",
    "failed_at",
    "partial_results",
)


def _reset_session_for_new_run(
    manager: SessionManager,
    session_id: str,
    *,
    fresh: bool,
) -> dict[str, Any]:
    """Reset run-scoped state on an existing session before a new run.

    Must be called with the cross-process worker lock held so a peer
    cancel cannot drop a fresh `.cancel` between the clear and the
    coroutine starting. Resets the four state surfaces that can leak
    from a prior run into a new one:

    - the durable `.cancel` marker (executor's between-step checkpoint),
    - the EventBus in-memory snapshot read by `/status`,
    - the EventBus per-session SSE queue read by `/events`,
    - run-scoped session metadata fields surfaced by `/sessions/{id}`.

    `fresh=True` (existing-session reuse via `POST /pipeline`) also
    clears `completed_steps`, `overall_progress`, `preview_images`, and
    `current_step` because the new run starts from scratch. `fresh=False`
    (regenerate) keeps those because a regenerate is incremental on top
    of the already-completed steps.

    Returns a snapshot of the prior values for every metadata field the
    reset overwrote. Pass it to `_restore_session_after_reset_failure`
    in an `except` block so a subsequent failure (e.g., `register()` or
    a config-write race) does not leave the session permanently in
    `pending` with all prior diagnostics wiped.
    """
    metadata = manager.get_session_metadata(session_id) or {}
    snapshot_keys: tuple[str, ...] = (
        "status",
        "current_step",
        "can_cancel",
    ) + _RUN_SCOPED_METADATA_FIELDS
    if fresh:
        snapshot_keys = snapshot_keys + (
            "completed_steps",
            "preview_images",
            "overall_progress",
        )
    snapshot = {key: metadata.get(key) for key in snapshot_keys}

    manager.clear_cancellation(session_id)
    get_event_bus().clear_session_state(session_id)

    metadata_reset: dict[str, Any] = {
        "status": "pending",
        "current_step": None,
        "can_cancel": True,
    }
    for field in _RUN_SCOPED_METADATA_FIELDS:
        metadata_reset[field] = None
    if fresh:
        metadata_reset["completed_steps"] = []
        metadata_reset["preview_images"] = []
        metadata_reset["overall_progress"] = {
            "current_step": 0,
            "total_steps": 9,
            "percent": 0,
            "estimated_remaining_seconds": None,
        }
    manager.update_session(session_id, metadata_reset)
    return snapshot


def _restore_session_after_reset_failure(
    manager: SessionManager,
    session_id: str,
    snapshot: dict[str, Any],
) -> None:
    """Re-apply the prior metadata snapshot after a post-reset failure.

    Called from `except` blocks when a step after `_reset_session_for_new_run`
    raises (validation, config-write race, register failure, etc.). Without
    this, the session would be permanently stuck in `pending` with prior
    diagnostics wiped and no executor coroutine ever scheduled.

    The bus snapshot and `.cancel` marker are deliberately not restored:
    the bus snapshot rebuilds lazily from disk on next read, and the
    `.cancel` marker reflected a pre-existing cancellation that the
    caller already chose to abandon by accepting the retry.
    """
    if not snapshot:
        return
    try:
        manager.update_session(session_id, snapshot)
    except Exception:
        logger.exception(
            "Failed to restore session metadata for %s after reset rollback",
            session_id,
        )


def _uses_per_prim_overrides(material_textures: dict[str, Any] | None) -> bool:
    """Return whether material overrides request per-prim texture units."""
    if not material_textures:
        return False
    return any(
        isinstance(override, dict) and bool(override.get("per_prim"))
        for override in material_textures.values()
    )


def _sync_texture_mode_for_overrides(
    pipeline_config: dict[str, Any],
    material_textures: dict[str, Any] | None,
) -> None:
    """Promote configs with per-prim overrides without downgrading stored mode."""
    if _uses_per_prim_overrides(material_textures):
        pipeline_config.setdefault("texture", {})["mode"] = "per_prim"


def _preserve_legacy_service_auto_prompting(pipeline_config: dict[str, Any]) -> None:
    """Backfill auto-prompt defaults for regenerated service configs."""
    auto_prompt = pipeline_config.get("auto_prompt")
    if not isinstance(auto_prompt, dict):
        return
    if "enabled" not in auto_prompt:
        auto_prompt["enabled"] = True
    if (
        auto_prompt.get("enabled", True)
        and "max_generated_materials" not in auto_prompt
    ):
        auto_prompt["max_generated_materials"] = (
            config.auto_prompt_max_generated_materials
        )


def _read_apply_cache_key_mode(marker_path: Path) -> str | None:
    """Read and validate the durable cached-apply key-mode marker."""
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise ValueError(f"Invalid apply cache key-mode marker: {marker_path}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid apply cache key-mode marker: {marker_path}")
    if payload.get("schema_version") != _APPLY_CACHE_KEY_MODE_MARKER_SCHEMA:
        raise ValueError(f"Invalid apply cache key-mode marker: {marker_path}")
    mode = payload.get("key_mode")
    if mode not in {
        _APPLY_CACHE_KEY_MODE_LEGACY,
        _APPLY_CACHE_KEY_MODE_PLAN,
    }:
        raise ValueError(f"Invalid apply cache key mode {mode!r}: {marker_path}")
    return str(mode)


async def _has_complete_plan_unit_texture_cache(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    plan_path: Path,
) -> bool:
    """Return whether every selected plan unit has one valid flat PBR triplet."""
    plan_payload = await asyncio.to_thread(plan_path.read_bytes)
    plan = validate_texture_plan_payload(plan_payload)
    selected_unit_ids = [unit.unit_id for unit in plan.selected_units]
    if not selected_unit_ids:
        return False
    expected_keys = {
        f"cache/textures/{unit_id}_{channel}.png"
        for unit_id in selected_unit_ids
        for channel in ("albedo", "normal", "orm")
    }

    def _all_local_textures_are_valid() -> bool:
        for key in sorted(expected_keys):
            if not is_valid_cached_texture_png(session_dir / key):
                return False
        return True

    if manager.uses_shared_store():
        # regenerate_pipeline hydrates cache/ before this check. Durable-key
        # membership prevents stale pod-local files from authorizing promotion;
        # validating the hydrated local snapshot then checks the exact bytes
        # that the executor will consume without re-downloading every texture.
        def _durable_keys_cover_hydrated_snapshot() -> bool:
            try:
                durable_keys = set(
                    manager.list_store_keys(session_id, "cache/textures/")
                )
                if not expected_keys.issubset(durable_keys):
                    return False
            except Exception:
                logger.warning(
                    "Could not list durable plan texture cache for %s",
                    session_id[:8],
                    exc_info=True,
                )
                return False
            return True

        if not await asyncio.to_thread(_durable_keys_cover_hydrated_snapshot):
            return False

    return await asyncio.to_thread(_all_local_textures_are_valid)


async def _persist_apply_cache_key_mode(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    key_mode: str,
) -> None:
    """Atomically persist and upload the cached blended-map key mode.

    The marker is uploaded before the regeneration job is registered. This
    makes the decision survive successful runs, failed runs, and worker
    replacement alike. Once promoted to plan-unit IDs, later cache loss must
    fail closed instead of falling back to stale legacy maps.
    """
    if key_mode not in {
        _APPLY_CACHE_KEY_MODE_LEGACY,
        _APPLY_CACHE_KEY_MODE_PLAN,
    }:
        raise ValueError(f"Invalid apply cache key mode: {key_mode!r}")
    marker_path = session_dir / _APPLY_CACHE_KEY_MODE_MARKER_KEY
    marker_payload = (
        json.dumps(
            {
                "schema_version": _APPLY_CACHE_KEY_MODE_MARKER_SCHEMA,
                "key_mode": key_mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    def _write_local_marker() -> None:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(marker_payload, encoding="utf-8")
            os.replace(temp_path, marker_path)
        finally:
            temp_path.unlink(missing_ok=True)

    await asyncio.to_thread(_write_local_marker)
    try:
        await asyncio.to_thread(
            manager.sync_to_store,
            session_id,
            _APPLY_CACHE_KEY_MODE_MARKER_KEY,
        )
    except Exception:
        # A local-only decision must never become authoritative after its
        # durable upload failed. The next request will hydrate whatever marker
        # the shared store actually contains (or remain markerless).
        await asyncio.to_thread(marker_path.unlink, missing_ok=True)
        raise


async def _promote_apply_cache_key_mode_after_artifact_sync(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    plan_path: Path,
) -> None:
    """Commit plan-unit apply mode after a complete cache is durable.

    Transitional pre-plan regeneration initially keeps its legacy marker so a
    failed or cancelled run cannot claim plan-keyed artifacts. Once full-plan
    generation, blending, and the executor's artifact sync all succeed, this
    finalizer proves that every selected plan unit has a valid durable PBR
    triplet before monotonically promoting the marker. Application may happen
    in the same request or later; the marker describes the blended cache.
    """
    if not await asyncio.to_thread(
        plan_path.is_file
    ) or not await _has_complete_plan_unit_texture_cache(
        manager,
        session_id,
        session_dir,
        plan_path,
    ):
        raise RuntimeError(
            "Successful plan-keyed regeneration did not produce a complete "
            "durable plan-unit texture cache; refusing to promote cached apply "
            "metadata"
        )
    await _persist_apply_cache_key_mode(
        manager,
        session_id,
        session_dir,
        _APPLY_CACHE_KEY_MODE_PLAN,
    )


def build_default_pipeline_config(
    session_id: str,
    usd_path: str,
    working_dir: str,
    material_textures: dict[str, Any] | None = None,
    user_prompt: str | None = None,
    auto_prompt_enabled: bool = True,
    texture_backend: str | None = None,
    texture_endpoint: str | None = None,
    backend_engine: str | None = None,
    backend_custom_parameters: dict[str, Any] | None = None,
    detail_policy: str | None = None,
    reference_image_uris: list[str] | None = None,
    turntable_video_uri: str | None = None,
    multiview_image_uris: list[str] | None = None,
    seed: int | None = None,
    strength: float | None = None,
    strict_scope: bool | None = None,
    uv_policy: str | None = None,
    uv_scope: str | None = None,
    uv_backend: str | None = None,
    uv_projection: str | None = None,
    uv_overwrite_existing: bool | None = None,
    uv_rebake_source_albedo: bool | None = None,
    uv_rebake_size: int | None = None,
    uv_normalize_out_of_range: bool | None = None,
    render_timeout_sec: int | None = None,
    planning_discovery_mode: str = "effective_bound",
    planning_unit_mode: str | None = None,
    explicit_material_paths: list[str] | None = None,
    explicit_prim_paths: list[str] | None = None,
    operator_override_cap: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Build a default pipeline config dict from ServiceConfig defaults.

    Args:
        session_id: Session identifier
        usd_path: Path to input USD file
        working_dir: Working directory for pipeline output
        material_textures: Per-material prompt/opacity overrides
        user_prompt: Aesthetic direction for auto-prompt generation
        auto_prompt_enabled: Whether to generate prompts for discovered
            materials missing from material_textures. Defaults to True to
            preserve legacy service behavior.

    Returns:
        Pipeline config dict compatible with config_to_context()
    """
    image_gen_config: dict[str, Any] = {
        "backend": config.image_gen_backend,
    }
    if config.image_gen_model:
        image_gen_config["model"] = config.image_gen_model
    if config.image_gen_base_url:
        image_gen_config["base_url"] = config.image_gen_base_url
    if config.image_gen_api_key_env:
        image_gen_config["api_key_env"] = config.image_gen_api_key_env
    elif config.image_gen_api_key:
        image_gen_config["api_key"] = config.image_gen_api_key

    texture_route = _resolve_texture_route(
        texture_backend=texture_backend,
        texture_endpoint=texture_endpoint,
        backend_engine=backend_engine,
        uv_scope=uv_scope,
        uv_rebake_source_albedo=uv_rebake_source_albedo,
        uv_rebake_size=uv_rebake_size,
    )
    resolved_uv_policy = uv_policy or config.uv_policy
    route_default_cap = backend_default_texture_cap(
        {
            **texture_route,
            "uv_policy": resolved_uv_policy,
        }
    )
    backend_default_cap = (
        config.texture_plan_uv_aware_default_cap
        if route_default_cap == TEXTURE_UV_AWARE_DEFAULT_CAP
        else config.texture_plan_default_cap
    )

    texture_section: dict[str, Any] = {
        "backend": texture_route["backend"],
        "detail_policy": detail_policy or "default",
        "image_gen": image_gen_config,
        "size": config.texture_size,
        "max_texture_units": config.max_texture_units,
        "workers": texture_route["workers"],
        "job_timeout_sec": texture_route["job_timeout_sec"],
        "failure_threshold": 0.0,
        "uv_policy": resolved_uv_policy,
        "uv_scope": texture_route["uv_scope"],
        "uv_backend": uv_backend or config.uv_backend,
        "uv_projection": uv_projection or config.uv_projection,
        "uv_mode": uv_projection or config.uv_projection,
        "uv_overwrite_existing": (
            config.uv_overwrite_existing
            if uv_overwrite_existing is None
            else uv_overwrite_existing
        ),
        "uv_rebake_source_albedo": texture_route["uv_rebake_source_albedo"],
        "uv_normalize_out_of_range": (
            config.uv_normalize_out_of_range
            if uv_normalize_out_of_range is None
            else uv_normalize_out_of_range
        ),
    }
    resolved_uv_rebake_size = texture_route["uv_rebake_size"]
    if resolved_uv_rebake_size is not None:
        texture_section["uv_rebake_size"] = resolved_uv_rebake_size
    resolved_texture_endpoint = texture_route["endpoint"]
    resolved_backend_engine = texture_route["engine"]
    if resolved_texture_endpoint:
        texture_section["endpoint"] = resolved_texture_endpoint
    if resolved_backend_engine:
        texture_section["engine"] = resolved_backend_engine
    if backend_custom_parameters:
        texture_section["custom_parameters"] = backend_custom_parameters
    if reference_image_uris:
        texture_section["reference_image_uris"] = reference_image_uris
    if turntable_video_uri:
        texture_section["turntable_video_uri"] = turntable_video_uri
    if multiview_image_uris:
        texture_section["multiview_image_uris"] = multiview_image_uris
    if seed is not None:
        texture_section["seed"] = seed
    if strength is not None:
        texture_section["strength"] = strength
    if strict_scope is not None:
        texture_section["strict_scope"] = strict_scope

    pipeline_config: dict[str, Any] = {
        "project": {
            "name": session_id,
            "session_id": session_id,
            "working_dir": working_dir,
        },
        "input": {
            "usd_path": usd_path,
        },
        "planning": {
            "source_asset": (f"session://{session_id}/input/{Path(usd_path).name}"),
            "discovery_mode": planning_discovery_mode,
            "unit_mode": planning_unit_mode
            or texture_section.get("mode", "per_material"),
            "explicit_material_paths": explicit_material_paths or [],
            "explicit_prim_paths": explicit_prim_paths or [],
            "backend_default_cap": backend_default_cap,
            "operator_override_cap": operator_override_cap,
            "plan_only": plan_only,
        },
        "texture": texture_section,
        "material_textures": material_textures or {},
        "auto_prompt": {
            "enabled": auto_prompt_enabled,
            "user_prompt": user_prompt or "",
            "default_opacity": config.blend_opacity,
            "max_generated_materials": config.auto_prompt_max_generated_materials,
            "llm": {
                "backend": config.llm_backend,
                "model": config.llm_model,
                **({"base_url": config.llm_base_url} if config.llm_base_url else {}),
                **(
                    {"api_key_env": config.llm_api_key_env}
                    if config.llm_api_key_env
                    else {}
                ),
                **(
                    {"api_key": config.llm_api_key}
                    if config.llm_api_key and not config.llm_api_key_env
                    else {}
                ),
            },
        },
        "variations": {"count": 1},
        "steps": {
            "prepare_uvs": {"enabled": True},
            "discover_materials": {"enabled": True},
            "generate_prompts": {"enabled": True},
            "render_previews": {
                "enabled": config.render_previews_enabled,
                "backend": DEFAULT_TEXTURE_RENDERING_BACKEND,
                "image_width": config.render_preview_image_width,
                "image_height": config.render_preview_image_height,
            },
            "generate_textures": {
                "enabled": True,
                "skip_existing": True,
                "max_workers": texture_route["workers"],
            },
            "blend_textures": {
                "enabled": True,
                "default_opacity": config.blend_opacity,
                "output_size": config.texture_size,
            },
            "apply_textures": {"enabled": True},
            "render": {
                "enabled": config.render_enabled,
                "backend": DEFAULT_TEXTURE_RENDERING_BACKEND,
                "image_width": config.render_image_width,
                "image_height": config.render_image_height,
            },
        },
    }
    resolved_render_timeout_sec = (
        config.render_timeout_sec if render_timeout_sec is None else render_timeout_sec
    )
    if resolved_render_timeout_sec is not None:
        pipeline_config["steps"]["render"]["timeout_sec"] = resolved_render_timeout_sec
    _sync_texture_mode_for_overrides(pipeline_config, material_textures)
    if planning_unit_mode is None:
        pipeline_config["planning"]["unit_mode"] = pipeline_config["texture"].get(
            "mode", "per_material"
        )
    return pipeline_config


@router.post(
    "/upload-usd",
    response_model=SessionCreated,
    status_code=201,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def upload_usd_immediate(
    usd_file: UploadFile | None = File(
        None,
        description="USD file to upload. Provide exactly one of usd_file or s3_uri.",
    ),
    s3_uri: str = Form(
        None,
        description=(
            "S3 URI to a USD file. Its exact bucket name must be listed in "
            "TA_S3_ALLOWED_BUCKETS; an empty allowlist rejects all client S3 inputs. "
            "Provide exactly one of usd_file or s3_uri."
        ),
    ),
) -> SessionCreated:
    """Upload a USD file and create a session for later pipeline execution.

    Provide exactly one of these two input modes:
    1. **File upload**: Provide ``usd_file`` (multipart).
    2. **S3 reference**: Provide ``s3_uri`` -- the service downloads server-side.

    Client S3 access is fail-closed: the exact bucket name must be listed in
    ``TA_S3_ALLOWED_BUCKETS``, and an empty allowlist rejects every client S3 URI
    before session-manager or S3 I/O.

    Use the returned session_id with ``POST /pipeline`` to start processing.
    """
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
        session_dir = await asyncio.to_thread(manager.create_session, session_id)
        try:
            local_path = await asyncio.to_thread(
                _download_s3_to_session,
                s3_uri,
                session_dir,
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
            s3_basename = s3_uri.rstrip("/").rsplit("/", 1)[-1] or local_path.name
            await asyncio.to_thread(
                manager.update_session,
                session_id,
                {
                    "status": "ready",
                    "config": {
                        "has_usd_upload": True,
                        "s3_uri": s3_uri,
                        "original_filename": s3_basename,
                        "input_extension": local_path.suffix.lower(),
                    },
                },
            )
            await asyncio.to_thread(manager.sync_to_store, session_id, "input/")
            return SessionCreated(
                session_id=session_id,
                status="ready",
                message=f"USD downloaded from S3 successfully ({size_mb:.1f}MB)",
                estimated_duration_minutes=0,
            )
        except HTTPException:
            await asyncio.to_thread(manager.delete_session, session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            await asyncio.to_thread(manager.delete_session, session_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download USD from S3: {e}",
            )

    # File upload path
    if usd_file and usd_file.filename:
        ext = Path(usd_file.filename).suffix.lower()
        if ext not in _VALID_USD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid USD file type: {ext}. "
                f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
            )

    session_dir = await asyncio.to_thread(manager.create_session, session_id)

    original_ext = (
        Path(usd_file.filename).suffix.lower()
        if usd_file and usd_file.filename
        else ".usd"
    )
    usd_path = session_dir / "input" / f"scene{original_ext}"

    max_bytes = config.max_upload_size_mb * 1024 * 1024
    try:
        total_bytes = await _stream_copy(usd_file, usd_path, max_bytes=max_bytes)
        size_mb = total_bytes / (1024 * 1024)

        logger.info(
            f"USD uploaded for session {session_id[:8]}: "
            f"{size_mb:.2f}MB ({original_ext})"
        )

        await asyncio.to_thread(
            manager.update_session,
            session_id,
            {
                "status": "ready",
                "config": {
                    "has_usd_upload": True,
                    "original_filename": usd_file.filename if usd_file else None,
                    "input_extension": original_ext,
                },
            },
        )
        await asyncio.to_thread(manager.sync_to_store, session_id, "input/")

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded successfully",
            estimated_duration_minutes=0,
        )

    except HTTPException:
        await asyncio.to_thread(manager.delete_session, session_id)
        raise
    except Exception as e:
        logger.error(f"Failed to upload USD: {e}")
        await asyncio.to_thread(manager.delete_session, session_id)
        raise HTTPException(status_code=500, detail=f"Failed to upload USD: {e}")


@router.post(
    "",
    response_model=SessionCreated,
    status_code=202,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def create_pipeline(
    usd_file: UploadFile | None = File(
        None,
        description=(
            "Lowest-priority USD source. Used only when session_id and s3_uri "
            "are both omitted."
        ),
    ),
    reference_image_file: UploadFile | None = File(
        None,
        description=(
            "Optional global reference image upload for projection backend "
            "conditioning. Added to reference_image_uris."
        ),
    ),
    session_id: str = Form(
        None,
        description=(
            "Highest-priority source: an existing session ID from /upload-usd. "
            "When provided, s3_uri and usd_file are ignored."
        ),
    ),
    s3_uri: str = Form(
        None,
        description=(
            "Second-priority source, used when session_id is omitted. Its exact "
            "bucket name must be listed in TA_S3_ALLOWED_BUCKETS; an empty allowlist "
            "rejects all selected client S3 inputs before session-manager or S3 I/O. "
            "When selected, usd_file is ignored."
        ),
    ),
    material_textures_json: str = Form(
        default="",
        description=(
            "Per-material texture config JSON. Shape: "
            '{"Material": {"prompt": "rusted steel", "opacity": 0.85, '
            '"per_prim": {"/World/Prim": {"prompt": "scratches", "opacity": 0.65}}}}. '
            "Material prompt is required and non-empty, opacity is optional "
            "and bounded to 0.0-1.0, unknown fields are rejected, and any "
            "per_prim entry runs the request in per-prim texture mode."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description="Aesthetic direction for auto-prompt generation (e.g. 'old and weathered'). "
        "Used to auto-generate prompts for materials not covered by material_textures_json.",
    ),
    auto_prompt_enabled: bool | None = Form(
        default=None,
        description=(
            "Whether to auto-generate prompts for discovered materials missing "
            "from material_textures_json. Defaults to true for legacy service "
            "behavior; set false for strict material_textures_json scope."
        ),
    ),
    texture_backend: str | None = Form(
        default=None,
        description="Texture backend override, for example 'service'.",
    ),
    texture_endpoint: str | None = Form(
        default=None,
        description="Texture variation backend endpoint for texture_backend='service'.",
    ),
    backend_engine: str | None = Form(
        default=None,
        description="Projection backend engine/model route hint.",
    ),
    backend_custom_parameters_json: str = Form(
        default="",
        description="Backend custom parameters JSON object.",
    ),
    detail_policy: TextureDetailPolicy | None = Form(
        default=None,
        description=(
            "Texture detail policy. Use 'surface_only' for AOI/CAD assets "
            "where traces, labels, holes, seams, or other semantic details "
            "already exist as geometry and textures should stay limited to "
            "subtle material surface variation."
        ),
    ),
    reference_image_uris_json: str = Form(
        default="",
        description="Global reference image URI JSON list.",
    ),
    turntable_video_uri: str | None = Form(
        default=None,
        description="Global turntable video URI for projection backend conditioning.",
    ),
    multiview_image_uris_json: str = Form(
        default="",
        description="Global multi-view image URI JSON list.",
    ),
    seed: int | None = Form(
        default=None,
        description="Texture backend seed override.",
    ),
    strength: float | None = Form(
        default=None,
        ge=0.0,
        le=1.0,
        description="Texture edit strength override.",
    ),
    strict_scope: bool | None = Form(
        default=None,
        description="Whether projection backend requests must preserve selected scope.",
    ),
    uv_policy: str | None = Form(
        default=None,
        description=(
            "UV preparation policy override, for example 'force_projection' "
            "for UV-aware texture backends."
        ),
    ),
    uv_scope: str | None = Form(
        default=None,
        description="UV projection scope override, for example 'target_prims'.",
    ),
    uv_backend: str | None = Form(
        default=None,
        description="UV preparation backend override.",
    ),
    uv_projection: str | None = Form(
        default=None,
        description="UV projection mode override, for example 'box'.",
    ),
    uv_overwrite_existing: bool | None = Form(
        default=None,
        description="Whether UV preparation may overwrite existing UV coordinates.",
    ),
    uv_rebake_source_albedo: bool | None = Form(
        default=None,
        description=(
            "Whether scoped UV projection should rebake source albedo, normal, "
            "and ORM maps into the generated UV layout."
        ),
    ),
    uv_rebake_size: int | None = Form(
        default=None,
        gt=0,
        description="Optional scoped source texture rebake resolution.",
    ),
    uv_normalize_out_of_range: bool | None = Form(
        default=None,
        description="Whether UV preparation should normalize out-of-range UVs.",
    ),
    render_timeout_sec: int | None = Form(
        default=None,
        gt=0,
        description=(
            "Optional final render request timeout in seconds. Texture generation "
            "and USD packaging can still complete when the renderer is unavailable "
            "or slow."
        ),
    ),
    plan_only: bool = Form(
        default=False,
        description=(
            "Discover and persist texture_plan.json without invoking prompt, "
            "image-generation, application, or render backends."
        ),
    ),
    discovery_mode: TextureDiscoveryMode = Form(
        default=TextureDiscoveryMode.EFFECTIVE_BOUND,
        description="Planning scope: effective_bound, explicit, or all_authored.",
    ),
    unit_mode: TextureUnitMode | None = Form(
        default=None,
        description="Planning unit mode: per_material, per_group, or per_prim.",
    ),
    explicit_material_paths_json: str = Form(
        default="",
        description="JSON list of absolute material paths for explicit discovery.",
    ),
    explicit_prim_paths_json: str = Form(
        default="",
        description="JSON list of absolute prim/subset paths for explicit discovery.",
    ),
    operator_override_cap: int | None = Form(
        default=None,
        ge=1,
        description=(
            "Intentional generation-unit cap override. Values above the backend "
            "default and no greater than 64 are recorded in the plan."
        ),
    ),
) -> SessionCreated:
    """Create and execute a texture generation pipeline.

    Provide at least one of these three input modes:
    1. **Existing session**: Provide ``session_id`` (from ``/upload-usd``).
    2. **S3 reference**: Provide ``s3_uri``, downloads from S3 server-side.
    3. **File upload**: Provide ``usd_file``, creates new session.

    When multiple sources are supplied, selection follows the legacy precedence
    ``session_id`` > ``s3_uri`` > ``usd_file`` and lower-priority fields are
    ignored. A selected S3 source is admitted only when its exact bucket name is
    listed in ``TA_S3_ALLOWED_BUCKETS``; an empty allowlist rejects every client
    S3 input before session-manager or S3 I/O.

    Optionally provide ``material_textures_json`` to specify per-material
    texture prompts, blend opacity, and per-prim overrides.
    """
    # Parse material_textures from JSON. Use 422 with a structured detail
    # list matching FastAPI's request-validation format and validate the
    # decoded wire shape before accepting the request.
    material_textures: dict[str, Any] | None = None
    if material_textures_json and material_textures_json.strip():
        try:
            decoded = json.loads(material_textures_json)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "json_invalid",
                        "loc": ["form", "material_textures_json"],
                        "msg": f"JSON decode error: {e}",
                    }
                ],
            )
        if not isinstance(decoded, dict):
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "dict_type",
                        "loc": ["form", "material_textures_json"],
                        "msg": (
                            "Input should be a JSON object mapping material "
                            "names to override dicts"
                        ),
                    }
                ],
            )
        bad_keys = [k for k, v in decoded.items() if not isinstance(v, dict)]
        if bad_keys:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "dict_type",
                        "loc": ["form", "material_textures_json", k],
                        "msg": (
                            "Per-material override must be an object with "
                            "prompt/opacity fields"
                        ),
                    }
                    for k in bad_keys
                ],
            )
        material_textures = _validate_material_textures(
            decoded, ["form", "material_textures_json"]
        )

    backend_custom_parameters = _decode_json_form_field(
        backend_custom_parameters_json,
        field_name="backend_custom_parameters_json",
        expected_type=dict,
    )
    reference_image_uris = _normalize_uri_list(
        _decode_json_form_field(
            reference_image_uris_json,
            field_name="reference_image_uris_json",
            expected_type=list,
        ),
        field_name="reference_image_uris_json",
    )
    multiview_image_uris = _normalize_uri_list(
        _decode_json_form_field(
            multiview_image_uris_json,
            field_name="multiview_image_uris_json",
            expected_type=list,
        ),
        field_name="multiview_image_uris_json",
    )
    explicit_material_paths = _normalize_uri_list(
        _decode_json_form_field(
            explicit_material_paths_json,
            field_name="explicit_material_paths_json",
            expected_type=list,
        ),
        field_name="explicit_material_paths_json",
    )
    explicit_prim_paths = _normalize_uri_list(
        _decode_json_form_field(
            explicit_prim_paths_json,
            field_name="explicit_prim_paths_json",
            expected_type=list,
        ),
        field_name="explicit_prim_paths_json",
    )
    if (
        operator_override_cap is not None
        and operator_override_cap > config.texture_plan_hard_cap
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"operator_override_cap cannot exceed the hard maximum of "
                f"{config.texture_plan_hard_cap}. Consolidate materials or narrow the "
                "explicit material/prim scope."
            ),
        )
    if discovery_mode is TextureDiscoveryMode.EXPLICIT and not (
        explicit_material_paths or explicit_prim_paths
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Explicit discovery requires explicit_material_paths_json or "
                "explicit_prim_paths_json."
            ),
        )
    texture_endpoint_text = texture_endpoint.strip() if texture_endpoint else None
    backend_engine_text = backend_engine.strip() if backend_engine else None
    uv_policy_text = uv_policy.strip() if uv_policy else None
    uv_scope_text = uv_scope.strip() if uv_scope else None
    uv_backend_text = uv_backend.strip() if uv_backend else None
    uv_projection_text = uv_projection.strip() if uv_projection else None
    turntable_video_uri_text = (
        turntable_video_uri.strip() if turntable_video_uri else None
    )
    _require_projection_endpoint(
        texture_backend=texture_backend,
        texture_endpoint=texture_endpoint_text,
    )
    planning_route = _resolve_texture_route(
        texture_backend=texture_backend,
        texture_endpoint=texture_endpoint_text,
        backend_engine=backend_engine_text,
        uv_scope=uv_scope_text,
        uv_rebake_source_albedo=uv_rebake_source_albedo,
        uv_rebake_size=uv_rebake_size,
    )
    effective_default_cap = backend_default_texture_cap(
        {
            "backend": planning_route["backend"],
            "engine": planning_route["engine"],
            "uv_policy": uv_policy_text or config.uv_policy,
        }
    )
    if (
        operator_override_cap is not None
        and operator_override_cap <= effective_default_cap
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "operator_override_cap must be greater than the effective "
                f"backend default cap of {effective_default_cap}; omit the "
                "override when the default already covers the request."
            ),
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

    worker_lock: Any | None = None
    reused_existing_session = False
    session_created_here = False
    delete_created_session_on_failure = False
    existing_metadata: dict[str, Any] | None = None

    if session_id:
        # Path 1: reuse existing session
        if not await asyncio.to_thread(manager.session_exists, session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        reused_existing_session = True

        # Prevent concurrent re-start of a running session. Reserve the
        # cross-process lock before reading or mutating session files so DELETE
        # and peer POST requests cannot interleave with config/metadata writes.
        job_registry = get_job_registry()
        if job_registry.is_running(session_id):
            raise HTTPException(
                status_code=409,
                detail="Session is already running. Cancel it first or wait for completion.",
            )
        if manager.uses_shared_store() and await asyncio.to_thread(
            manager.is_worker_active,
            session_id,
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Session is already running on another instance. "
                    "Cancel it first or wait for completion."
                ),
            )
        worker_lock = await _reserve_worker_slot(manager, session_id)

        session_dir = manager.get_session_dir(session_id)
        existing_metadata = await asyncio.to_thread(
            manager.get_session_metadata,
            session_id,
        )

    elif selected_s3_uri:
        # Path 2: new session with S3 download
        session_id = str(uuid.uuid4())
        session_dir = await asyncio.to_thread(manager.create_session, session_id)
        session_created_here = True

        try:
            local_path = await asyncio.to_thread(
                _download_s3_to_session,
                selected_s3_uri,
                session_dir,
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            await asyncio.to_thread(manager.delete_session, session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            await asyncio.to_thread(manager.delete_session, session_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download USD from S3: {e}",
            )

    elif selected_usd_file:
        # Path 3: new session with USD upload
        session_id = str(uuid.uuid4())
        session_dir = await asyncio.to_thread(manager.create_session, session_id)
        session_created_here = True

        try:
            if selected_usd_file.filename:
                ext = Path(selected_usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid USD file type: {ext}. "
                        f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}",
                    )

            original_ext = (
                Path(selected_usd_file.filename).suffix.lower()
                if selected_usd_file.filename
                else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
            max_bytes = config.max_upload_size_mb * 1024 * 1024
            total_bytes = await _stream_copy(
                selected_usd_file,
                usd_path,
                max_bytes=max_bytes,
            )
            size_mb = total_bytes / (1024 * 1024)

            logger.info(
                f"USD uploaded for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({original_ext})"
            )

        except HTTPException:
            await asyncio.to_thread(manager.delete_session, session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to save USD file: {e}")
            await asyncio.to_thread(manager.delete_session, session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save USD file: {e}")

    else:
        raise HTTPException(
            status_code=400,
            detail="One of usd_file, session_id, or s3_uri must be provided",
        )

    reset_snapshot: dict[str, Any] = {}
    try:
        if worker_lock is None:
            worker_lock = await _reserve_worker_slot(manager, session_id)

        # Find the input USD
        input_usd_path = _find_input_usd(session_dir)
        if (
            not input_usd_path
            and reused_existing_session
            and manager.uses_shared_store()
        ):
            input_usd_path = _input_usd_path_from_metadata(
                session_dir,
                existing_metadata,
            )
        if not input_usd_path:
            raise HTTPException(
                status_code=400, detail="Input USD not found for session"
            )

        uploaded_reference_uri = await _save_reference_image_upload(
            reference_image_file,
            session_dir,
        )
        if uploaded_reference_uri:
            reference_image_uris = [
                *(reference_image_uris or []),
                uploaded_reference_uri,
            ]

        # Build pipeline config
        user_prompt_text = user_prompt.strip() if user_prompt else None
        pipeline_config = build_default_pipeline_config(
            session_id=session_id,
            usd_path=str(input_usd_path),
            working_dir=str(session_dir / "cache"),
            material_textures=material_textures,
            user_prompt=user_prompt_text,
            auto_prompt_enabled=(
                True if auto_prompt_enabled is None else auto_prompt_enabled
            ),
            texture_backend=texture_backend.strip() if texture_backend else None,
            texture_endpoint=texture_endpoint_text,
            backend_engine=backend_engine_text,
            backend_custom_parameters=backend_custom_parameters,
            detail_policy=detail_policy.value if detail_policy else None,
            reference_image_uris=reference_image_uris,
            turntable_video_uri=turntable_video_uri_text,
            multiview_image_uris=multiview_image_uris,
            seed=seed,
            strength=strength,
            strict_scope=strict_scope,
            uv_policy=uv_policy_text,
            uv_scope=uv_scope_text,
            uv_backend=uv_backend_text,
            uv_projection=uv_projection_text,
            uv_overwrite_existing=uv_overwrite_existing,
            uv_rebake_source_albedo=uv_rebake_source_albedo,
            uv_rebake_size=uv_rebake_size,
            uv_normalize_out_of_range=uv_normalize_out_of_range,
            render_timeout_sec=render_timeout_sec,
            planning_discovery_mode=discovery_mode.value,
            planning_unit_mode=unit_mode.value if unit_mode else None,
            explicit_material_paths=explicit_material_paths,
            explicit_prim_paths=explicit_prim_paths,
            operator_override_cap=operator_override_cap,
            plan_only=plan_only,
        )

        # Save resolved config for audit / regeneration
        config_path = session_dir / "input" / "config.yaml"
        try:
            await asyncio.to_thread(
                _write_pipeline_config,
                config_path,
                pipeline_config,
            )
        except InlineSecretError:
            delete_created_session_on_failure = session_created_here
            raise HTTPException(
                status_code=400,
                detail=_INVALID_PIPELINE_CONFIG_DETAIL,
            ) from None

        # Update session metadata. Only fields that are safe for the public
        # ``/sessions`` and ``/sessions/{id}`` responses are persisted here:
        # the absolute ``usd_path`` is intentionally omitted because it would
        # leak the container's internal storage layout.
        input_extension = input_usd_path.suffix.lower()
        existing_config = (existing_metadata or {}).get("config") or {}
        if not isinstance(existing_config, dict):
            existing_config = {}
        original_filename = (
            existing_config.get("original_filename")
            if reused_existing_session
            else (
                selected_usd_file.filename
                if selected_usd_file and selected_usd_file.filename
                else None
            )
        )
        has_usd_upload = (
            bool(existing_config.get("has_usd_upload"))
            if reused_existing_session
            else bool(selected_usd_file and selected_usd_file.filename)
        )
        s3_uri_value = (
            existing_config.get("s3_uri")
            if reused_existing_session
            else selected_s3_uri
        )
        await asyncio.to_thread(
            manager.update_session,
            session_id,
            {
                "config": {
                    "project_name": session_id,
                    "input_extension": input_extension,
                    "original_filename": original_filename,
                    "has_usd_upload": has_usd_upload,
                    "s3_uri": s3_uri_value,
                    "material_textures": material_textures,
                },
            },
        )
        await asyncio.to_thread(manager.sync_to_store, session_id, "input/")

        # Reset run-scoped state on reused sessions only after every step
        # that could fail above has succeeded. The executor reads `.cancel`
        # and metadata when it starts; `/status` reads the bus snapshot.
        # Resetting earlier (then failing in validation/config write) would
        # leave the session permanently `pending` with prior diagnostics
        # wiped — see `_restore_session_after_reset_failure` for the
        # post-register rollback path.
        if reused_existing_session:
            reset_snapshot = await asyncio.to_thread(
                _reset_session_for_new_run,
                manager,
                session_id,
                fresh=True,
            )
        await get_event_bus().seed_pending_session(session_id)

        # Register and start pipeline execution
        job_registry = get_job_registry()
        await job_registry.register(
            session_id,
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                acquire_worker_lock=False,
                only_steps=(
                    ["discover_materials", "plan_textures"] if plan_only else None
                ),
                worker_owner_token=getattr(
                    worker_lock,
                    "_wu_shared_reservation_token",
                    None,
                ),
            ),
            on_never_started=_cancel_never_started_callback(
                manager,
                session_id,
            ),
            on_finished=_release_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
            on_queued_heartbeat=_heartbeat_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
        )
    except Exception:
        if worker_lock is not None:
            await asyncio.to_thread(
                manager.release_worker_lock,
                worker_lock,
                session_id,
            )
        if reset_snapshot:
            await asyncio.to_thread(
                _restore_session_after_reset_failure,
                manager,
                session_id,
                reset_snapshot,
            )
        if delete_created_session_on_failure:
            await asyncio.to_thread(manager.delete_session, session_id)
        raise

    logger.info(f"Pipeline registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Pipeline queued for execution",
        estimated_duration_minutes=10,
        plan_url=f"/pipeline/{session_id}/plan",
    )


@router.get(
    "/{session_id}/plan",
    response_model=TexturePlan,
    responses={404: JSON_RESPONSE},
)
async def get_pipeline_plan(session_id: str) -> TexturePlan:
    """Return the validated immutable texture plan when planning is complete."""
    manager = get_session_manager()
    if not await asyncio.to_thread(manager.session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    plan = await _load_texture_plan(manager, session_id)
    if plan is None:
        raise HTTPException(
            status_code=404,
            detail="Texture plan is not available yet. Check the status endpoint.",
        )
    return plan


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Uses the same merged disk+bus view as ``/sessions/{sid}`` so the two
    endpoints agree on every observable field for the same session, even
    when the executor's outer exception handler persists a terminal disk
    status without emitting a corresponding bus event.
    """
    active_status = _active_snapshot_status(session_id)
    if active_status is not None:
        plan = await _load_texture_plan(get_session_manager(), session_id)
        active_status.texture_plan = _texture_plan_status(session_id, plan)
        return active_status

    # Imported here to keep the cross-router dependency local rather than
    # introducing it at module load time.
    from .sessions_router import _build_session_view

    view = await asyncio.to_thread(_build_session_view, session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_images = view.get("preview_images", [])
    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    # Sanitize at read time too -- session.json files written before the
    # write-time scrubbing fix landed may still hold raw NVCF URLs or
    # absolute session paths in the failure diagnostics.
    storage_root = config.session_storage_path
    completed_steps = sanitize_payload(view.get("completed_steps", []), storage_root)
    if not isinstance(completed_steps, list):
        completed_steps = []

    plan = await _load_texture_plan(get_session_manager(), session_id)
    return PipelineStatus(
        session_id=session_id,
        status=view["status"],
        current_step=view.get("current_step"),
        completed_steps=completed_steps,
        overall_progress=view.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=view["can_cancel"],
        elapsed_seconds=view["elapsed_seconds"],
        created_at=view["created_at"],
        updated_at=view["updated_at"],
        error=sanitize_message(view.get("error"), storage_root),
        failed_step=view.get("failed_step"),
        failed_step_stats=sanitize_step_stats(
            view.get("failed_step_stats"), storage_root
        ),
        texture_plan=_texture_plan_status(session_id, plan),
    )


@router.get(
    "/{session_id}/results",
    response_model=PipelineResults | PipelineError,
)
async def get_pipeline_results(session_id: str):
    """Get pipeline execution results (only available when completed).

    Reads from the same merged disk+bus view as ``/sessions/{sid}`` and
    ``/pipeline/{sid}/status``: when the bus has reached a terminal status
    but ``_persist_status`` hasn't yet awaited its disk write, a disk-only
    read here would briefly return 202 ("still running") while the other
    two endpoints already report "completed".
    """
    from .sessions_router import _build_session_view

    view = await asyncio.to_thread(_build_session_view, session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status = view["status"]
    storage_root = config.session_storage_path

    if status == "completed":
        # Sanitize ``results`` for legacy session.json files written before
        # the executor's write-time scrubber landed. A completed run with
        # partial failures (threshold not hit) carries ``errors`` records
        # whose ``message`` field can still hold an NVCF function-
        # invocation URL or absolute session path.
        sanitized_stats = sanitize_step_stats(view.get("results", {}), storage_root)
        download_urls = _artifact_download_urls(session_id)
        if isinstance(sanitized_stats, dict) and sanitized_stats.get(
            "manifest_available"
        ):
            sanitized_stats["manifest_url"] = download_urls["manifest"]
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=sanitized_stats or {},
            download_urls=download_urls,
            duration_seconds=view.get("duration_seconds", 0),
            completed_at=view.get("completed_at", ""),
        )

    elif status == "failed":
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=sanitize_message(
                view.get("error", "Unknown error"), storage_root
            ),
            failed_step=view.get("failed_step", "unknown"),
            completed_steps=[s["name"] for s in view.get("completed_steps", [])],
            partial_results=sanitize_step_stats(
                view.get("partial_results"), storage_root
            ),
            failed_step_stats=sanitize_step_stats(
                view.get("failed_step_stats"), storage_root
            ),
        )

    else:
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post("/{session_id}/cancel")
async def cancel_pipeline(session_id: str):
    """Cancel a running pipeline."""
    job_registry = get_job_registry()
    manager = get_session_manager()

    metadata = await asyncio.to_thread(manager.get_session_metadata, session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    if metadata["status"] not in ["pending", "running", "cancelling"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel pipeline with status: {metadata['status']}",
        )

    # request_cancellation drops the `.cancel` marker (so the worker's
    # between-step is_cancelled() checkpoint sees it) and persists "cancelling"
    # to disk. The CANCELLING bus event then mirrors that into the in-memory
    # snapshot used by /status and notifies SSE subscribers. Both writers are
    # idempotent against terminal state — if the worker finished naturally in
    # the window after our is_running() guard, neither will downgrade the
    # final status. In a multi-process deployment, this disk marker is the
    # only shared cancellation signal; JobRegistry only knows about local
    # asyncio tasks.
    await asyncio.to_thread(manager.request_cancellation, session_id)
    event_bus = get_event_bus()
    snapshot = event_bus.get_snapshot(session_id)
    current_step_info = (snapshot or {}).get("current_step") or {}
    current_step = current_step_info.get("name", "pipeline")
    await event_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step=current_step,
            state=StepState.CANCELLING,
            message="Pipeline cancellation requested",
        )
    )

    if job_registry.is_running(session_id):
        # job_registry.cancel internally fires task.cancel() and waits up to 5s
        # for the worker to finish (cooperative path or asyncio cancellation).
        cancelled = await job_registry.cancel(session_id)

        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Failed to cancel pipeline. It may have already completed.",
            )

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Pipeline cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str):
    """Stream real-time progress events via Server-Sent Events (SSE).

    Example client (JavaScript):
        const eventSource = new EventSource(`/pipeline/${sessionId}/events`);
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Step: ${data.step}, Progress: ${data.percent}%`);
        });
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    local_job_active = get_job_registry().is_running(session_id)
    if not await asyncio.to_thread(manager.session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")
    if snapshot is None:
        metadata = await asyncio.to_thread(manager.get_session_metadata, session_id)
        final_state = (metadata or {}).get("status", "unknown")
        has_local_session = (
            manager.get_session_dir(session_id) / METADATA_KEY
        ).is_file()
        if (
            manager.uses_shared_store()
            and final_state in {"running", "cancelling"}
            and not local_job_active
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline is running on a different instance; use polling instead"
                ),
            )
        if (
            final_state not in terminal_states
            and not has_local_session
            and not local_job_active
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pipeline is running on a different instance; use polling instead"
                ),
            )

    # Register the per-session queue here in the route handler rather than
    # lazily inside the generator. EventSourceResponse runs the generator
    # body only once SSE iteration starts, which opens a window between the
    # session_exists() check above and the first queue.get(). A DELETE
    # landing in that window would leave cleanup_session() with no queue to
    # notify, then the generator would call get_queue() and silently
    # setdefault() a fresh queue for an already-deleted session. Resolving
    # the queue eagerly closes that race -- cleanup_session() will see the
    # queue, push the terminal sentinel, and the generator will pick it up
    # immediately on its first iteration.
    queue = event_bus.get_queue(session_id)

    async def event_generator():  # pragma: no cover - SSE transport loop
        """Generate SSE events from the session's event queue."""
        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}',
            }
            return

        if snapshot is None:
            metadata = await asyncio.to_thread(manager.get_session_metadata, session_id)
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
                    elif (
                        event.state == "completed"
                        and event.extra
                        and event.extra.get("pipeline_completed")
                    ):
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": f'{{"session_id": "{session_id}", "final_state": "{event.state}"}}',
                        }
                        break

                except TimeoutError:
                    if not await asyncio.to_thread(manager.session_exists, session_id):
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "deleted"}}'
                            ),
                        }
                        break
                    metadata = await asyncio.to_thread(
                        manager.get_session_metadata,
                        session_id,
                    )
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


@router.post(
    "/{session_id}/regenerate",
    response_model=SessionCreated,
    status_code=202,
)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> SessionCreated:
    """Regenerate specific pipeline steps from cached data.

    Useful for re-running texture generation with different prompts/opacity
    without re-discovering materials.
    """
    manager = get_session_manager()
    worker_lock = await _reserve_worker_slot(manager, session_id)
    reset_snapshot: dict[str, Any] = {}

    try:
        metadata = await asyncio.to_thread(manager.get_session_metadata, session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")

        if metadata["status"] in ["pending", "running", "cancelling"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot regenerate while pipeline is {metadata['status']}",
            )

        # Load the original config from session
        session_dir = manager.get_session_dir(session_id)
        for prefix in ("input/", "cache/"):
            await asyncio.to_thread(manager.sync_from_store, session_id, prefix)
        config_path = session_dir / "input" / "config.yaml"

        if not config_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Original config not found for session",
            )

        with open(config_path) as f:
            pipeline_config = yaml.safe_load(f)
        _preserve_legacy_service_auto_prompting(pipeline_config)
        # This flag is valid only for the current server-authorized apply-only
        # request. Never trust or carry a copy from the stored input config.
        pipeline_config.setdefault("planning", {}).pop(
            "allow_non_executable_cached_apply_plan",
            None,
        )

        # Determine which steps to re-run
        only_steps = [s.value for s in request.steps]
        plan_path = session_dir / "cache" / "texture_plan.json"
        regenerates_complete_plan_cache = not request.texture_unit_ids and {
            TexturePipelineStep.GENERATE_TEXTURES,
            TexturePipelineStep.BLEND_TEXTURES,
        }.issubset(request.steps)
        apply_cache_marker_path = session_dir / _APPLY_CACHE_KEY_MODE_MARKER_KEY
        apply_cache_key_mode: str | None = None
        steps_cfg = pipeline_config.get("steps", {}) or {}
        rebuild_runtime_units = bool(
            {
                TexturePipelineStep.GENERATE_TEXTURES,
                TexturePipelineStep.APPLY_TEXTURES,
            }.intersection(request.steps)
        )
        if rebuild_runtime_units:
            # Runtime PrimTextureUnit objects are deliberately not serialized.
            # Rebuild them from the source stage and the durable prompt cache so
            # incremental generate/apply runs do not need an LLM call.
            prompt_cache_path = (
                session_dir / "cache" / "prompts" / "material_prompts.json"
            )
            if prompt_cache_path.is_file():
                cached_prompts = json.loads(
                    prompt_cache_path.read_text(encoding="utf-8")
                )
                if isinstance(cached_prompts, dict):
                    pipeline_config["material_textures"] = {
                        **cached_prompts,
                        **(pipeline_config.get("material_textures") or {}),
                    }

        stored_key_mode: str | None = None
        if TexturePipelineStep.APPLY_TEXTURES in request.steps or (
            TexturePipelineStep.GENERATE_TEXTURES in request.steps
            and (not plan_path.is_file() or regenerates_complete_plan_cache)
        ):
            try:
                stored_key_mode = await asyncio.to_thread(
                    _read_apply_cache_key_mode,
                    apply_cache_marker_path,
                )
            except ValueError as err:
                logger.warning(
                    "Cannot regenerate cached apply for %s: %s",
                    session_id[:8],
                    err,
                )
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Cached apply metadata is malformed or unsupported: "
                        f"{_APPLY_CACHE_KEY_MODE_MARKER_KEY}. Restore a valid "
                        "marker from durable session state before retrying."
                    ),
                ) from err

        if TexturePipelineStep.APPLY_TEXTURES in request.steps:
            if stored_key_mode == _APPLY_CACHE_KEY_MODE_PLAN:
                # Promotion is monotonic. If the plan artifact was lost, rebuild
                # it below and remain on deterministic plan-unit IDs; never
                # downgrade to stale display-key maps.
                if (
                    manager.uses_shared_store()
                    and plan_path.is_file()
                    and not await _has_complete_plan_unit_texture_cache(
                        manager,
                        session_id,
                        session_dir,
                        plan_path,
                    )
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Plan-mode cached apply requires the complete "
                            "plan-unit texture cache in durable session storage. "
                            "Restore the missing cache/textures artifacts or "
                            "regenerate the complete texture set before applying."
                        ),
                    )
                apply_cache_key_mode = _APPLY_CACHE_KEY_MODE_PLAN
            elif not plan_path.is_file():
                apply_cache_key_mode = _APPLY_CACHE_KEY_MODE_LEGACY
            elif stored_key_mode == _APPLY_CACHE_KEY_MODE_LEGACY:
                apply_cache_key_mode = (
                    _APPLY_CACHE_KEY_MODE_PLAN
                    if await _has_complete_plan_unit_texture_cache(
                        manager,
                        session_id,
                        session_dir,
                        plan_path,
                    )
                    else _APPLY_CACHE_KEY_MODE_LEGACY
                )
            else:
                # A modern completed run may predate the durable marker. Adopt
                # plan IDs only with positive evidence. Missing/corrupt cache
                # files and shared-store listing failures are indeterminate,
                # not evidence that stale display-key files are authoritative.
                if not await _has_complete_plan_unit_texture_cache(
                    manager,
                    session_id,
                    session_dir,
                    plan_path,
                ):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Cached apply key mode cannot be determined safely: "
                            "texture_plan.json exists, but the complete plan-unit "
                            "texture cache is unavailable and no durable key-mode "
                            "marker exists. Restore cache/apply_cache_key_mode.json "
                            "or regenerate the complete texture set before applying."
                        ),
                    )
                apply_cache_key_mode = _APPLY_CACHE_KEY_MODE_PLAN
            planning_config = pipeline_config.setdefault("planning", {})
            planning_config["resume_apply_textures"] = True
            planning_config["apply_texture_plan_unit_ids"] = (
                apply_cache_key_mode == _APPLY_CACHE_KEY_MODE_PLAN
            )
            if (
                apply_cache_key_mode == _APPLY_CACHE_KEY_MODE_LEGACY
                and TexturePipelineStep.GENERATE_TEXTURES not in request.steps
            ):
                # A pre-plan cache can legitimately contain up to the hard
                # 64-unit limit even though current generation defaults to 32.
                # Persist the compatibility plan for scoping, but keep it
                # non-executable: cached apply may hydrate prompts/units from
                # it, while every current or future generation request still
                # has to pass the normal executable-plan gate.
                planning_config["plan_only"] = True
                planning_config["allow_non_executable_cached_apply_plan"] = True
                texture_config = pipeline_config.setdefault("texture", {})
                configured_unit_limit = texture_config.get("max_texture_units")
                if (
                    isinstance(configured_unit_limit, int)
                    and not isinstance(configured_unit_limit, bool)
                    and configured_unit_limit > 0
                ):
                    texture_config["max_texture_units"] = min(
                        configured_unit_limit,
                        config.texture_plan_hard_cap,
                    )
                else:
                    texture_config["max_texture_units"] = config.texture_plan_hard_cap
            apply_prerequisites = []
            if steps_cfg.get(TexturePipelineStep.PREPARE_UVS.value, {}).get(
                "enabled", True
            ):
                apply_prerequisites.append(TexturePipelineStep.PREPARE_UVS.value)
            apply_prerequisites.append(TexturePipelineStep.DISCOVER_MATERIALS.value)
            if not plan_path.is_file():
                # Pre-plan sessions used display-derived cache keys. Rebuild a
                # plan to satisfy prompt expansion, but retain those legacy keys
                # when hydrating the blended-map cache in the executor.
                apply_prerequisites.append(TexturePipelineStep.PLAN_TEXTURES.value)
            apply_prerequisites.append(TexturePipelineStep.GENERATE_PROMPTS.value)
            only_steps = list(dict.fromkeys([*apply_prerequisites, *only_steps]))

        if TexturePipelineStep.GENERATE_TEXTURES in request.steps:
            if (
                not plan_path.is_file()
                and stored_key_mode is None
                and apply_cache_key_mode is None
            ):
                # Before plans existed, blended maps used display-derived keys.
                # Record that fact before a generate-only request creates a plan
                # so a later apply request never has to infer the old key mode.
                apply_cache_key_mode = _APPLY_CACHE_KEY_MODE_LEGACY
            if not plan_path.is_file() and request.texture_unit_ids:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Targeted regeneration requires the immutable "
                        "texture_plan.json from the original run."
                    ),
                )
            planning_config = pipeline_config.setdefault("planning", {})
            prerequisites = ["discover_materials", "generate_prompts"]
            if plan_path.is_file():
                plan = validate_texture_plan_payload(plan_path.read_bytes())
                selected_ids = tuple(unit.unit_id for unit in plan.selected_units)
                regenerate_ids = tuple(request.texture_unit_ids or selected_ids)
                unknown_ids = sorted(set(regenerate_ids) - set(selected_ids))
                if unknown_ids:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Regeneration unit IDs are outside the approved texture "
                            f"plan: {', '.join(unknown_ids)}"
                        ),
                    )
                planning_config["resume_execution"] = True
                planning_config["regenerate_unit_ids"] = list(regenerate_ids)
            else:
                # Pre-WP0 sessions have no persisted plan. Rebuild one before
                # backend work, while still requiring an existing immutable
                # plan for exact unit-ID regeneration.
                prerequisites.insert(1, "plan_textures")

            only_steps = list(
                dict.fromkeys(
                    [
                        *prerequisites,
                        *only_steps,
                    ]
                )
            )

        disabled_requested = [
            s for s in only_steps if not steps_cfg.get(s, {}).get("enabled", True)
        ]
        if disabled_requested:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Requested step(s) are disabled in this deploy: "
                    f"{', '.join(disabled_requested)}. The default Docker "
                    "Compose deploy does not configure a rendering backend; "
                    "render_previews and render are disabled. Either deploy "
                    "with a rendering backend configured or omit these steps."
                ),
            )

        if apply_cache_key_mode is not None:
            await _persist_apply_cache_key_mode(
                manager,
                session_id,
                session_dir,
                apply_cache_key_mode,
            )

        promote_apply_cache_after_sync = (
            regenerates_complete_plan_cache
            and stored_key_mode != _APPLY_CACHE_KEY_MODE_PLAN
            and apply_cache_key_mode != _APPLY_CACHE_KEY_MODE_PLAN
        )

        async def _finalize_apply_cache_key_mode() -> None:
            await _promote_apply_cache_key_mode_after_artifact_sync(
                manager,
                session_id,
                session_dir,
                plan_path,
            )

        # Override material_textures if provided
        material_textures_config = request.material_textures_config()
        if material_textures_config is not None:
            pipeline_config["material_textures"] = {
                **(pipeline_config.get("material_textures") or {}),
                **material_textures_config,
            }
            _sync_texture_mode_for_overrides(pipeline_config, material_textures_config)

        # Regenerate is incremental — keep completed_steps / progress —
        # but every other run-scoped state surface must be reset so the
        # executor and `/status` cannot see prior-run remnants. Reset is
        # deferred until after every step that could fail above has
        # succeeded; the snapshot drives rollback if `register()` raises.
        reset_snapshot = await asyncio.to_thread(
            _reset_session_for_new_run,
            manager,
            session_id,
            fresh=False,
        )
        await get_event_bus().seed_pending_session(session_id)

        job_registry = get_job_registry()
        await job_registry.register(
            session_id,
            execute_pipeline_async(
                session_id=session_id,
                config_dict=pipeline_config,
                session_manager=manager,
                only_steps=only_steps,
                acquire_worker_lock=False,
                worker_owner_token=getattr(
                    worker_lock,
                    "_wu_shared_reservation_token",
                    None,
                ),
                on_artifacts_synced=(
                    _finalize_apply_cache_key_mode
                    if promote_apply_cache_after_sync
                    else None
                ),
            ),
            on_never_started=_cancel_never_started_callback(
                manager,
                session_id,
            ),
            on_finished=_release_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
            on_queued_heartbeat=_heartbeat_worker_slot_callback(
                manager,
                session_id,
                worker_lock,
            ),
        )
    except Exception:
        await asyncio.to_thread(manager.release_worker_lock, worker_lock, session_id)
        if reset_snapshot:
            await asyncio.to_thread(
                _restore_session_after_reset_failure,
                manager,
                session_id,
                reset_snapshot,
            )
        raise

    logger.info(f"Pipeline regeneration registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
        plan_url=f"/pipeline/{session_id}/plan",
    )


@router.get("/{session_id}/event-log")
async def get_event_log(session_id: str) -> dict[str, Any]:
    """Get the persisted event log for a session with sanitized diagnostics."""
    manager = get_session_manager()

    if not await asyncio.to_thread(manager.session_exists, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    storage_root = config.session_storage_path
    events = []
    try:
        event_log = await asyncio.to_thread(manager.get_event_log, session_id)
        for event in event_log:
            if isinstance(event, dict):
                if isinstance(event.get("message"), str):
                    event["message"] = sanitize_message(event["message"], storage_root)
                extra = event.get("extra")
                if isinstance(extra, dict):
                    event["extra"] = sanitize_step_stats(extra, storage_root)
                events.append(event)

        return {"events": events, "total": len(events)}

    except Exception as e:
        logger.error(f"Failed to load event log for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load event log: {e}")
