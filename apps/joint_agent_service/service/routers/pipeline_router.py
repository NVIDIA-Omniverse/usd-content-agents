# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline API endpoints - Core workflow operations."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import stat
import uuid
from collections.abc import Awaitable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from joint_agent.api.defaults import (
    DEFAULT_RENDER_BACKEND,
    build_default_pipeline_config,
)
from joint_agent.joint_rigger_options import (
    DEFAULT_CANDIDATE_READINESS_POLICY,
    DEFAULT_MISSING_DEPENDENCY_POLICY,
    DEFAULT_SERVICE_JOINT_RIGGER_ADAPTER,
    SUPPORTED_CANDIDATE_READINESS_POLICIES,
    SUPPORTED_JOINT_RIGGER_ADAPTERS,
    SUPPORTED_MISSING_DEPENDENCY_POLICIES,
)
from sse_starlette import EventSourceResponse
from world_understanding.functions.graphics.rendering_backend_factory import (
    RENDERING_BACKEND_NAMES,
    validate_rendering_backend_name,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    copy_open_file_to_confined,
    iter_open_regular_files,
    open_confined_directory,
    open_confined_directory_at,
)
from world_understanding.utils.credentials import (
    InlineSecretError,
    aensure_no_inline_secrets,
    ensure_no_inline_secrets,
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
from ..models.requests import RegenerateRequest
from ..models.responses import (
    S3_INPUT_ERROR_RESPONSES,
    PipelineCancellationAccepted,
    PipelineError,
    PipelineResults,
    PipelineRunCreated,
    PipelineStatus,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..runtime.registry import JobRegistry
from ..session.cache_publications import (
    PIPELINE_CONFIG_PUBLICATION_ID_FIELD,
    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD,
    cache_publication_path,
    cache_publication_prefix,
    parse_cache_publications,
    pipeline_config_publication_key,
)
from ..session.manager import ACTIVE_RUN_ID_FIELD, SessionManager
from ..workers.executor import (
    execute_pipeline_async,
    finalize_pipeline_run,
    maintain_run_claim,
)

logger = logging.getLogger(__name__)

_INVALID_PIPELINE_CONFIG_DETAIL = "Pipeline configuration is invalid"
_PIPELINE_CONFIG_INTEGRITY_DETAIL = (
    "Original config publication failed integrity verification"
)
_MAX_PIPELINE_CONFIG_PUBLICATION_BYTES = 4 * 1024 * 1024

# Create router
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


class _RunClaimGuard:
    """One already-running claim heartbeat transferable to the job registry."""

    def __init__(
        self,
        manager: SessionManager,
        session_id: str,
        run_id: str,
    ) -> None:
        self._task = asyncio.create_task(
            maintain_run_claim(manager, session_id, run_id),
            name=f"joint-run-claim-{session_id[:8]}",
        )

    @property
    def task(self) -> asyncio.Task[None]:
        return self._task

    def __await__(self):
        return self._task.__await__()

    def close(self) -> None:
        if not self._task.done():
            self._task.cancel()

    async def stop(self) -> None:
        self.close()
        await asyncio.gather(self._task, return_exceptions=True)


def _raise_stopped_run_claim(guard: _RunClaimGuard) -> None:
    try:
        guard.task.result()
    except asyncio.CancelledError as exc:
        cause: BaseException | None = exc
    except Exception as exc:
        cause = exc
    else:
        cause = None
    raise HTTPException(
        status_code=409,
        detail="Pipeline run claim was lost during admission",
    ) from cause


async def _await_with_run_claim(
    awaitable: Awaitable[Any],
    guard: _RunClaimGuard,
) -> Any:
    """Run one admission operation only while the reserved claim stays live."""
    if guard.task.done():
        _raise_stopped_run_claim(guard)

    operation = asyncio.create_task(_drain_admission_operation(awaitable))
    try:
        done, _ = await asyncio.wait(
            {operation, guard.task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if guard.task in done:
            _raise_stopped_run_claim(guard)
        return operation.result()
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


async def _drain_admission_operation(awaitable: Awaitable[Any]) -> Any:
    """Finish already-started thread work before propagating cancellation."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
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


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


async def _reserve_run_admission(
    manager: SessionManager,
    session_id: str,
    run_id: str,
) -> JobRegistry:
    """Reserve local admission first, then the distributed run generation."""

    job_registry = get_job_registry()
    if not await job_registry.reserve_admission(session_id, run_id):
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already active or still draining for this session",
        )
    try:
        reserved = await manager.reserve_run(session_id, run_id)
    except BaseException:
        await job_registry.release_admission(session_id, run_id)
        raise
    if not reserved:
        await job_registry.release_admission(session_id, run_id)
        raise HTTPException(
            status_code=409,
            detail="A pipeline run is already active for this session",
        )
    return job_registry


async def _cleanup_rejected_pipeline_session(
    manager: SessionManager,
    session_id: str,
    *,
    session_created_here: bool,
) -> None:
    """Best-effort cleanup for a rejected request-owned session."""
    if not session_created_here:
        return
    try:
        get_event_bus().cleanup_session(session_id)
    except Exception:  # pragma: no cover - defensive cleanup containment
        logger.error("Failed to clean up rejected pipeline event state")
    try:
        deleted = await manager.delete_session(session_id)
    except Exception:  # pragma: no cover - defensive cleanup containment
        logger.error("Failed to clean up rejected pipeline session")
        return
    if not deleted:  # pragma: no cover - manager reports store cleanup failure
        logger.error("Failed to clean up rejected pipeline session")


_JOINT_RIGGER_ARTIFACT_ROUTES = {
    "joint_rigger_output": "joint-rigger-output",
    "joint_rigger_diagnostics": "joint-rigger-diagnostics",
    "joint_rigger_validation": "joint-rigger-validation",
}
_PIPELINE_INPUT_PREFIX = "input/"
_PIPELINE_DATASET_PREFIX = "cache/dataset/"
_PIPELINE_PREDICTIONS_PREFIX = "cache/predictions/"
_CACHE_PREFIX_NAMESPACES = {
    _PIPELINE_DATASET_PREFIX: "dataset",
    _PIPELINE_PREDICTIONS_PREFIX: "predictions",
}


def _result_download_urls(session_id: str, metadata: dict) -> dict[str, str]:
    """Return only result URLs that should be advertised to clients."""
    urls = {
        "predictions": f"/artifacts/{session_id}/predictions",
        "articulation_candidates": f"/artifacts/{session_id}/articulation-candidates",
        "articulation_report": f"/artifacts/{session_id}/articulation-report",
        "report": f"/artifacts/{session_id}/report",
        "dataset": f"/artifacts/{session_id}/dataset",
    }

    results = metadata.get("results", {})
    joint_rigger_artifacts = results.get("joint_rigger_artifacts")
    if not isinstance(joint_rigger_artifacts, dict):
        joint_rigger_artifacts = {}

    for key, route in _JOINT_RIGGER_ARTIFACT_ROUTES.items():
        if joint_rigger_artifacts.get(key):
            urls[key] = f"/artifacts/{session_id}/{route}"

    return urls


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


def _apply_prediction_request_limit(pipeline_config: dict) -> None:
    """Clamp prediction workers to the operator-configured service cap."""
    step_config = pipeline_config.get("steps", {}).get("predict")
    if not isinstance(step_config, dict):
        return

    existing_workers = step_config.get("max_workers", config.vlm_max_workers)
    try:
        worker_limit = min(int(existing_workers), config.vlm_max_workers)
    except (TypeError, ValueError):
        worker_limit = config.vlm_max_workers

    step_config["max_workers"] = max(1, worker_limit)


def _form_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _form_bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _validated_choice(
    *,
    field_name: str,
    value: str | None,
    allowed_values: tuple[str, ...],
) -> str | None:
    text = _form_text(value)
    if text is None:
        return None
    if text not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be one of: {allowed}",
        )
    return text


def _joint_rigger_step_options(
    *,
    apply_joint_rigger: bool,
    joint_rigger_adapter: str | None,
    joint_rigger_on_missing_dependency: str | None,
    joint_rigger_on_unready_candidates: str | None,
    joint_rigger_template: str | None,
    joint_rigger_apply_masses: bool | None,
    joint_rigger_apply_collision: bool | None,
) -> dict[str, object] | None:
    """Validate and normalize opt-in Joint Rigger request settings."""
    joint_rigger_apply_masses = _form_bool_or_none(joint_rigger_apply_masses)
    joint_rigger_apply_collision = _form_bool_or_none(joint_rigger_apply_collision)
    option_supplied = any(
        value is not None
        for value in (
            _form_text(joint_rigger_adapter),
            _form_text(joint_rigger_on_missing_dependency),
            _form_text(joint_rigger_on_unready_candidates),
            _form_text(joint_rigger_template),
            joint_rigger_apply_masses,
            joint_rigger_apply_collision,
        )
    )
    if not apply_joint_rigger:
        if option_supplied:
            raise HTTPException(
                status_code=400,
                detail=(
                    "apply_joint_rigger must be true when Joint Rigger options "
                    "are provided"
                ),
            )
        return None

    adapter = _validated_choice(
        field_name="joint_rigger_adapter",
        value=joint_rigger_adapter,
        allowed_values=SUPPORTED_JOINT_RIGGER_ADAPTERS,
    )
    adapter = adapter or DEFAULT_SERVICE_JOINT_RIGGER_ADAPTER
    on_missing_dependency = _validated_choice(
        field_name="joint_rigger_on_missing_dependency",
        value=joint_rigger_on_missing_dependency,
        allowed_values=SUPPORTED_MISSING_DEPENDENCY_POLICIES,
    )
    on_unready_candidates = _validated_choice(
        field_name="joint_rigger_on_unready_candidates",
        value=joint_rigger_on_unready_candidates,
        allowed_values=SUPPORTED_CANDIDATE_READINESS_POLICIES,
    )
    template = _form_text(joint_rigger_template)

    if adapter == "owned_core":
        if joint_rigger_apply_masses is True:
            raise HTTPException(
                status_code=400,
                detail=(
                    "joint_rigger_apply_masses must be false for the "
                    "topology-only owned_core adapter"
                ),
            )
        if joint_rigger_apply_collision is True:
            raise HTTPException(
                status_code=400,
                detail=(
                    "joint_rigger_apply_collision must be false for the "
                    "topology-only owned_core adapter"
                ),
            )

    step_config: dict[str, object] = {
        "enabled": True,
        "adapter": adapter,
        "on_missing_dependency": (
            on_missing_dependency or DEFAULT_MISSING_DEPENDENCY_POLICY
        ),
        "on_unready_candidates": (
            on_unready_candidates or DEFAULT_CANDIDATE_READINESS_POLICY
        ),
    }
    if template is not None:
        step_config["joint_rigger_template"] = template
    if adapter == "owned_core":
        step_config["apply_masses"] = False
        step_config["apply_collision"] = False
    elif joint_rigger_apply_masses is not None:
        step_config["apply_masses"] = joint_rigger_apply_masses
    if adapter != "owned_core" and joint_rigger_apply_collision is not None:
        step_config["apply_collision"] = joint_rigger_apply_collision
    return step_config


def _apply_joint_rigger_step_options(
    pipeline_config: dict, step_options: dict[str, object] | None
) -> None:
    """Apply already-validated Joint Rigger settings to the generated config."""
    if step_options is None:
        return
    step_config = pipeline_config.setdefault("steps", {}).setdefault(
        "apply_joint_rigger", {}
    )
    step_config.update(step_options)
    if step_options.get("adapter") == "owned_core":
        # The service form has no explicit prediction-artifact input. Keep
        # owned_core candidate-only even when an alternate defaults builder is
        # supplied by an embedding application.
        step_config.pop("predictions_path", None)
        candidate_config = pipeline_config.setdefault("steps", {}).setdefault(
            "infer_articulation_candidates", {}
        )
        candidate_config["candidate_joint_types"] = ["revolute", "prismatic"]


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    chunk_size: int = 2 * 1024 * 1024,
    max_bytes: int | None = None,
) -> int:
    """Stream an upload into a temporary snapshot, then publish atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.upload")
    total_bytes = 0

    try:
        with snapshot.open("xb") as f:
            while True:
                data = await upload.read(chunk_size)
                if not data:
                    break
                total_bytes += len(data)
                if max_bytes is not None and total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the configured upload limit of "
                            f"{config.max_upload_size_mb} MiB"
                        ),
                    )
                f.write(data)
        snapshot.replace(dest)
    finally:
        snapshot.unlink(missing_ok=True)

    return total_bytes


def _configured_upload_limit_bytes() -> int | None:
    """Return the opt-in upload limit in bytes, or None when unbounded."""
    if config.max_upload_size_mb <= 0:
        return None
    return config.max_upload_size_mb * 1024 * 1024


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
    snapshot = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.download")

    try:
        download_file_from_s3(s3_uri, snapshot)
    except FileNotFoundError:
        snapshot.unlink(missing_ok=True)
        log_durable_failure(
            logger,
            "pipeline_s3_object_not_found",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(status_code=404, detail="S3 object not found") from None
    except PermissionError:
        snapshot.unlink(missing_ok=True)
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
        snapshot.unlink(missing_ok=True)
        log_durable_failure(
            logger,
            "pipeline_s3_download_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=502, detail="Failed to download from S3"
        ) from None

    try:
        max_bytes = _configured_upload_limit_bytes()
        if max_bytes is not None and snapshot.stat().st_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "S3 file exceeds the configured upload limit of "
                    f"{config.max_upload_size_mb} MiB"
                ),
            )
        snapshot.replace(local_path)
    finally:
        snapshot.unlink(missing_ok=True)

    return local_path


@router.post(
    "/upload-usd",
    response_model=SessionCreated,
    status_code=201,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def upload_usd_immediate(
    usd_file: UploadFile = File(
        None,
        description="USD file to upload. Provide exactly one of usd_file or s3_uri.",
    ),
    s3_uri: str = Form(
        None,
        description=(
            "S3 URI to a USD file. Its exact bucket name must be listed in "
            "JA_S3_ALLOWED_BUCKETS; an empty allowlist rejects all client S3 inputs. "
            "Provide exactly one of usd_file or s3_uri."
        ),
    ),
) -> SessionCreated:
    """Create a session from exactly one uploaded or authorized S3 USD source.

    Client S3 access is fail-closed: the exact bucket name must be listed in
    ``JA_S3_ALLOWED_BUCKETS``, and an empty allowlist rejects every client S3 URI
    before session-manager or S3 I/O.
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
        session_dir = await manager.create_session(session_id)
        try:
            local_path = _download_s3_to_session(s3_uri, session_dir)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({local_path.suffix})"
            )
            # Push input to store so other instances can find it
            try:
                await manager.sync_to_store(session_id)
            except Exception:
                log_durable_failure(
                    logger,
                    "pipeline_usd_sync_failed",
                    phase=FailurePhase.SYNC_UPLOAD,
                    retryable=True,
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
                phase=FailurePhase.LOCAL_PUBLICATION,
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

    try:
        total_bytes = await _stream_copy(
            usd_file,
            usd_path,
            max_bytes=_configured_upload_limit_bytes(),
        )
        size_mb = total_bytes / (1024 * 1024)

        logger.info(
            f"USD uploaded for session {session_id[:8]}: {size_mb:.2f}MB ({original_ext})"
        )

        # Push input to store so other instances can find it
        try:
            await manager.sync_to_store(session_id)
        except Exception:
            log_durable_failure(
                logger,
                "pipeline_usd_sync_failed",
                phase=FailurePhase.SYNC_UPLOAD,
                retryable=True,
            )

        return SessionCreated(
            session_id=session_id,
            status="ready",
            message="USD uploaded successfully",
            estimated_duration_minutes=0,
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
        raise HTTPException(status_code=500, detail="Failed to upload USD") from None


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in [".usd", ".usda", ".usdc", ".usdz"]:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


class _PipelineConfigPublicationTooLargeError(ValueError):
    """A serialized config cannot cross the immutable publication boundary."""


def _write_pipeline_config_publication(
    storage_path: Path,
    session_id: str,
    run_id: str,
    pipeline_config: dict[str, Any],
) -> tuple[Path, str, bytes]:
    """Descriptor-publish one bounded run config without following aliases."""

    ensure_no_inline_secrets(
        pipeline_config,
        context="joint pipeline configuration",
    )
    publication_bytes = yaml.safe_dump(
        pipeline_config,
        default_flow_style=False,
    ).encode("utf-8")
    if len(publication_bytes) > _MAX_PIPELINE_CONFIG_PUBLICATION_BYTES:
        raise _PipelineConfigPublicationTooLargeError

    publication_components = _pipeline_config_parent_components(session_id, run_id)
    publication_directory_name = publication_components[-1]
    publication_path = (
        storage_path / session_id / pipeline_config_publication_key(run_id)
    )
    with (
        _open_storage_root(storage_path) as storage_descriptor,
        _open_directory_chain(
            storage_descriptor,
            publication_components[:-1],
            create=True,
        ) as publication_parent_descriptor,
    ):
        publication_directory_created = False
        try:
            os.mkdir(
                publication_directory_name,
                mode=0o700,
                dir_fd=publication_parent_descriptor,
            )
            publication_directory_created = True
            with _open_directory_chain(
                publication_parent_descriptor,
                (publication_directory_name,),
            ) as publication_descriptor:
                try:
                    config_descriptor = os.open(
                        "config.yaml",
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=publication_descriptor,
                    )
                    with os.fdopen(config_descriptor, "wb") as config_file:
                        config_file.write(publication_bytes)
                        config_file.flush()
                        os.fsync(config_file.fileno())

                    # Re-open the complete path from the held storage root and
                    # compare identities before accepting a path binding. An
                    # ancestor swapped to a symlink cannot redirect the write
                    # or turn detached bytes into an accepted publication key.
                    with _open_directory_chain(
                        storage_descriptor,
                        publication_components,
                    ) as rebound_descriptor:
                        publication_stat = os.fstat(publication_descriptor)
                        rebound_stat = os.fstat(rebound_descriptor)
                        if (
                            publication_stat.st_dev != rebound_stat.st_dev
                            or publication_stat.st_ino != rebound_stat.st_ino
                        ):
                            raise _PipelineConfigIntegrityError(
                                "Publication directory binding changed"
                            )

                    os.fsync(publication_descriptor)
                    os.fsync(publication_parent_descriptor)
                except BaseException:
                    _remove_descriptor_entry(publication_descriptor, "config.yaml")
                    raise
        except BaseException:
            if publication_directory_created:
                _remove_descriptor_entry(
                    publication_parent_descriptor,
                    publication_directory_name,
                )
            raise
    return (
        publication_path,
        hashlib.sha256(publication_bytes).hexdigest(),
        publication_bytes,
    )


def _sha256_stream(
    stream: Any,
    *,
    max_bytes: int = _MAX_PIPELINE_CONFIG_PUBLICATION_BYTES,
) -> str:
    """Hash at most ``max_bytes`` while reading only one overflow byte."""
    digest = hashlib.sha256()
    bytes_read = 0
    while bytes_read <= max_bytes:
        read_size = min(2 * 1024 * 1024, max_bytes + 1 - bytes_read)
        chunk = stream.read(read_size)
        if not chunk:
            return digest.hexdigest()
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("Publication stream must be binary")
        bounded_chunk = chunk[:read_size]
        bytes_read += len(bounded_chunk)
        if bytes_read > max_bytes or len(chunk) > read_size:
            raise _PipelineConfigPublicationTooLargeError
        digest.update(bounded_chunk)
    raise _PipelineConfigPublicationTooLargeError


async def _persist_pipeline_config(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    pipeline_config: dict[str, Any],
    input_usd_path: Path,
) -> str:
    """Publish one immutable run-scoped config before scheduling execution."""

    session_dir = manager.get_session_dir(session_id)
    publication_key = pipeline_config_publication_key(run_id)
    try:
        input_key = input_usd_path.relative_to(session_dir).as_posix()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Input USD is outside the session",
        ) from exc
    if not input_key.startswith(_PIPELINE_INPUT_PREFIX):
        raise HTTPException(status_code=400, detail="Input USD is outside the session")
    try:
        _, publication_sha256, publication_bytes = await asyncio.to_thread(
            _write_pipeline_config_publication,
            manager.storage_path,
            session_id,
            run_id,
            pipeline_config,
        )
    except InlineSecretError:
        raise HTTPException(
            status_code=400,
            detail=_INVALID_PIPELINE_CONFIG_DETAIL,
        ) from None
    except _PipelineConfigPublicationTooLargeError:
        raise HTTPException(
            status_code=400,
            detail=_INVALID_PIPELINE_CONFIG_DETAIL,
        ) from None
    except Exception:
        log_durable_failure(
            logger,
            "joint_pipeline_config_local_publication_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline config could not be persisted",
        ) from None

    try:
        await manager.sync_to_store(
            session_id,
            prefix=input_key,
            overwrite=False,
        )
        await manager.sync_to_store(
            session_id,
            prefix=publication_key,
            overwrite=False,
        )
        publication_stream = await manager.store.open_read(
            session_id,
            publication_key,
        )
        try:
            persisted_sha256 = await asyncio.to_thread(
                _sha256_stream,
                publication_stream,
            )
        finally:
            publication_stream.close()
        persisted = await manager.store.exists(
            session_id, input_key
        ) and hmac.compare_digest(
            persisted_sha256,
            publication_sha256,
        )
        if persisted:
            await asyncio.to_thread(
                _replace_pipeline_config_from_validated_bytes,
                manager.storage_path,
                session_id,
                publication_bytes,
            )
    except Exception:
        log_durable_failure(
            logger,
            "joint_pipeline_config_sync_failed",
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )
        await _quarantine_pipeline_config_publication(
            manager,
            session_id,
            run_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline config could not be persisted",
        ) from None

    if not persisted:
        log_durable_failure(
            logger,
            "joint_pipeline_config_verification_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        await _quarantine_pipeline_config_publication(
            manager,
            session_id,
            run_id,
        )
        raise HTTPException(
            status_code=503,
            detail="Pipeline config could not be persisted",
        )
    return publication_sha256


class _PipelineConfigIntegrityError(RuntimeError):
    """A bound publication cannot be trusted as canonical configuration."""


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_descriptor_component(value: str) -> bool:
    """Return whether ``value`` is one portable, relative path component."""

    return bool(
        value
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


def _directory_open_flags() -> int:
    """Return the flags required for descriptor-relative safe traversal."""

    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not nofollow_flag:
        raise _PipelineConfigIntegrityError(
            "Safe publication directory traversal is unavailable"
        )
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory_flag | nofollow_flag


@contextmanager
def _open_storage_root(storage_path: Path) -> Iterator[int]:
    """Hold the canonical configured storage root without following its leaf."""

    try:
        canonical_root = storage_path.resolve(strict=True)
        descriptor = os.open(canonical_root, _directory_open_flags())
    except _PipelineConfigIntegrityError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise _PipelineConfigIntegrityError(
            "Publication storage root could not be verified"
        ) from None
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_directory_chain(
    parent_descriptor: int,
    components: tuple[str, ...],
    *,
    create: bool = False,
    create_mode: int = 0o700,
) -> Iterator[int]:
    """Traverse and hold real directories relative to one already-held root."""

    descriptors: list[int] = []
    try:
        current_descriptor = os.dup(parent_descriptor)
        descriptors.append(current_descriptor)
        for component in components:
            if not _safe_descriptor_component(component):
                raise _PipelineConfigIntegrityError(
                    "Publication path contains an invalid component"
                )
            if create:
                try:
                    os.mkdir(
                        component,
                        mode=create_mode,
                        dir_fd=current_descriptor,
                    )
                except FileExistsError:
                    pass
            next_descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current_descriptor,
            )
            descriptors.append(next_descriptor)
            current_descriptor = next_descriptor
            if create:
                os.fchmod(current_descriptor, create_mode)
        yield current_descriptor
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _pipeline_config_parent_components(
    session_id: str,
    publication_id: str,
) -> tuple[str, ...]:
    """Return validated components leading to one publication directory."""

    try:
        publication_key = Path(pipeline_config_publication_key(publication_id))
    except (TypeError, ValueError):
        raise _PipelineConfigIntegrityError(
            "Publication path is not canonical"
        ) from None
    if not _safe_descriptor_component(session_id):
        raise _PipelineConfigIntegrityError("Publication session ID is invalid")
    components = (session_id, *publication_key.parts[:-1])
    if not all(_safe_descriptor_component(component) for component in components):
        raise _PipelineConfigIntegrityError("Publication path is not canonical")
    return components


def _remove_descriptor_entry(parent_descriptor: int, name: str) -> None:
    """Remove one entry relative to a held directory without following it."""

    try:
        entry_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if stat.S_ISDIR(entry_stat.st_mode):
        os.rmdir(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _validate_pipeline_config_publication(
    storage_path: Path,
    session_id: str,
    publication_id: str,
    expected_sha256: str,
) -> bytes:
    """Read and verify the exact bytes later published as canonical config."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        with (
            _open_storage_root(storage_path) as storage_descriptor,
            _open_directory_chain(
                storage_descriptor,
                _pipeline_config_parent_components(session_id, publication_id),
            ) as publication_parent_descriptor,
        ):
            descriptor = os.open(
                "config.yaml",
                flags,
                dir_fd=publication_parent_descriptor,
            )
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise _PipelineConfigIntegrityError(
                        "Publication is not a regular file"
                    )
                with os.fdopen(descriptor, "rb", closefd=False) as publication_file:
                    publication_bytes = publication_file.read(
                        _MAX_PIPELINE_CONFIG_PUBLICATION_BYTES + 1
                    )
            finally:
                os.close(descriptor)
    except _PipelineConfigIntegrityError:
        raise
    except OSError:
        raise _PipelineConfigIntegrityError(
            "Publication could not be read safely"
        ) from None
    if len(publication_bytes) > _MAX_PIPELINE_CONFIG_PUBLICATION_BYTES:
        raise _PipelineConfigIntegrityError("Publication exceeds integrity limit")
    actual_sha256 = hashlib.sha256(publication_bytes).hexdigest()
    if not hmac.compare_digest(expected_sha256, actual_sha256):
        raise _PipelineConfigIntegrityError("Publication digest does not match")
    try:
        pipeline_config = yaml.safe_load(publication_bytes)
        if not isinstance(pipeline_config, dict):
            raise _PipelineConfigIntegrityError("Publication is not a mapping")
        ensure_no_inline_secrets(
            pipeline_config,
            context="joint pipeline configuration publication",
        )
    except _PipelineConfigIntegrityError:
        raise
    except Exception:
        raise _PipelineConfigIntegrityError(
            "Publication content failed validation"
        ) from None
    return publication_bytes


def _move_pipeline_config_to_quarantine(
    storage_path: Path,
    session_id: str,
    publication_id: str,
) -> None:
    """Move rejected bytes outside every session read/list surface."""
    with ExitStack() as stack:
        try:
            storage_descriptor = stack.enter_context(_open_storage_root(storage_path))
            source_descriptor = stack.enter_context(
                _open_directory_chain(
                    storage_descriptor,
                    _pipeline_config_parent_components(session_id, publication_id),
                )
            )
        except (_PipelineConfigIntegrityError, OSError, RuntimeError, ValueError):
            return

        try:
            publication_stat = os.stat(
                "config.yaml",
                dir_fd=source_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return

        # Never move a symlink, FIFO, socket, or device node into quarantine and
        # then operate on it. Descriptor-relative removal cannot follow an
        # attacker-swapped ancestor or leaf.
        if not stat.S_ISREG(publication_stat.st_mode):
            _remove_descriptor_entry(source_descriptor, "config.yaml")
            return

        source_file_descriptor = os.open(
            "config.yaml",
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=source_descriptor,
        )
        try:
            source_file_stat = os.fstat(source_file_descriptor)
            if (
                not stat.S_ISREG(source_file_stat.st_mode)
                or source_file_stat.st_dev != publication_stat.st_dev
                or source_file_stat.st_ino != publication_stat.st_ino
            ):
                raise OSError("Pipeline config quarantine source changed type")

            try:
                quarantine_components = (
                    ".quarantine",
                    session_id,
                    "pipeline_configs",
                    publication_id,
                )
                with _open_directory_chain(
                    storage_descriptor,
                    quarantine_components,
                    create=True,
                ) as quarantine_descriptor:
                    try:
                        os.stat(
                            "config.yaml",
                            dir_fd=quarantine_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        quarantine_name = "config.yaml"
                    else:
                        quarantine_name = f"config.{uuid.uuid4().hex}.yaml"

                    os.replace(
                        "config.yaml",
                        quarantine_name,
                        src_dir_fd=source_descriptor,
                        dst_dir_fd=quarantine_descriptor,
                    )
                    try:
                        quarantine_file_descriptor = os.open(
                            quarantine_name,
                            os.O_RDONLY
                            | getattr(os, "O_CLOEXEC", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=quarantine_descriptor,
                        )
                    except OSError:
                        _remove_descriptor_entry(
                            quarantine_descriptor,
                            quarantine_name,
                        )
                        raise
                    try:
                        quarantined_stat = os.fstat(quarantine_file_descriptor)
                        if (
                            not stat.S_ISREG(quarantined_stat.st_mode)
                            or quarantined_stat.st_dev != source_file_stat.st_dev
                            or quarantined_stat.st_ino != source_file_stat.st_ino
                        ):
                            _remove_descriptor_entry(
                                quarantine_descriptor,
                                quarantine_name,
                            )
                            raise OSError(
                                "Pipeline config quarantine source changed during move"
                            )
                        os.fchmod(quarantine_file_descriptor, 0o600)
                    finally:
                        os.close(quarantine_file_descriptor)
            except Exception:
                # A rejected publication must not remain on a normal read/list
                # surface when the quarantine destination itself is unsafe.
                _remove_descriptor_entry(source_descriptor, "config.yaml")
                raise
        finally:
            os.close(source_file_descriptor)


async def _quarantine_pipeline_config_publication(
    manager: SessionManager,
    session_id: str,
    publication_id: str,
) -> None:
    try:
        await asyncio.to_thread(
            _move_pipeline_config_to_quarantine,
            manager.storage_path,
            session_id,
            publication_id,
        )
    except Exception:
        log_durable_failure(
            logger,
            "joint_pipeline_config_quarantine_failed",
            phase=FailurePhase.ROLLBACK,
            retryable=False,
        )
    # The descriptor-relative move above already removes a local-store key.
    # Re-resolving the same path through the store would reopen an ancestor-swap
    # window after the safe move. Remote stores still need their backing object
    # deleted independently of the local cache quarantine.
    if getattr(manager.store, "kind", "local") == "local":
        return
    try:
        await manager.store.delete_key(
            session_id,
            pipeline_config_publication_key(publication_id),
        )
    except Exception:
        log_durable_failure(
            logger,
            "joint_pipeline_config_quarantine_store_delete_failed",
            phase=FailurePhase.ROLLBACK,
            retryable=True,
        )


async def _sync_pipeline_inputs_without_mutable_config(
    manager: SessionManager,
    session_id: str,
) -> None:
    """Hydrate input artifacts without accepting the mutable config copy."""
    input_keys = await manager.store.list_keys(
        session_id, prefix=_PIPELINE_INPUT_PREFIX
    )
    for input_key in input_keys:
        if input_key == "input/config.yaml":
            continue
        await manager.sync_from_store(
            session_id,
            prefix=input_key,
            overwrite=True,
        )


def _replace_pipeline_config_from_validated_bytes(
    storage_path: Path,
    session_id: str,
    publication_bytes: bytes,
) -> None:
    """Atomically publish the same bounded bytes accepted by validation."""
    temporary_name = f".config.yaml.{uuid.uuid4().hex}.tmp"
    with _open_storage_root(storage_path) as storage_descriptor:
        with _open_directory_chain(
            storage_descriptor,
            (session_id,),
        ) as session_descriptor:
            try:
                os.mkdir("input", mode=0o700, dir_fd=session_descriptor)
            except FileExistsError:
                pass
            with _open_directory_chain(
                session_descriptor,
                ("input",),
            ) as input_descriptor:
                temporary_descriptor = -1
                try:
                    temporary_descriptor = os.open(
                        temporary_name,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=input_descriptor,
                    )
                    with os.fdopen(temporary_descriptor, "wb") as config_file:
                        temporary_descriptor = -1
                        config_file.write(publication_bytes)
                        config_file.flush()
                        os.fsync(config_file.fileno())
                    os.replace(
                        temporary_name,
                        "config.yaml",
                        src_dir_fd=input_descriptor,
                        dst_dir_fd=input_descriptor,
                    )
                    os.fsync(input_descriptor)
                finally:
                    if temporary_descriptor >= 0:
                        os.close(temporary_descriptor)
                    try:
                        os.unlink(temporary_name, dir_fd=input_descriptor)
                    except FileNotFoundError:
                        pass


async def _restore_pipeline_config(
    manager: SessionManager,
    session_id: str,
    config_path: Path,
) -> None:
    """Refresh the canonical config in this instance's session root."""
    metadata = await manager.get_session_metadata(session_id)
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=404, detail="Session not found")
    publication_id = metadata.get(PIPELINE_CONFIG_PUBLICATION_ID_FIELD)
    if PIPELINE_CONFIG_PUBLICATION_ID_FIELD in metadata:
        try:
            publication_key = pipeline_config_publication_key(publication_id)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=409,
                detail="Original config publication binding is invalid",
            ) from None
        session_dir = manager.get_session_dir(session_id)
        try:
            if config_path != session_dir / "input" / "config.yaml":
                raise _PipelineConfigIntegrityError(
                    "Canonical config path is not session-owned"
                )
            await manager.sync_from_store(
                session_id,
                prefix=publication_key,
                overwrite=True,
            )
            expected_sha256 = metadata.get(PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD)
            if not _valid_sha256(expected_sha256):
                await _quarantine_pipeline_config_publication(
                    manager,
                    session_id,
                    publication_id,
                )
                raise _PipelineConfigIntegrityError(
                    "Publication digest binding is missing"
                )
            try:
                publication_bytes = await asyncio.to_thread(
                    _validate_pipeline_config_publication,
                    manager.storage_path,
                    session_id,
                    publication_id,
                    expected_sha256,
                )
            except _PipelineConfigIntegrityError:
                await _quarantine_pipeline_config_publication(
                    manager,
                    session_id,
                    publication_id,
                )
                raise
            await asyncio.to_thread(
                _replace_pipeline_config_from_validated_bytes,
                manager.storage_path,
                session_id,
                publication_bytes,
            )
            await _sync_pipeline_inputs_without_mutable_config(
                manager,
                session_id,
            )
        except _PipelineConfigIntegrityError:
            log_durable_failure(
                logger,
                "joint_pipeline_config_integrity_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=False,
            )
            raise HTTPException(
                status_code=409,
                detail=_PIPELINE_CONFIG_INTEGRITY_DETAIL,
            ) from None
        except Exception:
            log_durable_failure(
                logger,
                "joint_pipeline_config_restore_failed",
                phase=FailurePhase.PERSISTENCE_VERIFICATION,
                retryable=True,
            )
            raise HTTPException(
                status_code=409,
                detail="Original config publication is unavailable for session",
            ) from None
        return

    log_durable_failure(
        logger,
        "joint_pipeline_config_binding_missing",
        phase=FailurePhase.PERSISTENCE_VERIFICATION,
        retryable=False,
    )
    raise HTTPException(
        status_code=409,
        detail=_PIPELINE_CONFIG_INTEGRITY_DETAIL,
    ) from None


def _normalized_absolute_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return Path(os.path.normpath(value))


def _rebase_session_paths(
    value: Any,
    original_session_dir: Path,
    current_session_dir: Path,
) -> Any:
    """Rebase absolute config paths contained by the original session root."""
    if isinstance(value, dict):
        return {
            key: _rebase_session_paths(
                item,
                original_session_dir,
                current_session_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_session_paths(
                item,
                original_session_dir,
                current_session_dir,
            )
            for item in value
        ]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value

    normalized_path = Path(os.path.normpath(value))
    try:
        relative_path = normalized_path.relative_to(original_session_dir)
    except ValueError:
        return value
    return str(current_session_dir / relative_path)


def _load_rebased_pipeline_config(
    config_path: Path,
    session_id: str,
    current_session_dir: Path,
) -> dict[str, Any]:
    """Load a stored config and safely move its session-local paths."""
    try:
        with config_path.open(encoding="utf-8") as config_file:
            pipeline_config = yaml.safe_load(config_file)
    except (OSError, UnicodeError, yaml.YAMLError):
        log_durable_failure(
            logger,
            "pipeline_config_read_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        ) from None

    try:
        if not isinstance(pipeline_config, dict):
            raise ValueError("pipeline config must be a mapping")
        project = pipeline_config.get("project")
        input_config = pipeline_config.get("input")
        if not isinstance(project, dict) or not isinstance(input_config, dict):
            raise ValueError("pipeline config is missing path sections")

        configured_session_id = project.get("session_id")
        if configured_session_id not in (None, session_id):
            raise ValueError("pipeline config belongs to another session")

        original_working_dir = _normalized_absolute_path(
            project.get("working_dir"),
            "project.working_dir",
        )
        if original_working_dir.name != "cache":
            raise ValueError("project.working_dir is not session-local")
        original_session_dir = original_working_dir.parent
        if original_session_dir.name != session_id:
            raise ValueError("project.working_dir belongs to another session")

        input_usd_path = _normalized_absolute_path(
            input_config.get("usd_path"),
            "input.usd_path",
        )
        input_relative_path = input_usd_path.relative_to(original_session_dir)
        if not input_relative_path.parts or input_relative_path.parts[0] != "input":
            raise ValueError("input.usd_path is not session-local")
    except ValueError as exc:
        logger.warning(
            "Rejected original pipeline config for %s: %s",
            session_id[:8],
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        ) from exc

    local_session_dir = Path(os.path.abspath(current_session_dir))
    rebased_config = _rebase_session_paths(
        pipeline_config,
        original_session_dir,
        local_session_dir,
    )
    if not isinstance(rebased_config, dict):  # pragma: no cover - guarded above
        raise AssertionError("rebased pipeline config must remain a mapping")
    return rebased_config


def _session_contained_path(
    value: object,
    session_dir: Path,
    field_name: str,
) -> Path:
    """Return one absolute config path only when it stays inside the session."""
    if isinstance(value, Path):
        value = str(value)
    try:
        candidate = _normalized_absolute_path(value, field_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        ) from exc

    session_root = Path(os.path.abspath(session_dir))
    try:
        candidate.relative_to(session_root)
        candidate.resolve(strict=False).relative_to(session_root.resolve())
    except ValueError as exc:
        logger.warning(
            "Rejected regeneration path outside session for %s: %s",
            field_name,
            candidate,
        )
        raise HTTPException(
            status_code=400,
            detail="Original config contains a path outside the session",
        ) from exc
    return candidate


def _step_config(pipeline_config: dict[str, Any], step_name: str) -> dict[str, Any]:
    steps = pipeline_config.get("steps", {})
    if not isinstance(steps, dict):
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        )
    step_config = steps.get(step_name, {})
    if not isinstance(step_config, dict):
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        )
    return step_config


def _require_durable_path(
    path: Path,
    *,
    session_dir: Path,
    dataset_dir: Path,
    predictions_dir: Path,
    field_name: str,
) -> str:
    """Return the durable closure that owns a required session-local path."""
    input_dir = session_dir / "input"
    for prefix, root in (
        (_PIPELINE_INPUT_PREFIX, input_dir),
        (_PIPELINE_DATASET_PREFIX, dataset_dir),
        (_PIPELINE_PREDICTIONS_PREFIX, predictions_dir),
    ):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return prefix

    logger.warning(
        "Regeneration field %s is outside a durable artifact closure: %s",
        field_name,
        path,
    )
    raise HTTPException(
        status_code=409,
        detail=(
            "Regeneration prerequisites are unavailable because the original "
            "config references a non-durable session path"
        ),
    )


def _required_config_path(
    step_config: dict[str, Any],
    key: str,
    *,
    fallback: Path | None,
    session_dir: Path,
    field_name: str,
) -> Path:
    value = step_config.get(key)
    if value is None:
        if fallback is None:
            raise HTTPException(
                status_code=409,
                detail=f"Regeneration prerequisite is not configured: {field_name}",
            )
        value = str(fallback)
    return _session_contained_path(value, session_dir, field_name)


def _regeneration_prerequisite_plan(
    pipeline_config: dict[str, Any],
    only_steps: list[str],
    session_dir: Path,
) -> tuple[set[str], bool, set[Path], set[Path], set[Path]]:
    """Build the durable inputs needed after accounting for selected producers."""
    project = pipeline_config.get("project", {})
    if not isinstance(project, dict):
        raise HTTPException(
            status_code=400,
            detail="Original config is invalid for session",
        )
    working_dir = _session_contained_path(
        project.get("working_dir"),
        session_dir,
        "project.working_dir",
    )
    dataset_dir = working_dir / "dataset"
    predictions_dir = working_dir / "predictions"
    canonical_dataset = dataset_dir / "dataset.jsonl"
    canonical_predictions = predictions_dir / "predictions.jsonl"
    requested = set(only_steps)

    prefixes: set[str] = set()
    require_render_dataset = False
    dataset_paths: set[Path] = set()
    prediction_paths: set[Path] = set()
    additional_paths: set[Path] = set()

    if (
        "build_dataset_prepare_dataset" in requested
        and "build_dataset_usd" not in requested
    ):
        prefixes.add(_PIPELINE_DATASET_PREFIX)
        require_render_dataset = True

    if "predict" in requested and "build_dataset_prepare_dataset" not in requested:
        prefixes.add(_PIPELINE_DATASET_PREFIX)
        dataset_paths.add(canonical_dataset)

    if "consistency_pass" in requested and "predict" not in requested:
        consistency_config = _step_config(pipeline_config, "consistency_pass")
        predictions_path = _required_config_path(
            consistency_config,
            "predictions_path",
            fallback=canonical_predictions,
            session_dir=session_dir,
            field_name="steps.consistency_pass.predictions_path",
        )
        prefixes.add(
            _require_durable_path(
                predictions_path,
                session_dir=session_dir,
                dataset_dir=dataset_dir,
                predictions_dir=predictions_dir,
                field_name="steps.consistency_pass.predictions_path",
            )
        )
        prediction_paths.add(predictions_path)

    if "infer_articulation_candidates" in requested:
        candidate_config = _step_config(
            pipeline_config,
            "infer_articulation_candidates",
        )
        if not {"predict", "consistency_pass"}.intersection(requested):
            predictions_path = _required_config_path(
                candidate_config,
                "predictions_path",
                fallback=None,
                session_dir=session_dir,
                field_name="steps.infer_articulation_candidates.predictions_path",
            )
            prefixes.add(
                _require_durable_path(
                    predictions_path,
                    session_dir=session_dir,
                    dataset_dir=dataset_dir,
                    predictions_dir=predictions_dir,
                    field_name=("steps.infer_articulation_candidates.predictions_path"),
                )
            )
            prediction_paths.add(predictions_path)

        configured_dataset = candidate_config.get("dataset_path")
        adjudication = candidate_config.get("adjudication", {})
        require_source_images = isinstance(adjudication, dict) and bool(
            adjudication.get("require_source_images", False)
        )
        if configured_dataset is None and require_source_images:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Regeneration prerequisite is not configured: "
                    "steps.infer_articulation_candidates.dataset_path"
                ),
            )
        if configured_dataset is not None:
            dataset_path = _session_contained_path(
                configured_dataset,
                session_dir,
                "steps.infer_articulation_candidates.dataset_path",
            )
            produced_by_request = (
                "build_dataset_prepare_dataset" in requested
                and dataset_path == canonical_dataset
            )
            if not produced_by_request:
                prefixes.add(
                    _require_durable_path(
                        dataset_path,
                        session_dir=session_dir,
                        dataset_dir=dataset_dir,
                        predictions_dir=predictions_dir,
                        field_name=("steps.infer_articulation_candidates.dataset_path"),
                    )
                )
                dataset_paths.add(dataset_path)

        prim_metadata = candidate_config.get("prim_metadata_path")
        if prim_metadata is not None:
            prim_metadata_path = _session_contained_path(
                prim_metadata,
                session_dir,
                "steps.infer_articulation_candidates.prim_metadata_path",
            )
            prefixes.add(
                _require_durable_path(
                    prim_metadata_path,
                    session_dir=session_dir,
                    dataset_dir=dataset_dir,
                    predictions_dir=predictions_dir,
                    field_name=(
                        "steps.infer_articulation_candidates.prim_metadata_path"
                    ),
                )
            )
            additional_paths.add(prim_metadata_path)

    return (
        prefixes,
        require_render_dataset,
        dataset_paths,
        prediction_paths,
        additional_paths,
    )


def _relative_prerequisite_name(path: Path, session_dir: Path) -> str:
    try:
        return path.relative_to(session_dir).as_posix()
    except ValueError:  # pragma: no cover - containment is checked earlier
        return path.name


def _require_prerequisite_file(path: Path, session_dir: Path) -> None:
    if path.is_file():
        return
    raise HTTPException(
        status_code=409,
        detail=(
            "Regeneration prerequisite is unavailable: "
            f"{_relative_prerequisite_name(path, session_dir)}"
        ),
    )


def _iter_jsonl_rows(path: Path, session_dir: Path) -> Iterator[dict[str, Any]]:
    _require_prerequisite_file(path, session_dir)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row {line_number} is not an object")
                yield row
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid regeneration prerequisite %s", path, exc_info=True)
        raise HTTPException(
            status_code=409,
            detail=(
                "Regeneration prerequisite is invalid: "
                f"{_relative_prerequisite_name(path, session_dir)}"
            ),
        ) from exc


def _validate_referenced_file(
    raw_path: object,
    *,
    base_dir: Path,
    session_dir: Path,
    field_name: str,
) -> None:
    if not isinstance(raw_path, str) or not raw_path:
        raise HTTPException(
            status_code=409,
            detail=f"Regeneration prerequisite contains an invalid {field_name}",
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    path = _session_contained_path(path, session_dir, field_name)
    _require_prerequisite_file(path, session_dir)


def _validate_render_dataset(dataset_dir: Path, session_dir: Path) -> None:
    usd_dir = dataset_dir / "usd"
    _require_prerequisite_file(usd_dir / "dataset.json", session_dir)
    prims_path = usd_dir / "prims.jsonl"
    for row in _iter_jsonl_rows(prims_path, session_dir):
        renders = row.get("renders", [])
        if not isinstance(renders, list):
            raise HTTPException(
                status_code=409,
                detail="Regeneration prerequisite contains invalid render metadata",
            )
        for render in renders:
            if not isinstance(render, dict) or "path" not in render:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Regeneration prerequisite contains invalid render metadata"
                    ),
                )
            _validate_referenced_file(
                render["path"],
                base_dir=usd_dir,
                session_dir=session_dir,
                field_name="render path",
            )


def _dataset_image_paths(row: dict[str, Any]) -> list[object]:
    paths: list[object] = []
    images = row.get("images")
    if isinstance(images, list):
        for image in images:
            paths.append(image.get("path") if isinstance(image, dict) else image)
    image_path = row.get("image_path")
    if image_path is not None:
        paths.append(image_path)
    media = row.get("media")
    if isinstance(media, dict):
        media_images = media.get("images")
        if isinstance(media_images, list):
            for image in media_images:
                paths.append(image.get("path") if isinstance(image, dict) else image)
    return paths


def _validate_prepared_dataset(dataset_path: Path, session_dir: Path) -> None:
    for row in _iter_jsonl_rows(dataset_path, session_dir):
        for image_path in _dataset_image_paths(row):
            _validate_referenced_file(
                image_path,
                base_dir=dataset_path.parent,
                session_dir=session_dir,
                field_name="dataset image path",
            )


def _validate_jsonl(path: Path, session_dir: Path) -> None:
    for _ in _iter_jsonl_rows(path, session_dir):
        pass


def _replace_cache_namespace_from_publication(
    publication_dir: Path,
    cache_dir: Path,
) -> None:
    """Copy an immutable publication into a cache through one held session root."""

    session_dir = cache_dir.parent.parent
    try:
        publication_key = (
            publication_dir.absolute().relative_to(session_dir.absolute()).as_posix()
        )
        cache_parent_key = (
            cache_dir.parent.absolute().relative_to(session_dir.absolute()).as_posix()
        )
    except ValueError:
        raise RuntimeError("Cache paths must share one session root") from None
    if not publication_key or not cache_parent_key:
        raise RuntimeError("Cache paths must be descendants of the session root")

    cache_name = cache_dir.name
    staging_name = f".{cache_name}.{uuid.uuid4().hex}.staging"
    backup_name = f".{cache_name}.{uuid.uuid4().hex}.backup"
    backup_created = False
    swap_complete = False

    def remove_entry(parent_descriptor: int, name: str) -> None:
        try:
            metadata = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(name, dir_fd=parent_descriptor)
        else:
            os.unlink(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)

    try:
        with open_confined_directory(session_dir) as session_descriptor:
            with open_confined_directory_at(
                session_descriptor,
                publication_key,
            ) as publication_descriptor:
                with open_confined_directory_at(
                    session_descriptor,
                    cache_parent_key,
                    create=True,
                ) as cache_parent_descriptor:
                    os.mkdir(staging_name, dir_fd=cache_parent_descriptor)
                    try:
                        with open_confined_directory_at(
                            cache_parent_descriptor,
                            staging_name,
                        ) as staging_descriptor:
                            for artifact in iter_open_regular_files(
                                publication_descriptor
                            ):
                                copy_open_file_to_confined(
                                    staging_descriptor,
                                    artifact.relative_key,
                                    artifact.stream,
                                    artifact.metadata,
                                    overwrite=False,
                                )

                        try:
                            cache_metadata = os.stat(
                                cache_name,
                                dir_fd=cache_parent_descriptor,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            pass
                        else:
                            if not stat.S_ISDIR(cache_metadata.st_mode) or stat.S_ISLNK(
                                cache_metadata.st_mode
                            ):
                                raise RuntimeError(
                                    "Mutable cache target is not a directory: "
                                    f"{cache_dir}"
                                )
                            os.rename(
                                cache_name,
                                backup_name,
                                src_dir_fd=cache_parent_descriptor,
                                dst_dir_fd=cache_parent_descriptor,
                            )
                            backup_created = True

                        try:
                            os.rename(
                                staging_name,
                                cache_name,
                                src_dir_fd=cache_parent_descriptor,
                                dst_dir_fd=cache_parent_descriptor,
                            )
                            swap_complete = True
                            os.fsync(cache_parent_descriptor)
                        except BaseException:
                            if backup_created:
                                os.rename(
                                    backup_name,
                                    cache_name,
                                    src_dir_fd=cache_parent_descriptor,
                                    dst_dir_fd=cache_parent_descriptor,
                                )
                                backup_created = False
                                os.fsync(cache_parent_descriptor)
                            raise
                    finally:
                        remove_entry(cache_parent_descriptor, staging_name)
                        if swap_complete and backup_created:
                            remove_entry(cache_parent_descriptor, backup_name)
    except (ArtifactPathError, FileNotFoundError) as exc:
        raise RuntimeError("Cache publication path is not safe") from exc


def _remove_cache_namespace(cache_dir: Path) -> None:
    session_dir = cache_dir.parent.parent
    try:
        cache_parent_key = (
            cache_dir.parent.absolute().relative_to(session_dir.absolute()).as_posix()
        )
    except ValueError:
        raise RuntimeError("Cache path must be inside its session root") from None
    try:
        with open_confined_directory(session_dir) as session_descriptor:
            with open_confined_directory_at(
                session_descriptor,
                cache_parent_key,
                create=True,
            ) as cache_parent_descriptor:
                try:
                    metadata = os.stat(
                        cache_dir.name,
                        dir_fd=cache_parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    return
                if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                ):
                    shutil.rmtree(cache_dir.name, dir_fd=cache_parent_descriptor)
                else:
                    os.unlink(cache_dir.name, dir_fd=cache_parent_descriptor)
                os.fsync(cache_parent_descriptor)
    except FileNotFoundError:
        return
    except ArtifactPathError as exc:
        raise RuntimeError("Cache namespace path is not safe") from exc


async def _restore_cache_namespace(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    namespace: str,
) -> None:
    """Restore one bound immutable snapshot, with legacy stable-key fallback."""

    metadata = await manager.get_session_metadata(session_id)
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=404, detail="Session not found")
    bindings = parse_cache_publications(metadata)
    if bindings is None:
        await manager.sync_from_store(
            session_id,
            prefix=f"cache/{namespace}/",
            overwrite=True,
        )
        return

    run_id = bindings.get(namespace)
    if run_id is None:
        _remove_cache_namespace(session_dir / "cache" / namespace)
        raise HTTPException(
            status_code=409,
            detail=(
                "Regeneration prerequisites are unavailable because the selected "
                f"run has no {namespace} cache publication"
            ),
        )
    publication_prefix = cache_publication_prefix(run_id, namespace)
    await manager.sync_from_store(
        session_id,
        prefix=publication_prefix,
        overwrite=True,
    )
    publication_dir = cache_publication_path(session_dir, run_id, namespace)
    try:
        await asyncio.to_thread(
            _replace_cache_namespace_from_publication,
            publication_dir,
            session_dir / "cache" / namespace,
        )
    except (FileNotFoundError, OSError, RuntimeError):
        log_durable_failure(
            logger,
            "cache_publication_materialization_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "Regeneration prerequisites are unavailable because the selected "
                f"{namespace} cache publication is invalid"
            ),
        ) from None


async def _restore_regeneration_prerequisites(
    manager: SessionManager,
    session_id: str,
    session_dir: Path,
    pipeline_config: dict[str, Any],
    only_steps: list[str],
) -> None:
    """Restore and validate all cached inputs before accepting regeneration."""
    (
        prefixes,
        require_render_dataset,
        dataset_paths,
        prediction_paths,
        additional_paths,
    ) = _regeneration_prerequisite_plan(
        pipeline_config,
        only_steps,
        session_dir,
    )
    metadata = await manager.get_session_metadata(session_id)
    if isinstance(metadata, dict) and parse_cache_publications(metadata) is None:
        prefixes.update(_CACHE_PREFIX_NAMESPACES)

    try:
        for prefix in sorted(prefixes):
            namespace = _CACHE_PREFIX_NAMESPACES.get(prefix)
            if namespace is None:
                await manager.sync_from_store(
                    session_id,
                    prefix=prefix,
                    overwrite=True,
                )
            else:
                await _restore_cache_namespace(
                    manager,
                    session_id,
                    session_dir,
                    namespace,
                )
    except HTTPException:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "regeneration_prerequisite_restore_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Regeneration prerequisites could not be restored",
        ) from None

    working_dir = _session_contained_path(
        pipeline_config["project"]["working_dir"],
        session_dir,
        "project.working_dir",
    )
    if require_render_dataset:
        await asyncio.to_thread(
            _validate_render_dataset,
            working_dir / "dataset",
            session_dir,
        )
    for dataset_path in dataset_paths:
        await asyncio.to_thread(
            _validate_prepared_dataset,
            dataset_path,
            session_dir,
        )
    for predictions_path in prediction_paths:
        await asyncio.to_thread(
            _validate_jsonl,
            predictions_path,
            session_dir,
        )
    for additional_path in additional_paths:
        await asyncio.to_thread(
            _require_prerequisite_file,
            additional_path,
            session_dir,
        )


@router.post(
    "",
    response_model=PipelineRunCreated,
    status_code=202,
    responses=S3_INPUT_ERROR_RESPONSES,
)
async def create_pipeline(
    usd_file: UploadFile = File(
        None,
        description=(
            "Lowest-priority USD source. Used only when session_id and s3_uri "
            "are both omitted."
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
            "bucket name must be listed in JA_S3_ALLOWED_BUCKETS; an empty allowlist "
            "rejects all selected client S3 inputs before session-manager or S3 I/O. "
            "When selected, usd_file is ignored."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional)",
    ),
    render_backend: str = Form(
        default="",
        description=(
            "Rendering backend: 'remote' (default, via RENDER_ENDPOINT), "
            "'warp' (local CUDA), 'ovrtx' (local Vulkan), or 'mock' "
            "(deterministic CPU-only test images)"
        ),
        json_schema_extra={"enum": [*RENDERING_BACKEND_NAMES, ""]},
    ),
    apply_joint_rigger: bool = Form(
        default=False,
        description="Enable the opt-in Joint Rigger apply step.",
    ),
    joint_rigger_adapter: str = Form(
        default="",
        description=(
            "Joint Rigger adapter to use when apply_joint_rigger is true: "
            "'owned_core', 'mock', or 'usd_joint_rigger'. Defaults to "
            "'owned_core' when omitted."
        ),
    ),
    joint_rigger_on_missing_dependency: str = Form(
        default="",
        description="Missing dependency policy for the Joint Rigger adapter: 'skip' or 'block'.",
    ),
    joint_rigger_on_unready_candidates: str = Form(
        default="",
        description="Candidate readiness policy for the Joint Rigger adapter: 'warn' or 'block'.",
    ),
    joint_rigger_template: str = Form(
        default="",
        description="Template name for the provisional USD Joint Rigger.",
    ),
    joint_rigger_apply_masses: bool | None = Form(
        default=None,
        description=(
            "Whether the Joint Rigger should author masses when supported. "
            "owned_core requires false."
        ),
    ),
    joint_rigger_apply_collision: bool | None = Form(
        default=None,
        description=(
            "Whether the Joint Rigger should author collision when supported. "
            "owned_core requires false."
        ),
    ),
) -> PipelineRunCreated:
    """Create and execute a joint agent pipeline from at least one source.

    When multiple sources are supplied, selection follows the legacy precedence
    ``session_id`` > ``s3_uri`` > ``usd_file`` and lower-priority fields are
    ignored. A selected S3 source is admitted only when its exact bucket name is
    listed in ``JA_S3_ALLOWED_BUCKETS``; an empty allowlist rejects every client
    S3 input before session-manager or S3 I/O.
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
    joint_rigger_step_options = _joint_rigger_step_options(
        apply_joint_rigger=apply_joint_rigger,
        joint_rigger_adapter=joint_rigger_adapter,
        joint_rigger_on_missing_dependency=joint_rigger_on_missing_dependency,
        joint_rigger_on_unready_candidates=joint_rigger_on_unready_candidates,
        joint_rigger_template=joint_rigger_template,
        joint_rigger_apply_masses=joint_rigger_apply_masses,
        joint_rigger_apply_collision=joint_rigger_apply_collision,
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
    reused_existing_session = False
    session_created_here = False
    existing_metadata: dict[str, Any] | None = None

    if session_id:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        reused_existing_session = True
        session_dir = manager.get_session_dir(session_id)
        existing_metadata = await manager.get_session_metadata(session_id)

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
            total_bytes = await _stream_copy(
                selected_usd_file,
                usd_path,
                max_bytes=_configured_upload_limit_bytes(),
            )
            size_mb = total_bytes / (1024 * 1024)

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

    run_id = uuid.uuid4().hex
    try:
        job_registry = await _reserve_run_admission(manager, session_id, run_id)
    except BaseException:
        await _cleanup_rejected_pipeline_session(
            manager,
            session_id,
            session_created_here=session_created_here,
        )
        raise

    claim_guard = _RunClaimGuard(manager, session_id, run_id)
    registered = False
    try:
        input_usd_path = _find_input_usd(session_dir)
        if not input_usd_path:
            # May be on a different instance — pull input/ from store and retry
            pulled = await _await_with_run_claim(
                manager.sync_from_store(
                    session_id,
                    prefix="input/",
                    overwrite=True,
                ),
                claim_guard,
            )
            if pulled > 0:
                logger.info(
                    f"Pulled {pulled} input file(s) from store for session {session_id[:8]}"
                )
            input_usd_path = _find_input_usd(session_dir)
        if not input_usd_path:
            raise HTTPException(
                status_code=400, detail="Input USD not found for session"
            )

        pipeline_config = build_default_pipeline_config(
            session_id=session_id,
            usd_path=str(input_usd_path),
            working_dir=str(session_dir / "cache"),
            user_prompt=user_prompt_text,
            render_backend=render_backend_text,
            vlm_backend=config.vlm_backend,
            vlm_model=config.vlm_model,
            llm_backend=config.llm_backend,
            llm_model=config.llm_model,
            joint_rigger_output_suffix=(
                ".usdz"
                if joint_rigger_step_options
                and joint_rigger_step_options.get("adapter") == "owned_core"
                else ".usd"
            ),
        )
        _apply_render_request_limit(pipeline_config)
        _apply_prediction_request_limit(pipeline_config)
        _apply_joint_rigger_step_options(pipeline_config, joint_rigger_step_options)

        try:
            await aensure_no_inline_secrets(
                pipeline_config,
                context="joint pipeline configuration",
            )
        except InlineSecretError:
            raise HTTPException(
                status_code=400,
                detail=_INVALID_PIPELINE_CONFIG_DETAIL,
            ) from None
        publication_sha256 = await _await_with_run_claim(
            _persist_pipeline_config(
                manager,
                session_id,
                run_id,
                pipeline_config,
                input_usd_path,
            ),
            claim_guard,
        )

        existing_config = (existing_metadata or {}).get("config") or {}
        if not isinstance(existing_config, dict):
            existing_config = {}
        source_s3_uri = (
            existing_config.get("s3_uri")
            if reused_existing_session
            else selected_s3_uri
        )
        has_usd_upload = (
            bool(existing_config.get("has_usd_upload"))
            if reused_existing_session
            else bool(selected_usd_file and selected_usd_file.filename)
        )
        accepted = await _await_with_run_claim(
            manager.update_session_for_run(
                session_id,
                run_id,
                {
                    "status": "pending",
                    "current_step": None,
                    "results": {},
                    "error": None,
                    "failed_step": None,
                    PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
                    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: publication_sha256,
                    "config": {
                        "project_name": pipeline_config.get("project", {}).get(
                            "name", ""
                        ),
                        "usd_path": str(input_usd_path),
                        "has_usd_upload": has_usd_upload,
                        "s3_uri": source_s3_uri,
                        "user_prompt": user_prompt_text,
                    },
                },
            ),
            claim_guard,
        )
        if not accepted:
            raise HTTPException(status_code=409, detail="Pipeline run was superseded")

        await _await_with_run_claim(
            get_event_bus().seed_pending_session(session_id),
            claim_guard,
        )

        async def finish_run() -> None:
            await finalize_pipeline_run(manager, session_id, run_id)

        try:
            await job_registry.register(
                session_id,
                execute_pipeline_async(
                    session_id=session_id,
                    run_id=run_id,
                    config_dict=pipeline_config,
                    session_manager=manager,
                ),
                run_id=run_id,
                liveness_guard=claim_guard,
                on_finish=finish_run,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        registered = True

        logger.info(f"Pipeline registered for session {session_id}")

        return PipelineRunCreated(
            session_id=session_id,
            run_id=run_id,
            status="pending",
            message="Pipeline queued for execution",
            estimated_duration_minutes=15,
        )
    finally:
        try:
            try:
                if not registered:
                    await claim_guard.stop()
                    await manager.release_run(session_id, run_id)
            finally:
                await job_registry.release_admission(session_id, run_id)
        finally:
            await _cleanup_rejected_pipeline_session(
                manager,
                session_id,
                session_created_here=session_created_here and not registered,
            )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_pipeline_status(session_id: str) -> PipelineStatus:
    """Get pipeline execution status with detailed progress.

    Reads from in-memory event bus state for fast, real-time accuracy.
    Falls back to store-based SessionManager for completed/cross-instance sessions.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    # Only a locally owned run may use the in-memory fast path. A different
    # instance can accept a regeneration while this instance retains an older
    # terminal snapshot, so durable metadata is authoritative without ownership.
    metadata: dict[str, Any]
    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is not None and not get_job_registry().is_running(session_id):
        snapshot = None

    if snapshot:
        metadata = snapshot
        preview_images = snapshot.get("preview_images", [])
    else:
        # Fall back to store (works cross-instance)
        stored_metadata = await manager.get_session_metadata(session_id)
        if not stored_metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        metadata = stored_metadata
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
async def get_pipeline_results(session_id: str) -> PipelineResults | PipelineError:
    """Get terminal pipeline results or error details."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        return PipelineResults(
            session_id=session_id,
            status=status,
            stats=metadata.get("results", {}),
            download_urls=_result_download_urls(session_id, metadata),
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )

    elif status in {"failed", "cancelled"}:
        default_error = (
            "Pipeline run was cancelled" if status == "cancelled" else "Unknown error"
        )
        default_failed_step = "cancelled" if status == "cancelled" else "unknown"
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error") or default_error,
            failed_step=metadata.get("failed_step") or default_failed_step,
            completed_steps=[
                step["name"]
                for step in metadata.get("completed_steps", [])
                if isinstance(step, dict) and isinstance(step.get("name"), str)
            ],
            partial_results=metadata.get("partial_results"),
        )

    else:
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still {status}. Check status endpoint for progress.",
        )


@router.post(
    "/{session_id}/cancel",
    response_model=PipelineCancellationAccepted,
)
async def cancel_pipeline(
    session_id: str,
    run_id: str = Query(
        ...,
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
        description="Generation token returned when the pipeline run was accepted",
    ),
) -> PipelineCancellationAccepted:
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

    # Write cancel signal to store (visible to all instances)
    active_run_id = metadata.get(ACTIVE_RUN_ID_FIELD)
    if active_run_id != run_id or not await manager.request_cancellation(
        session_id,
        run_id,
    ):
        raise HTTPException(
            status_code=409,
            detail="Pipeline run changed before cancellation could be accepted",
        )

    # Also try local cancellation (fast path if this is the executing instance)
    if job_registry.is_running(session_id):
        await job_registry.cancel(session_id, run_id=run_id)

    return PipelineCancellationAccepted(
        session_id=session_id,
        run_id=run_id,
        status="cancelling",
        message="Pipeline cancellation requested",
    )


@router.get("/{session_id}/events")
async def stream_progress_events(session_id: str):
    """Stream real-time progress events via Server-Sent Events (SSE).

    Only works when connected to the instance running the pipeline.
    For cross-instance progress, use GET /pipeline/{session_id}/status (polling).
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is not None and not get_job_registry().is_running(session_id):
        snapshot = None
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


@router.post(
    "/{session_id}/regenerate",
    response_model=PipelineRunCreated,
    status_code=202,
)
async def regenerate_pipeline(
    session_id: str,
    request: RegenerateRequest,
) -> PipelineRunCreated:
    """Regenerate specific pipeline steps from cached data."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    # The leased claim is authoritative; crash-stale running metadata must not
    # prevent a new owner from reclaiming the session after lease expiry.
    run_id = uuid.uuid4().hex
    job_registry = await _reserve_run_admission(manager, session_id, run_id)

    claim_guard = _RunClaimGuard(manager, session_id, run_id)
    registered = False
    try:
        session_dir = manager.get_session_dir(session_id)
        config_path = session_dir / "input" / "config.yaml"
        await _await_with_run_claim(
            _restore_pipeline_config(manager, session_id, config_path),
            claim_guard,
        )
        pipeline_config = _load_rebased_pipeline_config(
            config_path,
            session_id,
            session_dir,
        )
        only_steps = [s.value for s in request.steps]
        await _await_with_run_claim(
            _restore_regeneration_prerequisites(
                manager,
                session_id,
                session_dir,
                pipeline_config,
                only_steps,
            ),
            claim_guard,
        )
        _apply_render_request_limit(pipeline_config)
        _apply_prediction_request_limit(pipeline_config)

        if request.user_prompt is not None:
            steps_section = pipeline_config.get("steps", {})
            prepare_dataset = steps_section.get("build_dataset_prepare_dataset", {})
            prompts = prepare_dataset.get("prompts", {})
            prompts["user"] = request.user_prompt
            prepare_dataset["prompts"] = prompts
            steps_section["build_dataset_prepare_dataset"] = prepare_dataset
            pipeline_config["steps"] = steps_section

        # _load_rebased_pipeline_config has already validated this mapping.
        input_config = pipeline_config["input"]
        input_usd_path = _session_contained_path(
            input_config.get("usd_path"),
            session_dir,
            "input.usd_path",
        )
        publication_sha256 = await _await_with_run_claim(
            _persist_pipeline_config(
                manager,
                session_id,
                run_id,
                pipeline_config,
                input_usd_path,
            ),
            claim_guard,
        )

        accepted = await _await_with_run_claim(
            manager.update_session_for_run(
                session_id,
                run_id,
                {
                    "status": "pending",
                    "current_step": None,
                    "can_cancel": True,
                    "results": {},
                    "error": None,
                    "failed_step": None,
                    PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
                    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: publication_sha256,
                },
            ),
            claim_guard,
        )
        if not accepted:
            raise HTTPException(status_code=409, detail="Pipeline run was superseded")

        await _await_with_run_claim(
            get_event_bus().seed_pending_session(session_id),
            claim_guard,
        )

        async def finish_run() -> None:
            await finalize_pipeline_run(manager, session_id, run_id)

        try:
            await job_registry.register(
                session_id,
                execute_pipeline_async(
                    session_id=session_id,
                    run_id=run_id,
                    config_dict=pipeline_config,
                    session_manager=manager,
                    only_steps=only_steps,
                ),
                run_id=run_id,
                liveness_guard=claim_guard,
                on_finish=finish_run,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        registered = True

        logger.info(f"Pipeline regeneration registered for session {session_id}")

        return PipelineRunCreated(
            session_id=session_id,
            run_id=run_id,
            status="pending",
            message=f"Regenerating steps: {', '.join(s.value for s in request.steps)}",
        )
    finally:
        try:
            if not registered:
                await claim_guard.stop()
                await manager.release_run(session_id, run_id)
        finally:
            await job_registry.release_admission(session_id, run_id)


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
