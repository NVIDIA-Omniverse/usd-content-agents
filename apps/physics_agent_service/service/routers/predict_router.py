# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict API endpoints — first-class predict workflow.

Distinct route group from ``/pipeline``:

* ``POST /predict`` — start an async prediction job (Mode A or Mode B).
* ``GET  /predict/{session_id}/status`` — current status (mirrors /pipeline).
* ``GET  /predict/{session_id}/results`` — completed predict results.
* ``GET  /predict/{session_id}/events`` — SSE progress stream.
* ``POST /predict/{session_id}/cancel`` — cancel an in-flight predict job.

The predict route is intentionally NOT a thin alias for ``/pipeline``: it
runs a prediction-only workflow that auto-detects whether the session
has a prepared dataset whose referenced images resolve (Mode A → just
predict) or needs the minimum upstream prep first (Mode B → optimize_usd
→ identify_asset → build_dataset_usd → build_dataset_prepare_dataset →
predict). The
``/pipeline`` workflow remains unchanged and continues to be the right
entry point for the full classify/apply flow.

Reuses shared infra (session manager, job registry, event bus, SSE,
cancellation, artifact storage) — only the route definitions and request /
response schemas live here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.api.defaults import build_default_pipeline_config
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
    PIPELINE_CONFIG_WRITE_FAILED_DETAIL,
    build_and_validate_pipeline_config,
    build_and_write_pipeline_config,
)
from ..models.responses import (
    PREDICT_INPUT_ERROR_RESPONSES,
    PipelineError,
    PipelineStatus,
    PredictResults,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..session.manager import SessionManager
from ..workers.predict_executor import (
    _dataset_jsonl_has_resolvable_images,
    execute_predict_async,
)

logger = logging.getLogger(__name__)

# Distinct prefix and tag — clients should treat /predict as its own route group,
# parallel to /pipeline and #36's planned /tune.
router = APIRouter(prefix="/predict", tags=["predict"])

# Global session manager (set by main app — same instance as /pipeline).
session_manager: SessionManager | None = None

_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
_DATASET_STAGE_FAILED_DETAIL = "Failed to stage dataset into session cache"
_PREDICT_START_FAILED_DETAIL = "Failed to start predict job"


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


def _validate_mode_b_options(
    *,
    render_backend: str,
    optimize_usd: bool,
    enable_deinstance: bool,
    enable_split: bool,
    enable_deduplicate: bool,
) -> str | None:
    """Validate request-only Mode B options before starting expensive work."""
    if optimize_usd and not any([enable_deinstance, enable_split, enable_deduplicate]):
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one optimization operation must be enabled when "
                "optimize_usd is true (enable_deinstance, enable_split, or "
                "enable_deduplicate)."
            ),
        )

    render_backend_text = render_backend.strip() if render_backend else None
    if render_backend_text is None:
        return None

    try:
        return validate_rendering_backend_name(render_backend_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    chunk_size: int = 2 * 1024 * 1024,
) -> int:
    """Stream upload file to disk in chunks; abort early on size overrun.

    Raises HTTPException(413) as soon as the running byte total passes
    ``max_bytes`` so an oversized upload cannot fill the session volume.
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
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: exceeds {max_bytes // (1024 * 1024)}MB"
                        ),
                    )
                f.write(data)
        return total_bytes
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


def _copy_to_sibling_temp(source: Path, target: Path) -> Path:
    """Copy ``source`` to an fsynced sibling without touching ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        shutil.copyfile(source, temporary_path)
        with temporary_path.open("rb") as staged_file:
            os.fsync(staged_file.fileno())
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass
class _FileSnapshot:
    """Bounded-memory, anonymous on-disk snapshot of one prior input file."""

    target: Path
    existed: bool
    backup: BinaryIO | None = None

    @classmethod
    def capture(cls, target: Path) -> _FileSnapshot:
        """Capture a regular file without retaining its contents on the heap."""
        if not target.exists():
            return cls(target=target, existed=False)
        if target.is_symlink() or not target.is_file():
            raise OSError("Predict input snapshot source is not a regular file")

        backup = tempfile.TemporaryFile(mode="w+b", dir=target.parent)
        try:
            with target.open("rb") as source:
                shutil.copyfileobj(source, backup, length=2 * 1024 * 1024)
            backup.flush()
            os.fsync(backup.fileno())
            backup.seek(0)
            return cls(target=target, existed=True, backup=backup)
        except Exception:
            backup.close()
            raise

    def restore(self) -> None:
        """Atomically restore this snapshot, or remove a newly created file."""
        if not self.existed:
            self.target.unlink(missing_ok=True)
            return
        if self.backup is None:
            raise RuntimeError("Predict input snapshot is unavailable")

        self.target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            self.backup.seek(0)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.target.parent,
                prefix=f".{self.target.name}.rollback.",
                suffix=".tmp",
                delete=False,
            ) as rollback_file:
                temporary_path = Path(rollback_file.name)
                shutil.copyfileobj(
                    self.backup,
                    rollback_file,
                    length=2 * 1024 * 1024,
                )
                rollback_file.flush()
                os.fsync(rollback_file.fileno())
            os.replace(temporary_path, self.target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Release the anonymous backup file."""
        if self.backup is not None:
            try:
                self.backup.close()
            except OSError:
                pass
            finally:
                self.backup = None


@dataclass
class _PredictInputsTransaction:
    """Rollback handle retained until worker registration succeeds."""

    snapshots: tuple[_FileSnapshot, ...]
    active: bool = True

    def commit(self) -> None:
        """Make the newly published inputs authoritative."""
        self.active = False
        for snapshot in self.snapshots:
            snapshot.close()

    def rollback(self) -> None:
        """Restore both files independently, reporting only a fixed failure."""
        if not self.active:
            return
        failed = False
        for snapshot in self.snapshots:
            try:
                snapshot.restore()
            except Exception:
                failed = True
            finally:
                snapshot.close()
        self.active = False
        if failed:
            raise RuntimeError("Predict input rollback failed")


async def _execute_predict_after_commit(
    start_gate: asyncio.Event,
    *,
    session_id: str,
    config_dict: dict[str, Any],
    manager: SessionManager,
    dataset_path: Path | None,
) -> None:
    """Prevent worker side effects until the request commits startup state."""
    await start_gate.wait()
    await execute_predict_async(
        session_id=session_id,
        config_dict=config_dict,
        session_manager=manager,
        dataset_path=dataset_path,
    )


async def _cleanup_failed_predict_session(
    *,
    manager: SessionManager,
    session_id: str,
    session_created_here: bool,
) -> None:
    """Best-effort cleanup for a request-owned session transaction."""
    if not session_created_here:
        return
    try:
        await manager.delete_session(session_id)
    except Exception:  # pragma: no cover - defensive cleanup containment
        logger.error("Failed to clean up rejected predict session")


async def _persist_predict_inputs_transactionally(
    *,
    predict_config: dict[str, Any],
    config_path: Path,
    dataset_source: Path | None,
    dataset_target: Path,
    manager: SessionManager,
    session_id: str,
    session_created_here: bool,
) -> _PredictInputsTransaction:
    """Publish worker config and an explicit dataset as one logical update.

    The dataset is copied to a sibling temporary file first, so a partial copy
    can never truncate the prior accepted artifact. Prior files are retained in
    bounded-memory anonymous disk snapshots until worker registration succeeds.
    The returned handle must be committed only after ``reservation.start()``
    completes.
    """
    staged_dataset: Path | None = None
    if dataset_source is not None:
        already_staged = (
            dataset_target.exists()
            and dataset_source.exists()
            and dataset_source.samefile(dataset_target)
        )
        if not already_staged:
            try:
                staged_dataset = _copy_to_sibling_temp(
                    dataset_source,
                    dataset_target,
                )
            except Exception:
                log_durable_failure(
                    logger,
                    "predict_dataset_stage_failed",
                    phase=FailurePhase.LOCAL_PUBLICATION,
                    retryable=True,
                )
                await _cleanup_failed_predict_session(
                    manager=manager,
                    session_id=session_id,
                    session_created_here=session_created_here,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_DATASET_STAGE_FAILED_DETAIL,
                ) from None

    snapshots: list[_FileSnapshot] = []
    try:
        snapshots.append(_FileSnapshot.capture(config_path))
        if staged_dataset is not None:
            snapshots.append(_FileSnapshot.capture(dataset_target))
    except Exception:
        for snapshot in snapshots:
            snapshot.close()
        if staged_dataset is not None:
            staged_dataset.unlink(missing_ok=True)
        log_durable_failure(
            logger,
            "predict_input_snapshot_failed",
            phase=FailurePhase.PERSISTENCE_VERIFICATION,
            retryable=True,
        )
        await _cleanup_failed_predict_session(
            manager=manager,
            session_id=session_id,
            session_created_here=session_created_here,
        )
        raise HTTPException(
            status_code=500,
            detail=PIPELINE_CONFIG_WRITE_FAILED_DETAIL,
        ) from None

    transaction = _PredictInputsTransaction(snapshots=tuple(snapshots))

    try:
        try:
            await build_and_write_pipeline_config(
                config_factory=lambda: predict_config,
                config_path=config_path,
                session_manager=manager,
                session_id=session_id,
                session_created_here=session_created_here,
            )
        except BaseException as exc:
            log_durable_failure(
                logger,
                "predict_config_publication_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            if session_created_here:
                transaction.commit()
            else:
                try:
                    transaction.rollback()
                except Exception:
                    log_durable_failure(
                        logger,
                        "predict_input_publication_restore_failed",
                        phase=FailurePhase.ROLLBACK,
                        retryable=True,
                    )
            # Contained HTTP failures have already performed ownership-aware
            # cleanup in build_and_write_pipeline_config. Other failures,
            # including cancellation after the writer quiesces, still need the
            # request transaction's one cleanup attempt.
            if not isinstance(exc, HTTPException):
                await _cleanup_failed_predict_session(
                    manager=manager,
                    session_id=session_id,
                    session_created_here=session_created_here,
                )
            if isinstance(exc, HTTPException | asyncio.CancelledError):
                raise
            if not isinstance(exc, Exception):
                raise
            raise HTTPException(
                status_code=500,
                detail=PIPELINE_CONFIG_WRITE_FAILED_DETAIL,
            ) from None

        if staged_dataset is not None:
            try:
                os.replace(staged_dataset, dataset_target)
                staged_dataset = None
            except Exception:
                log_durable_failure(
                    logger,
                    "predict_dataset_publication_failed",
                    phase=FailurePhase.LOCAL_PUBLICATION,
                    retryable=True,
                )
                if session_created_here:
                    transaction.commit()
                else:
                    try:
                        transaction.rollback()
                    except Exception:
                        log_durable_failure(
                            logger,
                            "predict_input_publication_restore_failed",
                            phase=FailurePhase.ROLLBACK,
                            retryable=True,
                        )
                await _cleanup_failed_predict_session(
                    manager=manager,
                    session_id=session_id,
                    session_created_here=session_created_here,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_DATASET_STAGE_FAILED_DETAIL,
                ) from None
    finally:
        if staged_dataset is not None:
            staged_dataset.unlink(missing_ok=True)
    return transaction


def _resolve_dataset_path_safely(raw_path: str, manager: SessionManager) -> Path:
    """Resolve and validate an absolute ``dataset_path`` arg.

    The path must canonicalize inside one of the allowed roots — the
    SessionManager's actual storage path plus any operator-provided extras
    from ``PA_DATASET_ALLOWED_ROOTS`` (colon-separated, read live so test
    fixtures that rebind env after import still apply). Anything outside is
    rejected with 403 to prevent the route from acting as a local-file-read
    primitive.
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise HTTPException(
            status_code=400, detail="dataset_path must be an absolute path"
        )
    try:
        real = candidate.resolve(strict=True)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path does not exist: {raw_path}",
        ) from e
    if not real.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path is not a regular file: {raw_path}",
        )
    if real.name != "dataset.jsonl":
        # Restrict to the canonical dataset filename so this route can't be
        # used to copy out arbitrary files (session.json, predictions.jsonl,
        # etc.) that happen to live under the session storage root.
        raise HTTPException(
            status_code=400,
            detail="dataset_path must point at a file named 'dataset.jsonl'",
        )

    env_roots = os.environ.get(
        "PA_DATASET_ALLOWED_ROOTS", config.dataset_allowed_roots or ""
    )
    extra_roots = [p for p in env_roots.split(":") if p.strip()]
    allowed_roots = [Path(manager.storage_path), *map(Path, extra_roots)]
    resolved_roots = []
    for root in allowed_roots:
        try:
            resolved_roots.append(root.resolve(strict=False))
        except OSError:
            continue

    for root in resolved_roots:
        try:
            common = Path(os.path.commonpath([str(real), str(root)]))
        except ValueError:
            # commonpath raises on different drives (Windows) or empty input.
            continue
        if common == root:
            return real

    raise HTTPException(
        status_code=403,
        detail=(
            "dataset_path resolves outside allowed roots. Set "
            "PA_DATASET_ALLOWED_ROOTS to opt-in additional locations."
        ),
    )


def _preflight_s3_object_size(s3_uri: str, max_bytes: int) -> None:
    """HEAD the S3 object and reject oversized payloads before any download.

    Raises HTTPException(413) when the advertised ``ContentLength`` already
    exceeds the configured cap, so a multi-GB object cannot fill the session
    volume during a transfer that we'd reject anyway.

    Network/permission/missing-object errors here are swallowed: we want the
    real download path to produce the canonical 404/403/502 response. This
    preflight is a best-effort fast-fail for the common case where
    ``ContentLength`` is available.
    """
    try:
        # Imported lazily so unit tests can monkey-patch s3_utils internals
        # without paying the import cost on every request.
        from world_understanding.utils.s3_utils import (
            _create_s3_client,
            _parse_s3_path,
        )

        bucket, key = _parse_s3_path(s3_uri)
        s3_client = _create_s3_client()
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        # Any failure (NoSuchBucket, AccessDenied, ProfileNotFound, network)
        # falls through to the real download_file_from_s3 path which has
        # full error-code translation. A failed preflight is not itself a
        # client-visible error.
        log_durable_failure(
            logger,
            "predict_s3_preflight_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        return

    content_length = head.get("ContentLength")
    if content_length is None:
        return
    size_bytes = int(content_length)
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"S3 file too large: {size_mb:.1f}MB. "
                f"Max: {max_bytes // (1024 * 1024)}MB"
            ),
        )


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

    # Reject oversized objects via head_object BEFORE writing anything to
    # disk. The post-download guard at the end of this function is kept as
    # a safety net in case ContentLength is missing or the object grows
    # between HEAD and GET.
    max_bytes = config.max_upload_size_mb * 1024 * 1024
    _preflight_s3_object_size(s3_uri, max_bytes)

    local_path = session_dir / "input" / f"scene{ext}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file_from_s3(s3_uri, local_path)
    except FileNotFoundError:
        log_durable_failure(
            logger,
            "predict_s3_object_not_found",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(status_code=404, detail="S3 object not found") from None
    except PermissionError:
        log_durable_failure(
            logger,
            "predict_s3_access_denied",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=False,
        )
        raise HTTPException(
            status_code=403, detail="Access denied to S3 object"
        ) from None
    except Exception:
        log_durable_failure(
            logger,
            "predict_s3_download_failed",
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
            detail=(
                f"S3 file too large: {size_mb:.1f}MB. "
                f"Max: {config.max_upload_size_mb}MB"
            ),
        )

    return local_path


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in _VALID_USD_EXTENSIONS:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


@router.post(
    "",
    response_model=SessionCreated,
    status_code=202,
    responses=PREDICT_INPUT_ERROR_RESPONSES,
)
async def create_predict(
    usd_file: UploadFile | None = File(
        None,
        description=(
            "USD file to predict on (optional if dataset_path, session_id or "
            "s3_uri is provided)."
        ),
    ),
    session_id: str | None = Form(
        None,
        description=(
            "Existing session ID (e.g. from POST /pipeline/upload-usd). When "
            "the session has a prepared dataset whose referenced images "
            "resolve, /predict runs Mode A (predict only)."
        ),
    ),
    s3_uri: str | None = Form(
        None,
        description="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    ),
    dataset_path: str | None = Form(
        None,
        description=(
            "Absolute path to a prepared dataset.jsonl on the server. When "
            "set and readable, forces Mode A (predict-only)."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional, used in Mode B)",
    ),
    render_backend: str = Form(
        default="",
        description=(
            "Rendering backend for Mode B: 'remote' (default), 'warp', "
            "'ovrtx', or 'mock'. Ignored in Mode A."
        ),
        json_schema_extra={"enum": [*RENDERING_BACKEND_NAMES, ""]},
    ),
    optimize_usd: bool = Form(
        default=False,
        description=(
            "Enable USD optimization step in Mode B (default: false). Ignored "
            "in Mode A."
        ),
    ),
    enable_deinstance: bool = Form(
        default=True,
        description="Enable deinstance op when optimize_usd=true (Mode B only).",
    ),
    enable_split: bool = Form(
        default=False,
        description="Enable split-meshes op when optimize_usd=true (Mode B only).",
    ),
    enable_deduplicate: bool = Form(
        default=False,
        description="Enable deduplicate op when optimize_usd=true (Mode B only).",
    ),
) -> SessionCreated:
    """Create and execute a prediction job.

    Two input modes (auto-detected at job start):

    * **Mode A — dataset already prepared.** Triggered when ``dataset_path``
      points at a readable dataset.jsonl, or when an existing ``session_id``
      has ``cache/dataset/dataset.jsonl`` and its referenced images resolve.
      Only the predict step runs; upstream prep (rendering, dataset prep) is
      skipped.
    * **Mode B — USD upload / s3_uri / fresh session_id.** When no runnable
      prepared dataset is present, /predict runs the minimum upstream steps
      (``optimize_usd`` if enabled → ``identify_asset`` → ``build_dataset_usd``
      → ``build_dataset_prepare_dataset``) before predicting. ``apply_physics``
      is intentionally not part of /predict — use POST /pipeline if you need
      the full classify/apply flow.

      A cached JSONL whose image references do not resolve falls back to Mode B
      when an input USD is available; without an input USD, the request returns
      HTTP 400 with recovery guidance.

    The detected mode is persisted to session metadata under ``predict_mode``
    and surfaced in the GET /predict/{id}/results response.
    """
    user_prompt_text = user_prompt.strip() if user_prompt else None

    # Reject ambiguous input combinations up front. The route advertises four
    # input sources, but only specific combinations are well-defined:
    #
    #   * exactly one of {usd_file, session_id, s3_uri} as the primary source
    #   * dataset_path may be supplied alone (pure Mode A) or together with
    #     session_id (override the session's prepared dataset)
    #   * dataset_path with usd_file or s3_uri is contradictory — Mode A would
    #     ignore the upload while the docs say dataset_path forces Mode A.
    #
    # Rejecting these combinations early prevents silent precedence games
    # where, for example, session_id won over a non-empty usd_file or a
    # dataset_path + s3_uri request still downloaded the S3 object.
    has_usd_file = usd_file is not None and (usd_file.filename or "").strip() != ""
    primary_sources = [
        ("usd_file", has_usd_file),
        ("session_id", bool(session_id)),
        ("s3_uri", bool(s3_uri)),
    ]
    provided_primary = [name for name, present in primary_sources if present]
    if len(provided_primary) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide exactly one of usd_file, session_id, or s3_uri "
                f"(got: {', '.join(provided_primary)})."
            ),
        )
    if dataset_path and (has_usd_file or s3_uri):
        # session_id + dataset_path is the one supported override.
        raise HTTPException(
            status_code=400,
            detail=(
                "dataset_path is incompatible with usd_file or s3_uri "
                "(those are Mode B inputs; dataset_path forces Mode A). "
                "Combine dataset_path with session_id instead, or send it alone."
            ),
        )

    # Authorize a client-controlled S3 source before any session-store or
    # network access. An upload or authorized S3 URI is unambiguously Mode B,
    # so validate every Mode-B-only option before creating a session, writing
    # upload bytes, or downloading an object. Existing sessions are validated
    # below after checking whether they contain a prepared Mode A dataset.
    if s3_uri:
        _validate_and_authorize_s3_usd_uri(s3_uri)

    render_backend_text = render_backend.strip() if render_backend else None
    if has_usd_file or s3_uri:
        render_backend_text = _validate_mode_b_options(
            render_backend=render_backend,
            optimize_usd=optimize_usd,
            enable_deinstance=enable_deinstance,
            enable_split=enable_split,
            enable_deduplicate=enable_deduplicate,
        )

    manager = get_session_manager()

    # Resolve dataset_path early — must canonicalize inside an allowed root
    # so /predict cannot be used as an arbitrary local-file-read primitive.
    resolved_dataset_path: Path | None = None
    if dataset_path:
        resolved_dataset_path = _resolve_dataset_path_safely(dataset_path, manager)

    # Track whether THIS request created the session — only sessions created
    # here are safe to delete on later validation failures. Reused sessions
    # belong to the caller and may already hold uploaded USDs / artifacts.
    session_created_here = False

    # Resolve session_dir / input USD
    if session_id:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        # Reject re-queuing while an earlier predict job on this session is
        # still in-flight or being torn down. Without this guard, two POSTs
        # for the same session could race on the same cache/ paths and the
        # second job's metadata would clobber the first's. Mirrors the
        # /pipeline/{id}/regenerate guard and uses the persisted store
        # status so it works cross-pod, plus the in-process JobRegistry
        # to close the same-pod TOCTOU window where two concurrent POSTs
        # both observe a terminal status before either writes "pending".
        # Cross-pod concurrent reruns are still a known limitation shared
        # with /pipeline (no distributed lock).
        if get_job_registry().is_running(session_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Predict already running on this instance for session "
                    f"{session_id}. Wait for it to finish or cancel first."
                ),
            )
        existing_metadata = await manager.get_session_metadata(session_id)
        existing_status = (existing_metadata or {}).get("status")
        if existing_status in ("pending", "running", "cancelling"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Predict already {existing_status} for session "
                    f"{session_id}. Wait for it to reach a terminal state "
                    f"or cancel it first."
                ),
            )
        session_dir = manager.get_session_dir(session_id)
    elif s3_uri:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
        try:
            # _download_s3_to_session is sync (boto3 + post-download size
            # check); run it on a thread so a slow transfer doesn't block
            # the request event loop and stall other handlers on this worker.
            local_path = await asyncio.to_thread(
                _download_s3_to_session, s3_uri, session_dir
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for /predict session "
                f"{session_id[:8]}: {size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception:
            log_durable_failure(
                logger,
                "predict_s3_ingest_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to download USD from S3"
            ) from None
    elif has_usd_file:
        # has_usd_file (computed above) treats UploadFile with an empty
        # filename as "no file", matching what FastAPI hands us when the
        # multipart field is absent.
        if usd_file is None:
            raise HTTPException(status_code=400, detail="USD upload is missing")
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
        try:
            if usd_file.filename:
                ext = Path(usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid USD file type: {ext}. "
                            f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}"
                        ),
                    )
            original_ext = (
                Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
            total_bytes = await _stream_copy(
                usd_file,
                usd_path,
                max_bytes=config.max_upload_size_mb * 1024 * 1024,
            )
            size_mb = total_bytes / (1024 * 1024)
            logger.info(
                f"USD uploaded for /predict session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({original_ext})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception:
            log_durable_failure(
                logger,
                "predict_usd_local_publication_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail="Failed to save USD file"
            ) from None
    elif resolved_dataset_path is not None:
        # Pure Mode A from explicit dataset path with no session/USD context.
        # We still need a session_dir for outputs.
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "One of usd_file, session_id, s3_uri, or dataset_path must be provided"
            ),
        )

    if session_id is None:
        raise RuntimeError("Predict session allocation did not produce an identifier")
    # Mode B (USD-driven) needs an input USD on disk so the renderer can run.
    # Mode A doesn't — it can predict from dataset.jsonl alone. We don't
    # require an input USD when dataset_path was supplied OR when the session
    # already has a runnable prepared dataset.
    #
    # The actual copy of the external dataset_path into the session cache is
    # deferred until *after* job_registry.reserve() succeeds, so a losing
    # concurrent rerun cannot clobber the winner's cached dataset.jsonl. We
    # only need to know *whether* mode A applies here, not have the file
    # staged yet.
    session_dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    session_has_dataset = session_dataset.exists()
    session_dataset_is_runnable = (
        session_has_dataset and _dataset_jsonl_has_resolvable_images(session_dataset)
    )
    will_be_mode_a = resolved_dataset_path is not None or session_dataset_is_runnable

    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path and not will_be_mode_a:
        # A prepared dataset may live on another instance. Resolve that Mode A
        # possibility first because its Mode-B-only options must remain ignored.
        # If no runnable dataset exists, the session is Mode B: validate its
        # options before pulling a potentially large input USD into the local
        # cache.
        await manager.sync_from_store(session_id, prefix="cache/dataset/")
        session_has_dataset = session_dataset.exists()
        session_dataset_is_runnable = (
            session_has_dataset
            and _dataset_jsonl_has_resolvable_images(session_dataset)
        )
        will_be_mode_a = (
            resolved_dataset_path is not None or session_dataset_is_runnable
        )
        if not will_be_mode_a:
            render_backend_text = _validate_mode_b_options(
                render_backend=render_backend,
                optimize_usd=optimize_usd,
                enable_deinstance=enable_deinstance,
                enable_split=enable_split,
                enable_deduplicate=enable_deduplicate,
            )
            pulled = await manager.sync_from_store(session_id, prefix="input/")
            if pulled > 0:
                logger.info(
                    f"Pulled {pulled} input file(s) from store for /predict session "
                    f"{session_id[:8]}"
                )
            input_usd_path = _find_input_usd(session_dir)

    if not input_usd_path and not will_be_mode_a:
        if resolved_dataset_path is None and session_has_dataset:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Session has a staged dataset.jsonl but its referenced "
                    "images are not present (likely a previous run staged the "
                    "JSONL alone, or the per-prim PNGs have not been synced "
                    "down on this instance), and no input USD is available to "
                    "rebuild from source. Re-supply dataset_path with the "
                    "original directory, upload the USD again, or provide "
                    "s3_uri."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "No input USD or prepared dataset found for this session. "
                "Provide usd_file, s3_uri, or dataset_path."
            ),
        )

    # The optimizer flags and render_backend are ignored in Mode A per the
    # docstring; only validate them when the request actually triggers
    # Mode B so a dataset-only call can't 400 (or 500 from
    # build_default_pipeline_config) on options that won't run.
    if not will_be_mode_a:
        render_backend_text = _validate_mode_b_options(
            render_backend=render_backend,
            optimize_usd=optimize_usd,
            enable_deinstance=enable_deinstance,
            enable_split=enable_split,
            enable_deduplicate=enable_deduplicate,
        )

    # Build a pipeline config dict so Mode B can drive the full upstream
    # workflow. In Mode A only the `predict` step + project/working_dir
    # actually matter — the rest is harmless.
    # When no input USD exists yet (Mode A from dataset_path with a fresh
    # session), use a sentinel string. Mode B never reaches that branch
    # because we already raised 400 above.
    usd_path_for_config = (
        str(input_usd_path)
        if input_usd_path
        else str(session_dir / "input" / "scene.usda")
    )

    from .pipeline_router import _apply_render_request_limit

    def prepare_predict_config() -> dict[str, Any]:
        prepared: dict[str, Any] = build_default_pipeline_config(
            session_id=session_id,
            usd_path=usd_path_for_config,
            working_dir=str(session_dir / "cache"),
            user_prompt=user_prompt_text,
            # Mode A ignores the render backend; suppress it so a typo'd
            # value can't make build_default_pipeline_config raise on a
            # request that wasn't going to render anything anyway.
            render_backend=None if will_be_mode_a else render_backend_text,
            optimize_usd=False if will_be_mode_a else optimize_usd,
            enable_deinstance=enable_deinstance,
            enable_split=enable_split,
            enable_deduplicate=enable_deduplicate,
        )
        # Mirror /pipeline: clamp render-step concurrency to the process-wide
        # cap, and disable apply_physics for this prediction-only route.
        _apply_render_request_limit(prepared)
        prepared.setdefault("steps", {}).setdefault("apply_physics", {})["enabled"] = (
            False
        )
        return prepared

    # Reject invalid request-derived configuration before claiming a job slot.
    # This build is side-effect-free; the exact validated object is persisted
    # only after the reservation is acquired below.
    predict_config = await build_and_validate_pipeline_config(
        config_factory=prepare_predict_config,
        session_manager=manager,
        session_id=session_id,
        session_created_here=session_created_here,
    )

    job_registry = get_job_registry()

    # Atomically claim the slot in the in-process registry BEFORE writing any
    # session state. A losing concurrent rerun for the same terminal
    # session must NOT mutate session metadata/config: under the previous
    # ordering it could pass the up-front is_running()/persisted-status
    # check, write `status=pending` + a new config block, and only then
    # get rejected by JobRegistry — leaving the winning job running with
    # the loser's metadata. Reserving first inverts the order: the loser
    # raises ValueError here, propagates as 409, and never touches
    # session state. See registry.JobRegistry.reserve().
    try:
        reservation = await job_registry.reserve(session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    async with reservation:
        # Configuration and an explicit dataset are one worker-input update.
        # Keep the accepted files untouched until staging succeeds, then
        # publish both transactionally before marking the job pending. A
        # rejected concurrent rerun, partial copy, or failed publish therefore
        # leaves the previous terminal session fully retryable.
        config_path = session_dir / "input" / "predict_config.yaml"
        session_dataset_target = session_dir / "cache" / "dataset" / "dataset.jsonl"
        input_transaction: _PredictInputsTransaction | None = None
        metadata_snapshot: dict[str, Any] | None = None
        start_gate = asyncio.Event()
        try:
            existing = await manager.get_session_metadata(session_id)
            if not isinstance(existing, dict):
                log_durable_failure(
                    logger,
                    "predict_metadata_snapshot_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_PREDICT_START_FAILED_DETAIL,
                )
            metadata_snapshot = deepcopy(existing)
            existing_config_value = existing.get("config")
            existing_config = (
                existing_config_value if isinstance(existing_config_value, dict) else {}
            )

            input_transaction = await _persist_predict_inputs_transactionally(
                predict_config=predict_config,
                config_path=config_path,
                dataset_source=resolved_dataset_path,
                dataset_target=session_dataset_target,
                manager=manager,
                session_id=session_id,
                session_created_here=session_created_here,
            )

            # Register a gated coroutine before the pending metadata transition.
            # The worker cannot mutate files/status until every startup surface is
            # committed and the gate is opened below.
            await reservation.start(
                _execute_predict_after_commit(
                    start_gate,
                    session_id=session_id,
                    config_dict=predict_config,
                    manager=manager,
                    dataset_path=resolved_dataset_path,
                )
            )
            await manager.update_session(
                session_id,
                {
                    "status": "pending",
                    "can_cancel": True,
                    "config": {
                        **existing_config,
                        "project_name": predict_config.get("project", {}).get(
                            "name", ""
                        ),
                        "usd_path": str(input_usd_path) if input_usd_path else None,
                        "has_usd_upload": existing_config.get("has_usd_upload", False)
                        or has_usd_file,
                        "s3_uri": s3_uri or existing_config.get("s3_uri"),
                        "user_prompt": user_prompt_text,
                        "optimize_usd": optimize_usd,
                        "enable_deinstance": enable_deinstance,
                        "enable_split": enable_split,
                        "enable_deduplicate": enable_deduplicate,
                        "predict_route": True,
                    },
                },
            )

            # Local status is mutated only after every fallible publication and
            # registration step has succeeded. Failed startup therefore leaves
            # the prior EventBus state in place without snapshot restoration.
            get_event_bus().cleanup_session(session_id)
            input_transaction.commit()
            start_gate.set()
        except BaseException as exc:
            try:
                await job_registry.cancel(session_id)
            except Exception:
                log_durable_failure(
                    logger,
                    "predict_start_task_cancel_failed",
                    phase=FailurePhase.ROLLBACK,
                    retryable=True,
                )

            if session_created_here:
                if input_transaction is not None:
                    input_transaction.commit()
                get_event_bus().cleanup_session(session_id)
                await _cleanup_failed_predict_session(
                    manager=manager,
                    session_id=session_id,
                    session_created_here=True,
                )
            else:
                if input_transaction is not None:
                    try:
                        input_transaction.rollback()
                    except Exception:
                        log_durable_failure(
                            logger,
                            "predict_start_input_restore_failed",
                            phase=FailurePhase.ROLLBACK,
                            retryable=True,
                        )
                if metadata_snapshot is not None:
                    try:
                        await manager.restore_session_metadata(
                            session_id,
                            deepcopy(metadata_snapshot),
                        )
                    except Exception:
                        log_durable_failure(
                            logger,
                            "predict_start_metadata_restore_failed",
                            phase=FailurePhase.ROLLBACK,
                            retryable=True,
                        )

            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=500,
                detail=_PREDICT_START_FAILED_DETAIL,
            ) from None

    logger.info(f"/predict registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Predict job queued for execution",
        estimated_duration_minutes=10,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_predict_status(session_id: str) -> PipelineStatus:
    """Get predict execution status with detailed progress.

    Reuses the same response schema as /pipeline/{id}/status so existing
    progress UIs work unchanged. Reads from the in-memory event bus first,
    falls back to the SessionManager store for cross-instance visibility.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot:
        metadata = snapshot
        preview_images = snapshot.get("preview_images", [])
    else:
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


@router.get("/{session_id}/results", response_model=PredictResults | PipelineError)
async def get_predict_results(session_id: str):
    """Get predict execution results (only available when completed)."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        results = metadata.get("results") or {}
        # Build download URLs. We only advertise dataset when one actually
        # exists for this session — Mode A from an external dataset_path
        # leaves the session's dataset/ dir empty, so claiming the URL works
        # would be a lie.
        download_urls: dict[str, str] = {
            "predictions": f"/artifacts/{session_id}/predictions",
            "report": f"/artifacts/{session_id}/report",
        }
        session_dir = manager.get_session_dir(session_id)
        dataset_local = session_dir / "cache" / "dataset" / "dataset.jsonl"
        if not dataset_local.exists():
            # Cross-instance case: the worker pod synced the dataset to the
            # shared store but this pod doesn't have the local copy yet.
            # Pull just the dataset prefix before deciding whether to advertise
            # the URL, so /artifacts/{id}/dataset can serve it on this pod too.
            try:
                await manager.sync_from_store(session_id, prefix="cache/dataset/")
            except Exception:  # noqa: BLE001
                log_durable_failure(
                    logger,
                    "predict_dataset_restore_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
        if dataset_local.exists():
            download_urls["dataset"] = f"/artifacts/{session_id}/dataset"

        # The worker normalizes PredictOutput.predictions_count into
        # results["predictions_made"], but if a future code path ever stores
        # a PredictOutput-shaped dict directly we still want the REST layer
        # to surface the right number — accept either key.
        predictions_count = int(
            results.get("predictions_made", results.get("predictions_count", 0))
        )

        return PredictResults(
            session_id=session_id,
            status=status,
            mode=metadata.get("predict_mode", "unknown"),
            steps_run=metadata.get("predict_steps_run", []),
            stats=results,
            predictions_count=predictions_count,
            failed_count=int(results.get("failed_count", 0)),
            predictions_path=results.get("predictions_path"),
            token_stats=results.get("token_stats", {}) or {},
            download_urls=download_urls,
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )

    if status == "failed":
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step=metadata.get("failed_step", "predict"),
            completed_steps=[s["name"] for s in metadata.get("completed_steps", [])],
            partial_results=metadata.get("partial_results"),
        )

    raise HTTPException(
        status_code=202,
        detail=f"Predict still {status}. Check status endpoint for progress.",
    )


@router.post("/{session_id}/cancel")
async def cancel_predict(session_id: str) -> dict[str, str]:
    """Cancel a running predict job."""
    job_registry = get_job_registry()
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    # Refuse to cancel a session that wasn't started via /predict. Without
    # this guard, /predict/{id}/cancel would happily cancel a /pipeline
    # session and respond "Predict cancellation requested" — confusing and
    # incorrect. The predict route stamps `predict_route: True` into
    # session metadata.config when it queues; we use that as the
    # discriminator. Sessions without a config block fall through to the
    # standard cancel semantics (typically just-created predict sessions
    # caught between create and the metadata stamp).
    session_config = metadata.get("config") or {}
    if session_config and not session_config.get("predict_route"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is not a predict session "
                "(it was created via /pipeline or /pipeline/upload-usd). Use "
                "POST /pipeline/{session_id}/cancel instead."
            ),
        )

    if metadata["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel predict job with status: {metadata['status']}",
        )

    await manager.request_cancellation(session_id)

    if job_registry.is_running(session_id):
        cancelled = await job_registry.cancel(session_id)
        if cancelled:
            # If the task was still queued (waiting on the JobRegistry
            # semaphore), CancelledError fires before execute_predict_async
            # ever enters its CancelledError handler, so the session would
            # otherwise be stuck on "cancelling" forever. Drive the metadata
            # to a terminal "cancelled" ourselves when the persisted state
            # is still mid-cancel.
            post_cancel = await manager.get_session_metadata(session_id)
            if post_cancel and post_cancel.get("status") in (
                "cancelling",
                "pending",
                "running",
            ):
                await manager.update_session(
                    session_id,
                    {
                        "status": "cancelled",
                        "cancelled_at": datetime.now(UTC).isoformat(),
                        "can_cancel": False,
                    },
                )

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Predict cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_predict_events(session_id: str) -> EventSourceResponse:
    """Stream real-time predict progress events via Server-Sent Events (SSE).

    Mirrors /pipeline/{id}/events semantics: only works when connected to the
    instance executing the job; for cross-instance progress, poll
    GET /predict/{id}/status instead.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")

    if snapshot is None:
        metadata = await manager.get_session_metadata(session_id)
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Predict is running on a different instance; use polling instead"
                ),
            )

    async def event_generator():  # pragma: no cover - SSE transport loop
        queue = event_bus.get_queue(session_id)

        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": (
                    f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}'
                ),
            }
            return

        if snapshot is None:
            metadata = await manager.get_session_metadata(session_id)
            if metadata and metadata.get("status") in terminal_states:
                final_state = metadata["status"]
                yield {
                    "event": "done",
                    "data": (
                        f'{{"session_id": "{session_id}", '
                        f'"final_state": "{final_state}"}}'
                    ),
                }
                return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_data = event.model_dump_json()
                    yield {"event": "progress", "data": event_data}

                    should_close = False
                    if event.state in ["failed", "cancelled"]:
                        should_close = True
                    elif event.extra and event.extra.get("pipeline_ready"):
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "{event.state}"}}'
                            ),
                        }
                        break

                except TimeoutError:
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        final_state = metadata["status"]
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "{final_state}"}}'
                            ),
                        }
                        break
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"/predict SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)
