# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable FastAPI service harness for Texture Variation API backends."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from .backend import TextureGenerationBackend, TextureGenerationBackendError
from .models import (
    AssetUploadResponse,
    CreateJobRequest,
    GenerationResult,
    HealthResponse,
    JobStatus,
)

logger = logging.getLogger(__name__)
_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_DEFAULT_MAX_UPLOAD_BYTES = 256 * 1024 * 1024
_UPLOAD_ROUTE_SEGMENT = "texture-variation-assets"


class ServiceBusyError(RuntimeError):
    """Raised when the service queue is full."""


class ServiceNotReadyError(RuntimeError):
    """Raised when the backend is not ready to accept work."""


@dataclass
class _UploadRecord:
    path: Path
    created_at: float
    claimed_job_id: str | None = None


class _JobRecord:
    def __init__(
        self,
        status: JobStatus,
        *,
        upload_ids: tuple[str, ...] = (),
    ) -> None:
        self.status = status
        self.upload_ids = upload_ids
        self.cancel_event = threading.Event()
        self.future: Future[None] | None = None
        self.completed_at: float | None = None


class TextureVariationService:
    """In-memory Texture Variation API job service."""

    def __init__(
        self,
        *,
        backend: TextureGenerationBackend,
        output_dir: Path,
        service_name: str = "texture-variation-api",
        version: str = "1.0.0",
        max_workers: int = 1,
        max_queue_size: int = 0,
        terminal_job_ttl_sec: float = 300.0,
        upload_ttl_sec: float = 300.0,
    ) -> None:
        self.backend = backend
        self.output_dir = output_dir
        self.service_name = service_name
        self.version = version
        self.max_workers = max(1, max_workers)
        self.max_queue_size = max(0, max_queue_size)
        self.terminal_job_ttl_sec = max(0.0, terminal_job_ttl_sec)
        self.upload_ttl_sec = max(0.0, upload_ttl_sec)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._jobs: dict[str, _JobRecord] = {}
        self._uploads: dict[str, _UploadRecord] = {}
        self._pending_output_cleanup: set[str] = set()
        self._pending_upload_cleanup: set[str] = set()
        self._lock = threading.Lock()

    def register_upload(self, asset_id: str, path: Path) -> None:
        """Register one fully written service-owned input artifact."""

        with self._lock:
            if asset_id in self._uploads:
                raise ValueError("Upload asset id already exists.")
            self._uploads[asset_id] = _UploadRecord(
                path=path.resolve(),
                created_at=time.monotonic(),
            )

    def get_upload(self, asset_id: str, filename: str) -> Path:
        """Return a registered input artifact without exposing arbitrary paths."""

        expired_upload_ids: list[str]
        with self._lock:
            expired_upload_ids = self._evict_unclaimed_uploads_locked()
            record = self._uploads.get(asset_id)
            path = (
                record.path
                if record is not None
                and asset_id not in self._pending_upload_cleanup
                and record.path.name == filename
                else None
            )
        self._cleanup_uploads(expired_upload_ids)
        if path is None or not path.is_file():
            raise KeyError(asset_id)
        return path

    def upload_id_for_job_path(self, job_id: str, path: Path) -> str | None:
        """Return the job-owned upload id associated with a local path."""

        resolved = path.resolve()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            for asset_id in record.upload_ids:
                upload = self._uploads.get(asset_id)
                if upload is not None and upload.path == resolved:
                    return asset_id
        return None

    def get_job_artifact(self, job_id: str, artifact_path: str) -> Path:
        """Resolve one backend artifact within its owning job directory."""

        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
        root = (self.output_dir / job_id).resolve()
        candidate = (root / artifact_path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise KeyError(artifact_path)
        return candidate

    def submit(self, request: CreateJobRequest) -> JobStatus:
        """Submit a request and return queued status."""
        backend_health = self.backend.health()
        if not backend_health.ready:
            raise ServiceNotReadyError(backend_health.error or "Backend is not ready.")

        job_id = f"vj-{uuid.uuid4().hex[:12]}"
        status = JobStatus(job_id=job_id, status="queued", progress=0)
        record: _JobRecord | None = None
        expired_job_ids: list[str] = []
        expired_upload_ids: list[str] = []
        busy_error: ServiceBusyError | None = None
        localized_request = request
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            expired_upload_ids = self._evict_unclaimed_uploads_locked()
            active_jobs = self._count_status_locked("processing")
            queued_jobs = self._count_status_locked("queued")
            if active_jobs + queued_jobs >= self.max_workers + self.max_queue_size:
                busy_error = ServiceBusyError("Texture generation queue is full.")
            else:
                localized_request, upload_ids = self._claim_request_uploads_locked(
                    request,
                    job_id=job_id,
                )
                record = _JobRecord(status, upload_ids=upload_ids)
                self._jobs[job_id] = record
                record.future = self._executor.submit(
                    self._run_job,
                    job_id,
                    localized_request,
                    record,
                )
        self._cleanup_job_outputs(expired_job_ids)
        self._cleanup_uploads(expired_upload_ids)
        if busy_error is not None:
            raise busy_error
        return status.model_copy(deep=True)

    def get_status(self, job_id: str) -> JobStatus:
        """Return a job status copy."""
        expired_job_ids: list[str] = []
        expired_upload_ids: list[str] = []
        status: JobStatus | None = None
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            expired_upload_ids = self._evict_unclaimed_uploads_locked()
            record = self._jobs.get(job_id)
            if record is not None:
                status = record.status.model_copy(deep=True)
        self._cleanup_job_outputs(expired_job_ids)
        self._cleanup_uploads(expired_upload_ids)
        if status is None:
            raise KeyError(job_id)
        return status

    def cancel(self, job_id: str) -> None:
        """Request cancellation.

        Queued jobs are marked cancelled when their future can be cancelled
        before execution. Running jobs receive a cooperative cancellation event;
        backends may finish if interruption is unsafe.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status.status in {"completed", "failed", "cancelled"}:
                raise ValueError(f"Job in terminal state: {record.status.status}")
            record.cancel_event.set()
            if record.future is not None and record.future.cancel():
                record.status.status = "cancelled"
                record.status.progress = 100
                record.status.message = "Cancelled before execution."
                record.completed_at = time.monotonic()

    def health(self) -> HealthResponse:
        """Return service and backend health."""
        backend_health = self.backend.health()
        expired_job_ids: list[str] = []
        expired_upload_ids: list[str] = []
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            expired_upload_ids = self._evict_unclaimed_uploads_locked()
            active_jobs = self._count_status_locked("processing")
            queued_jobs = self._count_status_locked("queued")
        self._cleanup_job_outputs(expired_job_ids)
        self._cleanup_uploads(expired_upload_ids)
        accepting = (
            backend_health.ready
            and active_jobs + queued_jobs < self.max_workers + self.max_queue_size
        )
        status = backend_health.status
        if backend_health.ready and active_jobs >= self.max_workers:
            status = "busy"
        return HealthResponse(
            status=status,
            service=self.service_name,
            version=self.version,
            backend=self.backend.name,
            ready=backend_health.ready,
            accepting_jobs=accepting,
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            max_workers=self.max_workers,
            max_queue_size=self.max_queue_size,
            warmup_complete=backend_health.warmup_complete,
            gpu_available=backend_health.gpu_available,
            capabilities=backend_health.capabilities,
            error=backend_health.error,
        )

    def shutdown(self) -> None:
        """Cancel queued work and stop the executor."""
        with self._lock:
            for record in self._jobs.values():
                record.cancel_event.set()
                if record.status.status == "queued":
                    record.status.status = "cancelled"
                    record.status.progress = 100
                    record.status.message = "Service is shutting down."
                    record.completed_at = time.monotonic()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _count_status_locked(self, status: str) -> int:
        return sum(
            1 for record in self._jobs.values() if record.status.status == status
        )

    def _evict_terminal_locked(self) -> list[str]:
        if self.terminal_job_ttl_sec > 0:
            now = time.monotonic()
            expired = [
                job_id
                for job_id, record in self._jobs.items()
                if record.completed_at is not None
                and now - record.completed_at > self.terminal_job_ttl_sec
            ]
            for job_id in expired:
                del self._jobs[job_id]
                self._pending_output_cleanup.add(job_id)
        return list(self._pending_output_cleanup)

    def _evict_unclaimed_uploads_locked(self) -> list[str]:
        if self.upload_ttl_sec > 0:
            now = time.monotonic()
            self._pending_upload_cleanup.update(
                asset_id
                for asset_id, record in self._uploads.items()
                if record.claimed_job_id is None
                and now - record.created_at > self.upload_ttl_sec
            )
        return list(self._pending_upload_cleanup)

    @staticmethod
    def _asset_id_from_uri(uri: str) -> tuple[str, str] | None:
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            return None
        parts = tuple(part for part in parsed.path.split("/") if part)
        try:
            marker = parts.index(_UPLOAD_ROUTE_SEGMENT)
            asset_id, filename = (
                unquote(part) for part in parts[marker + 1 : marker + 3]
            )
        except (ValueError, IndexError):
            return None
        if marker + 3 != len(parts):
            return None
        return asset_id, filename

    def _claim_uploaded_uri_locked(
        self,
        uri: str | None,
        *,
        job_id: str,
        upload_ids: set[str],
    ) -> str | None:
        if uri is None:
            return None
        identity = self._asset_id_from_uri(uri)
        if identity is None:
            return uri
        asset_id, filename = identity
        record = self._uploads.get(asset_id)
        if (
            record is None
            or asset_id in self._pending_upload_cleanup
            or record.path.name != filename
        ):
            raise ValueError("Texture variation input upload is missing or expired.")
        if record.claimed_job_id not in {None, job_id}:
            raise ValueError("Texture variation input upload is already in use.")
        upload_ids.add(asset_id)
        return record.path.as_uri()

    def _claim_request_uploads_locked(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
    ) -> tuple[CreateJobRequest, tuple[str, ...]]:
        upload_ids: set[str] = set()

        def localize(uri: str | None) -> str | None:
            return self._claim_uploaded_uri_locked(
                uri,
                job_id=job_id,
                upload_ids=upload_ids,
            )

        conditioning = request.conditioning.model_copy(
            update={
                "reference_image_uris": [
                    localize(uri) for uri in request.conditioning.reference_image_uris
                ],
                "turntable_video_uri": localize(
                    request.conditioning.turntable_video_uri
                ),
                "multiview_image_uris": [
                    localize(uri) for uri in request.conditioning.multiview_image_uris
                ],
            }
        )
        configuration = request.configuration
        if configuration.weathering is not None:
            configuration = configuration.model_copy(
                update={
                    "weathering": configuration.weathering.model_copy(
                        update={
                            "editable_mask_uri": localize(
                                configuration.weathering.editable_mask_uri
                            ),
                            "protected_mask_uri": localize(
                                configuration.weathering.protected_mask_uri
                            ),
                        }
                    )
                }
            )
        localized = request.model_copy(
            update={
                "source_asset_uri": localize(request.source_asset_uri),
                "conditioning": conditioning,
                "configuration": configuration,
            }
        )
        for asset_id in upload_ids:
            self._uploads[asset_id].claimed_job_id = job_id
        return localized, tuple(sorted(upload_ids))

    def _cleanup_job_outputs(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            if not self._cleanup_job_output(job_id):
                continue
            with self._lock:
                self._pending_output_cleanup.discard(job_id)

    def _cleanup_job_output(self, job_id: str) -> bool:
        job_dir = self.output_dir / job_id
        with self._lock:
            record = self._jobs.get(job_id)
            upload_ids = record.upload_ids if record is not None else ()
            if record is None:
                upload_ids = tuple(
                    asset_id
                    for asset_id, upload in self._uploads.items()
                    if upload.claimed_job_id == job_id
                )
        try:
            shutil.rmtree(job_dir)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to clean texture generation job output: %s", job_dir)
            return False
        for asset_id in upload_ids:
            with self._lock:
                upload = self._uploads.get(asset_id)
            if upload is None:
                continue
            try:
                shutil.rmtree(upload.path.parent)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Failed to clean texture generation upload: %s", upload.path
                )
                return False
            with self._lock:
                self._uploads.pop(asset_id, None)
        return True

    def _cleanup_uploads(self, asset_ids: list[str]) -> None:
        for asset_id in asset_ids:
            with self._lock:
                record = self._uploads.get(asset_id)
            if record is None:
                with self._lock:
                    self._pending_upload_cleanup.discard(asset_id)
                continue
            try:
                shutil.rmtree(record.path.parent)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Failed to clean texture generation upload: %s", record.path
                )
                continue
            with self._lock:
                self._uploads.pop(asset_id, None)
                self._pending_upload_cleanup.discard(asset_id)

    def _update(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = record.status.model_copy(update=kwargs)
            if record.status.status in _TERMINAL_STATES:
                record.completed_at = record.completed_at or time.monotonic()

    def _run_job(
        self,
        job_id: str,
        request: CreateJobRequest,
        record: _JobRecord,
    ) -> None:
        if record.cancel_event.is_set():
            self._update(
                job_id,
                status="cancelled",
                progress=100,
                message="Cancelled before execution.",
            )
            return

        self._update(
            job_id,
            status="processing",
            progress=5,
            message="Starting texture generation...",
        )
        output_dir = self.output_dir / job_id

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            result = self.backend.generate(
                request,
                job_id=job_id,
                output_dir=output_dir,
                cancel_event=record.cancel_event,
            )
        except TextureGenerationBackendError as exc:
            logger.exception("[%s] Texture variation job failed", job_id)
            self._update(
                job_id,
                status="failed",
                error_message=str(exc),
                result=exc.result,
            )
            return
        except Exception as exc:
            logger.exception("[%s] Texture variation job failed", job_id)
            if record.cancel_event.is_set():
                self._update(
                    job_id,
                    status="cancelled",
                    progress=100,
                    message="Cancellation requested while backend was running.",
                    error_message=None,
                )
            else:
                self._update(job_id, status="failed", error_message=str(exc))
            return

        if record.cancel_event.is_set():
            self._update(
                job_id,
                status="cancelled",
                progress=100,
                message="Cancellation requested while backend was running.",
                result=result,
            )
            return

        self._update(
            job_id,
            status="completed",
            progress=100,
            message=None,
            result=result,
        )


def _validated_upload_filename(filename: str) -> str:
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or any(ord(character) < 32 for character in filename)
    ):
        raise ValueError("Upload filename must be one safe filename component.")
    return filename


def _local_path_from_artifact_uri(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path_text = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        return Path(url2pathname(path_text)).expanduser().resolve()
    if parsed.scheme:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else None


def _publish_result_artifacts(
    result: GenerationResult,
    *,
    job_id: str,
    service: TextureVariationService,
    request: Request,
) -> GenerationResult:
    job_root = (service.output_dir / job_id).resolve()

    def publish(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: publish(item) for key, item in value.items()}
        if isinstance(value, list):
            return [publish(item) for item in value]
        if not isinstance(value, str):
            return value
        local_path = _local_path_from_artifact_uri(value)
        if local_path is None:
            return value
        if local_path.is_relative_to(job_root):
            relative = local_path.relative_to(job_root).as_posix()
            return str(
                request.url_for(
                    "get_texture_variation_artifact",
                    job_id=job_id,
                    artifact_path=relative,
                )
            )
        asset_id = service.upload_id_for_job_path(job_id, local_path)
        if asset_id is not None:
            return str(
                request.url_for(
                    "get_texture_variation_upload",
                    asset_id=asset_id,
                    filename=local_path.name,
                )
            )
        return value

    return GenerationResult.model_validate(publish(result.model_dump(mode="python")))


def create_app(
    *,
    backend: TextureGenerationBackend,
    output_dir: Path,
    title: str,
    version: str = "1.0.0",
    description: str = "",
    service_name: str = "texture-variation-api",
    max_workers: int = 1,
    max_queue_size: int = 0,
    terminal_job_ttl_sec: float = 300.0,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
    upload_ttl_sec: float = 300.0,
) -> FastAPI:
    """Create a FastAPI app for one texture generation backend."""
    if max_upload_bytes <= 0:
        raise ValueError("max_upload_bytes must be positive")
    service = TextureVariationService(
        backend=backend,
        output_dir=output_dir,
        service_name=service_name,
        version=version,
        max_workers=max_workers,
        max_queue_size=max_queue_size,
        terminal_job_ttl_sec=terminal_job_ttl_sec,
        upload_ttl_sec=upload_ttl_sec,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan,
    )
    app.state.texture_variation_service = service

    @app.post(
        "/v1/texture-variation-assets",
        status_code=201,
        response_model=AssetUploadResponse,
    )
    async def upload_texture_variation_asset(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
    ) -> AssetUploadResponse:
        try:
            safe_filename = _validated_upload_filename(filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raw_content_length = request.headers.get("content-length")
        if raw_content_length is not None:
            try:
                content_length = int(raw_content_length)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Content-Length must be an integer.",
                ) from exc
            if content_length > max_upload_bytes:
                raise HTTPException(status_code=413, detail="Upload is too large.")

        asset_id = f"va-{uuid.uuid4().hex}"
        upload_dir = service.output_dir / "_uploads" / asset_id
        destination = upload_dir / safe_filename
        temporary = upload_dir / f".{safe_filename}.part"
        written = 0
        stored = False
        try:
            upload_dir.mkdir(parents=True, exist_ok=False)
            with temporary.open("xb") as target:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Upload is too large.",
                        )
                    target.write(chunk)
            if written <= 0:
                raise HTTPException(status_code=400, detail="Upload is empty.")
            temporary.replace(destination)
            service.register_upload(asset_id, destination)
            stored = True
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Failed to store texture variation input.",
            ) from exc
        finally:
            if not stored:
                shutil.rmtree(upload_dir, ignore_errors=True)
        return AssetUploadResponse(
            asset_uri=str(
                request.url_for(
                    "get_texture_variation_upload",
                    asset_id=asset_id,
                    filename=safe_filename,
                )
            ),
            byte_size=written,
        )

    @app.get(
        "/v1/texture-variation-assets/{asset_id}/{filename}",
        name="get_texture_variation_upload",
        response_class=FileResponse,
    )
    async def get_texture_variation_upload(
        asset_id: str,
        filename: str,
    ) -> FileResponse:
        try:
            path = service.get_upload(asset_id, filename)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Upload not found") from exc
        return FileResponse(path)

    @app.post("/v1/texture-variations", status_code=202, response_model=JobStatus)
    async def create_job(request: CreateJobRequest) -> JobStatus:
        try:
            return service.submit(request)
        except ServiceNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/v1/texture-variations/{job_id}/artifacts/{artifact_path:path}",
        name="get_texture_variation_artifact",
        response_class=FileResponse,
    )
    async def get_texture_variation_artifact(
        job_id: str,
        artifact_path: str,
    ) -> FileResponse:
        try:
            path = service.get_job_artifact(job_id, artifact_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        return FileResponse(path)

    @app.get("/v1/texture-variations/{job_id}", response_model=JobStatus)
    async def get_status(job_id: str, request: Request) -> JobStatus:
        try:
            status = service.get_status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        if status.result is not None:
            status = status.model_copy(
                update={
                    "result": _publish_result_artifacts(
                        status.result,
                        job_id=job_id,
                        service=service,
                        request=request,
                    )
                }
            )
        return status

    @app.delete(
        "/v1/texture-variations/{job_id}",
        status_code=204,
        response_model=None,
    )
    async def cancel_job(job_id: str) -> None:
        try:
            service.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return service.health()

    return app
