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
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .backend import TextureGenerationBackend, TextureGenerationBackendError
from .models import CreateJobRequest, HealthResponse, JobStatus

logger = logging.getLogger(__name__)
_TERMINAL_STATES = {"completed", "failed", "cancelled"}


class ServiceBusyError(RuntimeError):
    """Raised when the service queue is full."""


class ServiceNotReadyError(RuntimeError):
    """Raised when the backend is not ready to accept work."""


class _JobRecord:
    def __init__(self, status: JobStatus) -> None:
        self.status = status
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
    ) -> None:
        self.backend = backend
        self.output_dir = output_dir
        self.service_name = service_name
        self.version = version
        self.max_workers = max(1, max_workers)
        self.max_queue_size = max(0, max_queue_size)
        self.terminal_job_ttl_sec = max(0.0, terminal_job_ttl_sec)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._jobs: dict[str, _JobRecord] = {}
        self._pending_output_cleanup: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, request: CreateJobRequest) -> JobStatus:
        """Submit a request and return queued status."""
        backend_health = self.backend.health()
        if not backend_health.ready:
            raise ServiceNotReadyError(backend_health.error or "Backend is not ready.")

        job_id = f"vj-{uuid.uuid4().hex[:12]}"
        status = JobStatus(job_id=job_id, status="queued", progress=0)
        record = _JobRecord(status)
        expired_job_ids: list[str] = []
        busy_error: ServiceBusyError | None = None
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            active_jobs = self._count_status_locked("processing")
            queued_jobs = self._count_status_locked("queued")
            if active_jobs + queued_jobs >= self.max_workers + self.max_queue_size:
                busy_error = ServiceBusyError("Texture generation queue is full.")
            else:
                self._jobs[job_id] = record
                record.future = self._executor.submit(
                    self._run_job,
                    job_id,
                    request,
                    record,
                )
        self._cleanup_job_outputs(expired_job_ids)
        if busy_error is not None:
            raise busy_error
        return status.model_copy(deep=True)

    def get_status(self, job_id: str) -> JobStatus:
        """Return a job status copy."""
        expired_job_ids: list[str] = []
        status: JobStatus | None = None
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            record = self._jobs.get(job_id)
            if record is not None:
                status = record.status.model_copy(deep=True)
        self._cleanup_job_outputs(expired_job_ids)
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
        with self._lock:
            expired_job_ids = self._evict_terminal_locked()
            active_jobs = self._count_status_locked("processing")
            queued_jobs = self._count_status_locked("queued")
        self._cleanup_job_outputs(expired_job_ids)
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

    def _cleanup_job_outputs(self, job_ids: list[str]) -> None:
        for job_id in job_ids:
            if not self._cleanup_job_output(job_id):
                continue
            with self._lock:
                self._pending_output_cleanup.discard(job_id)

    def _cleanup_job_output(self, job_id: str) -> bool:
        job_dir = self.output_dir / job_id
        try:
            shutil.rmtree(job_dir)
        except FileNotFoundError:
            return True
        except OSError:
            logger.warning("Failed to clean texture generation job output: %s", job_dir)
            return False
        return True

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
) -> FastAPI:
    """Create a FastAPI app for one texture generation backend."""
    service = TextureVariationService(
        backend=backend,
        output_dir=output_dir,
        service_name=service_name,
        version=version,
        max_workers=max_workers,
        max_queue_size=max_queue_size,
        terminal_job_ttl_sec=terminal_job_ttl_sec,
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

    @app.post("/v1/texture-variations", status_code=202, response_model=JobStatus)
    async def create_job(request: CreateJobRequest) -> JobStatus:
        try:
            return service.submit(request)
        except ServiceNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/v1/texture-variations/{job_id}", response_model=JobStatus)
    async def get_status(job_id: str) -> JobStatus:
        try:
            return service.get_status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

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
