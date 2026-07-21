# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU parent dispatcher for the OVRTX rendering API service.

The dispatcher keeps the public API stable while running one private worker
process per GPU. Each worker owns exactly one OVRTX renderer/daemon and keeps
the existing single-flight render lock inside ``Renderer``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests import HTTPError

logger = logging.getLogger(__name__)

_RENDERER_NOT_INITIALIZED = "Renderer not initialized"


@dataclass(frozen=True)
class WorkerSpec:
    """Static configuration for one private OVRTX worker."""

    gpu_id: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class WorkerState:
    """Runtime state for one private OVRTX worker."""

    spec: WorkerSpec
    process: subprocess.Popen[bytes] | None = None
    ready: bool = False
    renderer_initialized: bool = False
    daemon_running: bool = False
    daemon_pid: int | None = None
    daemon_completed_renders: int | None = None
    daemon_rss_bytes: int | None = None
    daemon_recycle_count: int | None = None
    daemon_last_recycle_reason: str | None = None
    daemon_pending_recycle_reason: str | None = None
    in_flight: int = 0
    restart_count: int = 0
    last_error: str | None = None
    last_health_at: float = 0.0
    next_restart_at: float = 0.0
    unhealthy_since: float | None = None
    restart_requested: bool = False
    status: str = "starting"

    def health_payload(self) -> dict[str, Any]:
        return {
            "gpu": self.spec.gpu_id,
            "port": self.spec.port,
            "ready": self.ready,
            "busy": self.in_flight > 0,
            "in_flight": self.in_flight,
            "status": self.status,
            "renderer_initialized": self.renderer_initialized,
            "daemon_running": self.daemon_running,
            "daemon_pid": self.daemon_pid,
            "daemon_completed_renders": self.daemon_completed_renders,
            "daemon_rss_bytes": self.daemon_rss_bytes,
            "daemon_recycle_count": self.daemon_recycle_count,
            "daemon_last_recycle_reason": self.daemon_last_recycle_reason,
            "daemon_pending_recycle_reason": self.daemon_pending_recycle_reason,
            "restart_count": self.restart_count,
            "last_error": self.last_error,
        }


def _clear_daemon_telemetry(worker: WorkerState) -> None:
    """Clear fields owned by the worker's current daemon process."""
    worker.daemon_pid = None
    worker.daemon_completed_renders = None
    worker.daemon_rss_bytes = None
    worker.daemon_recycle_count = None
    worker.daemon_last_recycle_reason = None
    worker.daemon_pending_recycle_reason = None


def parse_gpu_workers(raw_value: str | None) -> list[str]:
    """Parse ``OVRTX_GPU_WORKERS`` into GPU ids.

    Accepts a comma-separated GPU id list (``0,1,3``). A single positive
    integer without a comma is treated as a worker count, so ``2`` becomes
    ``["0", "1"]``. ``0`` disables dispatcher mode.
    """
    if raw_value is None or raw_value.strip() == "":
        return []

    value = raw_value.strip()
    if "," not in value:
        try:
            count = int(value)
        except ValueError:
            return [value]
        if count <= 0:
            return []
        return [str(index) for index in range(count)]

    return [part.strip() for part in value.split(",") if part.strip()]


class OVRTXDispatcher:
    """Supervisor and local load balancer for per-GPU OVRTX workers."""

    def __init__(
        self,
        *,
        gpu_ids: list[str],
        port_base: int = 8100,
        parent_port: int = 8000,
        health_interval_seconds: float = 5.0,
        worker_start_stagger_seconds: float = 0.0,
        request_timeout_seconds: float = 3600.0,
        queue_timeout_seconds: float = 60.0,
        restart_cooldown_seconds: float = 10.0,
    ) -> None:
        if not gpu_ids:
            raise ValueError("OVRTXDispatcher requires at least one GPU id")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("duplicate GPU id in OVRTX_GPU_WORKERS")
        if port_base <= parent_port < port_base + len(gpu_ids):
            raise ValueError(
                "OVRTX worker port range collides with parent service port "
                f"{parent_port}"
            )
        self._workers = [
            WorkerState(spec=WorkerSpec(gpu_id=gpu_id, port=port_base + index))
            for index, gpu_id in enumerate(gpu_ids)
        ]
        self._health_interval_seconds = health_interval_seconds
        self._worker_start_stagger_seconds = worker_start_stagger_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._queue_timeout_seconds = queue_timeout_seconds
        self._restart_cooldown_seconds = restart_cooldown_seconds
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._monitor_task: asyncio.Task[None] | None = None

    @property
    def total_workers(self) -> int:
        return len(self._workers)

    @property
    def ready_workers(self) -> int:
        with self._condition:
            return sum(1 for worker in self._workers if worker.ready)

    async def start(self) -> None:
        """Start all private workers and begin health monitoring."""
        started_workers: list[WorkerState] = []
        try:
            for index, worker in enumerate(self._workers):
                self._start_worker(worker)
                started_workers.append(worker)
                if (
                    self._worker_start_stagger_seconds > 0
                    and index < len(self._workers) - 1
                ):
                    await asyncio.sleep(self._worker_start_stagger_seconds)
        except Exception:
            logger.exception("OVRTX dispatcher startup failed; stopping workers")
            for worker in started_workers:
                self._stop_worker_process(worker)
            raise
        self._monitor_task = asyncio.create_task(self._monitor_workers())

    async def stop(self) -> None:
        """Stop monitor and terminate worker processes."""
        self._stop_event.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        deadline = time.monotonic() + 15.0
        for worker in self._workers:
            remaining = max(0.0, deadline - time.monotonic())
            self._stop_worker_process(worker, timeout_seconds=remaining)

    def render(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one render request to a ready private worker."""
        worker: WorkerState | None = None
        restart_worker = False
        try:
            worker = self._acquire_worker()
            logger.info(
                "Dispatching render to OVRTX worker gpu=%s port=%d",
                worker.spec.gpu_id,
                worker.spec.port,
            )
            response = requests.post(
                f"{worker.spec.base_url}/render",
                json=request_payload,
                timeout=self._request_timeout_seconds,
            )
            if response.status_code >= 500:
                response.raise_for_status()
            if response.status_code >= 400:
                return _worker_http_error_response(response)
            payload = response.json()
            if _is_renderer_not_initialized_response(payload):
                logger.warning(
                    "OVRTX worker gpu=%s reported an uninitialized renderer during "
                    "render; restarting worker",
                    worker.spec.gpu_id,
                )
                self._mark_worker_unhealthy(
                    worker,
                    _RENDERER_NOT_INITIALIZED,
                    restart_immediately=True,
                )
                restart_worker = True
            return payload
        except (requests.ConnectionError, requests.Timeout, HTTPError) as exc:
            if worker is not None:
                self._mark_worker_unhealthy(worker, str(exc))
                logger.exception(
                    "OVRTX worker gpu=%s render failed",
                    worker.spec.gpu_id,
                )
            else:
                logger.warning("OVRTX render dispatch failed before worker selection")
            return {"status": "exception", "error": str(exc), "images": {}}
        except Exception as exc:
            logger.exception("OVRTX render dispatch failed")
            return {"status": "exception", "error": str(exc), "images": {}}
        finally:
            if worker is not None:
                with self._condition:
                    worker.in_flight = max(0, worker.in_flight - 1)
                    self._condition.notify_all()
                if restart_worker:
                    self._restart_unhealthy_worker_if_due(worker)

    def health(self) -> dict[str, Any]:
        """Return aggregate dispatcher health."""
        with self._condition:
            workers = [worker.health_payload() for worker in self._workers]
            ready_workers = sum(1 for worker in self._workers if worker.ready)
            any_process_alive = any(
                worker.process is not None and worker.process.poll() is None
                for worker in self._workers
            )

        if ready_workers > 0:
            status = "healthy"
        elif any_process_alive:
            status = "initializing"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "service": "ovrtx-rendering-api",
            "version": "0.1.0",
            "renderer": "ovrtx",
            "gpu_initialized": ready_workers > 0,
            "renderer_initialized": ready_workers > 0,
            "daemon_running": ready_workers > 0,
            "ready_workers": ready_workers,
            "total_workers": len(workers),
            "workers": workers,
        }

    def _start_worker(self, worker: WorkerState) -> None:
        env = os.environ.copy()
        env["OVRTX_WORKER_MODE"] = "1"
        env["OVRTX_WORKER_GPU_INDEX"] = worker.spec.gpu_id
        env["CUDA_VISIBLE_DEVICES"] = worker.spec.gpu_id
        env["NVIDIA_VISIBLE_DEVICES"] = worker.spec.gpu_id

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "service.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(worker.spec.port),
        ]
        logger.info(
            "Starting OVRTX worker gpu=%s port=%d",
            worker.spec.gpu_id,
            worker.spec.port,
        )
        with self._condition:
            worker.status = "starting"
            worker.ready = False
            worker.renderer_initialized = False
            worker.daemon_running = False
            _clear_daemon_telemetry(worker)
            worker.last_error = None
            worker.unhealthy_since = None
            worker.restart_requested = False
            worker.process = None
            self._condition.notify_all()

        process = subprocess.Popen(cmd, env=env)
        with self._condition:
            worker.process = process
            self._condition.notify_all()

    async def _monitor_workers(self) -> None:
        while not self._stop_event.is_set():
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(self._check_worker, worker)
                    for worker in self._workers
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.error(
                        "OVRTX worker monitor check failed",
                        exc_info=(type(result), result, result.__traceback__),
                    )
            await asyncio.sleep(self._health_interval_seconds)

    def _check_worker(self, worker: WorkerState) -> None:
        process = worker.process
        if process is not None and process.poll() is not None:
            self._mark_worker_exited(worker, process.returncode)
            self._restart_worker_if_due(worker)
            return

        try:
            response = requests.get(
                f"{worker.spec.base_url}/health",
                timeout=2.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            with self._condition:
                worker.ready = False
                worker.renderer_initialized = False
                worker.daemon_running = False
                _clear_daemon_telemetry(worker)
                worker.status = (
                    "starting" if worker.status == "starting" else "unhealthy"
                )
                worker.last_error = str(exc)
                worker.last_health_at = time.time()
                if worker.status != "starting" and worker.unhealthy_since is None:
                    worker.unhealthy_since = time.monotonic()
                self._condition.notify_all()
            self._restart_unhealthy_worker_if_due(worker)
            return

        with self._condition:
            worker.ready = bool(payload.get("gpu_initialized"))
            worker.renderer_initialized = bool(payload.get("renderer_initialized"))
            worker.daemon_running = bool(payload.get("daemon_running"))
            worker.daemon_pid = payload.get("daemon_pid")
            worker.daemon_completed_renders = payload.get("daemon_completed_renders")
            worker.daemon_rss_bytes = payload.get("daemon_rss_bytes")
            worker.daemon_recycle_count = payload.get("daemon_recycle_count")
            worker.daemon_last_recycle_reason = payload.get(
                "daemon_last_recycle_reason"
            )
            worker.daemon_pending_recycle_reason = payload.get(
                "daemon_pending_recycle_reason"
            )
            worker.status = str(payload.get("status", "unknown"))
            worker.last_error = None if worker.ready else payload.get("error")
            worker.last_health_at = time.time()
            if worker.ready or worker.status == "initializing":
                worker.unhealthy_since = None
            elif worker.unhealthy_since is None:
                worker.unhealthy_since = time.monotonic()
            self._condition.notify_all()
        self._restart_unhealthy_worker_if_due(worker)

    def _mark_worker_exited(self, worker: WorkerState, returncode: int | None) -> None:
        with self._condition:
            if worker.status != "exited":
                worker.next_restart_at = (
                    time.monotonic() + self._restart_cooldown_seconds
                )
            worker.ready = False
            worker.renderer_initialized = False
            worker.daemon_running = False
            _clear_daemon_telemetry(worker)
            worker.status = "exited"
            worker.last_error = f"worker exited with code {returncode}"
            worker.unhealthy_since = None
            worker.restart_requested = False
            self._condition.notify_all()

    def _restart_worker_if_due(self, worker: WorkerState) -> None:
        with self._condition:
            if time.monotonic() < worker.next_restart_at:
                return
            worker.restart_count += 1
        self._start_worker(worker)

    def _restart_unhealthy_worker_if_due(self, worker: WorkerState) -> None:
        process_to_stop: subprocess.Popen[bytes] | None = None
        with self._condition:
            if worker.in_flight > 0:
                return
            if not worker.restart_requested:
                if worker.ready or worker.unhealthy_since is None:
                    return
                if (
                    time.monotonic() - worker.unhealthy_since
                    < self._restart_cooldown_seconds
                ):
                    return
            process_to_stop = worker.process
            worker.process = None
            worker.restart_count += 1
            worker.status = "restarting"
            worker.ready = False
            worker.renderer_initialized = False
            worker.daemon_running = False
            _clear_daemon_telemetry(worker)
            worker.last_error = (
                worker.last_error or "worker remained unhealthy; restarting"
            )
            worker.unhealthy_since = None
            worker.restart_requested = False
            self._condition.notify_all()

        if process_to_stop is not None and process_to_stop.poll() is None:
            self._stop_process(process_to_stop, worker.spec.gpu_id, timeout_seconds=5.0)
        self._start_worker(worker)

    def _mark_worker_unhealthy(
        self,
        worker: WorkerState,
        error: str,
        *,
        restart_immediately: bool = False,
    ) -> None:
        with self._condition:
            worker.ready = False
            worker.renderer_initialized = False
            worker.daemon_running = False
            _clear_daemon_telemetry(worker)
            worker.status = "unhealthy"
            worker.last_error = error
            now = time.monotonic()
            if restart_immediately:
                worker.unhealthy_since = now - self._restart_cooldown_seconds
                worker.restart_requested = True
            elif worker.unhealthy_since is None:
                worker.unhealthy_since = now
            self._condition.notify_all()

    def _stop_worker_process(
        self,
        worker: WorkerState,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        process = worker.process
        if process is None:
            return
        self._stop_process(process, worker.spec.gpu_id, timeout_seconds=timeout_seconds)

    @staticmethod
    def _stop_process(
        process: subprocess.Popen[bytes],
        gpu_id: str,
        *,
        timeout_seconds: float,
    ) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
                return
            except subprocess.TimeoutExpired:
                logger.warning("Killing OVRTX worker on GPU %s", gpu_id)
                process.kill()

        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("OVRTX worker on GPU %s did not exit after kill", gpu_id)

    def _acquire_worker(self) -> WorkerState:
        deadline = time.monotonic() + self._queue_timeout_seconds
        with self._condition:
            while True:
                ready_workers = [worker for worker in self._workers if worker.ready]
                if ready_workers:
                    worker = min(
                        ready_workers,
                        key=lambda item: (item.in_flight, item.spec.port),
                    )
                    if worker.in_flight == 0:
                        worker.in_flight += 1
                        return worker

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a ready OVRTX worker")
                self._condition.wait(timeout=min(1.0, remaining))


def _worker_http_error_response(response: requests.Response) -> dict[str, Any]:
    """Convert worker 4xx responses into the renderer's JSON error envelope."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict) and detail.get("status") == "blank_render":
            return {**detail, "images": detail.get("images", {})}

    message = response.text[:500]
    return {
        "status": "exception",
        "error": f"OVRTX worker returned HTTP {response.status_code}: {message}",
        "images": {},
    }


def _is_renderer_not_initialized_response(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "exception":
        return False
    error = payload.get("error")
    if not isinstance(error, str):
        return False
    normalized_error = error.strip()
    if normalized_error == _RENDERER_NOT_INITIALIZED:
        return True
    if _RENDERER_NOT_INITIALIZED in normalized_error:
        logger.warning(
            "OVRTX worker returned renderer-not-initialized text that did not "
            "match the exact restart sentinel: %r",
            normalized_error,
        )
    return False
