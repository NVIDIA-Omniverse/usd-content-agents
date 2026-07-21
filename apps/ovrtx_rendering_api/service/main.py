# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering API service using OVRTX.

A drop-in replacement for the Kit-based rendering-api container. Exposes
the same POST /render and GET /health endpoints with identical request/response
schemas, but uses OVRTX for local RTX rendering instead of Kit SDK.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from service.dispatcher import OVRTXDispatcher, parse_gpu_workers
from service.models import HealthResponse, RenderRequest
from service.renderer import Renderer

_renderer: Renderer | None = None
_warmup_task: asyncio.Task | None = None
_dispatcher: OVRTXDispatcher | None = None


def _configure_logging(root_logger: logging.Logger | None = None) -> None:
    """Ensure OVRTX startup logs are visible under uvicorn and tests."""
    root_logger = root_logger or logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


_configure_logging()
logger = logging.getLogger(__name__)


def _dispatcher_gpu_ids() -> list[str]:
    """Return configured dispatcher GPU ids for the parent process."""
    if os.environ.get("OVRTX_WORKER_MODE") == "1":
        return []
    return parse_gpu_workers(os.environ.get("OVRTX_GPU_WORKERS"))


def _create_renderer_sync() -> Renderer:
    """Construct the Renderer without running warm-up."""
    log_level = os.environ.get("OVRTX_LOG_LEVEL", "warn")
    # Kit-parity defaults: mode=pt + 500 step() iterations per frame
    # hits the convergence plateau at ~39.7 dB PSNR vs Kit on the
    # kit-gen-ai-service golden scene (see the cap sweep at
    # /tmp/ovrtx_cap.py). rt2 is available as an override for callers
    # that want real-time-path-tracing speed at the cost of ~12 dB
    # quality. ``num_sensor_updates`` here is ITERATION COUNT, not SPP. In
    # 0.2.0 validation the bundled samplesPerPixel / accumulationLimit schema
    # attributes were silently ignored (verified in /tmp/ovrtx_verify.py);
    # keep the step-loop guard until 0.3 GPU validation proves otherwise.
    num_sensor_updates = int(os.environ.get("OVRTX_NUM_SENSOR_UPDATES", "500"))
    render_mode = os.environ.get("OVRTX_RENDER_MODE", "pt")
    logger.info(
        "Starting OVRTX renderer (log_level=%s, num_sensor_updates=%d, render_mode=%s)",
        log_level,
        num_sensor_updates,
        render_mode,
    )
    logger.info(
        "OVRTX warm-up is running in the background; "
        "/health will report gpu_initialized=false until it completes. "
        "Cold startup commonly takes around 5 minutes."
    )
    renderer = Renderer(
        log_level=log_level,
        num_sensor_updates=num_sensor_updates,
        render_mode=render_mode,
    )
    return renderer


def _warm_up_renderer_sync(renderer: Renderer) -> None:
    """Run renderer warm-up and log failure without dropping the renderer."""
    if not renderer.warm_up():
        logger.error(
            "OVRTX warm-up failed; service is up but /health will report "
            "gpu_initialized=false until the renderer succeeds at least once."
        )


async def _background_init() -> None:
    """Initialize the renderer off the event loop."""
    global _renderer
    try:
        renderer = await asyncio.to_thread(_create_renderer_sync)
        _renderer = renderer
        await asyncio.to_thread(_warm_up_renderer_sync, renderer)
    except Exception:
        logger.exception("OVRTX renderer initialization failed")


async def _start_dispatcher() -> OVRTXDispatcher:
    """Start the multi-GPU parent dispatcher."""
    gpu_ids = _dispatcher_gpu_ids()
    parent_port = int(os.environ.get("OVRTX_PARENT_PORT", "8000"))
    port_base = int(os.environ.get("OVRTX_WORKER_PORT_BASE", "8100"))
    health_interval = float(os.environ.get("OVRTX_WORKER_HEALTH_INTERVAL", "5"))
    warmup_stagger = float(os.environ.get("OVRTX_WORKER_WARMUP_STAGGER_SECONDS", "0"))
    request_timeout = float(os.environ.get("OVRTX_WORKER_REQUEST_TIMEOUT", "3600"))
    queue_timeout = float(os.environ.get("OVRTX_WORKER_QUEUE_TIMEOUT", "60"))
    restart_cooldown = float(os.environ.get("OVRTX_WORKER_RESTART_COOLDOWN", "10"))
    dispatcher = OVRTXDispatcher(
        gpu_ids=gpu_ids,
        parent_port=parent_port,
        port_base=port_base,
        health_interval_seconds=health_interval,
        worker_start_stagger_seconds=warmup_stagger,
        request_timeout_seconds=request_timeout,
        queue_timeout_seconds=queue_timeout,
        restart_cooldown_seconds=restart_cooldown,
    )
    await dispatcher.start()
    return dispatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Schedule OVRTX renderer init in background so the app can serve
    /health immediately — orchestrators checking `service_healthy` don't
    block on native GPU init, which can take tens of seconds or hang on
    misconfigured hosts. /health reports gpu_initialized=true only once
    the background task has completed a successful warm_up.
    """
    global _dispatcher, _warmup_task
    dispatcher_gpu_ids = _dispatcher_gpu_ids()
    if dispatcher_gpu_ids:
        logger.info(
            "Starting OVRTX multi-GPU dispatcher for GPUs: %s",
            ",".join(dispatcher_gpu_ids),
        )
        _dispatcher = await _start_dispatcher()
        yield
        logger.info("Shutting down OVRTX multi-GPU dispatcher")
        await _dispatcher.stop()
        _dispatcher = None
        return

    logger.info("Scheduling OVRTX warm-up task")
    _warmup_task = asyncio.create_task(_background_init())

    yield

    logger.info("Shutting down OVRTX renderer")
    if _warmup_task is not None and not _warmup_task.done():
        _warmup_task.cancel()
    if _renderer is not None:
        _renderer.shutdown()


app = FastAPI(
    title="OVRTX Rendering API",
    description="USD rendering service using OVRTX local RTX renderer",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    if _dispatcher is not None:
        return HealthResponse(**_dispatcher.health())

    renderer = _renderer
    initializing = _warmup_task is not None and not _warmup_task.done()
    if renderer is None:
        status = "initializing" if initializing else "unhealthy"
        return HealthResponse(status=status)

    renderer_initialized = renderer.is_initialized
    daemon_running = renderer.daemon_running
    gpu_initialized = renderer.is_ready
    if gpu_initialized:
        status = "healthy"
    elif initializing:
        status = "initializing"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        gpu_initialized=gpu_initialized,
        renderer_initialized=renderer_initialized,
        daemon_running=daemon_running,
        **renderer.daemon_lifecycle,
    )


@app.post("/render")
def render(request: RenderRequest) -> dict[str, Any]:
    """Render a USD file.

    Accepts the same request body as the Kit-based rendering-api and returns
    the same V1 response format (images[frame][camera][sensor] = base64).

    This is a sync ``def`` (not ``async def``) so uvicorn runs it in a
    thread pool.  The OVRTX daemon serialises renders internally, so
    there is no concurrency benefit from ``async``; making it async would
    block the event loop and starve health-check responses, causing the
    orchestrator to kill the pod.
    """
    if _dispatcher is not None:
        return _dispatcher.render(request.model_dump())

    if _renderer is None:
        return {
            "status": "exception",
            "error": "Renderer not initialized",
            "images": {},
        }
    if not _renderer.is_ready:
        if _warmup_task is not None and not _warmup_task.done():
            return {
                "status": "exception",
                "error": "Renderer not initialized",
                "images": {},
            }
        logger.warning("Renderer not initialized; attempting recovery before render")
        if not _renderer.recover():
            return {
                "status": "exception",
                "error": "Renderer not initialized",
                "images": {},
            }

    settings = request.render_settings
    result = _renderer.render(
        url=request.url,
        camera_paths=settings.camera_paths,
        frame_start=settings.frame_range.start,
        frame_end=settings.frame_range.end,
        width=settings.camera_parameters.width,
        height=settings.camera_parameters.height,
        sensors=settings.sensors,
        num_sensor_updates=settings.num_sensor_updates,
        render_mode=settings.render_mode,
        material_target=settings.material_target,
    )
    return result
