# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering API service using OVRTX.

A drop-in replacement for the Kit-based rendering-api container. Exposes
the same POST /render and GET /health endpoints with identical request/response
schemas, but uses OVRTX for local RTX rendering instead of Kit SDK.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import os
import sys
import zlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from service.dispatcher import OVRTXDispatcher, parse_gpu_workers
from service.models import (
    HealthResponse,
    ProtocolV3RenderResponse,
    ProtocolV3RenderUploadParams,
    RenderRequest,
)
from service.protocol import PROTOCOL_VERSION
from service.renderer import IncompleteRenderOutputError, Renderer

_renderer: Renderer | None = None
_warmup_task: asyncio.Task | None = None
_dispatcher: OVRTXDispatcher | None = None
_MAX_BODY_BYTES = int(os.environ.get("OVRTX_MAX_BODY_BYTES", str(70 * 1024 * 1024)))
_MAX_SCENE_BYTES = _MAX_BODY_BYTES * 8
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_MAX_PROTOCOL_REQUEST_BYTES = _MAX_BODY_BYTES + _MULTIPART_OVERHEAD_BYTES


class _RequestBodyLimitExceeded(RuntimeError):
    """Raised before multipart parsing when a protocol upload exceeds its cap."""


class _ProtocolUploadBodyLimitMiddleware:
    """Bound `/render/upload` bytes at the ASGI receive layer.

    FastAPI resolves ``UploadFile`` before calling the endpoint, so handler-level
    reads cannot stop Starlette from spooling an arbitrarily large multipart body.
    This wrapper counts bytes before form parsing and reserves a small bounded
    allowance above the advertised file-part cap for JSON parameters and framing.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/render/upload"
        ):
            await self.app(scope, receive, send)
            return

        content_lengths = [
            value
            for name, value in scope.get("headers", ())
            if name.lower() == b"content-length"
        ]
        if content_lengths:
            try:
                if len(content_lengths) != 1:
                    raise ValueError
                content_length = int(content_lengths[0].decode("ascii"))
                if content_length < 0:
                    raise ValueError
            except (UnicodeDecodeError, ValueError):
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "invalid content-length"},
                )
                await response(scope, receive, send)
                return
            if content_length > self.max_request_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "request body too large"},
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_bytes:
                    raise _RequestBodyLimitExceeded
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyLimitExceeded:
            if response_started:
                raise
            response = JSONResponse(
                status_code=413,
                content={"detail": "request body too large"},
            )
            await response(scope, receive, send)


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
app.add_middleware(
    _ProtocolUploadBodyLimitMiddleware,
    max_request_bytes=_MAX_PROTOCOL_REQUEST_BYTES,
)

_default_openapi = app.openapi


def _openapi_with_nvcf_version() -> dict[str, Any]:
    """Build OpenAPI metadata that identifies the serving NVCF version."""
    schema: dict[str, Any] = _default_openapi()
    if version_id := os.getenv("NVCF_FUNCTION_VERSION_ID"):
        schema["info"]["x-nvcf-function-version-id"] = version_id
    return schema


app.openapi = _openapi_with_nvcf_version


@app.get("/live", response_model=None)
async def live() -> dict[str, Any] | JSONResponse:
    """Public package-owned usd-cli protocol identity.

    The v3 adapter intentionally advertises no optional transports: clients use
    the bounded gzip/none multipart floor and never infer CAS or zstd support.
    """
    return {
        "status": "alive",
        "protocol_version": PROTOCOL_VERSION,
        "engine": "ovrtx",
        "renderer": "ovrtx",
        "max_body_bytes": _MAX_BODY_BYTES,
        "max_scene_bytes": _MAX_SCENE_BYTES,
        "features": [],
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    if _dispatcher is not None:
        return HealthResponse(**_dispatcher.health())

    renderer = _renderer
    initializing = _warmup_task is not None and not _warmup_task.done()
    if renderer is None:
        status = "initializing" if initializing else "unhealthy"
        return HealthResponse(status=status, protocol_version=PROTOCOL_VERSION)

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
        protocol_version=PROTOCOL_VERSION,
        gpu_initialized=gpu_initialized,
        renderer_initialized=renderer_initialized,
        daemon_running=daemon_running,
        **renderer.daemon_lifecycle,
    )


def _render_http_response(result: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    """Map typed retryable renderer failures to an HTTP retry signal."""
    if result.get("retryable") is True:
        return JSONResponse(
            status_code=503,
            content=result,
            headers={"Retry-After": "1"},
        )
    return result


@app.post("/render", response_model=None)
def render(request: RenderRequest) -> dict[str, Any] | JSONResponse:
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
        return _render_http_response(_dispatcher.render(request.model_dump()))

    if _renderer is None:
        return _render_http_response(
            {
                "status": "exception",
                "error": "Renderer not initialized",
                "images": {},
            }
        )
    if not _renderer.is_ready:
        if _warmup_task is not None and not _warmup_task.done():
            return _render_http_response(
                {
                    "status": "exception",
                    "error": "Renderer not initialized",
                    "images": {},
                }
            )
        logger.warning("Renderer not initialized; attempting recovery before render")
        if not _renderer.recover():
            return _render_http_response(
                {
                    "status": "exception",
                    "error": "Renderer not initialized",
                    "images": {},
                }
            )

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
    return _render_http_response(result)


def _inflate_protocol_v3_upload(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as archive:
            expanded = archive.read(_MAX_SCENE_BYTES + 1)
    except (OSError, EOFError, zlib.error) as exc:
        raise HTTPException(
            status_code=400, detail="file is not valid gzip data"
        ) from exc
    if len(expanded) > _MAX_SCENE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="gzip body inflates past the scene size limit",
        )
    return expanded


@app.post("/render/upload", response_model=ProtocolV3RenderResponse)
async def render_upload(
    file: UploadFile, params: str = Form(...)
) -> ProtocolV3RenderResponse | JSONResponse:
    """Render package-owned usd-cli protocol-v3 multipart input."""
    try:
        parsed = ProtocolV3RenderUploadParams.model_validate_json(params)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid params: {exc}") from exc
    if _dispatcher is None and (_renderer is None or not _renderer.is_ready):
        raise HTTPException(status_code=503, detail="renderer is not ready")
    try:
        data = await file.read(_MAX_BODY_BYTES + 1)
    finally:
        await file.close()
    if len(data) > _MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    if _dispatcher is not None:
        dispatched = await asyncio.to_thread(
            _dispatcher.render_protocol_v3_upload,
            data=data,
            filename=file.filename or "scene.usdz",
            content_type=file.content_type or "application/octet-stream",
            params=parsed.model_dump_json(),
        )
        if dispatched.status_code >= 400:
            headers = (
                {"Retry-After": dispatched.retry_after}
                if dispatched.retry_after is not None
                else None
            )
            return JSONResponse(
                status_code=dispatched.status_code,
                content=dispatched.payload,
                headers=headers,
            )
        try:
            return ProtocolV3RenderResponse.model_validate(dispatched.payload)
        except ValidationError:
            logger.exception("Protocol-v3 worker returned an invalid response")
            return JSONResponse(
                status_code=502,
                content={"detail": "invalid protocol-v3 response from renderer worker"},
            )
    data = _inflate_protocol_v3_upload(data, parsed.compression)
    if _renderer is None or not _renderer.is_ready:
        raise HTTPException(status_code=503, detail="renderer is not ready")
    try:
        items = await asyncio.to_thread(
            _renderer.render_protocol_v3_upload,
            usdz_bytes=data,
            camera_paths=parsed.cameras,
            width=parsed.image_width,
            height=parsed.image_height,
            mode=parsed.mode,
            frames=parsed.frames,
            camera_defs=parsed.camera_defs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IncompleteRenderOutputError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        logger.exception("Protocol-v3 render failed")
        raise HTTPException(
            status_code=500, detail="protocol-v3 render failed"
        ) from exc
    return ProtocolV3RenderResponse(results=items)
