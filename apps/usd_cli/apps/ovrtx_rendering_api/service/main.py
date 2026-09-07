# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""usd-cli OVRTX rendering API.

FastAPI service that renders USD via usd_core's local OVRTX backend. Public `/live` is
liveness-only; authenticated `/ready` becomes successful after background GPU warm-up.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import os
import secrets
import shutil
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import Headers
from usd_core.remote_protocol import PROTOCOL_VERSION

from service import cas
from service.models import (
    HealthResponse,
    ManifestRenderRequest,
    NegotiateRequest,
    NegotiateResponse,
    PhysicsResponse,
    PhysicsUploadParams,
    RenderRequest,
    RenderResponse,
    RenderResultItem,
    RenderUploadParams,
)
from service.physics import Simulator
from service.renderer import Renderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_renderer: Renderer | None = None
_warmup_task: asyncio.Task | None = None
_init_state = "initializing"
_init_error: str | None = None
_render_lock = asyncio.Lock()
_simulator: Simulator | None = None
_physics_lock = asyncio.Lock()
MAX_BODY_BYTES = int(os.environ.get("OVRTX_MAX_BODY_BYTES", str(70 * 1024 * 1024)))
RENDER_TIMEOUT_S = float(os.environ.get("OVRTX_REQUEST_TIMEOUT", "1800"))
PHYSICS_TIMEOUT_S = float(os.environ.get("OVRTX_PHYSICS_TIMEOUT", "1800"))
# The staged/decompressed scene ceiling. ×8 matches the zstd decompression bound the
# bundle transport already applies — a manifest scene may not exceed what a maximally
# compressed bundle could have delivered. Advertised on /live as `max_scene_bytes`.
MAX_SCENE_BYTES = MAX_BODY_BYTES * 8
_cas_store: cas.BlobStore | None = None
_cas_lock = threading.Lock()


def _get_cas() -> cas.BlobStore:
    """The process-wide blob store (upload-dedup Phase 2), created lazily from
    OVRTX_CAS_DIR / OVRTX_CAS_MAX_BYTES (default: <tmp>/ovrtx_cas, 20 GiB)."""
    global _cas_store
    with _cas_lock:
        if _cas_store is None:
            root = os.environ.get("OVRTX_CAS_DIR") or os.path.join(
                tempfile.gettempdir(), "ovrtx_cas")
            max_bytes = int(os.environ.get("OVRTX_CAS_MAX_BYTES",
                                           str(20 * 1024 ** 3)))
            _cas_store = cas.BlobStore(root, max_bytes=max_bytes)
            logger.info("CAS blob store at %s (quota %.1f GB)",
                        root, max_bytes / 1024 ** 3)
        return _cas_store


def _create_renderer_sync() -> Renderer:
    log_level = os.environ.get("OVRTX_LOG_LEVEL", "warn")
    num_sensor_updates = int(os.environ.get("OVRTX_NUM_SENSOR_UPDATES", "64"))
    render_mode = os.environ.get("OVRTX_RENDER_MODE", "")
    logger.info(
        "Starting OVRTX renderer (log_level=%s, num_sensor_updates=%d, render_mode=%r); "
        "warm-up runs in the background — /health reports gpu_initialized=false until ready.",
        log_level, num_sensor_updates, render_mode,
    )
    return Renderer(log_level=log_level, num_sensor_updates=num_sensor_updates,
                    render_mode=render_mode)


async def _background_init() -> None:
    global _renderer, _init_state, _init_error
    try:
        renderer = await asyncio.to_thread(_create_renderer_sync)
        _renderer = renderer
        ok = await asyncio.to_thread(renderer.warm_up)
        if not ok:
            _init_state = "failed"
            _init_error = "OVRTX warm-up failed"
            logger.error("OVRTX warm-up failed; /health stays gpu_initialized=false.")
        else:
            _init_state = "ready"
    except Exception:  # noqa: BLE001
        _init_state = "failed"
        _init_error = "renderer initialization failed"
        logger.exception("OVRTX renderer initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warmup_task
    _warmup_task = asyncio.create_task(_background_init())
    yield
    if _warmup_task and not _warmup_task.done():
        _warmup_task.cancel()
    if _renderer is not None:
        await asyncio.to_thread(_renderer.close)
    if _simulator is not None:
        await asyncio.to_thread(_simulator.close)


app = FastAPI(title="usd-cli OVRTX Rendering API", version="0.1.0", lifespan=lifespan)


def authenticate(authorization: str | None = Header(default=None)) -> None:
    api_key = os.environ.get("OVRTX_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OVRTX_API_KEY is not configured")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(supplied, api_key):
        raise HTTPException(status_code=401, detail="invalid API key")


class _BodyTooLarge(Exception):
    """Raised inside the receive channel when a body passes the ceiling."""


class BodySizeLimitMiddleware:
    """Bound request bodies by the bytes that actually arrive.

    `Content-Length` is only a promise: chunked transfer omits it entirely, and
    a body may exceed whatever it declares. The limit therefore has to be
    enforced on the ASGI receive channel, before anything parses — by the time
    an endpoint reaches `await request.body()` or an `UploadFile`, the payload
    is already buffered in memory or spooled to disk, and body parsing is not
    ordered after `Depends(authenticate)`, so an unauthenticated client would
    otherwise decide how much this service stores.

    The bytes are counted, never accumulated, so multipart uploads keep
    streaming to their spool file exactly as before.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_length = Headers(scope=scope).get("content-length")
        if raw_length:
            try:
                declared = int(raw_length)
                if declared < 0:
                    raise ValueError
            except ValueError:
                await self._respond(scope, receive, send, 400, "invalid content-length")
                return
            if declared > self.max_bytes:
                await self._respond(scope, receive, send, 413, "request body too large")
                return

        received = 0
        response_started = False

        async def counted_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, tracking_send)
        except _BodyTooLarge:
            # Nothing can be said once the endpoint has begun replying.
            if response_started:
                raise
            await self._respond(scope, receive, send, 413, "request body too large")

    @staticmethod
    async def _respond(scope, receive, send, status: int, detail: str) -> None:
        await JSONResponse(status_code=status, content={"detail": detail})(
            scope, receive, send)


app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)


async def _decompress_body(data: bytes, compression: str) -> bytes:
    """Inflate the uploaded scene per its declared compression ("none" passes
    through). 400s carry the cause — an unsupported codec must not read as a
    corrupt upload."""
    if compression == "gzip":
        def _inflate_gzip() -> bytes:
            # Bounded inflate so a small gzip body can't balloon into RAM: read
            # at most the scene ceiling + 1 decompressed bytes and reject past
            # it, mirroring the zstd path's max_output_size cap below.
            # gzip.decompress() has no size guard and is a decompression bomb.
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                out = gz.read(MAX_SCENE_BYTES + 1)
            if len(out) > MAX_SCENE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="gzip body inflates past the scene size limit")
            return out

        try:
            return await asyncio.to_thread(_inflate_gzip)
        except (OSError, EOFError) as exc:
            raise HTTPException(status_code=400,
                                detail="file is not valid gzip data") from exc
    if compression == "zstd":
        if not _zstd_available():
            raise HTTPException(
                status_code=400,
                detail="this service cannot decode zstd (zstandard not installed) "
                       "— send gzip, or install zstandard and restart")
        import zstandard

        def _inflate() -> bytes:
            return zstandard.ZstdDecompressor().decompress(
                data, max_output_size=MAX_BODY_BYTES * 8)

        try:
            return await asyncio.to_thread(_inflate)
        except zstandard.ZstdError as exc:
            raise HTTPException(status_code=400,
                                detail="file is not valid zstd data") from exc
    return data


def _zstd_available() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/live")
async def live() -> dict:
    # protocol_version is public on purpose: clients verify compatibility here before
    # authenticating or uploading anything (see usd_core.remote_protocol).
    # max_body_bytes/features let clients adopt the real upload limit (a default
    # client cap fail-fasted a scene this service would have accepted) and pick
    # zstd only when this end can decode it. Non-secret capability facts.
    info: dict = {
        "status": "alive",
        "protocol_version": PROTOCOL_VERSION,
        # Engine identity is part of the workflow readiness contract:
        # a generic "remote" transport must never be mistaken for OVRTX.
        "engine": "ovrtx",
        "renderer": "ovrtx",
        "max_body_bytes": MAX_BODY_BYTES,
        # the staged-scene ceiling for the CAS manifest transport — the
        # client's pre-packaging fail-fast sizes against this instead of
        # the per-body limit (a manifest render has no single big body)
        "max_scene_bytes": MAX_SCENE_BYTES,
    }
    features = [
        "cas",  # content-addressed manifest transport (dedup plan Phase 2)
        "physics-simulate",
    ]
    if _zstd_available():
        features.append("zstd")
    info["features"] = features
    return info


@app.get("/health", dependencies=[Depends(authenticate)])
async def health() -> HealthResponse:
    return HealthResponse(
        status=_init_state,
        protocol_version=PROTOCOL_VERSION,
        gpu_initialized=_renderer.is_ready if _renderer else False,
        renderer_initialized=_renderer.is_initialized if _renderer else False,
        daemon_running=_renderer.daemon_running if _renderer else False,
        physics_daemon_running=_simulator.daemon_running if _simulator else False,
        error=_init_error,
    )


@app.get("/ready", dependencies=[Depends(authenticate)])
async def ready():
    if _init_state != "ready" or _renderer is None or not _renderer.is_ready:
        raise HTTPException(status_code=503, detail={"status": _init_state, "error": _init_error})
    return {"status": "ready"}


async def _run_locked(fn, args, *, lock: asyncio.Lock, timeout_s: float, what: str):
    """Run one worker call behind a bounded queue (one slot + one active job) with a
    timeout. Excess callers receive a quick overload response instead of building an
    unbounded work queue."""
    try:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=1.0)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail=f"{what} busy; retry later") from exc
        release_here = True
        task = asyncio.create_task(asyncio.to_thread(fn, *args))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            # A Python worker thread cannot be force-killed safely, and the task is
            # shielded, so it keeps running in both cases — a timeout here, or the
            # caller disconnecting and cancelling this coroutine. Either way the
            # queue must stay locked until that worker finishes; releasing on the
            # way out would let the next request drive the same OVRTX/ovphysx
            # backend concurrently.
            release_here = False
            task.add_done_callback(lambda _task: lock.release())
            raise
        finally:
            if release_here:
                lock.release()
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"{what} timed out") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", what)
        # Carry the CAUSE (sanitized, truncated) — a bare "render failed" sent
        # round-7 agents on multi-minute misdiagnosis detours while the real
        # error ("no LdrColor output", NVML/Vulkan init) sat in the server log.
        cause = "".join(c if c.isprintable() else " " for c in
                        f"{type(exc).__name__}: {exc}")[:300]
        raise HTTPException(status_code=500,
                            detail=f"{what} failed — {cause}") from exc


async def _run_render(fn, *args) -> list[dict]:
    """Run one renderer call behind the bounded GPU queue with the shared timeout."""
    if _renderer is None or _init_state != "ready":
        raise HTTPException(status_code=503, detail=f"renderer is {_init_state}")
    return await _run_locked(fn, args, lock=_render_lock, timeout_s=RENDER_TIMEOUT_S,
                             what="render")


@app.post("/render", dependencies=[Depends(authenticate)])
async def render(request: RenderRequest) -> RenderResponse:
    items = await _run_render(
        _renderer.render if _renderer else None,
        request.cameras, request.image_width, request.image_height, request.mode,
        request.usdz_base64, request.usd, request.frames, request.camera_defs)
    return RenderResponse(results=[RenderResultItem(**it) for it in items])


def _get_simulator() -> Simulator:
    global _simulator
    if _simulator is None:
        _simulator = Simulator()
    return _simulator


@app.post("/physics/simulate", dependencies=[Depends(authenticate)])
async def physics_simulate(file: UploadFile, params: str = Form(...)) -> PhysicsResponse:
    """Remote drop-settle simulation: multipart raw (optionally gzipped) flattened
    USD/USDA/USDZ scene in `file`, JSON simulation parameters in `params`. Returns the raw
    trajectory; the client authors the recording and metrics locally. Independent of the
    render pipeline — available even while the GPU renderer is still warming up."""
    try:
        parsed = PhysicsUploadParams.model_validate_json(params)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid params: {exc}") from exc
    data = await file.read()
    filename = file.filename or "scene.usda"
    await file.close()
    data = await _decompress_body(data, parsed.compression)
    logger.info("physics simulate: %.1f MB scene, body=%s duration=%.2fs dt=%g fps=%d",
                len(data) / (1024 * 1024), parsed.body_pattern, parsed.duration_s,
                parsed.dt, parsed.sample_fps)

    def _simulate():
        return _get_simulator().simulate_scene_bytes(
            data, filename=filename, body_pattern=parsed.body_pattern,
            duration_s=parsed.duration_s, dt=parsed.dt, sample_fps=parsed.sample_fps)

    result = await _run_locked(_simulate, (), lock=_physics_lock,
                               timeout_s=PHYSICS_TIMEOUT_S, what="simulation")
    return PhysicsResponse(**result)


@app.post("/render/upload", dependencies=[Depends(authenticate)])
async def render_upload(file: UploadFile, params: str = Form(...)) -> RenderResponse:
    """Multipart transport: raw (optionally gzipped) binary USDZ in `file`, JSON render
    parameters in the `params` form field. No base64 inflation — a scene near the body
    limit that would 413 as JSON uploads fine here."""
    try:
        parsed = RenderUploadParams.model_validate_json(params)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid params: {exc}") from exc
    data = await file.read()
    await file.close()
    data = await _decompress_body(data, parsed.compression)
    logger.info("multipart render: %.1f MB USDZ, %d camera(s), %dx%d %s%s",
                len(data) / (1024 * 1024), len(parsed.cameras),
                parsed.image_width, parsed.image_height, parsed.mode,
                f", {len(parsed.frames)} frame(s)" if parsed.frames else "")
    items = await _run_render(
        _renderer.render_usdz_bytes if _renderer else None,
        parsed.cameras, parsed.image_width, parsed.image_height, parsed.mode, data,
        parsed.frames, parsed.camera_defs)
    return RenderResponse(results=[RenderResultItem(**it) for it in items])


# ── CAS manifest transport (upload-dedup plan Phase 2) ──────────────────────────


@app.post("/render/negotiate", dependencies=[Depends(authenticate)])
async def render_negotiate(request: NegotiateRequest) -> NegotiateResponse:
    """Answer which of the manifest's blobs this node is missing. Present blobs are
    LRU-touched and prefetched into the page cache, so their disk warm-up overlaps
    the client's upload of the missing ones."""
    store = _get_cas()
    shas = [f.sha256 for f in request.files]
    missing = await asyncio.to_thread(store.missing, shas)
    absent = set(missing)
    await asyncio.to_thread(store.prefetch, [s for s in set(shas) if s not in absent])
    need = sum(f.size for f in request.files if f.sha256 in absent)
    logger.info("negotiate: %d file(s), %d blob(s) missing (~%.1f MB to upload)",
                len(request.files), len(missing), need / (1024 * 1024))
    return NegotiateResponse(missing=missing)


@app.put("/blobs/{sha256}", dependencies=[Depends(authenticate)])
async def put_blob(
    request: Request,
    sha256: str = PathParam(..., pattern=r"^[0-9a-f]{64}$"),
    compression: Literal["none", "gzip", "zstd"] = "none",
) -> dict:
    """Store one content-addressed blob (optionally wire-compressed). The digest is
    verified on write — a mismatch is a 400 and nothing is stored."""
    data = await request.body()
    if len(data) > MAX_BODY_BYTES:
        # Defense in depth: BodySizeLimitMiddleware already counts the streamed
        # bytes, so reaching this is a bug rather than the normal chunked path.
        raise HTTPException(status_code=413, detail="request body too large")
    data = await _decompress_body(data, compression)
    if len(data) > MAX_SCENE_BYTES:
        raise HTTPException(status_code=413,
                            detail="decompressed blob exceeds the scene limit")
    try:
        stored = await asyncio.to_thread(_get_cas().put, sha256, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"sha256": sha256, "size": len(data), "stored": stored}


@app.post("/render/manifest", dependencies=[Depends(authenticate)])
async def render_manifest(request: ManifestRenderRequest) -> RenderResponse:
    """Render a scene materialized from the blob store (hardlinked staging tree).

    Any blob the store no longer holds answers 409 with the missing digest list —
    the client re-uploads exactly those and retries; a partially-materialized scene
    is never rendered."""
    if sum(f.size for f in request.files) > MAX_SCENE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"manifest scene exceeds the {MAX_SCENE_BYTES} byte staged-scene "
                   "limit — raise OVRTX_MAX_BODY_BYTES or reduce the scene")
    store = _get_cas()
    staging = await asyncio.to_thread(
        tempfile.mkdtemp, prefix="stage_", dir=str(store.staging))
    try:
        try:
            root_path = await asyncio.to_thread(
                store.materialize, [f.model_dump() for f in request.files],
                request.root, staging, MAX_SCENE_BYTES)
        except cas.MissingBlobsError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": "blob(s) evicted or never uploaded — PUT them "
                                   "and retry", "missing": exc.missing}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info("manifest render: %d file(s) staged, %d camera(s), %dx%d %s%s",
                    len(request.files), len(request.cameras),
                    request.image_width, request.image_height, request.mode,
                    f", {len(request.frames)} frame(s)" if request.frames else "")
        items = await _run_render(
            _renderer.render_scene_path if _renderer else None,
            request.cameras, request.image_width, request.image_height, request.mode,
            str(root_path), request.frames, request.camera_defs)
        return RenderResponse(results=[RenderResultItem(**it) for it in items])
    finally:
        await asyncio.to_thread(shutil.rmtree, staging, True)
