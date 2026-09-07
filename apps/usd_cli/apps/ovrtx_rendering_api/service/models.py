# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request/response schemas — the contract usd_core's RemoteRenderBackend speaks.

See `src/usd_core/render/remote.py` and `src/usd_core/physics_runtime.py`:
  POST /render  {usdz_base64 | usd, cameras, image_width, image_height, mode}
                -> {results: [{camera, image_base64}]}
  POST /physics/simulate  multipart file + params {body_pattern, duration_s, dt, sample_fps}
                -> {trajectory, n_bodies, n_steps}
  GET  /health  -> {status, protocol_version, gpu_initialized, renderer_initialized, ...}
  GET  /live    -> {status, protocol_version}   # public; clients verify compatibility
                                                # here (usd_core.remote_protocol)

CAS manifest transport is feature-gated on `/live` advertising `features: ["cas"]`:
  POST /render/negotiate  {files: [{path, sha256, size}...]} -> {missing: [sha...]}
  PUT  /blobs/{sha}?compression=none|gzip|zstd   raw body, sha verified on write
  POST /render/manifest   {files, root, cameras, ...} -> {results: [...]}
                          (409 + {missing} when a blob was evicted mid-flight)

`usdz_base64` is the preferred transport: a self-contained USDZ bundle (geometry + textures)
so material maps resolve remotely. `usd` (flat USDA text, textures unresolved) is still
accepted for backward compatibility; exactly one of the two must be supplied.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Ceiling on the pixels one request may ask the renderer to produce in total.
#: The per-image and per-render-count limits are independent, so without an
#: aggregate budget a single valid request can ask for 240 renders at 4096x4096
#: — 4,026,531,840 pixels. Every PNG is base64-encoded and held in one response
#: before it is sent, so that request OOMs the service rather than failing.
#: The default admits the largest documented legitimate job (a 240-frame
#: animation at the default 1024x1024 = 251,658,240 pixels) and rejects the
#: order-of-magnitude abuses above it.
MAX_REQUEST_PIXELS = int(
    os.environ.get("OVRTX_MAX_REQUEST_PIXELS", str(268_435_456)))
#: Ceiling on `duration_s / dt`, the step count the physics daemon schedules
#: (`total = max(1, int(round(dur/dt)))`). `dt` alone is bounded only by
#: `gt=0`, so `duration_s=60` with `dt=1e-6` would queue 60,000,000 steps and
#: hold the simulator until its long timeout. The default admits a full 60 s
#: run at the default 1/240 s timestep (14,400 steps) with room to spare.
#: Same knob and default as usd_core.physics_runtime.MAX_PHYSICS_STEPS.
MAX_PHYSICS_STEPS = int(
    os.environ.get("OVRTX_MAX_PHYSICS_STEPS", str(120_000)))


def _check_render_budget(cameras: list[str], frames: list[float] | None,
                         width: int, height: int) -> None:
    """Shared per-image, per-count and aggregate render limits."""
    if width * height > 16_777_216:
        raise ValueError("image dimensions exceed 16,777,216 pixels")
    if frames is not None and len(cameras) * len(frames) > 240:
        raise ValueError("cameras × frames exceeds 240 renders per request")
    renders = len(cameras) * (len(frames) if frames is not None else 1)
    if renders * width * height > MAX_REQUEST_PIXELS:
        raise ValueError(
            f"total render output ({renders} renders x {width}x{height} = "
            f"{renders * width * height} pixels) exceeds the request budget of "
            f"{MAX_REQUEST_PIXELS} pixels")


class CameraDef(BaseModel):
    """A tool-authored render camera, sent as a spec instead of scene content
    (protocol v2). The client strips these cameras from the uploaded bundle —
    making the bundle viewpoint-independent and cacheable across renders — and
    the service re-authors them at `path` before rendering."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, max_length=2048,
                      description="Absolute prim path to author the camera at")
    matrix: list[float] = Field(..., min_length=16, max_length=16,
                                description="Row-major 4x4 local-to-world transform")
    focal_length: float | None = Field(None, gt=0)
    horizontal_aperture: float | None = Field(None, gt=0)
    vertical_aperture: float | None = Field(None, gt=0)
    clipping_range: list[float] | None = Field(None, min_length=2, max_length=2)
    projection: Literal["perspective", "orthographic"] | None = None

    @model_validator(mode="after")
    def _finite_and_sane(self) -> "CameraDef":
        # these values reach native USD/GPU code — reject NaN/Inf and inverted
        # clipping outright instead of letting the renderer misbehave
        import math

        values = list(self.matrix) + [v for v in (self.focal_length,
                                                  self.horizontal_aperture,
                                                  self.vertical_aperture)
                                      if v is not None]
        if self.clipping_range is not None:
            values += self.clipping_range
        if not all(math.isfinite(v) for v in values):
            raise ValueError("camera_defs values must be finite")
        if self.clipping_range is not None:
            near, far = self.clipping_range
            if not 0 < near < far:
                raise ValueError("clipping_range must satisfy 0 < near < far")
        return self


class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usdz_base64: str | None = Field(
        None, max_length=67_108_864,
        description="Base64 USDZ bundle (geometry + textures); preferred over `usd`")
    usd: str | None = Field(
        None, max_length=50_331_648,
        description="USD scene as USDA text (root layer export); textures unresolved")
    cameras: list[str] = Field(..., min_length=1, max_length=8,
                               description="Camera prim paths to render")
    image_width: int = Field(1024, ge=1, le=4096)
    image_height: int = Field(1024, ge=1, le=4096)
    mode: Literal["fast", "quality"] = Field(
        "quality", description="fast (rt1) or quality (rt2)"
    )
    frames: list[float] | None = Field(
        None, max_length=240,
        description="USD time codes for animation. When set, the scene is ingested once and "
                    "each frame is rendered at its time code (cameras × frames). When omitted, "
                    "a single render at the default time code.")
    camera_defs: list[CameraDef] | None = Field(
        None, max_length=8,
        description="Specs of tool-authored cameras stripped from the scene (protocol v2); "
                    "authored into the stage before rendering")

    @model_validator(mode="after")
    def _one_scene_source(self) -> "RenderRequest":
        if bool(self.usdz_base64) == bool(self.usd):
            raise ValueError("provide exactly one of 'usdz_base64' or 'usd'")
        _check_render_budget(self.cameras, self.frames,
                             self.image_width, self.image_height)
        return self


class RenderUploadParams(BaseModel):
    """The `params` form field of `POST /render/upload` — the multipart transport.

    The scene arrives as the raw (optionally gzipped) binary USDZ in the `file` part,
    so there is no base64 inflation and no JSON size ceiling beyond the body limit.
    """

    model_config = ConfigDict(extra="forbid")

    cameras: list[str] = Field(..., min_length=1, max_length=8)
    image_width: int = Field(1024, ge=1, le=4096)
    image_height: int = Field(1024, ge=1, le=4096)
    mode: Literal["fast", "quality"] = "quality"
    compression: Literal["none", "gzip", "zstd"] = "none"
    frames: list[float] | None = Field(
        None, max_length=240,
        description="USD time codes for animation; ingest once, render each frame "
                    "(cameras × frames). Omit for a single default-time render.")
    camera_defs: list[CameraDef] | None = Field(
        None, max_length=8,
        description="Specs of tool-authored cameras stripped from the bundle (protocol v2); "
                    "authored into the stage before rendering")

    @model_validator(mode="after")
    def _pixels(self) -> "RenderUploadParams":
        _check_render_budget(self.cameras, self.frames,
                             self.image_width, self.image_height)
        return self


class ManifestFile(BaseModel):
    """One file of a bundle manifest: its package-relative path and content hash.

    Paths are the packaged USDZ's zip-entry names — already a self-contained
    relative tree — and are re-validated here (and again in `service.cas`) because
    they are joined under a server staging directory."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, max_length=1024)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    size: int = Field(..., ge=0, le=1 << 40)

    @model_validator(mode="after")
    def _safe_path(self) -> "ManifestFile":
        from service.cas import validate_rel_path

        validate_rel_path(self.path)
        return self


class NegotiateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[ManifestFile] = Field(..., min_length=1, max_length=8192)


class NegotiateResponse(BaseModel):
    #: sha256 digests the store does not hold — upload exactly these via PUT /blobs
    missing: list[str]


class ManifestRenderRequest(BaseModel):
    """`POST /render/manifest` — render a scene materialized from the blob store.

    Mirrors RenderUploadParams, but the scene arrives as a manifest of
    content-addressed files (negotiated + uploaded beforehand) instead of a bundle
    body. `root` names the manifest entry to open as the stage's root layer."""

    model_config = ConfigDict(extra="forbid")

    files: list[ManifestFile] = Field(..., min_length=1, max_length=8192)
    root: str = Field(..., min_length=1, max_length=1024)
    cameras: list[str] = Field(..., min_length=1, max_length=8)
    image_width: int = Field(1024, ge=1, le=4096)
    image_height: int = Field(1024, ge=1, le=4096)
    mode: Literal["fast", "quality"] = "quality"
    frames: list[float] | None = Field(
        None, max_length=240,
        description="USD time codes for animation; ingest once, render each frame "
                    "(cameras × frames). Omit for a single default-time render.")
    camera_defs: list[CameraDef] | None = Field(
        None, max_length=8,
        description="Specs of tool-authored cameras stripped from the bundle (protocol v2); "
                    "authored into the stage before rendering")

    @model_validator(mode="after")
    def _sane(self) -> "ManifestRenderRequest":
        _check_render_budget(self.cameras, self.frames,
                             self.image_width, self.image_height)
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("manifest contains duplicate file paths")
        if self.root not in set(paths):
            raise ValueError("root must be one of the manifest's file paths")
        return self


class PhysicsUploadParams(BaseModel):
    """The `params` form field of `POST /physics/simulate` — remote drop-settle simulation.

    The scene arrives as the raw (optionally gzipped) flattened USD/USDA/USDZ in the
    `file` part; the response carries the raw trajectory so the client authors the
    recording and metrics locally (mirrors usd_core.physics_runtime's daemon contract).
    """

    model_config = ConfigDict(extra="forbid")

    body_pattern: str = Field(..., min_length=1, max_length=2048,
                              description="Rigid-body prim path/pattern to track")
    duration_s: float = Field(1.0, gt=0, le=60.0)
    dt: float = Field(1.0 / 240.0, gt=0, le=0.1)
    sample_fps: int = Field(30, ge=1, le=240)
    compression: Literal["none", "gzip", "zstd"] = "none"

    @model_validator(mode="after")
    def _bounded_step_count(self) -> "PhysicsUploadParams":
        # `duration_s` and `dt` are each bounded, but their ratio — the work the
        # daemon actually schedules — was not.
        steps = max(1, int(round(self.duration_s / self.dt)))
        if steps > MAX_PHYSICS_STEPS:
            raise ValueError(
                f"duration_s / dt schedules {steps} simulation steps, over the "
                f"limit of {MAX_PHYSICS_STEPS}; raise dt or shorten duration_s")
        return self


class PhysicsResponse(BaseModel):
    #: samples of [time_s, pose[7] (px py pz qx qy qz qw), vel[6] (lin+ang)]
    trajectory: list[tuple[float, list[float], list[float]]]
    n_bodies: int
    n_steps: int


class RenderResultItem(BaseModel):
    camera: str
    image_base64: str  # base64-encoded PNG
    #: USD time code this image was rendered at; present only for animation (frames) requests
    frame: float | None = None
    #: Factual settings returned by the executing OVRTX backend.
    ovrtx_render_mode: str | None = None
    ovrtx_num_sensor_updates: int | None = None
    active_aov: str | None = None


class RenderResponse(BaseModel):
    results: list[RenderResultItem]


class HealthResponse(BaseModel):
    status: str
    #: wire-contract version baked in at deploy time (usd_core.remote_protocol);
    #: clients refuse to run when it differs from their own
    protocol_version: int = 0
    gpu_initialized: bool
    renderer_initialized: bool
    daemon_running: bool
    #: ovphysx daemon state: it boots lazily on the first /physics/simulate call, so False
    #: means "not started yet", not "unavailable"
    physics_daemon_running: bool = False
    error: str | None = None
