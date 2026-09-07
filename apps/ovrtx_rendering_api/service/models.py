# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request and response models for the rendering API.

These models match the Kit-based rendering-api contract so this service
is a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from service.protocol import PROTOCOL_VERSION
from world_understanding.functions.graphics.material_targets import RenderMaterialTarget

# Render-mode strings the caller can pass — keep the tokens identical to the
# kit-gen-ai-service ``RenderMode`` enum (``rt1``/``rt2``/``pt``) so clients
# targeting either service can use the same request body.
RenderMode = Literal["rt1", "rt2", "pt"]


class FrameRange(BaseModel):
    """Frame range for multi-frame rendering."""

    start: int = 0
    end: int = 0


class CameraParameters(BaseModel):
    """Camera resolution parameters."""

    width: int = Field(default=1024, ge=1, le=8192)
    height: int = Field(default=1024, ge=1, le=8192)


class RenderSettings(BaseModel):
    """Render settings matching the Kit-based rendering-api contract."""

    camera_paths: list[str] = Field(default_factory=lambda: ["/Camera"])
    frame_range: FrameRange = Field(default_factory=FrameRange)
    camera_parameters: CameraParameters = Field(default_factory=CameraParameters)
    sensors: list[str] | None = None
    apply_background_mask: bool = False
    render_mode: RenderMode | None = Field(
        default=None,
        description=(
            "Selects the RTX path — ``rt1`` (ray-traced lighting), "
            "``rt2`` (real-time path tracing), or ``pt`` (offline path "
            "tracing, ground truth). Writes the ``omni:rtx:rendermode`` "
            "USD attribute on the RenderProduct, which OVRtx honors at "
            "``step()`` time. ``None`` falls back to the service's "
            "instance default (``OVRTX_RENDER_MODE`` env, default "
            "``pt`` — the only mode that reaches Kit-equivalent quality "
            "in 0.2.0 validation; rt2 capped at ~27 dB PSNR vs Kit)."
        ),
    )
    num_sensor_updates: int | None = Field(
        default=None,
        ge=1,
        le=5000,
        description=(
            "Number of progressive ``renderer.step(delta_time=0)`` "
            "iterations per frame. This is the guarded quality knob "
            "validated on OVRtx 0.2.0 — the bundled "
            "``omni:rtx:pt:samplesPerPixel`` / "
            "``omni:rtx:rt:accumulationLimit`` schema attributes are "
            "kept disabled until 0.3 GPU validation proves they affect "
            "output. ``None`` falls back to the instance "
            "default captured from ``OVRTX_NUM_SENSOR_UPDATES`` at service "
            "startup (500 — the convergence plateau at ~39.7 dB PSNR "
            "vs Kit on the kit golden scene). Lower values trade "
            "quality for wall-clock time (~9 ms per step at 512x512)."
        ),
    )
    material_target: RenderMaterialTarget | None = Field(
        default=None,
        description=(
            "Explicit material target for render export. ``None`` keeps the "
            "service/backend default. ``auto`` preserves authored/native "
            "material outputs, ``preview_surface`` requests that fallback "
            "explicitly, and ``openpbr_materialx`` asks the renderer to "
            "consume native OpenPBR/MaterialX without authoring render-only "
            "PreviewSurface fallbacks."
        ),
    )


class RenderRequest(BaseModel):
    """POST /render request body."""

    url: str
    force_render: bool = True
    render_settings: RenderSettings = Field(default_factory=RenderSettings)


class RenderResponse(BaseModel):
    """POST /render response body (V1 format).

    Structure: images[frame_number][camera_path][sensor_name] = base64 string
    """

    status: str = "success"
    error: str | None = None
    images: dict[str, dict[str, dict[str, str]]] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    requested_output_count: int | None = None
    output_count: int | None = None
    missing_output_count: int | None = None
    missing_camera_count: int | None = None
    ovrtx_render_mode: RenderMode | None = None
    ovrtx_num_sensor_updates: int | None = Field(default=None, ge=1, le=5000)
    active_aov: str | None = None


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "healthy"
    service: str = "ovrtx-rendering-api"
    version: str = "0.1.0"
    renderer: str = "ovrtx"
    protocol_version: int = PROTOCOL_VERSION
    gpu_initialized: bool = False
    renderer_initialized: bool = False
    daemon_running: bool = False
    daemon_pid: int | None = None
    daemon_completed_renders: int | None = None
    daemon_rss_bytes: int | None = None
    daemon_recycle_count: int | None = None
    daemon_last_recycle_reason: str | None = None
    daemon_pending_recycle_reason: str | None = None
    ready_workers: int | None = None
    total_workers: int | None = None
    workers: list[dict[str, Any]] | None = None


class ProtocolV3CameraDef(BaseModel):
    """One package-owned usd-cli render camera transmitted out of band."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, max_length=2048)
    matrix: list[float] = Field(..., min_length=16, max_length=16)
    focal_length: float | None = Field(None, gt=0)
    horizontal_aperture: float | None = Field(None, gt=0)
    vertical_aperture: float | None = Field(None, gt=0)
    clipping_range: list[float] | None = Field(None, min_length=2, max_length=2)
    projection: Literal["perspective", "orthographic"] | None = None

    @model_validator(mode="after")
    def _finite_and_sane(self) -> ProtocolV3CameraDef:
        values = list(self.matrix) + [
            value
            for value in (
                self.focal_length,
                self.horizontal_aperture,
                self.vertical_aperture,
            )
            if value is not None
        ]
        if self.clipping_range is not None:
            values.extend(self.clipping_range)
            near, far = self.clipping_range
            if not 0 < near < far:
                raise ValueError("clipping_range must satisfy 0 < near < far")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera values must be finite")
        return self


class ProtocolV3RenderUploadParams(BaseModel):
    """Multipart render parameters spoken by usd-cli remote protocol v3."""

    model_config = ConfigDict(extra="forbid")

    cameras: list[str] = Field(..., min_length=1, max_length=8)
    image_width: int = Field(1024, ge=1, le=4096)
    image_height: int = Field(1024, ge=1, le=4096)
    mode: Literal["fast", "quality"] = Field(
        "quality",
        description=(
            "Protocol-compatible client intent. The standalone service's "
            "OVRTX_RENDER_MODE instance policy remains authoritative; every "
            "result reports the exact mode that executed."
        ),
    )
    compression: Literal["none", "gzip"] = "none"
    frames: list[float] | None = Field(None, min_length=1, max_length=240)
    camera_defs: list[ProtocolV3CameraDef] | None = Field(None, max_length=8)

    @model_validator(mode="after")
    def _bounded_render(self) -> ProtocolV3RenderUploadParams:
        frames = self.frames or [0.0]
        if not all(math.isfinite(frame) for frame in frames):
            raise ValueError("frames must be finite")
        renders = len(self.cameras) * len(frames)
        if renders > 240:
            raise ValueError("cameras x frames exceeds 240 renders per request")
        if self.image_width * self.image_height > 16_777_216:
            raise ValueError("image dimensions exceed 16,777,216 pixels")
        if renders * self.image_width * self.image_height > 268_435_456:
            raise ValueError("total render output exceeds the request pixel budget")
        return self


class ProtocolV3RenderResultItem(BaseModel):
    """One protocol-v3 camera/frame render."""

    camera: str
    image_base64: str
    frame: float | None = None
    ovrtx_render_mode: RenderMode
    ovrtx_num_sensor_updates: int = Field(..., ge=1, le=5000)
    active_aov: str = Field(..., min_length=1)


class ProtocolV3RenderResponse(BaseModel):
    """Package-owned usd-cli protocol-v3 render response."""

    results: list[ProtocolV3RenderResultItem] = Field(..., max_length=240)
