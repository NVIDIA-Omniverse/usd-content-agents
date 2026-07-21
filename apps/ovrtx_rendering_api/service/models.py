# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request and response models for the rendering API.

These models match the Kit-based rendering-api contract so this service
is a drop-in replacement.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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

    width: int = 1024
    height: int = 1024


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


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "healthy"
    service: str = "ovrtx-rendering-api"
    version: str = "0.1.0"
    renderer: str = "ovrtx"
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
