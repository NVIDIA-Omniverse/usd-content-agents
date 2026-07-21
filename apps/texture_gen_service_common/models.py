# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Texture Variation API Pydantic models.

These models intentionally mirror Texture Agent's REST client contract so
service implementations do not drift from the client payloads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JobState = Literal["queued", "processing", "completed", "failed", "cancelled"]


class Conditioning(BaseModel):
    """Conditioning inputs for texture generation."""

    text_prompt: str | None = None
    reference_image_uris: list[str] = Field(default_factory=list)
    turntable_video_uri: str | None = None
    multiview_image_uris: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_at_least_one(self) -> Conditioning:
        has_prompt = bool(self.text_prompt and self.text_prompt.strip())
        has_refs = bool(self.reference_image_uris)
        has_video = bool(self.turntable_video_uri and self.turntable_video_uri.strip())
        has_multiview = bool(self.multiview_image_uris)
        if not (has_prompt or has_refs or has_video or has_multiview):
            raise ValueError(
                "At least one non-empty conditioning input is required "
                "(text_prompt, reference_image_uris, turntable_video_uri, "
                "or multiview_image_uris)."
            )
        return self


class Configuration(BaseModel):
    """Texture generation configuration."""

    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    seed: int | None = None
    variant_name: str | None = None
    engine: str | None = None
    texture_size: int | None = Field(default=None, ge=1, le=4096)
    custom_parameters: dict[str, Any] = Field(default_factory=dict)


class TextureTarget(BaseModel):
    """Selected material/prim scope for projection backends."""

    material_name: str | None = None
    material_path: str | None = None
    prim_paths: list[str] = Field(default_factory=list)
    mode: str = "per_material"
    strict_scope: bool = True


class BackendCapabilities(BaseModel):
    """Backend capability hints or reported response capabilities."""

    model_config = ConfigDict(extra="allow")

    image_conditioning: bool | None = None
    multiview: bool | None = None
    normal_map: bool | None = None
    orm: bool | None = None
    masks: bool | None = None
    coverage: bool | None = None
    geometry_output: str | None = None


class CreateJobRequest(BaseModel):
    """POST /v1/texture-variations request."""

    source_asset_uri: str
    target: TextureTarget | None = None
    conditioning: Conditioning
    configuration: Configuration = Field(default_factory=Configuration)
    capabilities: BackendCapabilities | None = None


class GeneratedTextures(BaseModel):
    """Backward-compatible generated texture URI fields.

    Projection backends may return albedo-only output. Missing normal/ORM maps
    must be paired with degraded-channel metadata and diagnostics.
    """

    albedo: str | None = None
    normal: str | None = None
    orm: str | None = None


class MapArtifact(BaseModel):
    """Normalized map artifact metadata."""

    uri: str
    width: int | None = None
    height: int | None = None
    mime_type: str = "image/png"
    colorspace: str | None = None
    packing: str | None = None


class GenerationResult(BaseModel):
    """Completed generation result."""

    variant_asset_uri: str
    variant_name: str
    generated_textures: GeneratedTextures
    maps: dict[str, MapArtifact] = Field(default_factory=dict)
    auxiliary_artifacts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)


class JobStatus(BaseModel):
    """Texture variation job status."""

    job_id: str
    status: JobState
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    result: GenerationResult | None = None
    error_message: str | None = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "healthy"
    service: str = "texture-variation-api"
    version: str = "1.0.0"
    backend: str
    ready: bool = False
    accepting_jobs: bool = False
    active_jobs: int = 0
    queued_jobs: int = 0
    max_workers: int = 1
    max_queue_size: int = 0
    warmup_complete: bool | None = None
    gpu_available: bool | None = None
    capabilities: BackendCapabilities | dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
