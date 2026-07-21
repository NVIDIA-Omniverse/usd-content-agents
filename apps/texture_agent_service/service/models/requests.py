# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request models for Texture Agent Service API."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TEXTURE_UNIT_ID_RE = re.compile(r"^tu_[0-9a-f]{20}$")


class TexturePipelineStep(StrEnum):
    """Available pipeline steps."""

    PREPARE_UVS = "prepare_uvs"
    DISCOVER_MATERIALS = "discover_materials"
    PLAN_TEXTURES = "plan_textures"
    GENERATE_PROMPTS = "generate_prompts"
    RENDER_PREVIEWS = "render_previews"
    GENERATE_TEXTURES = "generate_textures"
    BLEND_TEXTURES = "blend_textures"
    APPLY_TEXTURES = "apply_textures"
    RENDER = "render"


class TextureDetailPolicy(StrEnum):
    """Texture detail policy values accepted by the service API."""

    DEFAULT = "default"
    SURFACE_ONLY = "surface_only"


class PrimTextureOverride(BaseModel):
    """Per-prim prompt/opacity override nested under a material override."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    prompt: str | None = Field(default=None, min_length=1)
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    detail_policy: TextureDetailPolicy | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("Prompt must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def _requires_one_override(self) -> "PrimTextureOverride":
        if self.prompt is None and self.opacity is None and self.detail_policy is None:
            raise ValueError(
                "Per-prim override must include prompt, opacity, or detail_policy"
            )
        return self


class MaterialTextureOverride(BaseModel):
    """Per-material texture prompt/opacity override accepted by the API."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    prompt: str = Field(min_length=1)
    opacity: float | None = Field(default=None, ge=0.0, le=1.0)
    detail_policy: TextureDetailPolicy | None = None
    material_path: str | None = None
    prim_path: str | None = None
    prim_paths: list[str] | None = None
    reference_image_uris: list[str] | None = None
    turntable_video_uri: str | None = None
    multiview_image_uris: list[str] | None = None
    per_prim: dict[str, PrimTextureOverride] | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Prompt must be a non-empty string")
        return stripped

    @field_validator(
        "prim_paths",
        "reference_image_uris",
        "multiview_image_uris",
    )
    @classmethod
    def _strip_uri_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        stripped = [item.strip() for item in value if item and item.strip()]
        return stripped or None

    @field_validator("material_path", "prim_path", "turntable_video_uri")
    @classmethod
    def _strip_optional_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("per_prim")
    @classmethod
    def _validate_per_prim_keys(
        cls, value: dict[str, PrimTextureOverride] | None
    ) -> dict[str, PrimTextureOverride] | None:
        if value is None:
            return None
        for key in value:
            if not key or not key.strip():
                raise ValueError("Per-prim override keys must be non-empty")
        return value


class MaterialTextures(BaseModel):
    """Root model for API material texture overrides."""

    model_config = ConfigDict(extra="forbid")

    root: dict[str, MaterialTextureOverride] = Field(default_factory=dict)

    @field_validator("root")
    @classmethod
    def _validate_material_keys(
        cls, value: dict[str, MaterialTextureOverride]
    ) -> dict[str, MaterialTextureOverride]:
        for key in value:
            if not key or not key.strip():
                raise ValueError("Material override keys must be non-empty")
        return value

    def as_config(self) -> dict[str, dict[str, Any]]:
        """Return the plain dict shape consumed by the texture pipeline."""
        return {
            material: override.model_dump(exclude_none=True)
            for material, override in self.root.items()
        }


class RegenerateRequest(BaseModel):
    """Request to regenerate specific steps from cache."""

    steps: list[TexturePipelineStep] = Field(
        min_length=1,
        description="Steps to re-run from cache (at least one)",
    )

    material_textures: dict[str, MaterialTextureOverride] | None = Field(
        default=None,
        description="Override per-material prompt/opacity for regeneration",
    )
    texture_unit_ids: list[str] | None = Field(
        default=None,
        description=(
            "Exact approved texture-plan unit IDs to regenerate. Omit to "
            "regenerate every approved unit when generate_textures is selected."
        ),
    )

    @field_validator("texture_unit_ids")
    @classmethod
    def _validate_texture_unit_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("texture_unit_ids must not be empty when provided")
        if len(value) != len(set(value)):
            raise ValueError("texture_unit_ids must not contain duplicates")
        for unit_id in value:
            if _TEXTURE_UNIT_ID_RE.fullmatch(unit_id) is None:
                raise ValueError(
                    "texture_unit_ids must use canonical tu_<20 hex> identifiers"
                )
        return value

    @model_validator(mode="after")
    def _targeted_units_require_generation(self) -> "RegenerateRequest":
        if (
            self.texture_unit_ids
            and TexturePipelineStep.GENERATE_TEXTURES not in self.steps
        ):
            raise ValueError(
                "texture_unit_ids requires the generate_textures regeneration step"
            )
        return self

    @field_validator("material_textures")
    @classmethod
    def _validate_material_texture_keys(
        cls, value: dict[str, MaterialTextureOverride] | None
    ) -> dict[str, MaterialTextureOverride] | None:
        if value is None:
            return None
        for key in value:
            if not key or not key.strip():
                raise ValueError("Material override keys must be non-empty")
        return value

    def material_textures_config(self) -> dict[str, dict[str, Any]] | None:
        """Return material overrides in the plain dict format used by YAML config."""
        if self.material_textures is None:
            return None
        return {
            material: override.model_dump(exclude_none=True)
            for material, override in self.material_textures.items()
        }
