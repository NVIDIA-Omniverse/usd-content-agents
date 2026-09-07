# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for USD Agent Dataset Schema v0.2.

This module defines the unified dataset schema used by all World Understanding agents
that operate on USD data (material-agent, physics-agent, joint-agent, etc.).
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# dataset.json Models (Shared Configuration)
# =============================================================================


class DatasetMetadata(BaseModel):
    """Metadata about the dataset."""

    created: str = Field(description="ISO 8601 timestamp of creation")
    creator: str = Field(description="Creating agent name (e.g., 'material-agent')")
    source_usd: str | None = Field(default=None, description="Original USD file path")
    description: str | None = Field(default=None, description="Dataset description")
    num_entries: int = Field(description="Number of entries in dataset.jsonl")

    @field_validator("created", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        """Validate and normalize timestamp."""
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, str):
            # Validate it's a valid ISO 8601 timestamp
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                return v
            except ValueError as e:
                raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e
        raise TypeError(f"Expected datetime or str, got {type(v)}")


class TaskConfig(BaseModel):
    """Task configuration."""

    type: Literal["material_assignment", "iterative_classification", "detection"] = (
        Field(description="Task type")
    )
    description: str = Field(description="Task description")


class PromptConfig(BaseModel):
    """Configuration for a single inference prompt (can be multi-step)."""

    step_name: str = Field(description="Human-readable step name")
    step_index: int = Field(description="Step order (0-indexed)")
    system_prompt: str = Field(description="System prompt (shared across all entries)")
    output_format: dict[str, Any] | None = Field(
        default=None, description="Expected output structure"
    )
    classes: list[str] | None = Field(
        default=None, description="Valid classes for classification tasks"
    )
    temperature: float | None = Field(default=None, description="Model temperature")
    max_tokens: int | None = Field(default=None, description="Max tokens")


class InferenceConfig(BaseModel):
    """Inference configuration with prompts."""

    prompts: list[PromptConfig] = Field(
        description="List of prompts (one per step for multi-step tasks)"
    )

    @field_validator("prompts")
    @classmethod
    def validate_prompts_not_empty(cls, v: list[PromptConfig]) -> list[PromptConfig]:
        """Ensure at least one prompt is provided."""
        if not v:
            raise ValueError("At least one prompt configuration is required")
        return v

    @field_validator("prompts")
    @classmethod
    def validate_step_indices(cls, v: list[PromptConfig]) -> list[PromptConfig]:
        """Ensure step indices are sequential starting from 0."""
        if not v:
            return v

        indices = [p.step_index for p in v]
        expected = list(range(len(v)))

        if sorted(indices) != expected:
            raise ValueError(
                f"Step indices must be sequential starting from 0, got {indices}"
            )

        return v


class DatasetConfig(BaseModel):
    """Root model for dataset.json (v0.2).

    This represents the shared configuration for all entries in a dataset.
    """

    schema_version: Literal["0.2"] = Field(default="0.2", description="Schema version")
    metadata: DatasetMetadata = Field(description="Dataset metadata")
    task: TaskConfig = Field(description="Task configuration")
    inference: InferenceConfig = Field(description="Inference configuration")
    prims_file: str = Field(
        default="dataset.jsonl", description="Filename of per-entry data"
    )
    usd_model_file: str | None = Field(
        default="usd_model.json", description="Optional USD model hierarchy file"
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation."""
        # Ensure single-step vs multi-step consistency
        num_steps = len(self.inference.prompts)

        if self.task.type == "iterative_classification" and num_steps < 2:
            raise ValueError(
                "iterative_classification tasks require at least 2 prompts (multi-step)"
            )


# =============================================================================
# dataset.jsonl Entry Models (Per-Entry Data)
# =============================================================================


class SourceInfo(BaseModel):
    """Source information for a dataset entry."""

    type: Literal["usd_prim", "image", "point_cloud", "video_frame"] = Field(
        description="Source type"
    )
    prim_path: str | None = Field(default=None, description="USD prim path")
    usd_file: str | None = Field(default=None, description="Original USD filename")
    model_number: str | None = Field(default=None, description="Model/part number")


class ImageMetadata(BaseModel):
    """Metadata for a single image."""

    view: str | None = Field(
        default=None, description="Camera view angle (e.g., 'posx_posy_posz')"
    )
    camera: str | None = Field(default=None, description="USD camera prim path")
    render_mode: str | None = Field(
        default=None,
        description="Render mode ('prim_only', 'prim_with_stage', 'full_scene')",
    )
    vlm_prompt: str | None = Field(
        default=None,
        description="Per-image prompt override (e.g., for different render modes)",
    )
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")

    # Allow additional custom metadata
    model_config = {"extra": "allow"}


class ImageObject(BaseModel):
    """A single image (render, reference, or photo)."""

    path: str = Field(description="Relative path from dataset root")
    type: Literal["render", "reference", "photo"] = Field(description="Image type")
    metadata: ImageMetadata | None = Field(
        default=None, description="Image metadata (view, camera, render_mode, etc.)"
    )


class MediaConfig(BaseModel):
    """Media configuration for a dataset entry."""

    images: list[ImageObject] = Field(description="Rendered/captured images")
    reference_images: list[ImageObject] | None = Field(
        default=None, description="Reference images for context"
    )

    @field_validator("images")
    @classmethod
    def validate_images_not_empty(cls, v: list[ImageObject]) -> list[ImageObject]:
        """Ensure at least one image is provided."""
        if not v:
            raise ValueError("At least one image is required")
        return v


class StepResult(BaseModel):
    """Ground truth result for a single classification step."""

    step_index: int = Field(description="Which step (0-indexed)")
    step_name: str = Field(description="Step name (for clarity)")
    classification: str = Field(alias="class", description="Ground truth class")

    # Allow 'class' as alias for 'classification'
    model_config = {"populate_by_name": True}


class GroundTruthMetadata(BaseModel):
    """Metadata about ground truth annotations."""

    annotator: str | None = Field(default=None, description="Who annotated")
    source: str | None = Field(
        default=None, description="Annotation source (e.g., 'manual', 'oracle')"
    )
    date: str | None = Field(default=None, description="Annotation date (ISO 8601)")
    verified: bool | None = Field(default=None, description="Whether verified")

    # Allow additional custom metadata
    model_config = {"extra": "allow"}


class GroundTruth(BaseModel):
    """Ground truth labels for benchmarking."""

    material: str | None = Field(
        default=None, description="Ground truth material (material-agent)"
    )
    classification: str | None = Field(
        default=None, description="Final classification (physics-agent)"
    )
    step_results: list[StepResult] | None = Field(
        default=None, description="Per-step classifications for multi-step tasks"
    )
    metadata: GroundTruthMetadata | None = Field(
        default=None, description="Annotation metadata"
    )

    @model_validator(mode="after")
    def validate_at_least_one_label(self) -> "GroundTruth":
        """Ensure at least one ground truth label is provided."""
        if not self.material and not self.classification and not self.step_results:
            raise ValueError(
                "At least one of material, classification, or step_results must be provided"
            )
        return self


class UntrustedSpecReferencePage(BaseModel):
    """Converted specification page retained as non-visual provenance."""

    path: str = Field(description="Dataset-relative path to the converted PDF page")
    description: str | None = Field(
        default=None,
        description="Optional provenance-only description for the converted page",
    )


class UntrustedSpecEvidence(BaseModel):
    """Specification provenance excluded from visual model prompts and media."""

    extracted_text: str | None = Field(
        default=None,
        description="Untrusted extracted specification text for post-visual checks",
    )
    reference_pdf_pages: list[UntrustedSpecReferencePage] = Field(
        default_factory=list,
        description="Converted PDF pages retained outside visual inference media",
    )


class DatasetEntry(BaseModel):
    """Root model for dataset.jsonl entries (v0.2).

    Each line in dataset.jsonl is a JSON object conforming to this schema.
    """

    id: str = Field(description="Unique identifier (usually prim path)")
    source: SourceInfo = Field(description="Source information")
    user_prompt: str | None = Field(
        default=None, description="User prompt (for single-step tasks)"
    )
    user_prompts: list[str] | None = Field(
        default=None, description="User prompts (for multi-step tasks)"
    )
    media: MediaConfig = Field(description="Images and reference media")
    ground_truth: GroundTruth | None = Field(
        default=None, description="Ground truth labels (optional)"
    )
    usd_metadata: dict[str, Any] | None = Field(
        default=None, description="USD-specific metadata (geometry, hierarchy, etc.)"
    )
    untrusted_spec_evidence: UntrustedSpecEvidence | None = Field(
        default=None,
        description=(
            "Specification provenance used only for deterministic post-visual "
            "reconciliation"
        ),
    )

    @model_validator(mode="after")
    def validate_prompts(self) -> "DatasetEntry":
        """Ensure either user_prompt or user_prompts is present (but not both)."""
        has_single = self.user_prompt is not None
        has_multi = self.user_prompts is not None and len(self.user_prompts) > 0

        if not has_single and not has_multi:
            raise ValueError("Either user_prompt or user_prompts must be provided")

        if has_single and has_multi:
            raise ValueError("Cannot have both user_prompt and user_prompts")

        return self


# =============================================================================
# Helper Functions
# =============================================================================


def export_json_schema() -> dict[str, Any]:
    """Export JSON schemas for documentation.

    Returns:
        Dictionary with 'dataset_config' and 'dataset_entry' schemas
    """
    return {
        "dataset_config": DatasetConfig.model_json_schema(),
        "dataset_entry": DatasetEntry.model_json_schema(),
    }


def validate_dataset_config_file(file_path: str) -> DatasetConfig:
    """Validate a dataset.json file.

    Args:
        file_path: Path to dataset.json file

    Returns:
        Validated DatasetConfig instance

    Raises:
        ValidationError: If file doesn't conform to schema
        FileNotFoundError: If file doesn't exist
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return DatasetConfig(**data)


def validate_dataset_entry(entry_data: dict[str, Any]) -> DatasetEntry:
    """Validate a single dataset entry.

    Args:
        entry_data: Dictionary representing a dataset.jsonl entry

    Returns:
        Validated DatasetEntry instance

    Raises:
        ValidationError: If entry doesn't conform to schema
    """
    return DatasetEntry(**entry_data)
