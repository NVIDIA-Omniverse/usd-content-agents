# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration models for USD dataset building.

Shared by material-agent, physics-agent, and joint-agent for consistent USD dataset generation.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from world_understanding.rendering_backend_contract import (
    RENDERING_BACKEND_NAMES,
    validate_rendering_backend_name,
)

logger = logging.getLogger(__name__)


class RenderingModeConfig(BaseModel):
    """Configuration for a single rendering mode.

    Attributes:
        margin: Camera margin multiplier for framing
        cameras: List of camera directions (e.g., ["+x+y+z", "-x-y-z"])
        camera_focus_mode: "prim" (focus on prim) or "stage" (focus on entire stage)
        skip_occluded_images: Skip images where prim is occluded
        occlusion_pixel_threshold: Minimum visible pixels to keep image
        near_clip_margin: Near clipping plane margin
        far_clip_margin: Far clipping plane margin
    """

    margin: float = Field(
        default=1.2, description="Camera margin multiplier for framing"
    )
    cameras: list[str] = Field(
        default=["+x+y+z"], description="Camera directions for this mode"
    )
    camera_focus_mode: str = Field(
        default="prim",
        description="Focus mode: 'prim' (focus on prim) or 'stage' (focus on stage)",
    )
    skip_occluded_images: bool = Field(
        default=False, description="Skip images where prim is occluded"
    )
    occlusion_pixel_threshold: int = Field(
        default=20, description="Minimum visible pixels to keep image"
    )
    near_clip_margin: float = Field(
        default=0.1, description="Near clipping plane margin"
    )
    far_clip_margin: float = Field(default=0.1, description="Far clipping plane margin")

    @field_validator("camera_focus_mode")
    @classmethod
    def validate_camera_focus_mode(cls, v: str) -> str:
        """Validate camera focus mode."""
        valid_modes = {"prim", "stage"}
        if v not in valid_modes:
            raise ValueError(
                f"camera_focus_mode must be one of {valid_modes}, got '{v}'"
            )
        return v


class RendererConfig(BaseModel):
    """Renderer configuration for USD dataset building."""

    backend: str = Field(
        default="remote",
        description=(
            "Canonical rendering backend name (default: 'remote'): "
            + ", ".join(RENDERING_BACKEND_NAMES)
        ),
    )
    image_width: int = Field(default=512, description="Image width in pixels")
    image_height: int = Field(default=512, description="Image height in pixels")
    cull_style: str = Field(
        default="back",
        description="Culling style: 'back', 'front', or 'none'",
    )
    should_highlight_prim: bool = Field(
        default=False,
        description="Whether to highlight the target prim in renders",
    )
    should_assign_random_colors: bool = Field(
        default=True,
        description="Whether to assign random colors to non-highlighted prims",
    )
    camera_view_type: str = Field(
        default="corner",
        description="Camera view type: 'corner' (8 views) or 'side' (6 views)",
    )
    camera_directions: list[str] | str | None = Field(
        default=None,
        description="Optional custom camera directions (overrides camera_view_type)",
    )

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        """Reject backend names outside the shared canonical contract."""
        return validate_rendering_backend_name(value)

    @field_validator("camera_directions", mode="before")
    @classmethod
    def normalize_camera_directions(cls, v):
        """Normalize camera_directions to list (accept string or list)."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    # Old format: Per-mode margins (for backward compatibility with material_agent)
    camera_prim_with_stage_margin: float | None = Field(
        default=None,
        description="Margin for prim_with_stage mode (old format)",
    )
    camera_prim_only_margin: float | None = Field(
        default=None,
        description="Margin for prim_only mode (old format)",
    )
    camera_composition_margin: float | None = Field(
        default=None,
        description="Margin for composition mode (old format)",
    )

    # New format: Per-mode configuration with dict
    rendering_modes_config: dict[str, RenderingModeConfig] | None = Field(
        default=None,
        description="Per-mode rendering configuration (new format)",
    )

    # Optional advanced settings
    highlight_color: list[float] | None = Field(
        default=None,
        description="RGB color for highlighted prim [r, g, b] in range [0, 1]",
    )
    other_color_range: list[float] | None = Field(
        default=None,
        description="Color range for non-highlighted prims [min, max] in range [0, 1]",
    )
    background_color: list[float] | None = Field(
        default=None,
        description="RGB background color [r, g, b] in range [0, 1]",
    )
    horizontal_aperture: float = Field(
        default=1.0, description="Camera horizontal aperture in mm"
    )
    vertical_aperture: float = Field(
        default=1.0, description="Camera vertical aperture in mm"
    )
    near_clip_margin: float = Field(
        default=0.1, description="Near clipping plane margin"
    )
    far_clip_margin: float = Field(default=0.1, description="Far clipping plane margin")

    @field_validator("cull_style")
    @classmethod
    def validate_cull_style(cls, v: str) -> str:
        """Validate culling style."""
        valid_styles = {"back", "front", "none"}
        if v not in valid_styles:
            raise ValueError(f"cull_style must be one of {valid_styles}, got '{v}'")
        return v

    @field_validator("camera_view_type")
    @classmethod
    def validate_camera_view_type(cls, v: str) -> str:
        """Validate camera view type."""
        valid_types = {"corner", "side"}
        if v not in valid_types:
            raise ValueError(
                f"camera_view_type must be one of {valid_types}, got '{v}'"
            )
        return v

    def get_rendering_modes_config(
        self, rendering_modes: list[str] | dict[str, dict[str, Any]]
    ) -> dict[str, RenderingModeConfig]:
        """Parse rendering modes into unified per-mode configuration.

        Supports both old format (list with global settings) and new format
        (dict with per-mode settings).

        Args:
            rendering_modes: Either a list of mode names (old) or dict of mode configs (new)

        Returns:
            Dict mapping mode name to RenderingModeConfig

        Raises:
            ValueError: If rendering_modes format is invalid

        Examples:
            >>> # Old format
            >>> config.get_rendering_modes_config(["prim_with_stage", "prim_only"])

            >>> # New format
            >>> config.get_rendering_modes_config({
            ...     "prim_with_stage": {"margin": 3.0, "cameras": ["+x+y+z"]}
            ... })
        """
        if isinstance(rendering_modes, list):
            # Old format: Use global camera directions and per-mode margins
            camera_directions = self.camera_directions or ["+x+y+z"]
            if not isinstance(camera_directions, list):
                camera_directions = [camera_directions]

            modes_config = {}
            for mode_name in rendering_modes:
                # Determine margin and focus mode based on mode name
                if mode_name == "prim_with_stage":
                    margin = self.camera_prim_with_stage_margin or 3.0
                    focus_mode = "prim"
                    mode_cameras = camera_directions
                elif mode_name == "prim_only":
                    margin = self.camera_prim_only_margin or 1.2
                    focus_mode = "prim"
                    mode_cameras = (
                        camera_directions
                        if camera_directions != ["+x+y+z"]
                        else ["+x+y+z", "-x-y-z"]
                    )
                elif mode_name == "composition":
                    margin = self.camera_composition_margin or 6.0
                    focus_mode = "stage"
                    mode_cameras = ["+x", "+y", "+z"]
                else:
                    margin = 1.2  # Default margin
                    focus_mode = "prim"
                    mode_cameras = camera_directions

                modes_config[mode_name] = RenderingModeConfig(
                    margin=margin,
                    cameras=mode_cameras,
                    camera_focus_mode=focus_mode,
                    skip_occluded_images=False,
                    occlusion_pixel_threshold=20,
                    near_clip_margin=self.near_clip_margin,
                    far_clip_margin=self.far_clip_margin,
                )

            logger.debug(
                f"Parsed old format rendering config: {len(modes_config)} modes"
            )
            return modes_config

        elif isinstance(rendering_modes, dict):
            # New format: Use per-mode configuration
            modes_config = {}
            for mode_name, mode_dict in rendering_modes.items():
                # Parse cameras - can be list or single string
                cameras_raw = mode_dict.get("cameras", ["+x+y+z"])
                cameras = [cameras_raw] if isinstance(cameras_raw, str) else cameras_raw

                modes_config[mode_name] = RenderingModeConfig(
                    margin=mode_dict.get("margin", 1.2),
                    cameras=cameras,
                    camera_focus_mode=mode_dict.get("camera_focus_mode", "prim"),
                    skip_occluded_images=mode_dict.get("skip_occluded_images", False),
                    occlusion_pixel_threshold=mode_dict.get(
                        "occlusion_pixel_threshold", 20
                    ),
                    near_clip_margin=mode_dict.get(
                        "near_clip_margin", self.near_clip_margin
                    ),
                    far_clip_margin=mode_dict.get(
                        "far_clip_margin", self.far_clip_margin
                    ),
                )

            logger.debug(
                f"Parsed new format rendering config: {len(modes_config)} modes"
            )
            return modes_config

        else:
            raise ValueError(
                f"Invalid rendering_modes format: {type(rendering_modes)}. "
                f"Expected list (old format) or dict (new format)"
            )


class PrimFilters(BaseModel):
    """Prim filtering configuration."""

    types: list[str] | None = Field(
        default=None,
        description="USD prim types to include (e.g., ['UsdGeom.Mesh'])",
    )
    paths: list[str] | None = Field(
        default=None,
        description="Specific prim paths to include",
    )
    exclude_paths: list[str] | None = Field(
        default=None,
        description="Prim paths to exclude",
    )
    allowed_purposes: list[str] | None = Field(
        default=None,
        description=(
            "Effective UsdGeom purposes eligible for rendering. When unset, "
            "purpose does not affect selection and bbox metadata/camera framing "
            "remain default-purpose only. When set, the same purposes define "
            "selection, bbox metadata, and camera framing."
        ),
    )
    rigid_body_purpose_fallbacks: list[str] | None = Field(
        default=None,
        description=(
            "Otherwise excluded purposes that may be selected only when the "
            "nearest enabled rigid-body owner has no normally selected Gprim "
            "descendant. Intended for bounded recovery of guide-only moving "
            "assemblies without globally admitting guide helpers."
        ),
    )
    skip_fully_transparent: bool = Field(
        default=False,
        description=("Skip Gprims whose effective displayOpacity values are all zero"),
    )
    skip_unusable_bbox: bool = Field(
        default=False,
        description=("Skip Gprims without a finite, non-degenerate world-space bound"),
    )


class USDDatasetConfig(BaseModel):
    """Configuration for USD dataset building.

    This configuration is shared by both material-agent, physics-agent, and joint-agent
    for consistent USD dataset generation with multi-view rendering.

    Example:
        ```python
        config = USDDatasetConfig(
            usd_path=Path("model.usd"),
            output_dir=Path("output/dataset"),
            renderer=RendererConfig(backend="remote"),
        )
        ```
    """

    # Core paths (required)
    usd_path: Path = Field(description="Path to USD file to process")
    output_dir: Path = Field(description="Output directory for dataset")

    # Renderer configuration
    renderer: RendererConfig = Field(
        default_factory=RendererConfig,
        description="Renderer settings",
    )

    # Prim filtering
    prim_filters: PrimFilters | None = Field(
        default=None,
        description="Filters for which prims to include",
    )

    # Metadata extraction options
    extract_metadata: bool = Field(
        default=False,
        description="Extract additional prim metadata (extent, transforms, etc.)",
    )
    extract_material_bindings: bool = Field(
        default=True,
        description="Extract material bindings (resolved, direct, subassignments)",
    )
    extract_hierarchy: bool = Field(
        default=True,
        description="Extract hierarchy info (parent/children, ancestors, collections)",
    )
    extract_display_color: bool = Field(
        default=True,
        description="Extract display color from prims if available",
    )
    include_display_color_statistics: bool = Field(
        default=True,
        description="Include statistics about display colors in dataset",
    )

    # USD model building
    build_usd_model: bool = Field(
        default=True,
        description="Build USD model for efficient hierarchy queries",
    )
    export_usd_model: bool = Field(
        default=True,
        description="Export USD model as JSON alongside dataset manifest",
    )

    # USD stage preprocessing
    convert_prototypes_to_xforms: bool = Field(
        default=False,
        description="Convert abstract prototype prims (class/over) to concrete def prims",
    )
    prototype_names: list[str] | None = Field(
        default=None,
        description="Specific prototype prim names to convert (None = all with 'Prototype' in name)",
    )

    # Performance options
    batch_size: int = Field(
        default=10,
        description="Number of prims to render in parallel batches",
        gt=0,
    )
    skip_existing: bool = Field(
        default=False,
        description="Skip rendering if output files already exist",
    )
    num_workers: int | None = Field(
        default=None,
        description="Number of workers for parallel processing (None = sequential)",
    )

    # Rendering modes
    rendering_modes: list[str] = Field(
        default=["composition"],
        description="Rendering modes: 'composition', 'prim_only', etc.",
    )

    @field_validator("usd_path", "output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        if isinstance(v, Path):
            return v
        raise TypeError(f"Expected str or Path, got {type(v)}")

    @model_validator(mode="after")
    def validate_hierarchy_dependencies(self) -> "USDDatasetConfig":
        """Validate that hierarchy extraction dependencies are met."""
        if self.extract_hierarchy and not self.build_usd_model:
            raise ValueError(
                "extract_hierarchy requires build_usd_model=True for efficient queries"
            )
        return self

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        overrides: dict[str, Any] | None = None,
    ) -> "USDDatasetConfig":
        """Load configuration from YAML file with optional overrides.

        Args:
            config_path: Path to YAML configuration file
            overrides: Dictionary of override values (e.g., from CLI args)

        Returns:
            Validated USDDatasetConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid

        Example:
            ```python
            config = USDDatasetConfig.from_yaml(
                Path("config.yaml"),
                overrides={"usd_path": Path("custom.usd")}
            )
            ```
        """
        import yaml

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load YAML
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Apply overrides
        if overrides:
            data.update(overrides)

        # Resolve relative paths relative to config file directory
        config_dir = config_path.parent

        if "usd_path" in data:
            usd_path = Path(data["usd_path"])
            if not usd_path.is_absolute():
                data["usd_path"] = (config_dir / usd_path).resolve()

        if "output_dir" in data:
            output_dir = Path(data["output_dir"])
            if not output_dir.is_absolute():
                data["output_dir"] = (config_dir / output_dir).resolve()

        # Create and validate model
        return cls(**data)

    def to_context_dict(self) -> dict[str, Any]:
        """Convert config to context dictionary for workflow execution.

        Returns:
            Dictionary suitable for workflow initial_context

        Example:
            ```python
            config = USDDatasetConfig.from_yaml(Path("config.yaml"))
            context = config.to_context_dict()
            workflow.run(context)
            ```
        """
        return {
            "usd_path": str(self.usd_path),
            "output_dir": str(self.output_dir),
            "extract_metadata": self.extract_metadata,
            "extract_material_bindings": self.extract_material_bindings,
            "extract_hierarchy": self.extract_hierarchy,
            "extract_display_color": self.extract_display_color,
            "include_display_color_statistics": self.include_display_color_statistics,
            "build_usd_model": self.build_usd_model,
            "export_usd_model": self.export_usd_model,
            "convert_prototypes_to_xforms": self.convert_prototypes_to_xforms,
            "prototype_names": self.prototype_names,
            "batch_size": self.batch_size,
            "skip_existing": self.skip_existing,
            "num_workers": self.num_workers,
            "rendering_modes": self.rendering_modes,
            "renderer_config": self.renderer.model_dump(),
            "prim_filters": (
                self.prim_filters.model_dump() if self.prim_filters else None
            ),
        }
