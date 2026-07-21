# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD renderer provisioning task."""

import logging
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    CameraFocusMode,
    CameraSpec,
    CameraViewType,
    RenderingBackend,
    RenderingConfig,
)
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

# Rendering mode categories
RGB_RENDERING_MODES = [
    "composition",
    "prim_with_stage",
    "prim_only",
]
SENSOR_RENDERING_MODES = [
    "linear_depth",
    "depth",
    "instance_id_segmentation",
]
ALL_VALID_RENDERING_MODES = RGB_RENDERING_MODES + SENSOR_RENDERING_MODES


def validate_rendering_modes(
    modes: list[str],
    backend: RenderingBackend,
    base_mode_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate rendering modes against backend capabilities.

    Args:
        modes: List of requested rendering modes
        backend: Rendering backend instance
        base_mode_map: Optional mapping of custom mode names to core mode types.
            Used to resolve modes like "prim_only_original" to "prim_only" for
            validation.

    Returns:
        Tuple of (valid_modes, warnings):
            - valid_modes: List of modes supported by the backend
            - warnings: List of warning messages for unsupported modes
    """
    if not modes:
        return [], []

    # Resolve each mode to its base mode for validation
    resolved = (
        {m: base_mode_map.get(m, m) for m in modes}
        if base_mode_map
        else {m: m for m in modes}
    )

    # Separate RGB and sensor modes using resolved base modes
    rgb_modes = [m for m in modes if resolved[m] in RGB_RENDERING_MODES]
    sensor_modes = [m for m in modes if resolved[m] in SENSOR_RENDERING_MODES]
    invalid_modes = [m for m in modes if resolved[m] not in ALL_VALID_RENDERING_MODES]

    warnings = []
    valid_modes = rgb_modes.copy()

    # Warn about invalid modes
    if invalid_modes:
        warnings.append(
            f"Invalid rendering modes: {invalid_modes}. "
            f"Valid modes are: {ALL_VALID_RENDERING_MODES}"
        )

    # Check sensor support
    if sensor_modes:
        if not backend.supports_sensors():
            warnings.append(
                f"Backend '{backend.__class__.__name__}' does not "
                f"support sensor modes: {sensor_modes}. "
                f"These modes will be skipped. Use 'remote' backend for "
                f"sensor support."
            )
        else:
            # Check which sensor modes are supported
            supported = backend.get_supported_sensor_modes()
            unsupported = [m for m in sensor_modes if m not in supported]
            if unsupported:
                warnings.append(
                    f"Backend '{backend.__class__.__name__}' does not "
                    f"support sensor modes: {unsupported}. "
                    f"Supported sensor modes: {supported}"
                )
            # Add only supported sensor modes
            valid_modes.extend([m for m in sensor_modes if m in supported])

    return valid_modes, warnings


def parse_camera_configuration(
    renderer_config: dict[str, Any],
    global_defaults: dict[str, Any] | None = None,
) -> dict[str, list[CameraSpec]]:
    """Parse camera configuration from renderer config.

    Supports three configuration levels:
    1. Legacy (backward compatible): camera_view_type + camera_directions
    2. Enhanced simple: cameras.views with per-view settings
    3. Per-mode: rendering_modes with mode-specific cameras

    Args:
        renderer_config: Renderer configuration dictionary
        global_defaults: Optional global default settings

    Returns:
        Dict mapping render_mode -> list of CameraSpec
        Special key "__all__" means cameras apply to all modes
    """
    if global_defaults is None:
        global_defaults = {}

    # Extract global camera defaults from renderer_config
    camera_defaults = {
        "focal_length": renderer_config.get("camera_focal_length", 100.0),
        "horizontal_aperture": renderer_config.get("camera_horizontal_aperture", 1.0),
        "vertical_aperture": renderer_config.get("camera_vertical_aperture", 1.0),
        "near_clip_margin": renderer_config.get("near_clip_margin", 0.1),
        "far_clip_margin": renderer_config.get("far_clip_margin", 0.1),
        **global_defaults,
    }

    # Update with explicit camera_defaults if present
    if "camera_defaults" in renderer_config:
        camera_defaults.update(renderer_config["camera_defaults"])

    # LEVEL 3: Per-mode camera configuration
    if "rendering_modes" in renderer_config:
        modes_config = renderer_config["rendering_modes"]

        # Check if any mode has camera configuration
        has_per_mode_cameras = False
        if isinstance(modes_config, dict):
            for mode_config in modes_config.values():
                # Check for dict with cameras/reference OR direct list
                if isinstance(mode_config, list):
                    has_per_mode_cameras = True
                    break
                elif isinstance(mode_config, dict) and (
                    "cameras" in mode_config or "use_cameras_from" in mode_config
                ):
                    has_per_mode_cameras = True
                    break

        if has_per_mode_cameras:
            return _parse_per_mode_cameras(modes_config, camera_defaults)

    # LEVEL 2: Enhanced simple cameras configuration
    if "cameras" in renderer_config:
        cameras = _parse_camera_list(
            renderer_config["cameras"].get("views", []),
            {**camera_defaults, **renderer_config["cameras"].get("defaults", {})},
        )
        return {"__all__": cameras}

    # LEVEL 1: Legacy backward compatible format
    return {"__all__": _parse_legacy_cameras(renderer_config, camera_defaults)}


def _parse_per_mode_cameras(
    modes_config: dict[str, Any],
    global_defaults: dict[str, Any],
) -> dict[str, list[CameraSpec]]:
    """Parse per-mode camera configurations (Level 3)."""
    result = {}

    for mode_name, mode_config in modes_config.items():
        if isinstance(mode_config, str):
            # Simple string mode name - will use global cameras
            continue

        if not isinstance(mode_config, dict):
            # Simple list - shorthand for cameras
            if isinstance(mode_config, list):
                result[mode_name] = _parse_camera_list(mode_config, global_defaults)
            continue

        # Check for camera reference
        if "use_cameras_from" in mode_config:
            ref_mode = mode_config["use_cameras_from"]
            if ref_mode in result:
                result[mode_name] = result[ref_mode]
            else:
                logger.warning(
                    f"Mode '{mode_name}' references '{ref_mode}' cameras, "
                    f"but '{ref_mode}' is not yet defined. Skipping."
                )
            continue

        # Extract mode-level settings (applied to all cameras in this mode)
        mode_settings = {
            k: v
            for k, v in mode_config.items()
            if k
            in [
                "margin",
                "focal_length",
                "horizontal_aperture",
                "vertical_aperture",
                "near_clip_margin",
                "far_clip_margin",
            ]
        }

        # Merge: global defaults < mode settings
        mode_defaults = {**global_defaults, **mode_settings}

        # Parse cameras for this mode
        if "cameras" in mode_config:
            cameras_config = mode_config["cameras"]
            result[mode_name] = _parse_camera_list(cameras_config, mode_defaults)

    return result


def parse_occlusion_settings(
    renderer_config: dict[str, Any],
) -> dict[str, bool]:
    """Parse per-mode skip_occluded_images settings from renderer config.

    Args:
        renderer_config: Renderer configuration dictionary

    Returns:
        Dict mapping render_mode -> skip_occluded_images bool
    """
    result = {}

    # Check for per-mode occlusion settings
    if "rendering_modes" in renderer_config:
        modes_config = renderer_config["rendering_modes"]

        if isinstance(modes_config, dict):
            for mode_name, mode_config in modes_config.items():
                if (
                    isinstance(mode_config, dict)
                    and "skip_occluded_images" in mode_config
                ):
                    result[mode_name] = mode_config["skip_occluded_images"]

    return result


def parse_focus_mode_settings(
    renderer_config: dict[str, Any],
) -> dict[str, "CameraFocusMode"]:
    """Parse per-mode camera_focus_mode settings from renderer config.

    Args:
        renderer_config: Renderer configuration dictionary

    Returns:
        Dict mapping render_mode -> CameraFocusMode
    """
    from world_understanding.functions.graphics.rendering import CameraFocusMode

    result = {}

    # Check for per-mode focus settings
    if "rendering_modes" in renderer_config:
        modes_config = renderer_config["rendering_modes"]

        if isinstance(modes_config, dict):
            for mode_name, mode_config in modes_config.items():
                if isinstance(mode_config, dict) and "camera_focus_mode" in mode_config:
                    focus_mode_str = mode_config["camera_focus_mode"]
                    try:
                        result[mode_name] = CameraFocusMode(focus_mode_str)
                    except ValueError:
                        logger.warning(
                            f"Invalid camera_focus_mode '{focus_mode_str}' for mode '{mode_name}'. "
                            f"Valid values: 'prim', 'stage'"
                        )

    return result


def parse_original_material_settings(
    renderer_config: dict[str, Any],
) -> dict[str, bool]:
    """Parse per-mode use_original_materials settings from renderer config.

    Args:
        renderer_config: Renderer configuration dictionary

    Returns:
        Dict mapping render_mode -> use_original_materials bool
    """
    result = {}

    if "rendering_modes" in renderer_config:
        modes_config = renderer_config["rendering_modes"]

        if isinstance(modes_config, dict):
            for mode_name, mode_config in modes_config.items():
                if (
                    isinstance(mode_config, dict)
                    and "use_original_materials" in mode_config
                ):
                    result[mode_name] = mode_config["use_original_materials"]

    return result


def parse_base_mode_settings(
    renderer_config: dict[str, Any],
) -> dict[str, str]:
    """Parse per-mode base_mode settings from renderer config.

    Custom mode names (e.g., "prim_only_original") can map to core mode types
    (e.g., "prim_only") via a base_mode field. This allows YAML configs to have
    multiple variants of the same core mode.

    Args:
        renderer_config: Renderer configuration dictionary

    Returns:
        Dict mapping render_mode -> base_mode string
    """
    result = {}

    if "rendering_modes" in renderer_config:
        modes_config = renderer_config["rendering_modes"]

        if isinstance(modes_config, dict):
            for mode_name, mode_config in modes_config.items():
                if isinstance(mode_config, dict) and "base_mode" in mode_config:
                    result[mode_name] = mode_config["base_mode"]

    return result


def _parse_camera_list(
    cameras_config: list[Any],
    defaults: dict[str, Any],
) -> list[CameraSpec]:
    """Parse a list of camera configurations."""
    cameras = []

    for cam_config in cameras_config:
        if isinstance(cam_config, str):
            # Shorthand: just direction string
            cameras.append(
                CameraSpec(
                    direction=cam_config,
                    margin=defaults.get("margin"),
                    focal_length=defaults.get("focal_length"),
                    horizontal_aperture=defaults.get("horizontal_aperture"),
                    vertical_aperture=defaults.get("vertical_aperture"),
                    near_clip_margin=defaults.get("near_clip_margin"),
                    far_clip_margin=defaults.get("far_clip_margin"),
                )
            )
        elif isinstance(cam_config, dict):
            # Full spec: merge with defaults
            spec = {**defaults, **cam_config}
            cameras.append(
                CameraSpec(
                    direction=spec.get("direction"),
                    view_type=spec.get("view_type"),
                    margin=spec.get("margin"),
                    focal_length=spec.get("focal_length"),
                    horizontal_aperture=spec.get("horizontal_aperture"),
                    vertical_aperture=spec.get("vertical_aperture"),
                    near_clip_margin=spec.get("near_clip_margin"),
                    far_clip_margin=spec.get("far_clip_margin"),
                    name=spec.get("name"),
                )
            )

    return cameras


def _parse_legacy_cameras(
    renderer_config: dict[str, Any],
    defaults: dict[str, Any],
) -> list[CameraSpec]:
    """Parse legacy camera configuration (Level 1 - backward compatible)."""
    # Support both "camera_directions" and "camera_corners" parameter names
    camera_directions = renderer_config.get(
        "camera_corners", renderer_config.get("camera_directions")
    )

    # Get camera view type
    camera_view_type_str = renderer_config.get("camera_view_type", "corner")
    try:
        camera_view_type = CameraViewType(camera_view_type_str)
    except ValueError:
        logger.warning(
            f"Invalid camera_view_type: {camera_view_type_str}. Falling back to CORNER."
        )
        camera_view_type = CameraViewType.CORNER

    # If no directions specified, use defaults based on view type
    if camera_directions is None:
        if camera_view_type == CameraViewType.SIDE:
            camera_directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
        else:  # CORNER
            camera_directions = [
                "+x+y+z",
                "-x+y+z",
                "-x-y+z",
                "+x-y+z",
                "+x+y-z",
                "-x+y-z",
                "-x-y-z",
                "+x-y-z",
            ]

    # Get margin (legacy configs use camera_margin)
    margin = renderer_config.get("camera_margin", defaults.get("margin", 1.0))

    # Build CameraSpec list (let CameraSpec auto-infer view_type from direction)
    return [
        CameraSpec(
            direction=direction,
            view_type=None,  # Let CameraSpec auto-infer from direction
            margin=margin,
            focal_length=defaults.get("focal_length"),
            horizontal_aperture=defaults.get("horizontal_aperture"),
            vertical_aperture=defaults.get("vertical_aperture"),
            near_clip_margin=defaults.get("near_clip_margin"),
            far_clip_margin=defaults.get("far_clip_margin"),
        )
        for direction in camera_directions
    ]


class USDRendererProvisioningTask(Task):
    """Provision USD renderer backend and configuration."""

    def __init__(self):
        self.name = "USDRendererProvisioning"
        self.description = "Provision USD renderer backend and resources"

    def run(self, context: dict[str, Any], object_store: ObjectStore) -> dict[str, Any]:
        """Provision USD renderer based on configuration.

        Expected context inputs:
            - renderer_config: Renderer configuration dict

        Updates context with:
            - rendering_backend: Initialized renderer backend
            - rendering_config: RenderingConfig instance

        Stores in object_store:
            - rendering_backend: The backend instance (for large objects)
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        renderer_config = context.get("renderer_config", {})
        backend_type = renderer_config.get("backend", "remote")

        listener.info(f"Provisioning {backend_type} USD renderer backend")

        rendering_backend = create_rendering_backend(backend_type, renderer_config)
        listener.info(f"Using {backend_type} rendering backend")

        # Get camera view type
        camera_view_type_str = renderer_config.get("camera_view_type", "corner")
        try:
            camera_view_type = CameraViewType(camera_view_type_str.lower())
        except ValueError:
            listener.warning(
                f"Invalid camera_view_type '{camera_view_type_str}', using 'corner'"
            )
            camera_view_type = CameraViewType.CORNER

        # Get camera directions or use defaults based on view type
        # Support both "camera_directions" and "camera_corners" parameter names
        # "camera_corners" takes precedence (used by SimReady unified configs)
        camera_directions = renderer_config.get(
            "camera_corners", renderer_config.get("camera_directions")
        )
        if camera_directions is None:
            if camera_view_type == CameraViewType.SIDE:
                # Default side view directions (6 cardinal directions)
                camera_directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
            else:  # CameraViewType.CORNER
                # Default corner view directions (8 corners)
                camera_directions = [
                    "+x+y+z",
                    "-x+y+z",
                    "-x-y+z",
                    "+x-y+z",
                    "+x+y-z",
                    "-x+y-z",
                    "-x-y-z",
                    "+x-y-z",
                ]

        # Create rendering configuration
        # Note: image_height is stored separately as RenderingConfig doesn't
        # have this field
        # Set camera_name_prefix based on view type
        if camera_view_type == CameraViewType.CORNER:
            camera_name_prefix = "CornerViewCamera"
        else:
            camera_name_prefix = "SideViewCamera"

        # Extract color configurations
        highlight_color = renderer_config.get("highlight_color", [1.0, 0.0, 0.0])
        if isinstance(highlight_color, list):
            highlight_color = tuple(highlight_color)

        other_color_range = renderer_config.get("other_color_range", [0.1, 0.2])
        if isinstance(other_color_range, list):
            other_color_range = tuple(other_color_range)

        background_color = renderer_config.get("background_color", [0, 0, 0])
        if isinstance(background_color, list):
            background_color = tuple(background_color)

        # Parse per-mode camera configuration (supports new multi-view system)
        # This will return an empty dict if no per-mode cameras are configured
        camera_specs = parse_camera_configuration(renderer_config)

        # Parse per-mode occlusion settings
        per_mode_skip_occluded = parse_occlusion_settings(renderer_config)

        # Parse per-mode focus settings
        per_mode_focus_mode = parse_focus_mode_settings(renderer_config)

        # Parse per-mode original material settings
        per_mode_use_original_materials = parse_original_material_settings(
            renderer_config
        )

        # Parse per-mode base mode settings
        per_mode_base_mode = parse_base_mode_settings(renderer_config)

        rendering_config = RenderingConfig(
            image_width=renderer_config.get("image_width", 512),
            cull_style=renderer_config.get("cull_style", "back"),
            should_highlight_prim=renderer_config.get("should_highlight_prim", False),
            should_assign_random_colors=renderer_config.get(
                "should_assign_random_colors", True
            ),
            composition_show_full_scene=renderer_config.get(
                "composition_show_full_scene", True
            ),
            # Newly configurable parameters
            highlight_color=highlight_color,
            other_color_range=other_color_range,
            background_color=background_color,
            use_background_color=renderer_config.get("use_background_color", True),
            should_reset_materials=renderer_config.get("should_reset_materials", True),
            use_lights=renderer_config.get("use_lights", False),
            # Contour configuration
            contour_method=renderer_config.get("contour_method", "red"),
            contour_black_threshold=renderer_config.get("contour_black_threshold", 20),
            # Camera configuration (legacy)
            camera_view_type=camera_view_type,
            camera_name_prefix=camera_name_prefix,
            camera_ordering=camera_directions,
            # Camera margin configuration (legacy)
            camera_prim_focus_margin=renderer_config.get(
                "camera_prim_focus_margin", 1.1
            ),
            camera_prim_with_stage_margin=renderer_config.get(
                "camera_prim_with_stage_margin", 3.0
            ),
            camera_composition_margin=renderer_config.get(
                "camera_composition_margin", 3.0
            ),
            # Per-mode camera specifications (new multi-view system)
            camera_specs=camera_specs,
            # Occlusion detection configuration
            skip_occluded_images=renderer_config.get("skip_occluded_images", False),
            occlusion_pixel_threshold=renderer_config.get(
                "occlusion_pixel_threshold", 10
            ),
            # Per-mode occlusion settings
            per_mode_skip_occluded=per_mode_skip_occluded,
            # Per-mode focus settings
            per_mode_focus_mode=per_mode_focus_mode,
            # Per-mode original material settings
            per_mode_use_original_materials=per_mode_use_original_materials,
            # Per-mode base mode mapping
            per_mode_base_mode=per_mode_base_mode,
        )

        # Store rendering modes configuration
        # Users can enable multiple rendering modes in the config
        # Rendering modes include both RGB modes (composition, prim_with_stage, prim_only)
        # and sensor modes (linear_depth, depth, instance_id_segmentation)
        # Check context first (for unified pipeline), then renderer_config, then defaults
        rendering_modes_raw = context.get(
            "rendering_modes",
            renderer_config.get("rendering_modes", ["prim_with_stage", "prim_only"]),
        )

        # Extract mode names from either list or dict format
        if isinstance(rendering_modes_raw, dict):
            # Dict format: keys are mode names
            rendering_modes = list(rendering_modes_raw.keys())
        elif isinstance(rendering_modes_raw, list):
            # List format: already mode names
            rendering_modes = rendering_modes_raw
        else:
            # Single string: convert to list
            rendering_modes = [rendering_modes_raw]

        # Validate rendering modes against backend capabilities
        valid_modes, validation_warnings = validate_rendering_modes(
            rendering_modes, rendering_backend, base_mode_map=per_mode_base_mode
        )

        # Log warnings for invalid/unsupported modes
        for warning in validation_warnings:
            listener.warning(warning)

        # Use validated modes, or fall back to defaults if none are valid
        if not valid_modes:
            # Default to prim_with_stage and prim_only modes
            listener.warning(
                "No valid rendering modes specified. "
                "Defaulting to: ['prim_with_stage', 'prim_only']"
            )
            valid_modes = ["prim_with_stage", "prim_only"]

        # Store all rendering modes (unified list for config compatibility)
        context["rendering_modes"] = valid_modes

        # Also store RGB and sensor modes separately for easier processing
        # Use base_mode resolution for custom mode names
        rgb_modes = [
            m
            for m in valid_modes
            if per_mode_base_mode.get(m, m) in RGB_RENDERING_MODES
        ]
        sensor_modes = [
            m
            for m in valid_modes
            if per_mode_base_mode.get(m, m) in SENSOR_RENDERING_MODES
        ]

        context["rgb_rendering_modes"] = rgb_modes
        context["sensor_rendering_modes"] = sensor_modes

        if sensor_modes:
            listener.info(
                f"Enabled rendering modes: RGB={rgb_modes}, Sensors={sensor_modes}"
            )
        else:
            listener.info(f"Enabled rendering modes: {rgb_modes}")

        # Store image_height separately in context for backends that support it
        context["image_height"] = renderer_config.get("image_height", 512)

        # Store backend in object store (it may be large)
        object_store.set("rendering_backend", rendering_backend)
        object_store.set("rendering_config", rendering_config)

        # Also keep references in context
        context["rendering_backend"] = rendering_backend
        context["rendering_config"] = rendering_config

        listener.info(
            f"USD renderer provisioned with config: "
            f"size={rendering_config.image_width}x{context['image_height']}, "
            f"cull={rendering_config.cull_style}, "
            f"camera_view_type={camera_view_type.value}, "
            f"num_views={len(camera_directions)}, "
            f"rendering_modes={rendering_modes}"
        )

        return context
