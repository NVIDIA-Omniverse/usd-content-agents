# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified task for rendering USD with optional flattening.

This task provides a flexible render capability that:
1. Takes an arbitrary USD file (from apply step or any source)
2. Optionally flattens it for rendering
3. Renders it to specified output path(s)
4. Supports both standalone and workflow-integrated usage patterns
"""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image
from pxr import Usd
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.render_validation import (
    RENDER_BLANK_IMAGE,
    ImageValidationResult,
    validate_image_artifact,
)
from world_understanding.functions.graphics.rendering import (
    CameraFocusMode,
    CameraViewType,
    RenderingConfig,
    format_direction_for_filename,
)
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
    validate_rendering_backend_name,
)
from world_understanding.utils.image_utils import paste_on_background
from world_understanding.utils.usd.camera import (
    add_corner_view_camera,
    add_focused_corner_view_camera,
    add_focused_side_view_camera,
    add_side_view_camera,
)
from world_understanding.utils.usd.prim import get_bbox_from_prim
from world_understanding.utils.usd.stage import prepare_stage_for_render

logger = logging.getLogger(__name__)

_SIDE_VIEW_DIRECTIONS = {"+x", "-x", "+y", "-y", "+z", "-z"}
_FORCED_SERIAL_RENDERING_BACKENDS = frozenset({"ovrtx", "warp"})


class BlankRenderStatsDict(TypedDict, total=False):
    reason: str
    unique_colors: int
    dominant_color_ratio: float
    luma_std: float


def _blank_final_render_error(
    output_path: str,
    stats: BlankRenderStatsDict,
    backend_type: str,
) -> str:
    details = (
        f"Blank final render detected at {output_path}: "
        f"reason={stats.get('reason', 'unknown')}, "
        f"unique_colors={stats.get('unique_colors', 'unknown')}, "
        f"dominant_color_ratio={stats.get('dominant_color_ratio', 'unknown')}, "
        f"luma_std={stats.get('luma_std', 'unknown')}."
    )
    if backend_type == "ovrtx":
        return (
            f"{details} Check OVRTX rendering endpoint logs, renderer lighting, "
            "and WU_OVRTX_DEFAULT_HDRI."
        )
    if backend_type == "remote":
        return f"{details} Check remote rendering endpoint logs."
    return f"{details} Check the {backend_type} rendering backend configuration."


def _scene_render_max_workers(
    *,
    render_config: dict[str, Any],
    backend_type: str,
    camera_count: int,
) -> int:
    """Resolve worker count without parallelizing local GPU backends."""
    configured_max_workers = render_config.get("max_workers")
    if backend_type in _FORCED_SERIAL_RENDERING_BACKENDS:
        max_workers = 1
    elif configured_max_workers is not None:
        max_workers = max(1, int(configured_max_workers))
    elif backend_type == "remote":
        max_workers = 1
    else:
        max_workers = 2
    return min(camera_count, max_workers)


def _validation_issue_codes(validation: ImageValidationResult) -> list[str]:
    return [issue.code for issue in validation.issues]


def _is_blank_render_validation(validation: ImageValidationResult) -> bool:
    return RENDER_BLANK_IMAGE in _validation_issue_codes(validation)


def _failed_render_attempt_path(
    output_path: Path,
    *,
    attempt: int,
    reason: str,
) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.{reason}_attempt_{attempt}{output_path.suffix}"
    )


def _preserve_failed_render_stage(
    stage: Usd.Stage,
    *,
    output_base_path: Path,
    output_path: Path,
    attempt: int,
) -> Path | None:
    debug_dir = output_base_path / "_render_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stage_path = debug_dir / f"{output_path.stem}.attempt_{attempt}.usda"
    if stage.GetRootLayer().Export(str(stage_path)):
        return stage_path
    return None


class RenderTask(Task):
    """Unified task to flatten and render a USD file.

    This task handles multiple usage patterns:
    - Standalone rendering with explicit paths
    - Workflow-integrated rendering with flexible context keys
    - Optional rendering that can be skipped via flag
    - Optional flattening before rendering

    Input context keys (flexible):
        Path inputs (in priority order):
        - output_usd_path: USD file with materials applied (preferred)
        - render_usd_path: Pre-flattened USD for rendering (workflow usage)
        - input_usd_path: Original input USD file path (fallback)

        Output directory (in priority order):
        - output_base_path: Explicit output directory
        - render_output_dir: Alternative output directory name
        - (defaults to input USD parent directory)

        Configuration:
        - render_enabled: Whether to perform rendering (default: True)
        - flatten_before_render: Whether to flatten USD before rendering (default: True)
        - render_config: Rendering configuration dictionary with:
            - backend: Rendering backend ("remote", "warp", "ovrtx", or
                "mock"; default: "remote")
            - image_width: Image width in pixels (default: 1024)
            - image_height: Image height in pixels (default: image_width)
            - camera_corners: List of camera corners to render from (default: ["+x+y+z"])
            - camera_corner: Alternative single corner specification
            - camera_margin: Camera margin multiplier (default: 1.2)
            - background_color: Background color as [R, G, B] in 0-1 range (default: [1.0, 1.0, 1.0])
            - max_retries: REST renderer retry count (optional)
            - retry_delay: REST renderer retry delay (optional)
            - retry_backoff_factor: REST renderer backoff factor (optional)
            - retry_jitter: REST renderer jitter (optional)
            - max_attempts: Per-camera render attempts after task-level validation (optional)
            - allow_partial_renders: If True, return successfully rendered cameras
                when other requested cameras fail validation (default: False)
            - max_workers: Camera render worker count override (optional).
                Remote rendering defaults to 1 to avoid sharing a USD stage
                across concurrent REST render requests.
            OvRTX-specific keys (backend == "ovrtx"):
            - log_level: Logging verbosity for OvRTX subprocess (str, default: "warn")
            - ovrtx_venv_dir: Path to the OvRTX virtual environment directory
                (str, optional; defaults to ~/.cache/wu/ovrtx_venv)
            - num_sensor_updates: Progressive path-tracer step iterations
                per frame (int, default: 500). Sized for ``render_mode="pt"``;
                rt2 caps quality regardless of step count, so paired with
                ``pt`` is the configuration used here.
            - render_mode: OvRTX render mode (``rt1`` | ``rt2`` | ``pt``,
                default ``"pt"``). The material-agent default is ``pt``
                (Kit's ground-truth mode) because final renders here are
                presentation-quality. The 500-step ``num_sensor_updates``
                default is sized for the ``pt`` convergence plateau; ``rt2``
                caps at ~27 dB PSNR regardless of step count and is the
                fast-iteration default of the underlying backend, which is
                not what this task needs.

    Output context keys:
        - flattened_usd_path: Path to flattened USD (if flattening was done)
        - rendered_image_paths: List of all rendered image paths
        - rendered_image_path: Path to the first rendered image (backward compatibility)
        - rendering_skipped: Boolean indicating if rendering was skipped
        - rendering_stats: Dictionary with rendering statistics
    """

    def __init__(self):
        """Initialize the render task."""
        self.name = "Render"
        self.description = "Flatten and render USD file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Flatten and render the USD file.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with rendering results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Test listener immediately
        listener.info("🎬 Render task starting...")

        # Check if rendering is enabled (default True for backward compatibility)
        render_enabled = context.get("render_enabled", True)
        if not render_enabled:
            listener.info(
                "Rendering is disabled (render_enabled=False), skipping render task"
            )
            context["rendering_skipped"] = True
            return context

        # Get USD file path - support multiple context key patterns
        # Priority: output_usd_path > render_usd_path > input_usd_path
        # Note: output_usd_path takes priority because it represents the freshly
        # created USD file with materials applied (e.g., from apply/assign steps)
        input_usd_path = (
            context.get("output_usd_path")
            or context.get("render_usd_path")
            or context.get("input_usd_path")
        )

        if not input_usd_path:
            raise ValueError(
                "No USD file path found. Provide one of: output_usd_path, render_usd_path, or input_usd_path"
            )

        input_usd_path = Path(input_usd_path)
        if not input_usd_path.exists():
            listener.warning(
                f"USD file not found: {input_usd_path}, skipping rendering"
            )
            context["rendering_skipped"] = True
            return context

        render_config = context.get("render_config", {})
        backend_type = validate_rendering_backend_name(
            render_config.get("backend", "remote")
        )

        # Get output directory - support multiple context key patterns
        # Priority: output_base_path > render_output_dir > input USD parent dir
        output_base_path = context.get("output_base_path") or context.get(
            "render_output_dir"
        )
        if not output_base_path:
            output_base_path = input_usd_path.parent
            listener.info(
                f"No output directory specified, using input USD parent: {output_base_path}"
            )

        output_base_path = Path(output_base_path)
        output_base_path.mkdir(parents=True, exist_ok=True)

        flatten_before_render = context.get("flatten_before_render", True)

        # Track which USD path to use for output naming (before potential flattening)
        original_usd_path = input_usd_path

        listener.info(f"Rendering USD: {input_usd_path}")
        listener.info(f"Output directory: {output_base_path}")
        listener.info(f"Flatten before render: {flatten_before_render}")

        # Step 1: Prepare the USD stage for rendering
        stage = Usd.Stage.Open(str(input_usd_path))
        if not stage:
            raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

        prepared_stage, preparation_metadata = prepare_stage_for_render(
            stage,
            flatten=flatten_before_render,
            normalize_materials=True,
        )
        render_asset_base_dir = preparation_metadata.get("asset_base_dir")
        listener.info(f"Render stage preparation: {preparation_metadata}")

        # Export the prepared stage. Flattening produces a self-contained stage;
        # the non-flatten path still writes a converted temp layer so the
        # original USD is not mutated.
        if flatten_before_render:
            flattened_usd_path = output_base_path / f"{input_usd_path.stem}_flat.usd"
            listener.info(f"Flattening USD to: {flattened_usd_path}")

            try:
                # Save the flattened stage
                prepared_stage.GetRootLayer().Export(str(flattened_usd_path))

                listener.info(f"✓ Flattened USD saved to: {flattened_usd_path}")
                context["flattened_usd_path"] = str(flattened_usd_path)

                # Use flattened USD for rendering
                usd_to_render = flattened_usd_path

            except Exception as e:
                listener.error(f"Failed to flatten USD: {e}")
                raise RuntimeError(f"USD flattening failed: {e}") from e
        else:
            converted_path = output_base_path / f"{input_usd_path.stem}_converted.usda"
            prepared_stage.GetRootLayer().Export(str(converted_path))
            usd_to_render = converted_path
            listener.info(f"Converted MDL shaders, rendering from: {converted_path}")

        # Step 2: Render the USD
        listener.info(f"Starting rendering from: {usd_to_render}")

        # Extract render settings
        image_width = render_config.get("image_width", 1024)
        image_height = render_config.get("image_height", image_width)

        # Support both single corner (string) and multiple corners (list)
        camera_corners_config = render_config.get(
            "camera_corners"
        ) or render_config.get("camera_corner", "+x+y+z")
        if isinstance(camera_corners_config, str):
            camera_corners = [camera_corners_config]
        else:
            camera_corners = camera_corners_config

        camera_margin = render_config.get("camera_margin", 1.2)

        # Background color: config uses 0-1 range, convert to 0-255 for PIL
        bg_color_normalized = render_config.get("background_color", [1.0, 1.0, 1.0])
        background_color = tuple(int(c * 255) for c in bg_color_normalized)
        allow_partial_renders = bool(render_config.get("allow_partial_renders", False))

        listener.info("Rendering configuration:")
        listener.info(f"  Backend: {backend_type}")
        listener.info(f"  Image size: {image_width}x{image_height}")
        listener.info(
            f"  Camera corners: {', '.join(camera_corners)} ({len(camera_corners)} views)"
        )
        listener.info(f"  Camera margin: {camera_margin}")
        listener.info(f"  Background color: {background_color}")
        listener.info(f"  Allow partial renders: {allow_partial_renders}")

        # Open the USD stage for rendering
        stage = Usd.Stage.Open(str(usd_to_render))
        if not stage:
            raise RuntimeError(
                f"Failed to open USD stage for rendering: {usd_to_render}"
            )

        # Calculate apertures based on desired aspect ratio
        aspect_ratio = image_width / image_height
        if aspect_ratio >= 1.0:
            # Landscape or square: keep horizontal at 36, adjust vertical
            horizontal_aperture = 36.0
            vertical_aperture = 36.0 / aspect_ratio
        else:
            # Portrait: keep vertical at 36, adjust horizontal
            vertical_aperture = 36.0
            horizontal_aperture = 36.0 * aspect_ratio

        listener.info(
            f"Camera apertures: {horizontal_aperture:.2f}mm x {vertical_aperture:.2f}mm "
            f"(aspect ratio: {aspect_ratio:.2f})"
        )

        # Clear existing material bindings before rendering if requested.
        # This shows only the newly-assigned materials from the pipeline,
        # making it easier to verify predictions against a neutral surface.
        clear_materials = render_config.get("clear_materials", False)
        if clear_materials:
            from world_understanding.utils.usd.prim import nullify_materials

            listener.info("Clearing original material bindings (clear_materials=True)")
            nullify_materials(stage)

        # Scope to prim_path: hide everything outside the subtree and
        # focus the camera on the target prim only.
        prim_path = render_config.get("prim_path")
        focus_prim = stage.GetPrimAtPath(prim_path) if prim_path else None
        if focus_prim and focus_prim.IsValid():
            listener.info(f"Isolating prim for render: {prim_path}")
            from world_understanding.functions.graphics.rendering import (
                hide_prims_outside_subtree,
            )

            hide_prims_outside_subtree(stage, prim_path)
            listener.info(f"Hidden prims outside {prim_path} subtree")
        elif prim_path:
            listener.warning(f"Prim '{prim_path}' not found, rendering full scene")
            prim_path = None
            focus_prim = None

        if focus_prim:
            scene_bbox = get_bbox_from_prim(focus_prim)
        else:
            scene_bbox = get_bbox_from_prim(stage.GetPseudoRoot())
        aligned_range = scene_bbox.ComputeAlignedRange()
        bbox_min = aligned_range.GetMin()
        bbox_max = aligned_range.GetMax()

        scene_size_x = bbox_max[0] - bbox_min[0]
        scene_size_y = bbox_max[1] - bbox_min[1]
        scene_size_z = bbox_max[2] - bbox_min[2]

        listener.info(
            f"Scene bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to "
            f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]"
        )
        listener.info(
            f"Scene dimensions: {scene_size_x:.2f} × {scene_size_y:.2f} × {scene_size_z:.2f}"
        )

        # Final renders use the OVRTX path-traced quality profile unless the
        # caller overrides it. Other backends ignore these OVRTX-only options.
        backend_config = {
            "num_sensor_updates": 500,
            "render_mode": "pt",
            **render_config,
        }
        rendering_backend = create_rendering_backend(backend_type, backend_config)
        listener.info(f"Using {backend_type} rendering backend")
        # Set up rendering configuration
        rendering_config = RenderingConfig(
            image_width=image_width,
            cull_style="back",
            # For final render, don't modify materials or colors
            should_reset_materials=False,
            should_highlight_prim=False,
            should_assign_random_colors=False,
            # Use white background by default for clean presentation
            background_color=background_color,
            use_background_color=True,
            # Use lights if available in the scene
            use_lights=True,
            # Focus on the entire stage
            camera_focus_mode=CameraFocusMode.STAGE,
            camera_view_type=CameraViewType.CORNER,
        )

        # Create all cameras
        camera_infos = []
        listener.info(f"Creating {len(camera_corners)} camera(s)...")

        for i, camera_corner in enumerate(camera_corners):
            camera_path = (
                f"/RenderCamera_{i}" if len(camera_corners) > 1 else "/RenderCamera"
            )

            listener.info(
                f"  [{i + 1}/{len(camera_corners)}] Creating camera at {camera_path} (direction: {camera_corner})"
            )

            # Add a view camera scoped to the prim bbox when requested. Single-axis
            # directions are side views; compound directions are corner views.
            is_side_view = camera_corner in _SIDE_VIEW_DIRECTIONS
            if focus_prim:
                add_camera = (
                    add_focused_side_view_camera
                    if is_side_view
                    else add_focused_corner_view_camera
                )
                add_camera(
                    prim_to_focus=focus_prim,
                    camera_path=camera_path,
                    direction=camera_corner,
                    margin=camera_margin,
                    min_distance=0,
                    focal_length=50.0,
                    horizontal_aperture=horizontal_aperture,
                    vertical_aperture=vertical_aperture,
                    near_clip_margin=0.1,
                    far_clip_margin=0.1,
                )
            else:
                add_camera = (
                    add_side_view_camera if is_side_view else add_corner_view_camera
                )
                add_camera(
                    stage,
                    margin=camera_margin,
                    camera_path=camera_path,
                    direction=camera_corner,
                    focal_length=50.0,
                    horizontal_aperture=horizontal_aperture,
                    vertical_aperture=vertical_aperture,
                    near_clip_margin=0.1,
                    far_clip_margin=0.1,
                )

            # Generate output filename with corner suffix if multiple cameras
            # Use the original USD path for naming (before flattening)
            base_name = original_usd_path.stem

            if len(camera_corners) > 1:
                # Use standard direction formatting: "+x+y+z" -> "posx_posy_posz"
                corner_suffix = format_direction_for_filename(camera_corner)
                output_filename = f"{base_name}_{corner_suffix}.png"
            else:
                output_filename = f"{base_name}.png"

            output_image_path = output_base_path / output_filename

            camera_infos.append(
                {
                    "camera_path": camera_path,
                    "camera_corner": camera_corner,
                    "output_path": output_image_path,
                    "index": i,
                }
            )

        # Save the stage with all cameras
        stage.Save()
        render_validation_results: list[dict[str, Any]] = []
        validation_retry_count = 0

        # Define rendering function for parallel execution
        def render_single_camera(camera_info: dict) -> dict:
            """Render a single camera view."""
            camera_path = camera_info["camera_path"]
            output_path = camera_info["output_path"]
            corner = camera_info["camera_corner"]
            index = camera_info["index"]
            camera_validation_results: list[dict[str, Any]] = []

            listener.info(
                f"[{index + 1}/{len(camera_corners)}] Rendering {corner} to {output_path.name}"
            )

            try:
                # Remote render functions can occasionally return HTTP 200
                # with body {"status": "exception"} on single full-scene renders
                # (seen on the final post-apply render step in CI). Retry a
                # couple of times before giving up on the camera.
                max_attempts = int(
                    render_config.get(
                        "max_attempts",
                        3 if backend_type == "remote" else 1,
                    )
                )
                render_result = None
                for attempt in range(max_attempts):
                    stage_for_render = stage
                    if backend_type == "remote":
                        # Reopen the serialized render stage for every REST
                        # attempt. This avoids re-exporting a potentially stale
                        # in-memory stage object after cameras/material edits.
                        stage_for_render = Usd.Stage.Open(str(usd_to_render))
                        if not stage_for_render:
                            return {
                                "success": False,
                                "error": (
                                    "Failed to reopen serialized USD stage for "
                                    f"render attempt: {usd_to_render}"
                                ),
                                "camera_path": camera_path,
                                "corner": corner,
                                "index": index,
                                "validation_results": camera_validation_results,
                            }

                    render_result = rendering_backend.render(
                        stage=stage_for_render,
                        cameras=[camera_path],
                        image_width=image_width,
                        image_height=image_height,
                        cull_style=rendering_config.cull_style,
                        frames="0",  # Single frame render
                        base_dir=render_asset_base_dir,
                    )
                    if (
                        not (
                            render_result
                            and render_result.get("successful_cameras", 0) > 0
                        )
                        and attempt < max_attempts - 1
                    ):
                        attempt_error = "No successful renders returned"
                        if render_result and "results" in render_result:
                            for r in render_result["results"]:
                                if "error" in r:
                                    attempt_error = r["error"]
                                    break
                        listener.warning(
                            f"Render {corner} attempt {attempt + 1}/{max_attempts} failed: {attempt_error}; retrying"
                        )
                        time.sleep(2 * (attempt + 1))
                        continue

                    # Check if rendering was successful
                    if not (
                        render_result
                        and render_result.get("successful_cameras", 0) > 0
                        and "results" in render_result
                        and len(render_result["results"]) > 0
                    ):
                        error_msg = "No successful renders returned"
                        if render_result and "results" in render_result:
                            for result in render_result["results"]:
                                if "error" in result:
                                    error_msg = result["error"]
                                    break
                        if attempt < max_attempts - 1:
                            listener.warning(
                                f"Render {corner} attempt {attempt + 1}/{max_attempts} failed: {error_msg}; retrying"
                            )
                            time.sleep(2 * (attempt + 1))
                            continue
                        return {
                            "success": False,
                            "error": error_msg,
                            "camera_path": camera_path,
                            "corner": corner,
                            "index": index,
                            "validation_results": camera_validation_results,
                        }

                    # Get the first camera result
                    camera_result = render_result["results"][0]

                    # Save the image
                    if not ("images" in camera_result and camera_result["images"]):
                        error_msg = "No image data in result"
                        if attempt < max_attempts - 1:
                            listener.warning(
                                f"Render {corner} attempt {attempt + 1}/{max_attempts} failed: {error_msg}; retrying"
                            )
                            time.sleep(2 * (attempt + 1))
                            continue
                        return {
                            "success": False,
                            "error": error_msg,
                            "camera_path": camera_path,
                            "corner": corner,
                            "index": index,
                            "validation_results": camera_validation_results,
                        }

                    # Get the first image (we only rendered one frame)
                    image = camera_result["images"][0]

                    # Check if it's a PIL Image or raw data
                    if hasattr(image, "save"):
                        # It's a PIL Image, use it directly
                        pass
                    elif isinstance(image, dict) and "image" in image:
                        # For remote REST backends, image data might be in a dict.
                        if isinstance(image["image"], bytes):
                            img_bytes = image["image"]
                        else:
                            # Decode base64 if needed
                            img_bytes = base64.b64decode(image["image"])
                        image = Image.open(BytesIO(img_bytes))
                    else:
                        error_msg = "Unexpected image format"
                        if attempt < max_attempts - 1:
                            listener.warning(
                                f"Render {corner} attempt {attempt + 1}/{max_attempts} failed: {error_msg}; retrying"
                            )
                            time.sleep(2 * (attempt + 1))
                            continue
                        return {
                            "success": False,
                            "error": error_msg,
                            "camera_path": camera_path,
                            "corner": corner,
                            "index": index,
                            "validation_results": camera_validation_results,
                        }

                    # Apply background color if specified
                    if rendering_config.use_background_color:
                        # Convert to RGBA if needed
                        if image.mode != "RGBA":
                            image = image.convert("RGBA")

                        # Apply background color
                        image = paste_on_background(image, background_color)

                    validation = validate_image_artifact(
                        image,
                        backend=backend_type,
                        low_contrast_std_threshold=-1.0,
                        low_contrast_percentile_range_threshold=-1.0,
                    )
                    validation_payload = {
                        "camera_corner": corner,
                        "camera_path": camera_path,
                        "attempt": attempt + 1,
                        "output_path": str(output_path),
                        "validation": validation.to_dict(),
                    }
                    camera_validation_results.append(validation_payload)
                    if _is_blank_render_validation(validation):
                        failed_attempt_path = _failed_render_attempt_path(
                            output_path,
                            attempt=attempt + 1,
                            reason="blank",
                        )
                        image.save(str(failed_attempt_path))
                        _preserve_failed_render_stage(
                            stage_for_render,
                            output_base_path=output_base_path,
                            output_path=output_path,
                            attempt=attempt + 1,
                        )
                        if attempt < max_attempts - 1:
                            listener.warning(
                                f"Render {corner} attempt {attempt + 1}/{max_attempts} returned a blank image; retrying"
                            )
                            time.sleep(2 * (attempt + 1))
                            continue
                        return {
                            "success": False,
                            "error": "Rendered image failed validation: blank image",
                            "camera_path": camera_path,
                            "corner": corner,
                            "index": index,
                            "validation": validation.to_dict(),
                            "failed_attempt_path": str(failed_attempt_path),
                            "attempts": attempt + 1,
                            "validation_results": camera_validation_results,
                        }

                    if not validation.passed:
                        listener.warning(
                            f"Render {corner} produced a low-quality image: "
                            f"{_validation_issue_codes(validation)}"
                        )

                    # Save the final image
                    image.save(str(output_path))

                    listener.info(
                        f"✓ Successfully rendered {corner} to {output_path.name}"
                    )

                    # Emit per-camera rendering event
                    try:
                        listener.event(
                            "rendering.completed",
                            {
                                "camera_corner": corner,
                                "output_path": str(output_path),
                                "image_width": image_width,
                                "image_height": image_height,
                                "backend": backend_type,
                                "attempts": attempt + 1,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to emit rendering event for {corner}: {e}"
                        )

                    return {
                        "success": True,
                        "output_path": str(output_path),
                        "camera_path": camera_path,
                        "corner": corner,
                        "index": index,
                        "attempts": attempt + 1,
                        "validation": validation.to_dict(),
                        "validation_results": camera_validation_results,
                    }

                return {
                    "success": False,
                    "error": "Render attempts exhausted",
                    "camera_path": camera_path,
                    "corner": corner,
                    "index": index,
                    "validation_results": camera_validation_results,
                }

            except Exception as e:
                listener.error(f"Rendering from {corner} failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "camera_path": camera_path,
                    "corner": corner,
                    "index": index,
                    "validation_results": camera_validation_results,
                }

        # Render all cameras in parallel
        listener.info(f"Rendering {len(camera_infos)} view(s) in parallel...")
        rendered_results = []
        failed_renders = []

        # Determine max workers based on number of cameras and backend
        max_workers = _scene_render_max_workers(
            render_config=render_config,
            backend_type=backend_type,
            camera_count=len(camera_infos),
        )
        listener.info(f"Using {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rendering tasks
            future_to_camera = {
                executor.submit(render_single_camera, cam_info): cam_info
                for cam_info in camera_infos
            }

            # Process results as they complete
            for future in as_completed(future_to_camera):
                cam_info = future_to_camera[future]
                try:
                    result = future.result()
                    validation_retry_count += max(0, int(result.get("attempts", 1)) - 1)
                    result_validation = result.get("validation_results", [])
                    if isinstance(result_validation, list):
                        render_validation_results.extend(result_validation)
                    if result["success"]:
                        rendered_results.append(result)
                    else:
                        failed_renders.append(result)
                        listener.error(
                            f"Failed to render {result['corner']}: {result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    listener.error(
                        f"Exception rendering {cam_info['camera_corner']}: {e}"
                    )
                    failed_renders.append(
                        {
                            "success": False,
                            "error": str(e),
                            "corner": cam_info["camera_corner"],
                        }
                    )

        # Remove render cameras from the stage so they don't pollute the
        # output USD served to users (the flat file is also the download).
        for cam_info in camera_infos:
            cam_path = cam_info["camera_path"]
            if stage.GetPrimAtPath(cam_path):
                stage.RemovePrim(cam_path)
        stage.Save()
        listener.info("Cleaned up render camera prims from output USD")

        context["render_validation"] = render_validation_results
        context["rendering_stats"] = {
            "total_images": len(rendered_results),
            "failed_renders": len(failed_renders),
            "image_width": image_width,
            "image_height": image_height,
            "backend": backend_type,
            "validation_retry_count": validation_retry_count,
        }

        # Check if any renders failed
        if failed_renders:
            listener.warning(
                f"{len(failed_renders)}/{len(camera_infos)} renders failed"
            )
            if not allow_partial_renders or len(rendered_results) == 0:
                context["rendering_errors"] = failed_renders
                raise RuntimeError(
                    f"{len(failed_renders)}/{len(camera_infos)} camera renders failed. "
                    f"First error: {failed_renders[0].get('error', 'Unknown')}"
                )
            context["rendering_errors"] = failed_renders

        rendered_results.sort(key=lambda result: result.get("index", 0))
        rendered_image_paths = [result["output_path"] for result in rendered_results]

        # Update context with results
        if rendered_image_paths:
            context["rendered_image_paths"] = rendered_image_paths
            context["rendered_image_path"] = rendered_image_paths[
                0
            ]  # Backward compatibility
            context["rendering_skipped"] = False
            context["rendering_stats"] = {
                "total_images": len(rendered_image_paths),
                "failed_renders": len(failed_renders),
                "image_width": image_width,
                "image_height": image_height,
                "backend": backend_type,
                "validation_retry_count": validation_retry_count,
            }

            listener.info("✓ Rendering complete:")
            listener.info(f"  • Total images rendered: {len(rendered_image_paths)}")
            listener.info(f"  • Failed renders: {len(failed_renders)}")
            listener.info(f"  • Validation retries: {validation_retry_count}")
            listener.info(f"  • Image size: {image_width}x{image_height}")
            for img_path in rendered_image_paths:
                listener.info(f"  • {img_path}")

            # Emit overall rendering completion event
            try:
                listener.event(
                    "rendering.all_completed",
                    {
                        "total_images": len(rendered_image_paths),
                        "failed_renders": len(failed_renders),
                        "image_width": image_width,
                        "image_height": image_height,
                        "backend": backend_type,
                        "rendered_image_paths": rendered_image_paths,
                        "camera_corners": camera_corners,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to emit overall rendering event: {e}")
        else:
            context["rendering_skipped"] = True  # pragma: no cover - defensive fallback

        return context
