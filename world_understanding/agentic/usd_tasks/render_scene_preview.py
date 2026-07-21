# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render lightweight whole-scene preview images from a USD file.

This is a self-contained, reusable task that:
- Loads a fresh USD stage from ``usd_path``
- Creates cameras focused on the whole scene bounding box
- Provisions its own rendering backend from ``render_config``
- Saves images to ``output_dir/preview/``
- Outputs ``rendered_preview_paths`` in context

Both the material agent (``render_preview`` pipeline step) and the asset agent
(``identify_asset`` workflow) share this task so rendering logic stays in one place.
"""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from pxr import Usd

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    add_corner_view_camera,
    add_side_view_camera,
    format_direction_for_filename,
)
from world_understanding.functions.graphics.rendering_backend_factory import (
    create_rendering_backend,
    validate_rendering_backend_name,
)
from world_understanding.utils.image_utils import paste_on_background
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.usd.prim import (
    get_bbox_from_prim,
    nullify_materials,
    remove_all_lights,
)
from world_understanding.utils.usd.stage import load_stage, prepare_stage_for_render

logger = logging.getLogger(__name__)


class RenderScenePreviewTask(Task):
    """Render whole-scene preview images from a USD file.

    This task is **self-contained**: it provisions its own rendering backend
    and loads a fresh USD stage so that it can be dropped into any workflow
    without requiring ``USDRendererProvisioningTask`` or ``USDLoadingTask``.

    Input context keys:
        - usd_path: Path to the USD file
        - render_config: Dictionary with rendering settings:
            - backend: ``"remote"`` (default), ``"warp"``, ``"ovrtx"``, or
              ``"mock"``
            - image_width: int (default 512)
            - image_height: int (default image_width)
            - cameras: list of direction strings (default ``["+x+y+z"]``)
            - camera_margin: float (default 3.0)
            - background_color: [R, G, B] 0.0-1.0 (default ``[0.0, 0.0, 0.0]``)
            - should_reset_materials: bool (default True)
            - use_lights: bool (default False)
            - flatten_before_render: bool (default False)
        - prim_filters: Optional dict with prim filtering (same schema as
          ``build_dataset_usd``). When provided, prims that do **not** match
          the filter are deactivated so only relevant geometry appears in
          the preview. Supported keys:
            - types: list of type strings (default ``["UsdGeom.Mesh"]``)
            - skip_instances: bool (default True)
            - skip_prototypes: bool (default False)
            - exclude_paths: list of prim path prefixes to hide
        - output_dir: Directory to write preview images into

    Output context keys:
        - rendered_preview_paths: List of rendered image paths
        - composition_images: Alias of ``rendered_preview_paths`` (for IdentifyAssetTask compat)
    """

    def __init__(self) -> None:
        self.name = "RenderScenePreview"
        self.description = "Render whole-scene preview images"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Render preview images of the whole scene.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with rendered_preview_paths and composition_images
        """
        listener = get_listener(context, logger_name=__name__)

        # ── Resolve inputs ────────────────────────────────────────────
        usd_path = context.get("usd_path")
        if not usd_path:
            raise ValueError("usd_path not found in context")

        usd_path = Path(usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(f"USD file not found: {usd_path}")

        render_config: dict[str, Any] = context.get("render_config", {})
        output_dir = Path(context.get("output_dir", "."))

        backend_type = validate_rendering_backend_name(
            render_config.get("backend", "remote")
        )
        image_width = render_config.get("image_width", 512)
        image_height = render_config.get("image_height", image_width)
        camera_directions: list[str] = render_config.get("cameras", ["+x+y+z"])
        camera_margin = render_config.get("camera_margin", 3.0)
        bg_color = render_config.get("background_color", [0.0, 0.0, 0.0])
        should_reset_materials = render_config.get("should_reset_materials", True)
        use_lights = render_config.get("use_lights", False)
        flatten_before_render = render_config.get("flatten_before_render", False)

        # Normalise background_color to 0-255 ints
        # Accepts 0.0-1.0 floats (like the render step) for consistency
        background_color = tuple(min(255, max(0, int(c * 255))) for c in bg_color)

        preview_dir = output_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        listener.info(
            f"Rendering {len(camera_directions)} preview view(s) at "
            f"{image_width}x{image_height} using {backend_type}"
        )

        # ── Load a fresh stage ────────────────────────────────────────
        listener.info(f"Loading USD stage for preview: {usd_path}")
        preview_stage = load_stage(str(usd_path))

        # Optionally flatten
        if flatten_before_render:
            listener.info("Flattening stage before preview render")
            preview_stage, preparation_metadata = prepare_stage_for_render(
                preview_stage,
                flatten=True,
                normalize_materials=False,
            )
            render_asset_base_dir = preparation_metadata.get("asset_base_dir")
            listener.info(f"Preview stage preparation: {preparation_metadata}")
        else:
            render_asset_base_dir = str(usd_path.parent)

        # ── Scene bounding box ────────────────────────────────────────
        scene_bbox = get_bbox_from_prim(preview_stage.GetPseudoRoot())
        aligned_range = scene_bbox.ComputeAlignedRange()
        bbox_min = aligned_range.GetMin()
        bbox_max = aligned_range.GetMax()
        max_stage_size = max(
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2],
        )

        # ── Create cameras ────────────────────────────────────────────
        camera_root = "/PreviewCameras"
        camera_paths: list[str] = []

        for direction in camera_directions:
            dir_suffix = format_direction_for_filename(direction)
            camera_path = f"{camera_root}/Preview_{dir_suffix}"

            num_axes = sum(1 for c in direction if c in "xyz")
            cam_fn = add_corner_view_camera if num_axes >= 2 else add_side_view_camera
            cam_fn(
                preview_stage,
                margin=camera_margin,
                camera_path=camera_path,
                direction=direction,
                focal_length=50.0,
                horizontal_aperture=36.0,
                vertical_aperture=36.0,
                near_clip_margin=0.1,
                far_clip_margin=0.1,
                max_scene_size=max_stage_size,
                time=Usd.TimeCode.Default(),
            )
            camera_paths.append(camera_path)

        # ── Flatten for remote renderer ───────────────────────────────
        # The remote backend exports the stage via GetRootLayer().Export()
        # which only writes the root layer.  For stages with payloads,
        # references, or USDZ archives the geometry lives in sub-layers
        # and would be lost.  Flatten now (after cameras are added) so
        # that a single root layer contains everything the renderer needs.
        if backend_type == "remote" and not flatten_before_render:
            listener.info(
                "Flattening stage for remote renderer (resolving payloads/references)"
            )
            preview_stage, preparation_metadata = prepare_stage_for_render(
                preview_stage,
                flatten=True,
                normalize_materials=False,
            )
            render_asset_base_dir = preparation_metadata.get(
                "asset_base_dir",
                render_asset_base_dir,
            )
            listener.info(f"Preview stage preparation: {preparation_metadata}")

        # Strip materials / lights if requested
        # NOTE: This must happen AFTER the remote-renderer flatten, because
        # SetActive(False) writes to the session layer which is
        # discarded by Flatten().
        if should_reset_materials:
            listener.info("Resetting materials (nullifying all material bindings)")
            nullify_materials(preview_stage)

        if not use_lights:
            listener.info("Removing scene lights")
            remove_all_lights(preview_stage)

        # ── Apply prim filters (deactivate non-matching prims) ────────
        prim_filters: dict[str, Any] = context.get("prim_filters", {})
        if prim_filters:
            deactivated = self._apply_prim_filters(
                preview_stage, prim_filters, listener
            )
            if deactivated > 0:
                listener.info(f"Deactivated {deactivated} prims not matching filters")

        # ── Provision rendering backend ───────────────────────────────
        rendering_backend = create_rendering_backend(backend_type, render_config)

        # ── Render ────────────────────────────────────────────────────
        rendered_paths: list[str] = []

        def _render_camera(idx: int, cam_path: str, direction: str) -> str | None:
            """Render a single camera and return saved path or None."""
            try:
                result = rendering_backend.render(
                    preview_stage,
                    cameras=[cam_path],
                    image_width=image_width,
                    image_height=image_height,
                    cull_style="back",
                    frames="0",
                    base_dir=render_asset_base_dir,
                )

                for cam_result in result.get("results", []):
                    images = cam_result.get("images", [])
                    for frame_idx, image in enumerate(images):
                        if image is None:
                            continue

                        image = self._to_pil(image)

                        # Apply background colour
                        if image.mode != "RGBA":
                            image = image.convert("RGBA")
                        image = paste_on_background(image, background_color)

                        dir_suffix = format_direction_for_filename(direction)
                        filename = f"preview_{dir_suffix}_frame{frame_idx}.png"
                        save_path = preview_dir / filename
                        image.save(str(save_path))
                        listener.info(f"Saved preview: {save_path.name}")
                        return str(save_path)
            except Exception as e:
                listener.error(f"Preview render failed for {direction}: {e}")
            return None

        # Local GPU backends (warp, ovrtx) share in-process USD state and/or
        # CUDA contexts that are not thread-safe — concurrent rendering from
        # multiple threads causes segfaults or USD clip-cache assertions.
        # Only the remote HTTP backend is safe to parallelise.
        if backend_type == "remote":
            max_workers = min(len(camera_paths), 4)
        else:
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_render_camera, i, cp, d): d
                for i, (cp, d) in enumerate(
                    zip(camera_paths, camera_directions, strict=True)
                )
            }
            for future in as_completed(futures):
                path = future.result()
                if path:
                    rendered_paths.append(path)

        listener.info(f"Rendered {len(rendered_paths)} preview images to {preview_dir}")

        # ── Update context ────────────────────────────────────────────
        context["rendered_preview_paths"] = rendered_paths
        # Alias for IdentifyAssetTask compatibility
        context["composition_images"] = rendered_paths

        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_prim_filters(
        stage: "Usd.Stage",
        filters: dict[str, Any],
        listener: Any,
    ) -> int:
        """Deactivate prims that do not match the filter criteria.

        This makes the preview show only geometry the pipeline will process,
        matching the same ``prim_filters`` schema used by ``build_dataset_usd``.

        Args:
            stage: USD stage (will be mutated)
            filters: Same schema as prim_filters in build_dataset_usd
            listener: Event listener for logging

        Returns:
            Number of prims deactivated
        """
        prim_types_str = filters.get("types", ["UsdGeom.Mesh"])
        skip_instances = filters.get("skip_instances", True)
        skip_prototypes = filters.get("skip_prototypes", False)
        exclude_paths = filters.get("exclude_paths", [])

        # Resolve type strings to USD schema classes
        prim_types = []
        for type_str in prim_types_str:
            # Support "UsdGeom.Mesh" style strings
            parts = type_str.split(".")
            if len(parts) == 2:
                module_name, class_name = parts
                try:
                    from pxr import UsdGeom

                    module = {"UsdGeom": UsdGeom}.get(module_name)
                    if module:
                        prim_types.append(getattr(module, class_name))
                except (AttributeError, ImportError):
                    listener.warning(f"Unknown prim type: {type_str}")

        if not prim_types:
            return 0

        # Collect geometry prims that match the filter
        matching_paths: set[str] = set()

        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            prim_path = str(prim.GetPath())

            # Skip excluded paths
            if any(prim_path.startswith(ex) for ex in exclude_paths):
                continue

            # Skip instances
            if skip_instances and (prim.IsInstance() or prim.IsInstanceProxy()):
                continue

            # Skip prototypes
            if skip_prototypes and prim.IsInPrototype():
                continue

            # Check type match
            for pt in prim_types:
                if prim.IsA(pt):
                    matching_paths.add(prim_path)
                    break

        if not matching_paths:
            listener.warning("No prims matched prim_filters — skipping deactivation")
            return 0

        # Build set of ancestor paths that must stay active
        ancestors: set[str] = set()
        for path in matching_paths:
            parts = path.split("/")
            for i in range(1, len(parts)):
                ancestors.add("/".join(parts[:i]))

        # Deactivate imageable (geometry) prims that are NOT in the matching set
        # and NOT ancestors of matching prims.  Only touch prims that are
        # Imageable (geometry / lights / cameras) to avoid breaking non-visual
        # structure.
        deactivated = 0
        for prim in stage.TraverseAll():
            prim_path = str(prim.GetPath())
            if prim_path in matching_paths or prim_path in ancestors:
                continue
            if not prim.IsA(UsdGeom.Imageable):
                continue
            # Don't deactivate cameras we created
            if prim_path.startswith("/PreviewCameras"):
                continue
            if prim.IsActive():
                prim.SetActive(False)
                deactivated += 1

        return deactivated

    @staticmethod
    def _to_pil(image: Any) -> Image.Image:
        """Normalise various image formats to PIL Image."""
        if hasattr(image, "save"):
            return image  # type: ignore[return-value]
        if isinstance(image, dict) and "image" in image:
            raw = image["image"]
            if isinstance(raw, bytes):
                return Image.open(BytesIO(raw))
            try:
                return Image.open(BytesIO(base64.b64decode(raw)))
            except Exception as decode_err:
                raise ValueError(
                    f"Failed to decode image data: {decode_err}"
                ) from decode_err
        raise ValueError(f"Unexpected image format: {type(image)}")
