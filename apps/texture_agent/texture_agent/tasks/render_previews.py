# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Render material previews on a sphere."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom, UsdShade
from world_understanding.agentic.tasks import Task

from texture_agent.config.rendering_backends import (
    DEFAULT_TEXTURE_RENDERING_BACKEND,
    validate_texture_rendering_backend,
)
from texture_agent.functions.material_discovery import MaterialInfo
from texture_agent.tasks.render_results import render_result_items

logger = logging.getLogger(__name__)

# Default thumbnail template from the material agent
_DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "material_agent"
    / "data"
    / "templates"
    / "thumbnail_template.usd"
)

_TEMPLATE_CAMERA = "/Root/thumbnail_CAM"
_TEMPLATE_SPHERE = "/Root/Sphere"


def _render_result_items(results: Any) -> list[dict[str, Any]]:
    """Normalize renderer results using the preview diagnostic contract."""
    return render_result_items(results, producer="Rendering backend")


class RenderMaterialPreviewsTask(Task):
    """Render each material on a sphere for visual reference.

    Composes a stage using the thumbnail template (sphere + camera + lights),
    references each material from the input USD, and renders through the shared
    rendering-backend contract.

    Context keys read:
        discovered_materials (list[MaterialInfo]): From DiscoverMaterialsTask.
        usd_path (str): Input USD containing the materials.
        render_preview_config (dict): backend, image_width, image_height.
        working_dir (str): Working directory.

    Context keys written:
        material_previews (dict[str, str]): Material name -> preview image path.
    """

    def __init__(self) -> None:
        self.name = "RenderMaterialPreviews"
        self.description = "Render material previews on a sphere"

    def _compose_preview_stage(
        self,
        template_path: Path,
        usd_path: str,
        material: MaterialInfo,
    ) -> Usd.Stage:
        """Compose a preview stage with the material on the template sphere."""
        stage = Usd.Stage.CreateInMemory()
        root = stage.GetRootLayer()
        root.subLayerPaths.append(str(template_path))

        # Create Looks scope and reference the material
        UsdGeom.Scope.Define(stage, "/Root/Looks")
        mat_dest = f"/Root/Looks/{material.name}"
        mat_prim = stage.OverridePrim(mat_dest)
        mat_prim.GetReferences().AddReference(str(usd_path), material.prim_path)

        # Bind to sphere
        sphere = stage.GetPrimAtPath(_TEMPLATE_SPHERE)
        if sphere.IsValid():
            binding_api = UsdShade.MaterialBindingAPI.Apply(sphere)
            mat = UsdShade.Material(stage.GetPrimAtPath(mat_dest))
            binding_api.Bind(mat)

        # Flatten so every backend receives resolved composition arcs.
        flat_layer = stage.Flatten()
        flat_stage = Usd.Stage.Open(flat_layer)

        # Convert custom MDL to built-in for renderer compatibility.
        try:
            from world_understanding.utils.usd.material import (
                convert_custom_mdl_to_builtin,
            )

            convert_custom_mdl_to_builtin(flat_stage)
        except ImportError:
            pass

        return flat_stage

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        scoped_materials = context.get("texture_plan_scoped_materials")
        materials: list[MaterialInfo]
        if scoped_materials is None:
            materials = context["discovered_materials"]
        else:
            materials = scoped_materials
        config: dict = context.get("render_preview_config", {})
        backend_type = validate_texture_rendering_backend(
            config.get("backend", DEFAULT_TEXTURE_RENDERING_BACKEND),
            step_name="render_previews",
        )

        if not materials:
            logger.info("No scoped materials to preview")
            context["material_previews"] = {}
            return context

        usd_path: str = context["usd_path"]
        working_dir = Path(context["working_dir"])

        template = config.get("template_scene")
        if template:
            template_path = Path(template)
        else:
            template_path = _DEFAULT_TEMPLATE

        if not template_path.exists():
            logger.warning(
                "Thumbnail template not found: %s. Skipping previews.",
                template_path,
            )
            context["material_previews"] = {}
            return context

        image_width = config.get("image_width", 512)
        image_height = config.get("image_height", image_width)

        from world_understanding.functions.graphics.rendering_backend_factory import (
            create_rendering_backend,
        )

        rendering_backend = create_rendering_backend(backend_type, config)
        logger.info("Using %s rendering backend for material previews", backend_type)

        out_dir = working_dir / "previews"
        out_dir.mkdir(parents=True, exist_ok=True)

        previews: dict[str, str] = {}

        for mat in materials:
            logger.info("Rendering preview for %s", mat.name)
            try:
                flat_stage = self._compose_preview_stage(template_path, usd_path, mat)

                render_result = rendering_backend.render(
                    stage=flat_stage,
                    image_width=image_width,
                    image_height=image_height,
                    cameras=[_TEMPLATE_CAMERA],
                    base_dir=Path(usd_path).parent,
                    render_slot_timeout_sec=config.get("render_slot_timeout_sec"),
                )
                results = _render_result_items(render_result)

                if (
                    results
                    and results[0].get("images")
                    and str(results[0].get("status", "success")) == "success"
                ):
                    out_path = out_dir / f"{mat.name}_preview.png"
                    results[0]["images"][0].save(str(out_path))
                    previews[mat.name] = str(out_path)
                    logger.info("  Saved preview: %s", out_path)
                else:
                    logger.warning("  No image returned for %s", mat.name)

            except Exception:
                logger.exception("  Failed to render preview for %s", mat.name)

        context["material_previews"] = previews
        logger.info("Rendered %d material previews", len(previews))
        return context
