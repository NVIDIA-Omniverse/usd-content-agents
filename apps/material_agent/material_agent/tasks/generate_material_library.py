# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate an asset-specific material library package."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.config import get_api_key_for_model_config
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.models.vision_language_models import create_vlm
from world_understanding.utils.credentials import apply_vlm_nim_env_override
from world_understanding.utils.llm_parsing import extract_json_from_llm_response

from material_agent.material_library_generation import (
    MaterialGenerationPlan,
    TextureGenerationSettings,
    build_generated_material_library,
    validate_generated_material_library,
)

logger = logging.getLogger(__name__)


class GenerateMaterialLibraryTask(Task):
    """Generate textures, USD material definitions, and canonical materials.yaml."""

    def __init__(self) -> None:
        self.name = "GenerateMaterialLibrary"
        self.description = "Generate an asset-specific material library"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        output_dir = Path(context.get("output_dir", "generated_material_library"))
        output_dir.mkdir(parents=True, exist_ok=True)

        plan = self._load_or_create_plan(context, listener)
        texture_settings = self._texture_settings_from_context(context)

        listener.info(
            f"Generating material library with {len(plan.materials)} material(s)"
        )
        authoring_config = context.get("material_authoring") or {}
        use_default_prototypes = bool(
            authoring_config.get("use_default_prototypes", True)
        )
        library = build_generated_material_library(
            plan,
            output_dir,
            texture_settings=texture_settings,
            prototype_materials_data=(
                context.get("prototype_materials_data")
                if use_default_prototypes
                else None
            ),
            prototype_materials_path=(
                context.get("prototype_materials_path")
                if use_default_prototypes
                else None
            ),
            prototype_min_score=float(
                authoring_config.get("prototype_min_score", 0.75)
            ),
            material_profile=context.get("material_profile", "auto"),
            write_debug_plan=bool(context.get("write_material_generation_plan", True)),
            include_generation_metadata=bool(
                context.get("include_generation_metadata", True)
            ),
        )

        validation = validate_generated_material_library(
            library.materials_manifest_path
        )
        if not validation.ok:
            raise ValueError(
                "Generated material library failed validation: "
                + "; ".join(validation.errors)
            )

        materials_data = library.materials_data
        context.update(
            {
                "generated_material_library_path": str(library.material_library_path),
                "material_library_path": str(library.material_library_path),
                "generated_materials_yaml_path": str(library.materials_manifest_path),
                "materials_manifest_path": str(library.materials_manifest_path),
                "material_generation_plan_path": (
                    str(library.generation_plan_path)
                    if library.generation_plan_path
                    else context.get("material_generation_plan_path")
                ),
                "generated_material_entries": materials_data["entries"],
                "generated_materials_data": materials_data,
                "materials_data": materials_data,
                "generation_validation": {
                    "ok": validation.ok,
                    "errors": list(validation.errors),
                    "warnings": list(validation.warnings),
                    "metadata": validation.metadata,
                },
            }
        )

        listener.info(
            "Generated materials.yaml with "
            f"{len(materials_data['entries'])} entries: "
            f"{library.materials_manifest_path}"
        )
        return context

    def _load_or_create_plan(
        self, context: dict[str, Any], listener: Any
    ) -> MaterialGenerationPlan:
        plan_path = context.get("material_generation_plan_path")
        if plan_path:
            path = Path(plan_path)
            if not path.exists():
                raise FileNotFoundError(f"Material generation plan not found: {path}")
            with open(path, encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            return MaterialGenerationPlan.from_dict(data, base_dir=path.parent)

        inline_plan = context.get("material_generation_plan")
        if inline_plan:
            base_dir = context.get("material_generation_plan_base_dir")
            return MaterialGenerationPlan.from_dict(inline_plan, base_dir=base_dir)

        return self._create_plan_with_vlm(context, listener)

    def _create_plan_with_vlm(
        self, context: dict[str, Any], listener: Any
    ) -> MaterialGenerationPlan:
        image_caption_pairs = self._collect_image_caption_pairs(context)
        if not image_caption_pairs:
            raise ValueError(
                "generate_material_library requires material_generation_plan_path, "
                "material_generation_plan, or at least one reference/preview image "
                "for VLM material planning"
            )

        vlm = context.get("vlm")
        vlm_config = apply_vlm_nim_env_override(context.get("vlm_config", {}))
        if vlm is None:
            backend = vlm_config.get("backend", "nim")
            model = vlm_config.get("model")
            listener.info(
                "Provisioning VLM for material generation planning: "
                f"{backend}{f' / {model}' if model else ''}"
            )
            model_kwargs: dict[str, Any] = {}
            for key in ("model", "base_url", "timeout"):
                if key in vlm_config:
                    model_kwargs[key] = vlm_config[key]
            api_key = get_api_key_for_model_config(backend, vlm_config, "VLM")
            if api_key:
                model_kwargs["api_key"] = api_key
            vlm = create_vlm(backend, **model_kwargs)

        response_text = vlm.generate_with_image_caption_pairs(
            image_caption_pairs=image_caption_pairs,
            final_prompt=self._build_planning_prompt(context),
            system_prompt=(
                "You are a senior material artist creating a compact PBR material "
                "library for 3D asset material assignment. Return only JSON."
            ),
        )
        plan_data = extract_json_from_llm_response(
            response_text,
            expected_keys=["materials"],
        )
        if isinstance(plan_data, list):
            plan_data = {"materials": plan_data}
        if not isinstance(plan_data, dict):
            raise ValueError("VLM material planner did not return a JSON object")
        return MaterialGenerationPlan.from_dict(plan_data)

    def _collect_image_caption_pairs(
        self, context: dict[str, Any]
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        sources = (
            ("reference_images", "User reference image"),
            ("generated_reference_image_paths", "Generated reference image"),
            ("composition_images", "3D preview composition image"),
            ("rendered_preview_paths", "3D preview image"),
        )
        seen: set[str] = set()
        for key, label in sources:
            for raw_path in context.get(key, [])[:4]:
                path = str(raw_path)
                if path in seen:
                    continue
                seen.add(path)
                pairs.append((f"{label} {len(pairs) + 1}:", path))
        return pairs[:8]

    def _build_planning_prompt(self, context: dict[str, Any]) -> str:
        input_usd_path = context.get("input_usd_path", "")
        identification = context.get("identification") or {}
        material_guidance = context.get("material_guidance") or context.get(
            "planning_guidance"
        )
        parts = [
            "Create a material_generation_plan JSON object for this asset.",
            "Identify the small set of reusable materials needed from the images.",
            "Do not assign materials per prim. Produce general library materials.",
            "Each material must specify color, material substance, and finish.",
            "Create separate materials for every visually distinct color/finish region in the reference images, including interior wells, trays, liners, rims, recesses, controls, caps, inserts, and accents.",
            "For transparent, translucent, frosted, glass, or acrylic materials, include opacity, transmission, ior, and thin_walled PBR hints. Use transmission near 1.0 for glass/acrylic rather than representing it as opaque gray paint.",
            "Do not infer a functional viewing window from object category alone. Only use glass/acrylic when the image clearly shows see-through transparency, visible contents behind the surface, or glass-like refraction.",
            "If a large circular lid insert is opaque gray/silver with strong specular edge highlights, create a metallic lid insert/bezel material. If the lid insert is ambiguous between metal and glass, include both a silver metallic lid insert material and a glass/acrylic material rather than collapsing it into one frosted window.",
            "Name materials broadly enough for prediction to reuse them across similar parts. For example, prefer 'Dark Charcoal Matte Interior Plastic' over a narrow name such as 'Dark Grey Rubber Seal' when the dark appearance covers a whole chamber or tray.",
            "Prefer 2-8 materials. Names must be unique and descriptive.",
            "",
            "Return JSON with this shape:",
            "{",
            '  "version": 1,',
            '  "asset": {"usd_path": "...", "asset_summary": "..."},',
            '  "materials": [',
            "    {",
            '      "id": "stable_lowercase_id",',
            '      "name": "Generated Blue Glossy Plastic",',
            '      "color": "saturated blue",',
            '      "material": "plastic",',
            '      "finish": "glossy molded finish",',
            '      "description": "short manifest and retrieval description",',
            '      "appearance_prompt": "texture generation prompt for a seamless PBR albedo map",',
            '      "base_color_hint": [0.0, 0.1, 0.9],',
            '      "pbr_hints": {"metallic": 0.0, "roughness": 0.35, "opacity": 1.0, "transmission": 0.0, "ior": 1.5, "thin_walled": false},',
            '      "intended_parts": [',
            '        {"semantic_label": "rails", "evidence": "why this material is needed"}',
            "      ]",
            "    }",
            "  ]",
            "}",
        ]
        if input_usd_path:
            parts.append(f"\nUSD path: {input_usd_path}")
        if identification:
            parts.append("\nAsset identification:")
            parts.append(json.dumps(identification, indent=2, sort_keys=True))
        if material_guidance:
            parts.append("\nUser/team material guidance:")
            if isinstance(material_guidance, list):
                parts.extend(f"- {item}" for item in material_guidance)
            else:
                parts.append(str(material_guidance))
        return "\n".join(parts)

    def _texture_settings_from_context(
        self, context: dict[str, Any]
    ) -> TextureGenerationSettings:
        data = context.get("texture_generation") or {}
        return TextureGenerationSettings(
            texture_size=int(data.get("texture_size", 1024)),
            backend=data.get("backend"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            api_key=data.get("api_key"),
            api_key_env_var=data.get("api_key_env_var"),
            seed=data.get("seed"),
            color_correct_albedo=bool(data.get("color_correct_albedo", True)),
            albedo_color_correction_strength=float(
                data.get("albedo_color_correction_strength", 1.0)
            ),
        )
