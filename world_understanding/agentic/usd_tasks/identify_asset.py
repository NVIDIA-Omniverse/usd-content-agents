# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify asset type and description from preview images and USD metadata.

Runs a single VLM call on preview/composition images to identify the whole
asset (type, subtype, description, expected colors) before per-component
classification. The identification can be used to auto-generate reference
image prompts without manual per-asset descriptions.

Shared across all agents (physics-agent, joint-agent, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.config import get_api_key_for_model_config
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.credentials import apply_vlm_nim_env_override
from world_understanding.utils.llm_parsing import extract_json_from_llm_response
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class IdentifyAssetTask(Task):
    """Run VLM inference on preview images to identify the asset.

    Uses preview renders (from ``render_preview``) and optionally the USD
    filename and prim names to identify what the 3D object is and describe
    its expected real-world appearance.

    Input context keys:
        - composition_images or rendered_preview_paths: list[str] — images
        - vlm: VLM instance (from model provisioning)
        - vlm_invoke_kwargs: dict — optional VLM invoke kwargs
        - identify_system_prompt: str — optional system prompt override
        - reference_images: list[str] — optional scene reference images
        - usd_path or input_usd_path: str — USD file path (for filename hint)
        - output_dir: str — directory to save identification.json

    Output context keys:
        - identification: dict with asset_type, asset_subtype,
          asset_description, expected_colors, confidence, reasoning
        - identification_path: str — path to identification.json
        - image_gen_prompt: str — auto-generated prompt for reference
          image generation (consumed by ``GenerateReferenceImageTask``)
    """

    def __init__(self) -> None:
        self.name = "IdentifyAsset"
        self.description = "Identify asset from preview images and USD metadata"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        vlm = context.get("vlm")
        if isinstance(vlm, dict):
            context.setdefault("vlm_config", vlm)
            vlm = None
        elif vlm is not None and not hasattr(vlm, "generate_with_image_caption_pairs"):
            raise TypeError(
                "vlm must be a provisioned vision-language model or a VLM config dict"
            )
        images = context.get("composition_images") or context.get(
            "rendered_preview_paths", []
        )
        system_prompt = context.get("identify_system_prompt", "")
        output_dir = Path(context.get("output_dir", "."))
        reference_images: list[str] = context.get("reference_images", [])

        # Self-contained: provision VLM if not already in context
        if vlm is None:
            vlm_config = apply_vlm_nim_env_override(context.get("vlm_config", {}))
            backend = vlm_config.get("backend", "nim")
            model = vlm_config.get("model")
            listener.info(
                f"Provisioning VLM for identification: {backend}"
                + (f" / {model}" if model else "")
            )
            from world_understanding.functions.models.vision_language_models import (
                create_vlm,
            )

            model_kwargs: dict[str, Any] = {}
            for key in ("model", "base_url", "timeout"):
                if key in vlm_config:
                    model_kwargs[key] = vlm_config[key]
            if model and "model" not in model_kwargs:
                model_kwargs["model"] = model

            api_key = get_api_key_for_model_config(backend, vlm_config, "VLM")
            if api_key:
                model_kwargs["api_key"] = api_key

            vlm = create_vlm(backend, **model_kwargs)
            listener.info(f"VLM provisioned: {vlm.model_name}")

        if not images:
            logger.warning("No preview images available for identification")
            identification = self._fallback_identification(
                "No preview images available"
            )
            self._save_and_update_context(context, identification, output_dir)
            return context

        listener.info(f"Identifying asset from {len(images)} preview images")

        # Gather hints from USD metadata
        usd_path = context.get("usd_path") or context.get("input_usd_path", "")
        usd_filename = Path(usd_path).stem if usd_path else ""
        prim_names = self._extract_prim_name_hints(usd_path)

        # Build the user prompt with chain-of-thought
        user_prompt = self._build_user_prompt(
            usd_filename, prim_names, has_reference=bool(reference_images)
        )

        # Build image content for VLM
        images_to_use = images[:4]

        try:
            # Build image-caption pairs for VLM
            image_caption_pairs: list[tuple[str, str]] = []

            # Add scene reference images first (if available)
            for idx, ref_path in enumerate(reference_images[:2]):
                image_caption_pairs.append(
                    (f"Scene reference image {idx + 1}:", ref_path)
                )

            # Add preview images
            for idx, img_path in enumerate(images_to_use):
                image_caption_pairs.append(
                    (f"3D preview of the asset (view {idx + 1}):", img_path)
                )

            response_text = vlm.generate_with_image_caption_pairs(
                image_caption_pairs=image_caption_pairs,
                final_prompt=user_prompt,
                system_prompt=system_prompt
                or "You are an expert at identifying 3D objects.",
            )

            identification = self._parse_identification(response_text)

            listener.info(
                f"Identified: {identification.get('asset_type', '?')} / "
                f"{identification.get('asset_subtype', '?')} "
                f"(confidence: {identification.get('confidence', '?')})"
            )
            listener.info(
                f"Description: {identification.get('asset_description', '')[:200]}"
            )

        except Exception as e:
            logger.error("Asset identification failed: %s", e, exc_info=True)
            identification = self._fallback_identification(str(e))

        self._save_and_update_context(context, identification, output_dir)

        # Auto-generate image_gen_prompt for GenerateReferenceImageTask
        prompt = self._build_image_gen_prompt(identification)
        context["image_gen_prompt"] = prompt
        listener.info(f"Auto-generated reference image prompt: {prompt[:200]}...")

        return context

    def _build_user_prompt(
        self, usd_filename: str, prim_names: list[str], has_reference: bool
    ) -> str:
        """Build a chain-of-thought identification prompt."""
        parts = [
            "Analyze the 3D preview images and identify this object.\n",
        ]

        if usd_filename:
            parts.append(f"HINT: The USD file is named '{usd_filename}'.\n")

        if prim_names:
            names_str = ", ".join(prim_names[:20])
            if len(prim_names) > 20:
                names_str += f" ... ({len(prim_names)} total)"
            parts.append(f"HINT: Internal part names include: {names_str}\n")

        if has_reference:
            parts.append(
                "A scene reference image is also provided — the object may "
                "appear somewhere in that scene. Use it to determine the "
                "correct real-world colors.\n"
            )

        parts.append(
            "\nRespond with JSON:\n"
            "{\n"
            '  "asset_type": "category (e.g., vehicle, robot, tool, '
            'furniture, industrial_equipment)",\n'
            '  "asset_subtype": "specific type (e.g., AGV, humanoid, '
            'forklift, drill)",\n'
            '  "asset_description": "brief description of the object",\n'
            '  "expected_colors": "describe the real-world color scheme '
            '(e.g., white body, black bumpers, orange branding)",\n'
            '  "confidence": "high/medium/low",\n'
            '  "reasoning": "how you identified it"\n'
            "}"
        )

        return "\n".join(parts)

    def _extract_prim_name_hints(self, usd_path: str) -> list[str]:
        """Extract readable prim name hints from the USD file."""
        if not usd_path or not Path(usd_path).exists():
            return []

        try:
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(str(usd_path))
            names: list[str] = []
            seen: set[str] = set()
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    # Get the parent Xform name (more descriptive than mesh)
                    parent = prim.GetParent()
                    name = parent.GetName() if parent else prim.GetName()
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
            return names
        except Exception as e:
            logger.debug("Could not extract prim names from %s: %s", usd_path, e)
            return []

    def _parse_identification(self, response_text: str) -> dict[str, Any]:
        """Parse VLM response into identification dict."""
        text = response_text.strip()
        result = extract_json_from_llm_response(
            text,
            expected_keys=["asset_type", "asset_subtype"],
        )
        if isinstance(result, dict):
            result.setdefault("asset_type", "unknown")
            result.setdefault("asset_subtype", "unknown")
            result.setdefault("asset_description", "")
            result.setdefault("expected_colors", "")
            result.setdefault("confidence", "medium")
            result.setdefault("reasoning", "")
            return result

        logger.warning("Could not parse identification JSON from VLM response")
        return self._fallback_identification(f"Could not parse response: {text[:200]}")

    def _fallback_identification(self, reason: str) -> dict[str, Any]:
        """Return a fallback identification when VLM fails."""
        return {
            "asset_type": "unknown",
            "asset_subtype": "unknown",
            "asset_description": reason,
            "expected_colors": "",
            "confidence": "low",
            "reasoning": reason,
        }

    def _build_image_gen_prompt(self, identification: dict[str, Any]) -> str:
        """Build a reference image generation prompt from identification."""
        asset_type = identification.get("asset_type", "object")
        asset_subtype = identification.get("asset_subtype", "")
        description = identification.get("asset_description", "")
        colors = identification.get("expected_colors", "")

        parts = ["Generate a photorealistic product photograph of"]

        if asset_subtype and asset_subtype != "unknown":
            parts.append(f"a {asset_subtype}")
        elif asset_type and asset_type != "unknown":
            parts.append(f"a {asset_type}")
        else:
            parts.append("the object shown in the 3D preview")

        if description:
            parts.append(f"({description})")

        parts.append(".")

        if colors:
            parts.append(f"Colors: {colors}.")

        parts.append(
            "Keep the exact geometry and proportions from the 3D preview. "
            "Use professional product photography lighting with a clean "
            "neutral background."
        )

        return " ".join(parts)

    def _save_and_update_context(
        self,
        context: dict[str, Any],
        identification: dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Save identification to JSON and update context."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "identification.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(identification, f, indent=2, ensure_ascii=False)
        logger.info("Saved identification to %s", output_path)

        context["identification"] = identification
        context["identification_path"] = str(output_path)
