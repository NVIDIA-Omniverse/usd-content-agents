# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-based texture prompt generation for discovered materials.

Given a list of MaterialInfo and a user-level aesthetic direction,
generates per-material texture prompts using a chat LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from texture_agent.functions.material_discovery import MaterialInfo

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert at describing PBR (Physically Based Rendering) material textures \
for 3D assets. Given a list of material names and their physical properties, you \
generate detailed texture descriptions suitable for an AI image generation model \
that produces flat, tileable PBR albedo texture maps.

Rules:
- Each description should focus on SURFACE APPEARANCE: color variation, weathering \
patterns, scratches, stains, patina, wear patterns, etc.
- Descriptions must be compatible with flat texture map generation -- do NOT mention \
3D geometry, lighting, or camera angles.
- Use the material name to infer the base material type (e.g., "Aluminum_Brushed" \
is brushed aluminum, "Plastic_Dark_Blue" is dark blue plastic).
- Use the base_color RGB values to match the expected color palette.
- Use metalness and roughness to inform surface character (metalness=1.0 means \
metal, roughness=0.1 means very glossy, roughness=0.8 means matte/rough).
- When a user_prompt is provided (e.g., "old and weathered"), incorporate that \
aesthetic direction into ALL descriptions consistently.
- Suggest an opacity value between 0.5 and 1.0 indicating how strongly the \
texture should override the base color (0.7 = subtle, 0.9 = heavy).

Return ONLY a JSON object with this exact structure:
{
  "materials": {
    "<material_name>": {
      "prompt": "<texture description>",
      "opacity": <float between 0.5 and 1.0>
    }
  }
}\
"""

_USER_PROMPT_TEMPLATE = """\
Generate texture prompts for the following materials.
{user_direction}

Materials:
{materials_list}\
"""

_MATERIAL_ENTRY_TEMPLATE = """\
- Name: {name}
  Base Color (RGB linear): ({r:.3f}, {g:.3f}, {b:.3f})
  Metalness: {metalness}
  Roughness: {roughness}\
"""


def _format_materials_list(materials: list[MaterialInfo]) -> str:
    """Format materials into a readable list for the LLM prompt."""
    entries = []
    for mat in materials:
        entries.append(
            _MATERIAL_ENTRY_TEMPLATE.format(
                name=mat.name,
                r=mat.base_color[0],
                g=mat.base_color[1],
                b=mat.base_color[2],
                metalness=(
                    mat.base_metalness if mat.base_metalness is not None else "unknown"
                ),
                roughness=(
                    mat.specular_roughness
                    if mat.specular_roughness is not None
                    else "unknown"
                ),
            )
        )
    return "\n".join(entries)


def _fallback_prompt_for_material(mat: MaterialInfo, user_prompt: str = "") -> str:
    """Generate a basic prompt from material name when LLM fails.

    When the user supplied an aesthetic direction, lead with it and apply it
    to the material by name — e.g. ``"rusted steampunk, applied to aluminum
    matte"``. Without a user prompt, fall back to a generic description so
    the image-gen model still has something to work with.
    """
    readable = mat.name.replace("_", " ").lower()
    if user_prompt:
        return f"{user_prompt}, applied to {readable}"
    return f"realistic {readable} surface texture"


def _fallback_prompts(
    materials: list[MaterialInfo],
    user_prompt: str,
    default_opacity: float,
) -> dict[str, dict[str, Any]]:
    """Generate fallback prompts when LLM call fails entirely."""
    logger.warning("Using fallback prompt generation (LLM failed)")
    return {
        mat.name: {
            "prompt": _fallback_prompt_for_material(mat, user_prompt),
            "opacity": default_opacity,
        }
        for mat in materials
    }


def generate_texture_prompts(
    materials: list[MaterialInfo],
    llm: Any,
    user_prompt: str = "",
    default_opacity: float = 0.80,
    fallback_material_names: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate texture prompts for materials using an LLM.

    Args:
        materials: Discovered materials needing prompts.
        llm: A LangChain chat model instance.
        user_prompt: User-level aesthetic direction
            (e.g., "old and weathered", "brand new factory condition").
        default_opacity: Fallback opacity if LLM doesn't provide one.

    Returns:
        Dict mapping material name -> {"prompt": str, "opacity": float}.
    """
    if not materials:
        return {}

    from langchain_core.messages import HumanMessage, SystemMessage
    from world_understanding.utils.llm_parsing import (
        extract_json_from_llm_response,
    )

    user_direction = (
        f"Overall aesthetic direction: {user_prompt}"
        if user_prompt
        else "No specific aesthetic direction -- generate neutral, realistic textures."
    )

    materials_list = _format_materials_list(materials)
    user_message = _USER_PROMPT_TEMPLATE.format(
        user_direction=user_direction,
        materials_list=materials_list,
    )

    logger.info(
        "Generating prompts for %d materials (user_prompt=%r)",
        len(materials),
        user_prompt[:80] if user_prompt else "(none)",
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        if fallback_material_names is not None:
            fallback_material_names.update(mat.name for mat in materials)
        return _fallback_prompts(materials, user_prompt, default_opacity)

    result = extract_json_from_llm_response(
        response.content, expected_keys=["materials"]
    )
    if not result or "materials" not in result:
        logger.error(
            "Failed to parse LLM response for prompt generation. Response: %s",
            response.content[:500],
        )
        if fallback_material_names is not None:
            fallback_material_names.update(mat.name for mat in materials)
        return _fallback_prompts(materials, user_prompt, default_opacity)

    generated = result["materials"]

    # Validate and normalize
    prompts: dict[str, dict[str, Any]] = {}
    for mat in materials:
        entry = generated.get(mat.name, {})
        prompt = entry.get("prompt", "")
        if not prompt:
            logger.warning(
                "LLM response omitted a prompt for material %r; using fallback prompt",
                mat.name,
            )
            if fallback_material_names is not None:
                fallback_material_names.add(mat.name)
            prompt = _fallback_prompt_for_material(mat, user_prompt)
        opacity = entry.get("opacity", default_opacity)
        opacity = max(0.3, min(1.0, float(opacity)))
        prompts[mat.name] = {"prompt": prompt, "opacity": opacity}

    logger.info("Generated prompts for %d materials", len(prompts))
    return prompts
