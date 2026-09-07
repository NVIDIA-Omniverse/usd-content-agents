# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private adapter from refinement source state to legacy library authoring."""

from __future__ import annotations

from material_agent.material_library_generation.schema import MaterialRecipe, PBRHints

from .contracts import MaterialRefinementSource


def source_material_recipe(source: MaterialRefinementSource) -> MaterialRecipe:
    """Build the internal recipe required to author source evidence."""

    return candidate_material_recipe(
        source,
        appearance_prompt=(
            source.description
            or f"Existing {source.name} material supplied for refinement."
        ),
        base_color=source.base_color,
        roughness=source.roughness,
        metallic=source.metallic,
    )


def candidate_material_recipe(
    source: MaterialRefinementSource,
    *,
    appearance_prompt: str,
    base_color: tuple[float, float, float],
    roughness: float,
    metallic: float,
    material_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> MaterialRecipe:
    """Adapt source identity and PBR state for existing material-library writers."""

    resolved_name = name or source.name
    recipe = MaterialRecipe(
        id=material_id if material_id is not None else source.material_id,
        name=resolved_name,
        description=(
            description
            or source.description
            or f"Material derived from supplied source {resolved_name}."
        ),
        appearance_prompt=appearance_prompt,
        base_color_hint=base_color,
        pbr_hints=PBRHints(
            roughness=roughness,
            metallic=metallic,
            opacity=source.opacity,
            transmission=source.transmission,
            ior=source.ior,
            thin_walled=source.thin_walled,
        ),
    )
    recipe.validate()
    return recipe


__all__ = ["candidate_material_recipe", "source_material_recipe"]
