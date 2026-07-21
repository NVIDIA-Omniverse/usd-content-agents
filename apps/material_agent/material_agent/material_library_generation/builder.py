# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level builder for generated material library packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from material_agent.material_library_generation.manifests import (
    write_generation_plan,
    write_materials_manifest,
)
from material_agent.material_library_generation.prototypes import (
    load_material_prototypes_from_data,
    load_material_prototypes_from_manifest,
    select_material_prototype,
)
from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    GeneratedMaterialLibrary,
    MaterialGenerationPlan,
)
from material_agent.material_library_generation.texture_generation import (
    TextureGenerationSettings,
    generate_texture_maps,
)
from material_agent.material_library_generation.usd_authoring import (
    write_material_library_usd,
)
from material_agent.material_profiles import MaterialProfile, normalize_material_profile


def build_generated_material_library(
    plan: MaterialGenerationPlan,
    package_dir: str | Path,
    *,
    texture_settings: TextureGenerationSettings | None = None,
    image_model: Any | None = None,
    prototype_materials_data: dict[str, Any] | None = None,
    prototype_materials_path: str | Path | None = None,
    prototype_min_score: float = 0.75,
    material_profile: str | MaterialProfile = "auto",
    write_debug_plan: bool = True,
    include_generation_metadata: bool = True,
) -> GeneratedMaterialLibrary:
    """Generate textures, author USD, and write `materials.yaml` for a plan."""
    plan.validate()
    material_profile = normalize_material_profile(material_profile)
    package_dir = Path(package_dir)
    textures_dir = package_dir / "textures"
    package_dir.mkdir(parents=True, exist_ok=True)

    generation_plan_path: Path | None = None
    if write_debug_plan:
        generation_plan_path = package_dir / "material_generation_plan.yaml"
        write_generation_plan(generation_plan_path, plan, ())

    prototypes = ()
    if prototype_materials_data:
        prototypes = load_material_prototypes_from_data(prototype_materials_data)
    elif prototype_materials_path:
        prototypes = load_material_prototypes_from_manifest(prototype_materials_path)

    generated: list[GeneratedMaterial] = []
    for recipe in plan.materials:
        texture_maps = generate_texture_maps(
            recipe,
            textures_dir / recipe.material_id,
            settings=texture_settings,
            image_model=image_model,
        )
        prototype_source = None
        prototype_match = select_material_prototype(
            recipe,
            prototypes,
            min_score=prototype_min_score,
        )
        if prototype_match is not None:
            prototype, score = prototype_match
            prototype_source = prototype.to_source_dict(score=score)
        generated.append(
            GeneratedMaterial(
                recipe=recipe,
                textures=texture_maps,
                prototype_source=prototype_source,
            )
        )

    material_library_path = package_dir / "material_library.usda"
    write_material_library_usd(
        material_library_path,
        generated,
        material_profile=material_profile,
    )

    materials_manifest_path = package_dir / "materials.yaml"
    write_materials_manifest(
        materials_manifest_path,
        material_library_path,
        generated,
        include_generation_metadata=include_generation_metadata,
    )

    if write_debug_plan:
        write_generation_plan(generation_plan_path, plan, generated)

    return GeneratedMaterialLibrary(
        package_dir=package_dir,
        material_library_path=material_library_path,
        materials_manifest_path=materials_manifest_path,
        generation_plan_path=generation_plan_path,
        materials=tuple(generated),
    )
