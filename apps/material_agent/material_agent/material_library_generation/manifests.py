# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest writers for generated material libraries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    MaterialGenerationPlan,
)


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def material_entry(
    material: GeneratedMaterial,
    *,
    include_generation_metadata: bool = True,
) -> dict[str, Any]:
    """Return one Material Agent manifest entry."""
    recipe = material.recipe
    entry: dict[str, Any] = {
        "name": recipe.name,
        "description": recipe.description,
        "binding": recipe.binding,
    }
    if include_generation_metadata:
        entry["source"] = "generated"
        entry["generation_id"] = recipe.material_id
        if material.prototype_source:
            entry["prototype_source"] = dict(material.prototype_source)
        if recipe.intended_parts:
            entry["intended_parts"] = [
                part.semantic_label for part in recipe.intended_parts
            ]
    return entry


def write_materials_manifest(
    manifest_path: str | Path,
    library_path: str | Path,
    materials: list[GeneratedMaterial] | tuple[GeneratedMaterial, ...],
    *,
    include_generation_metadata: bool = True,
) -> Path:
    """Write canonical Material Agent `materials.yaml` for a generated library."""
    manifest_path = Path(manifest_path)
    library_path = Path(library_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "library_path": _relative_path(library_path, manifest_path.parent),
        "entries": [
            material_entry(
                material,
                include_generation_metadata=include_generation_metadata,
            )
            for material in materials
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return manifest_path


def write_generation_plan(
    plan_path: str | Path,
    plan: MaterialGenerationPlan,
    materials: list[GeneratedMaterial] | tuple[GeneratedMaterial, ...],
) -> Path:
    """Write the non-canonical recipe/debug plan used to create the library."""
    plan_path = Path(plan_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    texture_paths: dict[str, dict[str, str]] = {}
    for material in materials:
        texture_paths[material.recipe.material_id] = material.textures.as_relative_dict(
            plan_path.parent
        )

    with open(plan_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(plan.to_dict(texture_paths), stream, sort_keys=False)
    return plan_path
