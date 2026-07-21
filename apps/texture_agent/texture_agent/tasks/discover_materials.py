# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Discover materials in a USD stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import (
    discover_effective_materials_from_file,
)

logger = logging.getLogger(__name__)


class DiscoverMaterialsTask(Task):
    """Discover materials in a USD file.

    Reads the input USD, traverses material prims, and extracts their
    OpenPBR attributes and common MaterialX/MDL shader-network properties.

    Context keys read:
        usd_path (str): Path to the input USD file.
        prim_paths (list[str], optional): Legacy material prim path scope.
        material_prim_paths (list[str], optional): Exact material path scope.
        prim_scope_paths (list[str], optional): Geometry prim scope.
        upstream_assignment_paths (list[str], optional): Upstream assignment scope.
        working_dir (str): Working directory for output.

    Context keys written:
        discovered_materials (list[MaterialInfo]): Discovered materials.
        effective_material_discovery (EffectiveMaterialDiscovery): Full result.
        effective_materials (list[MaterialInfo]): Effectively bound materials.
        material_discovery_counts (dict[str, int]): Auditable plan counts.
    """

    def __init__(self) -> None:
        self.name = "DiscoverMaterials"
        self.description = "Discover materials and texture metadata in the USD stage"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        material_prim_paths = context.get("material_prim_paths")
        if material_prim_paths is None:
            material_prim_paths = context.get("prim_paths")
        prim_scope_paths = context.get("prim_scope_paths")
        upstream_assignment_paths = context.get("upstream_assignment_paths")

        logger.info("Discovering materials in %s", usd_path)
        discovery = discover_effective_materials_from_file(
            usd_path,
            material_prim_paths=material_prim_paths,
            prim_scope_paths=prim_scope_paths,
            upstream_assignment_paths=upstream_assignment_paths,
        )
        materials = list(discovery.authored_materials)
        if material_prim_paths:
            material_path_set = set(material_prim_paths)
            scoped_materials = {
                material.prim_path: material
                for material in materials
                if material.prim_path in material_path_set
                or material_path_set.intersection(material.material_alias_paths)
            }
            # Instance-proxy aliases reduce to one representative composed
            # material path in effective discovery. Keep that representative
            # available to legacy downstream steps as well.
            scoped_materials.update(
                {
                    material.prim_path: material
                    for material in discovery.effective_materials
                }
            )
            materials = [scoped_materials[path] for path in sorted(scoped_materials)]

        context["discovered_materials"] = materials
        context["effective_material_discovery"] = discovery
        context["effective_materials"] = list(discovery.effective_materials)
        context["material_discovery_counts"] = {
            "authored_material_count": discovery.authored_material_count,
            "renderable_prim_count": discovery.renderable_prim_count,
            "renderable_subset_count": discovery.renderable_subset_count,
            "effective_bound_material_count": (
                discovery.effective_bound_material_count
            ),
        }

        # Save discovery results to working dir
        working_dir = context.get("working_dir")
        if working_dir:
            out_dir = Path(working_dir) / "discovery"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = [
                {
                    "name": m.name,
                    "prim_path": m.prim_path,
                    "base_color": list(m.base_color),
                    "has_existing_texture": m.has_existing_texture,
                    "bound_prims": len(m.bound_prim_paths),
                    "bound_subsets": len(m.bound_subset_paths),
                }
                for m in materials
            ]
            (out_dir / "materials.json").write_text(json.dumps(summary, indent=2))

        # Log summary table
        logger.info("Discovered %d materials:", len(materials))
        for m in materials:
            logger.info(
                "  %-30s base_color=(%.2f, %.2f, %.2f) texture=%s prims=%d",
                m.name,
                *m.base_color,
                "yes" if m.has_existing_texture else "no",
                len(m.bound_prim_paths),
            )

        return context
