# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic delivery audits for material-assignment workflows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

BINDING_AUDIT_SCHEMA_VERSION = "content-agent-material-binding-audit.v1"


def _bound_material_prim_paths(prims: Iterable[Any]) -> set[str]:
    from pxr import UsdShade

    prim_list = list(prims)
    bound_paths: set[str] = set()
    for purpose in (
        UsdShade.Tokens.allPurpose,
        UsdShade.Tokens.preview,
        UsdShade.Tokens.full,
    ):
        materials, _relationships = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(
            prim_list,
            purpose,
        )
        for prim, material in zip(prim_list, materials, strict=True):
            if material and material.GetPrim().IsValid():
                bound_paths.add(str(prim.GetPath()))
    return bound_paths


def _mesh_subset_coverage(mesh: Any, bound_prim_paths: set[str]) -> tuple[int, int]:
    from pxr import UsdShade

    face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    total_faces = len(face_counts)
    bound_faces: set[int] = set()
    for subset in UsdShade.MaterialBindingAPI(mesh.GetPrim()).GetMaterialBindSubsets():
        if str(subset.GetPrim().GetPath()) not in bound_prim_paths:
            continue
        for index in subset.GetIndicesAttr().Get() or []:
            if isinstance(index, int) and 0 <= index < total_faces:
                bound_faces.add(index)
    return len(bound_faces), total_faces


def audit_usd_material_bindings(usd_path: Path) -> dict[str, Any]:
    """Count composed bindings on active, visible, render-purpose meshes."""

    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD stage: {usd_path}")

    mesh_count = 0
    visible_meshes: list[Any] = []
    bound_mesh_count = 0
    subset_bound_mesh_count = 0
    partially_bound_mesh_count = 0
    unbound_paths: list[str] = []
    partially_bound_paths: list[str] = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        visible_meshes.append(UsdGeom.Mesh(prim))

    binding_prims = [mesh.GetPrim() for mesh in visible_meshes]
    for mesh in visible_meshes:
        binding_prims.extend(
            subset.GetPrim()
            for subset in UsdShade.MaterialBindingAPI(
                mesh.GetPrim()
            ).GetMaterialBindSubsets()
        )
    bound_prim_paths = _bound_material_prim_paths(binding_prims)
    visible_mesh_count = len(visible_meshes)
    for mesh in visible_meshes:
        prim = mesh.GetPrim()
        if str(prim.GetPath()) in bound_prim_paths:
            bound_mesh_count += 1
            continue

        bound_faces, total_faces = _mesh_subset_coverage(mesh, bound_prim_paths)
        if total_faces > 0 and bound_faces == total_faces:
            bound_mesh_count += 1
            subset_bound_mesh_count += 1
        elif bound_faces > 0:
            partially_bound_mesh_count += 1
            partially_bound_paths.append(str(prim.GetPath()))
        else:
            unbound_paths.append(str(prim.GetPath()))

    strict_coverage = (
        round(100.0 * bound_mesh_count / visible_mesh_count, 1)
        if visible_mesh_count
        else None
    )
    return {
        "schema_version": BINDING_AUDIT_SCHEMA_VERSION,
        "usd_path": str(usd_path),
        "mesh_count": mesh_count,
        "visible_render_mesh_count": visible_mesh_count,
        "bound_visible_mesh_count": bound_mesh_count,
        "subset_bound_visible_mesh_count": subset_bound_mesh_count,
        "partially_bound_visible_mesh_count": partially_bound_mesh_count,
        "unbound_visible_mesh_count": len(unbound_paths),
        "final_mesh_binding_coverage_score": strict_coverage,
        "unbound_visible_mesh_paths": unbound_paths,
        "partially_bound_visible_mesh_paths": partially_bound_paths,
    }
