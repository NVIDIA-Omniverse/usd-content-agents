# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for native USD analytic geometry in the shared mesh loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from geometry_repair.mesh_io import _canonicalize_flattened_prototypes, load_meshes


def _analytic_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)

    cube = UsdGeom.Cube.Define(stage, "/Asset/CollisionCube")
    cube.GetSizeAttr().Set(2.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    shared = UsdGeom.Sphere.Define(stage, "/Asset/SharedPhysicsSphere")
    shared.GetRadiusAttr().Set(0.4)
    UsdPhysics.CollisionAPI.Apply(shared.GetPrim())

    cylinder = UsdGeom.Cylinder.Define(stage, "/Asset/RenderCylinder")
    cylinder.GetRadiusAttr().Set(0.5)
    cylinder.GetHeightAttr().Set(4.0)
    cylinder.GetAxisAttr().Set(UsdGeom.Tokens.y)

    sphere = UsdGeom.Sphere.Define(stage, "/Asset/RenderSphere")
    sphere.GetRadiusAttr().Set(0.75)
    UsdGeom.Cone.Define(stage, "/Asset/RenderCone")
    UsdGeom.Capsule.Define(stage, "/Asset/RenderCapsule")
    stage.GetRootLayer().Save()
    return path


def test_load_meshes_tessellates_native_usd_analytic_primitives(tmp_path: Path) -> None:
    source = _analytic_stage(tmp_path / "analytic.usda")

    render_meshes, metadata = load_meshes(source)
    render_by_path = {mesh.path: mesh for mesh in render_meshes}
    assert set(render_by_path) == {
        "/Asset/RenderCapsule",
        "/Asset/RenderCone",
        "/Asset/RenderCylinder",
        "/Asset/RenderSphere",
        "/Asset/SharedPhysicsSphere",
    }
    assert metadata["source_collision_paths"] == [
        "/Asset/CollisionCube",
        "/Asset/SharedPhysicsSphere",
    ]
    assert render_by_path["/Asset/SharedPhysicsSphere"].role == "render"
    cylinder_extents = np.ptp(
        render_by_path["/Asset/RenderCylinder"].world_vertices_m,
        axis=0,
    )
    assert np.allclose(cylinder_extents, (0.01, 0.04, 0.01), atol=1e-9)

    all_meshes, _ = load_meshes(source, include_guide_purpose=True)
    all_by_path = {mesh.path: mesh for mesh in all_meshes}
    collision = all_by_path["/Asset/CollisionCube"]
    assert collision.role == "collision"
    assert np.allclose(np.ptp(collision.world_vertices_m, axis=0), 0.02, atol=1e-9)
    assert len(collision.triangles) == 12


def _flattened_prototype_stage(path: Path, assignments: dict[str, tuple[int, str]]) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Asset")
    for material_name, (prototype_index, value) in assignments.items():
        prototype_path = f"/Flattened_Prototype_{prototype_index}"
        prototype = UsdGeom.Xform.Define(stage, prototype_path).GetPrim()
        prototype.CreateAttribute("semanticValue", Sdf.ValueTypeNames.String).Set(value)
        material = UsdGeom.Xform.Define(stage, f"/Asset/{material_name}").GetPrim()
        material.GetReferences().AddInternalReference(prototype_path)
    stage.GetRootLayer().Save()
    return path


def test_flattened_prototype_names_are_canonical_across_source_numbering(
    tmp_path: Path,
) -> None:
    first = _flattened_prototype_stage(
        tmp_path / "first.usda",
        {"Brass": (9, "brass"), "Steel": (2, "steel")},
    )
    second = _flattened_prototype_stage(
        tmp_path / "second.usda",
        {"Brass": (3, "brass"), "Steel": (8, "steel")},
    )

    _canonicalize_flattened_prototypes(first)
    _canonicalize_flattened_prototypes(second)

    first_stage = Usd.Stage.Open(str(first))
    second_stage = Usd.Stage.Open(str(second))
    assert first_stage and second_stage
    assert first_stage.GetRootLayer().ExportToString() == (
        second_stage.GetRootLayer().ExportToString()
    )
    assert first_stage.GetPrimAtPath("/Asset/Brass").GetAttribute("semanticValue").Get() == (
        "brass"
    )
    assert first_stage.GetPrimAtPath("/Asset/Steel").GetAttribute("semanticValue").Get() == (
        "steel"
    )
