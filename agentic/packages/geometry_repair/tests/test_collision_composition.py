# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-safe render and collision composition regressions."""

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from geometry_repair.collision import compose_asset
from geometry_repair.orchestrator import _validate_final_usd_package


def _root_mesh_stage(path: Path, root_name: str) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, f"/{root_name}")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 0.0, 1.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()
    return path


def test_compose_without_replacement_preserves_source_root_collider(tmp_path: Path) -> None:
    render = _root_mesh_stage(tmp_path / "render.usda", "RenderMesh")

    composed = compose_asset(render, None, tmp_path / "asset.usda")

    stage = Usd.Stage.Open(str(composed))
    assert stage is not None
    assert str(stage.GetDefaultPrim().GetPath()) == "/GeometryRepairAsset"
    render_prim = stage.GetPrimAtPath("/GeometryRepairAsset/Render")
    assert render_prim.IsA(UsdGeom.Mesh)
    assert render_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.CollisionAPI(render_prim).GetCollisionEnabledAttr().Get() is not False
    assert (
        stage.GetDefaultPrim().GetCustomDataByKey("geometryRepairDisabledSourceColliderCount") == 0
    )
    validation = _validate_final_usd_package(
        render,
        None,
        composed,
        tmp_path / "validation.json",
    )
    assert validation["status"] == "pass"
    assert validation["composition"]["source_active_collision_count"] == 1
    assert validation["composition"]["composed_render_active_collision_count"] == 1


def test_compose_with_replacement_disables_source_and_keeps_replacement(tmp_path: Path) -> None:
    render = _root_mesh_stage(tmp_path / "render.usda", "RenderMesh")
    collision = _root_mesh_stage(tmp_path / "collision.usda", "CollisionMesh")

    composed = compose_asset(render, collision, tmp_path / "asset.usda")

    stage = Usd.Stage.Open(str(composed))
    assert stage is not None
    render_prim = stage.GetPrimAtPath("/GeometryRepairAsset/Render")
    collision_prim = stage.GetPrimAtPath("/GeometryRepairAsset/Collision")
    assert render_prim.IsA(UsdGeom.Mesh)
    assert collision_prim.IsA(UsdGeom.Mesh)
    assert UsdPhysics.CollisionAPI(render_prim).GetCollisionEnabledAttr().Get() is False
    assert UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get() is not False
    assert (
        stage.GetDefaultPrim().GetCustomDataByKey("geometryRepairDisabledSourceColliderCount") == 1
    )
    validation = _validate_final_usd_package(
        render,
        collision,
        composed,
        tmp_path / "validation.json",
    )
    assert validation["status"] == "pass"
    assert validation["composition"]["composed_render_active_collision_count"] == 0
    assert validation["composition"]["composed_collision_active_count"] == 1


def test_compose_rejects_missing_replacement_layer(tmp_path: Path) -> None:
    render = _root_mesh_stage(tmp_path / "render.usda", "RenderMesh")

    with pytest.raises(FileNotFoundError, match="collision layer does not exist"):
        compose_asset(render, tmp_path / "missing.usda", tmp_path / "asset.usda")
