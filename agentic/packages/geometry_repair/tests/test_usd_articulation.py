# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for source-authoritative USD articulation extraction."""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from geometry_repair.advanced_profiles import evaluate_articulated_geometry
from geometry_repair.usd_articulation import (
    extract_usd_articulation_request,
    normalize_nested_rigid_body_xforms,
)


def _slider_stage(path: Path, *, include_child_geometry: bool = True) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Slider").GetPrim()
    stage.SetDefaultPrim(root)

    base = UsdGeom.Xform.Define(stage, "/Slider/Base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    base_cube = UsdGeom.Cube.Define(stage, "/Slider/Base/BaseCollider")
    base_cube.GetSizeAttr().Set(2.0)
    UsdPhysics.CollisionAPI.Apply(base_cube.GetPrim())

    child = UsdGeom.Xform.Define(stage, "/Slider/Base/Drawer")
    UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
    if include_child_geometry:
        child_cube = UsdGeom.Cube.Define(stage, "/Slider/Base/Drawer/DrawerCollider")
        child_cube.GetSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(child_cube.GetPrim())

    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Slider/Base/Drawer/SlideJoint")
    joint.GetBody0Rel().SetTargets([base.GetPath()])
    joint.GetBody1Rel().SetTargets([child.GetPath()])
    joint.GetAxisAttr().Set(UsdGeom.Tokens.y)
    joint.GetLowerLimitAttr().Set(-2.0)
    joint.GetUpperLimitAttr().Set(0.0)
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(1.0, 0.0, 0.0))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
    stage.GetRootLayer().Save()
    return path


def test_extract_usd_articulation_builds_authoritative_request(tmp_path: Path) -> None:
    source = _slider_stage(tmp_path / "slider.usda")

    result = extract_usd_articulation_request(source, sample_count=5)

    assert result.status == "ready"
    assert result.request is not None
    assert len(result.request.semantic_links) == 2
    assert len(result.request.link_mappings) == 2
    assert len(result.request.joints) == 1
    joint = result.request.joints[0]
    assert joint.axis_world == (0.0, 1.0, 0.0)
    assert joint.origin_world_m == (0.02, 0.02, 0.03)
    assert joint.lower_limit == -0.02
    assert joint.upper_limit == 0.0
    assert joint.units == "meters"
    assert len(result.request.adjacent_link_exclusions) == 1

    evidence = evaluate_articulated_geometry(
        source,
        source,
        semantic_links=result.request.semantic_links,
        link_mappings=result.request.link_mappings,
        joints=result.request.joints,
        adjacent_link_exclusions=result.request.adjacent_link_exclusions,
        max_candidate_pairs=result.request.max_candidate_pairs,
    )
    assert evidence.status == "pass"
    assert evidence.certification_eligible is True


def test_extract_usd_articulation_blocks_missing_link_geometry(tmp_path: Path) -> None:
    source = _slider_stage(tmp_path / "incomplete.usda", include_child_geometry=False)

    result = extract_usd_articulation_request(source)

    assert result.status == "blocked"
    assert result.request is None
    assert any("visible geometry is missing" in blocker for blocker in result.blockers)
    assert any("collision geometry is missing" in blocker for blocker in result.blockers)


def test_normalize_nested_rigid_bodies_preserves_world_transforms(tmp_path: Path) -> None:
    source = _slider_stage(tmp_path / "nested.usda")
    source_stage = Usd.Stage.Open(str(source))
    before = UsdGeom.XformCache().GetLocalToWorldTransform(
        source_stage.GetPrimAtPath("/Slider/Base/Drawer")
    )

    output = tmp_path / "normalized.usda"
    report = normalize_nested_rigid_body_xforms(source, output)

    assert report.status == "pass"
    assert report.normalized_prim_paths == ["/Slider/Base/Drawer"]
    assert report.maximum_world_transform_drift == 0.0
    normalized = Usd.Stage.Open(str(output))
    child = UsdGeom.Xformable(normalized.GetPrimAtPath("/Slider/Base/Drawer"))
    assert child.GetResetXformStack() is True
    after = UsdGeom.XformCache().GetLocalToWorldTransform(child.GetPrim())
    assert before == after
