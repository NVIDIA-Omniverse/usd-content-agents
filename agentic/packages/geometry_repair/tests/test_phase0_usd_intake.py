# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 tests for non-mutating USD dependency and prim-role intake."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pxr", reason="OpenUSD Python bindings are not installed")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from geometry_repair.diagnosis import _usd_semantic_issues
from geometry_repair.models import GeometryMetrics
from geometry_repair.usd_intake import (
    _layer_dependencies,
    _material_fact,
    _primvars,
    inventory_usd_stage,
)


def _write_inventory_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    Usd.ModelAPI(root.GetPrim()).SetKind("component")
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))

    material = UsdShade.Material.Define(stage, "/Asset/Looks/Plastic")
    render_cube = UsdGeom.Cube.Define(stage, "/Asset/RenderCube")
    render_cube.CreateSizeAttr(0.25)
    render_cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdShade.MaterialBindingAPI.Apply(render_cube.GetPrim()).Bind(material)

    mesh = UsdGeom.Mesh.Define(stage, "/Asset/AttributedMesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    primvar.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(0, 1)]))
    subset = UsdGeom.Subset.Define(stage, "/Asset/AttributedMesh/RegionA")
    subset.CreateElementTypeAttr(UsdGeom.Tokens.face)
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr(Vt.IntArray([0]))
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)

    collision = UsdGeom.Mesh.Define(stage, "/Asset/CollisionProxy")
    collision.CreatePointsAttr(mesh.GetPointsAttr().Get())
    collision.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    collision.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
    collision.CreatePurposeAttr(UsdGeom.Tokens.guide)

    disabled = UsdGeom.Sphere.Define(stage, "/Asset/DisabledVisualCollision")
    disabled.CreateRadiusAttr(0.125)
    UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)

    guide = UsdGeom.Cylinder.Define(stage, "/Asset/GuideShape")
    guide.CreateRadiusAttr(0.05)
    guide.CreateHeightAttr(0.4)
    guide.CreateAxisAttr(UsdGeom.Tokens.x)
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)

    UsdGeom.Xform.Define(stage, "/Asset/helper_locator")
    stage.GetRootLayer().Save()
    return path


def _by_path(report):
    return {record.path: record for record in report.prims}


def test_inventory_records_joint_rigid_body_and_time_sample_facts(tmp_path: Path) -> None:
    source = tmp_path / "articulated.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    left = UsdGeom.Xform.Define(stage, "/Asset/Left")
    right = UsdGeom.Xform.Define(stage, "/Asset/Right")
    UsdPhysics.RigidBodyAPI.Apply(left.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(right.GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/Left/Render")
    UsdGeom.Cube.Define(stage, "/Asset/Right/Render")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/Hinge")
    joint.CreateBody0Rel().SetTargets([left.GetPath()])
    joint.CreateBody1Rel().SetTargets([right.GetPath()])
    right.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0.0))
    right.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.1, 0.0, 0.0), Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()

    report = inventory_usd_stage(source)

    assert report.joint_prim_paths == ["/Asset/Hinge"]
    assert report.rigid_body_prim_paths == ["/Asset/Left", "/Asset/Right"]
    assert "/Asset/Right" in report.time_varying_prim_paths
    assert "/Asset/Right" in report.geometry_time_varying_prim_paths


def test_inventory_records_brep_and_semantic_diagnosis_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "brep_with_mesh.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/Render")
    stage.DefinePrim("/Asset/ExactSource", "BrepArray")
    stage.GetRootLayer().Save()

    report = inventory_usd_stage(source)
    issues = _usd_semantic_issues(
        {
            "joint_prim_paths": report.joint_prim_paths,
            "rigid_body_prim_paths": report.rigid_body_prim_paths,
            "time_varying_prim_paths": report.time_varying_prim_paths,
            "geometry_time_varying_prim_paths": (report.geometry_time_varying_prim_paths),
            "skeleton_prim_paths": report.skeleton_prim_paths,
            "animation_prim_paths": report.animation_prim_paths,
            "brep_prim_paths": report.brep_prim_paths,
        },
        GeometryMetrics(source_format="usd", source_part_paths=["/Asset/Render"]),
        "rigid_pick_place",
    )

    assert report.brep_prim_paths == ["/Asset/ExactSource"]
    unsupported = next(
        issue for issue in issues if issue.issue_id == "brep:validation_not_evaluated"
    )
    assert unsupported.blocking_profiles == ["rigid_pick_place"]
    assert unsupported.affected_prim_paths == ["/Asset/ExactSource"]


def test_material_animation_is_not_misclassified_as_dynamic_geometry(tmp_path: Path) -> None:
    source = tmp_path / "animated_material.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/Render")
    shader = UsdShade.Shader.Define(stage, "/Asset/Looks/Shader")
    roughness = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    roughness.Set(0.2, Usd.TimeCode(0.0))
    roughness.Set(0.8, Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()

    report = inventory_usd_stage(source)

    assert "/Asset/Looks/Shader" in report.time_varying_prim_paths
    assert "/Asset/Looks/Shader" not in report.geometry_time_varying_prim_paths


def test_semantic_facts_block_dynamic_source_under_rigid_profile() -> None:
    facts = {
        "joint_prim_paths": ["/Asset/Hinge"],
        "rigid_body_prim_paths": ["/Asset/Left", "/Asset/Right"],
        "time_varying_prim_paths": ["/Asset/Right"],
        "geometry_time_varying_prim_paths": ["/Asset/Right"],
        "skeleton_prim_paths": [],
        "animation_prim_paths": [],
    }
    metrics = GeometryMetrics(
        source_format="usd",
        source_part_paths=["/Asset/Left/Render", "/Asset/Right/Render"],
    )

    rigid = _usd_semantic_issues(facts, metrics, "rigid_pick_place")
    articulated = _usd_semantic_issues(facts, metrics, "articulated_rigid")

    assert rigid[0].issue_id == "semantic:dynamic_source_conflicts_with_rigid_profile"
    assert rigid[0].blocking_profiles == ["rigid_pick_place"]
    assert not any(
        issue.issue_id == "semantic:fused_or_incomplete_articulated_links" for issue in articulated
    )


def test_material_only_time_samples_do_not_conflict_with_rigid_profile() -> None:
    issues = _usd_semantic_issues(
        {
            "joint_prim_paths": [],
            "rigid_body_prim_paths": [],
            "time_varying_prim_paths": ["/Asset/Looks/Shader"],
            "geometry_time_varying_prim_paths": [],
            "skeleton_prim_paths": [],
            "animation_prim_paths": [],
        },
        GeometryMetrics(source_format="usd", source_part_paths=["/Asset/Render"]),
        "rigid_pick_place",
    )

    assert not any(
        issue.issue_id == "semantic:dynamic_source_conflicts_with_rigid_profile" for issue in issues
    )


def test_inventory_records_roles_authored_evidence_and_native_primitives(
    tmp_path: Path,
) -> None:
    source = _write_inventory_stage(tmp_path / "inventory.usda")
    before = source.read_bytes()

    report = inventory_usd_stage(source)

    assert report.layer_readable is True
    assert report.stage_readable is True
    assert report.composition_complete is True
    assert report.source_unchanged is True
    assert source.read_bytes() == before
    assert report.default_prim_path == "/Asset"
    assert report.meters_per_unit == pytest.approx(1.0)
    assert report.up_axis == "Z"

    prims = _by_path(report)
    render = prims["/Asset/RenderCube"]
    assert render.role == "render"
    assert render.visibility == "invisible"
    assert render.native_primitive is not None
    assert render.native_primitive.schema_type == "Cube"
    assert render.native_primitive.parameters == {"size": pytest.approx(0.25)}
    assert render.material.computed_material_path == "/Asset/Looks/Plastic"
    assert render.material.computed_binding_status == "bound"
    assert render.material.warnings == []
    assert render.inspection_warnings == []
    assert any(
        evidence.signal == "geometry_schema" and evidence.supports == "render" and evidence.decisive
        for evidence in render.role_evidence
    )
    assert not any(
        evidence.signal == "visibility" and evidence.decisive for evidence in render.role_evidence
    )

    attributed = prims["/Asset/AttributedMesh"]
    primvars = {item.name: item for item in attributed.primvars}
    assert {"displayColor", "displayOpacity", "st"} <= set(primvars)
    assert primvars["st"].interpolation == "faceVarying"
    assert primvars["st"].value_count == 3
    assert primvars["st"].authored is True
    assert primvars["displayColor"].authored is False
    assert attributed.inspection_warnings == []
    assert len(attributed.material.subsets) == 1
    assert attributed.material.subsets[0].path == "/Asset/AttributedMesh/RegionA"
    assert attributed.material.subsets[0].index_count == 1
    assert attributed.material.subsets[0].material_binding_targets == ["/Asset/Looks/Plastic"]

    collision = prims["/Asset/CollisionProxy"]
    assert collision.role == "source_collision"
    assert collision.collision.api_applied is True
    assert collision.collision.enabled is True
    assert collision.collision.enabled_authored is True
    assert collision.collision.approximation == "convexHull"
    assert collision.purpose == "guide"

    disabled = prims["/Asset/DisabledVisualCollision"]
    assert disabled.role == "disabled_visual_collision_api"
    assert disabled.collision.enabled is False
    assert disabled.native_primitive is not None
    assert disabled.native_primitive.parameters["radius"] == pytest.approx(0.125)

    guide = prims["/Asset/GuideShape"]
    assert guide.role == "guide"
    assert guide.native_primitive is not None
    assert guide.native_primitive.parameters == {
        "radius": pytest.approx(0.05),
        "height": pytest.approx(0.4),
        "axis": "X",
    }
    assert prims["/Asset/helper_locator"].role == "helper"
    assert prims["/Asset/Looks/Plastic"].role == "unknown"
    assert prims["/Asset/RenderCube"].parent_path == "/Asset"
    assert prims["/Asset"].world_transform is not None


def test_optional_material_and_primvar_failures_remain_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_inventory_stage(tmp_path / "inspection_failure.usda")
    stage = Usd.Stage.Open(str(source))
    prim = stage.GetPrimAtPath("/Asset/AttributedMesh")

    class BrokenMaterialBindingApi:
        def __init__(self, _prim) -> None:
            pass

        def ComputeBoundMaterial(self):
            raise RuntimeError("material inspection unavailable")

    class BrokenPrimvarsApi:
        def __init__(self, _prim) -> None:
            pass

        def GetPrimvars(self):
            raise RuntimeError("primvar inspection unavailable")

    monkeypatch.setattr(UsdShade, "MaterialBindingAPI", BrokenMaterialBindingApi)
    material = _material_fact(prim)
    monkeypatch.setattr(UsdGeom, "PrimvarsAPI", BrokenPrimvarsApi)
    primvars, warnings = _primvars(prim)

    assert material.computed_material_path is None
    assert material.computed_binding_status == "not_evaluated"
    assert material.warnings == [
        "computed material binding inspection failed: RuntimeError: material inspection unavailable"
    ]
    assert primvars == []
    assert warnings == ["primvar inspection failed: RuntimeError: primvar inspection unavailable"]


def test_dependency_inventory_hashes_only_allowlisted_local_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture_dir = source_dir / "textures"
    texture_dir.mkdir()
    local_texture = texture_dir / "albedo.png"
    local_texture.write_bytes(b"local texture bytes")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"must not be hashed by default")

    source = source_dir / "dependencies.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    prim = root.GetPrim()
    prim.CreateAttribute("inputs:local", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png")
    )
    prim.CreateAttribute("inputs:missing", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/missing.png")
    )
    prim.CreateAttribute("inputs:remote", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("https://example.invalid/never-fetch.png")
    )
    prim.CreateAttribute("inputs:outside", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(outside))
    )
    stage.GetRootLayer().Save()

    report = inventory_usd_stage(source)
    records = {record.authored_path: record for record in report.dependencies}

    local = records["textures/albedo.png"]
    assert local.status == "resolved_local"
    assert local.local_path == str(local_texture.resolve())
    assert local.sha256 is not None
    assert local.size_bytes == len(b"local texture bytes")
    assert records["textures/missing.png"].status == "unresolved_local"
    assert records["https://example.invalid/never-fetch.png"].status == ("remote_not_fetched")
    assert records["https://example.invalid/never-fetch.png"].remote_scheme == "https"
    assert records[str(outside)].status == "outside_allowed_roots"
    assert records[str(outside)].sha256 is None
    assert report.unresolved_dependencies == sorted(
        [
            "https://example.invalid/never-fetch.png",
            str(outside),
            "textures/missing.png",
        ]
    )


def test_dependency_metadata_inspection_failure_is_explicit() -> None:
    class BrokenSpec:
        def ListInfoKeys(self):
            return ["brokenAssetMetadata"]

        def GetInfo(self, _key):
            raise RuntimeError("metadata decoder unavailable")

    class FakeLayer:
        realPath = "/portable/source.usda"
        identifier = "source.usda"
        subLayerPaths = []

        def GetObjectAtPath(self, _path):
            return BrokenSpec()

        def Traverse(self, _root, visitor):
            visitor(Sdf.Path("/Asset"))

        def GetExternalReferences(self):
            return []

        def GetExternalAssetDependencies(self):
            return []

    warnings: list[str] = []
    dependencies = _layer_dependencies(FakeLayer(), warnings=warnings)

    assert dependencies == []
    assert warnings == [
        "dependency metadata inspection failed for /portable/source.usda:/Asset "
        "key 'brokenAssetMetadata': RuntimeError: metadata decoder unavailable"
    ]


def test_unsafe_composition_dependency_uses_root_only_inventory(tmp_path: Path) -> None:
    source = tmp_path / "missing_reference.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Asset/LocalCube")
    root.GetPrim().GetReferences().AddReference("missing.usda", "/ReferencedAsset")
    stage.GetRootLayer().Save()

    report = inventory_usd_stage(source)

    assert report.composition_complete is False
    assert report.stage_readable is True
    assert report.dependencies[0].kind == "reference"
    assert report.dependencies[0].status == "unresolved_local"
    assert "/Asset/LocalCube" in _by_path(report)
    assert any("isolated root-layer view" in warning for warning in report.warnings)
