# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic USD physics sanity inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pxr.Usd", reason="USD (pxr) not available in this env")
pytest.importorskip("pxr.UsdGeom", reason="USD geometry bindings not available")
pytest.importorskip("pxr.UsdPhysics", reason="USD physics bindings not available")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from world_understanding.functions.physics import (  # noqa: E402
    infer_physics_expected,
    inspect_usd_physics,
)
from world_understanding.functions.physics.physics_sanity import (  # noqa: E402
    PhysicsSanityFinding,
    PhysicsSanityResult,
    _authored_number,
    _check_mass_scale,
    _world_bbox_info,
)


def _new_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


def _save(stage: Usd.Stage) -> None:
    stage.GetRootLayer().Save()


def _add_scene(stage: Usd.Stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityMagnitudeAttr(9.81)


def _add_valid_body(stage: Usd.Stage, body_path: str = "/World/Body") -> None:
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr(10.0)
    mass_api.CreateDensityAttr(1000.0)
    collider = UsdGeom.Cube.Define(stage, f"{body_path}/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)


def _codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def test_infer_physics_expected_prefers_explicit_flag() -> None:
    assert infer_physics_expected("visual only, no physics", expect_physics=True)
    assert not infer_physics_expected("simulate gravity", expect_physics=False)


def test_infer_physics_expected_from_task_text() -> None:
    assert infer_physics_expected("Check collision, mass, and rigid body setup")
    assert not infer_physics_expected("Visual only, no physics checks")
    assert not infer_physics_expected()
    assert not infer_physics_expected("Validate the cases that fall through")


def test_open_failure_is_reported_cleanly(tmp_path: Path) -> None:
    result = inspect_usd_physics(tmp_path / "missing.usda", expect_physics=True)

    assert not result.opened
    assert not result.passed
    assert _codes(result) == {"physics.usd_open_failed"}


def test_falsey_stage_open_is_reported_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "empty.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(Usd.Stage, "Open", lambda _path: None)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert not result.opened
    assert not result.passed
    assert _codes(result) == {"physics.usd_open_failed"}


def test_sanity_dataclasses_serialize_passed_state() -> None:
    finding = PhysicsSanityFinding(
        code="physics.example",
        severity="warn",
        message="example warning",
        prim_path="/World/Body",
        details={"answer": 42},
    )
    result = PhysicsSanityResult(
        usd_path="asset.usda",
        opened=True,
        physics_expected=True,
        findings=[finding],
    )

    assert finding.to_dict() == {
        "code": "physics.example",
        "severity": "warn",
        "message": "example warning",
        "prim_path": "/World/Body",
        "details": {"answer": 42},
    }
    assert result.to_dict()["passed"] is True


def test_valid_simple_physics_usd_passes(tmp_path: Path) -> None:
    usd_path = tmp_path / "valid_physics.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert result.passed
    assert result.findings == []
    assert result.summary["physics_scene_count"] == 1
    assert result.summary["rigid_body_count"] == 1
    assert result.summary["collider_count"] == 1


def test_single_mesh_rigid_body_group_is_not_multi_mesh(tmp_path: Path) -> None:
    usd_path = tmp_path / "single_mesh_body.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    mesh = UsdGeom.Mesh.Define(stage, "/World/SoloMesh").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.invalid_rigid_body_authoring" not in _codes(result)


def test_usd_validation_runs_basic_and_physics_categories(tmp_path: Path) -> None:
    from world_understanding.functions.graphics.validate_usd import (
        PHYSICS_VALIDATION_CATEGORIES,
        is_available,
        validate_usd,
    )

    if not is_available():
        pytest.skip("usd-validation-nvidia not installed")

    usd_path = tmp_path / "valid_physics.usda"
    stage = _new_stage(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    _add_scene(stage)
    _add_valid_body(stage)
    _save(stage)

    result = validate_usd(usd_path, categories=list(PHYSICS_VALIDATION_CATEGORIES))

    assert result["status"] == "success"
    assert set(result["categories_checked"]) == set(PHYSICS_VALIDATION_CATEGORIES)
    assert result["summary"]["failures"] == 0
    assert result["summary"]["errors"] == 0


def test_density_zero_with_authored_mass_is_sentinel(tmp_path: Path) -> None:
    # SimReady / UsdPhysics convention: ``MassAPI.density == 0`` paired with a
    # non-zero ``MassAPI.mass`` means "use authored mass directly; ignore
    # density for mass computation". The inspector must not raise
    # ``physics.property_out_of_range`` for that sentinel. See #172.
    usd_path = tmp_path / "density_zero_with_mass.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr(24.68)
    mass_api.CreateDensityAttr(0.0)
    collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert result.passed
    assert "physics.property_out_of_range" not in _codes(result)


def test_density_zero_without_mass_still_flagged(tmp_path: Path) -> None:
    # The sentinel exemption only applies when a non-zero mass is authored on
    # the same prim. A bare ``density == 0`` with no mass is still invalid.
    usd_path = tmp_path / "density_zero_no_mass.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateDensityAttr(0.0)
    collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.property_out_of_range" in _codes(result)


def test_no_physics_usd_fails_when_physics_expected(tmp_path: Path) -> None:
    usd_path = tmp_path / "no_physics.usda"
    stage = _new_stage(usd_path)
    UsdGeom.Cube.Define(stage, "/World/Cube")
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert not result.passed
    assert {
        "physics.expected_but_missing",
        "physics.no_physics_scene",
        "physics.no_rigid_bodies",
        "physics.no_colliders",
    }.issubset(_codes(result))


def test_no_physics_usd_passes_when_physics_not_expected(tmp_path: Path) -> None:
    usd_path = tmp_path / "visual_only.usda"
    stage = _new_stage(usd_path)
    UsdGeom.Cube.Define(stage, "/World/Cube")
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=False)

    assert result.passed
    assert result.findings == []


def test_physics_scene_without_rigid_bodies_fails(tmp_path: Path) -> None:
    usd_path = tmp_path / "scene_only.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    UsdGeom.Cube.Define(stage, "/World/Cube")
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert not result.passed
    assert "physics.no_rigid_bodies" in _codes(result)
    assert "physics.no_colliders" in _codes(result)
    assert "physics.expected_but_missing" not in _codes(result)


def test_rigid_body_without_colliders_fails(tmp_path: Path) -> None:
    usd_path = tmp_path / "body_without_colliders.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert not result.passed
    matching = [
        finding for finding in result.findings if finding.code == "physics.no_colliders"
    ]
    assert any(finding.prim_path == "/World/Body" for finding in matching)


def test_multi_mesh_per_mesh_rigid_bodies_are_flagged(tmp_path: Path) -> None:
    usd_path = tmp_path / "bad_multi_mesh.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    stage.DefinePrim("/World/Geometry", "Scope")
    for name in ("Aluminum", "Rubber", "Plastic"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Geometry/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    invalid = [
        finding
        for finding in result.findings
        if finding.code == "physics.invalid_rigid_body_authoring"
    ]
    assert len(invalid) == 1
    assert invalid[0].severity == "fail"
    assert invalid[0].prim_path == "/World/Geometry"
    assert invalid[0].details["suggested_rigid_body_prim"] == "/World/Geometry"
    assert set(invalid[0].details["rigid_body_meshes"]) == {
        "/World/Geometry/Aluminum",
        "/World/Geometry/Rubber",
        "/World/Geometry/Plastic",
    }


def test_multi_mesh_xform_geometry_container_is_flagged(tmp_path: Path) -> None:
    usd_path = tmp_path / "bad_xform_geometry.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    UsdGeom.Xform.Define(stage, "/World/Geometry")
    for name in ("PartA", "PartB"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Geometry/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    invalid = [
        finding
        for finding in result.findings
        if finding.code == "physics.invalid_rigid_body_authoring"
    ]
    assert len(invalid) == 1
    assert invalid[0].prim_path == "/World/Geometry"
    assert invalid[0].details["suggested_rigid_body_prim"] == "/World/Geometry"


def test_independent_sibling_mesh_rigid_bodies_pass_by_default(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "independent_mesh_bodies.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    for name in ("BoxA", "BoxB", "BoxC"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.invalid_rigid_body_authoring" not in _codes(result)
    assert result.passed


def test_root_sibling_mesh_rigid_bodies_are_not_auto_flagged(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "root_mesh_bodies.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    for name in ("RootBodyA", "RootBodyB"):
        mesh = UsdGeom.Mesh.Define(stage, f"/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.invalid_rigid_body_authoring" not in _codes(result)


def test_nested_mesh_parent_lookup_skips_mesh_ancestors(tmp_path: Path) -> None:
    usd_path = tmp_path / "nested_mesh_body.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    UsdGeom.Mesh.Define(stage, "/World/OuterMesh")
    inner = UsdGeom.Mesh.Define(stage, "/World/OuterMesh/InnerMesh").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(inner).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(inner).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.invalid_rigid_body_authoring" not in _codes(result)


def test_non_container_scope_sibling_mesh_bodies_are_not_auto_flagged(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "scope_mesh_bodies.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    stage.DefinePrim("/World/Parts", "Scope")
    for name in ("PartA", "PartB"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Parts/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.invalid_rigid_body_authoring" not in _codes(result)


def test_single_asset_policy_flags_independent_sibling_mesh_bodies(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "single_asset_mesh_bodies.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    for name in ("PartA", "PartB"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(mesh).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(
        usd_path,
        expect_physics=True,
        single_asset=True,
    )

    assert "physics.invalid_rigid_body_authoring" in _codes(result)


def test_out_of_range_properties_are_reported(tmp_path: Path) -> None:
    usd_path = tmp_path / "bad_properties.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)

    body = stage.GetPrimAtPath("/World/Body")
    mass_api = UsdPhysics.MassAPI(body)
    mass_api.CreateMassAttr(-1.0)
    mass_api.CreateDensityAttr(-10.0)

    material = stage.DefinePrim("/World/Looks/BadPhysicsMaterial", "Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material)
    material_api.CreateStaticFrictionAttr(-0.1)
    material_api.CreateDynamicFrictionAttr(12.0)
    material_api.CreateRestitutionAttr(1.5)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    property_findings = [
        finding
        for finding in result.findings
        if finding.code == "physics.property_out_of_range"
    ]
    reported = {
        (finding.prim_path, finding.details["property"])
        for finding in property_findings
    }
    assert ("/World/Body", "mass") in reported
    assert ("/World/Body", "density") in reported
    assert ("/World/Looks/BadPhysicsMaterial", "static_friction") in reported
    assert ("/World/Looks/BadPhysicsMaterial", "dynamic_friction") in reported
    assert ("/World/Looks/BadPhysicsMaterial", "restitution") in reported


def test_zero_mass_is_out_of_range(tmp_path: Path) -> None:
    usd_path = tmp_path / "zero_mass.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Body"))
    mass_api.CreateMassAttr(0.0)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert any(
        finding.code == "physics.property_out_of_range"
        and finding.prim_path == "/World/Body"
        and finding.details["property"] == "mass"
        for finding in result.findings
    )


def test_suspicious_mass_scale_is_warned(tmp_path: Path) -> None:
    usd_path = tmp_path / "mass_scale.usda"
    stage = _new_stage(usd_path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/LargeBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr(25_000.0)
    mass_api.CreateDensityAttr(2700.0)
    collider = UsdGeom.Cube.Define(stage, "/World/LargeBody/Collider").GetPrim()
    UsdGeom.Cube(collider).CreateSizeAttr(8.0)
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.mass_scale_suspicious" in _codes(result)
    warning = next(
        finding
        for finding in result.findings
        if finding.code == "physics.mass_scale_suspicious"
    )
    assert warning.severity == "warn"
    assert warning.prim_path == "/World/LargeBody"


def test_mass_scale_uses_stage_units(tmp_path: Path) -> None:
    usd_path = tmp_path / "centimeter_units.usda"
    stage = _new_stage(usd_path)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/MeterBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr(1000.0)
    mass_api.CreateDensityAttr(1000.0)
    collider = UsdGeom.Cube.Define(stage, "/World/MeterBody/Collider").GetPrim()
    UsdGeom.Cube(collider).CreateSizeAttr(100.0)
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert "physics.mass_scale_suspicious" not in _codes(result)


def test_authored_number_ignores_bool_and_non_numeric_values() -> None:
    class FakeAttr:
        def __init__(self, value: object) -> None:
            self.value = value

        def HasAuthoredValueOpinion(self) -> bool:
            return True

        def Get(self) -> object:
            return self.value

    assert _authored_number(FakeAttr(True)) is None
    assert _authored_number(FakeAttr("heavy")) is None


def test_mass_scale_ignores_missing_bbox_info() -> None:
    class FakePrim:
        def GetPath(self) -> str:
            return "/World/Body"

    class RaisingBBoxCache:
        def ComputeWorldBound(self, _prim: object) -> object:
            raise RuntimeError("bbox unavailable")

    findings: list[PhysicsSanityFinding] = []
    _check_mass_scale(
        findings,
        FakePrim(),
        mass=1_000.0,
        bbox_cache=RaisingBBoxCache(),
        meters_per_unit=1.0,
        kilograms_per_unit=1.0,
    )

    assert findings == []


def test_world_bbox_info_returns_none_for_empty_range() -> None:
    class FakePrim:
        pass

    class EmptyRange:
        def IsEmpty(self) -> bool:
            return True

    class FakeBound:
        def ComputeAlignedRange(self) -> EmptyRange:
            return EmptyRange()

    class FakeBBoxCache:
        def ComputeWorldBound(self, _prim: object) -> FakeBound:
            return FakeBound()

    assert _world_bbox_info(FakePrim(), FakeBBoxCache()) is None


def test_disabled_rigid_body_is_excluded(tmp_path: Path) -> None:
    usd_path = tmp_path / "disabled_body.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    body = UsdGeom.Xform.Define(stage, "/World/DisabledBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(False)
    _save(stage)

    result = inspect_usd_physics(usd_path, expect_physics=True)

    assert result.summary["rigid_body_count"] == 0
    assert not any(
        finding.prim_path == "/World/DisabledBody" for finding in result.findings
    )


def test_asset_validator_physics_issues_are_consumed(tmp_path: Path) -> None:
    usd_path = tmp_path / "valid_physics.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)
    _save(stage)

    result = inspect_usd_physics(
        usd_path,
        expect_physics=True,
        asset_validator_report={
            "status": "success",
            "issues": [
                {
                    "severity": "warning",
                    "rule": "RigidBodyRule",
                    "category": "Physics",
                    "message": "Synthetic physics validator issue",
                    "at": "/World/Body",
                }
            ],
        },
    )

    assert "physics.asset_validator_issue" in _codes(result)
    finding = result.findings[0]
    assert finding.severity == "warn"
    assert finding.prim_path == "/World/Body"


def test_asset_validator_error_report_is_warned(tmp_path: Path) -> None:
    usd_path = tmp_path / "valid_physics.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)
    _save(stage)

    result = inspect_usd_physics(
        usd_path,
        expect_physics=True,
        asset_validator_report={"status": "error", "error": "validator unavailable"},
    )

    assert "physics.asset_validator_unavailable" in _codes(result)
    warning = next(
        finding
        for finding in result.findings
        if finding.code == "physics.asset_validator_unavailable"
    )
    assert warning.severity == "warn"


def test_asset_validator_mass_failure_is_preserved(tmp_path: Path) -> None:
    usd_path = tmp_path / "valid_physics.usda"
    stage = _new_stage(usd_path)
    _add_scene(stage)
    _add_valid_body(stage)
    _save(stage)

    result = inspect_usd_physics(
        usd_path,
        expect_physics=True,
        asset_validator_report={
            "status": "success",
            "issues": [
                {
                    "severity": "failure",
                    "rule": "MassChecker",
                    "category": "Physics",
                    "message": "Density value is invalid for PhysX.",
                    "at": "/World/Body",
                },
                {
                    "severity": "warning",
                    "rule": "NormalsChecker",
                    "category": "Geometry",
                    "message": "Geometry issue only",
                    "at": "/World/Body/Collider",
                },
            ],
        },
    )

    validator_findings = [
        finding
        for finding in result.findings
        if finding.code == "physics.asset_validator_issue"
    ]
    assert len(validator_findings) == 1
    assert validator_findings[0].severity == "fail"
    assert validator_findings[0].prim_path == "/World/Body"
