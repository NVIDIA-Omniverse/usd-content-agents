# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for `apply_physics` against a real USD asset.

These exercise the full `apply_physics` function on the bundled lightbulb
asset and validate that the output USD has the expected UsdPhysics schemas
authored correctly.

Skipped if `pxr` isn't importable.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr", reason="USD (pxr) not available in this env")
from pxr import Ar, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402

from physics_agent.functions.apply_physics import (  # noqa: E402
    PhysicsAuthoringError,
    _apply_predictions_to_stage,
    _create_physics_material,
    _create_usdz_package,
    _is_runtime_resolved_asset_path,
    apply_physics,
)

LIGHTBULB_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "examples" / "Lightbulb01"
)
LIGHTBULB_USDZ = LIGHTBULB_DIR / "light_bulb_01.usdz"


@pytest.fixture
def lightbulb_usdz() -> Path:
    if not LIGHTBULB_USDZ.exists():
        pytest.skip(f"Lightbulb example asset missing: {LIGHTBULB_USDZ}")
    return LIGHTBULB_USDZ


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _glass_props() -> dict:
    return {
        "density": 2500,
        "estimated_mass_kg": 0.008,
        "static_friction": 0.4,
        "dynamic_friction": 0.3,
        "restitution": 0.5,
    }


def _metal_props() -> dict:
    return {
        "density": 8900,
        "estimated_mass_kg": 0.01,
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.3,
    }


def _mass_scale_warning() -> list[dict]:
    return [
        {
            "code": "mass_scale_suspicious",
            "severity": "warning",
            "message": "synthetic scale warning",
        }
    ]


def test_create_physics_material_defines_looks_scope() -> None:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    _create_physics_material(stage, "/Root/Looks/PhysMat", 0.4, 0.3, 0.2)

    looks = stage.GetPrimAtPath("/Root/Looks")
    assert looks.IsA(UsdGeom.Scope)
    assert stage.GetPrimAtPath("/Root/Looks/PhysMat").IsA(UsdShade.Material)


def _write_instanced_usd(tmp_path: Path) -> Path:
    asset_path = tmp_path / "referenced_model.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    model = UsdGeom.Xform.Define(asset_stage, "/Model")
    asset_stage.SetDefaultPrim(model.GetPrim())
    UsdGeom.Cube.Define(asset_stage, "/Model/Cube")
    asset_stage.GetRootLayer().Save()

    stage_path = tmp_path / "instanced_scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    instance = UsdGeom.Xform.Define(stage, "/World/Inst").GetPrim()
    instance.GetReferences().AddReference(str(asset_path), "/Model")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    return stage_path


def _write_default_prim_instance_usd(
    tmp_path: Path,
    *,
    with_physics_scene: bool = False,
) -> Path:
    asset_path = tmp_path / "referenced_default_model.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    model = UsdGeom.Xform.Define(asset_stage, "/Model")
    asset_stage.SetDefaultPrim(model.GetPrim())
    UsdGeom.Cube.Define(asset_stage, "/Model/Cube")
    if with_physics_scene:
        scene = UsdPhysics.Scene.Define(asset_stage, "/Model/PhysicsScene")
        scene.CreateGravityMagnitudeAttr(123.0)
    asset_stage.GetRootLayer().Save()

    stage_path = tmp_path / "default_prim_instance.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    world.GetPrim().GetReferences().AddReference(str(asset_path), "/Model")
    world.GetPrim().SetInstanceable(True)
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().Save()
    return stage_path


def _write_mass_authored_usd(tmp_path: Path) -> Path:
    stage_path = tmp_path / "mass_authored.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim()
    mass_api = UsdPhysics.MassAPI.Apply(cube)
    mass_api.CreateMassAttr(123.0)
    stage.GetRootLayer().Save()
    return stage_path


def _write_sublayered_usd(
    tmp_path: Path,
    *,
    root_mass: float | None = None,
    weaker_mass: float | None = None,
) -> Path:
    sublayer_path = tmp_path / "payload.usda"
    sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
    sublayer_world = UsdGeom.Xform.Define(sublayer_stage, "/World")
    sublayer_stage.SetDefaultPrim(sublayer_world.GetPrim())
    sublayer_cube = UsdGeom.Cube.Define(sublayer_stage, "/World/Cube").GetPrim()
    if weaker_mass is not None:
        UsdPhysics.MassAPI.Apply(sublayer_cube).CreateMassAttr(weaker_mass)
    sublayer_stage.GetRootLayer().Save()

    root_path = tmp_path / "layered.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("./payload.usda")
    root_world = UsdGeom.Xform.Define(root_stage, "/World")
    root_stage.SetDefaultPrim(root_world.GetPrim())
    if root_mass is not None:
        root_cube = UsdGeom.Cube.Define(root_stage, "/World/Cube").GetPrim()
        UsdPhysics.MassAPI.Apply(root_cube).CreateMassAttr(root_mass)
    root_stage.GetRootLayer().Save()
    return root_path


def _write_parent_relative_sublayered_usd(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "asset"
    shared_dir = tmp_path / "shared"
    asset_dir.mkdir()
    shared_dir.mkdir()

    sublayer_path = shared_dir / "payload.usda"
    sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
    UsdGeom.Xform.Define(sublayer_stage, "/World")
    UsdGeom.Cube.Define(sublayer_stage, "/World/Cube")
    sublayer_stage.GetRootLayer().Save()

    root_path = asset_dir / "layered.usda"
    root_stage = Usd.Stage.CreateNew(str(root_path))
    root_stage.GetRootLayer().subLayerPaths.append("../shared/payload.usda")
    root_world = UsdGeom.Xform.Define(root_stage, "/World")
    root_stage.SetDefaultPrim(root_world.GetPrim())
    root_stage.GetRootLayer().Save()
    return root_path


def _write_articulated_robot_usd(tmp_path: Path) -> Path:
    stage_path = tmp_path / "articulated_robot.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())

    base_link = UsdGeom.Xform.Define(stage, "/Robot/base_link").GetPrim()
    arm_link = UsdGeom.Xform.Define(stage, "/Robot/arm_link").GetPrim()
    for link in (base_link, arm_link):
        UsdPhysics.RigidBodyAPI.Apply(link).CreateRigidBodyEnabledAttr(True)

    UsdGeom.Cube.Define(stage, "/Robot/base_link/visuals/base_mesh")
    UsdGeom.Cube.Define(stage, "/Robot/arm_link/visuals/arm_mesh")

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/base_to_arm_joint")
    joint.CreateBody0Rel().SetTargets([base_link.GetPath()])
    joint.CreateBody1Rel().SetTargets([arm_link.GetPath()])
    stage.GetRootLayer().Save()
    return stage_path


def _enabled_rigid_body_paths(stage: Usd.Stage) -> list[str]:
    paths: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        if enabled is not False:
            paths.append(str(prim.GetPath()))
    return paths


def test_apply_physics_authors_expected_schemas(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Full-pipeline equivalent: applies physics to three prims and verifies
    schemas, attributes, PhysicsScene gravity, material cache, and binding
    purpose are all authored correctly on the output USD.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
            {
                "id": "/light_bulb_01/Geometry/Bulb_Screw_Cap",
                "classification": {"physical_properties": _metal_props()},
            },
            {
                "id": "/light_bulb_01/Geometry/Filament_Big",
                "classification": {"physical_properties": _metal_props()},
            },
        ],
    )

    out = apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    # apply_physics returns an absolute path per its docstring.
    assert Path(out).is_absolute()
    assert Path(out).exists()

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None

    # PhysicsScene created at the default-prim root, with gravity matching Z-up.
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert len(scenes) == 1, f"expected 1 PhysicsScene, got {len(scenes)}"
    scene = UsdPhysics.Scene(scenes[0])
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81, rel=1e-3)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    expected_dir = (0, 0, -1) if up_axis == UsdGeom.Tokens.z else (0, -1, 0)
    gravity_dir = scene.GetGravityDirectionAttr().Get()
    assert tuple(gravity_dir) == expected_dir

    # Each predicted mesh becomes a collider on the asset's rigid body —
    # CollisionAPI + MeshCollisionAPI + MassAPI(density only) +
    # MaterialBindingAPI. NO per-mesh RigidBodyAPI; that lives on the
    # default prim only (the asset's rigid-body parent).
    expected_collider_schemas = {
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsMassAPI",
        "MaterialBindingAPI",
    }
    for prim_path, props in [
        ("/light_bulb_01/Geometry/Bulb_Main", _glass_props()),
        ("/light_bulb_01/Geometry/Bulb_Screw_Cap", _metal_props()),
        ("/light_bulb_01/Geometry/Filament_Big", _metal_props()),
    ]:
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), f"missing prim {prim_path}"
        applied = set(prim.GetAppliedSchemas())
        assert expected_collider_schemas.issubset(applied), (
            f"{prim_path} applied schemas {applied} missing "
            f"{expected_collider_schemas - applied}"
        )
        assert "PhysicsRigidBodyAPI" not in applied, (
            f"{prim_path} is a collider, not a rigid body — RigidBodyAPI "
            "should live on the asset's default prim, not on each mesh."
        )
        mass_api = UsdPhysics.MassAPI(prim)
        # Per-collider mass is NOT authored — mass is aggregated on the body.
        assert not mass_api.GetMassAttr().HasAuthoredValueOpinion(), (
            f"{prim_path} has per-collider mass authored; mass belongs on "
            "the asset's rigid body (default prim), not on each collider."
        )
        # Per-collider density IS preserved so engines can derive inertia.
        assert mass_api.GetDensityAttr().Get() == pytest.approx(
            props["density"], rel=1e-3
        )
        assert (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            == "convexHull"
        )
        # Physics-purpose material binding preserves visual bindings on the prim.
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("physics")
        assert rel, f"{prim_path} has no physics-purpose material binding"
        targets = rel.GetTargets()
        assert len(targets) == 1

    # The asset's rigid body lives on the default prim with the
    # aggregated mass = sum of predicted estimated_mass_kg.
    body = stage.GetDefaultPrim()
    assert body.IsValid()
    body_schemas = set(body.GetAppliedSchemas())
    assert {"PhysicsRigidBodyAPI", "PhysicsMassAPI"}.issubset(body_schemas), (
        f"default prim {body.GetPath()} missing rigid body schemas: {body_schemas}"
    )
    assert UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is True
    expected_aggregate = (
        _glass_props()["estimated_mass_kg"]
        + _metal_props()["estimated_mass_kg"]
        + _metal_props()["estimated_mass_kg"]
    )
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(
        expected_aggregate, rel=1e-3
    )

    # Material cache de-duplicated the two metal prims onto a single PhysMat.
    looks = stage.GetPrimAtPath("/light_bulb_01/Looks")
    assert looks.IsValid()
    phys_mats = [c for c in looks.GetChildren() if c.GetName().startswith("PhysMat_")]
    assert len(phys_mats) == 2, (
        f"expected 2 cached PhysMats (glass + metal), got {len(phys_mats)}: "
        f"{[m.GetName() for m in phys_mats]}"
    )


@pytest.mark.parametrize(
    ("meters_per_unit", "expected_gravity_stage_units"),
    [(None, 981.0), (0.01, 981.0), (0.1, 98.1), (1.0, 9.81)],
)
def test_apply_physics_authors_earth_gravity_in_customer_stage_units(
    tmp_path: Path,
    meters_per_unit: float | None,
    expected_gravity_stage_units: float,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "output.usda"
    predictions = tmp_path / "predictions.jsonl"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Cube")
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.GetRootLayer().Save()
    _write_jsonl(
        predictions,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )

    apply_physics(str(source), str(predictions), str(output))

    result = Usd.Stage.Open(str(output))
    assert result is not None
    assert UsdGeom.StageHasAuthoredMetersPerUnit(result) is (
        meters_per_unit is not None
    )
    resolved_mpu = float(UsdGeom.GetStageMetersPerUnit(result))
    scene_prim = next(prim for prim in result.Traverse() if prim.IsA(UsdPhysics.Scene))
    gravity = float(UsdPhysics.Scene(scene_prim).GetGravityMagnitudeAttr().Get())
    assert gravity == pytest.approx(expected_gravity_stage_units, rel=1e-6)
    assert gravity * resolved_mpu == pytest.approx(9.81, rel=1e-6)


def test_apply_physics_preserves_existing_customer_gravity(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "output.usda"
    predictions = tmp_path / "predictions.jsonl"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.Cube.Define(stage, "/World/Cube")
    customer_scene = UsdPhysics.Scene.Define(stage, "/World/CustomerPhysics")
    customer_scene.CreateGravityMagnitudeAttr(123.0)
    stage.GetRootLayer().Save()
    _write_jsonl(
        predictions,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )

    apply_physics(str(source), str(predictions), str(output))

    result = Usd.Stage.Open(str(output))
    assert result is not None
    scenes = [prim for prim in result.Traverse() if prim.IsA(UsdPhysics.Scene)]
    assert [str(prim.GetPath()) for prim in scenes] == ["/World/CustomerPhysics"]
    gravity = UsdPhysics.Scene(scenes[0]).GetGravityMagnitudeAttr().Get()
    assert gravity == pytest.approx(123.0)


def test_apply_physics_deinstances_default_prim_before_scene_authoring(
    tmp_path: Path,
) -> None:
    source = _write_default_prim_instance_usd(tmp_path)
    output = tmp_path / "output.usda"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _glass_props()},
            }
        ],
    )

    apply_physics(str(source), str(predictions), str(output))

    result = Usd.Stage.Open(str(output))
    assert result is not None
    assert not result.GetDefaultPrim().IsInstanceable()
    assert result.GetPrimAtPath("/World/PhysicsScene").IsA(UsdPhysics.Scene)
    cube = result.GetPrimAtPath("/World/Cube")
    assert not cube.IsInstanceProxy()
    assert cube.HasAPI(UsdPhysics.CollisionAPI)


def test_apply_physics_detects_existing_instance_proxy_scene(tmp_path: Path) -> None:
    source = _write_default_prim_instance_usd(tmp_path, with_physics_scene=True)
    output = tmp_path / "output.usda"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions, [])

    apply_physics(
        str(source),
        str(predictions),
        str(output),
        allow_empty_predictions=True,
        author_rigid_body=False,
    )

    result = Usd.Stage.Open(str(output))
    assert result is not None
    assert result.GetDefaultPrim().IsInstanceable()
    scene_traversal = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    scenes = [
        prim
        for prim in Usd.PrimRange.Stage(result, scene_traversal)
        if prim.IsA(UsdPhysics.Scene)
    ]
    assert [str(prim.GetPath()) for prim in scenes] == ["/World/PhysicsScene"]
    assert scenes[0].IsInstanceProxy()
    assert UsdPhysics.Scene(scenes[0]).GetGravityMagnitudeAttr().Get() == pytest.approx(
        123.0
    )


def test_apply_physics_deinstances_existing_proxy_scene_for_normal_authoring(
    tmp_path: Path,
) -> None:
    source = _write_default_prim_instance_usd(tmp_path, with_physics_scene=True)
    output = tmp_path / "output.usda"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _glass_props()},
            }
        ],
    )

    apply_physics(str(source), str(predictions), str(output))

    result = Usd.Stage.Open(str(output))
    assert result is not None
    assert not result.GetDefaultPrim().IsInstanceable()
    scenes = [prim for prim in result.Traverse() if prim.IsA(UsdPhysics.Scene)]
    assert [str(prim.GetPath()) for prim in scenes] == ["/World/PhysicsScene"]
    assert UsdPhysics.Scene(scenes[0]).GetGravityMagnitudeAttr().Get() == pytest.approx(
        123.0
    )
    cube = result.GetPrimAtPath("/World/Cube")
    assert not cube.IsInstanceProxy()
    assert cube.HasAPI(UsdPhysics.CollisionAPI)


def test_apply_physics_rejects_instance_proxy_default_prim(tmp_path: Path) -> None:
    source = _write_instanced_usd(tmp_path)
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    proxy = stage.GetPrimAtPath("/World/Inst/Cube")
    assert proxy.IsInstanceProxy()

    class _StageWithProxyDefault:
        def GetDefaultPrim(self) -> Usd.Prim:
            return proxy

    with pytest.raises(PhysicsAuthoringError, match="default prim .*instance proxy"):
        _apply_predictions_to_stage(
            _StageWithProxyDefault(),  # type: ignore[arg-type]
            source,
            [],
            "convexHull",
            "classification",
            "skip_mass",
            allow_empty_predictions=True,
            author_rigid_body=False,
        )


def test_apply_physics_rejects_empty_predictions_by_default(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "out" / "light_bulb_01_physics.usda"
    _write_jsonl(predictions_path, [])

    with pytest.raises(PhysicsAuthoringError, match="No predictions"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
        )

    assert not output_usd.exists()
    assert not output_usd.parent.exists()


def test_apply_physics_allows_empty_predictions_when_opted_in(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"
    _write_jsonl(predictions_path, [])

    out = apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        allow_empty_predictions=True,
    )

    assert Path(out) == output_usd.resolve()
    assert output_usd.exists()
    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders == []


def test_apply_physics_allows_all_skipped_predictions_when_opted_in(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {},
            },
        ],
    )

    out = apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        allow_empty_predictions=True,
    )

    assert Path(out) == output_usd.resolve()
    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders == []


def test_apply_physics_rejects_malformed_prediction_json_even_when_empty_allowed(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"
    predictions_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(PhysicsAuthoringError, match="Malformed JSON"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
            allow_empty_predictions=True,
        )

    assert not output_usd.exists()


def test_apply_physics_rejects_malformed_prediction_json_after_valid_record(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "out" / "light_bulb_01_physics.usda"
    valid_prediction = {
        "id": "/RootNode/Looks/Glass",
        "classification": {"physical_properties": _glass_props()},
    }
    predictions_path.write_text(
        json.dumps(valid_prediction) + "\n{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(PhysicsAuthoringError, match="Malformed JSON"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
        )

    assert not output_usd.exists()
    assert not output_usd.parent.exists()


def test_apply_physics_preserves_lightbulb_usdz_package(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """USDZ input should produce a USDZ package without flattening payloads away."""
    predictions_path = tmp_path / "predictions.jsonl"
    output_usdz = tmp_path / "light_bulb_01_physics.usdz"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )

    out = apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usdz),
    )

    assert Path(out) == output_usdz.resolve()
    assert zipfile.is_zipfile(output_usdz)
    with zipfile.ZipFile(output_usdz) as archive:
        members = set(archive.namelist())
    assert "light_bulb_01.usda" in members
    assert "Payload/Contents.usda" in members
    assert "Payload/Geometry.usdc" in members
    assert "Payload/Textures/obs_light_bulb_01_a.png" in members

    stage = Usd.Stage.Open(str(output_usdz))
    assert stage is not None
    payload_only_prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Internal_Chamber")
    assert payload_only_prim.IsValid()
    bulb = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert bulb.IsValid()
    assert "PhysicsCollisionAPI" in bulb.GetAppliedSchemas()
    assert "PhysicsRigidBodyAPI" in stage.GetDefaultPrim().GetAppliedSchemas()


def test_apply_physics_rejects_in_place_output(tmp_path: Path) -> None:
    stage_path = _write_mass_authored_usd(tmp_path)
    before = stage_path.read_bytes()
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, [])

    with pytest.raises(ValueError, match="output_path must differ"):
        apply_physics(
            usd_path=str(stage_path),
            predictions_path=str(predictions_path),
            output_path=str(stage_path),
        )
    assert stage_path.read_bytes() == before


def test_apply_physics_rejects_empty_approved_dependency_roots(
    tmp_path: Path,
) -> None:
    stage_path = _write_mass_authored_usd(tmp_path)

    with pytest.raises(ValueError, match="must not be empty"):
        apply_physics(
            usd_path=str(stage_path),
            predictions_path=str(tmp_path / "predictions.jsonl"),
            output_path=str(tmp_path / "output.usda"),
            approved_dependency_roots=[],
        )


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("filesystem-root", "must not contain filesystem roots"),
        ("missing-root", "must contain existing directories"),
    ],
)
def test_apply_physics_rejects_invalid_approved_dependency_roots(
    tmp_path: Path,
    invalid_kind: str,
    message: str,
) -> None:
    stage_path = _write_mass_authored_usd(tmp_path)
    invalid_root = (
        Path(tmp_path.anchor)
        if invalid_kind == "filesystem-root"
        else tmp_path / "missing-root"
    )

    with pytest.raises(ValueError, match=message):
        apply_physics(
            usd_path=str(stage_path),
            predictions_path=str(tmp_path / "missing-predictions.jsonl"),
            output_path=str(tmp_path / "output.usda"),
            approved_dependency_roots=[invalid_root],
        )


def test_apply_physics_expands_and_forwards_tilde_dependency_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    approved_root = home / "approved"
    approved_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    stage_path = _write_mass_authored_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )
    captured: dict[str, tuple[Path, ...]] = {}

    def capture_export(
        _stage: Usd.Stage,
        _output: Path,
        _skipped_mass_paths: set[str],
        approved_dependency_roots: tuple[Path, ...],
        package_asset_root: Path | None = None,
    ) -> None:
        assert package_asset_root is None
        captured["approved_dependency_roots"] = approved_dependency_roots

    monkeypatch.setattr(
        "physics_agent.functions.apply_physics._export_flattened_stage",
        capture_export,
    )

    apply_physics(
        usd_path=str(stage_path),
        predictions_path=str(predictions_path),
        output_path=str(tmp_path / "output" / "physics.usda"),
        approved_dependency_roots=["~/approved"],
    )

    assert captured["approved_dependency_roots"] == (approved_root.resolve(),)


def test_apply_physics_preserves_existing_same_dir_output_on_failure(
    tmp_path: Path,
) -> None:
    stage_path = _write_mass_authored_usd(tmp_path)
    output_usd = tmp_path / "mass_authored_physics.usda"
    before = b"previous-good-output"
    output_usd.write_bytes(before)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Missing",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="No physics schemas were applied"):
        apply_physics(
            usd_path=str(stage_path),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
        )

    assert output_usd.read_bytes() == before


@pytest.mark.parametrize("symlink_kind", ["leaf", "parent"])
def test_apply_physics_rejects_output_symlink_escape_without_publication(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    stage_path = _write_mass_authored_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _metal_props()},
            },
        ],
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_kind == "leaf":
        requested_parent = tmp_path
        outside_output = outside / "sentinel.usda"
        outside_output.write_bytes(b"outside-must-survive")
        output = requested_parent / "physics.usda"
        output.symlink_to(outside_output)
        message = "symlink USD output"
    else:
        requested_parent = tmp_path / "requested"
        outside_output = outside / "physics.usda"
        outside_output.write_bytes(b"outside-must-survive")
        requested_parent.symlink_to(outside, target_is_directory=True)
        output = requested_parent / "physics.usda"
        message = "symlink or non-directory ancestor"

    with pytest.raises(RuntimeError, match=message):
        apply_physics(
            usd_path=str(stage_path),
            predictions_path=str(predictions_path),
            output_path=str(output),
        )

    assert outside_output.read_bytes() == b"outside-must-survive"
    assert requested_parent.is_symlink() is (symlink_kind == "parent")
    if symlink_kind == "leaf":
        assert output.is_symlink()
        assert not (requested_parent / "physics.usda_assets").exists()
    assert not (outside / "physics.usda_assets").exists()


def test_create_usdz_package_preserves_existing_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_layer = tmp_path / "root.usda"
    root_layer.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "existing.usdz"
    before = b"previous-good-package"
    output.write_bytes(before)

    monkeypatch.setattr(
        "physics_agent.functions.apply_physics.UsdUtils.CreateNewUsdzPackage",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="Failed to create USDZ package"):
        _create_usdz_package(root_layer, output)
    assert output.read_bytes() == before


def test_create_usdz_package_removes_owned_stale_sidecar(tmp_path: Path) -> None:
    from world_understanding.functions.graphics import so_export

    root_layer = tmp_path / "root.usda"
    root_layer.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "output.usdz"
    sidecar = tmp_path / so_export.portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / so_export.PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        so_export.PORTABLE_SIDECAR_MARKER_BYTES
    )
    (sidecar / "stale.png").write_bytes(b"stale")

    _create_usdz_package(root_layer, output)

    assert zipfile.is_zipfile(output)
    assert not sidecar.exists()


def test_create_usdz_package_uses_path_export_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    monkeypatch.setattr(so_export, "_SUPPORTS_DIRECTORY_DESCRIPTORS", False)
    root_layer = tmp_path / "root.usda"
    stage = Usd.Stage.CreateNew(str(root_layer))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()
    output = tmp_path / "output.usdz"

    _create_usdz_package(root_layer, output)

    assert zipfile.is_zipfile(output)
    assert Usd.Stage.Open(str(output)) is not None
    assert not list(tmp_path.glob(".usd_output_*"))


def test_create_usdz_package_refuses_foreign_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    root_layer = tmp_path / "root.usda"
    root_layer.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "output.usdz"
    output.write_bytes(b"previous-good-package")
    sidecar = tmp_path / so_export.portable_sidecar_name(output)
    sidecar.write_bytes(b"foreign-sidecar")
    package_calls = 0

    def create_package(*_args, **_kwargs):
        nonlocal package_calls
        package_calls += 1
        return True

    monkeypatch.setattr(
        "physics_agent.functions.apply_physics.UsdUtils.CreateNewUsdzPackage",
        create_package,
    )

    with pytest.raises(RuntimeError, match="non-exporter-owned USD sidecar"):
        _create_usdz_package(root_layer, output)

    assert package_calls == 0
    assert output.read_bytes() == b"previous-good-package"
    assert sidecar.read_bytes() == b"foreign-sidecar"


def test_apply_physics_uses_path_export_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.graphics import so_export

    monkeypatch.setattr(so_export, "_SUPPORTS_DIRECTORY_DESCRIPTORS", False)
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.GetRootLayer().Save()
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )
    output = tmp_path / "physics" / "result.usda"

    result = apply_physics(str(source), str(predictions), str(output))

    assert Path(result) == output.resolve()
    authored = Usd.Stage.Open(str(output))
    assert authored is not None
    cube = authored.GetPrimAtPath("/World/Cube")
    assert cube.HasAPI(UsdPhysics.CollisionAPI)
    assert cube.HasAPI(UsdPhysics.MassAPI)
    assert not list(output.parent.glob(".result_export_*"))


def test_apply_physics_output_is_self_contained_and_relocatable(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Output must open cleanly without source-sibling files.

    Regression: apply_physics previously exported just the root layer, which
    preserved the source asset's relative payload refs (`./Payload/...`)
    that only resolved against the *source* directory. When written to
    `{working_dir}/physics/` — with no Payload sibling — reopening the
    output produced composition errors and missing prims. The fix flattens
    the stage before export and copies any package-local asset dependencies
    beside the output with rewritten relative paths.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    produce_dir = tmp_path / "physics"
    produce_dir.mkdir()
    output_usd = produce_dir / "light_bulb_01_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    # No source sibling Payload directory is present; any dependency sidecars
    # are generated under the output stem and referenced from the output.
    assert output_usd in list(produce_dir.iterdir())
    assert not (produce_dir / "Payload").exists()

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None

    # A prim that only exists inside the source payload must still be present
    # and valid on reopen — proves the payload was actually composed in.
    payload_only_prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Internal_Chamber")
    assert payload_only_prim.IsValid(), (
        "payload-provided prim should be composed into a self-contained output"
    )

    # And the prim we authored physics onto should carry collider schemas.
    bulb = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert bulb.IsValid()
    bulb_schemas = bulb.GetAppliedSchemas()
    assert "PhysicsCollisionAPI" in bulb_schemas
    # Rigid body is on the default prim, not on each collider.
    assert "PhysicsRigidBodyAPI" not in bulb_schemas
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()


def test_apply_physics_usdz_to_usda_preserves_bare_mdl_asset_paths(
    tmp_path: Path,
) -> None:
    """Bare MDL asset tokens should survive USDZ input to USDA output.

    Omniverse USDZ assets can author shader source assets like
    ``@OmniPBR.mdl@``. These are intentionally unresolved by hosts that do not
    have the Omniverse MDL search path; writing a USDA must preserve the token
    instead of trying to bundle it into a new USDZ package.
    """
    root_usda = tmp_path / "root.usda"
    root_usda.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Cube "Geom"
    {
        rel material:binding = </World/Looks/OmniPBR>
    }

    def Scope "Looks"
    {
        def Material "OmniPBR"
        {
            token outputs:surface.connect = </World/Looks/OmniPBR/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:implementationSource = "sourceAsset"
                asset info:mdl:sourceAsset = @OmniPBR.mdl@
                token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                token outputs:surface
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    input_usdz = tmp_path / "asset.usdz"
    with zipfile.ZipFile(input_usdz, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root_usda, "root.usda")
    assert Usd.Stage.Open(str(input_usdz)) is not None

    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Geom",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )
    output_usda = tmp_path / "asset_physics.usda"

    apply_physics(
        usd_path=str(input_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usda),
    )

    output_text = output_usda.read_text(encoding="utf-8")
    assert "asset info:mdl:sourceAsset = @OmniPBR.mdl@" in output_text
    stage = Usd.Stage.Open(str(output_usda))
    assert stage is not None
    assert "PhysicsRigidBodyAPI" in stage.GetDefaultPrim().GetAppliedSchemas()


def test_apply_physics_usdz_to_usda_rewrites_package_asset_paths(
    tmp_path: Path,
) -> None:
    """USDZ package assets should stay portable after flattening to USDA."""
    root_usda = tmp_path / "root.usda"
    root_usda.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Cube "Geom"
    {
        rel material:binding = </World/Looks/Preview>
    }

    def Scope "Looks"
    {
        def Material "Preview"
        {
            token outputs:surface.connect = </World/Looks/Preview/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:file = @./Textures/diffuse.png@
                token outputs:surface
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    texture = tmp_path / "Textures" / "diffuse.png"
    texture.parent.mkdir()
    texture.write_bytes(b"png")
    input_usdz = tmp_path / "textured.usdz"
    with zipfile.ZipFile(input_usdz, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root_usda, "root.usda")
        archive.write(texture, "Textures/diffuse.png")

    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Geom",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )
    output_usda = tmp_path / "textured_physics.usda"

    apply_physics(
        usd_path=str(input_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usda),
    )

    assets_dir = tmp_path / "textured_physics.usda_assets"
    output_text = output_usda.read_text(encoding="utf-8")
    assert str(tmp_path) not in output_text
    stage = Usd.Stage.Open(str(output_usda))
    assert stage is not None
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Preview/Shader"))
    texture_asset = shader.GetInput("file").Get()
    assert isinstance(texture_asset, Sdf.AssetPath)
    assert texture_asset.path.startswith(f"{assets_dir.name}/")
    assert not Path(texture_asset.path).is_absolute()
    resolved_texture = Path(texture_asset.resolvedPath).resolve()
    assert resolved_texture.is_relative_to(assets_dir.resolve())
    assert resolved_texture.read_bytes() == b"png"
    assert "PhysicsRigidBodyAPI" in stage.GetDefaultPrim().GetAppliedSchemas()

    del stage
    root_usda.unlink()
    input_usdz.unlink()
    shutil.rmtree(texture.parent)
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    delivered_output = Path(shutil.move(str(output_usda), delivery / output_usda.name))
    delivered_assets = Path(shutil.move(str(assets_dir), delivery / assets_dir.name))

    delivered_stage = Usd.Stage.Open(str(delivered_output))
    assert delivered_stage is not None
    delivered_shader = UsdShade.Shader(
        delivered_stage.GetPrimAtPath("/World/Looks/Preview/Shader")
    )
    delivered_texture = delivered_shader.GetInput("file").Get()
    assert (
        Path(delivered_texture.resolvedPath)
        .resolve()
        .is_relative_to(delivered_assets.resolve())
    )
    assert Path(delivered_texture.resolvedPath).read_bytes() == b"png"
    assert "PhysicsRigidBodyAPI" in delivered_stage.GetDefaultPrim().GetAppliedSchemas()


def test_apply_physics_usdz_to_usda_localizes_time_sampled_assets_after_move(
    tmp_path: Path,
) -> None:
    """Time-sampled package textures survive source deletion and delivery move."""
    source_dir = tmp_path / "source"
    textures_dir = source_dir / "Textures"
    textures_dir.mkdir(parents=True)
    root_usda = source_dir / "root.usda"
    root_usda.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Cube "Geom"
    {
        asset inputs:file.timeSamples = {
            0: @./Textures/frame_0.png@,
            1: @./Textures/frame_1.png@,
        }
    }
}
""",
        encoding="utf-8",
    )
    (textures_dir / "frame_0.png").write_bytes(b"frame-zero")
    (textures_dir / "frame_1.png").write_bytes(b"frame-one")
    input_usdz = source_dir / "animated.usdz"
    with zipfile.ZipFile(input_usdz, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root_usda, "root.usda")
        archive.write(textures_dir / "frame_0.png", "Textures/frame_0.png")
        archive.write(textures_dir / "frame_1.png", "Textures/frame_1.png")

    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Geom",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )
    output_dir = tmp_path / "work"
    output_usda = output_dir / "animated_physics.usda"
    apply_physics(
        usd_path=str(input_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usda),
    )

    sidecar = output_dir / "animated_physics.usda_assets"
    assert sidecar.is_dir()
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    delivered_output = Path(shutil.move(str(output_usda), delivery / output_usda.name))
    delivered_sidecar = Path(shutil.move(str(sidecar), delivery / sidecar.name))
    shutil.rmtree(source_dir)

    stage = Usd.Stage.Open(str(delivered_output))
    assert stage is not None
    attr = stage.GetPrimAtPath("/World/Geom").GetAttribute("inputs:file")
    assert attr.GetTimeSamples() == [0.0, 1.0]
    for time_code, expected in ((0.0, b"frame-zero"), (1.0, b"frame-one")):
        asset = attr.Get(time_code)
        assert asset.path.startswith(f"{delivered_sidecar.name}/")
        resolved = Ar.GetResolver().Resolve(asset.resolvedPath)
        resolver_asset = Ar.GetResolver().OpenAsset(resolved)
        assert resolver_asset is not None
        assert bytes(resolver_asset.GetBuffer()) == expected


def test_apply_physics_usdz_to_usda_rejects_unresolved_package_asset(
    tmp_path: Path,
) -> None:
    """A non-runtime package dependency must fail closed instead of dangling."""
    root_usda = tmp_path / "root.usda"
    root_usda.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Cube "Geom"
    {
        asset inputs:file = @../missing.png@
    }
}
""",
        encoding="utf-8",
    )
    input_usdz = tmp_path / "dangling.usdz"
    with zipfile.ZipFile(input_usdz, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root_usda, "root.usda")

    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Geom",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )
    output_usda = tmp_path / "dangling_physics.usda"

    with pytest.raises(RuntimeError, match="unresolved asset"):
        apply_physics(
            usd_path=str(input_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usda),
        )

    assert not output_usda.exists()
    assert not (tmp_path / "dangling_physics.usda_assets").exists()


def test_apply_physics_respects_custom_output_key(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """When `predict.output_key` != 'classification', apply_physics should read
    physical_properties from the configured key rather than silently skipping.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                # Stored under "analysis" — default "classification" would miss it
                "analysis": {"physical_properties": _glass_props()},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        output_key="analysis",
    )

    stage = Usd.Stage.Open(str(output_usd))
    prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert prim.IsValid()
    # Per-collider schemas only; rigid body lives on the default prim.
    assert "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
    assert "PhysicsRigidBodyAPI" not in prim.GetAppliedSchemas()
    body = stage.GetDefaultPrim()
    assert UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is True
    # Single prediction → aggregate mass equals that prediction's mass.
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(
        _glass_props()["estimated_mass_kg"], rel=1e-3
    )


def test_apply_physics_accepts_one_nested_output_key_wrapper(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Back-compat for predictions saved as classification.classification."""

    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {
                    "classification": {
                        "component_type": "optical",
                        "physical_properties": _glass_props(),
                    }
                },
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        output_key="classification",
    )

    stage = Usd.Stage.Open(str(output_usd))
    prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert prim.IsValid()
    # The collider is on the predicted mesh, but RigidBodyAPI lives on the
    # default prim (asset anchor) per the apply_physics contract.
    assert "PhysicsCollisionAPI" in prim.GetAppliedSchemas()
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    assert UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Get() is True


def test_apply_physics_default_policy_omits_suspicious_mass(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"
    props = {
        "density": 2700,
        "estimated_mass_kg": 25000,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.3,
    }

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": props},
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    # Per-collider density still authored on the predicted mesh; mass
    # never authored at the collider level.
    prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    prim_mass_api = UsdPhysics.MassAPI(prim)
    assert "PhysicsMassAPI" in prim.GetAppliedSchemas()
    assert not prim_mass_api.GetMassAttr().HasAuthoredValueOpinion()
    assert prim_mass_api.GetDensityAttr().Get() == pytest.approx(2700, rel=1e-3)
    # Body's aggregate mass blocked under skip_mass when any prediction
    # is suspicious — engine derives it from per-collider density × volume.
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    body_mass_api = UsdPhysics.MassAPI(body)
    assert not body_mass_api.GetMassAttr().HasAuthoredValueOpinion()


def test_apply_physics_skip_mass_policy_removes_existing_mass(
    tmp_path: Path,
) -> None:
    usd_path = _write_mass_authored_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "mass_authored_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "density": 2700,
                        "estimated_mass_kg": 25000,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                        "restitution": 0.3,
                    }
                },
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    apply_physics(
        usd_path=str(usd_path),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    # Pre-existing per-collider mass on /World/Cube is blocked under
    # skip_mass + suspicious — defensive cleanup so stale mass specs
    # in the input don't confuse downstream consumers.
    cube = stage.GetPrimAtPath("/World/Cube")
    cube_mass_api = UsdPhysics.MassAPI(cube)
    assert not cube_mass_api.GetMassAttr().HasAuthoredValueOpinion()
    assert cube_mass_api.GetDensityAttr().Get() == pytest.approx(2700, rel=1e-3)
    # Body /World has the rigid body; aggregate mass blocked.
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    body_mass_api = UsdPhysics.MassAPI(body)
    assert not body_mass_api.GetMassAttr().HasAuthoredValueOpinion()


def test_apply_physics_skip_mass_policy_blocks_weaker_mass(
    tmp_path: Path,
) -> None:
    usd_path = _write_sublayered_usd(
        tmp_path,
        root_mass=123.0,
        weaker_mass=1.0,
    )
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "layered_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "density": 2700,
                        "estimated_mass_kg": 25000,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                        "restitution": 0.3,
                    }
                },
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    apply_physics(
        usd_path=str(usd_path),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    cube = stage.GetPrimAtPath("/World/Cube")
    cube_mass_api = UsdPhysics.MassAPI(cube)
    assert cube_mass_api.GetMassAttr().Get() is None
    assert cube_mass_api.GetDensityAttr().Get() == pytest.approx(2700, rel=1e-3)


def test_apply_physics_same_suffix_different_directory_flattens_dependencies(
    tmp_path: Path,
) -> None:
    usd_path = _write_sublayered_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "out" / "layered_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "density": 2700,
                        "estimated_mass_kg": 1.0,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                        "restitution": 0.3,
                    }
                },
            },
        ],
    )

    apply_physics(
        usd_path=str(usd_path),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None
    assert stage.GetRootLayer().subLayerPaths == []
    cube = stage.GetPrimAtPath("/World/Cube")
    assert cube.IsValid()
    assert "PhysicsCollisionAPI" in cube.GetAppliedSchemas()


def test_apply_physics_localizes_optimizer_package_texture_after_relocation(
    tmp_path: Path,
) -> None:
    """Optimizer output must not retain dependencies on the upload tree."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    package_root = upload_dir / "root.usda"
    package_root.write_text(
        '#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    texture = upload_dir / "albedo.png"
    texture.write_bytes(b"generated-texture")
    roughness = upload_dir / "roughness.png"
    roughness.write_bytes(b"roughness-texture")
    source_package = upload_dir / "Body_textured.usdz"
    with zipfile.ZipFile(
        source_package,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.write(package_root, "root.usda")
        archive.write(texture, "0/albedo.png")

    optimizer_dir = tmp_path / "cache" / "optimize_usd"
    optimizer_dir.mkdir(parents=True)
    optimized_usd = optimizer_dir / "scene_optimized.usdc"
    optimized_stage = Usd.Stage.CreateNew(str(optimized_usd))
    world = UsdGeom.Xform.Define(optimized_stage, "/World")
    optimized_stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(optimized_stage, "/World/Geom")
    looks = UsdGeom.Scope.Define(optimized_stage, "/World/Looks")
    material = UsdShade.Material.Define(
        optimized_stage,
        looks.GetPath().AppendChild("Paint"),
    )
    shader = UsdShade.Shader.Define(
        optimized_stage,
        material.GetPath().AppendChild("Texture"),
    )
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(f"{source_package}[0/albedo.png]")
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(roughness))
    )
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    shader.GetPrim().CreateAttribute(
        "info:remote:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("https://example.invalid/runtime.png"))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    optimized_stage.GetRootLayer().Save()

    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Geom",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )
    physics_dir = tmp_path / "cache" / "physics"
    output_usd = physics_dir / "scene_physics.usda"

    apply_physics(
        usd_path=str(optimized_usd),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        approved_dependency_roots=[tmp_path],
    )

    output_stage = Usd.Stage.Open(str(output_usd))
    output_asset = (
        UsdShade.Shader(output_stage.GetPrimAtPath("/World/Looks/Paint/Texture"))
        .GetInput("file")
        .Get()
    )
    assert output_asset.path.startswith("scene_physics.usda_assets/")
    assert str(upload_dir) not in output_asset.path

    del optimized_stage, output_stage
    shutil.rmtree(upload_dir)
    shutil.rmtree(optimizer_dir)
    downloaded_dir = tmp_path / "downloaded"
    physics_dir.replace(downloaded_dir)
    moved_output = downloaded_dir / output_usd.name

    moved_stage = Usd.Stage.Open(str(moved_output))
    assert moved_stage is not None
    assert (
        "PhysicsCollisionAPI"
        in moved_stage.GetPrimAtPath("/World/Geom").GetAppliedSchemas()
    )
    assert "PhysicsRigidBodyAPI" in moved_stage.GetDefaultPrim().GetAppliedSchemas()

    moved_shader = UsdShade.Shader(
        moved_stage.GetPrimAtPath("/World/Looks/Paint/Texture")
    )
    moved_asset = moved_shader.GetInput("file").Get()
    moved_roughness = moved_shader.GetInput("roughness").Get()
    assert moved_roughness.path.startswith("scene_physics.usda_assets/")
    assert Path(moved_roughness.resolvedPath).read_bytes() == b"roughness-texture"
    mdl_asset = moved_shader.GetPrim().GetAttribute("info:mdl:sourceAsset").Get()
    assert mdl_asset.path == "OmniPBR.mdl"
    remote_asset = moved_shader.GetPrim().GetAttribute("info:remote:sourceAsset").Get()
    assert remote_asset.path == "https://example.invalid/runtime.png"
    assert Path(moved_asset.resolvedPath).read_bytes() == b"generated-texture"

    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(moved_output)))
    assert [
        path for path in unresolved if not _is_runtime_resolved_asset_path(path)
    ] == []


def test_apply_physics_non_usdz_to_usdz_flattens_parent_relative_dependencies(
    tmp_path: Path,
) -> None:
    usd_path = _write_parent_relative_sublayered_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usdz = tmp_path / "out" / "layered_physics.usdz"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {
                    "physical_properties": {
                        "density": 2700,
                        "estimated_mass_kg": 1.0,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                        "restitution": 0.3,
                    }
                },
            },
        ],
    )

    apply_physics(
        usd_path=str(usd_path),
        predictions_path=str(predictions_path),
        output_path=str(output_usdz),
        approved_dependency_roots=[tmp_path],
    )

    assert zipfile.is_zipfile(output_usdz)
    with zipfile.ZipFile(output_usdz) as archive:
        assert f"{output_usdz.stem}.usda" in archive.namelist()

    stage = Usd.Stage.Open(str(output_usdz))
    assert stage is not None
    assert stage.GetRootLayer().subLayerPaths == []
    cube = stage.GetPrimAtPath("/World/Cube")
    assert cube.IsValid()
    assert "PhysicsCollisionAPI" in cube.GetAppliedSchemas()


def test_apply_physics_warn_policy_authors_suspicious_mass(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        mass_scale_policy="warn",
    )

    stage = Usd.Stage.Open(str(output_usd))
    # warn policy includes the suspicious prediction in the body's
    # aggregate mass; per-collider mass remains unauthored under the
    # monolithic topology.
    body = stage.GetDefaultPrim()
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(
        _glass_props()["estimated_mass_kg"], rel=1e-3
    )


def test_apply_physics_warn_policy_replaces_existing_mass(
    tmp_path: Path,
) -> None:
    usd_path = _write_mass_authored_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "mass_authored_warn.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Cube",
                "classification": {"physical_properties": _glass_props()},
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    apply_physics(
        usd_path=str(usd_path),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        mass_scale_policy="warn",
    )

    stage = Usd.Stage.Open(str(output_usd))
    # warn policy authors the predicted mass on the asset's body
    # (default prim /World), aggregating from all colliders.
    body = stage.GetDefaultPrim()
    assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(
        _glass_props()["estimated_mass_kg"], rel=1e-3
    )


def test_apply_physics_fail_policy_rejects_suspicious_mass(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
                "quality_warnings": _mass_scale_warning(),
            },
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="mass/scale QA warning"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
            mass_scale_policy="fail",
        )

    assert not output_usd.exists()


def test_apply_physics_rejects_invalid_mass_scale_policy(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="mass_scale_policy"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(tmp_path / "out.usda"),
            mass_scale_policy="bad",
        )

    with pytest.raises(ValueError, match="allow_empty_predictions"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(tmp_path / "out.usda"),
            allow_empty_predictions="yes",  # type: ignore[arg-type]
        )


def test_apply_physics_default_output_key_on_mismatched_data_fails_loudly(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Sanity companion to the custom-output-key test: when data is under
    `analysis` but we read with the default `classification`, fail instead
    of exporting a USD with no authored physics schemas.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "analysis": {"physical_properties": _glass_props()},
            },
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="No physics schemas were applied"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
            # default output_key="classification" — mismatched, so no physics applies
        )

    assert not output_usd.exists()


def test_apply_physics_skips_nonexistent_prims_and_missing_properties(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """A prediction targeting a prim that isn't on the stage, or one whose
    classification lacks `physical_properties`, should be skipped (warned in
    logs, no schema authored) without failing the run.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            # Valid
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
            # Missing prim
            {
                "id": "/light_bulb_01/Geometry/Nonexistent",
                "classification": {"physical_properties": _glass_props()},
            },
            # Missing physical_properties
            {
                "id": "/light_bulb_01/Geometry/Internal_Chamber",
                "classification": {},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    # Valid predicted prim got collider schemas
    bulb = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert "PhysicsCollisionAPI" in bulb.GetAppliedSchemas()
    # Body lives on the default prim regardless of how many predictions
    # were skipped (one valid prediction is enough).
    body = stage.GetDefaultPrim()
    assert "PhysicsRigidBodyAPI" in body.GetAppliedSchemas()
    # Skipped prim stayed clean — no physics schemas, including no collider.
    internal = stage.GetPrimAtPath("/light_bulb_01/Geometry/Internal_Chamber")
    assert internal.IsValid()
    internal_schemas = internal.GetAppliedSchemas()
    assert "PhysicsRigidBodyAPI" not in internal_schemas
    assert "PhysicsCollisionAPI" not in internal_schemas


def test_apply_physics_preserves_existing_articulated_rigid_body_hierarchy(
    tmp_path: Path,
) -> None:
    """Predictions for visual meshes under robot links must not create nested
    enabled rigid bodies. Link-level rigid bodies and joints should be preserved,
    while visual meshes can still receive collider/material properties.
    """
    robot_usd = _write_articulated_robot_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "articulated_robot_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/Robot/base_link/visuals/base_mesh",
                "classification": {"physical_properties": _metal_props()},
            },
            {
                "id": "/Robot/arm_link/visuals/arm_mesh",
                "classification": {"physical_properties": _metal_props()},
            },
        ],
    )

    apply_physics(
        usd_path=str(robot_usd),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None
    assert set(_enabled_rigid_body_paths(stage)) == {
        "/Robot/base_link",
        "/Robot/arm_link",
    }
    assert stage.GetPrimAtPath("/Robot/base_to_arm_joint").IsValid()

    for mesh_path in (
        "/Robot/base_link/visuals/base_mesh",
        "/Robot/arm_link/visuals/arm_mesh",
    ):
        mesh = stage.GetPrimAtPath(mesh_path)
        assert mesh.IsValid()
        assert "PhysicsRigidBodyAPI" not in mesh.GetAppliedSchemas()
        assert "PhysicsCollisionAPI" in mesh.GetAppliedSchemas()
        assert "PhysicsMassAPI" in mesh.GetAppliedSchemas()
        assert "MaterialBindingAPI" in mesh.GetAppliedSchemas()


def test_apply_physics_preserves_articulation_when_prediction_targets_container(
    tmp_path: Path,
) -> None:
    """Predictions above robot links must not create a parent rigid body.

    The container can still carry collision, mass, and material properties,
    but the existing link-level rigid bodies and joints remain the articulation.
    """
    robot_usd = _write_articulated_robot_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "articulated_robot_container_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/Robot",
                "classification": {"physical_properties": _metal_props()},
            }
        ],
    )

    apply_physics(
        usd_path=str(robot_usd),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None
    assert set(_enabled_rigid_body_paths(stage)) == {
        "/Robot/base_link",
        "/Robot/arm_link",
    }
    assert stage.GetPrimAtPath("/Robot/base_to_arm_joint").IsValid()

    robot = stage.GetPrimAtPath("/Robot")
    assert robot.IsValid()
    assert "PhysicsRigidBodyAPI" not in robot.GetAppliedSchemas()
    assert "PhysicsCollisionAPI" in robot.GetAppliedSchemas()
    assert "PhysicsMassAPI" in robot.GetAppliedSchemas()
    assert "MaterialBindingAPI" in robot.GetAppliedSchemas()


def test_apply_physics_fails_on_instance_proxy_targets(tmp_path: Path) -> None:
    """Direct authoring onto instance proxy descendants should not export a
    partial-success USD. Optimized pipeline runs must author onto the
    deinstanced USD instead.
    """
    instanced_usd = _write_instanced_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "instanced_scene_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Inst/Cube",
                "classification": {"physical_properties": _glass_props()},
            }
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="instance proxy"):
        apply_physics(
            usd_path=str(instanced_usd),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
        )

    assert not output_usd.exists()
