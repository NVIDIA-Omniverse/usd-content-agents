# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from world_understanding.functions.physics import physics_topology as topology_module
from world_understanding.functions.physics.physics_topology import (
    PhysicsTopologyPlanError,
    apply_physics_topology_plan,
    inspect_physics_components,
    inspect_physics_topology,
    sha256_file,
)


def _write_nested_physics_asset(path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    asset = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(asset)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body").GetPrim()
    inner = UsdGeom.Xform.Define(stage, "/Asset/Body/Inner").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(inner).CreateRigidBodyEnabledAttr(True)

    visual = UsdGeom.Mesh.Define(stage, "/Asset/Body/Inner/Visual")
    visual.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(-1, -1, 0), Gf.Vec3f(1, -1, 0), Gf.Vec3f(0, 1, 1)])
    )
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])

    collider = UsdGeom.Cube.Define(stage, "/Asset/Body/Inner/Collision")
    collider.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    collider.CreateDisplayOpacityAttr([0.0])

    helper = UsdGeom.Cube.Define(stage, "/Asset/Body/Inner/reg_bbox")
    helper.CreateSizeAttr(2.0)
    helper.CreateDisplayOpacityAttr([0.0])

    root_joint = UsdPhysics.FixedJoint.Define(stage, "/Asset/Body/RootFixedJoint")
    root_joint.CreateBody0Rel().SetTargets([asset.GetPath()])
    root_joint.CreateBody1Rel().SetTargets([body.GetPath()])
    inner_joint = UsdPhysics.FixedJoint.Define(
        stage, "/Asset/Body/Inner/InnerFixedJoint"
    )
    inner_joint.CreateBody0Rel().SetTargets([body.GetPath()])
    inner_joint.CreateBody1Rel().SetTargets([inner.GetPath()])
    stage.GetRootLayer().Save()


def _write_articulation_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    base = UsdGeom.Xform.Define(stage, "/World/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base).CreateRigidBodyEnabledAttr(True)
    collider = UsdGeom.Cube.Define(stage, "/World/Robot/base/Collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    stage.GetRootLayer().Save()


def _write_articulation_asset_with_external_fixed_joint(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    base = UsdGeom.Xform.Define(stage, "/World/Robot/base").GetPrim()
    loose = UsdGeom.Xform.Define(stage, "/World/Loose").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(loose).CreateRigidBodyEnabledAttr(True)
    base_collider = UsdGeom.Cube.Define(stage, "/World/Robot/base/Collision")
    loose_collider = UsdGeom.Cube.Define(stage, "/World/Loose/Collision")
    UsdPhysics.CollisionAPI.Apply(base_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    UsdPhysics.CollisionAPI.Apply(loose_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Joints/ExternalFixedJoint")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([loose.GetPath()])
    stage.GetRootLayer().Save()


def _write_fixed_joint_with_child_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    left = UsdGeom.Xform.Define(stage, "/World/Left").GetPrim()
    right = UsdGeom.Xform.Define(stage, "/World/Right").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(left).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(right).CreateRigidBodyEnabledAttr(True)
    left_collider = UsdGeom.Cube.Define(stage, "/World/Left/Collision")
    right_collider = UsdGeom.Cube.Define(stage, "/World/Right/Collision")
    UsdPhysics.CollisionAPI.Apply(left_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    UsdPhysics.CollisionAPI.Apply(right_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/FixedJoint")
    joint.CreateBody0Rel().SetTargets([left.GetPath()])
    joint.CreateBody1Rel().SetTargets([right.GetPath()])
    UsdGeom.Cube.Define(stage, "/World/FixedJoint/DebugVisual")
    stage.GetRootLayer().Save()


def _write_body_with_unowned_collider_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    body_collider = UsdGeom.Cube.Define(stage, "/World/Body/Collision")
    UsdPhysics.CollisionAPI.Apply(body_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)
    stage.GetRootLayer().Save()


def _write_body_with_static_compound_collider_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    body_collider = UsdGeom.Cube.Define(stage, "/World/Body/Collision")
    UsdPhysics.CollisionAPI.Apply(body_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    static_root = UsdGeom.Xform.Define(stage, "/World/StaticCompound").GetPrim()
    UsdPhysics.CollisionAPI.Apply(static_root).CreateCollisionEnabledAttr(True)
    UsdGeom.Cube.Define(stage, "/World/StaticCompound/Visual")
    stage.GetRootLayer().Save()


def _write_static_scoped_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    material = UsdShade.Material.Define(stage, "/World/Looks/Plastic")
    scoped = UsdGeom.Cube.Define(stage, "/World/Scoped/Visual")
    UsdShade.MaterialBindingAPI(scoped.GetPrim()).Bind(material)
    UsdPhysics.CollisionAPI.Apply(scoped.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdGeom.Cube.Define(stage, "/World/Outside")
    stage.GetRootLayer().Save()


def _write_disabled_joint_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    left = UsdGeom.Xform.Define(stage, "/World/Left").GetPrim()
    right = UsdGeom.Xform.Define(stage, "/World/Right").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(left).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(right).CreateRigidBodyEnabledAttr(True)
    left_collider = UsdGeom.Cube.Define(stage, "/World/Left/Collision")
    right_collider = UsdGeom.Cube.Define(stage, "/World/Right/Collision")
    UsdPhysics.CollisionAPI.Apply(left_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    UsdPhysics.CollisionAPI.Apply(right_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/DisabledFixedJoint")
    joint.CreateBody0Rel().SetTargets([left.GetPath()])
    joint.CreateBody1Rel().SetTargets([right.GetPath()])
    joint.CreateJointEnabledAttr(False)
    stage.GetRootLayer().Save()


def _write_non_fixed_joint_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    left = UsdGeom.Xform.Define(stage, "/World/Left").GetPrim()
    right = UsdGeom.Xform.Define(stage, "/World/Right").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(left).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(right).CreateRigidBodyEnabledAttr(True)
    left_collider = UsdGeom.Cube.Define(stage, "/World/Left/Collision")
    right_collider = UsdGeom.Cube.Define(stage, "/World/Right/Collision")
    UsdPhysics.CollisionAPI.Apply(left_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    UsdPhysics.CollisionAPI.Apply(right_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
    joint.CreateBody0Rel().SetTargets([left.GetPath()])
    joint.CreateBody1Rel().SetTargets([right.GetPath()])
    stage.GetRootLayer().Save()


def _write_body_with_unowned_visual_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    body_collider = UsdGeom.Cube.Define(stage, "/World/Body/Collision")
    UsdPhysics.CollisionAPI.Apply(body_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    UsdGeom.Cube.Define(stage, "/World/LooseVisual")
    stage.GetRootLayer().Save()


def _write_instanceable_target_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    target = UsdGeom.Xform.Define(stage, "/World/Target").GetPrim()
    target.SetInstanceable(True)
    stage.GetRootLayer().Save()


def _write_scope_target_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/ScopeTarget")
    stage.GetRootLayer().Save()


def _write_inherited_guide_visual_asset(path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    guide = UsdGeom.Xform.Define(stage, "/World/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Guide/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(-1, -1, 0), Gf.Vec3f(1, -1, 0), Gf.Vec3f(0, 1, 1)])
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()


def _write_scoped_ancestor_body_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.RigidBodyAPI.Apply(world).CreateRigidBodyEnabledAttr(True)
    geometry = UsdGeom.Xform.Define(stage, "/World/Geometry")
    collider = UsdGeom.Cube.Define(stage, "/World/Geometry/Collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Geometry/FixedJoint")
    joint.CreateBody0Rel().SetTargets([world.GetPath()])
    joint.CreateBody1Rel().SetTargets([geometry.GetPath()])
    stage.GetRootLayer().Save()


def test_component_inspection_separates_visual_collider_and_helper_roles(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    _write_nested_physics_asset(asset)

    topology = inspect_physics_topology(asset)
    result = inspect_physics_components(asset)

    assert topology["enabled_rigid_body_count"] == 2
    assert topology["enabled_collider_count"] == 1
    assert {finding["code"] for finding in topology["findings"]} >= {
        "nested_enabled_rigid_body",
        "fixed_joint_to_non_rigid_root",
        "rigid_body_without_collider",
    }
    assert result["component_count"] == 1
    component = result["components"][0]
    assert component["visual_evidence_paths"] == ["/Asset/Body/Inner/Visual"]
    assert component["collider_paths"] == ["/Asset/Body/Inner/Collision"]
    assert component["helper_paths"] == ["/Asset/Body/Inner/reg_bbox"]
    assert component["rigid_body_paths"] == ["/Asset/Body", "/Asset/Body/Inner"]
    assert "fixed_joint_to_non_rigid_root" in component["topology_findings"]


def test_component_inspection_honors_inherited_guide_purpose(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "guide.usda"
    _write_inherited_guide_visual_asset(asset)

    result = inspect_physics_components(asset)

    assert result["component_count"] == 1
    component = result["components"][0]
    assert component["visual_evidence_paths"] == []
    assert component["helper_paths"] == ["/World/Guide/Mesh"]


def test_scoped_component_inspection_preserves_ancestor_body_owner(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "scoped_ancestor.usda"
    _write_scoped_ancestor_body_asset(asset)

    topology = inspect_physics_topology(asset, root_prim_path="/World/Geometry")
    result = inspect_physics_components(asset, root_prim_path="/World/Geometry")

    assert topology["colliders"][0]["owner_rigid_body_path"] == "/World"
    component = result["components"][0]
    assert component["body_root_path"] == "/World"
    assert component["rigid_body_paths"] == ["/World"]
    assert component["collider_paths"] == ["/World/Geometry/Collision"]


def test_ancestor_rigid_body_lookup_handles_missing_scope(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    asset = tmp_path / "scoped_ancestor.usda"
    _write_scoped_ancestor_body_asset(asset)

    stage = Usd.Stage.Open(str(asset))

    assert (
        topology_module._ancestor_enabled_rigid_body_paths(
            stage,
            "/World/Missing",
            UsdPhysics,
        )
        == set()
    )


def test_sha256_file_hashes_plain_file_bytes(tmp_path: Path) -> None:
    plain = tmp_path / "payload.txt"
    plain.write_text("plain fixture\n", encoding="utf-8")

    assert (
        sha256_file(plain) == f"sha256:{hashlib.sha256(plain.read_bytes()).hexdigest()}"
    )


def test_sha256_file_tracks_composed_usd_dependencies(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency.usda"
    root = tmp_path / "root.usda"
    _write_nested_physics_asset(dependency)
    root.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Asset"\n    subLayers = [ @dependency.usda@ ]\n)\n',
        encoding="utf-8",
    )
    original_digest = sha256_file(root)

    with dependency.open("a", encoding="utf-8") as stream:
        stream.write('\ndef Xform "DependencyChange" {}\n')

    assert sha256_file(root) != original_digest
    with pytest.raises(PhysicsTopologyPlanError, match="digest mismatch"):
        apply_physics_topology_plan(
            input_usd_path=root,
            output_usd_path=tmp_path / "stale.usda",
            expected_source_digest=original_digest,
            mobility_intent="movable",
            operations=[],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )


def test_component_inspection_keeps_unowned_colliders_separate(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "unowned.usda"
    _write_body_with_unowned_collider_asset(asset)

    result = inspect_physics_components(asset)

    assert result["component_count"] == 2
    by_root = {
        component["body_root_path"]: component for component in result["components"]
    }
    assert by_root["/World/Body"]["component_role"] == "body"
    assert by_root["/World/Body"]["collider_paths"] == ["/World/Body/Collision"]
    assert by_root["/World/Ground"]["component_role"] == "unowned_static"
    assert by_root["/World/Ground"]["collider_paths"] == ["/World/Ground"]
    assert by_root["/World/Ground"]["rigid_body_paths"] == []


def test_component_inspection_assigns_static_descendants_to_ancestor_collider(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "static_compound.usda"
    _write_body_with_static_compound_collider_asset(asset)

    result = inspect_physics_components(asset)

    by_root = {
        component["body_root_path"]: component for component in result["components"]
    }
    assert result["component_count"] == 2
    assert by_root["/World/StaticCompound"]["component_role"] == "unowned_static"
    assert by_root["/World/StaticCompound"]["collider_paths"] == [
        "/World/StaticCompound"
    ]
    assert by_root["/World/StaticCompound"]["visual_evidence_paths"] == [
        "/World/StaticCompound/Visual"
    ]


def test_component_inspection_scopes_static_collider_and_material_evidence(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "static_scoped.usda"
    _write_static_scoped_asset(asset)

    result = inspect_physics_components(asset, root_prim_path="/World/Scoped")

    assert result["component_count"] == 1
    component = result["components"][0]
    assert component["body_root_path"] == "/World/Scoped"
    assert component["visual_evidence_paths"] == ["/World/Scoped/Visual"]
    assert component["collider_paths"] == ["/World/Scoped/Visual"]
    assert component["material_evidence"] == [
        {
            "prim_path": "/World/Scoped/Visual",
            "material_path": "/World/Looks/Plastic",
            "material_name": "Plastic",
        }
    ]


def test_component_inspection_keeps_disabled_joint_bodies_separate(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "disabled_joint.usda"
    _write_disabled_joint_asset(asset)

    result = inspect_physics_components(asset)

    assert result["component_count"] == 2
    assert sorted(
        component["body_root_path"] for component in result["components"]
    ) == ["/World/Left", "/World/Right"]


def test_component_inspection_keeps_joint_connected_bodies_separate(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "hinge.usda"
    _write_non_fixed_joint_asset(asset)

    result = inspect_physics_components(asset)

    by_root = {
        component["body_root_path"]: component for component in result["components"]
    }
    assert result["component_count"] == 2
    assert sorted(by_root) == ["/World/Left", "/World/Right"]
    assert by_root["/World/Left"]["rigid_body_paths"] == ["/World/Left"]
    assert by_root["/World/Right"]["rigid_body_paths"] == ["/World/Right"]
    assert by_root["/World/Left"]["joint_paths"] == ["/World/Hinge"]
    assert by_root["/World/Right"]["joint_paths"] == ["/World/Hinge"]


def test_component_inspection_creates_unowned_visual_component_with_body_groups(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "unowned_visual.usda"
    _write_body_with_unowned_visual_asset(asset)

    result = inspect_physics_components(asset)

    by_root = {
        component["body_root_path"]: component for component in result["components"]
    }
    assert by_root["/World/LooseVisual"]["component_role"] == "unowned_static"
    assert by_root["/World/LooseVisual"]["visual_evidence_paths"] == [
        "/World/LooseVisual"
    ]


def test_topology_inspection_root_scope_includes_stage_descendants(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    _write_nested_physics_asset(asset)

    topology = inspect_physics_topology(asset, root_prim_path="/")

    assert topology["enabled_rigid_body_count"] == 2
    assert topology["enabled_collider_count"] == 1


def test_topology_plan_writes_verified_derivative_without_mutating_source(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)
    source_bytes = asset.read_bytes()

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="movable",
        operations=[
            {
                "op": "remove_rigid_body_api",
                "prim_path": "/Asset/Body/Inner",
            },
            {
                "op": "remove_fixed_joint",
                "prim_path": "/Asset/Body/RootFixedJoint",
            },
            {
                "op": "remove_fixed_joint",
                "prim_path": "/Asset/Body/Inner/InnerFixedJoint",
            },
            {"op": "ensure_rigid_body_api", "prim_path": "/Asset/Body"},
        ],
        invariants={
            "enabled_collider_count": 1,
            "reject_articulation_changes": True,
        },
    )

    assert asset.read_bytes() == source_bytes
    assert output.is_file()
    assert report["before"]["enabled_rigid_body_count"] == 2
    assert report["after"]["rigid_body_paths"] == ["/Asset/Body"]
    assert report["after"]["enabled_collider_count"] == 1
    assert report["after"]["joints"] == []
    assert report["after"]["findings"] == []
    assert report["after"]["source_digest"] == report["output_digest"]
    assert report["after_components"]["source_digest"] == report["output_digest"]


def test_topology_plan_deinstances_instanceable_targets(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    asset = tmp_path / "instanceable.usda"
    output = tmp_path / "prepared.usda"
    _write_instanceable_target_asset(asset)

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=[
            {
                "op": "ensure_rigid_body_api",
                "prim_path": "/World/Target",
            }
        ],
        invariants={
            "enabled_collider_count": 0,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    target = prepared.GetPrimAtPath("/World/Target")
    assert report["applied_operations"] == [
        {"op": "ensure_rigid_body_api", "prim_path": "/World/Target"}
    ]
    assert not target.IsInstanceable()


def test_topology_plan_rejects_non_xformable_rigid_body_target(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "scope.usda"
    _write_scope_target_asset(asset)

    with pytest.raises(PhysicsTopologyPlanError, match="Xformable"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/World/ScopeTarget",
                }
            ],
            invariants={
                "enabled_collider_count": 0,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_rejects_articulation_root_descendant_and_ancestor_edits(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "articulation.usda"
    _write_articulation_asset(asset)

    for target in ["/World/Robot", "/World/Robot/base", "/World"]:
        with pytest.raises(
            PhysicsTopologyPlanError,
            match="reject_articulation_changes=true",
        ):
            apply_physics_topology_plan(
                input_usd_path=asset,
                output_usd_path=tmp_path
                / f"{target.strip('/').replace('/', '_')}.usda",
                expected_source_digest=sha256_file(asset),
                mobility_intent="preserve",
                operations=[
                    {
                        "op": "ensure_rigid_body_api",
                        "prim_path": target,
                    }
                ],
                invariants={
                    "enabled_collider_count": 1,
                    "reject_articulation_changes": True,
                },
            )


def test_topology_plan_rejects_external_joint_targeting_articulation_root(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "external_joint.usda"
    _write_articulation_asset_with_external_fixed_joint(asset)

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="reject_articulation_changes=true",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "remove_fixed_joint",
                    "prim_path": "/World/Joints/ExternalFixedJoint",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_rejects_non_fixed_joint_endpoint_removal(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "hinge.usda"
    _write_non_fixed_joint_asset(asset)

    with pytest.raises(PhysicsTopologyPlanError, match="non-fixed joint endpoint"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "remove_rigid_body_api",
                    "prim_path": "/World/Left",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_rejects_fixed_joint_removal_with_children(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "fixed_joint_child.usda"
    _write_fixed_joint_with_child_asset(asset)

    with pytest.raises(PhysicsTopologyPlanError, match="child subtree"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "remove_fixed_joint",
                    "prim_path": "/World/FixedJoint",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_validates_non_fixed_joint_endpoint_ownership_after_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "hinge.usda"
    _write_non_fixed_joint_asset(asset)
    monkeypatch.setattr(
        topology_module,
        "_non_fixed_joint_endpoint_paths",
        lambda _topology: [],
    )

    with pytest.raises(PhysicsTopologyPlanError, match="endpoint ownership"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "remove_rigid_body_api",
                    "prim_path": "/World/Left",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_private_path_and_bounds_guards_cover_defensive_branches() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World")

    assert topology_module._path_in_scope("/World/Cube", "/World")
    assert topology_module._path_is_or_under("/World", "/")
    assert topology_module._display_opacity(xform.GetPrim(), UsdGeom) is None
    assert topology_module._component_bounds(stage, [], Usd, UsdGeom) == {}
    assert topology_module._component_bounds(stage, ["/World"], Usd, UsdGeom) == {}


def test_topology_plan_rejects_stale_digest_and_preserve_removals(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    _write_nested_physics_asset(asset)

    with pytest.raises(FileNotFoundError, match="Input USD not found"):
        sha256_file(tmp_path / "missing.usda")

    with pytest.raises(PhysicsTopologyPlanError, match="digest mismatch"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "stale.usda",
            expected_source_digest="sha256:stale",
            mobility_intent="movable",
            operations=[],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    with pytest.raises(PhysicsTopologyPlanError, match="USDZ package"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usdz",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    packaged = tmp_path / "packaged.usdz"
    packaged.write_bytes(b"not a real package")
    with pytest.raises(PhysicsTopologyPlanError, match="USDZ package inputs"):
        apply_physics_topology_plan(
            input_usd_path=packaged,
            output_usd_path=tmp_path / "prepared-from-usdz.usda",
            expected_source_digest="sha256:not-used",
            mobility_intent="movable",
            operations=[],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    with pytest.raises(PhysicsTopologyPlanError, match="RigidBodyAPI"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "noop.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "remove_rigid_body_api",
                    "prim_path": "/Asset",
                }
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    with pytest.raises(PhysicsTopologyPlanError, match="forbids"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "preserve.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {
                    "op": "remove_rigid_body_api",
                    "prim_path": "/Asset/Body/Inner",
                }
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    with pytest.raises(PhysicsTopologyPlanError, match="static' forbids"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "static.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="static",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/Asset/Body",
                }
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )

    with pytest.raises(PhysicsTopologyPlanError, match="articulation_changes=true"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "unsafe.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": False,
            },
        )


def test_topology_plan_rejects_nested_rigid_body_result(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "body.usda"
    _write_body_with_unowned_collider_asset(asset)

    with pytest.raises(PhysicsTopologyPlanError, match="nested enabled rigid bodies"):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="movable",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/World",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_allows_preexisting_nested_rigid_body_findings(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=[],
        invariants={
            "enabled_collider_count": 1,
            "reject_articulation_changes": True,
        },
    )

    assert output.is_file()
    assert any(
        finding["code"] == "nested_enabled_rigid_body"
        for finding in report["after"]["findings"]
    )
