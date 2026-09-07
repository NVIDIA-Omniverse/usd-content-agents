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


def _write_non_fixed_joint_with_inherited_endpoint_owner(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    parent = UsdGeom.Xform.Define(stage, "/World/Parent").GetPrim()
    endpoint = UsdGeom.Xform.Define(stage, "/World/Parent/Endpoint").GetPrim()
    other = UsdGeom.Xform.Define(stage, "/World/Other").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(parent).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.RigidBodyAPI.Apply(other).CreateRigidBodyEnabledAttr(True)
    endpoint_collider = UsdGeom.Cube.Define(stage, "/World/Parent/Endpoint/Collision")
    other_collider = UsdGeom.Cube.Define(stage, "/World/Other/Collision")
    UsdPhysics.CollisionAPI.Apply(
        endpoint_collider.GetPrim()
    ).CreateCollisionEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(other_collider.GetPrim()).CreateCollisionEnabledAttr(
        True
    )
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Slider")
    joint.CreateBody0Rel().SetTargets([parent.GetPath()])
    joint.CreateBody1Rel().SetTargets([endpoint.GetPath()])
    stage.GetRootLayer().Save()


def _write_filing_cabinet_articulation(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    cabinet = UsdGeom.Xform.Define(stage, "/Cabinet").GetPrim()
    stage.SetDefaultPrim(cabinet)
    fixed_body = UsdGeom.Xform.Define(stage, "/Cabinet/__JointAgent").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(fixed_body)
    for collider_index in range(2):
        collider = UsdGeom.Cube.Define(
            stage,
            f"/Cabinet/__JointAgent/Collider_{collider_index + 1}",
        )
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(
            True
        )
    for drawer_index in range(1, 7):
        shelf_path = f"/Cabinet/Shelf_{drawer_index}"
        UsdGeom.Xform.Define(stage, shelf_path)
        frame = UsdGeom.Xform.Define(stage, f"{shelf_path}/DrawerFrame").GetPrim()
        for collider_index in range(3):
            collider = UsdGeom.Cube.Define(
                stage,
                f"{shelf_path}/Collider_{collider_index + 1}",
            )
            UsdPhysics.CollisionAPI.Apply(
                collider.GetPrim()
            ).CreateCollisionEnabledAttr(True)
        joint = UsdPhysics.PrismaticJoint.Define(
            stage,
            f"/Cabinet/__JointAgent/DrawerJoint_{drawer_index}",
        )
        joint.CreateBody0Rel().SetTargets([fixed_body.GetPath()])
        joint.CreateBody1Rel().SetTargets([frame.GetPath()])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.45)
    stage.GetRootLayer().Save()


def _filing_cabinet_topology_plan() -> tuple[
    list[dict[str, str]], list[dict[str, str]]
]:
    fixed_body = "/Cabinet/__JointAgent"
    operations = [
        {"op": "ensure_rigid_body_api", "prim_path": fixed_body},
        *[
            {
                "op": "ensure_rigid_body_api",
                "prim_path": f"/Cabinet/Shelf_{drawer_index}",
            }
            for drawer_index in range(1, 7)
        ],
    ]
    promotions: list[dict[str, str]] = []
    for drawer_index in range(1, 7):
        joint_path = f"/Cabinet/__JointAgent/DrawerJoint_{drawer_index}"
        shelf_path = f"/Cabinet/Shelf_{drawer_index}"
        promotions.extend(
            [
                {
                    "joint_prim_path": joint_path,
                    "relationship": "body0",
                    "relationship_target_path": fixed_body,
                    "requested_rigid_body_ancestor_path": fixed_body,
                },
                {
                    "joint_prim_path": joint_path,
                    "relationship": "body1",
                    "relationship_target_path": f"{shelf_path}/DrawerFrame",
                    "requested_rigid_body_ancestor_path": shelf_path,
                },
            ]
        )
    return operations, promotions


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


def _write_instanced_physics_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    prototype_path = path.with_name("prototype.usda")
    prototype = Usd.Stage.CreateNew(str(prototype_path))
    model = UsdGeom.Xform.Define(prototype, "/Model").GetPrim()
    prototype.SetDefaultPrim(model)
    UsdPhysics.RigidBodyAPI.Apply(model).CreateRigidBodyEnabledAttr(True)
    collider = UsdGeom.Cube.Define(prototype, "/Model/Visual")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    prototype.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    instance = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddReference(str(prototype_path))
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()


def _write_instanced_topology_target_asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    prototype_path = path.with_name("topology_prototype.usda")
    prototype = Usd.Stage.CreateNew(str(prototype_path))
    model = UsdGeom.Xform.Define(prototype, "/Model").GetPrim()
    prototype.SetDefaultPrim(model)
    body = UsdGeom.Xform.Define(prototype, "/Model/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    collider = UsdGeom.Cube.Define(prototype, "/Model/Body/Collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    prototype.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for name in ("Selected", "Unrelated"):
        instance = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        instance.GetReferences().AddReference(str(prototype_path))
        instance.SetInstanceable(True)
    stage.GetRootLayer().Save()


def _write_instanced_fixed_joint_asset(path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    prototype_path = path.with_name("fixed_joint_prototype.usda")
    prototype = Usd.Stage.CreateNew(str(prototype_path))
    model = UsdGeom.Xform.Define(prototype, "/Model").GetPrim()
    prototype.SetDefaultPrim(model)
    body = UsdGeom.Xform.Define(prototype, "/Model/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    door = UsdGeom.Xform.Define(prototype, "/Model/Door").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(door).CreateRigidBodyEnabledAttr(True)
    collider = UsdGeom.Cube.Define(prototype, "/Model/Body/Collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    joint = UsdPhysics.FixedJoint.Define(prototype, "/Model/RootFixedJoint")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Model/Body")])
    door_joint = UsdPhysics.RevoluteJoint.Define(prototype, "/Model/DoorJoint")
    door_joint.CreateBody0Rel().SetTargets([Sdf.Path("/Model/Body")])
    door_joint.CreateBody1Rel().SetTargets([Sdf.Path("/Model/Door")])
    prototype.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for name in ("Selected", "Unrelated"):
        instance = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        instance.GetReferences().AddReference(str(prototype_path))
        instance.SetInstanceable(True)
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


def test_component_inspection_traverses_instance_proxies(tmp_path: Path) -> None:
    asset = tmp_path / "instanced.usda"
    _write_instanced_physics_asset(asset)

    topology = inspect_physics_topology(asset)
    result = inspect_physics_components(asset)

    proxy_path = "/World/Instance/Visual"
    assert topology["enabled_rigid_body_count"] == 1
    assert topology["enabled_collider_count"] == 1
    assert topology["colliders"][0]["prim_path"] == proxy_path
    assert result["component_count"] == 1
    component = result["components"][0]
    assert component["body_root_path"] == "/World/Instance"
    assert component["visual_evidence_paths"] == [proxy_path]
    assert component["collider_paths"] == [proxy_path]
    assert component["bounds_m"]


def test_topology_inspection_allows_reset_nested_rigid_body(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    asset = tmp_path / "nested-reset.usda"
    _write_nested_physics_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    inner = stage.GetPrimAtPath("/Asset/Body/Inner")
    UsdGeom.Xformable(inner).SetResetXformStack(True)
    stage.GetRootLayer().Save()

    topology = inspect_physics_topology(asset)

    assert not any(
        finding["code"] == "nested_enabled_rigid_body"
        for finding in topology["findings"]
    )


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


def test_topology_plan_resets_ensured_nested_body_without_moving_it(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Usd, UsdGeom

    asset = tmp_path / "nested.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)
    source_stage = Usd.Stage.Open(str(asset))
    assert source_stage is not None
    source_parent = UsdGeom.Xformable(source_stage.GetPrimAtPath("/Asset/Body"))
    source_parent.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
    source_body = UsdGeom.Xformable(source_stage.GetPrimAtPath("/Asset/Body/Inner"))
    source_body.AddTranslateOp().Set(Gf.Vec3d(3.0, 4.0, 5.0))
    source_stage.GetRootLayer().Save()
    before_world = source_body.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=[
            {"op": "ensure_rigid_body_api", "prim_path": "/Asset/Body"},
            {"op": "ensure_rigid_body_api", "prim_path": "/Asset/Body/Inner"},
        ],
        invariants={
            "enabled_collider_count": 1,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    prepared_body = UsdGeom.Xformable(prepared.GetPrimAtPath("/Asset/Body/Inner"))
    assert prepared_body.GetResetXformStack()
    assert (
        prepared_body.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        == before_world
    )
    assert not any(
        finding["code"] == "nested_enabled_rigid_body"
        for finding in report["after"]["findings"]
    )
    assert report["applied_operations"][1]["reset_xform_stack"] == "preserve_world"


def test_topology_plan_annotates_only_the_ensure_operation_that_resets(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "nested.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)

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
                "op": "ensure_rigid_body_api",
                "prim_path": "/Asset/Body/Inner",
            },
        ],
        invariants={
            "enabled_collider_count": 1,
            "reject_articulation_changes": True,
        },
    )

    assert "reset_xform_stack" not in report["applied_operations"][0]
    assert report["applied_operations"][1]["reset_xform_stack"] == "preserve_world"


def test_topology_plan_refuses_reset_with_animated_ancestry(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    asset = tmp_path / "animated-nested.usda"
    _write_nested_physics_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    parent = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset/Body"))
    rotation = parent.AddRotateZOp()
    rotation.Set(0.0, Usd.TimeCode(0.0))
    rotation.Set(90.0, Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="animated or connected transform ancestry",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {"op": "ensure_rigid_body_api", "prim_path": "/Asset/Body"},
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/Asset/Body/Inner",
                },
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_refuses_reset_with_animated_xform_op_order(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    asset = tmp_path / "animated-order-nested.usda"
    _write_nested_physics_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    parent = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset/Body"))
    parent.AddTranslateOp()
    order_attribute = parent.GetPrim().GetAttribute("xformOpOrder")
    order = order_attribute.Get()
    order_attribute.Set(order, Usd.TimeCode(0.0))
    order_attribute.Set(order, Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="animated or connected transform ancestry",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/Asset/Body/Inner",
                },
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_refuses_reset_with_connected_xform_op(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom

    asset = tmp_path / "connected-op-nested.usda"
    _write_nested_physics_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    parent = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset/Body"))
    translate = parent.AddTranslateOp()
    translate.GetAttr().AddConnection(Sdf.Path("/Driver.outputs:translate"))
    stage.GetRootLayer().Save()

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="animated or connected transform ancestry",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/Asset/Body/Inner",
                },
            ],
            invariants={
                "enabled_collider_count": 1,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_stops_ancestry_checks_at_an_existing_reset(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    asset = tmp_path / "reset-boundary.usda"
    output = tmp_path / "prepared.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    world = UsdGeom.Xform.Define(stage, "/World")
    world_translate = world.AddTranslateOp()
    world_translate.Set(Gf.Vec3d(100.0, 0.0, 0.0))
    world_translate.GetAttr().AddConnection(Sdf.Path("/Driver.outputs:translate"))
    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    boundary = UsdGeom.Xform.Define(stage, "/World/Body/Reset")
    boundary.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    boundary.SetResetXformStack(True)
    inner = UsdGeom.Xform.Define(stage, "/World/Body/Reset/Inner")
    inner.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    before_world = inner.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=[
            {
                "op": "ensure_rigid_body_api",
                "prim_path": "/World/Body/Reset/Inner",
            },
        ],
        invariants={
            "enabled_collider_count": 0,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    prepared_inner = UsdGeom.Xformable(
        prepared.GetPrimAtPath("/World/Body/Reset/Inner")
    )
    assert prepared_inner.GetResetXformStack()
    assert (
        prepared_inner.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        == before_world
    )


def test_world_preserving_reset_is_idempotent_for_an_existing_reset(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    asset = tmp_path / "already-reset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    xformable = UsdGeom.Xform.Define(stage, "/World/Body")
    xformable.SetResetXformStack(True)

    topology_module._reset_xform_stack_preserving_world(
        xformable.GetPrim(),
        Usd,
        UsdGeom,
    )

    assert xformable.GetResetXformStack()


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


def test_topology_plan_deinstances_selected_proxy_target_only(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    asset = tmp_path / "instanced_topology.usda"
    output = tmp_path / "prepared.usda"
    _write_instanced_topology_target_asset(asset)

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="movable",
        operations=[
            {
                "op": "remove_rigid_body_api",
                "prim_path": "/World/Selected/Body",
            }
        ],
        invariants={
            "enabled_collider_count": 2,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    selected = prepared.GetPrimAtPath("/World/Selected")
    selected_body = prepared.GetPrimAtPath("/World/Selected/Body")
    unrelated = prepared.GetPrimAtPath("/World/Unrelated")
    unrelated_body = prepared.GetPrimAtPath("/World/Unrelated/Body")
    assert report["applied_operations"] == [
        {"op": "remove_rigid_body_api", "prim_path": "/World/Selected/Body"}
    ]
    assert not selected.IsInstanceable()
    assert not selected_body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert unrelated.IsInstanceable()
    assert unrelated_body.IsInstanceProxy()
    assert unrelated_body.HasAPI(UsdPhysics.RigidBodyAPI)


def test_topology_plan_removes_fixed_joint_from_selected_instance_only(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    asset = tmp_path / "instanced_fixed_joint.usda"
    output = tmp_path / "prepared.usda"
    _write_instanced_fixed_joint_asset(asset)

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="movable",
        operations=[
            {
                "op": "remove_fixed_joint",
                "prim_path": "/World/Selected/RootFixedJoint",
            }
        ],
        invariants={
            "enabled_collider_count": 2,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    selected = prepared.GetPrimAtPath("/World/Selected")
    selected_joint = prepared.GetPrimAtPath("/World/Selected/RootFixedJoint")
    selected_door_joint = prepared.GetPrimAtPath("/World/Selected/DoorJoint")
    unrelated = prepared.GetPrimAtPath("/World/Unrelated")
    unrelated_joint = prepared.GetPrimAtPath("/World/Unrelated/RootFixedJoint")
    unrelated_door_joint = prepared.GetPrimAtPath("/World/Unrelated/DoorJoint")
    assert report["applied_operations"] == [
        {
            "op": "remove_fixed_joint",
            "prim_path": "/World/Selected/RootFixedJoint",
        }
    ]
    assert not selected.IsInstanceable()
    assert selected_joint.IsValid()
    assert not selected_joint.IsActive()
    assert not selected_door_joint.IsInstanceProxy()
    assert selected_door_joint.IsA(UsdPhysics.RevoluteJoint)
    assert unrelated.IsInstanceable()
    assert unrelated_joint.IsInstanceProxy()
    assert unrelated_joint.IsA(UsdPhysics.FixedJoint)
    assert unrelated_door_joint.IsInstanceProxy()
    assert unrelated_door_joint.IsA(UsdPhysics.RevoluteJoint)


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


def test_topology_plan_allows_explicit_exact_endpoint_owner_promotion(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "inherited-endpoint.usda"
    output = tmp_path / "prepared.usda"
    _write_non_fixed_joint_with_inherited_endpoint_owner(asset)

    before = inspect_physics_topology(asset)
    assert before["joints"][0]["body1_rigid_body_paths"] == ["/World/Parent"]

    apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=[
            {
                "op": "ensure_rigid_body_api",
                "prim_path": "/World/Parent/Endpoint",
            }
        ],
        joint_endpoint_owner_promotions=[
            {
                "joint_prim_path": "/World/Slider",
                "relationship": "body1",
                "relationship_target_path": "/World/Parent/Endpoint",
                "requested_rigid_body_ancestor_path": "/World/Parent/Endpoint",
            }
        ],
        invariants={
            "enabled_collider_count": 2,
            "reject_articulation_changes": True,
        },
    )

    after = inspect_physics_topology(output)
    assert after["joints"][0]["body1_targets"] == ["/World/Parent/Endpoint"]
    assert after["joints"][0]["body1_rigid_body_paths"] == ["/World/Parent/Endpoint"]


def test_non_fixed_joint_structural_signature_skips_disabled_joint_and_missing_attr(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    asset = tmp_path / "disabled-hinge.usda"
    _write_non_fixed_joint_asset(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    joint = stage.GetPrimAtPath("/World/Hinge")

    assert topology_module._joint_attribute_signature(joint, "physics:missing") == (
        False,
    )
    joint.GetAttribute("physics:jointEnabled").Set(False)
    assert topology_module._non_fixed_joint_structural_signature(stage) == ()


def test_endpoint_owner_promotion_must_change_the_requested_owner(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "inherited-endpoint.usda"
    _write_non_fixed_joint_with_inherited_endpoint_owner(asset)
    topology = inspect_physics_topology(asset)

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="did not produce the requested owner change",
    ):
        topology_module._validate_joint_endpoint_owner_promotions(
            topology,
            topology,
            ensured_paths={"/World/Parent/Endpoint"},
            promotions=[
                {
                    "joint_prim_path": "/World/Slider",
                    "relationship": "body1",
                    "relationship_target_path": "/World/Parent/Endpoint",
                    "requested_rigid_body_ancestor_path": "/World/Parent/Endpoint",
                }
            ],
        )


def test_topology_plan_promotes_filing_cabinet_joint_owners_fail_closed(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "filing-cabinet.usda"
    output = tmp_path / "prepared.usda"
    _write_filing_cabinet_articulation(asset)
    operations, promotions = _filing_cabinet_topology_plan()
    before = inspect_physics_topology(asset)
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    joint_signature = topology_module._non_fixed_joint_structural_signature(stage)

    report = apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="preserve",
        operations=operations,
        joint_endpoint_owner_promotions=promotions,
        invariants={
            "enabled_collider_count": 20,
            "reject_articulation_changes": True,
        },
    )

    after = inspect_physics_topology(output)
    prepared_stage = Usd.Stage.Open(str(output))
    assert prepared_stage is not None
    assert (
        before["articulation_root_paths"]
        == after["articulation_root_paths"]
        == ["/Cabinet/__JointAgent"]
    )
    assert (
        topology_module._non_fixed_joint_structural_signature(prepared_stage)
        == joint_signature
    )
    assert after["enabled_rigid_body_count"] == 7
    assert after["enabled_collider_count"] == 20
    collider_counts: dict[str, int] = {}
    for collider in after["colliders"]:
        owner = collider["owner_rigid_body_path"]
        assert isinstance(owner, str)
        collider_counts[owner] = collider_counts.get(owner, 0) + 1
    assert collider_counts == {
        "/Cabinet/__JointAgent": 2,
        **{f"/Cabinet/Shelf_{index}": 3 for index in range(1, 7)},
    }
    assert len(report["applied_joint_endpoint_owner_promotions"]) == 12
    for joint in after["joints"]:
        drawer_index = joint["prim_path"].rsplit("_", 1)[-1]
        assert joint["joint_type"] == "PhysicsPrismaticJoint"
        assert joint["body0_targets"] == ["/Cabinet/__JointAgent"]
        assert joint["body0_rigid_body_paths"] == ["/Cabinet/__JointAgent"]
        assert joint["body1_targets"] == [f"/Cabinet/Shelf_{drawer_index}/DrawerFrame"]
        assert joint["body1_rigid_body_paths"] == [f"/Cabinet/Shelf_{drawer_index}"]


def test_topology_plan_fails_closed_when_exported_derivative_cannot_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    asset = tmp_path / "nested.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)
    original_open = Usd.Stage.Open
    derivative_opens = 0

    def fail_for_derivative(root_layer: object, *args: object, **kwargs: object):
        nonlocal derivative_opens
        if isinstance(root_layer, str) and Path(root_layer).name.startswith(
            ".prepared."
        ):
            derivative_opens += 1
            if derivative_opens == 3:
                return None
        return original_open(root_layer, *args, **kwargs)

    monkeypatch.setattr(Usd.Stage, "Open", fail_for_derivative)

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="Failed to reopen the exported topology derivative",
    ):
        apply_physics_topology_plan(
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

    assert not output.exists()


def test_topology_plan_fails_closed_on_joint_structural_signature_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "hinge.usda"
    output = tmp_path / "prepared.usda"
    _write_non_fixed_joint_asset(asset)
    topology = inspect_physics_topology(asset)
    original_signature = topology_module._non_fixed_joint_structural_signature
    signature_reads = 0

    def drift_after_source(stage: object):
        nonlocal signature_reads
        signature_reads += 1
        signature = original_signature(stage)
        if signature_reads == 1:
            return signature
        return (*signature, ("forced-test-drift",))

    monkeypatch.setattr(
        topology_module,
        "_non_fixed_joint_structural_signature",
        drift_after_source,
    )

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="changed a non-fixed joint prim",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=output,
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[],
            invariants={
                "enabled_collider_count": topology["enabled_collider_count"],
                "reject_articulation_changes": True,
            },
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("unlisted", "outside the explicit promotion allowlist"),
        ("mismatched_target", "does not match one unambiguous relationship target"),
        ("duplicate", "duplicate joint endpoint owner promotion is ambiguous"),
        ("non_ancestor", "must be the relationship target or a strict ancestor"),
        ("unlisted_ensure", "ensure_rigid_body_api operation is not covered"),
    ],
)
def test_filing_cabinet_owner_promotion_rejects_invalid_allowlists(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    asset = tmp_path / f"filing-cabinet-{mutation}.usda"
    _write_filing_cabinet_articulation(asset)
    operations, promotions = _filing_cabinet_topology_plan()
    if mutation == "unlisted":
        promotions = promotions[1:]
    elif mutation == "mismatched_target":
        promotions[0] = {
            **promotions[0],
            "relationship_target_path": "/Cabinet/Shelf_1/DrawerFrame",
        }
    elif mutation == "duplicate":
        promotions.append(dict(promotions[0]))
    elif mutation == "non_ancestor":
        promotions[1] = {
            **promotions[1],
            "requested_rigid_body_ancestor_path": "/Cabinet/Shelf_2",
        }
    elif mutation == "unlisted_ensure":
        operations.append(
            {
                "op": "ensure_rigid_body_api",
                "prim_path": "/Cabinet/Shelf_1/DrawerFrame",
            }
        )

    with pytest.raises(PhysicsTopologyPlanError, match=error_match):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / f"prepared-{mutation}.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=operations,
            joint_endpoint_owner_promotions=promotions,
            invariants={
                "enabled_collider_count": 20,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_rejects_non_endpoint_owner_promotion(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    asset = tmp_path / "inherited-leaf-endpoint.usda"
    _write_non_fixed_joint_with_inherited_endpoint_owner(asset)
    stage = Usd.Stage.Open(str(asset))
    assert stage is not None
    leaf = UsdGeom.Xform.Define(stage, "/World/Parent/Endpoint/Leaf").GetPrim()
    joint = stage.GetPrimAtPath("/World/Slider")
    joint.GetRelationship("physics:body1").SetTargets([leaf.GetPath()])
    stage.GetRootLayer().Save()

    with pytest.raises(
        PhysicsTopologyPlanError,
        match="outside the explicit promotion allowlist",
    ):
        apply_physics_topology_plan(
            input_usd_path=asset,
            output_usd_path=tmp_path / "prepared.usda",
            expected_source_digest=sha256_file(asset),
            mobility_intent="preserve",
            operations=[
                {
                    "op": "ensure_rigid_body_api",
                    "prim_path": "/World/Parent/Endpoint",
                }
            ],
            invariants={
                "enabled_collider_count": 2,
                "reject_articulation_changes": True,
            },
        )


def test_topology_plan_does_not_reset_an_ensured_then_removed_body(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    asset = tmp_path / "ensure-remove.usda"
    output = tmp_path / "prepared.usda"
    _write_nested_physics_asset(asset)

    apply_physics_topology_plan(
        input_usd_path=asset,
        output_usd_path=output,
        expected_source_digest=sha256_file(asset),
        mobility_intent="movable",
        operations=[
            {
                "op": "ensure_rigid_body_api",
                "prim_path": "/Asset/Body/Inner",
            },
            {
                "op": "remove_rigid_body_api",
                "prim_path": "/Asset/Body/Inner",
            },
        ],
        invariants={
            "enabled_collider_count": 1,
            "reject_articulation_changes": True,
        },
    )

    prepared = Usd.Stage.Open(str(output))
    assert prepared is not None
    inner = prepared.GetPrimAtPath("/Asset/Body/Inner")
    assert not inner.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not UsdGeom.Xformable(inner).GetResetXformStack()


def test_private_path_and_bounds_guards_cover_defensive_branches() -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World")

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
