# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic native authoring witnesses; these do not qualify benchmark assets."""
import copy
import math

import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from content_agent_workflows.physics.workflow import (
    PhysicsComponent, PhysicsComponentTargetDecision, PhysicsMassProperties,
    _merge_rebased_component_decisions, add_physics_component_target_catalog,
    resolve_physics_v2_patch_targets,
)
from content_agent_workflows.physics.usd_cli_ops import physics_patch_from_workflow_decisions
from usd_core.physics import apply_operations, apply_rigid_body


MASS_PROPERTIES = {
    "center_of_mass": [0.02, 0.03, 0.04],
    "diagonal_inertia": [0.1, 0.2, 0.25],
    "principal_axes": [math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
}


def target_patch():
    components = [
        PhysicsComponent(component_id="moving", body_root_path="/World/Moving",
                         visual_evidence_paths=["/World/Moving/Mesh"]),
        PhysicsComponent(component_id="static", component_role="unowned_static",
                         body_root_path="/World/Static", visual_evidence_paths=["/World/Static/Mesh"]),
    ]
    catalog = add_physics_component_target_catalog({"components": [c.model_dump() for c in components]})
    decisions = []
    for c in catalog["components"]:
        dynamic = c["component_role"] == "body"
        d = dict(decision_id=c["component_id"], component_id=c["component_id"],
                 collider_target_ids=[c["authoring_targets"][0]["target_id"]],
                 collision_mode="author_on_targets", inferred_material_family="wood",
                 collision_approximation="convexHull", physical_properties={
                     "estimated_mass_kg": 2.0 if dynamic else 0.0,
                     "density": 500.0 if dynamic else 0.0,
                     "static_friction": 0.6, "dynamic_friction": 0.6, "restitution": 0.0},
                 confidence=1.0, rationale="Synthetic material and mobility fixture.")
        if dynamic:
            d["mass_properties"] = copy.deepcopy(MASS_PROPERTIES)
        PhysicsComponentTargetDecision.model_validate(d)
        decisions.append(d)
    return components, {"schema_version": "content-agent-workflows.physics-decision-patch.v2",
                        "asset": "fixture.usda", "source_digest": "sha256:fixture",
                        "decisions": decisions, "unresolved_components": []}


def resolve(components, patch):
    return resolve_physics_v2_patch_targets(patch, components=components, source_digest="sha256:fixture")[1]


def test_native_target_resolution_and_saved_massapi_preserve_static_fixture(tmp_path):
    components, patch = target_patch()
    decisions = resolve(components, patch)
    assert [d.component_role for d in decisions] == ["body", "unowned_static"]
    operations = physics_patch_from_workflow_decisions(
        [d.model_dump(mode="json") for d in decisions], author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene", physics_material_scope_path="/World/Looks")
    assert [d["path"] for d in operations["rigid_bodies"]] == ["/World/Moving"]
    stage = Usd.Stage.CreateNew(str(tmp_path / "native.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    for name in ("Moving", "Static"):
        UsdGeom.Xform.Define(stage, "/World/" + name)
        UsdGeom.Cube.Define(stage, "/World/" + name + "/Mesh")
    apply_operations(stage, operations)
    stage.GetRootLayer().Save()
    saved = Usd.Stage.Open(str(tmp_path / "native.usda"))
    moving = saved.GetPrimAtPath("/World/Moving")
    assert moving.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not saved.GetPrimAtPath("/World/Static").HasAPI(UsdPhysics.RigidBodyAPI)
    assert all(saved.GetPrimAtPath("/World/" + name + "/Mesh").HasAPI(UsdPhysics.CollisionAPI)
               for name in ("Moving", "Static"))
    props = UsdPhysics.MassAPI(moving)
    assert props.GetMassAttr().Get() == 2
    assert list(props.GetCenterOfMassAttr().Get()) == pytest.approx(MASS_PROPERTIES["center_of_mass"])
    assert list(props.GetDiagonalInertiaAttr().Get()) == pytest.approx(MASS_PROPERTIES["diagonal_inertia"])
    q = props.GetPrincipalAxesAttr().Get()
    assert [q.GetReal(), *q.GetImaginary()] == pytest.approx(MASS_PROPERTIES["principal_axes"])


@pytest.mark.parametrize("key,value", [
    ("center_of_mass", [0, 0]), ("center_of_mass", [False, 0, 0]),
    ("center_of_mass", [float("nan"), 0, 0]), ("center_of_mass", [1e100, 0, 0]),
    ("diagonal_inertia", [0, 1, 1]), ("diagonal_inertia", [-1, 1, 1]),
    ("diagonal_inertia", [1e-100, 1e-100, 1e-100]),
    ("diagonal_inertia", [1, 1, 3]), ("diagonal_inertia", [1, float("inf"), 1]),
    ("principal_axes", [0, 0, 0, 0]), ("principal_axes", [2, 0, 0, 0]),
    ("principal_axes", [1, 0, 0]),
])
def test_invalid_vectors_rejected_before_any_native_stage_mutation(key, value):
    bad = {**MASS_PROPERTIES, key: value}
    with pytest.raises(ValueError):
        PhysicsMassProperties.model_validate(bad)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/First")
    UsdGeom.Xform.Define(stage, "/World/Invalid")
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_operations(stage, {"scene_paths": ["/World/PhysicsScene"], "rigid_bodies": [
            {"path": "/World/First", "mass": 1},
            {"path": "/World/Invalid", "mass": 1, "mass_properties": bad}]})
    assert stage.GetRootLayer().ExportToString() == before


def test_dynamic_zero_mass_keeps_default_behavior_without_explicit_vectors():
    operations = physics_patch_from_workflow_decisions([{
        "mass_authoring_path": "/World/Dynamic", "collider_paths": ["/World/Dynamic/Mesh"],
        "physical_properties": {"estimated_mass_kg": 0, "density": 0},
        "collision_approximation": "convexHull"}], author_rigid_body=True,
        physics_scene_path="/World/PhysicsScene")
    assert operations["rigid_bodies"] == [{"path": "/World/Dynamic", "density": 0, "mass": 0}]
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Dynamic")
    apply_rigid_body(stage, "/World/Dynamic", mass=0, density=0)
    prim = stage.GetPrimAtPath("/World/Dynamic")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().HasAuthoredValueOpinion()


def test_child_cannot_change_inspected_component_mobility():
    components, patch = target_patch()
    patch["decisions"][1]["component_role"] = "body"
    with pytest.raises(RuntimeError, match="component_role"):
        resolve(components, patch)


def test_static_component_cannot_request_rigid_mass_vectors():
    components, patch = target_patch()
    patch["decisions"][1]["mass_properties"] = MASS_PROPERTIES
    with pytest.raises(RuntimeError, match="Static components"):
        resolve(components, patch)


def test_mass_vectors_require_reauthoring_when_body_frame_changes():
    components, patch = target_patch()
    decision = resolve(components, patch)[0]
    changed = components[0].model_copy(update={"body_root_path": "/World/NewFrame"})
    with pytest.raises(RuntimeError, match="body frame"):
        _merge_rebased_component_decisions(changed, [decision])
