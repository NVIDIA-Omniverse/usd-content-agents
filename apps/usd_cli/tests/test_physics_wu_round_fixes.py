# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-13 regressions retained at the workflow-neutral usd-cli boundary.

The low-level authoring API applies only explicitly listed schema targets and reports
every authored API. Workflow decisions, target inference, and preservation policy are
intentionally outside usd-cli. API removal remains available as a mechanical edit.
"""

from __future__ import annotations

import pytest

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from usd_core import physics
from usd_core.edit import remove_api
from usd_core.physics_topology import inspect_topology, source_digest


N_COLLIDERS = 34  # the task-13 asset's preauthored collider count


def test_topology_inspection_reports_composed_facts_without_workflow_policy(tmp_path):
    path = tmp_path / "topology.usda"
    stage = Usd.Stage.CreateNew(str(path))
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    collider = UsdGeom.Mesh.Define(stage, "/World/Body/Collider").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.CollisionAPI.Apply(collider)
    stage.GetRootLayer().Save()

    report = inspect_topology(path)

    assert report["schema_version"] == "usd-cli.physics-topology.v1"
    assert report["source_digest"] == source_digest(path)
    assert report["rigid_body_paths"] == ["/World/Body"]
    assert report["enabled_collider_count"] == 1
    assert report["colliders"] == [
        {
            "prim_path": "/World/Body/Collider",
            "owner_rigid_body_path": "/World/Body",
            "type_name": "Mesh",
        }
    ]
    assert report["findings"] == []


def test_topology_inspection_binds_unsaved_live_stage_contents(tmp_path):
    path = tmp_path / "live-topology.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.GetRootLayer().Save()
    disk_report = inspect_topology(path)

    body = UsdGeom.Xform.Define(stage, "/World/UnsavedBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    live_report = inspect_topology(path, stage=stage)

    assert disk_report["rigid_body_paths"] == []
    assert live_report["rigid_body_paths"] == ["/World/UnsavedBody"]
    assert live_report["source_digest"] != disk_report["source_digest"]


def test_topology_inspection_binds_anonymous_session_layer_edits(tmp_path):
    path = tmp_path / "session-topology.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    stage.GetRootLayer().Save()
    disk_report = inspect_topology(path)

    stage.SetEditTarget(stage.GetSessionLayer())
    UsdPhysics.RigidBodyAPI.Apply(body)
    live_report = inspect_topology(path, stage=stage)

    assert live_report["rigid_body_paths"] == ["/World/Body"]
    assert live_report["source_digest"] != disk_report["source_digest"]
    assert not stage.GetRootLayer().dirty


def _body_with_colliders(n: int = N_COLLIDERS):
    """Task-13's asset in miniature: an Xform body root whose n child meshes each carry
    a preauthored PhysicsCollisionAPI. The root itself is NOT a collider."""
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World")
    UsdGeom.Xform.Define(s, "/World/Body")
    for i in range(n):
        mesh = UsdGeom.Mesh.Define(s, f"/World/Body/mesh_{i}")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return s


def _collider_count(stage) -> int:
    return physics.validate_schema(stage)["checks"]["colliders"]


def test_validate_schema_reports_only_enabled_rigid_bodies():
    stage = Usd.Stage.CreateInMemory()
    enabled = UsdGeom.Xform.Define(stage, "/World/Enabled").GetPrim()
    disabled = UsdGeom.Xform.Define(stage, "/World/Disabled").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(enabled)
    disabled_api = UsdPhysics.RigidBodyAPI.Apply(disabled)
    disabled_api.CreateRigidBodyEnabledAttr(False)

    checks = physics.validate_schema(stage)["checks"]

    assert checks["rigid_bodies"] == 2
    assert checks["enabled_rigid_bodies"] == 1


# ── BUG: collider preservation + authored_apis reporting ─────────────────────────────
def test_apply_only_authors_explicit_targets():
    """Adding a body and material must not infer a collider on the body root."""
    stage = _body_with_colliders()
    record = physics.apply_operations(
        stage,
        {
            "scene_paths": ["/PhysicsScenario"],
            "rigid_bodies": [
                {"path": "/World/Body", "mass": 1.2, "density": 800.0}
            ],
            "materials": [
                {
                    "path": "/World/PhysicsMaterial",
                    "static_friction": 0.4,
                    "dynamic_friction": 0.3,
                    "restitution": 0.1,
                }
            ],
            "bindings": [
                {
                    "target_path": "/World/Body",
                    "material_path": "/World/PhysicsMaterial",
                }
            ],
        },
    )
    assert _collider_count(stage) == N_COLLIDERS
    body = stage.GetPrimAtPath("/World/Body")
    assert not body.HasAPI(UsdPhysics.CollisionAPI)
    assert body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert record["rigid_body"] == ["/World/Body"]
    assert record["collision"] == []
    apis = record["authored_apis"]
    assert "PhysicsRigidBodyAPI" in apis["/World/Body"]
    assert "PhysicsMassAPI" in apis["/World/Body"]
    assert "PhysicsCollisionAPI" not in apis["/World/Body"]
    assert record["material"] == ["/World/PhysicsMaterial"]
    assert "PhysicsMaterialAPI" in apis["/World/PhysicsMaterial"]
    assert "MaterialBindingAPI" in apis["/World/Body"]


def test_apply_explicit_collision_targets():
    """Exactly the caller-listed collider operation paths get colliders."""
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World/Body")
    for i in range(3):
        UsdGeom.Mesh.Define(s, f"/World/Body/mesh_{i}")
    targets = ["/World/Body/mesh_0", "/World/Body/mesh_1"]
    rec = physics.apply_operations(
        s,
        {
            "rigid_bodies": [{"path": "/World/Body", "mass": 1.2}],
            "colliders": [
                {"path": path, "approximation": "convexHull"} for path in targets
            ],
        },
    )
    assert rec["collision"] == targets
    for p in targets:
        prim = s.GetPrimAtPath(p)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert rec["authored_apis"][p] == ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    assert not s.GetPrimAtPath("/World/Body/mesh_2").HasAPI(UsdPhysics.CollisionAPI)
    assert not s.GetPrimAtPath("/World/Body").HasAPI(UsdPhysics.CollisionAPI)

def test_apply_does_not_infer_a_collider_for_a_fresh_body():
    """Target selection belongs to the workflow, including collider-less bodies."""
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(s, "/World/Body")
    rec = physics.apply_operations(
        s, {"rigid_bodies": [{"path": "/World/Body", "mass": 1.2}]}
    )
    assert rec["collision"] == []
    assert "PhysicsCollisionAPI" not in rec["authored_apis"]["/World/Body"]
    assert _collider_count(s) == 0


def test_apply_updates_an_explicit_existing_collider():
    """A root that already carries CollisionAPI may be re-authored in place (updating
    its approximation adds no new collider), so the collider count is unchanged."""
    s = Usd.Stage.CreateInMemory()
    body = UsdGeom.Mesh.Define(s, "/World/Body").GetPrim()
    UsdPhysics.CollisionAPI.Apply(body)
    assert _collider_count(s) == 1
    rec = physics.apply_operations(
        s,
        {"colliders": [{"path": "/World/Body", "approximation": "convexHull"}]},
    )
    assert _collider_count(s) == 1
    assert rec["collision"] == ["/World/Body"]
    assert "PhysicsCollisionAPI" in rec["authored_apis"]["/World/Body"]
    approx = UsdPhysics.MeshCollisionAPI(body).GetApproximationAttr().Get()
    assert approx == "convexHull"


def test_apply_multiple_explicit_colliders_does_not_touch_the_body_root():
    s = _body_with_colliders(0)
    for i in range(2):
        UsdGeom.Mesh.Define(s, f"/World/Body/mesh_{i}")
    targets = ["/World/Body/mesh_0", "/World/Body/mesh_1"]
    rec = physics.apply_operations(
        s,
        {
            "rigid_bodies": [{"path": "/World/Body", "mass": 1.2}],
            "colliders": [
                {"path": path, "approximation": "convexHull"} for path in targets
            ],
        },
    )
    assert rec["collision"] == targets
    assert not s.GetPrimAtPath("/World/Body").HasAPI(UsdPhysics.CollisionAPI)
    assert _collider_count(s) == 2


# ── CAPABILITY: remove_api round-trips ────────────────────────────────────────────────
def test_remove_api_rigid_body_round_trip():
    """The exact task-13 remediation: RemoveAPI + delete the schema's authored
    properties so no opinions linger."""
    s = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(s, "/World/Body").GetPrim()
    api = UsdPhysics.RigidBodyAPI.Apply(prim)
    api.CreateVelocityAttr().Set((1.0, 2.0, 3.0))
    api.CreateKinematicEnabledAttr().Set(True)
    summary = remove_api(s, "/World/Body", "PhysicsRigidBodyAPI")
    assert summary["path"] == "/World/Body"
    assert summary["api"] == "PhysicsRigidBodyAPI"
    assert summary["removed"] is True
    assert summary["properties_removed"] == ["physics:kinematicEnabled",
                                             "physics:velocity"]
    assert summary["properties_masked"] == []  # everything truly removed here
    # pre-removal capture (for the Session's real undo) reflects the authored values
    captured = {p["name"]: p for p in summary["captured"]["properties"]}
    assert captured["physics:velocity"]["value"] == (1.0, 2.0, 3.0)
    assert captured["physics:kinematicEnabled"]["value"] is True
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert "PhysicsRigidBodyAPI" not in prim.GetAppliedSchemas()
    assert not prim.GetAttribute("physics:velocity").IsValid()
    assert not prim.GetAttribute("physics:kinematicEnabled").IsValid()


def test_remove_api_accepts_short_names_and_any_casing():
    s = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(s, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr().Set(2.5)
    summary = remove_api(s, "/World/Body", "massapi")  # short + lowercase
    assert summary["api"] == "PhysicsMassAPI"
    assert summary["properties_removed"] == ["physics:mass"]
    assert not prim.HasAPI(UsdPhysics.MassAPI)
    summary2 = remove_api(s, "/World/Body", "RigidBodyAPI")  # short, no Physics prefix
    assert summary2["api"] == "PhysicsRigidBodyAPI"
    assert list(prim.GetAppliedSchemas()) == []


def test_remove_api_material_binding_purpose():
    """Removing MaterialBindingAPI also deletes the per-purpose binding relationships
    (its prim definition declares none — they are dynamic)."""
    s = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(s, "/World/Body").GetPrim()
    physics.apply_operations(
        s,
        {
            "materials": [
                {"path": "/World/PhysicsMaterial", "static_friction": 0.5}
            ],
            "bindings": [
                {
                    "target_path": "/World/Body",
                    "material_path": "/World/PhysicsMaterial",
                }
            ],
        },
    )
    assert prim.GetRelationship("material:binding:physics").IsValid()
    summary = remove_api(s, "/World/Body", "MaterialBindingAPI")
    assert summary["api"] == "MaterialBindingAPI"
    assert summary["properties_removed"] == ["material:binding:physics"]
    assert not prim.HasAPI(UsdShade.MaterialBindingAPI)
    assert not prim.GetRelationship("material:binding:physics").IsValid()


def test_remove_api_fixes_nested_rigid_body():
    """Task-13's actual repair loop: a nested RigidBodyAPI fails validate; remove_api
    on the child makes validate green again."""
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World/Body")
    UsdGeom.Cube.Define(s, "/World/Body/inner")
    physics.define_physics_scene(s, "/World/PhysicsScene")
    physics.apply_rigid_body(s, "/World/Body", mass=1.0)
    physics.apply_rigid_body(s, "/World/Body/inner", mass=0.5)
    assert any("nested rigid body" in i for i in physics.validate_schema(s)["issues"])
    summary = remove_api(s, "/World/Body/inner", "RigidBodyAPI")
    assert summary["removed"]
    remove_api(s, "/World/Body/inner", "MassAPI")
    assert physics.validate_schema(s)["ok"]


def test_remove_api_errors():
    s = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Cube.Define(s, "/World/Body").GetPrim()
    with pytest.raises(ValueError, match="no prim at /Nope"):
        remove_api(s, "/Nope", "RigidBodyAPI")
    with pytest.raises(ValueError, match="unknown API schema"):
        remove_api(s, "/World/Body", "TotallyMadeUpAPI")
    with pytest.raises(ValueError, match="not applied"):
        remove_api(s, "/World/Body", "PhysicsRigidBodyAPI")
    # ambiguity is an error, not a guess
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    with pytest.raises(ValueError, match="ambiguous"):
        remove_api(s, "/World/Body", "CollisionAPI")
    # ...but the full name still resolves
    assert remove_api(s, "/World/Body", "PhysicsMeshCollisionAPI")["removed"]
    assert remove_api(s, "/World/Body", "CollisionAPI")["api"] == "PhysicsCollisionAPI"


def test_simulate_output_file_path_not_directory(tmp_path, monkeypatch):
    """`-o recording.usda` must not become a directory anywhere in the chain.

    The low-level simulation path must forward the requested recording file to
    physics_runtime rather than creating a directory with the same suffix.
    """
    from usd_core.config import Config
    from usd_core.session import Session
    from usd_core import physics_runtime

    seen = {}

    def fake_simulate_scene(scene_usd, output_dir, **kw):
        seen["scene_usd"] = scene_usd
        seen["output_dir"] = output_dir
        seen.update(kw)
        return {"ok": True, "failures": [], "warnings": [],
                "metrics": {}, "scene_usd": "", "recording_usda": "",
                "trajectory_jsonl": "", "report_path": "", "executor": "stub",
                "engine": kw.get("engine", "ovphysx"), "n_bodies": 1}

    monkeypatch.setattr(physics_runtime, "simulate_scene", fake_simulate_scene)

    s = Session(Config())
    stage_path = tmp_path / "prop.usda"
    st = Usd.Stage.CreateNew(str(stage_path))
    cube = UsdGeom.Cube.Define(st, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(cube)
    st.GetRootLayer().Save()
    s.open_stage(str(stage_path))

    out = tmp_path / "runs" / "recording.usda"
    r = s.physics_simulate(
        scene=str(stage_path),
        body="/World/Body",
        body_pattern="/World/Body*",
        rest_position=[0.0, 0.0, 0.0],
        world_up=[0.0, 0.0, 1.0],
        output=str(out),
        relax_ovphysx_address_space_limit=True,
    )
    assert r.ok, getattr(r, "issues", r)
    # the session must hand physics_runtime the ORIGINAL -o (file) path and
    # must not have created a directory named recording.usda
    assert seen["output_dir"] == str(out)
    assert not out.is_dir()
    assert seen["scene_usd"] == str(stage_path)
    assert seen["body_path"] == "/World/Body"
    assert seen["body_pattern"] == "/World/Body*"
    assert seen["relax_address_space_limit"] is True
