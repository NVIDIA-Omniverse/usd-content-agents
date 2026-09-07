# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level physics simulation accepts an explicit workflow-authored scenario."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_simulate_scene_uses_explicit_scenario_without_drop_policy(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom
    from usd_core import physics_runtime

    dependency = tmp_path / "body.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    UsdGeom.Cube.Define(dependency_stage, "/World/Body")
    dependency_stage.GetRootLayer().Save()
    scene = tmp_path / "workflow_scenario.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    stage.GetRootLayer().subLayerPaths = [dependency.name]
    stage.GetRootLayer().Save()
    captured: dict[str, object] = {}
    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)

    def fake_remote(scene_usd, **kwargs):
        captured["scene_usd"] = scene_usd
        remote_stage = Usd.Stage.Open(str(scene_usd))
        captured["has_body"] = remote_stage.GetPrimAtPath("/World/Body").IsValid()
        captured["sublayers"] = list(remote_stage.GetRootLayer().subLayerPaths)
        captured.update(kwargs)
        return {
            "trajectory": [(0.0, [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0])],
            "n_bodies": 1,
            "n_steps": 1,
        }

    monkeypatch.setattr(physics_runtime, "evaluate_remote", fake_remote)
    report = physics_runtime.simulate_scene(
        scene,
        str(tmp_path / "out"),
        body_path="/World/Body",
        body_pattern="/World/Body*",
        rest_position=[0.0, 0.0, 1.0],
        world_up=[0.0, 0.0, 1.0],
        remote={"base_url": "http://ovrtx.test"},
    )

    assert captured["scene_usd"] != scene.resolve()
    assert not Path(captured["scene_usd"]).exists()
    assert captured["has_body"] is True
    assert captured["sublayers"] == []
    assert captured["body_pattern"] == "/World/Body*"
    assert report["scenario"]["body_path"] == "/World/Body"
    assert report["scenario"]["body_pattern"] == "/World/Body*"
    assert "drop_height_m" not in report
    assert Path(report["recording_usda"]).is_file()
    recording = Usd.Stage.Open(report["recording_usda"])
    body = recording.GetPrimAtPath("/World/Body")
    assert body.IsValid()
    assert not recording.GetPrimAtPath("/World/Body*").IsValid()
    assert body.GetAttribute("xformOp:translate").GetNumTimeSamples() == 1


def test_apply_operations_requires_explicit_schema_targets(tmp_path: Path) -> None:
    from pxr import Usd, UsdPhysics
    from usd_core.physics import apply_operations

    scene = tmp_path / "authoring.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Body", "Xform")
    stage.DefinePrim("/World/Body/Collider", "Cube")

    record = apply_operations(
        stage,
        {
            "scene_paths": ["/PhysicsScenario"],
            "rigid_bodies": [{"path": "/World/Body", "mass": 1.0}],
            "colliders": [{"path": "/World/Body/Collider", "approximation": "convexHull"}],
            "materials": [{"path": "/Looks/Physics", "static_friction": 0.6}],
            "bindings": [{"target_path": "/World/Body", "material_path": "/Looks/Physics"}],
        },
    )

    assert record["scene"] == ["/PhysicsScenario"]
    assert record["rigid_body"] == ["/World/Body"]
    assert record["collision"] == ["/World/Body/Collider"]
    assert stage.GetPrimAtPath("/PhysicsScenario").IsA(UsdPhysics.Scene)


def test_apply_operations_rejects_all_bad_approximations_before_authoring() -> None:
    from pxr import Usd, UsdPhysics
    from usd_core.physics import apply_operations

    stage = Usd.Stage.CreateInMemory()
    body = stage.DefinePrim("/World/Body", "Xform")
    stage.DefinePrim("/World/Body/Collider", "Cube")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="unknown collision approximation"):
        apply_operations(
            stage,
            {
                "scene_paths": ["/PhysicsScenario"],
                "rigid_bodies": [{"path": "/World/Body", "mass": 1.0}],
                "colliders": [
                    {"path": "/World/Body/Collider", "approximation": "bogus"}
                ],
            },
        )

    assert stage.GetRootLayer().ExportToString() == before
    assert not stage.GetPrimAtPath("/PhysicsScenario").IsValid()
    assert not body.HasAPI(UsdPhysics.RigidBodyAPI)


def test_apply_operations_rejects_binding_to_non_material_prim() -> None:
    from pxr import Usd, UsdShade
    from usd_core.physics import apply_operations

    stage = Usd.Stage.CreateInMemory()
    body = stage.DefinePrim("/World/Body", "Xform")
    stage.DefinePrim("/World/NotMaterial", "Xform")
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="not a Material"):
        apply_operations(
            stage,
            {
                "bindings": [
                    {
                        "target_path": "/World/Body",
                        "material_path": "/World/NotMaterial",
                    }
                ]
            },
        )

    assert stage.GetRootLayer().ExportToString() == before
    assert not body.HasAPI(UsdShade.MaterialBindingAPI)
