# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from pydantic import ValidationError

from geometry_repair.advanced_profiles import ContactRichProbeInput, SourceEvidence
from geometry_repair.runtime import _author_contact_insertion_scene


def _collision_cube(path: Path, root_path: str) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, f"{root_path}/Collision")
    cube.CreateSizeAttr(0.02)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    return path


def _probe(**updates) -> ContactRichProbeInput:
    values = {
        "probe_id": "peg_hole",
        "moving_part_id": "peg",
        "receiver_part_id": "hole",
        "receiver_collision_paths": ["/Receiver/Collision"],
        "axis_origin_world_m": (0.0, 0.0, 0.0),
        "axis_world": (0.0, 0.0, 1.0),
        "approach_direction": "along_axis",
        "path_points_world_m": [(0.0, 0.0, -0.02), (0.0, 0.0, 0.0)],
        "axis_tolerance_m": 0.001,
        "angular_tolerance_deg": 2.0,
        "moving_envelope_radius_m": 0.01,
        "required_radial_clearance_m": 0.001,
        "maximum_clearance_erosion_m": 0.0002,
        "seated_stop_tolerance_m": 0.001,
        "minimum_protected_feature_m": 0.001,
        "collision_representation": "static_triangle_mesh",
        "sample_step_m": 0.001,
        "evidence": [
            SourceEvidence(
                fact_kind="source_fact",
                source_ref="fixture#peg_hole",
                summary="Test-only authored dimensions.",
                confidence=1.0,
            )
        ],
    }
    values.update(updates)
    return ContactRichProbeInput(**values)


def test_contact_probe_requires_positive_clearance_after_erosion() -> None:
    with pytest.raises(ValidationError, match="maximum_clearance_erosion_m"):
        _probe(maximum_clearance_erosion_m=0.001)


def test_contact_insertion_scene_separates_static_and_dynamic_collision(tmp_path: Path) -> None:
    receiver = _collision_cube(tmp_path / "receiver.usda", "/ReceiverSource")
    moving = _collision_cube(tmp_path / "moving.usda", "/MovingSource")

    output = _author_contact_insertion_scene(
        receiver,
        moving,
        tmp_path / "contact_scene.usda",
        insertion_axis=(0.0, 0.0, 1.0),
        seated_root_translation_m=(0.1, -0.2, 0.3),
        separation_m=0.04,
        approach_speed_m_s=0.05,
        moving_mass_kg=0.1,
    )

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    receiver_collider = stage.GetPrimAtPath("/Receiver/Collider_0000")
    moving_collider = stage.GetPrimAtPath("/Male/Collider_0000")
    assert receiver_collider.HasAPI(UsdPhysics.CollisionAPI)
    assert moving_collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(receiver_collider).GetApproximationAttr().Get() == "none"
    assert (
        UsdPhysics.MeshCollisionAPI(moving_collider).GetApproximationAttr().Get()
        == "convexDecomposition"
    )
    male = stage.GetPrimAtPath("/Male")
    assert male.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(male).GetMassAttr().Get() == pytest.approx(0.1)
    assert UsdPhysics.RigidBodyAPI(male).GetVelocityAttr().Get() == Gf.Vec3f(0.0, 0.0, 0.05)
    transform = UsdGeom.Xformable(male).GetLocalTransformation()
    assert transform.ExtractTranslation() == Gf.Vec3d(0.1, -0.2, 0.26)
