# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for USD scene construction helpers."""

from __future__ import annotations

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from world_understanding.utils.usd import scene as scene_utils
from world_understanding.utils.usd import stage as stage_utils


def test_add_ground_plane_explicit_z_axis_geometry_and_material() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    prim = scene_utils.add_ground_plane(
        stage,
        center=(1.0, 2.0, 3.0),
        extent=2.0,
        friction=0.7,
        restitution=0.25,
    )

    assert prim.GetPath().pathString == "/World/GroundPlane"
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    assert [(point[0], point[1], point[2]) for point in points] == [
        (-1.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 4.0, 0.0),
        (-1.0, 4.0, 0.0),
    ]
    assert mesh.GetDoubleSidedAttr().Get() is True
    assert UsdPhysics.CollisionAPI(prim)

    material = UsdShade.Material.Get(stage, "/World/GroundPlaneMaterial")
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    assert material_api.GetStaticFrictionAttr().Get() == pytest.approx(0.7)
    assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(0.7)
    assert material_api.GetRestitutionAttr().Get() == pytest.approx(0.25)
    assert material.GetSurfaceOutput().HasConnectedSource()


def test_add_ground_plane_derives_center_extent_for_y_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(
        stage_utils,
        "get_scene_extent",
        lambda _stage: {"bounding_box": {"min": [-1, 1, -2], "max": [3, 5, 8]}},
    )

    prim = scene_utils.add_ground_plane(stage)

    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    assert [(point[0], point[1], point[2]) for point in points] == [
        (-39.0, 0.0, -37.0),
        (-39.0, 0.0, 43.0),
        (41.0, 0.0, 43.0),
        (41.0, 0.0, -37.0),
    ]


def test_add_ground_plane_requires_bbox_or_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(stage_utils, "get_scene_extent", lambda _stage: {})
    with pytest.raises(ValueError, match="stage bbox is empty"):
        scene_utils.add_ground_plane(stage)

    monkeypatch.setattr(
        stage_utils,
        "get_scene_extent",
        lambda _stage: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(ValueError, match="stage bbox is empty"):
        scene_utils.add_ground_plane(stage)
