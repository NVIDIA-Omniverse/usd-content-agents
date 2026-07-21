# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for USD camera extraction and projection helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from world_understanding.functions.graphics import usd_camera


def _write_camera_stage(path: Path, *, orthographic: bool = False) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera.GetHorizontalApertureAttr().Set(40.0)
    camera.GetVerticalApertureAttr().Set(20.0)
    camera.GetHorizontalApertureOffsetAttr().Set(2.0)
    camera.GetVerticalApertureOffsetAttr().Set(-1.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.25, 500.0))
    if orthographic:
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))
    stage.GetRootLayer().Save()
    return path


def test_extract_project_unproject_and_json_roundtrip(tmp_path: Path) -> None:
    usd_path = _write_camera_stage(tmp_path / "camera.usda")

    params = usd_camera.extract_camera_parameters(
        str(usd_path),
        "/World/Camera",
        image_width=1000,
        image_height=None,
        time_code=0,
    )

    assert params["projection"] == "perspective"
    assert params["image_height"] == 500
    assert params["near"] == pytest.approx(0.25)
    assert params["far"] == pytest.approx(500.0)
    assert params["K"][0][0] == pytest.approx(1250.0)
    assert params["K"][0][2] == pytest.approx(450.0)
    assert params["fov_x_rad"] == pytest.approx(2.0 * math.atan(0.4))
    assert len(params["camera_world_transform"]) == 4

    u, v, depth = usd_camera.project_point([0, 0, -10], params)
    assert depth == pytest.approx(-10.0)
    assert u == pytest.approx(params["K"][0][2])
    assert v == pytest.approx(params["K"][1][2])
    assert all(
        math.isnan(value) for value in usd_camera.project_point([0, 0, 0], params)[:2]
    )

    origin, direction = usd_camera.unproject_pixel((u, v), params)
    assert origin == pytest.approx((0.0, 0.0, 0.0))
    assert direction == pytest.approx((0.0, 0.0, -1.0))
    point = usd_camera.unproject_pixel((u, v), params, depth=10.0)
    assert point == pytest.approx((0.0, 0.0, -10.0))

    json_path = tmp_path / "camera.json"
    usd_camera.save_camera_json(params, json_path)
    assert usd_camera.load_camera_json(json_path)["camera_path"] == "/World/Camera"


def test_orthographic_projection_and_validation_errors(tmp_path: Path) -> None:
    usd_path = _write_camera_stage(tmp_path / "ortho.usda", orthographic=True)
    params = usd_camera.extract_camera_parameters(
        str(usd_path),
        "/World/Camera",
        image_width=100,
        image_height=50,
    )

    assert params["projection"] == "orthographic"
    assert params["is_orthographic_like"] is True
    u, v, depth = usd_camera.project_point([1, 2, -10], params)
    assert depth == pytest.approx(-10.0)
    assert u != params["K"][0][2]
    assert v != params["K"][1][2]

    origin, direction = usd_camera.unproject_pixel([u, v], params)
    assert direction == pytest.approx((0.0, 0.0, -1.0))
    assert origin[2] == pytest.approx(0.0)
    point = usd_camera.unproject_pixel([u, v], params, depth=5.0)
    assert point[2] == pytest.approx(-5.0)

    with pytest.raises(ValueError, match="world_point must be 3D"):
        usd_camera.project_point([1, 2], params)
    with pytest.raises(ValueError, match="pixel_coord must be 2D"):
        usd_camera.unproject_pixel([1, 2, 3], params)
    with pytest.raises(ValueError, match="Camera prim not found"):
        usd_camera.extract_camera_parameters(str(usd_path), "/World", 100, 50)

    matrix_values = usd_camera._matrix4_to_list(Gf.Matrix4d(1.0))
    assert (
        usd_camera._matrix4_to_list(usd_camera._list_to_matrix4(matrix_values))
        == matrix_values
    )


def test_extract_camera_parameters_defaults_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAttr:
        def __init__(self, value: object) -> None:
            self.value = value

        def Get(self, _time: object = None) -> object:
            return self.value

    class FakeCamera:
        def __init__(self, _prim: object) -> None:
            pass

        def GetProjectionAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetFocalLengthAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetHorizontalApertureAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetVerticalApertureAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetHorizontalApertureOffsetAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetVerticalApertureOffsetAttr(self) -> FakeAttr:
            return FakeAttr(None)

        def GetClippingRangeAttr(self) -> FakeAttr:
            return FakeAttr(None)

    class FakePrim:
        def IsA(self, _schema: object) -> bool:
            return True

    class FakeStage:
        def GetPrimAtPath(self, _path: object) -> FakePrim:
            return FakePrim()

    class FakeXformCache:
        def __init__(self, _time: object) -> None:
            pass

        def GetLocalToWorldTransform(self, _prim: object) -> Gf.Matrix4d:
            return Gf.Matrix4d(1.0)

    monkeypatch.setattr(usd_camera.Usd.Stage, "Open", lambda _path: FakeStage())
    monkeypatch.setattr(usd_camera.UsdGeom, "Camera", FakeCamera)
    monkeypatch.setattr(usd_camera.UsdGeom, "XformCache", FakeXformCache)

    params = usd_camera.extract_camera_parameters("fake.usda", "/Camera", 360)
    assert params["projection"] == "perspective"
    assert params["image_height"] == 240
    assert params["near"] == pytest.approx(0.1)
    assert params["far"] == pytest.approx(10000.0)
    assert np.array(params["world_to_camera"]).shape == (4, 4)

    monkeypatch.setattr(usd_camera.Usd.Stage, "Open", lambda _path: None)
    with pytest.raises(ValueError, match="Failed to open USD file"):
        usd_camera.extract_camera_parameters("missing.usda", "/Camera", 100)
