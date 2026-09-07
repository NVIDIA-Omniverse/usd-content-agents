# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PR-6: camera control — fit, look-at, orbit, create, use, list.

Cameras are authored into the in-memory stage; `camera list` reflects them. Each test
re-opens its scene so it starts from a clean (no managed camera) state.
"""

from __future__ import annotations

import pytest
from pxr import Usd, UsdGeom
from usd_core.camera import clipping_range
from usd_core.config import Config
from usd_core.session import Session

from conftest import ASSETS, SPRAY


def test_clipping_range_preserves_sub_centimeter_fitted_assets():
    near, far = clipping_range(
        (-0.00006, -0.00018, -0.00001),
        (0.00008, 0.00045, 0.00049),
        (0.00088, 0.001, 0.00124),
        (0.00001, 0.00014, 0.00024),
    )

    assert 1e-6 <= near < far < 0.01


def test_clipping_range_has_stable_span_for_zero_extent_at_camera_plane():
    near, far = clipping_range(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )

    assert near == pytest.approx(1e-6)
    assert far - near == pytest.approx(1e-5)


def test_playback_focus_camera_is_world_space_not_below_animated_default(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body")
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.Cube.Define(stage, "/Body/Geometry")
    translate = body.AddTranslateOp()
    translate.Set((0.0, 0.0, 0.0), Usd.TimeCode(0.0))
    translate.Set((0.0, 0.0, -10.0), Usd.TimeCode(10.0))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(10.0)
    session = Session(Config(project_dir=tmp_path), name="playback-camera")

    focus_range = session._playback_focus_range(stage, "/Body")
    camera_path = session._ensure_camera_on(stage, focus="/Body")

    assert focus_range.GetMin()[2] == pytest.approx(-11.0)
    assert focus_range.GetMax()[2] == pytest.approx(1.0)
    assert camera_path == "/usd_cam"
    assert stage.GetPrimAtPath(camera_path).GetParent().IsPseudoRoot()


def test_playback_focus_range_bounds_extreme_authored_time_span(
    tmp_path, monkeypatch
):
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body")
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.Cube.Define(stage, "/Body/Geometry")
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(1_000_000_000.0)
    session = Session(Config(project_dir=tmp_path), name="bounded-playback-camera")
    observed_frames = []
    real_bbox_cache = UsdGeom.BBoxCache

    class RecordingBBoxCache:
        def __init__(self, *_args, **_kwargs):
            self.time = None

        def SetTime(self, value):
            self.time = value.GetValue()
            observed_frames.append(self.time)

        def ComputeWorldBound(self, _prim):
            return real_bbox_cache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            ).ComputeWorldBound(stage.GetPrimAtPath("/Body"))

    monkeypatch.setattr(UsdGeom, "BBoxCache", RecordingBBoxCache)

    result = session._playback_focus_range(stage, "/Body")

    assert result is not None
    assert len(observed_frames) == 256
    assert observed_frames[0] == 0.0
    assert observed_frames[-1] == 1_000_000_000.0


@pytest.mark.parametrize("asset", ASSETS, ids=[a.id for a in ASSETS])
def test_camera_fit_frames_the_scene(project, asset):
    project.open(asset)
    env = project.cli("camera", "fit", "@n1", json=True, expect_ok=True).json()
    assert env["summary"]["camera"].endswith("usd_cam")
    assert env["summary"]["framed"] == ["@n1"]
    assert isinstance(env["summary"]["distance"], (int, float))
    assert env["summary"]["distance"] > 0


def test_camera_orbit_reports_position(project):
    project.open(SPRAY)
    env = project.cli(
        "camera", "orbit", "@n1", "--az=45", "--el=20", json=True, expect_ok=True
    ).json()
    assert env["summary"]["az"] == 45.0
    assert env["summary"]["el"] == 20.0
    assert env["summary"]["distance"] > 0
    assert env["summary"]["camera"].endswith("usd_cam")
    transform = env["summary"]["camera_world_transform"]
    assert len(transform) == 4
    assert all(len(row) == 4 for row in transform)


def test_camera_look_at(project):
    project.open(SPRAY)
    env = project.cli("camera", "look-at", "@n1", json=True, expect_ok=True).json()
    assert env["summary"]["target"] == "@n1"
    assert env["summary"]["camera"].endswith("usd_cam")


def test_camera_create_then_use_and_list(project):
    project.open(SPRAY)
    created = project.cli(
        "camera", "create", "--name", "hero", "--at=3,3,3", "--look-at=0,0,0",
        json=True, expect_ok=True,
    ).json()
    cam_path = created["summary"]["camera"]
    assert cam_path.endswith("/hero")

    use = project.cli("camera", "use", cam_path, json=True, expect_ok=True).json()
    assert use["summary"]["active"] == cam_path

    listing = project.cli("camera", "list", json=True, expect_ok=True).json()
    assert cam_path in listing["data"]["cameras"]
    assert listing["summary"]["active"] == cam_path
    assert listing["summary"]["count"] >= 1
