# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

try:
    from pxr import Gf, Usd, UsdGeom

    HAS_USD = True
except ImportError:
    HAS_USD = False

pytestmark = pytest.mark.skipif(not HAS_USD, reason="USD not available")


def _build_cube_stage(up_axis: str | None = None) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    if up_axis is not None:
        UsdGeom.SetStageUpAxis(stage, up_axis)
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    return stage, cube.GetPrim()


def _clipping_range(
    camera: UsdGeom.Camera, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> tuple[float, float]:
    value = camera.GetClippingRangeAttr().Get(time)
    return float(value[0]), float(value[1])


def test_camera_helpers_forward_explicit_bbox_purposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.utils.usd.camera as camera_module

    stage, cube_prim = _build_cube_stage(UsdGeom.Tokens.z)
    real_get_bbox = camera_module.get_bbox_from_prim
    recorded_purposes: list[tuple[str, ...] | None] = []

    def recording_get_bbox(prim: Usd.Prim, **kwargs: Any) -> Gf.BBox3d:
        purposes = kwargs.get("included_purposes")
        recorded_purposes.append(tuple(purposes) if purposes is not None else None)
        return real_get_bbox(prim, **kwargs)

    monkeypatch.setattr(camera_module, "get_bbox_from_prim", recording_get_bbox)
    purposes = ("default", "render")
    camera_module.add_side_view_camera(
        stage,
        camera_path="/Cameras/SidePurposes",
        included_purposes=purposes,
    )
    camera_module.add_focused_side_view_camera(
        cube_prim,
        camera_path="/Cameras/FocusedSidePurposes",
        included_purposes=purposes,
    )
    camera_module.add_corner_view_camera(
        stage,
        camera_path="/Cameras/CornerPurposes",
        included_purposes=purposes,
    )
    camera_module.add_focused_corner_view_camera(
        cube_prim,
        camera_path="/Cameras/FocusedCornerPurposes",
        included_purposes=purposes,
    )

    assert recorded_purposes == [purposes, purposes, purposes, purposes]


def test_side_framing_caps_effective_scene_size() -> None:
    from world_understanding.utils.usd.camera import (
        compute_camera_framing_position_sides,
    )

    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (100.0, 200.0, 300.0)

    uncapped_position, look_at = compute_camera_framing_position_sides(
        bbox_min,
        bbox_max,
        direction="+x",
    )
    capped_position, capped_look_at = compute_camera_framing_position_sides(
        bbox_min,
        bbox_max,
        direction="+x",
        max_scene_size=10.0,
    )

    assert capped_position[0] < uncapped_position[0]
    assert capped_look_at == look_at


def test_direction_weight_parser_skips_tokens_without_axes() -> None:
    from world_understanding.utils.usd.camera import _parse_direction_weights

    assert _parse_direction_weights("? -2y !") == (1.0, -2.0, 1.0)


def test_corner_framing_supports_per_axis_overrides() -> None:
    from world_understanding.utils.usd.camera import (
        compute_camera_framing_position_corners,
    )

    camera_position, look_at = compute_camera_framing_position_corners(
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        cam_x=10.0,
        cam_y=11.0,
        cam_z=12.0,
        target_x=1.0,
        target_y=2.0,
        target_z=3.0,
        max_scene_size=0.5,
    )

    assert camera_position == (10.0, 11.0, 12.0)
    assert look_at == (1.0, 2.0, 3.0)


def test_side_camera_updates_existing_prim_at_time_sample() -> None:
    from world_understanding.utils.usd.camera import add_side_view_camera

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.z)
    camera = add_side_view_camera(
        stage,
        camera_path="/Cameras/Side",
        direction="+x",
        near_clip=5.0,
        far_clip=1.0,
    )
    assert _clipping_range(camera) == pytest.approx((5.0, 5.5))

    update_time = Usd.TimeCode(4.0)
    updated_camera = add_side_view_camera(
        stage,
        camera_path="/Cameras/Side",
        direction="-y",
        focal_length=35.0,
        horizontal_aperture=20.0,
        vertical_aperture=21.0,
        near_clip=2.0,
        far_clip=3.0,
        time=update_time,
    )

    assert updated_camera.GetPrim() == camera.GetPrim()
    assert updated_camera.GetFocalLengthAttr().Get(update_time) == 35.0
    assert updated_camera.GetHorizontalApertureAttr().Get(update_time) == 20.0
    assert updated_camera.GetVerticalApertureAttr().Get(update_time) == 21.0
    assert _clipping_range(updated_camera, update_time) == pytest.approx((2.0, 3.0))


def test_focused_side_camera_uses_target_prim_stage() -> None:
    from world_understanding.utils.usd.camera import add_focused_side_view_camera

    _stage, cube_prim = _build_cube_stage(UsdGeom.Tokens.z)

    camera = add_focused_side_view_camera(
        cube_prim,
        camera_path="/Cameras/FocusedSide",
        direction="-y",
        near_clip=0.1,
        far_clip=10.0,
    )

    assert str(camera.GetPrim().GetPath()) == "/Cameras/FocusedSide"
    assert camera.GetPrim().GetStage() == cube_prim.GetStage()


def test_side_camera_uses_fallback_up_for_y_up_parallel_view() -> None:
    from world_understanding.utils.usd.camera import add_side_view_camera

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.y)

    camera = add_side_view_camera(
        stage,
        camera_path="/Cameras/YUpSide",
        direction="+y",
        near_clip=0.1,
        far_clip=20.0,
    )

    matrix = UsdGeom.Xformable(camera).GetLocalTransformation()
    right = Gf.Vec3d(matrix[0][0], matrix[1][0], matrix[2][0])
    assert right.GetLength() == pytest.approx(1.0)


def test_print_camera_prim_info_reports_transform_vectors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from world_understanding.utils.usd.camera import (
        add_side_view_camera,
        print_camera_prim_info,
    )

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.z)
    camera = add_side_view_camera(
        stage, camera_path="/Cameras/Printable", direction="+z"
    )

    print_camera_prim_info(camera)

    output = capsys.readouterr().out
    assert "Position:" in output
    assert "Rotation:" in output
    assert "Forward:" in output
    assert "Up:" in output
    assert "Right:" in output


def test_corner_camera_updates_existing_prim_at_time_sample() -> None:
    from world_understanding.utils.usd.camera import add_corner_view_camera

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.z)
    camera = add_corner_view_camera(
        stage,
        camera_path="/Cameras/Corner",
        direction="+x+y+z",
        near_clip=4.0,
        far_clip=1.0,
    )
    assert _clipping_range(camera) == pytest.approx((4.0, 4.4))

    update_time = Usd.TimeCode(7.0)
    updated_camera = add_corner_view_camera(
        stage,
        camera_path="/Cameras/Corner",
        direction="-x+y-z",
        focal_length=44.0,
        horizontal_aperture=22.0,
        vertical_aperture=23.0,
        near_clip=1.0,
        far_clip=2.0,
        time=update_time,
    )

    assert updated_camera.GetPrim() == camera.GetPrim()
    assert updated_camera.GetFocalLengthAttr().Get(update_time) == 44.0
    assert updated_camera.GetHorizontalApertureAttr().Get(update_time) == 22.0
    assert updated_camera.GetVerticalApertureAttr().Get(update_time) == 23.0
    assert _clipping_range(updated_camera, update_time) == pytest.approx((1.0, 2.0))


def test_corner_camera_uses_fallback_up_for_z_up_parallel_view() -> None:
    from world_understanding.utils.usd.camera import add_corner_view_camera

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.z)

    camera = add_corner_view_camera(
        stage,
        camera_path="/Cameras/ZUpFallback",
        cam_x=0.0,
        cam_y=0.0,
        cam_z=10.0,
        target_x=0.0,
        target_y=0.0,
        target_z=0.0,
        near_clip=0.1,
        far_clip=20.0,
    )

    matrix = UsdGeom.Xformable(camera).GetLocalTransformation()
    right = Gf.Vec3d(matrix[0][0], matrix[1][0], matrix[2][0])
    assert right.GetLength() == pytest.approx(1.0)


def test_corner_camera_uses_fallback_up_for_y_up_parallel_view() -> None:
    from world_understanding.utils.usd.camera import add_corner_view_camera

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.y)

    camera = add_corner_view_camera(
        stage,
        camera_path="/Cameras/YUpFallback",
        cam_x=0.0,
        cam_y=10.0,
        cam_z=0.0,
        target_x=0.0,
        target_y=0.0,
        target_z=0.0,
        near_clip=0.1,
        far_clip=20.0,
    )

    matrix = UsdGeom.Xformable(camera).GetLocalTransformation()
    right = Gf.Vec3d(matrix[0][0], matrix[1][0], matrix[2][0])
    assert right.GetLength() == pytest.approx(1.0)


def test_focused_corner_camera_uses_target_prim_stage() -> None:
    from world_understanding.utils.usd.camera import add_focused_corner_view_camera

    _stage, cube_prim = _build_cube_stage(UsdGeom.Tokens.z)

    camera = add_focused_corner_view_camera(
        cube_prim,
        camera_path="/Cameras/FocusedCorner",
        direction="-x+y-z",
        near_clip=0.1,
        far_clip=20.0,
    )

    assert str(camera.GetPrim().GetPath()) == "/Cameras/FocusedCorner"
    assert camera.GetPrim().GetStage() == cube_prim.GetStage()


def test_camera_discovery_helpers_return_camera_prims_and_paths() -> None:
    from world_understanding.utils.usd.camera import (
        add_corner_view_camera,
        add_side_view_camera,
        get_all_camera_paths,
        get_all_cameras,
    )

    stage, _cube = _build_cube_stage(UsdGeom.Tokens.z)
    add_side_view_camera(stage, camera_path="/Cameras/Side", direction="+x")
    add_corner_view_camera(stage, camera_path="/Cameras/Corner", direction="+x+y+z")

    paths = {str(path) for path in get_all_camera_paths(stage)}
    cameras = get_all_cameras(stage)

    assert paths == {"/Cameras/Side", "/Cameras/Corner"}
    assert {str(camera.GetPrim().GetPath()) for camera in cameras} == paths
