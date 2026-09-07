# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the auto-computed clipping range used by
``add_side_view_camera`` / ``_setup_side_view_camera``.

USD clipping planes are measured along the camera's view axis, NOT
Euclidean distance from the camera position. The autoclip path in
``camera.py`` projects bbox corners onto the view axis. These tests
guard against regressing back to Euclidean distance.
"""

from __future__ import annotations

import pytest

try:
    from pxr import Usd, UsdGeom

    HAS_USD = True
except ImportError:
    HAS_USD = False

pytestmark = pytest.mark.skipif(not HAS_USD, reason="USD not available")


def test_direction_weight_parser_ignores_malformed_numeric_token() -> None:
    from world_understanding.utils.usd.camera import _parse_direction_weights

    assert _parse_direction_weights("+1.2.3x-y+0.5z") == (1.0, -1.0, 0.5)


def test_direction_weight_parser_ignores_terminal_malformed_token() -> None:
    from world_understanding.utils.usd.camera import _parse_direction_weights

    assert _parse_direction_weights("+1.2.3x") == (1.0, 1.0, 1.0)


def _build_unit_cube_stage() -> Usd.Stage:
    """Unit cube centered at origin, stage Z-up."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Cube.Define(stage, "/Cube")
    return stage


def _clipping_range(camera_prim: UsdGeom.Camera) -> tuple[float, float]:
    rng = camera_prim.GetClippingRangeAttr().Get()
    return float(rng[0]), float(rng[1])


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y", "+z", "-z"])
def test_single_axis_camera_does_not_clip_scene_front(direction: str) -> None:
    """For an axis-aligned side camera, the auto-near plane must not fall
    behind the front face of the unit cube (which is exactly 1 unit
    closer than the camera origin along the view axis).

    With a margin near 0 (10% in the test below), Euclidean-distance
    clipping placed near *past* the actual front face when the camera
    sat off-axis from the cube center, occluding the scene. View-axis
    projection keeps near at the front face minus the margin.
    """
    from world_understanding.utils.usd.camera import add_side_view_camera

    stage = _build_unit_cube_stage()
    cam = add_side_view_camera(
        stage,
        camera_path=f"/Cameras/Side_{direction.replace('+', 'p').replace('-', 'n')}",
        direction=direction,
        margin=1.0,
        near_clip_margin=0.0,
        far_clip_margin=0.0,
    )
    near, far = _clipping_range(cam)

    # UsdGeom.Cube default size=2 → bbox extends ±1 along every axis.
    # With margin=1.0 the camera sits beyond the bbox; the near plane
    # (with margin=0) should land at scene-front along the view axis,
    # i.e., far - near should equal exactly 2 (the cube depth along
    # view). Euclidean clipping placed near larger than the true axial
    # depth, so far - near would be smaller than the true cube extent.
    assert near > 0.0, f"degenerate near plane for direction {direction}"
    assert far > near, f"degenerate clipping range for direction {direction}"
    assert (far - near) == pytest.approx(2.0, abs=1e-3), (
        f"clipping span {far - near} != cube depth 2.0 along {direction}; "
        "Euclidean distance regression suspected."
    )


def test_corner_camera_near_plane_contains_full_scene() -> None:
    """For a +x+y+z corner camera, near must be small enough that no
    bbox corner sits in front of it along the view axis.

    The camera looks at the origin, so the world-space view direction
    is just (origin - cam_pos).normalized — derive it that way to keep
    the test independent of how the rotation matrix is laid out.
    """
    from world_understanding.utils.usd.camera import add_corner_view_camera

    stage = _build_unit_cube_stage()
    cam = add_corner_view_camera(
        stage,
        camera_path="/Cameras/Corner_pxpypz",
        direction="+x+y+z",
        margin=1.0,
        near_clip_margin=0.0,
        far_clip_margin=0.0,
    )
    near, far = _clipping_range(cam)

    cam_xform = UsdGeom.Xformable(cam.GetPrim())
    cam_matrix = cam_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cam_pos = cam_matrix.ExtractTranslation()
    # Camera frames the unit cube centered at origin → look_at = origin.
    cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
    norm = (cx * cx + cy * cy + cz * cz) ** 0.5
    # View direction = (origin - cam_pos)/|...| = -cam_pos/|cam_pos|.
    view = (-cx / norm, -cy / norm, -cz / norm)

    # All 8 corners of the unit cube projected onto view axis must fall
    # within [near, far].
    # UsdGeom.Cube has default size=2.0 → bbox extends ±1 along each
    # axis. (Not ±0.5 as a "unit cube" name might suggest.)
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                depth = dx * view[0] + dy * view[1] + dz * view[2]
                assert near <= depth + 1e-4, (
                    f"corner ({x},{y},{z}) at depth {depth} clipped by near {near}"
                )
                assert depth <= far + 1e-4, (
                    f"corner ({x},{y},{z}) at depth {depth} clipped by far {far}"
                )


def test_near_plane_uses_view_axis_not_euclidean() -> None:
    """Direct numeric check: with the unit cube and a +x camera, the
    front face sits exactly 1.0 closer along view axis than the camera
    origin, while the off-axis corners are sqrt(0.5² + 0.5²) ≈ 0.707
    further by Euclidean distance. The auto-near plane must reflect the
    axial 1.0, not the Euclidean ≈ 1.225.
    """
    from world_understanding.utils.usd.camera import add_side_view_camera

    stage = _build_unit_cube_stage()
    cam = add_side_view_camera(
        stage,
        camera_path="/Cameras/Side_FrontFace",
        direction="+x",
        margin=1.0,
        near_clip_margin=0.0,
        far_clip_margin=0.0,
    )
    near, far = _clipping_range(cam)

    cam_pos = (
        UsdGeom.Xformable(cam.GetPrim())
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    cam_x = float(cam_pos[0])
    # UsdGeom.Cube default size=2 → front face at x=+1, back at x=-1.
    front_depth = cam_x - 1.0
    back_depth = cam_x + 1.0
    # Diagonal corners (off-axis) sit at sqrt((front_depth)² + 0.5² + 0.5²)
    # > front_depth Euclidean. View-axis projection keeps them at
    # front_depth (since y, z are perpendicular to view). So near must
    # equal front_depth (no margin).
    assert near == pytest.approx(front_depth, abs=1e-4), (
        f"near={near} but front face at axial depth {front_depth}; "
        "regression to Euclidean distance suspected."
    )
    assert far == pytest.approx(back_depth, abs=1e-4)


@pytest.mark.parametrize("camera_kind", ["side", "corner"])
def test_auto_clipping_keeps_millimeter_scale_asset_visible(camera_kind: str) -> None:
    """The near-plane floor must scale below a sub-millimeter scene depth."""
    from world_understanding.utils.usd.camera import (
        add_corner_view_camera,
        add_side_view_camera,
    )

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cube = UsdGeom.Cube.Define(stage, "/TinyCube")
    cube.CreateSizeAttr(0.001)
    if camera_kind == "side":
        camera = add_side_view_camera(
            stage,
            camera_path="/Cameras/TinySide",
            direction="+x",
            margin=1.2,
        )
    else:
        camera = add_corner_view_camera(
            stage,
            camera_path="/Cameras/TinyCorner",
            direction="+x+y+z",
            margin=1.2,
        )

    near, far = _clipping_range(camera)

    assert 0.0 < near < far < 0.01
