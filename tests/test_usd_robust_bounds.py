# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for robust USD bounds used by render camera framing."""

import math

import pytest

try:
    from pxr import Usd, UsdGeom, UsdSkel

    HAS_USD = True
except ImportError:
    HAS_USD = False

from world_understanding.utils.usd.camera import add_corner_view_camera
from world_understanding.utils.usd.prim import get_bbox_from_prim

pytestmark = pytest.mark.skipif(not HAS_USD, reason="USD not available")


def _make_skelroot_stage() -> "Usd.Stage":
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")
    UsdSkel.Root.Define(stage, "/Root/SkelRoot")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/SkelRoot/Mesh")
    mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 2, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    return stage


def test_get_bbox_from_prim_falls_back_to_skelroot_descendant_meshes():
    stage = _make_skelroot_stage()
    root = stage.GetPrimAtPath("/Root")

    # This documents the USD behavior that broke subasset render framing:
    # SkelRoot is Boundable, but its parent bound is empty unless we union
    # descendant mesh bounds explicitly.
    direct_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    direct_range = direct_cache.ComputeWorldBound(root).ComputeAlignedRange()
    assert direct_range.IsEmpty()

    bbox_range = get_bbox_from_prim(root).ComputeAlignedRange()
    assert not bbox_range.IsEmpty()
    assert tuple(bbox_range.GetMin()) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(bbox_range.GetMax()) == pytest.approx((1.0, 2.0, 0.0))


def test_get_bbox_from_prim_returns_zero_bbox_when_no_valid_bounds():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Root")

    bbox_range = get_bbox_from_prim(stage.GetPrimAtPath("/Root")).ComputeAlignedRange()

    assert tuple(bbox_range.GetMin()) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(bbox_range.GetMax()) == pytest.approx((0.0, 0.0, 0.0))


def test_corner_camera_uses_robust_stage_bounds_for_skelroot_stages():
    stage = _make_skelroot_stage()

    camera = add_corner_view_camera(
        stage,
        camera_path="/Camera",
        direction="+x+y+z",
        margin=1.2,
        focal_length=50.0,
        horizontal_aperture=36.0,
        vertical_aperture=36.0,
    )

    transform = UsdGeom.Xformable(camera).GetLocalTransformation(Usd.TimeCode(0))
    position = transform.ExtractTranslation()
    assert all(math.isfinite(position[i]) for i in range(3))
    assert all(abs(position[i]) < 100.0 for i in range(3))
