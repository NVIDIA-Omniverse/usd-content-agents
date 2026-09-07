# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic AOV helpers rasterize tessellated Gprims for OVRTX render artifacts."""

from __future__ import annotations

import numpy as np
import pytest

GPRIMS = ["Cube", "Sphere", "Cylinder", "Cone", "Capsule"]


def _scene_with(prim_type: str):
    from pxr import Usd, UsdGeom
    from usd_core.camera import author_camera, fit_distance, orbit_position

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    getattr(UsdGeom, prim_type).Define(stage, "/w/shape")
    cam = author_camera(stage, "/w/cam", orbit_position((0, 0, 0), fit_distance([4, 4, 4]),
                                                        30, 20, up_axis_y=True), (0, 0, 0))
    return stage, cam.GetPrim().GetPath().pathString


@pytest.mark.parametrize("prim_type", GPRIMS)
def test_gprim_appears_in_aovs(prim_type):
    """Each analytic Gprim tessellates into non-empty depth + segmentation + normals."""
    from usd_core import raster

    stage, cam = _scene_with(prim_type)
    out = raster.render_aovs(stage, cam, 160, 120,
                             ["depth", "segmentation", "normals"], lambda p: p)
    seg = np.asarray(out["segmentation"])
    depth = np.asarray(out["depth"])
    linear_depth = out["linear_depth"]
    normals = np.asarray(out["normals"])
    assert (seg.sum(axis=2) > 0).sum() > 200, f"{prim_type}: empty segmentation"
    assert (depth > 0).sum() > 200, f"{prim_type}: empty depth"
    assert linear_depth.dtype == np.float32
    assert np.isfinite(linear_depth).sum() > 200, f"{prim_type}: empty linear depth"
    assert np.all(linear_depth[np.isfinite(linear_depth)] > 0)
    assert (normals.sum(axis=2) > 0).sum() > 200, f"{prim_type}: empty normals"
    assert "/w/shape" in out["legend"]


def test_linear_depth_preserves_metric_scale_and_nan_background():
    from usd_core import raster

    encoded = np.full(2, raster._ENC_EMPTY, dtype=np.int64)
    encoded[0] = np.int64(
        int(np.asarray(2.5, dtype=np.float32).view(np.uint32)) << raster._ID_BITS
    )

    depth = raster._linear_depth_meters(
        encoded,
        2,
        1,
        meters_per_unit=0.01,
    )

    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(0.025)
    assert np.isnan(depth[0, 1])


def test_rasterizer_uses_reciprocal_perspective_depth_interpolation():
    """Guard the z-buffer against the old affine camera-depth interpolation."""
    from usd_core import raster

    # Pixel (0, 0), sampled at (0.5, 0.5), has barycentrics
    # (0.5, 0.25, 0.25). Perspective-correct camera depth is therefore
    # 1 / (0.5 / 1 + 0.25 / 2 + 0.25 / 4), not the affine value 2.0.
    projected = np.asarray(
        [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0], [0.0, 2.0, 4.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    encoded = np.full(4, raster._ENC_EMPTY, dtype=np.int64)

    raster._rasterize_mesh(
        projected,
        triangles,
        encoded,
        None,
        None,
        (1, 2, 3),
        None,
        2,
        2,
    )
    depth = raster._linear_depth_meters(encoded, 2, 2, meters_per_unit=1.0)

    assert depth[0, 0] == pytest.approx(1.0 / 0.6875, rel=1e-6)
    assert depth[0, 0] != pytest.approx(2.0)
