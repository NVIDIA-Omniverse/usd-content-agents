# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Winding normalization for geometry shared by raster and space ingest."""

from __future__ import annotations

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from usd_core import raster  # noqa: E402
from usd_core.raster import _cube, _iter_geometry  # noqa: E402
from usd_core.space.adapter import scene_geometry  # noqa: E402


@pytest.mark.parametrize(
    ("left_handed", "mirrored"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_iter_geometry_normalizes_orientation_and_mirrors(left_handed, mirrored):
    """Every emitted cube face must retain an outward world-space normal."""
    stage = Usd.Stage.CreateInMemory()
    points, triangles = _cube(2.0)
    if left_handed:
        # In a left-handed USD mesh the opposite vertex order denotes its outward
        # side; author the raw indices accordingly rather than merely setting a token.
        triangles = triangles[:, [0, 2, 1]]
    cube = UsdGeom.Mesh.Define(stage, "/World/Cube")
    cube.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    cube.CreateFaceVertexCountsAttr([3] * len(triangles))
    cube.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    if left_handed:
        cube.GetOrientationAttr().Set(UsdGeom.Tokens.leftHanded)
    if mirrored:
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(-1.0, 1.0, 1.0))

    _path, _ref, points, triangles = next(_iter_geometry(stage, lambda path: path))
    centers = points[triangles].mean(axis=1)
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                       points[triangles[:, 2]] - points[triangles[:, 0]])
    assert np.all(np.einsum("ij,ij->i", normals, centers) > 0.0)


def test_scoped_space_ingest_culls_before_mesh_geometry_is_loaded(monkeypatch):
    """An out-of-scope mesh must never reach the expensive tessellation helper."""
    stage = Usd.Stage.CreateInMemory()
    near = UsdGeom.Cube.Define(stage, "/World/Near")
    UsdGeom.Xformable(near).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    far = UsdGeom.Cube.Define(stage, "/World/Far")
    # USD's default metre scale is centimetres; keep this unambiguously outside the
    # two-metre detector scope after the stage-to-detector conversion.
    UsdGeom.Xformable(far).AddTranslateOp().Set(Gf.Vec3d(1_000.0, 0.0, 0.0))
    original = raster._prim_local_geometry

    def guarded(prim):
        if prim.GetPath().pathString == "/World/Far":
            raise AssertionError("out-of-scope prim was tessellated")
        return original(prim)

    monkeypatch.setattr(raster, "_prim_local_geometry", guarded)
    scene = scene_geometry(stage, lambda path: path,
                           scope=((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
                           scope_pad=0.0)
    assert scene.object_paths == ["/World/Near"]
