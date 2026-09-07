# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic geometry helpers for tests/examples (no USD, no Kit).

Build analytic scenes with known free-space answers so the detector can be
validated without any external asset or Kit runtime.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from usd_core.space.geometry import SceneGeometry


def box_mesh(lo: Sequence[float], hi: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned closed box as (vertices (8,3), triangles (12,3))."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        dtype=np.float32,
    )
    f = np.array(
        [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
         [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
        dtype=np.int32,
    )
    return v, f


def ramp_mesh(lo: Sequence[float], hi: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """A single inclined surface (2 triangles) rising along +x from ``lo`` to ``hi``.

    The surface spans x in [x0,x1], y in [y0,y1], with height z0 at x0 rising
    linearly to z1 at x1 — i.e. a ramp of slope ``(z1-z0)/(x1-x0)``. One object id.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array(
        [[x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]],
        dtype=np.float32,
    )
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return v, f


def scene_from_meshes(meshes: List[Tuple[np.ndarray, np.ndarray]],
                      names: List[str] | None = None) -> SceneGeometry:
    """Compose a SceneGeometry from arbitrary (vertices, triangles) meshes."""
    verts: List[np.ndarray] = []
    tris: List[np.ndarray] = []
    oids: List[np.ndarray] = []
    paths: List[str] = []
    vbase = 0
    for i, (v, f) in enumerate(meshes):
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int32)
        verts.append(v)
        tris.append((f + vbase).astype(np.int32))
        oids.append(np.full(f.shape[0], i, dtype=np.int32))
        paths.append(names[i] if names else f"mesh_{i}")
        vbase += v.shape[0]
    return SceneGeometry(
        vertices=np.concatenate(verts).astype(np.float32),
        indices=np.concatenate(tris).astype(np.int32),
        tri_object_id=np.concatenate(oids).astype(np.int32),
        object_paths=paths,
        meters_per_unit=1.0,
    )


def scene_from_boxes(boxes: List[Tuple[Sequence[float], Sequence[float]]], names: List[str] | None = None) -> SceneGeometry:
    """Compose a SceneGeometry from AABB boxes; each box is a distinct object id."""
    verts: List[np.ndarray] = []
    tris: List[np.ndarray] = []
    oids: List[np.ndarray] = []
    paths: List[str] = []
    vbase = 0
    for i, (lo, hi) in enumerate(boxes):
        v, f = box_mesh(lo, hi)
        verts.append(v)
        tris.append((f + vbase).astype(np.int32))
        oids.append(np.full(f.shape[0], i, dtype=np.int32))
        paths.append(names[i] if names else f"box_{i}")
        vbase += v.shape[0]
    return SceneGeometry(
        vertices=np.concatenate(verts).astype(np.float32),
        indices=np.concatenate(tris).astype(np.int32),
        tri_object_id=np.concatenate(oids).astype(np.int32),
        object_paths=paths,
        meters_per_unit=1.0,
    )
