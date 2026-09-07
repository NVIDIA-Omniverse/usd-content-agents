# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""World-space bounding-box helpers (PR-5 precursor; used by camera fit).

Vendored/adapted from world-understanding
`world_understanding/functions/graphics/usd_spatial.py` (get_world_bbox) — pure pxr,
no GPU. Per the Q2 decision (vendor minimal code), copied rather than imported.
"""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom


def get_bbox(stage: Usd.Stage, prim_path: str, time: Usd.TimeCode | None = None) -> Gf.Range3d | None:
    """World-space aligned bbox range for a prim, or None if not computable."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    tc = time if time is not None else Usd.TimeCode.Default()
    cache = UsdGeom.BBoxCache(tc, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return rng


def get_world_bbox(stage: Usd.Stage, prim_path: str) -> dict | None:
    """Plain-dict bbox: {min, max, size, center, volume}. Mirrors usd_spatial.py."""
    rng = get_bbox(stage, prim_path)
    if rng is None:
        return None
    bmin, bmax = rng.GetMin(), rng.GetMax()
    size = bmax - bmin
    center = (bmin + bmax) / 2.0
    return {
        "min": [bmin[0], bmin[1], bmin[2]],
        "max": [bmax[0], bmax[1], bmax[2]],
        "size": [size[0], size[1], size[2]],
        "center": [center[0], center[1], center[2]],
        "volume": float(size[0] * size[1] * size[2]),
    }


def combined_bbox(stage: Usd.Stage, prim_paths: list[str]) -> Gf.Range3d | None:
    """Union of the world bboxes of several prims (for `camera fit @a @b @c`)."""
    union: Gf.Range3d | None = None
    for p in prim_paths:
        rng = get_bbox(stage, p)
        if rng is None:
            continue
        union = Gf.Range3d(rng) if union is None else Gf.Range3d.GetUnion(union, rng)
    return union


def scene_bbox(stage: Usd.Stage) -> Gf.Range3d | None:
    """World bbox of the whole scene (default prim, else pseudo-root)."""
    root = stage.GetDefaultPrim()
    path = root.GetPath().pathString if root and root.IsValid() else "/"
    return get_bbox(stage, path)
