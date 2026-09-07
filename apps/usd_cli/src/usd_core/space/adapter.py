# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD stage → `SceneGeometry`, and detector-frame results back to stage frame.

This module is the whole USD-facing half of empty-space detection. It exists
because upstream's `esd.usd_scene.load_scene` is not usable on usd-cli's scenes:
it walks a plain `stage.Traverse()` and handles `UsdGeom.Mesh` only, so it
silently returns *near-empty geometry* on native-instanced stages, and misses
every analytic Gprim (`Cube`/`Sphere`/…) that `usd-cli create` authors, every
deactivated prim, and every invisible prim. "Near-empty geometry" is the worst
possible failure here: detection succeeds and reports the whole scope as free.

Instead we drive `usd_core.raster._iter_geometry`, the AOV rasterizer's ingest,
which already handles all four cases and is exercised by every `render` call.
One ingest path, one set of bugs.

Two transforms `_iter_geometry` does not do, both mandatory:

**Units.** `_iter_geometry` yields stage units; the detector is metres throughout
(`--size 0.3,0.3,0.3` means 30 cm regardless of the asset). Scale by
`UsdGeom.GetStageMetersPerUnit`. SimReady assets are commonly centimetres, where
skipping this makes every size threshold off by 100×.

**Up-axis.** The detector's grid is hard Z-up: `nx,ny` tile the XY plane, spans
run along Z, and slope is measured from XY. On a Y-up stage that is not a
labelling difference — the floor would be rasterised as a *wall*, gravity would
point along +Y, and every result would be meaningless. So Y-up stages are rotated
into Z-up on ingest and every result is rotated back before it reaches the user.
`to_stage_*` below are that inverse; results must never be returned raw.
"""

from __future__ import annotations

import numpy as np

from .geometry import SceneGeometry

# Y-up → Z-up is +90° about X: (x, y, z) → (x, −z, y). Y (up) ↦ Z (up), and the
# frame stays right-handed, so winding — and therefore the sign of the face
# normals the slope classifier reads — is preserved.
_YUP_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]],
    dtype=np.float64,
)  # row-vector convention: v_zup = v_yup @ _YUP_TO_ZUP


def stage_up_axis(stage) -> str:
    """`"Y"` or `"Z"` — the stage's declared up axis (USD defaults to Y)."""
    from pxr import UsdGeom

    return "Y" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y else "Z"


class Frame:
    """The stage↔detector coordinate mapping for one stage.

    Carries the metersPerUnit scale and the up-axis rotation together, because a
    result converted with only one of the two is silently wrong rather than
    obviously broken. Every number crossing out of the detector goes through a
    `to_stage_*` method here.
    """

    __slots__ = ("meters_per_unit", "up_axis")

    def __init__(self, meters_per_unit: float, up_axis: str) -> None:
        self.meters_per_unit = float(meters_per_unit) or 1.0
        self.up_axis = up_axis

    @classmethod
    def of(cls, stage) -> "Frame":
        from pxr import UsdGeom

        mpu = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
        return cls(mpu, stage_up_axis(stage))

    @property
    def rotates(self) -> bool:
        return self.up_axis == "Y"

    # ── stage → detector ──────────────────────────────────────────────────────
    def to_detector_points(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) stage-unit points → metres, Z-up."""
        out = np.asarray(pts, dtype=np.float64) * self.meters_per_unit
        return out @ _YUP_TO_ZUP if self.rotates else out

    def to_detector_bbox(self, bmin, bmax) -> tuple[np.ndarray, np.ndarray]:
        """A stage-frame AABB → the detector-frame AABB enclosing it.

        Re-derives min/max after the rotation: +90° about X maps the Y extent onto
        Z and the Z extent onto −Y, so the corners swap places and a naive
        component-wise transform would produce an inverted (min > max) box.
        """
        corners = np.array(
            [[x, y, z] for x in (bmin[0], bmax[0])
             for y in (bmin[1], bmax[1])
             for z in (bmin[2], bmax[2])],
            dtype=np.float64,
        )
        det = self.to_detector_points(corners)
        return det.min(axis=0), det.max(axis=0)

    def to_detector_size(self, size) -> np.ndarray:
        """An object's (sx, sy, sz) full size → detector frame.

        A size is an unsigned extent, not a point: rotate the axes and take
        magnitudes, so the object's *height* stays the component along the
        detector's up axis (Z) rather than becoming negative.
        """
        s = np.asarray(size, dtype=np.float64) * self.meters_per_unit
        return np.abs(s @ _YUP_TO_ZUP) if self.rotates else s

    # ── detector → stage ──────────────────────────────────────────────────────
    def to_stage_point(self, pt) -> list[float]:
        """A detector-frame point (metres, Z-up) → stage units and up-axis."""
        p = np.asarray(pt, dtype=np.float64)
        if self.rotates:
            p = p @ _YUP_TO_ZUP.T  # orthonormal: transpose == inverse
        return (p / self.meters_per_unit).tolist()


    def to_stage_bbox(self, bmin, bmax) -> tuple[list[float], list[float]]:
        lo = np.asarray(self.to_stage_point(bmin), dtype=np.float64)
        hi = np.asarray(self.to_stage_point(bmax), dtype=np.float64)
        # The inverse rotation flips the sign of one axis, so the transformed
        # "min" corner is not componentwise minimal any more.
        return np.minimum(lo, hi).tolist(), np.maximum(lo, hi).tolist()

    def to_stage_xy_polygon(self, outline, height) -> list[list[float]]:
        """A detector XY polygon at height `z` → stage-frame 3D points.

        Prism outlines are 2D in the detector's XY (ground) plane with the height
        band carried separately; on a Y-up stage that plane is XZ, so the polygon
        cannot be returned as bare (x, y) pairs.

        `height` is either one number for the whole loop, or one per vertex. The
        per-vertex form is what lets a polygon lie ON a sloped surface instead of
        hovering over it at some representative height: a region is connected by the
        step between adjacent cells, so a continuous ramp is one polygon whose
        vertices are metres apart in z.
        """
        if isinstance(height, (int, float)):
            zs = [float(height)] * len(outline)
        else:
            zs = [float(h) for h in height]
        return [self.to_stage_point((float(x), float(y), z))
                for (x, y), z in zip(outline, zs)]


def scene_geometry(stage, ref_for_path, *, scope=None, scope_pad=0.0) -> SceneGeometry:
    """Triangulate `stage` into detector-frame `SceneGeometry` (metres, Z-up).

    Args:
        stage: an open `Usd.Stage`.
        ref_for_path: `RefTable.ref_for_path` — accepted for signature parity with
            `_iter_geometry` and to keep a single ingest contract. Region→prim
            attribution goes through `object_paths`, which stays index-aligned with
            `tri_object_id`, so refs are resolved by the caller at report time
            rather than baked in here.
        scope: optional detector-frame `(min, max)` AABB. Prims whose world bounds
            miss it are skipped before their triangles are transformed — the
            scope-cull that keeps a 37M-triangle stage from being fully ingested to
            answer a question about one shelf.
        scope_pad: grow the scope by this much (detector metres) *for the cull only*.
            Callers pass the query's cell size. Two reasons, and both are correctness
            rather than tuning:

            1. **Conservative rasterisation.** A triangle just outside the scope still
               lands in the boundary cells it overlaps, so culling flush with the scope
               silently removes solid that the heightfield would have recorded.
            2. **The boundary-touch case.** A scope derived from a prim's bbox carries
               that bbox's float32 noise, so the floor a query rests on can sit a few
               hundred nanometres below the scope's own minimum and be culled by a
               strict comparison. That happened: a ground plane whose top face was
               7e-10 lost to a scope minimum of 2.29e-7 — a 228 nm miss that emptied
               every slat-gap column on a pallet, and, because a column with no
               geometry yields no free span at all, punched holes that the footprint
               erosion then widened into a fragmented surface.

            Culling is an optimisation; it must never change the answer. When in doubt
            keep the prim — the cost is transforming some triangles, and the cost of
            being wrong is inventing or destroying free space.

    Returns an empty-but-valid `SceneGeometry` when nothing survives, never `None`.
    """
    from pxr import Usd, UsdGeom
    from usd_core.raster import _iter_geometry

    frame = Frame.of(stage)
    include_prim = None
    if scope is None:
        scope_lo = scope_hi = None
    else:
        pad = float(scope_pad)
        lo_arr = np.asarray(scope[0], dtype=np.float64)
        hi_arr = np.asarray(scope[1], dtype=np.float64)
        # A relative floor on top of the caller's pad: float32 geometry around a
        # coordinate of magnitude M carries ~1e-7*M of noise, so a fixed pad alone is
        # not scale-safe on a warehouse laid out hundreds of metres from the origin.
        eps = 1e-6 * float(np.max(np.abs(np.concatenate([lo_arr, hi_arr]))) + 1.0)
        pad = max(pad, eps)
        scope_lo, scope_hi = lo_arr - pad, hi_arr + pad

        # Computing a prim's USD world bound is deliberately done before
        # `_iter_geometry` reads Mesh points or tessellates analytic Gprims.  The
        # detector only needs a conservative whole-prim reject: boundary-crossing
        # prims remain intact, avoiding the free-space holes that clipped triangles
        # would create.
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)

        def include_prim(prim) -> bool:
            bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty():
                return True  # preserve the existing ingest path for unusual prims
            prim_lo, prim_hi = frame.to_detector_bbox(bounds.GetMin(), bounds.GetMax())
            return not ((prim_hi < scope_lo).any() or (prim_lo > scope_hi).any())

    all_v: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    all_oid: list[np.ndarray] = []
    object_paths: list[str] = []
    vbase = 0

    for path, _ref, world, tris in _iter_geometry(
            stage, ref_for_path, include_prim=include_prim):
        pts = frame.to_detector_points(world)
        if scope_lo is not None:
            # Reject on the prim's own AABB — cheap, and conservative: a prim
            # straddling the scope boundary is kept whole, because a triangle
            # clipped mid-face would open a hole the detector reads as free space.
            if (pts.max(axis=0) < scope_lo).any() or (pts.min(axis=0) > scope_hi).any():
                continue
        oid = len(object_paths)
        object_paths.append(path)
        all_v.append(pts.astype(np.float32))
        all_i.append((np.asarray(tris, dtype=np.int64) + vbase).astype(np.int32))
        all_oid.append(np.full(len(tris), oid, dtype=np.int32))
        vbase += pts.shape[0]

    if not all_v:
        return SceneGeometry(
            vertices=np.zeros((0, 3), np.float32),
            indices=np.zeros((0, 3), np.int32),
            tri_object_id=np.zeros((0,), np.int32),
            object_paths=[],
            meters_per_unit=frame.meters_per_unit,
        )
    return SceneGeometry(
        vertices=np.concatenate(all_v, axis=0),
        indices=np.concatenate(all_i, axis=0),
        tri_object_id=np.concatenate(all_oid, axis=0),
        object_paths=object_paths,
        meters_per_unit=frame.meters_per_unit,
    )
