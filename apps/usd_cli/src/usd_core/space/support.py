# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable-support region detection — the "where can an object rest *stably*" query.

This is empty-space detection with two optional **stability gates** baked in, so a
consumer (object placement, scene-graph, world-query) gets stable-support regions
from one call instead of re-deriving support quality itself:

1. **Support distribution** (``support_margin``, metres, absolute) — the support under
   the footprint must sit on both sides of the object centre in more than one direction,
   never a narrow strip (anti-perch / anti-tightrope). Enforced by a morphological
   OPENING of the raw support region by the margin (deletes thin rims / corners / strips
   while keeping broad surfaces at full extent), plus, on the overhang path, a CoG-bracket.
2. **Support area** (``min_support_area`` / ``min_support_area_ratio``) — the total
   support-surface area must be at least an absolute floor OR a fraction of the object's
   footprint area (anti-point-balance). When a margin opening is enabled, the floor is
   rechecked on its surviving support surface, so a deleted thin appendage cannot keep a
   small perched patch alive.

Both gates default OFF, so with no gate the output is exactly ESD's raw free-support
regions. Yaw (Z-rotation) is intentionally NOT handled here — it is a pure footprint
transform (``(max(sx,sy), max(sx,sy))``) the caller applies before calling in.

Stability model: geometric footprint-over-support with a **uniform-density centroid** —
NOT a mass-weighted centre of mass, and no tipping dynamics. That is the correct default;
true CoM needs physics. The gate math is a straight port of the placement layer's
``feasible.support_regions`` (which this replaces), so results are bit-for-bit identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ._warp import warp as _warp
from .extract import (_connected_components, detect_regions, detect_regions_warp,
                      full_support_centers_warp)
from .gapfill import fill_empty_columns
from .heightfield import DEFAULT_MERGE_GAP, MAX_SOLID, build_spans_raster
from .device_mesh import DeviceMesh
from .errors import SpaceInvalidArgument, SpaceOutOfRange
from .smooth import smooth_spans_warp
from .geometry import SceneGeometry

wp = _warp()

_Cell = Tuple[int, int]

# Public library callers bypass Session._space_guard, so derive this from its same
# 4 GiB solid-heightfield allocation ceiling.  `heightfield` holds two float32 and
# five int32 MAX_SOLID-length arrays per cell; the separate free-span buffers and
# host copies remain additional headroom, just as they do for the Session guard.
MAX_SUPPORT_CELLS = (4 * 1024 ** 3) // (MAX_SOLID * 28)

# The summed-area-table erosion grid is a pair of small NumPy arrays, not the
# MAX_SOLID-layer Warp raster guarded above.  Keep its independent work-space
# ceiling so a legal raster query can still evaluate a large footprint.
MAX_SUPPORT_WINDOW_CELLS = 8_000_000

#: Work ceiling for the entire overhang query (``tau < 1``), in footprint-cell tests. The
#: full-support path is a summed-area table and does not care how big the window is;
#: this one has to look at *which* cells under the footprint are supported, so it stays
#: proportional to (candidate centres x footprint cells). The grid guard bounds the
#: first factor and `--cell` the second, but their product is unbounded: 500x500 cells
#: under a footprint 450 cells wide is 5e10 tests, i.e. a daemon that has stopped
#: answering. Refuse with advice instead — a coarser `--cell` is the fix, and it is the
#: same answer at the resolution that query can actually afford.
MAX_OVERHANG_WORK = 200_000_000
# The current partial-support Warp kernel is dense: each raw region launches over the
# whole grid even though only its candidate centres can be accepted. Beyond this total,
# use the existing work-bounded CPU oracle: fragmented shelves remain queryable instead
# of becoming device-dependent rejections.
MAX_PARTIAL_SUPPORT_DENSE_WORK = 200_000_000


@wp.kernel
def _partial_support_kernel(
    region: wp.array(dtype=wp.int32),
    candidates: wp.array(dtype=wp.int32),
    floors: wp.array(dtype=wp.float32),
    ceilings: wp.array(dtype=wp.float32),
    span_min: wp.array(dtype=wp.float32),
    span_max: wp.array(dtype=wp.float32),
    span_count: wp.array(dtype=wp.int32),
    solid_count: wp.array(dtype=wp.int32),
    nx: wp.int32, ny: wp.int32, ms: wp.int32,
    ex: wp.int32, ey: wp.int32, tau: wp.float64, margin: wp.float32,
    height: wp.float64, valid: wp.array(dtype=wp.int32),
    out_floor: wp.array(dtype=wp.float32), out_ceiling: wp.array(dtype=wp.float32),
):
    gid = wp.tid()
    if candidates[gid] == 0:
        valid[gid] = 0
        return
    cx, cy = gid // ny, gid % ny
    supported, minx, maxx, miny, maxy = int(0), int(nx), int(-1), int(ny), int(-1)
    floor, ceiling = wp.float64(-1.0e30), wp.float64(1.0e30)
    dx = int(-ex)
    while dx <= ex:
        dy = int(-ey)
        while dy <= ey:
            x, y = cx + dx, cy + dy
            if x >= 0 and y >= 0 and x < nx and y < ny:
                cell = x * ny + y
                if region[cell] != 0:
                    supported += 1
                    minx, maxx = wp.min(minx, x), wp.max(maxx, x)
                    miny, maxy = wp.min(miny, y), wp.max(maxy, y)
                    floor = wp.max(floor, wp.float64(floors[cell]))
                    ceiling = wp.min(ceiling, wp.float64(ceilings[cell]))
            dy += 1
        dx += 1
    total = (2 * ex + 1) * (2 * ey + 1)
    if (supported == 0
            or wp.float64(supported) / wp.float64(total) + wp.float64(1.0e-9) < tau):
        valid[gid] = 0
        return
    if float(cx - minx) + 1.0e-9 < margin or float(maxx - cx) + 1.0e-9 < margin or float(cy - miny) + 1.0e-9 < margin or float(maxy - cy) + 1.0e-9 < margin:
        valid[gid] = 0
        return
    z1 = floor + height
    dx = int(-ex)
    while dx <= ex:
        dy = int(-ey)
        while dy <= ey:
            x, y = cx + dx, cy + dy
            if x < 0 or y < 0 or x >= nx or y >= ny:
                valid[gid] = 0
                return
            else:
                cell = x * ny + y
                if region[cell] == 0 and solid_count[cell] != 0:
                    clear = int(0)
                    k = int(0)
                    base = cell * ms
                    while k < span_count[cell]:
                        if (wp.float64(span_min[base + k]) <= floor + wp.float64(1.0e-6)
                                and wp.float64(span_max[base + k]) >= z1 - wp.float64(1.0e-6)):
                            clear = 1
                            break
                        k += 1
                    if clear == 0:
                        valid[gid] = 0
                        return
            dy += 1
        dx += 1
    valid[gid] = 1
    out_floor[gid] = wp.float32(floor)
    out_ceiling[gid] = wp.float32(ceiling)


@dataclass
class SupportRegionCells:
    """One stable-support region as grid cells (ESD-native; no world polygon).

    ``cells`` are the valid object-CENTRE cells (for full support the whole footprint
    rests on support; for overhang the CoG is bracketed). ``ceiling_z`` is the lowest
    ceiling, and ``support_z`` the conservative rest height — the HIGHEST floor over the
    region, which is what the clearance test must use.

    ``floors`` is that height per cell, and it is the field that keeps a region honest.
    Neighbouring cells join when their floors are within the climb tolerance, so a
    continuous ramp is deliberately ONE region: the criterion is the step between
    adjacent cells, not the total rise across the region. That means the floors inside
    one region legitimately vary — by 0.5 m over a gentle 2 m ramp — and collapsing
    them to a single number reports a flat surface where there is a slope. The caller
    maps cells -> world polygon and carries these heights onto it.
    """

    cells: List[_Cell]
    support_z: float
    ceiling_z: float
    floors: Dict[_Cell, float] = field(default_factory=dict)


@dataclass
class SupportRegionResult:
    """Result of :func:`detect_support_regions` — regions + grid metadata."""

    regions: List[SupportRegionCells]
    origin: Tuple[float, float]
    cell_size: float
    nx: int
    ny: int
    #: Cells whose true solid-layer count exceeded the rasteriser's fixed buffer
    #: (``heightfield.MAX_SOLID``). Those cells were treated as fully solid, which is
    #: fail-safe — the answer can only lose a resting surface, never invent one — but
    #: the caller has no other way to tell it happened, so it is carried out here.
    solid_overflow_cells: int = 0
    #: Cells with more usable free levels than ``heightfield.MAX_FREE``. The levels
    #: dropped are the HIGHEST ones, so unlike the solid overflow this one removes real
    #: regions from the answer rather than hiding a surface behind extra solid.
    free_overflow_cells: int = 0
    #: Empty columns can inherit several supported levels from surrounding slats.  If
    #: more such levels fit than the fixed free-span buffer, the upper ones are dropped
    #: and the caller must not mistake the resulting answer for a complete rack view.
    gapfill_overflow_cells: int = 0


def _window_counts(mask: np.ndarray, ex: int, ey: int) -> np.ndarray:
    """For every cell, how many of its ``(2ex+1)x(2ey+1)`` neighbours are set.

    A summed-area table, so the cost is ``O(nx*ny)`` **independent of the window
    size**. The naive form — walk the window per cell — is what the erosion below
    used to do, and the window is not small: a ``--cell`` an order of magnitude under
    the object puts ``ex``/``ey`` in the hundreds, and a broad flat floor is the worst
    case because nothing short-circuits. Measured on the old code, a 200x200 grid at
    ``ex=ey=60`` took 6 s and the shapes the grid guard still permits (500x500 at
    ``ex=ey=225``) ran for minutes — a daemon that has stopped answering.
    """
    nx, ny = mask.shape
    padded_cells = (nx + 2 * ex) * (ny + 2 * ey)
    if padded_cells > MAX_SUPPORT_WINDOW_CELLS:
        raise SpaceOutOfRange(
            f"footprint window expands the {nx}x{ny} support grid to "
            f"{nx + 2 * ex}x{ny + 2 * ey} = {padded_cells} cells, over the "
            f"{MAX_SUPPORT_WINDOW_CELLS} cell budget. Use a smaller --size/--margin, a "
            f"larger --cell, or a smaller --scope.")
    padded = np.zeros((nx + 2 * ex, ny + 2 * ey), dtype=np.int32)
    padded[ex:ex + nx, ey:ey + ny] = mask
    sat = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int32)
    sat[1:, 1:] = padded.cumsum(0).cumsum(1)
    w, h = 2 * ex + 1, 2 * ey + 1
    return (sat[w:w + nx, h:h + ny] - sat[0:nx, h:h + ny]
            - sat[w:w + nx, 0:ny] + sat[0:nx, 0:ny])


def _as_mask(cells: Set[_Cell], pad_x: int = 0, pad_y: int = 0):
    """Cell set -> dense bool grid over its bounding box, plus that box's origin."""
    arr = np.fromiter((v for c in cells for v in c), dtype=np.int64,
                      count=2 * len(cells)).reshape(-1, 2)
    ox = int(arr[:, 0].min()) - pad_x
    oy = int(arr[:, 1].min()) - pad_y
    nx = int(arr[:, 0].max()) - ox + 1 + pad_x
    ny = int(arr[:, 1].max()) - oy + 1 + pad_y
    mask = np.zeros((nx, ny), dtype=bool)
    mask[arr[:, 0] - ox, arr[:, 1] - oy] = True
    return mask, ox, oy


def _from_mask(mask: np.ndarray, ox: int, oy: int) -> Set[_Cell]:
    xs, ys = np.nonzero(mask)
    return set(zip((xs + ox).tolist(), (ys + oy).tolist()))


def _erode_centers(cellset: Set[_Cell], ex: int, ey: int) -> Set[_Cell]:
    """Erode-ONLY: keep cells whose full (2ex+1)x(2ey+1) footprint is inside the set.

    Survivors are exactly the valid object CENTRES (footprint fully on support). This is
    the erode half of ``extract._erode_region`` WITHOUT the dilation — that opening
    dilates the border back and over-reports centres, which is why we erode-only here.
    """
    if (ex <= 0 and ey <= 0) or not cellset:
        return set(cellset)
    mask, ox, oy = _as_mask(cellset)
    full = (2 * ex + 1) * (2 * ey + 1)
    return _from_mask(_window_counts(mask, ex, ey) == full, ox, oy)


def _dilate(cells: Set[_Cell], ex: int, ey: int) -> Set[_Cell]:
    """All cells within Chebyshev (ex, ey) of any cell — the reach of a footprint centre."""
    if not cells:
        return set()
    if ex <= 0 and ey <= 0:
        return set(cells)
    # Grow the frame first: the answer reaches ex/ey beyond the input's own bbox.
    mask, ox, oy = _as_mask(cells, pad_x=ex, pad_y=ey)
    mask = np.pad(mask, ((0, ex), (0, ey)))
    return _from_mask(_window_counts(mask, ex, ey) > 0, ox, oy)


#: Slack for the overhang clearance test, mirroring ``extract._HEIGHT_EPS``. Spans are
#: float32 and the rest height is a max over them, so an object that fits exactly lands
#: a hair outside the span it fits in.
_CLEAR_EPS = 1.0e-6

#: A column the rasteriser found NO geometry in — free at every height. Modelled as one
#: unbounded free span so the cover test below needs no special case.
_OPEN_COLUMN = ((-math.inf, math.inf),)


def _column_probe(spans: dict):
    """Build ``clears(ix, iy, z0, z1)`` — is column ``(ix, iy)`` free over ``[z0, z1)``?

    This is the question the overhang path has to ask about the part of the footprint
    that hangs OFF the support, and it cannot be answered from the region: those cells
    are outside it by definition. It is answered from the span grid instead.

    Zero free spans is ambiguous on its own — the column is either open air (no geometry
    at all, the case an overhang is *for*) or solid (including a cell the rasteriser gave
    up on after ``MAX_SOLID`` overflow and conservatively filled in). ``solid_count``
    separates them, which is the same reason ``gapfill`` needs it.

    Sound but deliberately not tight. A "no" can mean the free air there was filtered out
    rather than absent: in ``terrain`` slope mode the span above a ramp is dropped (it is
    real air, just not a rest surface), and a sealed cavity's interior is dropped (there
    the rejection is right — you cannot reach into a closed box). Both err toward refusing
    a placement, never toward inventing one, which is the direction the rest of this
    module already leans. The public CLI uses terrain slope semantics, so overhanging
    above a ramp requires full support (``tau=1``).

    Off the grid is a "no". The scope clips the geometry too, so a column outside it has
    no evidence either way, and refusing keeps ``tau < 1`` no more permissive at the scope
    border than ``tau = 1``, where a footprint leaving the grid can never survive the
    erosion. Widen ``--scope`` if the answer should extend further.
    """
    cnt = spans["span_count"]
    lo = spans["span_min"]
    hi = spans["span_max"]
    solid = spans["solid_count"]
    nx, ny = int(spans["nx"]), int(spans["ny"])
    cache: Dict[_Cell, Tuple[Tuple[float, float], ...]] = {}

    def clears(ix: int, iy: int, z0: float, z1: float) -> bool:
        if not (0 <= ix < nx and 0 <= iy < ny):
            return False
        levels = cache.get((ix, iy))
        if levels is None:
            if int(solid[ix, iy]) == 0:
                levels = _OPEN_COLUMN
            else:
                levels = tuple((float(lo[ix, iy, k]), float(hi[ix, iy, k]))
                               for k in range(int(cnt[ix, iy])))
            cache[(ix, iy)] = levels
        for a, b in levels:
            if a <= z0 + _CLEAR_EPS and b >= z1 - _CLEAR_EPS:
                return True
        return False

    return clears


def _partial_support_centers(region_cells: Set[_Cell], bt: Dict[_Cell, Tuple[float, float]],
                             ex: int, ey: int, tau: float, mc: int, sz: float, clears,
                             candidates: Set[_Cell] | None = None
                             ) -> Dict[_Cell, Tuple[float, float]]:
    """Valid overhang centres: footprint >= ``tau`` supported, CoG bracketed by the
    supported cells with ``mc`` cells (an ABSOLUTE margin) on every side, AND the
    overhanging remainder of the footprint clear of geometry over the object's height.

    Returns ``{center_cell: (support_z, ceiling_z)}``. A centre may lie OUTSIDE the region
    (overhang) as long as enough of the footprint stays on it and the margin holds — a
    cantilever with support on one side only fails the bracket, so nothing point-balances.

    The bracket is a TIPPING test, not a clearance one: it asks whether the object stays
    put, never whether there is room for the part sticking out. Those are separate
    questions and only the first used to be asked, so a deck with a pillar beside it
    rising above deck level reported centres whose body ran straight through the pillar —
    the supported half cleared, and the half over the pillar was not looked at because
    its cells are not in ``region_cells``. Hence ``clears``, applied to exactly those
    cells at the rest height this centre actually sits at.
    """
    n_fp = (2 * ex + 1) * (2 * ey + 1)
    mx = my = float(mc)
    out: Dict[_Cell, Tuple[float, float]] = {}
    for c in candidates if candidates is not None else _dilate(region_cells, ex, ey):
        cx, cy = c
        sup = [(cx + dx, cy + dy) for dx in range(-ex, ex + 1)
               for dy in range(-ey, ey + 1) if (cx + dx, cy + dy) in region_cells]
        if (len(sup) / n_fp) + 1.0e-9 < tau:
            continue
        sxs = [s[0] for s in sup]
        sys_ = [s[1] for s in sup]
        if not ((cx - min(sxs)) >= mx - 1.0e-9 and (max(sxs) - cx) >= mx - 1.0e-9
                and (cy - min(sys_)) >= my - 1.0e-9 and (max(sys_) - cy) >= my - 1.0e-9):
            continue
        # Clearance LAST: it is the only test that touches the span grid, and by here
        # most candidates are already gone, so it runs over survivors rather than over
        # every dilated cell. The supported cells need no test — the component clearance
        # check downstream covers them with the region's own floor/ceiling extremes.
        z0 = max(bt[s][0] for s in sup)
        z1 = z0 + sz
        blocked = False
        for dx in range(-ex, ex + 1):
            for dy in range(-ey, ey + 1):
                n = (cx + dx, cy + dy)
                if n not in region_cells and not clears(n[0], n[1], z0, z1):
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue
        out[c] = (z0, min(bt[s][1] for s in sup))
    return out


def _partial_support_centers_warp(
    region_cells: Set[_Cell],
    bt: Dict[_Cell, Tuple[float, float]],
    spans: dict,
    ex: int,
    ey: int,
    tau: float,
    mc: int,
    sz: float,
    candidates: Set[_Cell],
    *,
    device: str,
    device_spans: tuple | None = None,
) -> Dict[_Cell, Tuple[float, float]]:
    """CUDA implementation of the ``tau < 1`` support/clearance gate.

    The public span representation is still host-owned while Phase 3 moves the
    gapfill/smooth chain onto the device.  This adapter intentionally uploads one
    dense query snapshot and downloads only the accepted centres; it replaces the
    expensive Python ``candidate x footprint`` loops without changing their
    conservative free-span predicate.
    """
    nx, ny, ms = int(spans["nx"]), int(spans["ny"]), int(spans["max_spans"])
    n = nx * ny
    region = np.zeros(n, dtype=np.int32)
    floor = np.zeros(n, dtype=np.float32)
    ceiling = np.zeros(n, dtype=np.float32)
    for (ix, iy), (z0, z1) in bt.items():
        cell = ix * ny + iy
        if (ix, iy) in region_cells:
            region[cell] = 1
            floor[cell] = z0
            ceiling[cell] = z1
    candidate = np.zeros(n, dtype=np.int32)
    for ix, iy in candidates:
        if 0 <= ix < nx and 0 <= iy < ny:
            candidate[ix * ny + iy] = 1

    d_region = wp.array(region, dtype=wp.int32, device=device)
    d_candidate = wp.array(candidate, dtype=wp.int32, device=device)
    d_floor = wp.array(floor, dtype=wp.float32, device=device)
    d_ceiling = wp.array(ceiling, dtype=wp.float32, device=device)
    if device_spans is None:
        device_spans = _partial_support_device_spans(spans, device=device)
    d_span_min, d_span_max, d_span_count, d_solid_count = device_spans
    valid = wp.zeros(n, dtype=wp.int32, device=device)
    out_floor = wp.zeros(n, dtype=wp.float32, device=device)
    out_ceiling = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        _partial_support_kernel,
        dim=n,
        inputs=[d_region, d_candidate, d_floor, d_ceiling, d_span_min, d_span_max,
                d_span_count, d_solid_count, wp.int32(nx), wp.int32(ny), wp.int32(ms),
                wp.int32(ex), wp.int32(ey), wp.float64(tau), wp.float32(mc), wp.float64(sz)],
        outputs=[valid, out_floor, out_ceiling],
        device=device,
    )
    valid_host = valid.numpy()
    floor_host = out_floor.numpy()
    ceiling_host = out_ceiling.numpy()
    return {
        (idx // ny, idx % ny): (float(floor_host[idx]), float(ceiling_host[idx]))
        for idx in np.flatnonzero(valid_host)
    }


def _partial_support_device_spans(spans: dict, *, device: str) -> tuple:
    """Upload the query-invariant span snapshot once for partial-support regions."""
    return (
        wp.array(
            np.ascontiguousarray(spans["span_min"].reshape(-1), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        ),
        wp.array(
            np.ascontiguousarray(spans["span_max"].reshape(-1), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        ),
        wp.array(
            np.ascontiguousarray(spans["span_count"].reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
        wp.array(
            np.ascontiguousarray(spans["solid_count"].reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def _cuda_survivors_by_label(
    cuda_labels: np.ndarray, cuda_centres: np.ndarray
) -> dict[int, Set[_Cell]]:
    """Group CUDA-proven centre cells by their exact connected-component label.

    Kept pure NumPy so CPU CI covers the host-side handoff from Warp labels to the
    public support-region serializer without requiring a physical CUDA device.
    """
    active = cuda_centres & (cuda_labels >= 0)
    ix, iy, slots = np.nonzero(active)
    if not ix.size:
        return {}
    labels = cuda_labels[ix, iy, slots]
    order = np.argsort(labels, kind="stable")
    ix, iy, labels = ix[order], iy[order], labels[order]
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    return {
        int(group_labels[0]): set(zip(group_ix.tolist(), group_iy.tolist()))
        for group_ix, group_iy, group_labels in zip(
            np.split(ix, boundaries), np.split(iy, boundaries), np.split(labels, boundaries)
        )
    }


def detect_support_regions(
    scene: SceneGeometry,
    scope_min: Sequence[float],
    scope_max: Sequence[float],
    cell_size: float,
    object_size: Sequence[float],
    *,
    tau: float = 1.0,
    support_margin: float = 0.0,
    min_support_area: float = 0.0,
    min_support_area_ratio: float = 0.0,
    smooth: bool = True,
    merge_gap: float | None = None,
    slope_mode: str = "terrain",
    slope_threshold_deg: float = 20.0,
    vertical_cutoff_deg: float = 80.0,
    device: str = "cpu",
    max_cells: int = MAX_SUPPORT_CELLS,
) -> SupportRegionResult:
    """Detect stable-support feasible regions for one object footprint.

    Parameters
    ----------
    scene : ESD ``SceneGeometry`` (triangle SoA).
    scope_min, scope_max : world AABB of the region of interest.
    cell_size : grid cell size in metres (the caller resolves its own policy).
    object_size : ``(sx, sy, sz)`` full object size in metres; ``sz`` sets the required
        vertical clearance. Required — the gates need a
        real footprint. If the object may be yaw-rotated, pass its bounding square.
    tau : required support fraction in (0, 1]. ``1`` = full support; ``<1`` allows overhang.
    merge_gap : coincidence tolerance for the solid-layer merge, in metres. ``None``
        uses :data:`~usd_core.space.heightfield.DEFAULT_MERGE_GAP`. Lower it below the
        object height when placing something thinner than the default, which otherwise
        fuses the gap it would rest in; see that constant for why it is invisible
        otherwise.
    support_margin : feature 1 (metres, absolute). ``0`` = off.
    min_support_area, min_support_area_ratio : feature 2 (m^2 / fraction of footprint).
        ``0`` = off; the effective floor is ``max(min_support_area, ratio * sx*sy)``;
        with a positive margin, it is applied again after support-surface opening.
    slope_mode : how inclined surfaces (ramps) are handled — ``"terrain"`` (default,
        keep true height but exclude the incline as a rest surface), ``"ignore"``
        (drop ramps entirely), or ``"flatten"`` (solid up to the ramp peak). A face
        is a ramp when it tilts in ``(slope_threshold_deg, vertical_cutoff_deg)``.

    Returns a :class:`SupportRegionResult`. With both gates off and ``tau=1`` the regions
    are exactly the erode-only valid-centre sets of ESD's raw free-support regions.
    """
    sx, sy, sz = float(object_size[0]), float(object_size[1]), float(object_size[2])
    if not (sx > 0.0 and sy > 0.0 and sz > 0.0 and
            math.isfinite(sx) and math.isfinite(sy) and math.isfinite(sz)):
        raise SpaceInvalidArgument(f"object_size must be positive finite (sx,sy,sz), got {object_size!r}")
    if not (0.0 < float(tau) <= 1.0):
        raise SpaceInvalidArgument(f"tau must be in (0, 1], got {tau!r}")
    if float(support_margin) < 0.0:
        raise SpaceInvalidArgument(f"support_margin must be >= 0, got {support_margin!r}")
    cs = float(cell_size)
    if not (cs > 0.0 and math.isfinite(cs)):
        raise SpaceInvalidArgument(f"cell_size must be positive finite, got {cell_size!r}")
    mg = DEFAULT_MERGE_GAP if merge_gap is None else float(merge_gap)
    if not (mg >= 0.0 and math.isfinite(mg)):
        raise SpaceInvalidArgument(f"merge_gap must be >= 0 and finite, got {merge_gap!r}")
    if slope_mode not in ("ignore", "flatten", "terrain"):
        raise SpaceInvalidArgument(
            f"slope_mode must be 'ignore'|'flatten'|'terrain', got {slope_mode!r}")

    smin = np.asarray(scope_min, dtype=np.float64)
    smax = np.asarray(scope_max, dtype=np.float64)
    # DoS backstop before the raster kernel allocates nx*ny*MAX_SOLID.
    nx_est = int(math.ceil(max(0.0, float(smax[0] - smin[0])) / cs))
    ny_est = int(math.ceil(max(0.0, float(smax[1] - smin[1])) / cs))
    if nx_est * ny_est > int(max_cells):
        raise SpaceInvalidArgument(
            f"grid {nx_est}x{ny_est} = {nx_est * ny_est} cells exceeds max_cells={max_cells} "
            f"(cell_size={cs} too small for scope)")

    # "Is this the same surface?" is a question about the object being placed, not
    # about a fixed number of centimetres. Upstream's 0.1 / 0.2 m are a rounding error
    # under a warehouse pallet and larger than an entire millimetre-scale board — there
    # a 50 mm step between two platforms merges into one region, which then reports the
    # higher floor as its rest height while half of it is a drop. So anchor both to the
    # object's own height, capped at the upstream constants so nothing ever merges more
    # freely than before; sz = 0.3 m reproduces 0.1 / 0.2 exactly.
    bottom_tol = min(0.1, sz / 3.0)
    top_tol = min(0.2, 2.0 * sz / 3.0)

    # ESD's own pipeline: raster heightfield -> optional gap-bridge smooth -> raw regions.
    rc = DeviceMesh(scene, device=device)
    # slope_mode: inclined surfaces (ramps) would otherwise be rasterised as a flat
    # slab up to their peak, reporting a stable platform where there is only a slope.
    # "terrain" (default) keeps the ramp geometry but excludes it as a rest surface.
    spans = build_spans_raster(
        rc, smin, smax, cs, sz, merge_gap=mg, slope_mode=slope_mode,
        slope_threshold_deg=slope_threshold_deg, vertical_cutoff_deg=vertical_cutoff_deg)
    if smooth:
        # Gap-bridging (rest-across-slats) in two steps, both driven by the SIDES.
        #
        # `fill_empty_columns` first (usd-cli addition): a gap column with no geometry
        # under it has no span at all, and the upstream pass below can only lift a span
        # that exists — so without this the gap stays a hole and the footprint erosion
        # later widens it into a break in the surface.
        spans = fill_empty_columns(
            spans, x_length_threshold=sx, y_length_threshold=sy,
            section_height_threshold=sz, bottom_tolerance=bottom_tol)
        # Then upstream's pass, which lifts a gap column's floor to the level its
        # neighbours agree on, requiring BOTH opposing sides to match so a real deck
        # edge is never extended over a void. Conservative (floors only rise, ceilings
        # only fall) so it can't invent support.
        spans = smooth_spans_warp(
            spans, x_length_threshold=sx, y_length_threshold=sy,
            section_height_threshold=sz, bottom_tolerance=bottom_tol, device=device)
    # erosion=None -> RAW regions (this function applies erode-only + gates itself).
    cuda_labels = None
    if str(device).startswith("cuda"):
        raw_regions, cuda_labels = detect_regions_warp(
            spans, sz, bottom_tolerance=bottom_tol, top_tolerance=top_tol,
            device=device, return_labels=True, return_slots=True)
    else:
        raw_regions = detect_regions(spans, sz, bottom_tolerance=bottom_tol,
                                     top_tolerance=top_tol, erosion=None)
    ox, oy = spans["origin"]
    cs = float(spans["cell_size"])
    nx, ny = int(spans["nx"]), int(spans["ny"])

    # Half-footprint in cells (matches ESD's ex,ey = ceil((len/2)/cell)).
    ex = int(math.ceil((sx / 2.0) / cs))
    ey = int(math.ceil((sy / 2.0) / cs))
    # ABSOLUTE margin in cells (independent of object size) — a small fixed buffer removes
    # thin rings / corners / strips (perch) WITHOUT rejecting a large object that
    # legitimately fills a large-but-tight surface.
    mc = int(round(float(support_margin) / cs)) if support_margin > 0.0 else 0
    # Anti-point-balance area floor, measured on the SUPPORT SURFACE vs the footprint.
    area_floor = max(float(min_support_area), float(min_support_area_ratio) * (sx * sy))
    cell_area = cs * cs
    full_support = float(tau) >= 1.0 - 1.0e-9
    cuda_centres = None
    cuda_survivors_by_label: dict[int, Set[_Cell]] | None = None
    if full_support and mc == 0 and cuda_labels is not None:
        cuda_centres = full_support_centers_warp(
            cuda_labels, spans["span_count"], ex, ey, device=device)
        if cuda_centres is not None:
            cuda_survivors_by_label = _cuda_survivors_by_label(cuda_labels, cuda_centres)
    # One probe for the whole query: its per-column cache is worth keeping across
    # regions, since neighbouring regions overhang onto the same columns.
    clears = None if full_support else _column_probe(spans)

    regions: List[SupportRegionCells] = []
    overhang_work = 0
    cuda_dense_work = 0
    cuda_partial_enabled = str(device).startswith("cuda")
    cuda_span_snapshot: tuple | None = None
    for region in raw_regions:
        bt: Dict[_Cell, Tuple[float, float]] = {
            (int(m[0]), int(m[1])): (float(m[2]), float(m[3])) for m in region}
        if len(bt) * cell_area < area_floor - 1.0e-9:
            continue  # support surface too small vs footprint -> point-balance / corner perch
        # Distribution guard: morphologically OPEN the support region by the margin
        # (erode mc -> dilate mc). Deletes thin rims / corners / strips narrower than
        # 2*mc cells (the perch sources) while keeping BROAD surfaces at full extent.
        region_set: Set[_Cell] = set(bt)
        if mc > 0:
            region_set = _dilate(_erode_centers(region_set, mc, mc), mc, mc) & set(bt)
            if not region_set:
                continue
            if area_floor > 0.0:
                # ``area_floor`` is a support-surface requirement, not a count of
                # output object-centre cells. Opening can split a raw region across
                # a deleted thin bridge, so qualify each physical patch independently.
                opened = _connected_components({c: bt[c] for c in region_set})
                region_set = {
                    (int(ix), int(iy))
                    for component in opened
                    if len(component) * cell_area >= area_floor - 1.0e-9
                    for ix, iy, _floor, _ceiling in component
                }
                if not region_set:
                    continue
        if full_support:
            if cuda_centres is None:
                survivors = _erode_centers(region_set, ex, ey)
            else:
                # ``return_slots`` carries the exact source slot, avoiding any
                # tolerance-based floor/ceiling lookup between close rack levels.
                first_ix, first_iy, _floor, _ceiling, slot = region[0]
                label = int(cuda_labels[int(first_ix), int(first_iy), int(slot)])
                survivors = (cuda_survivors_by_label or {}).get(label, set()) & region_set
            zof = {c: (bt[c][0], bt[c][1]) for c in survivors}
        else:
            # Overhang path: footprint >= tau supported + CoG bracketed by the margin.
            n_fp = (2 * ex + 1) * (2 * ey + 1)
            candidates = _dilate(region_set, ex, ey)
            work = len(candidates) * n_fp
            overhang_work += work
            if overhang_work > MAX_OVERHANG_WORK:
                raise SpaceOutOfRange(
                    f"--tau {tau} over {len(raw_regions)} raw regions with a "
                    f"{2 * ex + 1}x{2 * ey + 1}-cell footprint needs ~{overhang_work:.2g} "
                    f"cell tests in total, over the {MAX_OVERHANG_WORK:.2g} budget. Use a coarser "
                    f"--cell (the overhang test cannot resolve below the footprint "
                    f"anyway), a smaller --scope, or --tau 1.")
            if cuda_partial_enabled:
                cuda_dense_work += nx * ny
                if cuda_dense_work > MAX_PARTIAL_SUPPORT_DENSE_WORK:
                    cuda_partial_enabled = False
            if cuda_partial_enabled:
                if cuda_span_snapshot is None:
                    cuda_span_snapshot = _partial_support_device_spans(spans, device=device)
                zof = _partial_support_centers_warp(
                    region_set, bt, spans, ex, ey, float(tau), mc, sz, candidates,
                    device=device, device_spans=cuda_span_snapshot)
            else:
                zof = _partial_support_centers(region_set, bt, ex, ey, float(tau), mc,
                                               sz, clears, candidates)
            survivors = set(zof)
        if not survivors:
            continue
        for comp in _connected_components({c: zof[c] for c in survivors}):
            cells = [(int(m[0]), int(m[1])) for m in comp]
            support_z = max(float(m[2]) for m in comp)  # conservative: highest floor
            ceiling_z = min(float(m[3]) for m in comp)  # conservative: lowest ceiling
            if ceiling_z - support_z < sz - 1.0e-6:     # object no longer clears here
                continue
            regions.append(SupportRegionCells(
                cells=sorted(cells), support_z=support_z, ceiling_z=ceiling_z,
                floors={(int(m[0]), int(m[1])): float(m[2]) for m in comp}))

    return SupportRegionResult(
        regions=regions, origin=(float(ox), float(oy)), cell_size=cs, nx=nx, ny=ny,
        solid_overflow_cells=int(spans.get("solid_overflow_cells", 0)),
        free_overflow_cells=int(spans.get("free_overflow_cells", 0)),
        gapfill_overflow_cells=int(spans.get("gapfill_overflow_cells", 0)))


__all__ = [
    "detect_support_regions",
    "SupportRegionResult",
    "SupportRegionCells",
    "MAX_SUPPORT_CELLS",
    "MAX_SUPPORT_WINDOW_CELLS",
    "DEFAULT_MERGE_GAP",
]
