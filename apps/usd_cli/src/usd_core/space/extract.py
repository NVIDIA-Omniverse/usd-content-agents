# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free-box extraction from the height-span grid.

Given the per-column height spans (from heightfield.build_spans_raster), group cells into
axis-aligned free boxes suitable for placing a target object. Faithful to the
original in spirit (cluster by floor height into shelf "levels", grow connected
regions, emit boxes that fit the footprint/clearance thresholds) with the
**conservative footprint-min height clamp** so a box never intersects geometry
under a tilt (see docs/specs/rotation-handling.md): within a region the box
bottom is the highest floor and the box top the lowest ceiling.

This is a CPU (numpy) pass over the SoA the Warp kernel produced; it is the next
target to move into kernels for Phase 3, but is device-independent and correct
as the reference oracle now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ._warp import warp as _warp

wp = _warp()

# Slack for the "does the free height clear the object?" test. A span whose height
# is exactly the object's lands ~1e-16..1e-8 below it once float rounding is in play
# — spans are stored float32, and a floor at a non-trivial height (a deck at z=0.2
# under a ceiling at z=0.5) does not subtract to exactly 0.3. A strict ``>=`` then
# silently drops a bay that genuinely fits. Compare against ``threshold - eps`` so
# a height that rounds to exactly the object height is kept. 1 micron is far below
# any real placement tolerance, so this never admits a genuinely too-short box.
_HEIGHT_EPS = 1.0e-6

# Synchronous propagation advances a label one 4-neighbour edge per launch.  Beyond
# this, a CUDA query spends more time launching kernels than the CPU union-find takes;
# fall back to the reference rather than making a fine-grid request look accelerated.
MAX_WARP_LABEL_PASSES = 512
# A fixed-point pass touches every dense span slot, and every active slot compares the
# populated slots in four neighbours. Do not first launch hundreds of passes on a grid
# that cannot finish within the bounded comparison budget before falling back.
MAX_WARP_LABEL_WORK = 200_000_000
# The first Phase-1 kernel is a direct exact window check.  It is excellent for the
# ordinary 3x3/5x5 placement footprint, but intentionally falls back before a huge
# object turns every active span into an unbounded GPU nested loop.
MAX_WARP_FULL_SUPPORT_WINDOW_CELLS = 4_096
# The same-label footprint kernel is a nested active-span × window × slot scan.
# Its per-window cap alone does not bound a fragmented fine-grid query.
MAX_WARP_FULL_SUPPORT_WORK = 200_000_000


@wp.kernel
def _init_region_labels(
    smin: wp.array(dtype=wp.float32),
    smax: wp.array(dtype=wp.float32),
    cnt: wp.array(dtype=wp.int32),
    threshold: wp.float64,
    ms: wp.int32,
    active_ids: wp.array(dtype=wp.int32),
    labels: wp.array(dtype=wp.int32),
):
    """One active label per compacted free span; inactive slots stay ``-1``."""
    gid = active_ids[wp.tid()]
    cell = gid // ms
    slot = gid % ms
    if slot < cnt[cell] and (wp.float64(smax[gid]) - wp.float64(smin[gid])) >= threshold:
        labels[gid] = gid
    else:
        labels[gid] = -1


@wp.kernel
def _propagate_region_labels(
    smin: wp.array(dtype=wp.float32),
    smax: wp.array(dtype=wp.float32),
    cnt: wp.array(dtype=wp.int32),
    labels: wp.array(dtype=wp.int32),
    nx: wp.int32,
    ny: wp.int32,
    ms: wp.int32,
    bottom_tol: wp.float64,
    top_tol: wp.float64,
    active_ids: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    """One synchronous min-label relaxation over 4-neighbour compatible spans."""
    gid = active_ids[wp.tid()]
    cell = gid // ms
    label = labels[gid]
    if label < 0:
        out[gid] = -1
        return
    ix = cell // ny
    iy = cell % ny
    best = label
    b, t = wp.float64(smin[gid]), wp.float64(smax[gid])
    for direction in range(4):
        ax, ay = ix, iy
        if direction == 0:
            ax += 1
        elif direction == 1:
            ax -= 1
        elif direction == 2:
            ay += 1
        else:
            ay -= 1
        if ax < 0 or ay < 0 or ax >= nx or ay >= ny:
            continue
        other_cell = ax * ny + ay
        other_base = other_cell * ms
        other_slot = int(0)
        while other_slot < cnt[other_cell]:
            other_gid = other_base + other_slot
            other_label = labels[other_gid]
            if (other_label >= 0
                    and wp.abs(b - wp.float64(smin[other_gid])) <= bottom_tol
                    and wp.abs(t - wp.float64(smax[other_gid])) <= top_tol):
                best = wp.min(best, other_label)
            other_slot += 1
    out[gid] = best
    if best != label:
        wp.atomic_add(changed, 0, 1)


@wp.kernel
def _same_label_window(
    labels: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    nx: wp.int32,
    ny: wp.int32,
    ms: wp.int32,
    ex: wp.int32,
    ey: wp.int32,
    keep: wp.array(dtype=wp.int32),
):
    """Keep a span only when its full rectangular footprint shares its label."""
    gid = wp.tid()
    cell = gid // ms
    label = labels[gid]
    if label < 0:
        keep[gid] = 0
        return
    ix = cell // ny
    iy = cell % ny
    dx = -ex
    while dx <= ex:
        dy = -ey
        while dy <= ey:
            ax, ay = ix + dx, iy + dy
            if ax < 0 or ay < 0 or ax >= nx or ay >= ny:
                keep[gid] = 0
                return
            else:
                other_base = (ax * ny + ay) * ms
                other_slot = int(0)
                found = int(0)
                while other_slot < counts[ax * ny + ay]:
                    if labels[other_base + other_slot] == label:
                        found = 1
                        break
                    other_slot += 1
                if found == 0:
                    keep[gid] = 0
                    return
            dy += 1
        dx += 1
    keep[gid] = 1


def full_support_centers_warp(labels: np.ndarray, counts: np.ndarray, ex: int, ey: int, *, device: str = "cuda:0") -> np.ndarray | None:
    """Boolean `(cell, span)` centre mask for a full footprint on one region label.

    This is deliberately label-based rather than a plain 2-D support mask: a shelf
    above a lower bay, a slope band, or a neighbouring free layer cannot certify a
    footprint for the current surface.
    """
    window_cells = (2 * ex + 1) * (2 * ey + 1)
    if window_cells > MAX_WARP_FULL_SUPPORT_WINDOW_CELLS:
        return None
    active = int(np.count_nonzero(labels >= 0))
    max_slots = int(np.max(counts, initial=0))
    if active * window_cells * max_slots > MAX_WARP_FULL_SUPPORT_WORK:
        return None
    nx, ny, ms = labels.shape
    flat = np.ascontiguousarray(labels.reshape(-1), dtype=np.int32)
    count_flat = np.ascontiguousarray(counts.reshape(-1), dtype=np.int32)
    source = wp.array(flat, dtype=wp.int32, device=device)
    count_source = wp.array(count_flat, dtype=wp.int32, device=device)
    keep = wp.zeros(flat.size, dtype=wp.int32, device=device)
    wp.launch(_same_label_window, dim=flat.size,
              inputs=[source, count_source, wp.int32(nx), wp.int32(ny), wp.int32(ms),
                      wp.int32(ex), wp.int32(ey)], outputs=[keep], device=device)
    return keep.numpy().reshape(nx, ny, ms).astype(bool)


def detect_regions_warp(
    spans: dict,
    cell_height_threshold: float,
    bottom_tolerance: float = 0.1,
    top_tolerance: float = 0.2,
    *,
    device: str = "cuda:0",
    return_labels: bool = False,
    return_slots: bool = False,
) -> (list[list[tuple[int, int, float, float]]]
      | tuple[list[list[tuple[int, int, float, float]]], np.ndarray | None]):
    """CUDA connected components with the exact CPU compatibility predicate.

    Labels propagate the minimum active node id. A synchronous pass advances a label
    by one in-region edge, so non-convex components must run to a fixed point rather
    than using the grid's width/height as a convergence bound. If the bounded pass
    budget is exhausted, this returns the exact CPU reference result instead.

    The compact result is downloaded once and grouped into the existing public region
    representation; ``return_slots`` is an internal support-query option that carries
    each member's exact source span slot alongside its public floor/ceiling values.
    """
    nx, ny, ms = int(spans["nx"]), int(spans["ny"]), int(spans["max_spans"])
    n = nx * ny * ms
    slots = np.arange(ms, dtype=np.int32)[None, None, :]
    # Kernels and CPU extraction subtract stored float32 bounds as float64. Keep the
    # compact launch mask identical at the threshold boundary; otherwise a float32
    # subtraction can discard a valid span before Warp sees it.
    span_height = (
        spans["span_max"].astype(np.float64) - spans["span_min"].astype(np.float64)
    )
    active_mask = (
        (slots < spans["span_count"][:, :, None])
        & (span_height >= cell_height_threshold - _HEIGHT_EPS)
    )
    active_ids = np.flatnonzero(active_mask.reshape(-1)).astype(np.int32)
    active = int(active_ids.size)
    if active == 0:
        return ([], None) if return_labels else []
    neighbour_slots = int(np.max(spans["span_count"], initial=0))
    # Labels remain dense for O(1) neighbour lookup, but launch only populated,
    # height-eligible spans.  The budget reflects that compact work plus each active
    # node's four populated-neighbour scans, not the heightfield's fixed 16-slot
    # storage capacity.
    work = active * (1 + MAX_WARP_LABEL_PASSES * (1 + 4 * neighbour_slots))
    if work > MAX_WARP_LABEL_WORK:
        fallback = detect_regions(
            spans, cell_height_threshold, bottom_tolerance, top_tolerance, device=None,
            return_slots=return_slots)
        return (fallback, None) if return_labels else fallback
    smin = np.ascontiguousarray(spans["span_min"].reshape(-1), dtype=np.float32)
    smax = np.ascontiguousarray(spans["span_max"].reshape(-1), dtype=np.float32)
    cnt = np.ascontiguousarray(spans["span_count"].reshape(-1), dtype=np.int32)
    dmin = wp.array(smin, dtype=wp.float32, device=device)
    dmax = wp.array(smax, dtype=wp.float32, device=device)
    dcnt = wp.array(cnt, dtype=wp.int32, device=device)
    dactive = wp.array(active_ids, dtype=wp.int32, device=device)
    labels = wp.full(n, -1, dtype=wp.int32, device=device)
    scratch = wp.full(n, -1, dtype=wp.int32, device=device)
    wp.launch(_init_region_labels, dim=active,
              inputs=[dmin, dmax, dcnt, wp.float64(cell_height_threshold - _HEIGHT_EPS),
                      wp.int32(ms), dactive], outputs=[labels], device=device)
    converged = False
    for _ in range(MAX_WARP_LABEL_PASSES):
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_propagate_region_labels, dim=active,
                  inputs=[dmin, dmax, dcnt, labels, wp.int32(nx), wp.int32(ny),
                          wp.int32(ms), wp.float64(bottom_tolerance), wp.float64(top_tolerance),
                          dactive],
                  outputs=[scratch, changed], device=device)
        labels, scratch = scratch, labels
        if int(changed.numpy()[0]) == 0:
            converged = True
            break
    if not converged:
        fallback = detect_regions(
            spans, cell_height_threshold, bottom_tolerance, top_tolerance, device=None,
            return_slots=return_slots)
        return (fallback, None) if return_labels else fallback
    labels_host = labels.numpy().reshape(nx, ny, ms)
    slots = np.arange(ms, dtype=np.int32)[None, None, :]
    active = (labels_host >= 0) & (slots < spans["span_count"][:, :, None])
    ix, iy, k = np.nonzero(active)
    if ix.size == 0:
        result = []
    else:
        group_labels = labels_host[ix, iy, k]
        order = np.argsort(group_labels, kind="stable")
        ix, iy, k, group_labels = (
            ix[order], iy[order], k[order], group_labels[order]
        )
        boundaries = np.flatnonzero(np.diff(group_labels)) + 1
        groups = zip(
            np.split(ix, boundaries),
            np.split(iy, boundaries),
            np.split(k, boundaries),
        )
        if return_slots:
            result = [
                [
                    (int(x), int(y), float(spans["span_min"][x, y, slot]),
                     float(spans["span_max"][x, y, slot]), int(slot))
                    for x, y, slot in zip(xs, ys, ks)
                ]
                for xs, ys, ks in groups
            ]
        else:
            result = [
                [
                    (int(x), int(y), float(spans["span_min"][x, y, slot]),
                     float(spans["span_max"][x, y, slot]))
                    for x, y, slot in zip(xs, ys, ks)
                ]
                for xs, ys, ks in groups
            ]
    return (result, labels_host) if return_labels else result


def _connected_components(cells_bt: dict):
    """4-connected components of a {(ix,iy): (b,t)} dict. Returns list of
    lists of (ix, iy, b, t). Pure grid adjacency (band matching already done
    by the caller when it built cells_bt)."""
    seen = set()
    out = []
    for start in cells_bt:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            cx, cy = stack.pop()
            b, t = cells_bt[(cx, cy)]
            comp.append((cx, cy, b, t))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cx + dx, cy + dy)
                if n in cells_bt and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(comp)
    return out


def _erode_region(members, ex: int, ey: int):
    """Morphologically OPEN one region by the object's half-footprint (ex, ey).

    Ports the original's erode (label_spacious_span) + recover (recover_erosion_
    spans) as an opening: erode so a cell survives only if the full
    (2ex+1)x(2ey+1) footprint rectangle centred on it lies inside the free region
    (label_spacious_span), then dilate the survivors back — but only onto cells
    that were free in the ORIGINAL region. Opening deletes exactly the areas the
    footprint cannot fit (thin strips, concave nubs, small pockets) while keeping
    the placeable bulk at its true extent. Can split a region, so re-split.
    """
    if ex <= 0 and ey <= 0:
        return [members]
    # Opening is geometric and only needs the first four public fields.  Keep the
    # complete source member by cell, though: the internal ``return_slots`` option
    # adds a fifth field and must survive the optional erosion path too.
    members_by_cell = {}
    bt = {}
    for member in members:
        cell = (member[0], member[1])
        members_by_cell.setdefault(cell, []).append(member)
        bt[cell] = (member[2], member[3])
    # Erode: keep cells whose full footprint rectangle is free.
    centres = set()
    for (cx, cy) in bt:
        ok = True
        for dx in range(-ex, ex + 1):
            for dy in range(-ey, ey + 1):
                if (cx + dx, cy + dy) not in bt:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            centres.add((cx, cy))
    if not centres:
        return []
    # Dilate the surviving centres back, constrained to originally-free cells.
    opened = {}
    for (cx, cy) in centres:
        for dx in range(-ex, ex + 1):
            for dy in range(-ey, ey + 1):
                n = (cx + dx, cy + dy)
                if n in bt and n not in opened:
                    opened[n] = bt[n]
    components = _connected_components(opened)
    if all(len(member) == 4 for member in members):
        return components
    return [
        [member for ix, iy, _bottom, _top in component
         for member in members_by_cell[(ix, iy)]]
        for component in components
    ]


def detect_regions(spans: dict, cell_height_threshold: float,
                   bottom_tolerance: float = 0.1, top_tolerance: float = 0.2,
                   erosion: tuple | None = None, device: str | None = None,
                   return_slots: bool = False):
    """Group span-cells into free regions by matching floor (within
    bottom_tolerance) AND ceiling (within top_tolerance) across 4-neighbours.

    Returns a list of regions; each region is a list of ``(ix, iy, bottom, top)``.
    ``return_slots=True`` is an internal extractor option that adds the source span
    slot as a fifth member, preserving the Warp extractor's representation on CPU
    fallback paths.

    erosion=(ex, ey) (cells) erodes each region by the object's half-footprint
    so surviving cells are valid object centres (polygon path only) — mirrors the
    original's label_spacious_span. Erosion may split a region into several.
    """
    if device is not None and str(device).startswith("cuda"):
        region_list = detect_regions_warp(
            spans, cell_height_threshold, bottom_tolerance, top_tolerance, device=device,
            return_slots=return_slots)
        if erosion is not None and (erosion[0] > 0 or erosion[1] > 0):
            ex, ey = int(erosion[0]), int(erosion[1])
            eroded = []
            for members in region_list:
                # ``slot`` is an internal cell-layer identity, not part of the
                # ordinary region contract.  Preserve it only for the explicit
                # internal request; otherwise retain the historical one-member-per-
                # cell, four-field erosion result.
                eroded.extend(_erode_region(
                    members if return_slots else [member[:4] for member in members], ex, ey
                ))
            return eroded
        return region_list

    nx, ny = spans["nx"], spans["ny"]
    smin, smax, cnt = spans["span_min"], spans["span_max"], spans["span_count"]
    ms = spans["max_spans"]
    node_id = -np.ones((nx, ny, ms), dtype=np.int64)
    nodes: List[Tuple[int, int, float, float, int]] = []
    for ix in range(nx):
        for iy in range(ny):
            for k in range(int(cnt[ix, iy])):
                b = float(smin[ix, iy, k]); t = float(smax[ix, iy, k])
                # `- _HEIGHT_EPS` for the reason the constant was defined and then
                # never used: spans are float32, so a bay of exactly the object's
                # height rounds a hair below it and a strict `>=` drops a shelf that
                # genuinely fits. The rasteriser (`build_spans_raster`) and the
                # clearance re-test (`detect_support_regions`) both already allow it.
                if (t - b) >= cell_height_threshold - _HEIGHT_EPS:
                    node_id[ix, iy, k] = len(nodes)
                    nodes.append((ix, iy, b, t, k))
    if not nodes:
        return []
    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for nid, (ix, iy, b, t, _slot) in enumerate(nodes):
        for dx, dy in ((1, 0), (0, 1)):
            ax, ay = ix + dx, iy + dy
            if ax >= nx or ay >= ny:
                continue
            for k2 in range(int(cnt[ax, ay])):
                nid2 = node_id[ax, ay, k2]
                if nid2 < 0:
                    continue
                b2, t2 = nodes[nid2][2], nodes[nid2][3]
                if abs(b - b2) <= bottom_tolerance and abs(t - t2) <= top_tolerance:
                    ra, rb = find(nid), find(nid2)
                    if ra != rb:
                        parent[rb] = ra
    regions: Dict[int, list] = {}
    for nid, nd in enumerate(nodes):
        regions.setdefault(find(nid), []).append(nd)
    region_list = list(regions.values())

    if erosion is not None and (erosion[0] > 0 or erosion[1] > 0):
        ex, ey = int(erosion[0]), int(erosion[1])
        eroded = []
        for members in region_list:
            # See the CUDA-dispatch branch above: only the internal caller that
            # explicitly requests source slots receives five-field members.
            eroded.extend(_erode_region(
                members if return_slots else [member[:4] for member in members], ex, ey
            ))
        return eroded
    if return_slots:
        return region_list
    return [[member[:4] for member in region] for region in region_list]


@dataclass
class FreePrism:
    """A 2.5D free-space prism: a polygon-with-holes extruded over a height band.

    Non-overlapping by construction (one per connected free region), so it
    represents non-convex free areas exactly — no need for overlapping boxes.
    Matches the original's detected_space_2d schema.
    """
    id: int
    min_height: float
    max_height: float
    outline: List[List[float]]              # [[x, y], ...] world CCW
    holes: List[List[List[float]]] = field(default_factory=list)



def _boundary_loops(cells: set):
    """Directed boundary loops of a cell set, interior-on-the-left.

    Yields loops of integer grid corners; outer loop is CCW (positive signed
    area), holes are CW (negative). Handles holes and multiple loops.
    """
    from collections import defaultdict

    edges = defaultdict(list)  # start corner -> list of end corners
    for (x, y) in cells:
        if (x, y - 1) not in cells:
            edges[(x, y)].append((x + 1, y))        # bottom, +x
        if (x + 1, y) not in cells:
            edges[(x + 1, y)].append((x + 1, y + 1))  # right, +y
        if (x, y + 1) not in cells:
            edges[(x + 1, y + 1)].append((x, y + 1))  # top, -x
        if (x - 1, y) not in cells:
            edges[(x, y + 1)].append((x, y))          # left, -y

    loops = []
    for start in list(edges.keys()):
        while edges.get(start):
            loop = [start]
            cur = edges[start].pop()
            guard = 0
            while cur != start and guard < 10_000_000:
                loop.append(cur)
                nxts = edges.get(cur)
                if not nxts:
                    break
                cur = nxts.pop()
                guard += 1
            if len(loop) >= 4:
                loops.append(loop)
    return loops


def _simplify_collinear(pts):
    """Drop collinear vertices from an axis-aligned integer loop."""
    if len(pts) < 3:
        return pts
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        # keep b only if it is a turn (cross product != 0)
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            out.append(b)
    return out or pts


def _signed_area(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def prism_from_cells(rid: int, cells: set, bb: float, tt: float,
                     cs: float, ox: float, oy: float) -> "FreePrism":
    """Build one FreePrism (outline + holes, world coords) from a region cell set
    and its height band."""
    loops = _boundary_loops(cells)
    outline = None
    holes = []
    best_area = 0.0
    world_loops = []
    for lp in loops:
        lp = _simplify_collinear(lp)
        wl = [(ox + gx * cs, oy + gy * cs) for gx, gy in lp]
        world_loops.append(wl)
    for wl in world_loops:
        a = _signed_area(wl)
        if a > 0:
            if a > best_area:
                best_area = a
                outline = wl
        elif a < 0:
            holes.append(wl)
    if outline is None:
        if not world_loops:
            return None
        outline = max(world_loops, key=lambda w: abs(_signed_area(w)))
    return FreePrism(id=rid, min_height=bb, max_height=tt,
                     outline=[list(p) for p in outline],
                     holes=[[list(p) for p in h] for h in holes])


def prisms_from_region_cells(region_cells, cell_size: float, origin) -> List["FreePrism"]:
    """Convert [(cells, min_h, max_h), ...] into prisms."""
    ox, oy = origin
    out = []
    rid = 0
    for cells, bb, tt in region_cells:
        p = prism_from_cells(rid, cells, bb, tt, cell_size, ox, oy)
        if p is not None:
            out.append(p)
            rid += 1
    return out
