# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window height-span smoothing — pallet / slatted-surface gap bridging.

Faithful port of the original detector's ``preprocess_height_span`` pass
(``space_detect_manager.py:cache_height_span_info`` ->
``space_detect_helper.py:update_height_deque`` / ``update_height_span_section``
/ ``_merge_height_ranges`` / ``_filter_height_ranges`` / ``split_height_span``).

Motivation (the "Empty Space Detection Challenge" slide): a pallet / grating /
slatted deck has small **gaps** between the slats. A down-cast ray that falls
through a gap hits the floor far below, so that column's support floor is much
lower than the slat columns beside it. Without correction the region grouper
(``extract.detect_regions``, which merges cells only when BOTH floor and ceiling
match) sees the gap columns as a different level and fragments the deck into
slat-sized slivers full of holes — even though an object larger than a gap can
obviously rest across the slats.

This pass **harmonises** each column's span to the surrounding surface over a
window of ``object-radius`` cells, so gap columns acquire the slats' floor (and
ceiling) and merge into one continuous placement surface.

Safety — two independent guards, so this can only ever be conservative:

1. **Monotone bounds.** A span's floor is only ever RAISED and its ceiling only
   ever LOWERED, and never past its own original ``[b, t]``. So the smoothed
   column never claims vertical free space the raycast did not find.
2. **Both-sides-must-match (the real correctness invariant).** A column's floor
   is lifted to a neighbouring level only when that level is present on BOTH
   opposing sides within the window (``merge(-x, +x)`` requires matching floors;
   likewise ``-y, +y``), and is not blocked by a taller obstacle poking up from
   the perpendicular axis. This is what stops the deck from being extended out
   over a void at a real edge: a cliff column (deck on one side, void on the
   other, all within the object radius) is left untouched, so an object is never
   floated past the true surface boundary.

The window is ``ceil((object_length / 2) / cell_size)`` cells per axis — the
object radius in cells, exactly the original ``calculate_erosion_scale``.

NOTE: this is a correctness-first CPU pass (the original was pure-Python too). It
is O(nx * ny * spans * 4 * W); a Warp-kernel version (windowed max-floor /
min-ceiling with a connectivity gate) is a natural follow-up for the 256^2 hot
path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    from ._warp import warp as _warp

    wp = _warp()
    _HAS_WARP = True
except Exception:  # noqa: BLE001 — no warp: smooth_spans_warp falls back to the numpy path
    _HAS_WARP = False


def _dir_interval(
    smin: np.ndarray,
    smax: np.ndarray,
    cnt: np.ndarray,
    nx: int,
    ny: int,
    ix: int,
    iy: int,
    b: float,
    t: float,
    dx: int,
    dy: int,
    w: int,
    sec: float,
) -> Optional[Tuple[float, float]]:
    """Connected free interval seen from ``(ix, iy)`` looking ``w`` cells along
    ``(dx, dy)``.

    Starts from the column's own span ``[b, t]`` and, at each step, intersects
    the neighbour span that overlaps the running interval by at least ``sec``
    (the ``relative_connected`` test in the original). Returns
    ``(floor, ceiling)`` = ``(max of floors, min of ceilings)`` over the walked
    run, or ``None`` if nothing is connected in that direction (a void / edge /
    boundary) or the surviving interval is thinner than ``sec``.
    """
    fl, ce = float(b), float(t)
    cx, cy = ix, iy
    connected = False
    for _ in range(w):
        cx += dx
        cy += dy
        if cx < 0 or cy < 0 or cx >= nx or cy >= ny:
            break
        best_k = -1
        best_ov = sec  # require overlap >= sec to be "connected"
        for k in range(int(cnt[cx, cy])):
            nb = float(smin[cx, cy, k])
            nt = float(smax[cx, cy, k])
            ov = min(ce, nt) - max(fl, nb)
            if ov >= best_ov:
                best_ov = ov
                best_k = k
        if best_k < 0:
            break  # no connected span this way -> stop the run
        nb = float(smin[cx, cy, best_k])
        nt = float(smax[cx, cy, best_k])
        fl = max(fl, nb)
        ce = min(ce, nt)
        connected = True
    if not connected:
        return None
    if (ce - fl) < sec:
        return None
    return (fl, ce)


def _merge(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
    bottom_tol: float,
    sec: float,
) -> Optional[Tuple[float, float]]:
    """Original ``_merge_height_ranges`` for a single interval per side: keep the
    common interval only when the two opposing floors agree within
    ``bottom_tol`` (both sides are the same level)."""
    if a is None or b is None:
        return None
    if abs(a[0] - b[0]) < bottom_tol:
        fl = max(a[0], b[0])
        ce = min(a[1], b[1])
        if (ce - fl) >= sec:
            return (fl, ce)
    return None


def _blocked(
    cand: Optional[Tuple[float, float]],
    perp: Optional[Tuple[float, float]],
    tol: float,
) -> bool:
    """Original ``_filter_height_ranges`` predicate: a candidate range is blocked
    when a perpendicular neighbour's floor pokes strictly inside it
    (``f_min > t_min + tol and f_min < t_max``) — i.e. a taller obstacle sits
    across the bridging direction."""
    if cand is None or perp is None:
        return False
    pf = perp[0]
    return (pf > cand[0] + tol) and (pf < cand[1])


def smooth_spans(
    spans: dict,
    x_length_threshold: float,
    y_length_threshold: float,
    section_height_threshold: float,
    bottom_tolerance: float = 0.1,
    tol: float = 0.02,
) -> dict:
    """Return a new spans dict with narrow-gap columns harmonised to the
    surrounding surface (see module docstring).

    Args:
        spans: the SoA dict from ``heightfield.build_spans_raster``.
        x_length_threshold, y_length_threshold: object footprint (metres). The
            per-axis window is ``ceil((length / 2) / cell_size)`` cells.
        section_height_threshold: minimum free height for a span to count
            (``cell_height_threshold`` in the caller); also the connectivity
            overlap threshold.
        bottom_tolerance: floors within this agree (both-sides-match test).
        tol: perpendicular-block tolerance (original default 0.02).

    Does not mutate ``spans``; only ``span_min`` / ``span_max`` are rewritten
    (``span_count`` and every other field are preserved — this pass never adds,
    drops, or reorders spans, it only tightens each span's bounds).
    """
    nx, ny, ms = int(spans["nx"]), int(spans["ny"]), int(spans["max_spans"])
    cs = float(spans["cell_size"])
    smin = spans["span_min"]
    smax = spans["span_max"]
    cnt = spans["span_count"]

    wx = int(math.ceil((float(x_length_threshold) / 2.0) / cs)) if cs > 0 else 0
    wy = int(math.ceil((float(y_length_threshold) / 2.0) / cs)) if cs > 0 else 0
    if wx <= 0 and wy <= 0:
        return spans  # object smaller than a cell -> nothing to bridge

    sec = float(section_height_threshold)
    out_min = smin.copy()
    out_max = smax.copy()

    for ix in range(nx):
        for iy in range(ny):
            c = int(cnt[ix, iy])
            for k in range(c):
                b = float(smin[ix, iy, k])
                t = float(smax[ix, iy, k])

                left = _dir_interval(smin, smax, cnt, nx, ny, ix, iy, b, t, -1, 0, wx, sec)
                right = _dir_interval(smin, smax, cnt, nx, ny, ix, iy, b, t, 1, 0, wx, sec)
                down = _dir_interval(smin, smax, cnt, nx, ny, ix, iy, b, t, 0, -1, wy, sec)
                up = _dir_interval(smin, smax, cnt, nx, ny, ix, iy, b, t, 0, 1, wy, sec)

                horizontal = _merge(left, right, bottom_tolerance, sec)
                vertical = _merge(down, up, bottom_tolerance, sec)

                # Original update_height_span_section: start from horizontal,
                # drop it if a perpendicular (down/up) obstacle blocks it; else
                # fall back to vertical filtered by (left/right).
                final = horizontal
                if final is not None and (_blocked(final, down, tol) or _blocked(final, up, tol)):
                    final = None
                if final is None:
                    v = vertical
                    if v is not None and (_blocked(v, left, tol) or _blocked(v, right, tol)):
                        v = None
                    final = v

                if final is None:
                    continue  # safe fallback: keep the raw span unchanged

                # Clamp to the column's own span; floor only rises, ceiling only
                # falls. Keep only if a real section survives, else leave raw.
                nf = max(b, final[0])
                nc = min(t, final[1])
                if (nc - nf) >= sec:
                    out_min[ix, iy, k] = nf
                    out_max[ix, iy, k] = nc

    result = dict(spans)
    result["span_min"] = out_min
    result["span_max"] = out_max
    return result


# ---------------------------------------------------------------------------
# Warp (GPU / device-agnostic) implementation.
#
# One thread per (cell, span-slot): each thread reads the INPUT span arrays
# (read-only, shared) and writes only its own slot in the OUTPUT arrays, so the
# whole pass is race-free and embarrassingly parallel — the same structure as
# Runs on 'cpu' or 'cuda' from one source.
# The numpy ``smooth_spans`` above stays the readable reference oracle; a parity
# test pins this kernel to it (see tests/test_smooth.py).
# ---------------------------------------------------------------------------

if _HAS_WARP:

    @wp.func
    def _wp_dir_interval(
        smin: wp.array(dtype=wp.float32),
        smax: wp.array(dtype=wp.float32),
        scnt: wp.array(dtype=wp.int32),
        nx: wp.int32,
        ny: wp.int32,
        ms: wp.int32,
        ix: wp.int32,
        iy: wp.int32,
        b: wp.float32,
        t: wp.float32,
        dx: wp.int32,
        dy: wp.int32,
        w: wp.int32,
        sec: wp.float32,
    ) -> wp.vec3:
        # returns (floor, ceiling, ok) — ok==1.0 when a section >= sec is
        # connected along (dx,dy) within w cells, matching _dir_interval above.
        fl = b
        ce = t
        cx = ix
        cy = iy
        connected = float(0.0)
        step = int(0)
        while step < w:
            step += 1
            cx += dx
            cy += dy
            if cx < 0 or cy < 0 or cx >= nx or cy >= ny:
                break
            cell = cx * ny + cy
            cnt = scnt[cell]
            base = cell * ms
            best_k = int(-1)
            best_ov = sec
            k = int(0)
            while k < cnt:
                nb = smin[base + k]
                nt = smax[base + k]
                ov = wp.min(ce, nt) - wp.max(fl, nb)
                if ov >= best_ov:
                    best_ov = ov
                    best_k = k
                k += 1
            if best_k < 0:
                break
            fl = wp.max(fl, smin[base + best_k])
            ce = wp.min(ce, smax[base + best_k])
            connected = 1.0
        ok = float(0.0)
        if connected > 0.5 and (ce - fl) >= sec:
            ok = 1.0
        return wp.vec3(fl, ce, ok)

    @wp.func
    def _wp_merge(a: wp.vec3, b: wp.vec3, bottom_tol: wp.float32, sec: wp.float32) -> wp.vec3:
        if a[2] < 0.5 or b[2] < 0.5:
            return wp.vec3(0.0, 0.0, 0.0)
        if wp.abs(a[0] - b[0]) < bottom_tol:
            fl = wp.max(a[0], b[0])
            ce = wp.min(a[1], b[1])
            if (ce - fl) >= sec:
                return wp.vec3(fl, ce, 1.0)
        return wp.vec3(0.0, 0.0, 0.0)

    @wp.func
    def _wp_blocked(cand: wp.vec3, perp: wp.vec3, tol: wp.float32) -> wp.float32:
        if cand[2] < 0.5 or perp[2] < 0.5:
            return 0.0
        pf = perp[0]
        if pf > cand[0] + tol and pf < cand[1]:
            return 1.0
        return 0.0

    @wp.kernel
    def _smooth_kernel(
        smin: wp.array(dtype=wp.float32),
        smax: wp.array(dtype=wp.float32),
        scnt: wp.array(dtype=wp.int32),
        nx: wp.int32,
        ny: wp.int32,
        ms: wp.int32,
        wx: wp.int32,
        wy: wp.int32,
        sec: wp.float32,
        bottom_tol: wp.float32,
        tol: wp.float32,
        out_min: wp.array(dtype=wp.float32),
        out_max: wp.array(dtype=wp.float32),
    ):
        gid = wp.tid()
        cell = gid // ms
        k = gid % ms
        cnt = scnt[cell]
        if k >= cnt:
            out_min[gid] = smin[gid]  # passthrough unused slot
            out_max[gid] = smax[gid]
            return
        ix = cell // ny
        iy = cell % ny
        b = smin[gid]
        t = smax[gid]

        left = _wp_dir_interval(smin, smax, scnt, nx, ny, ms, ix, iy, b, t, -1, 0, wx, sec)
        right = _wp_dir_interval(smin, smax, scnt, nx, ny, ms, ix, iy, b, t, 1, 0, wx, sec)
        down = _wp_dir_interval(smin, smax, scnt, nx, ny, ms, ix, iy, b, t, 0, -1, wy, sec)
        up = _wp_dir_interval(smin, smax, scnt, nx, ny, ms, ix, iy, b, t, 0, 1, wy, sec)

        horizontal = _wp_merge(left, right, bottom_tol, sec)
        vertical = _wp_merge(down, up, bottom_tol, sec)

        final = horizontal
        if final[2] > 0.5:
            if _wp_blocked(final, down, tol) > 0.5 or _wp_blocked(final, up, tol) > 0.5:
                final = wp.vec3(0.0, 0.0, 0.0)
        if final[2] < 0.5:
            v = vertical
            if v[2] > 0.5:
                if _wp_blocked(v, left, tol) > 0.5 or _wp_blocked(v, right, tol) > 0.5:
                    v = wp.vec3(0.0, 0.0, 0.0)
            final = v

        nf = b
        nc = t
        if final[2] > 0.5:
            cand_f = wp.max(b, final[0])
            cand_c = wp.min(t, final[1])
            if (cand_c - cand_f) >= sec:
                nf = cand_f
                nc = cand_c
        out_min[gid] = nf
        out_max[gid] = nc


def smooth_spans_warp(
    spans: dict,
    x_length_threshold: float,
    y_length_threshold: float,
    section_height_threshold: float,
    bottom_tolerance: float = 0.1,
    tol: float = 0.02,
    device: str = "cpu",
) -> dict:
    """Device-agnostic Warp version of :func:`smooth_spans` (same semantics,
    same output). Falls back to the numpy path if Warp is unavailable.

    ``device`` is 'cpu' or 'cuda'; the kernel runs identically on both.
    """
    if not _HAS_WARP:
        return smooth_spans(spans, x_length_threshold, y_length_threshold,
                            section_height_threshold, bottom_tolerance, tol)

    nx, ny, ms = int(spans["nx"]), int(spans["ny"]), int(spans["max_spans"])
    cs = float(spans["cell_size"])
    wx = int(math.ceil((float(x_length_threshold) / 2.0) / cs)) if cs > 0 else 0
    wy = int(math.ceil((float(y_length_threshold) / 2.0) / cs)) if cs > 0 else 0
    if wx <= 0 and wy <= 0:
        return spans

    smin = np.ascontiguousarray(spans["span_min"].reshape(-1), dtype=np.float32)
    smax = np.ascontiguousarray(spans["span_max"].reshape(-1), dtype=np.float32)
    scnt = np.ascontiguousarray(spans["span_count"].reshape(-1), dtype=np.int32)
    n = nx * ny * ms

    d_smin = wp.array(smin, dtype=wp.float32, device=device)
    d_smax = wp.array(smax, dtype=wp.float32, device=device)
    d_scnt = wp.array(scnt, dtype=wp.int32, device=device)
    d_omin = wp.zeros(n, dtype=wp.float32, device=device)
    d_omax = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(
        _smooth_kernel,
        dim=n,
        inputs=[d_smin, d_smax, d_scnt, wp.int32(nx), wp.int32(ny), wp.int32(ms),
                wp.int32(wx), wp.int32(wy), wp.float32(section_height_threshold),
                wp.float32(bottom_tolerance), wp.float32(tol)],
        outputs=[d_omin, d_omax],
        device=device,
    )

    result = dict(spans)
    result["span_min"] = d_omin.numpy().reshape(nx, ny, ms)
    result["span_max"] = d_omax.numpy().reshape(nx, ny, ms)
    return result
