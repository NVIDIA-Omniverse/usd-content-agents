# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window gap-bridging (esd.smooth) — pallet / slatted-surface handling.

Reproduces the original detector's ``preprocess_height_span`` behaviour on the
port: narrow gaps between slats are harmonised into the surrounding deck so an
object larger than a gap can rest across the slats, while real edges (cliffs) and
gaps wider than the object are left alone (never floating a placement over void).

Scene (cell_size 0.05, scope [0,1]^2 x [0,1]):
  * ground slab z in [-0.02, 0]
  * slat A  x in [0.20, 0.40], slat B x in [0.40+gap, 0.60+gap], both z in
    [0.40, 0.50], running the full y in [0.20, 0.80].
So a slat column has support floor 0.50 (top span [0.50, 1.0]); a gap column
casts through to the ground -> open span [0.0, 1.0] (floor 0.0).
"""


import numpy as np
import pytest


wp = pytest.importorskip("warp")
from usd_core.space.smooth import smooth_spans, smooth_spans_warp  # noqa: E402

CELL = 0.05
OBJ = 0.2  # footprint -> window ceil((0.2/2)/0.05) = 2 cells


#: The smoother's contract is over a spans dict, so the fixture builds one directly
#: instead of sampling geometry. The sampler that used to produce it (the original
#: single-ray `kernels.build_spans`) is gone, and going through the rasteriser instead
#: would defeat the point: it bridges a 1-cell gap at rasterisation time, so there
#: would be nothing left for these tests to observe the smoother doing.
def _slatted_spans(gap_cells: int):
    """Spans for a slatted deck over a ground plane, as a raycast sampler saw them.

    Slat A spans x in [0.20, 0.40]; the gap is `gap_cells` wide; slat B follows.
    A slat column supports at 0.5 and also has the crawlspace below it. A gap column
    fell through to the ground, so it is one open span — that fall-through is exactly
    what the sliding window is being asked to repair.
    """
    n = 20                                   # 1.0 m / 0.05
    smin = np.zeros((n, n, 2), dtype=np.float32)
    smax = np.zeros((n, n, 2), dtype=np.float32)
    cnt = np.zeros((n, n), dtype=np.int32)
    slat_a = range(4, 8)                                    # x in [0.20, 0.40)
    b0 = 8 + gap_cells
    slat_b = range(b0, b0 + 4)
    for ix in range(n):
        for iy in range(n):
            on_slat = iy in range(4, 16) and (ix in slat_a or ix in slat_b)
            if on_slat:
                smin[ix, iy, 0], smax[ix, iy, 0] = 0.0, 0.40   # under the deck
                smin[ix, iy, 1], smax[ix, iy, 1] = 0.50, 1.0   # on the deck
                cnt[ix, iy] = 2
            else:
                smin[ix, iy, 0], smax[ix, iy, 0] = 0.0, 1.0    # open to the ground
                cnt[ix, iy] = 1
    return {"nx": n, "ny": n, "max_spans": 2,
            "origin": (0.0, 0.0), "cell_size": CELL,
            "min_z": 0.0, "max_z": 1.0,
            "span_min": smin, "span_max": smax, "span_count": cnt}


def _spans(res, ix, iy):
    c = int(res["span_count"][ix, iy])
    return [
        (round(float(res["span_min"][ix, iy, k]), 3), round(float(res["span_max"][ix, iy, k]), 3))
        for k in range(c)
    ]


def _smooth(raw):
    return smooth_spans(raw, x_length_threshold=OBJ, y_length_threshold=OBJ,
                        section_height_threshold=0.1, bottom_tolerance=0.1)


# ix at y=0.5 (iy=10, inside the slats' y-range):
#   ix 4..7 slat A, ix 8 gap (1-cell), ix 9.. slat B
IY = 10


def test_precondition_gap_is_open_slat_is_high():
    raw = _slatted_spans(gap_cells=1)
    # A slat column supports at 0.5; the top free span is [0.5, 1.0].
    assert (0.5, 1.0) in _spans(raw, 6, IY)
    # The 1-cell gap column fell through to the ground -> single open span.
    assert _spans(raw, 8, IY) == [(0.0, 1.0)]


def test_narrow_gap_is_bridged():
    """A gap narrower than the object radius: the gap column's floor is lifted to
    the surrounding slat level so it merges into the deck."""
    raw = _slatted_spans(gap_cells=1)
    out = _smooth(raw)
    assert _spans(out, 8, IY) == [(0.5, 1.0)], "gap column should join the slat deck at z=0.5"


def test_slat_column_unchanged():
    raw = _slatted_spans(gap_cells=1)
    out = _smooth(raw)
    assert _spans(out, 6, IY) == _spans(raw, 6, IY), "a real slat column must not move"


def test_cliff_edge_not_bridged():
    """SAFETY INVARIANT: a column just outside the pallet (ground on one side,
    slat on the other within the window) must NOT be lifted — otherwise the deck
    region would extend out over open floor and float a placement past the real
    edge. ix=3 is left of slat A (which starts at x=0.20 -> ix=4)."""
    raw = _slatted_spans(gap_cells=1)
    assert _spans(raw, 3, IY) == [(0.0, 1.0)]
    out = _smooth(raw)
    assert _spans(out, 3, IY) == [(0.0, 1.0)], "deck must not extend past its real edge"


def test_wide_gap_not_bridged():
    """A gap wider than the object cannot be spanned: its centre column, with no
    slat within the window on either side, keeps its true (low) floor."""
    raw = _slatted_spans(gap_cells=5)  # 0.25 m > object 0.2 m
    # gap spans ix 8..12; centre ix=10 has open columns for the full window (2).
    assert _spans(raw, 10, IY) == [(0.0, 1.0)]
    out = _smooth(raw)
    assert _spans(out, 10, IY) == [(0.0, 1.0)], "wide-gap centre must stay unbridged"


def test_monotone_bounds():
    """Floors only rise, ceilings only fall, never past the original span — so
    smoothing can never invent vertical free space."""
    raw = _slatted_spans(gap_cells=1)
    out = _smooth(raw)
    nx, ny = raw["nx"], raw["ny"]
    for ix in range(nx):
        for iy in range(ny):
            c = int(raw["span_count"][ix, iy])
            assert int(out["span_count"][ix, iy]) == c
            for k in range(c):
                assert out["span_min"][ix, iy, k] >= raw["span_min"][ix, iy, k] - 1e-9
                assert out["span_max"][ix, iy, k] <= raw["span_max"][ix, iy, k] + 1e-9


def test_idempotent():
    # Bridging is stable under re-application (up to float32 noise; any real
    # second-order change would be O(slat height) = 0.5, not O(1e-6)).
    raw = _slatted_spans(gap_cells=1)
    once = _smooth(raw)
    twice = _smooth(once)
    assert np.allclose(once["span_min"], twice["span_min"], atol=1e-6)
    assert np.allclose(once["span_max"], twice["span_max"], atol=1e-6)


def test_does_not_mutate_input():
    raw = _slatted_spans(gap_cells=1)
    before_min = raw["span_min"].copy()
    _smooth(raw)
    assert np.array_equal(raw["span_min"], before_min)


def test_noop_when_window_is_zero():
    raw = _slatted_spans(gap_cells=1)
    out = smooth_spans(raw, x_length_threshold=0.0, y_length_threshold=0.0,
                       section_height_threshold=0.1)
    assert out is raw  # window collapses to 0 -> untouched dict returned




def test_warp_cpu_matches_numpy_oracle():
    """The device-agnostic Warp kernel reproduces the numpy reference (float32
    tolerance; any real bridge decision is O(slat height) = 0.5, far above)."""
    raw = _slatted_spans(gap_cells=1)
    oracle = _smooth(raw)
    warp_cpu = smooth_spans_warp(raw, OBJ, OBJ, 0.1, bottom_tolerance=0.1, device="cpu")
    assert np.allclose(oracle["span_min"], warp_cpu["span_min"], atol=1e-4)
    assert np.allclose(oracle["span_max"], warp_cpu["span_max"], atol=1e-4)


def test_warp_cpu_cuda_parity():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    raw = _slatted_spans(gap_cells=1)
    cpu = smooth_spans_warp(raw, OBJ, OBJ, 0.1, bottom_tolerance=0.1, device="cpu")
    cuda = smooth_spans_warp(raw, OBJ, OBJ, 0.1, bottom_tolerance=0.1, device="cuda")
    assert np.allclose(cpu["span_min"], cuda["span_min"], atol=1e-5)
    assert np.allclose(cpu["span_max"], cuda["span_max"], atol=1e-5)
