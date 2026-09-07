# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slope handling in the rasterised heightfield (esd.raster.build_spans_raster).

A tilted (ramp) triangle used to write its WHOLE z-range into every cell it
covered, so an incline was reported as a flat wall at its peak and the free space
beside/under it was destroyed. ``slope_mode`` fixes this:

  * "terrain" (default) — the ramp is a true stepped heightfield (support height
    rises with x) and the incline itself is excluded as a rest surface, so the
    only free span rests on the floor UNDER the ramp;
  * "flatten" — the ramp is a solid up to its peak, a flat platform at z=peak;
  * "ignore" — the ramp is invisible (the floor is reported straight through it).

Decks and vertical walls are unaffected by the mode (regression).
"""


import numpy as np
import pytest


wp = pytest.importorskip("warp")
from usd_core.space.heightfield import build_spans_raster  # noqa: E402
from usd_core.space.device_mesh import DeviceMesh  # noqa: E402
from space_synthetic import ramp_mesh, box_mesh, scene_from_boxes, scene_from_meshes  # noqa: E402


def _ramp_scene():
    # flat floor at z=0, plus a ramp rising along +x from z=0 (x=1) to z=1 (x=3).
    floor = box_mesh([-1, 0, -0.1], [5, 3, 0.0])
    ramp = ramp_mesh([1, 0, 0.0], [3, 3, 1.0])  # slope = 0.5/unit-x -> ~26.6 deg
    return scene_from_meshes([floor, ramp], names=["floor", "ramp"])


def _build(scene, mode, device="cpu", cht=0.2):
    rc = DeviceMesh(scene, device=device)
    return build_spans_raster(rc, (-1, 0, -0.5), (5, 3, 3.0), 0.2, cht, slope_mode=mode)


def _spans(sp, x, y):
    ox, oy = sp["origin"]
    cs = sp["cell_size"]
    ix = int((x - ox) / cs)
    iy = int((y - oy) / cs)
    c = int(sp["span_count"][ix, iy])
    return [(round(float(sp["span_min"][ix, iy, k]), 3),
             round(float(sp["span_max"][ix, iy, k]), 3)) for k in range(c)]


def test_terrain_is_stepped_not_peak():
    sp = _build(_ramp_scene(), "terrain")
    low = _spans(sp, 1.5, 1.5)   # low on the ramp
    high = _spans(sp, 2.5, 1.5)  # higher on the ramp
    # One free span per cell, resting on the FLOOR under the ramp, up to the local
    # ramp underside — the height rises with x (a real slope, not a flat peak).
    assert len(low) == 1 and len(high) == 1, (low, high)
    assert low[0][0] == 0.0 and high[0][0] == 0.0
    assert 0.0 < low[0][1] < high[0][1] < 1.0, (low, high)


def test_terrain_excludes_the_incline_as_support():
    # No free span ever RESTS on the ramp surface (steep layer excluded). Every
    # emitted span over the ramp rests on the floor below (floor == 0.0); near the
    # bottom the under-ramp gap is thinner than the object and is correctly dropped.
    sp = _build(_ramp_scene(), "terrain")
    saw_span = False
    for x in (1.3, 1.9, 2.7):
        for (lo, hi) in _spans(sp, x, 1.5):
            assert lo == 0.0, (x, lo, hi)  # never rests on the incline
            saw_span = True
    assert saw_span  # the wider under-ramp region does yield placements


def test_flatten_is_a_flat_platform_at_peak():
    sp = _build(_ramp_scene(), "flatten")
    # The ramp becomes a solid up to its peak (z=1) at every covered cell: the
    # only free space is ABOVE the peak, identical low and high on the ramp.
    assert _spans(sp, 1.5, 1.5) == [(1.0, 3.0)]
    assert _spans(sp, 2.5, 1.5) == [(1.0, 3.0)]


def test_ignore_reports_floor_through_ramp():
    sp = _build(_ramp_scene(), "ignore")
    # The ramp contributes nothing; the floor is reported straight through it.
    assert _spans(sp, 1.5, 1.5) == [(0.0, 3.0)]
    assert _spans(sp, 2.5, 1.5) == [(0.0, 3.0)]


def test_open_floor_identical_across_modes():
    # A cell with no ramp above it is the same in every mode.
    want = [(0.0, 3.0)]
    for mode in ("terrain", "flatten", "ignore"):
        sp = _build(_ramp_scene(), mode)
        assert _spans(sp, 4.5, 1.5) == want, mode


def test_axis_aligned_scene_unaffected_by_mode():
    # Decks (horizontal) and walls (vertical) are never ramps -> byte-identical
    # results regardless of slope_mode (no behaviour change for warehouse geo).
    scene = scene_from_boxes([
        ([-1, -1, -0.1], [5, 3, 0.0]),
        ([0.0, 0.0, 1.0], [2.0, 3, 1.1]),
        ([3.0, 0.0, 0.0], [3.1, 3, 2.0]),
    ])
    ref = None
    for mode in ("terrain", "flatten", "ignore"):
        rc = DeviceMesh(scene, device="cpu")
        sp = build_spans_raster(rc, (-1, -1, -0.5), (5, 3, 3.0), 0.2, 0.2, slope_mode=mode)
        if ref is None:
            ref = sp
        else:
            assert np.array_equal(ref["span_count"], sp["span_count"]), mode
            assert np.allclose(ref["span_min"], sp["span_min"]), mode
            assert np.allclose(ref["span_max"], sp["span_max"]), mode


def test_gentle_slope_under_threshold_is_a_deck():
    """A ramp gentler than the threshold is a rest surface, at its OWN height.

    The face is clipped to each cell, so the deck rasterises where it actually is —
    z = 0.25 at x = 5 on a 0.5-over-8 rise — rather than as a slab up to its peak.
    Two spans follow, and both are real: the object-height gap between the ground
    and the underside of this floating deck, and the space resting on the deck. The
    whole-triangle over-mark used to hide the first and misreport the second.
    """
    floor = box_mesh([-1, 0, -0.1], [9, 3, 0.0])
    gentle = ramp_mesh([1, 0, 0.0], [9, 3, 0.5])  # slope ~0.0625/unit -> ~3.6 deg
    scene = scene_from_meshes([floor, gentle])
    rc = DeviceMesh(scene, device="cpu")
    sp = build_spans_raster(rc, (-1, 0, -0.5), (9, 3, 3.0), 0.2, 0.2,
                            slope_mode="terrain", slope_threshold_deg=20.0)
    s = _spans(sp, 5.0, 1.5)
    on_deck = [span for span in s if span[0] > 0.2]
    assert len(on_deck) == 1, s
    # The deck's true height at x=5, plus at most the cell's own rise (0.2 * 0.0625).
    assert 0.25 <= on_deck[0][0] <= 0.25 + 0.2 * 0.0625 + 1e-3, on_deck
    under_deck = [span for span in s if span[0] <= 0.2]
    assert len(under_deck) == 1 and under_deck[0][1] <= 0.25 + 1e-3, s


def test_invalid_slope_mode_raises():
    scene = _ramp_scene()
    rc = DeviceMesh(scene, device="cpu")
    with pytest.raises(ValueError):
        build_spans_raster(rc, (-1, 0, -0.5), (5, 3, 3.0), 0.2, 0.2, slope_mode="bogus")


def test_cpu_cuda_parity_terrain():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    cpu = _build(_ramp_scene(), "terrain", device="cpu")
    cuda = _build(_ramp_scene(), "terrain", device="cuda")
    assert np.array_equal(cpu["span_count"], cuda["span_count"])
    assert np.allclose(cpu["span_min"], cuda["span_min"], atol=1e-5)
    assert np.allclose(cpu["span_max"], cuda["span_max"], atol=1e-5)


def test_a_tilted_face_is_clipped_to_each_cell_not_marked_to_its_peak():
    """The rasteriser records the face's z range WITHIN a column, as Recast does.

    A tilted triangle used to write its whole z-range into every cell it covered, so a
    ramp below `slope_threshold_deg` — a deck that happens to fall away — reported a
    flat surface at its high end. The error was the size of the triangle, not of the
    cell: on this 0.5-over-2 m ramp the low end was reported 0.475 m above where the
    surface actually is, and a query answered `ok: true` for an object resting there.
    """
    ramp = ramp_mesh([-1, -1, 0.5], [1, 1, 0.0])  # 0.5 m fall over 2 m -> ~14 deg
    rc = DeviceMesh(scene_from_meshes([ramp], names=["ramp"]), device="cpu")
    sp = build_spans_raster(rc, (-1, -1, -0.1), (1, 1, 3.0), 0.05, 0.3)
    cs = sp["cell_size"]
    for x, truth in ((-0.9, 0.475), (-0.5, 0.375), (0.0, 0.25), (0.5, 0.125), (0.9, 0.025)):
        floor = _spans(sp, x, 0.0)[0][0]
        # Within the cell's own rise: the face is bounded by the four cell corners.
        assert abs(floor - truth) <= 0.25 * cs + 1e-3, (x, floor, truth)
