# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slat-gap bridging end to end — original tests, not ported from upstream.

The rest of `tests/space/` is upstream's compute-layer suite; this covers a usd-cli
addition (`space.gapfill`), so it is not part of a re-sync.

The bug it guards: `smooth` bridges a slatted deck by lifting a gap column's floor to
the level its neighbours agree on — a function of the left and right heights alone,
which is right. But it "never adds, drops, or reorders spans", so it can only lift a
span that exists. Under the rasterised span builder a gap column with no geometry under
it has no span, the lift silently does not happen, and the footprint erosion in
`detect_support_regions` widens each such hole into a break — turning one continuous EUR
pallet deck into disconnected strips.

`gapfill` supplies the missing span, at the NEIGHBOURS' agreed height. The obvious
alternative — having the span builder emit a full-height span for every empty column —
was tried and reverted, and `test_empty_space_is_not_a_support_surface` is why: it
floors every empty cell at the scope bottom, and the region grouper then joins them into
a large phantom resting surface wherever the scene is simply empty.
"""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from usd_core.space import gapfill  # noqa: E402
from usd_core.space import extract as extract_module  # noqa: E402
from usd_core.space.errors import SpaceOutOfRange  # noqa: E402
from usd_core.space.extract import (MAX_WARP_FULL_SUPPORT_WINDOW_CELLS,
                                    MAX_WARP_LABEL_PASSES, detect_regions,
                                    detect_regions_warp, full_support_centers_warp)  # noqa: E402
from usd_core.space.gapfill import fill_empty_columns  # noqa: E402
from usd_core.space.heightfield import build_spans_raster  # noqa: E402
from usd_core.space.device_mesh import DeviceMesh  # noqa: E402
from usd_core.space.smooth import smooth_spans_warp  # noqa: E402
from space_synthetic import scene_from_boxes  # noqa: E402

CELL = 0.02
SCOPE_MIN = (0.0, 0.0, 0.0)
SCOPE_MAX = (1.0, 1.0, 0.6)
DECK_TOP = 0.30


def _slatted_deck(gap: float, slat: float = 0.16):
    """Slats along X, separated along Y by `gap`, over OPEN AIR.

    Open air underneath is the point: it is what makes the gap columns empty, which is
    the case that used to produce no span at all.
    """
    boxes, y = [], 0.0
    while y < 1.0:
        boxes.append(((0.0, y, DECK_TOP - 0.02), (1.0, min(y + slat, 1.0), DECK_TOP)))
        y += slat + gap
    return scene_from_boxes(boxes)


def _stacked_slatted_decks(gap: float, slat: float = 0.16):
    """Two aligned slatted decks: each gap is empty through both levels."""
    boxes, y = [], 0.0
    while y < 1.0:
        top = min(y + slat, 1.0)
        boxes.extend(
            [
                ((0.0, y, 0.28), (1.0, top, 0.30)),
                ((0.0, y, 1.08), (1.0, top, 1.10)),
            ]
        )
        y += slat + gap
    return scene_from_boxes(boxes)


def _lower_floor_with_upper_slatted_deck(gap: float, slat: float = 0.16):
    """One continuous lower floor, then an upper deck whose slat gaps stay open."""
    boxes = [((0.0, 0.0, 0.28), (1.0, 1.0, 0.30))]
    y = 0.0
    while y < 1.0:
        boxes.append(((0.0, y, 1.08), (1.0, min(y + slat, 1.0), 1.10)))
        y += slat + gap
    return scene_from_boxes(boxes)


def _spans(scene, obj, smooth=True):
    """The real support pipeline: raster -> gapfill -> smooth."""
    raw = build_spans_raster(
        DeviceMesh(scene, device="cpu"), SCOPE_MIN, SCOPE_MAX, CELL, obj[2]
    )
    if not smooth:
        return raw
    filled = fill_empty_columns(raw, obj[0], obj[1], obj[2])
    return smooth_spans_warp(
        filled,
        x_length_threshold=obj[0],
        y_length_threshold=obj[1],
        section_height_threshold=obj[2],
        bottom_tolerance=0.1,
        device="cpu",
    )


def _floors(spans, ix):
    return [
        float(spans["span_min"][ix, iy, 0])
        if int(spans["span_count"][ix, iy])
        else None
        for iy in range(spans["ny"])
    ]


def _layered_gap_spans(*, upper_ceiling=2.0, upper_right_floor=1.1):
    """Three columns: two multi-level slats around one completely open gap."""
    smin = np.zeros((3, 1, 3), dtype=np.float32)
    smax = np.zeros((3, 1, 3), dtype=np.float32)
    count = np.zeros((3, 1), dtype=np.int32)
    # Each slat has a lower bay and an upper resting surface.  The central gap has
    # no geometry at all, so gapfill must synthesize BOTH layers from its sides.
    for ix, upper_floor in ((0, 1.1), (2, upper_right_floor)):
        smin[ix, 0, :2] = (0.0, upper_floor)
        smax[ix, 0, :2] = (1.0, upper_ceiling)
        count[ix, 0] = 2
    return {
        "nx": 3,
        "ny": 1,
        "max_spans": 3,
        "cell_size": 0.1,
        "span_min": smin,
        "span_max": smax,
        "span_count": count,
        "solid_count": np.zeros((3, 1), dtype=np.int32),
    }


def test_gapfill_preserves_every_aligned_layer_and_common_headroom():
    """A multi-level slatted rack must bridge its upper deck, not only slot zero."""
    bridged = fill_empty_columns(
        _layered_gap_spans(), 0.2, 0.2, 0.2, bottom_tolerance=0.02
    )
    assert int(bridged["span_count"][1, 0]) == 2
    levels = [
        (float(bridged["span_min"][1, 0, k]), float(bridged["span_max"][1, 0, k]))
        for k in range(2)
    ]
    assert levels[0] == pytest.approx((0.0, 1.0))
    assert levels[1] == pytest.approx((1.1, 2.0))


def test_gapfill_rejects_misaligned_or_too_short_upper_layer():
    """Both sides must align and their shared ceiling must clear the object."""
    stepped = fill_empty_columns(
        _layered_gap_spans(upper_right_floor=1.15), 0.2, 0.2, 0.2, bottom_tolerance=0.02
    )
    assert int(stepped["span_count"][1, 0]) == 1

    low_ceiling = fill_empty_columns(
        _layered_gap_spans(upper_ceiling=1.25), 0.2, 0.2, 0.2, bottom_tolerance=0.02
    )
    assert int(low_ceiling["span_count"][1, 0]) == 1


def test_gapfill_adds_upper_deck_above_an_existing_lower_free_span():
    """A slat gap can be open above a real lower floor without losing either level."""
    spans = _layered_gap_spans()
    # The central gap has a measured lower-floor span through the entire upper opening.
    # It is not an empty column, which is the rack topology the old guard skipped.
    spans["span_min"][1, 0, 0] = 0.0
    spans["span_max"][1, 0, 0] = 2.0
    spans["span_count"][1, 0] = 1

    bridged = fill_empty_columns(spans, 0.2, 0.2, 0.2, bottom_tolerance=0.02)
    assert int(bridged["span_count"][1, 0]) == 2
    levels = [
        (float(bridged["span_min"][1, 0, k]), float(bridged["span_max"][1, 0, k]))
        for k in range(2)
    ]
    assert levels[0] == pytest.approx((0.0, 2.0))
    assert levels[1] == pytest.approx((1.1, 2.0))


def test_gapfill_skips_lower_only_gap_columns_to_find_upper_deck():
    """A continuous lower floor must not shadow farther upper-deck slats."""
    smin = np.zeros((5, 1, 2), dtype=np.float32)
    smax = np.zeros((5, 1, 2), dtype=np.float32)
    count = np.ones((5, 1), dtype=np.int32)
    # Every column has the lower free band.  Only the two outer slats carry the
    # upper level, so an any-span nearest map would stop at the lower-only gaps.
    smin[:, 0, 0], smax[:, 0, 0] = 0.0, 2.0
    smin[[0, 4], 0, 1], smax[[0, 4], 0, 1] = 1.1, 2.0
    count[[0, 4], 0] = 2
    bridged = fill_empty_columns(
        {
            "nx": 5,
            "ny": 1,
            "max_spans": 2,
            "cell_size": 0.1,
            "span_min": smin,
            "span_max": smax,
            "span_count": count,
            "solid_count": np.zeros((5, 1), dtype=np.int32),
        },
        0.6,
        0.2,
        0.2,
        bottom_tolerance=0.02,
    )
    assert int(bridged["span_count"][2, 0]) == 2
    assert tuple(bridged["span_min"][2, 0, :2]) == pytest.approx((0.0, 1.1))


def test_gapfill_never_adds_an_upper_level_through_a_solid():
    """A lower span below a wall is not evidence that the wall's upper interval is free."""
    spans = _layered_gap_spans()
    spans["span_min"][1, 0, 0] = 0.0
    spans["span_max"][1, 0, 0] = 1.0
    spans["span_count"][1, 0] = 1
    spans["solid_count"][1, 0] = 1

    bridged = fill_empty_columns(spans, 0.2, 0.2, 0.2, bottom_tolerance=0.02)
    assert int(bridged["span_count"][1, 0]) == 1


def test_gapfill_reports_when_crossing_levels_exceed_slot_capacity():
    """Synthetic levels must never be silently truncated when both axes find one."""
    smin = np.zeros((3, 3, 1), dtype=np.float32)
    smax = np.zeros((3, 3, 1), dtype=np.float32)
    count = np.zeros((3, 3), dtype=np.int32)
    # X witnesses a lower deck; Y witnesses an independent upper deck.  This is the
    # smallest possible case where an empty column has two valid synthetic levels but
    # the storage has one slot.
    for ix, iy, floor, ceiling in (
        (0, 1, 0.0, 1.0),
        (2, 1, 0.0, 1.0),
        (1, 0, 1.1, 2.0),
        (1, 2, 1.1, 2.0),
    ):
        smin[ix, iy, 0], smax[ix, iy, 0], count[ix, iy] = floor, ceiling, 1
    bridged = fill_empty_columns(
        {
            "nx": 3,
            "ny": 3,
            "max_spans": 1,
            "cell_size": 0.1,
            "span_min": smin,
            "span_max": smax,
            "span_count": count,
            "solid_count": np.zeros((3, 3), dtype=np.int32),
        },
        0.2,
        0.2,
        0.2,
        bottom_tolerance=0.02,
    )
    assert int(bridged["span_count"][1, 1]) == 1
    assert int(bridged["gapfill_overflow_cells"]) == 1


def test_gapfill_uses_storage_capacity_not_observed_level_count():
    """An initially one-level scene may still bridge two valid levels into an empty slot."""
    smin = np.zeros((3, 3, 2), dtype=np.float32)
    smax = np.zeros((3, 3, 2), dtype=np.float32)
    count = np.zeros((3, 3), dtype=np.int32)
    # X witnesses a lower deck and Y witnesses an independent upper deck.  No source
    # column has more than one observed level, but the empty centre has storage for both.
    for ix, iy, floor, ceiling in (
        (0, 1, 0.0, 1.0),
        (2, 1, 0.0, 1.0),
        (1, 0, 1.1, 2.0),
        (1, 2, 1.1, 2.0),
    ):
        smin[ix, iy, 0], smax[ix, iy, 0], count[ix, iy] = floor, ceiling, 1
    bridged = fill_empty_columns(
        {
            "nx": 3,
            "ny": 3,
            "max_spans": 2,
            "cell_size": 0.1,
            "span_min": smin,
            "span_max": smax,
            "span_count": count,
            "solid_count": np.zeros((3, 3), dtype=np.int32),
        },
        0.2,
        0.2,
        0.2,
        bottom_tolerance=0.02,
    )
    assert int(bridged["span_count"][1, 1]) == 2
    assert tuple(bridged["span_min"][1, 1, :2]) == pytest.approx((0.0, 1.1))
    assert int(bridged.get("gapfill_overflow_cells", 0)) == 0


def test_gapfill_uniform_single_level_bulk_path_matches_general_path(monkeypatch):
    """The fast pallet-deck path is an optimisation, not a second geometry rule."""
    n, slots = 17, 16
    spans = {
        "nx": n,
        "ny": n,
        "max_spans": slots,
        "cell_size": 0.1,
        "span_min": np.zeros((n, n, slots), dtype=np.float32),
        "span_max": np.full((n, n, slots), 2.0, dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.ones((n, n), dtype=np.int32),
    }
    # Regular open slat gaps, including intersections, all bridged at one level.
    spans["span_count"][::4, :] = 0
    spans["solid_count"][::4, :] = 0
    spans["span_count"][:, ::5] = 0
    spans["solid_count"][:, ::5] = 0

    vectorized = gapfill._fill_single_level_vectorized
    calls = []

    def counted_fast_path(*args, **kwargs):
        calls.append(True)
        return vectorized(*args, **kwargs)

    monkeypatch.setattr(gapfill, "_fill_single_level_vectorized", counted_fast_path)
    fast = fill_empty_columns(spans, 0.4, 0.4, 0.3)
    assert calls, "uniform pallet input must select the vectorized acceleration path"
    assert np.count_nonzero(fast["span_count"] > spans["span_count"]) > 0, (
        "the fixture must bridge a real gap for fast/general parity to be meaningful"
    )
    monkeypatch.setattr(
        gapfill, "_fill_single_level_vectorized", lambda *_a, **_kw: None
    )
    general = fill_empty_columns(spans, 0.4, 0.4, 0.3)

    np.testing.assert_array_equal(fast["span_count"], general["span_count"])
    np.testing.assert_allclose(fast["span_min"], general["span_min"])
    np.testing.assert_allclose(fast["span_max"], general["span_max"])
    assert fast.get("gapfill_overflow_cells", 0) == general.get(
        "gapfill_overflow_cells", 0
    )


def test_gapfill_fast_path_preserves_just_over_tolerance_step():
    """Float32 subtraction must not bridge a step the scalar proof rejects."""
    smin = np.zeros((3, 1, 1), dtype=np.float32)
    smax = np.ones((3, 1, 1), dtype=np.float32)
    count = np.ones((3, 1), dtype=np.int32)
    count[1, 0] = 0
    smin[0, 0, 0] = np.float32(1.03e-9)
    smin[2, 0, 0] = np.float32(0.0200000014)
    spans = {
        "nx": 3,
        "ny": 1,
        "max_spans": 1,
        "cell_size": 0.1,
        "span_min": smin,
        "span_max": smax,
        "span_count": count,
        "solid_count": np.zeros((3, 1), dtype=np.int32),
    }

    bridged = fill_empty_columns(
        spans, 0.2, 0.2, 0.2, bottom_tolerance=0.02
    )

    assert int(bridged["span_count"][1, 0]) == 0


def test_gapfill_multilevel_candidate_prefilter_matches_full_scan(monkeypatch):
    """The rack prefilter may prune work, never a valid upper-deck addition."""
    spans = _layered_gap_spans()
    spans["span_min"][1, 0, 0] = 0.0
    spans["span_max"][1, 0, 0] = 2.0
    spans["span_count"][1, 0] = 1

    filtered = fill_empty_columns(spans, 0.2, 0.2, 0.2, bottom_tolerance=0.02)
    monkeypatch.setattr(
        gapfill,
        "_candidate_cells",
        lambda source, **_kw: np.ones(source["span_count"].shape, dtype=bool),
    )
    full_scan = fill_empty_columns(spans, 0.2, 0.2, 0.2, bottom_tolerance=0.02)

    np.testing.assert_array_equal(filtered["span_count"], full_scan["span_count"])
    np.testing.assert_allclose(filtered["span_min"], full_scan["span_min"])
    np.testing.assert_allclose(filtered["span_max"], full_scan["span_max"])
    assert filtered.get("gapfill_overflow_cells", 0) == full_scan.get(
        "gapfill_overflow_cells", 0
    )


def test_gapfill_prefilter_keeps_distinct_floor_beyond_tolerance():
    """A level farther than one tolerance from measured support must reach the proof."""
    smin = np.zeros((3, 1, 2), dtype=np.float32)
    smax = np.full((3, 1, 2), 2.0, dtype=np.float32)
    count = np.ones((3, 1), dtype=np.int32)
    # The centre's measured bay contains the raised deck but its floor is distinct.
    # `2 * tolerance` pruning used to call 0.03 "known" and skip the valid addition.
    smin[0, 0, 0] = smin[2, 0, 0] = 0.03
    bridged = fill_empty_columns(
        {
            "nx": 3, "ny": 1, "max_spans": 2, "cell_size": 0.1,
            "span_min": smin, "span_max": smax, "span_count": count,
            "solid_count": np.zeros((3, 1), dtype=np.int32),
        },
        0.2, 0.2, 0.2, bottom_tolerance=0.02,
    )
    assert int(bridged["span_count"][1, 0]) == 2
    assert tuple(bridged["span_min"][1, 0, :2]) == pytest.approx((0.0, 0.03))


def test_warp_region_labels_match_cpu_for_multilevel_gap():
    """Warp labels preserve both layers and the disconnected upper deck on CI CPU."""
    spans = _layered_gap_spans()
    cpu = detect_regions(spans, 0.2, bottom_tolerance=0.02, top_tolerance=0.02)
    warp_regions = detect_regions_warp(
        spans, 0.2, bottom_tolerance=0.02, top_tolerance=0.02, device="cpu"
    )

    def canonical(regions):
        return sorted(
            tuple(sorted((x, y, round(b, 6), round(t, 6)) for x, y, b, t in region))
            for region in regions
        )

    assert canonical(warp_regions) == canonical(cpu)


def test_warp_region_labels_converge_on_serpentine_component():
    """A non-convex component needs its graph diameter, not grid extent, in passes."""
    n = 9
    smin = np.zeros((n, n, 1), dtype=np.float32)
    smax = np.ones((n, n, 1), dtype=np.float32)
    count = np.zeros((n, n), dtype=np.int32)
    # Horizontal runs connect at alternating ends through one-cell vertical links.
    # Its 4-neighbour geodesic diameter exceeds ``nx + ny - 2`` despite the tiny grid.
    for iy in range(0, n, 2):
        count[:, iy] = 1
        if iy + 1 < n:
            count[n - 1 if (iy // 2) % 2 == 0 else 0, iy + 1] = 1
    spans = {
        "nx": n, "ny": n, "max_spans": 1, "span_min": smin,
        "span_max": smax, "span_count": count,
    }
    cpu = detect_regions(spans, 0.2, bottom_tolerance=0.02, top_tolerance=0.02)
    warp_regions = detect_regions_warp(
        spans, 0.2, bottom_tolerance=0.02, top_tolerance=0.02, device="cpu"
    )
    assert len(cpu) == len(warp_regions) == 1
    assert sorted(cpu[0]) == sorted(warp_regions[0])


def test_warp_region_labels_preserve_python_float_tolerance_boundary():
    """Warp must not round a Python tolerance up to a matching float32 endpoint."""
    upper = np.nextafter(np.float32(0.1), np.float32(np.inf))
    spans = {
        "nx": 2, "ny": 1, "max_spans": 1,
        "span_min": np.array([[[0.0]], [[upper]]], dtype=np.float32),
        "span_max": np.array([[[1.0]], [[1.0 + upper]]], dtype=np.float32),
        "span_count": np.ones((2, 1), dtype=np.int32),
    }
    cpu = detect_regions(spans, 0.2, bottom_tolerance=0.1, top_tolerance=0.1)
    warp_regions = detect_regions_warp(
        spans, 0.2, bottom_tolerance=0.1, top_tolerance=0.1, device="cpu"
    )
    assert len(cpu) == len(warp_regions) == 2


def test_warp_active_mask_preserves_float64_height_threshold_boundary():
    """The compact CUDA launch mask must accept every CPU-eligible stored span."""
    spans = {
        "nx": 1,
        "ny": 1,
        "max_spans": 1,
        "span_min": np.array([[[1.543e-6]]], dtype=np.float32),
        "span_max": np.array([[[0.300000548]]], dtype=np.float32),
        "span_count": np.ones((1, 1), dtype=np.int32),
    }
    cpu = detect_regions(spans, 0.3)
    warp_regions = detect_regions_warp(spans, 0.3, device="cpu")
    assert warp_regions == cpu
    assert len(warp_regions) == 1


def test_warp_label_large_grid_falls_back_to_cpu_reference():
    """Fine grids above the launch bound remain correct, rather than launch-storming."""
    n = MAX_WARP_LABEL_PASSES // 2 + 2
    spans = {
        "nx": n, "ny": n, "max_spans": 1,
        "span_min": np.zeros((n, n, 1), dtype=np.float32),
        "span_max": np.ones((n, n, 1), dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
    }
    assert detect_regions_warp(spans, 0.2, device="cpu") == detect_regions(spans, 0.2)


def test_warp_label_work_preflight_falls_back_to_cpu(monkeypatch):
    """A dense query above the pass-work budget must skip Warp launches entirely."""
    spans = {
        "nx": 3, "ny": 3, "max_spans": 1,
        "span_min": np.zeros((3, 3, 1), dtype=np.float32),
        "span_max": np.ones((3, 3, 1), dtype=np.float32),
        "span_count": np.ones((3, 3), dtype=np.int32),
    }
    monkeypatch.setattr(extract_module, "MAX_WARP_LABEL_WORK", 1)
    assert detect_regions_warp(spans, 0.2, device="cpu") == detect_regions(spans, 0.2)


@pytest.mark.parametrize("fallback", ["preflight", "passes"])
def test_warp_region_fallback_keeps_requested_span_slots(monkeypatch, fallback):
    """``return_slots`` has one five-member contract on Warp and CPU fallback."""
    spans = {
        "nx": 2, "ny": 1, "max_spans": 2,
        "span_min": np.array([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32),
        "span_max": np.array([[[0.5, 1.5]], [[0.5, 1.5]]], dtype=np.float32),
        "span_count": np.full((2, 1), 2, dtype=np.int32),
    }
    if fallback == "preflight":
        monkeypatch.setattr(extract_module, "MAX_WARP_LABEL_WORK", 1)
    else:
        monkeypatch.setattr(extract_module, "MAX_WARP_LABEL_PASSES", 0)

    regions, labels = detect_regions_warp(
        spans, 0.2, device="cpu", return_labels=True, return_slots=True
    )

    assert labels is None
    assert {member[4] for region in regions for member in region} == {0, 1}
    assert all(len(member) == 5 for region in regions for member in region)
    slots_only = detect_regions_warp(spans, 0.2, device="cpu", return_slots=True)
    assert {member[4] for region in slots_only for member in region} == {0, 1}
    assert all(len(member) == 5 for region in slots_only for member in region)


@pytest.mark.parametrize("device", [None, "cuda:0"])
def test_erosion_keeps_requested_span_slots(monkeypatch, device):
    """Opening must preserve the extractor's optional source-slot contract."""
    spans = {
        "nx": 3, "ny": 1, "max_spans": 2,
        "span_min": np.array([[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32),
        "span_max": np.array([[[0.5, 1.5]], [[0.5, 1.5]], [[0.5, 1.5]]], dtype=np.float32),
        "span_count": np.full((3, 1), 2, dtype=np.int32),
    }
    if device is not None:
        monkeypatch.setattr(
            extract_module,
            "detect_regions_warp",
            lambda source, *_args, **_kwargs: detect_regions(source, 0.2, return_slots=True),
        )

    regions = detect_regions(
        spans, 0.2, erosion=(1, 0), device=device, return_slots=True
    )

    assert {member[4] for region in regions for member in region} == {0, 1}
    assert all(len(member) == 5 for region in regions for member in region)


@pytest.mark.parametrize("device", [None, "cuda:0"])
def test_erosion_default_result_keeps_four_field_one_cell_contract(monkeypatch, device):
    """Opening must not expose internal slots or duplicate a cell by its layers."""
    spans = {
        "nx": 3, "ny": 1, "max_spans": 2,
        "span_min": np.array([[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32),
        "span_max": np.array([[[0.5, 1.5]], [[0.5, 1.5]], [[0.5, 1.5]]], dtype=np.float32),
        "span_count": np.full((3, 1), 2, dtype=np.int32),
    }
    if device is not None:
        monkeypatch.setattr(
            extract_module,
            "detect_regions_warp",
            lambda source, *_args, **_kwargs: detect_regions(source, 0.2, return_slots=True),
        )

    regions = detect_regions(spans, 0.2, erosion=(1, 0), device=device)

    assert all(len(member) == 4 for region in regions for member in region)
    assert all(
        len({(member[0], member[1]) for member in region}) == len(region)
        for region in regions
    )


def test_gapfill_candidate_prefilter_keeps_float64_clearance_boundary():
    """A float64-valid bridge must reach the scalar proof, not be prefiltered out."""
    # These stored endpoints differ by 0.7621355187147856 in Python float but by
    # 0.7621355056762695 after a float32 subtraction.  The midpoint must be
    # accepted by the scalar proof, and used to be pruned by this prefilter.
    floor = np.float32(0.01590554602444172)
    ceiling = np.float32(0.7780410647392273)
    min_height = 0.7621355121955276
    assert float(ceiling) - float(floor) >= min_height
    assert float(np.float32(ceiling - floor)) < min_height
    smin = np.zeros((3, 1, 2), dtype=np.float32)
    smax = np.zeros((3, 1, 2), dtype=np.float32)
    count = np.array([[2], [0], [2]], dtype=np.int32)
    smin[0, 0, 0] = smin[2, 0, 0] = floor
    smax[0, 0, 0] = smax[2, 0, 0] = ceiling
    spans = {
        "nx": 3, "ny": 1, "max_spans": 2, "cell_size": 0.1,
        "span_min": smin, "span_max": smax, "span_count": count,
        "solid_count": np.zeros((3, 1), dtype=np.int32),
    }
    bridged = fill_empty_columns(spans, 0.2, 0.2, min_height)
    assert int(bridged["span_count"][1, 0]) == 1


def test_warp_label_budget_charges_neighbour_span_cross_product(monkeypatch):
    """A multi-level rack falls back before its per-neighbour slot scans launch."""
    spans = {
        "nx": 3, "ny": 3, "max_spans": 4,
        "span_min": np.zeros((3, 3, 4), dtype=np.float32),
        "span_max": np.ones((3, 3, 4), dtype=np.float32),
        "span_count": np.full((3, 3), 4, dtype=np.int32),
    }
    monkeypatch.setattr(extract_module, "MAX_WARP_LABEL_WORK", 100_000)
    monkeypatch.setattr(
        extract_module.wp, "launch", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    assert detect_regions_warp(spans, 0.2, device="cpu") == detect_regions(spans, 0.2)


def test_warp_labels_compact_active_spans_not_fixed_capacity(monkeypatch):
    """A single-level query can use Warp even though the raster reserves 16 slots."""
    spans = {
        "nx": 20, "ny": 10, "max_spans": 16,
        "span_min": np.zeros((20, 10, 16), dtype=np.float32),
        "span_max": np.ones((20, 10, 16), dtype=np.float32),
        "span_count": np.ones((20, 10), dtype=np.int32),
    }
    # Fixed-capacity accounting would charge 1,641,600 visits and fall back. The
    # compact 200-node launch charges 512,200 comparisons and runs on CPU Warp CI.
    monkeypatch.setattr(extract_module, "MAX_WARP_LABEL_WORK", 1_000_000)
    regions, labels = detect_regions_warp(
        spans, 0.2, device="cpu", return_labels=True
    )
    assert labels is not None
    assert regions == detect_regions(spans, 0.2)


def test_warp_full_support_large_window_returns_cpu_fallback_signal():
    labels = np.zeros((3, 3, 1), dtype=np.int32)
    half_window = int(MAX_WARP_FULL_SUPPORT_WINDOW_CELLS**0.5) + 1
    assert full_support_centers_warp(
        labels, np.ones((3, 3), dtype=np.int32), half_window, half_window, device="cuda:0"
    ) is None


def test_warp_full_support_work_preflight_returns_cpu_fallback_signal(monkeypatch):
    """A legal window still falls back when all active spans make it too expensive."""
    labels = np.zeros((3, 3, 1), dtype=np.int32)
    monkeypatch.setattr(extract_module, "MAX_WARP_FULL_SUPPORT_WORK", 1)
    assert full_support_centers_warp(
        labels, np.ones((3, 3), dtype=np.int32), 1, 1, device="cpu"
    ) is None


def test_warp_full_support_cpu_kernel_matches_expected_centres():
    """CPU CI executes the default tau=1 same-label footprint kernel itself."""
    labels = np.zeros((5, 5, 1), dtype=np.int32)
    centres = full_support_centers_warp(
        labels, np.ones((5, 5), dtype=np.int32), 1, 1, device="cpu"
    )
    expected = np.zeros((5, 5, 1), dtype=bool)
    expected[1:4, 1:4, 0] = True
    np.testing.assert_array_equal(centres, expected)


def test_warp_full_support_cpu_kernel_keeps_span_levels_separate():
    """A lower level cannot certify an upper span at the same grid cell."""
    labels = np.zeros((5, 5, 2), dtype=np.int32)
    labels[:, :, 1] = 1
    labels[2, 2, 1] = -1
    centres = full_support_centers_warp(
        labels, np.full((5, 5), 2, dtype=np.int32), 1, 1, device="cpu"
    )
    assert centres is not None
    assert centres[2, 2, 0]
    assert not centres[2, 2, 1]


def test_gapfill_reports_overflow_when_no_synthetic_span_can_be_added():
    """A full measured column must retain its overflow warning after rejecting a level."""
    smin = np.zeros((3, 1, 1), dtype=np.float32)
    smax = np.zeros((3, 1, 1), dtype=np.float32)
    count = np.ones((3, 1), dtype=np.int32)
    # The centre's measured lower band contains the virtual upper deck, but its one
    # free-span slot is already occupied. Both neighbours prove the upper deck.
    smin[:, 0, 0] = (1.1, 0.0, 1.1)
    smax[:, 0, 0] = (2.0, 2.0, 2.0)
    bridged = fill_empty_columns(
        {
            "nx": 3,
            "ny": 1,
            "max_spans": 1,
            "cell_size": 0.1,
            "span_min": smin,
            "span_max": smax,
            "span_count": count,
            "solid_count": np.zeros((3, 1), dtype=np.int32),
        },
        0.2,
        0.2,
        0.2,
        bottom_tolerance=0.02,
    )
    assert int(bridged["span_count"][1, 0]) == 1
    assert int(bridged["gapfill_overflow_cells"]) == 1


def test_gapfill_large_footprint_uses_linear_nearest_support_sweeps():
    """A 400-cell half-footprint must not turn an 800x800 grid into billion probes."""
    shape = (800, 800, 1)
    empty = {
        "nx": 800,
        "ny": 800,
        "max_spans": 1,
        "cell_size": 0.1,
        "span_min": np.zeros(shape, dtype=np.float32),
        "span_max": np.zeros(shape, dtype=np.float32),
        "span_count": np.zeros(shape[:2], dtype=np.int32),
        "solid_count": np.zeros(shape[:2], dtype=np.int32),
    }
    # No support exists, so the result is unchanged.  The assertion proves the request
    # reaches the linear pass rather than being rejected by the former radius product.
    assert fill_empty_columns(empty, 80.0, 80.0, 0.2) is empty


def test_gapfill_retains_a_bounded_work_budget_for_direct_callers(monkeypatch):
    """The guard is independent of footprint radius and charges level-pair work."""
    monkeypatch.setattr(gapfill, "MAX_GAPFILL_WORK", 14)
    with pytest.raises(SpaceOutOfRange, match="gap bridging would perform"):
        fill_empty_columns(_layered_gap_spans(), 0.2, 0.2, 0.2)


def test_gapfill_budget_charges_actual_multilevel_candidates(monkeypatch):
    """Only selected deep-level candidates consume the exact-proof budget."""
    spans = {
        "nx": 3, "ny": 1, "max_spans": 4, "cell_size": 0.1,
        "span_min": np.zeros((3, 1, 4), dtype=np.float32),
        "span_max": np.ones((3, 1, 4), dtype=np.float32),
        "span_count": np.array([[4], [0], [4]], dtype=np.int32),
        "solid_count": np.zeros((3, 1), dtype=np.int32),
    }
    # The compatibility prefilter costs 75 and is allowed. Keep the rejection branch
    # covered separately from its witness-count arithmetic, which is asserted below.
    monkeypatch.setattr(gapfill, "MAX_GAPFILL_WORK", 100)
    original_bound = gapfill._exact_proof_work

    def over_budget(*args, **kwargs):
        assert original_bound(*args, **kwargs) == 28
        return 101

    monkeypatch.setattr(gapfill, "_exact_proof_work", over_budget)
    with pytest.raises(SpaceOutOfRange, match="exact candidate operations"):
        fill_empty_columns(spans, 0.2, 0.2, 0.2)


def test_gapfill_exact_budget_charges_slot_walk_and_double_existing_scan(monkeypatch):
    """A deep non-open slat gap cannot bypass the direct-call daemon guard."""
    levels = 8
    spans = {
        "nx": 3, "ny": 1, "max_spans": levels, "cell_size": 0.1,
        "span_min": np.zeros((3, 1, levels), dtype=np.float32),
        "span_max": np.ones((3, 1, levels), dtype=np.float32),
        "span_count": np.full((3, 1), levels, dtype=np.int32),
        "solid_count": np.zeros((3, 1), dtype=np.int32),
    }
    # The centre is a non-open slat gap with a deep measured lower stack. Its floors
    # differ from both witnesses, so the prefilter selects it for an exact proof.
    spans["span_min"][0, 0] = np.arange(levels, dtype=np.float32)
    spans["span_min"][2, 0] = np.arange(levels, dtype=np.float32)
    spans["span_min"][1, 0] = 10.0 + np.arange(levels, dtype=np.float32)

    # Prefilter work is 3 * (6*8 + 1) = 147. The old exact estimate was 144;
    # the full proof bound is 8 slots + 8 originals + 16 pair checks + 64 merge
    # comparisons + 2*8*8 existing-span scans = 224.
    monkeypatch.setattr(gapfill, "MAX_GAPFILL_WORK", 200)
    with pytest.raises(SpaceOutOfRange, match="224 exact candidate operations"):
        fill_empty_columns(spans, 0.2, 0.2, 0.2)


def test_gapfill_sparse_deep_column_does_not_charge_the_whole_scope(monkeypatch):
    """One deep rack bay cannot reject a broad otherwise single-level query."""
    n = 50
    spans = {
        "nx": n, "ny": n, "max_spans": 4, "cell_size": 0.1,
        "span_min": np.zeros((n, n, 4), dtype=np.float32),
        "span_max": np.ones((n, n, 4), dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.zeros((n, n), dtype=np.int32),
    }
    spans["span_count"][0, 0] = 4  # remote deep bay, not a bridge witness
    # The previous global m² charge was 522,500. The linear prefilter is 62,500 and
    # produces no exact candidates, so a normal object-relative scope remains valid.
    monkeypatch.setattr(gapfill, "MAX_GAPFILL_WORK", 100_000)
    assert fill_empty_columns(spans, 0.2, 0.2, 0.2) is spans


def test_gapfill_prefilter_keeps_the_former_multilevel_budget_envelope(monkeypatch):
    """Vectorised prefilter accounting must not reject a formerly admitted grid."""
    spans = {
        "nx": 4, "ny": 1, "max_spans": 4, "cell_size": 0.1,
        "span_min": np.zeros((4, 1, 4), dtype=np.float32),
        "span_max": np.ones((4, 1, 4), dtype=np.float32),
        "span_count": np.ones((4, 1), dtype=np.int32),
        "solid_count": np.zeros((4, 1), dtype=np.int32),
    }
    spans["span_count"][0, 0] = 4  # a remote deep column, not a bridge witness
    # The pre-acceleration guard admitted 4 * (6 * 4 + 1) = 100 units.  The
    # accelerated NumPy prefilter must preserve that envelope rather than charging
    # its internal map operations as Python proof work.
    monkeypatch.setattr(gapfill, "MAX_GAPFILL_WORK", 100)
    assert fill_empty_columns(spans, 0.2, 0.2, 0.2) is spans


def test_multi_level_slatted_decks_keep_the_upper_support_region():
    """The public pipeline must bridge each level, not only the lowest deck."""
    from usd_core.space.support import detect_support_regions

    result = detect_support_regions(
        _stacked_slatted_decks(gap=0.06),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        CELL,
        (0.2, 0.2, 0.1),
        smooth=True,
        device="cpu",
    )
    levels = {round(region.support_z, 2) for region in result.regions}
    assert 0.3 in levels
    assert 1.1 in levels, levels


def test_continuous_lower_floor_does_not_hide_upper_slatted_deck():
    """The real rack topology keeps the upper support region connected across gaps."""
    from usd_core.space.support import detect_support_regions

    result = detect_support_regions(
        _lower_floor_with_upper_slatted_deck(gap=0.06),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
        CELL,
        (0.2, 0.2, 0.1),
        smooth=True,
        device="cpu",
    )
    upper = [region for region in result.regions if abs(region.support_z - 1.1) < 1e-3]
    assert len(upper) == 1, (
        f"upper slatted deck fragmented into {len(upper)} support regions"
    )


def test_empty_space_is_not_a_support_surface():
    """Empty columns must NOT become a resting surface at the scope floor.

    This is the invariant that rules out "just emit a full-height span for every empty
    column". A lone pad surrounded by nothing must yield support on the pad only —
    never a scope-sized region hanging at the bottom of the window.
    """
    obj = (0.4, 0.4, 0.4)
    pad = scene_from_boxes([([-0.15, -0.15, -0.5], [0.15, 0.15, 0.0])], names=["pad"])
    spans = build_spans_raster(
        DeviceMesh(pad, device="cpu"), (-0.6, -0.6, -0.05), (0.6, 0.6, 2.0), 0.1, obj[2]
    )
    iy = spans["ny"] // 2
    floors = [
        float(spans["span_min"][ix, iy, 0])
        if int(spans["span_count"][ix, iy])
        else None
        for ix in range(spans["nx"])
    ]
    assert any(f is None for f in floors), (
        f"empty space was given a floor at the scope bottom: {floors}"
    )
    assert any(f is not None and abs(f) < 1e-6 for f in floors), (
        "the pad itself must support"
    )

    bridged = fill_empty_columns(spans, obj[0], obj[1], obj[2])
    after = [
        float(bridged["span_min"][ix, iy, 0])
        if int(bridged["span_count"][ix, iy])
        else None
        for ix in range(bridged["nx"])
    ]
    assert any(f is None for f in after), (
        f"bridging invented support in open space around an isolated pad: {after}"
    )


def test_gap_narrower_than_object_is_bridged():
    """A 60 mm gap under a 200 mm object: one surface, at the slat height."""
    obj = (0.2, 0.2, 0.1)
    scene = _slatted_deck(gap=0.06)
    ix = (
        build_spans_raster(
            DeviceMesh(scene, device="cpu"), SCOPE_MIN, SCOPE_MAX, CELL, obj[2]
        )["nx"]
        // 2
    )
    before = _floors(_spans(scene, obj, smooth=False), ix)
    after = _floors(_spans(scene, obj), ix)
    assert any(f is None for f in before), "gap columns should start with no span"
    for value in after:
        assert value == pytest.approx(DECK_TOP, abs=1e-3), (
            f"smoothing left a column at {value} instead of the slat top {DECK_TOP}"
        )


def test_bridged_deck_is_one_region():
    """The user-visible symptom: a slatted deck must not fragment into strips."""
    obj = (0.2, 0.2, 0.1)
    spans = _spans(_slatted_deck(gap=0.06), obj)
    regions = detect_regions(
        spans, obj[2], bottom_tolerance=0.1, top_tolerance=0.2, erosion=None
    )
    at_deck = [
        r
        for r in regions
        if abs(
            float(spans["span_min"][next(iter(r))[0], next(iter(r))[1], 0]) - DECK_TOP
        )
        < 1e-3
    ]
    assert len(at_deck) == 1, (
        f"the deck fragmented into {len(at_deck)} regions; it is one continuous surface"
    )


def test_gap_wider_than_object_is_not_bridged():
    """An object cannot span a gap wider than itself — that would invent support."""
    obj = (0.06, 0.06, 0.1)  # 60 mm object
    spans = _spans(_slatted_deck(gap=0.20), obj)  # 200 mm gap
    ix = spans["nx"] // 2
    assert any(f is None for f in _floors(spans, ix)), (
        "a 60 mm object was floated across a 200 mm gap"
    )


def test_deck_edge_is_not_extended_over_the_void():
    """Support on one side only is an edge, not a gap: the both-sides rule must hold."""
    obj = (0.2, 0.2, 0.1)
    scene = scene_from_boxes([((0.0, 0.0, DECK_TOP - 0.02), (1.0, 0.5, DECK_TOP))])
    spans = _spans(scene, obj)
    ix = spans["nx"] // 2
    assert int(spans["span_count"][ix, spans["ny"] - 1]) == 0, (
        "open air past the deck edge was given support — an object would float"
    )


def test_solid_column_gets_no_free_span():
    """A column solid to the ceiling must stay unsupported — nothing may rest inside it."""
    obj = (0.3, 0.3, 0.1)
    scene = scene_from_boxes(
        [
            ((0.0, 0.00, DECK_TOP - 0.02), (1.0, 1.00, DECK_TOP)),  # continuous deck
            ((0.0, 0.49, -0.05), (1.0, 0.51, SCOPE_MAX[2] + 0.1)),  # wall through it
        ]
    )
    spans = _spans(scene, obj, smooth=False)
    ix = spans["nx"] // 2
    solid = [iy for iy in range(spans["ny"]) if int(spans["span_count"][ix, iy]) == 0]
    assert solid, "the wall's own columns should be fully solid"


def test_detect_support_regions_bridges_without_being_asked():
    """The wiring, not just the helper.

    Every test above calls `fill_empty_columns` itself, so they would all still pass if
    `detect_support_regions` stopped calling it. This one goes through the public entry
    point: a slatted deck must come back as ONE support region.
    """
    from usd_core.space.support import detect_support_regions

    obj = (0.2, 0.2, 0.1)
    result = detect_support_regions(
        _slatted_deck(gap=0.06), SCOPE_MIN, SCOPE_MAX, CELL, obj, smooth=True
    )
    at_deck = [r for r in result.regions if abs(r.support_z - DECK_TOP) < 1e-3]
    assert len(at_deck) == 1, (
        f"the deck came back as {len(at_deck)} support regions; a slatted deck an "
        "object can span is one continuous surface"
    )
