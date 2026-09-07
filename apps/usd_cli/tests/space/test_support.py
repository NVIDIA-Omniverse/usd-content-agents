# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable-support region detection (esd.support.detect_support_regions).

Exercises the two stability gates that moved into ESD from the placement layer:
  * feature 1 (support_margin) — a morphological opening that deletes thin strips
    (tightrope / perch) but keeps broad surfaces dense;
  * feature 2 (min_support_area[_ratio]) — rejects supports too small vs the footprint
    (point-balance), meaningful on the tau<1 overhang path;
plus the tau full-vs-partial semantics and argument validation. Scenes are synthetic
AABB boxes with known support surfaces (empty columns emit no free span, so support
only ever rests on real geometry).
"""


import numpy as np
import pytest


wp = pytest.importorskip("warp")
from usd_core.space.errors import SpaceInvalidArgument, SpaceOutOfRange  # noqa: E402
from usd_core.space.extract import detect_regions  # noqa: E402
from usd_core.space.geometry import SceneGeometry  # noqa: E402
from usd_core.space import support as support_module  # noqa: E402
from usd_core.space.support import detect_support_regions  # noqa: E402
from space_synthetic import box_mesh, scene_from_boxes  # noqa: E402

CS = 0.1
OBJ = (0.4, 0.4, 0.3)


def _canonical_regions(result):
    """Device-independent support-region identity for CPU/Warp parity assertions."""
    return sorted(
        (tuple(sorted(region.cells)), round(region.support_z, 6), round(region.ceiling_z, 6))
        for region in result.regions
    )


def _flat_floor():
    # A 2x2 m slab whose top is z=0 — a broad support surface.
    return scene_from_boxes([([-1.0, -1.0, -0.5], [1.0, 1.0, 0.0])], names=["floor"])


def _pad(half=0.15):
    # A single small pad, top z=0, footprint (2*half) square — the only support.
    return scene_from_boxes([([-half, -half, -0.5], [half, half, 0.0])], names=["pad"])


def _strip():
    # A thin support strip: 1 cell wide in x (0.1 m), 1.0 m long in y, top z=0.
    return scene_from_boxes([([-0.05, -0.5, -0.5], [0.05, 0.5, 0.0])], names=["strip"])


def _stacked_floors():
    """Two broad support surfaces whose free spans occupy distinct slots."""
    return scene_from_boxes(
        [([-1.0, -1.0, -0.5], [1.0, 1.0, 0.0]),
         ([-1.0, -1.0, 1.0], [1.0, 1.0, 1.1])],
        names=["lower", "upper"],
    )


def _pad_with_thin_tail():
    """A small square pad whose raw area is inflated by a narrow tail.

    A margin opening removes the tail but preserves the broad pad.  This isolates the
    contract that ``min_support_area`` applies to the post-opening support surface.
    """
    return scene_from_boxes(
        [
            ([-0.15, -0.15, -0.5], [0.15, 0.15, 0.0]),
            ([0.15, -0.05, -0.5], [0.65, 0.05, 0.0]),
        ],
        names=["pad", "tail"],
    )


def _scope(x0, y0, x1, y1, ztop=2.0):
    # z-min just below the support top so only the free span ABOVE it qualifies.
    return (x0, y0, -0.05), (x1, y1, ztop)


def _run(scene, scope, **kw):
    smin, smax = scope
    kw.setdefault("smooth", False)  # deterministic unit behaviour (no gap-bridging)
    return detect_support_regions(scene, smin, smax, CS, OBJ, **kw)


# --- basic / full support -----------------------------------------------------------

def test_full_support_flat_floor_defaults():
    res = _run(_flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0))
    assert res.regions, "a broad flat floor must yield a full-support region"
    assert abs(res.regions[0].support_z - 0.0) < 1e-6
    # Deterministic: identical call -> identical region cell-sets.
    res2 = _run(_flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0))
    assert [r.cells for r in res.regions] == [r.cells for r in res2.regions]


def test_cuda_full_support_matches_cpu_flat_floor():
    """Phase-1 same-label erosion keeps the CPU full-footprint centres exactly."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene, scope = _flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0)
    cpu = _run(scene, scope, device="cpu")
    cuda = _run(scene, scope, device="cuda:0")
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def test_cuda_full_support_matches_cpu_for_upper_span_slot():
    """Vectorised CUDA-centre recovery retains an upper support level, not slot zero."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene, scope = _stacked_floors(), _scope(-1.0, -1.0, 1.0, 1.0, ztop=2.0)
    cpu = _run(scene, scope, device="cpu")
    cuda = _run(scene, scope, device="cuda:0")
    assert len(cpu.regions) == 2
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def test_cuda_host_mapping_uses_exact_slot_for_nearby_levels_without_cuda(monkeypatch):
    """The CUDA serializer must not re-identify a level by tolerant float matching.

    This exercises the host-side handoff with ``device='cuda:0'`` but replaces the
    device operations with deterministic arrays.  It therefore runs in ordinary CPU
    CI while reproducing the formerly ambiguous case: two different span slots have
    floor/ceiling values within the old 1e-6 re-lookup tolerance.
    """
    n, slots = 3, 2
    span_min = np.zeros((n, n, slots), dtype=np.float32)
    span_max = np.ones((n, n, slots), dtype=np.float32)
    span_min[:, :, 1] = 4.0e-7
    span_max[:, :, 1] += 4.0e-7
    spans = {
        "origin": (0.0, 0.0), "cell_size": CS, "nx": n, "ny": n,
        "max_spans": slots, "span_min": span_min, "span_max": span_max,
        "span_count": np.full((n, n), slots, dtype=np.int32),
        "solid_count": np.zeros((n, n), dtype=np.int32),
    }
    labels = np.empty((n, n, slots), dtype=np.int32)
    labels[:, :, 0] = 10
    labels[:, :, 1] = 20
    upper_region = [
        (ix, iy, float(span_min[ix, iy, 1]), float(span_max[ix, iy, 1]), 1)
        for ix in range(n) for iy in range(n)
    ]
    centres = np.zeros((n, n, slots), dtype=bool)
    centres[1, 1, 1] = True

    monkeypatch.setattr(support_module, "DeviceMesh", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(support_module, "build_spans_raster", lambda *_args, **_kwargs: spans)

    def fake_regions(*_args, **kwargs):
        assert kwargs["return_labels"] and kwargs["return_slots"]
        return [upper_region], labels

    monkeypatch.setattr(support_module, "detect_regions_warp", fake_regions)
    monkeypatch.setattr(support_module, "full_support_centers_warp",
                        lambda *_args, **_kwargs: centres)

    result = detect_support_regions(
        object(), (0.0, 0.0, 0.0), (0.3, 0.3, 1.2), CS, (0.1, 0.1, 0.3),
        smooth=False, device="cuda:0",
    )
    assert len(result.regions) == 1
    assert result.regions[0].cells == [(1, 1)]
    assert result.regions[0].support_z == pytest.approx(4.0e-7)


def test_cuda_partial_support_falls_back_to_cpu_oracle_without_cuda(monkeypatch):
    """A dense-work cap flips a CUDA tau<1 query to the CPU oracle mid-query.

    The fallback decision is host code; replacing only the GPU ingress lets CPU CI
    prove that a partial-support request remains correct without a CUDA machine.
    """
    n = 3
    spans = {
        "origin": (0.0, 0.0), "cell_size": CS, "nx": n, "ny": n,
        "max_spans": 1,
        "span_min": np.zeros((n, n, 1), dtype=np.float32),
        "span_max": np.ones((n, n, 1), dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.zeros((n, n), dtype=np.int32),
    }
    region = [(ix, iy, 0.0, 1.0, 0) for ix in range(n) for iy in range(n)]
    labels = np.zeros((n, n, 1), dtype=np.int32)

    monkeypatch.setattr(support_module, "DeviceMesh", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(support_module, "build_spans_raster", lambda *_args, **_kwargs: spans)
    monkeypatch.setattr(support_module, "detect_regions_warp",
                        lambda *_args, **_kwargs: ([region], labels))
    monkeypatch.setattr(support_module, "MAX_PARTIAL_SUPPORT_DENSE_WORK", 1)
    monkeypatch.setattr(
        support_module, "_partial_support_centers_warp",
        lambda *_args, **_kwargs: pytest.fail("dense-work cap did not disable CUDA path"),
    )
    calls = []

    def cpu_oracle(*_args, **_kwargs):
        calls.append(True)
        return {(1, 1): (0.0, 1.0)}

    monkeypatch.setattr(support_module, "_partial_support_centers", cpu_oracle)
    result = detect_support_regions(
        object(), (0.0, 0.0, 0.0), (0.3, 0.3, 1.2), CS, (0.1, 0.1, 0.3),
        tau=0.5, smooth=False, device="cuda:0",
    )
    assert calls == [True]
    assert len(result.regions) == 1
    assert result.regions[0].cells == [(1, 1)]


def test_full_support_needs_full_footprint():
    # A 0.3 m pad is smaller than the 0.4 m footprint -> tau=1 (full support) finds
    # NO centre; tau<1 (overhang) does.
    scene, scope = _pad(0.15), _scope(-0.6, -0.6, 0.6, 0.6)
    assert _run(scene, scope, tau=1.0).regions == []
    assert _run(scene, scope, tau=0.3).regions, "overhang must allow the smaller support"


def test_cuda_partial_support_matches_cpu_pad(monkeypatch):
    """Phase 4 preserves the oracle result and dispatches the CUDA kernel."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene, scope = _pad(0.15), _scope(-0.6, -0.6, 0.6, 0.6)
    cpu = _run(scene, scope, tau=0.3, device="cpu")

    def cpu_oracle_must_not_run(*_args, **_kwargs):
        raise AssertionError("CUDA tau<1 query fell back to the Python overhang oracle")

    monkeypatch.setattr(support_module, "_partial_support_centers", cpu_oracle_must_not_run)
    cuda = _run(scene, scope, tau=0.3, device="cuda:0")
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def test_cuda_partial_support_matches_cpu_margin_gate():
    """The CUDA CoG bracket retains the absolute support-margin semantics."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene, scope = _flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0)
    cpu = _run(scene, scope, tau=0.5, support_margin=0.1, device="cpu")
    cuda = _run(scene, scope, tau=0.5, support_margin=0.1, device="cuda:0")
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def test_partial_support_warp_reuses_query_span_snapshot(monkeypatch):
    """Per-region adapters consume a supplied immutable device snapshot unchanged."""
    n = 3
    spans = {
        "nx": n,
        "ny": n,
        "max_spans": 1,
        "span_min": np.zeros((n, n, 1), dtype=np.float32),
        "span_max": np.full((n, n, 1), 2.0, dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.zeros((n, n), dtype=np.int32),
    }
    snapshot = support_module._partial_support_device_spans(spans, device="cpu")
    monkeypatch.setattr(
        support_module,
        "_partial_support_device_spans",
        lambda *_args, **_kwargs: pytest.fail("adapter re-uploaded invariant spans"),
    )
    expected = {(1, 1): (0.0, 2.0)}
    for _ in range(2):
        assert support_module._partial_support_centers_warp(
            {(1, 1)}, {(1, 1): (0.0, 2.0)}, spans, 0, 0, 1.0, 0, 0.3, {(1, 1)},
            device="cpu", device_spans=snapshot,
        ) == expected


def test_partial_support_warp_preserves_tau_boundary_precision():
    """A tau just above 3/9 must reject rather than round down on the Warp path."""
    n = 3
    spans = {
        "nx": n, "ny": n, "max_spans": 1,
        "span_min": np.zeros((n, n, 1), dtype=np.float32),
        "span_max": np.full((n, n, 1), 2.0, dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.zeros((n, n), dtype=np.int32),
    }
    support_cells = {(0, 1), (1, 1), (2, 1)}
    bt = {cell: (0.0, 2.0) for cell in support_cells}
    assert support_module._partial_support_centers_warp(
        support_cells, bt, spans, 1, 1, 1.0 / 3.0 + 5.0e-9, 0, 0.3, {(1, 1)},
        device="cpu",
    ) == {}


def test_partial_support_warp_preserves_large_coordinate_clearance():
    """Float32 addition must not round the required overhang clearance downward."""
    floor = 1_000_000.0
    n = 3
    spans = {
        "nx": n, "ny": n, "max_spans": 1,
        "span_min": np.full((n, n, 1), floor, dtype=np.float32),
        "span_max": np.full((n, n, 1), floor + 0.25, dtype=np.float32),
        "span_count": np.ones((n, n), dtype=np.int32),
        "solid_count": np.ones((n, n), dtype=np.int32),
    }
    spans["solid_count"][1, 1] = 0
    assert support_module._partial_support_centers_warp(
        {(1, 1)}, {(1, 1): (floor, floor + 1.0)}, spans,
        1, 1, 1.0 / 9.0, 0, 0.28, {(1, 1)}, device="cpu",
    ) == {}


# --- feature 2: support-area gate (tau<1) -------------------------------------------

def test_area_ratio_gate_rejects_small_pad():
    # A 0.3 m pad rasterises to ~0.16 m^2 of support (conservative overlap); the 0.4 m
    # object footprint is 0.16 m^2. ratio off: overhang finds the pad. ratio=1.5 requires
    # 0.24 m^2 of support -> the pad is too small -> dropped.
    scene, scope = _pad(0.15), _scope(-0.6, -0.6, 0.6, 0.6)
    assert _run(scene, scope, tau=0.3, min_support_area_ratio=0.0).regions
    assert _run(scene, scope, tau=0.3, min_support_area_ratio=1.5).regions == []


def test_absolute_area_gate_rejects_small_pad():
    scene, scope = _pad(0.15), _scope(-0.6, -0.6, 0.6, 0.6)
    assert _run(scene, scope, tau=0.3, min_support_area=0.5).regions == []
    assert _run(scene, scope, tau=0.3, min_support_area=0.01).regions


# --- feature 1: distribution margin (anti-tightrope) --------------------------------

def test_margin_removes_thin_strip():
    scene, scope = _strip(), _scope(-0.6, -0.6, 0.6, 0.6)
    # Without a margin, a centre balances on the 1-cell strip (tightrope).
    assert _run(scene, scope, tau=0.15).regions
    # A margin opens the region: a 1-cell strip is narrower than 2*mc -> erased.
    assert _run(scene, scope, tau=0.15, support_margin=0.1).regions == []


def test_margin_keeps_broad_surface():
    # The same margin that erases a thin strip must NOT erase a broad floor.
    scene, scope = _flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0)
    assert _run(scene, scope, support_margin=0.2).regions, "broad surface survives opening"


def test_margin_without_area_gate_skips_component_relabeling(monkeypatch):
    """Opening alone need not relabel components: every nonempty patch qualifies."""
    scene, scope = _flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0)
    original = support_module._connected_components
    calls = 0

    def _count_components(cells):
        nonlocal calls
        calls += 1
        return original(cells)

    monkeypatch.setattr(support_module, "_connected_components", _count_components)
    assert _run(scene, scope, support_margin=0.2).regions
    # The final serializer groups output centres once.  The opening itself must not
    # add a second raw-support relabel when its area threshold is disabled.
    assert calls == 1


def test_area_floor_applies_after_margin_opening():
    """A deleted thin tail cannot make an undersized remaining pad pass the area gate."""
    scene, scope = _pad_with_thin_tail(), _scope(-0.8, -0.4, 0.8, 0.4)
    # The raw pad+tail surface exceeds 0.20 m², while the surviving 0.30 m square
    # pad does not.  The partial-support path would otherwise yield centre cells.
    assert _run(scene, scope, tau=0.3, min_support_area=0.20).regions
    assert _run(scene, scope, tau=0.3, support_margin=0.1).regions
    assert _run(
        scene, scope, tau=0.3, support_margin=0.1, min_support_area=0.20
    ).regions == []


def test_overhang_budget_accumulates_across_raw_regions(monkeypatch):
    """Several cheap pads must not bypass a query-wide Python-work budget."""
    one_pad = _pad(0.15)
    two_pads = scene_from_boxes(
        [([-0.65, -0.15, -0.5], [-0.35, 0.15, 0.0]),
         ([0.35, -0.15, -0.5], [0.65, 0.15, 0.0])],
        names=["left", "right"])
    scope = _scope(-1.0, -0.5, 1.0, 0.5)
    # One region remains within this deliberately low test budget; two independent
    # regions exceed it only when their candidate work is accumulated.
    monkeypatch.setattr(support_module, "MAX_OVERHANG_WORK", 2_000)
    assert _run(one_pad, scope, tau=0.3).regions
    with pytest.raises(SpaceOutOfRange, match="in total"):
        _run(two_pads, scope, tau=0.3)


def test_cuda_partial_support_budget_falls_back_to_cpu(monkeypatch):
    """Fragmented regions retain CPU semantics after the dense Warp budget is spent."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene, scope = _pad(0.15), _scope(-0.6, -0.6, 0.6, 0.6)
    expected = _run(scene, scope, tau=0.3, device="cpu")
    monkeypatch.setattr(support_module, "MAX_PARTIAL_SUPPORT_DENSE_WORK", 0)
    actual = _run(scene, scope, tau=0.3, device="cuda:0")
    assert _canonical_regions(actual) == _canonical_regions(expected)


# --- argument validation ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"tau": 0.0}, {"tau": 1.5}, {"support_margin": -0.1},
    {"merge_gap": -0.1}, {"merge_gap": float("nan")},
])
def test_invalid_args_raise(bad):
    scene, scope = _flat_floor(), _scope(-1.0, -1.0, 1.0, 1.0)
    with pytest.raises(SpaceInvalidArgument):
        _run(scene, scope, **bad)


def test_bad_object_size_raises():
    scene = _flat_floor()
    (smin, smax) = _scope(-1.0, -1.0, 1.0, 1.0)
    for obj in [(0.0, 0.4, 0.3), (0.4, 0.4, -1.0), (float("inf"), 0.4, 0.3)]:
        with pytest.raises(SpaceInvalidArgument):
            detect_support_regions(scene, smin, smax, CS, obj, smooth=False)


def test_dos_guard_rejects_oversized_grid():
    scene = _flat_floor()
    with pytest.raises(SpaceInvalidArgument):
        # 2 m / 1e-4 = 20000 cells/axis -> 4e8 cells > MAX_SUPPORT_CELLS.
        detect_support_regions(scene, (-1, -1, -0.05), (1, 1, 2.0), 1e-4, OBJ)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- a bay exactly as tall as the object --------------------------------------------

def _exact_bay(clearance=0.3, floor_top=0.2):
    """A deck at `floor_top` under a roof `clearance` above it — an exact fit."""
    return scene_from_boxes(
        [([-1.0, -1.0, 0.0], [1.0, 1.0, floor_top]),
         ([-1.0, -1.0, floor_top + clearance], [1.0, 1.0, floor_top + clearance + 0.2])],
        names=["deck", "roof"])


def test_a_bay_exactly_the_object_height_still_fits():
    """An exact fit is a fit. Spans are float32, so `0.5 - 0.2` lands ~3e-9 below 0.3
    and a strict `>=` in the region grouper silently dropped the whole deck — while
    both the rasteriser and the later clearance re-test allowed it, so nothing else
    in the pipeline disagreed.
    """
    scene = _exact_bay()
    scope = ((-1.0, -1.0, 0.15), (1.0, 1.0, 0.55))
    res = detect_support_regions(scene, *scope, CS, (0.4, 0.4, 0.3), smooth=False)
    assert res.regions, "a 0.30 m object was refused a bay of exactly 0.30 m"
    assert abs(res.regions[0].support_z - 0.2) < 1e-6

    # ...and the epsilon is a rounding allowance, not a licence: a bay genuinely
    # shorter than the object is still refused.
    short = _exact_bay(clearance=0.29)
    assert detect_support_regions(short, (-1.0, -1.0, 0.15), (1.0, 1.0, 0.55), CS,
                                  (0.4, 0.4, 0.3), smooth=False).regions == []


# --- morphology cost: the shapes the grid guard still permits -------------------------

def _naive_erode(cells, ex, ey):
    return {(x, y) for (x, y) in cells
            if all((x + dx, y + dy) in cells
                   for dx in range(-ex, ex + 1) for dy in range(-ey, ey + 1))}


def _naive_dilate(cells, ex, ey):
    return {(x + dx, y + dy) for (x, y) in cells
            for dx in range(-ex, ex + 1) for dy in range(-ey, ey + 1)}


def test_morphology_matches_the_naive_definition():
    """The summed-area form has to be the same function, not an approximation of it."""
    import random

    from usd_core.space.support import _dilate, _erode_centers

    random.seed(7)
    for _ in range(200):
        n = random.randint(1, 12)
        cells = {(random.randint(0, n), random.randint(0, n))
                 for _ in range(random.randint(1, n * n))}
        ex, ey = random.randint(0, 3), random.randint(0, 3)
        assert _erode_centers(cells, ex, ey) == _naive_erode(cells, ex, ey), (cells, ex, ey)
        assert _dilate(cells, ex, ey) == _naive_dilate(cells, ex, ey), (cells, ex, ey)


def test_erosion_cost_does_not_follow_the_footprint():
    """A big grid under a big footprint must stay interactive.

    The per-cell window walk this replaced was `cells x (2ex+1)(2ey+1)`: 200x200 at
    ex=ey=60 measured 6 s, and 500x500 at ex=ey=225 — a shape `_SPACE_MAX_CELLS` still
    permits — ran for minutes with the daemon unable to answer anything else. The
    budget here is deliberately loose; it is checking the exponent, not the constant.
    """
    import time

    from usd_core.space.support import _erode_centers

    cells = {(x, y) for x in range(500) for y in range(500)}
    start = time.perf_counter()
    survivors = _erode_centers(cells, 225, 225)
    elapsed = time.perf_counter() - start
    assert len(survivors) == 50 * 50, len(survivors)
    assert elapsed < 20.0, f"500x500 erosion at ex=ey=225 took {elapsed:.1f}s"


def test_overhang_work_is_bounded_rather_than_unbounded():
    """`tau < 1` still looks at each footprint cell, so it refuses instead of hanging."""
    from usd_core.space.errors import SpaceOutOfRange

    # 4 m of floor at 5 mm cells under a 3 m object: ~640k centres x ~360k footprint
    # cells. Correct to compute, impossible to wait for.
    scene = scene_from_boxes([([-2.0, -2.0, -0.5], [2.0, 2.0, 0.0])], names=["floor"])
    with pytest.raises(SpaceOutOfRange) as excinfo:
        detect_support_regions(scene, (-2.0, -2.0, -0.05), (2.0, 2.0, 2.0), 0.005,
                               (3.0, 3.0, 0.3), tau=0.5, smooth=False,
                               max_cells=800_000)
    assert "--cell" in str(excinfo.value)


def test_more_levels_than_the_buffer_holds_is_reported_not_swallowed():
    """A column with more usable levels than `MAX_FREE` must say so.

    The rasteriser keeps 16 free spans per column and a rack can have more. The ones
    it drops are the HIGHEST, so unlike the solid-layer overflow this is not fail-safe:
    the upper bays are simply absent from the answer while `ok` stays true. Twenty
    shelves in one column reports sixteen levels and a non-zero overflow count.
    """
    shelves = [([-1.0, -1.0, i * 0.5], [1.0, 1.0, i * 0.5 + 0.05]) for i in range(20)]
    scene = scene_from_boxes(shelves, names=[f"shelf{i}" for i in range(20)])
    res = detect_support_regions(scene, (-1.0, -1.0, -0.05), (1.0, 1.0, 11.0), 0.1,
                                 (0.3, 0.3, 0.3), smooth=False)
    levels = sorted({round(r.support_z, 2) for r in res.regions})
    assert len(levels) == 16, levels          # the buffer's worth, not all nineteen
    assert res.free_overflow_cells > 0, "the dropped levels were not reported"


def _thin_walled_crate(oid=0, floor=None):
    """A sealed crate with 0.5 mm walls, optionally resting on a floor of its own.

    Thin sheets are how real assets are built, and `resting on` means the crate's
    underside is coplanar with what holds it — which is exactly what the merge sees.
    """
    parts = [([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0005], oid),
             ([-0.5, -0.5, 0.55], [0.5, 0.5, 0.5505], oid)]
    if floor is not None:
        parts.insert(0, ([-1.0, -1.0, -0.1], [1.0, 1.0, 0.0], floor))
    vs, fs, oids, base = [], [], [], 0
    for lo, hi, o in parts:
        v, f = box_mesh(lo, hi)
        vs.append(v); fs.append(f + base); base += len(v)
        oids.append(np.full(len(f), o, np.int32))
    return SceneGeometry(
        vertices=np.concatenate(vs), indices=np.concatenate(fs),
        tri_object_id=np.concatenate(oids),
        object_paths=[f"/o{i}" for i in range(max(p[2] for p in parts) + 1)],
        meters_per_unit=1.0)


def test_a_sealed_crate_stays_sealed_when_it_rests_on_something():
    """Touching another object must not open a closed one up.

    The cavity test asks whether a gap's floor and ceiling belong to the same object.
    Merging coincident solid layers used to answer "no owner" (`-1`) for the merged
    layer, and a crate standing on a floor merges by definition — resting IS coplanar
    contact. So the crate's sealed interior came back as placeable the moment it was
    put down, while the identical crate floating in space was correctly excluded.
    """
    scope = ((-1.0, -1.0, -0.2), (1.0, 1.0, 1.5))
    def interior(scene):
        res = detect_support_regions(scene, *scope, 0.05, (0.2, 0.2, 0.2), smooth=False)
        return [r for r in res.regions if 0.0 < r.support_z < 0.5]

    assert interior(_thin_walled_crate()) == [], "a lone crate's interior is not free space"
    on_floor = interior(_thin_walled_crate(oid=0, floor=1))
    assert on_floor == [], (
        "the crate's sealed interior opened up when it was set down: "
        f"{[round(r.support_z, 3) for r in on_floor]}")


def _watertight_crate_on_floor(reverse_ids=False):
    """A surface-mesh crate meets a floor at exactly its zero-thickness bottom face."""
    parts = [([-1.0, -1.0, -0.1], [1.0, 1.0, 0.0], 1 if reverse_ids else 0),
             ([-0.5, -0.5, 0.0], [0.5, 0.5, 0.55], 0 if reverse_ids else 1)]
    vs, fs, oids, base = [], [], [], 0
    for lo, hi, oid in parts:
        vertices, faces = box_mesh(lo, hi)
        vs.append(vertices)
        fs.append(faces + base)
        base += len(vertices)
        oids.append(np.full(len(faces), oid, np.int32))
    return SceneGeometry(
        vertices=np.concatenate(vs), indices=np.concatenate(fs),
        tri_object_id=np.concatenate(oids), object_paths=["/floor", "/crate"],
        meters_per_unit=1.0)


@pytest.mark.parametrize("reverse_ids", [False, True])
def test_coplanar_floor_keeps_watertight_crate_interior_sealed(reverse_ids):
    """Exact contact preserves each object's span, independently of object-id order."""
    # Restrict XY to the crate interior: a returned floor at z=0 can only be the
    # erroneously exposed cavity, while z=0.55 is the legitimate space above its roof.
    result = detect_support_regions(
        _watertight_crate_on_floor(reverse_ids), (-0.4, -0.4, -0.2), (0.4, 0.4, 1.0),
        0.05, (0.1, 0.1, 0.1), smooth=False, device="cpu")
    cavity = [region for region in result.regions if region.support_z < 0.55 - 1e-6]
    assert cavity == [], (
        "coplanar floor contact exposed the sealed crate interior: "
        f"{[round(region.support_z, 3) for region in cavity]}")


@pytest.mark.parametrize("reverse_ids", [False, True])
def test_cuda_coplanar_crate_keeps_cpu_sealed_result(reverse_ids):
    """Phase-2 labels and Phase-1 erosion must not reopen a sealed cavity."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene = _watertight_crate_on_floor(reverse_ids)
    args = ((-0.4, -0.4, -0.2), (0.4, 0.4, 1.0), 0.05, (0.1, 0.1, 0.1))
    cpu = detect_support_regions(scene, *args, smooth=False, device="cpu")
    cuda = detect_support_regions(scene, *args, smooth=False, device="cuda:0")
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def _deck_beside_a_pillar():
    """A deck at z=1.0 covering x<=0, and a pillar at x in [0.1,0.6] rising to z=1.5.

    The pillar has to stand ABOVE the deck top for the overhang to hit anything — build
    it shorter and an object hanging off the deck edge passes over it harmlessly, which
    is why this case is easy to reason past.
    """
    return scene_from_boxes([
        ([-1.0, -1.0, 0.9], [0.0, 1.0, 1.0]),
        ([0.1, -1.0, 0.0], [0.6, 1.0, 1.5]),
    ], names=["deck", "pillar"])


def test_overhang_does_not_reach_into_a_pillar():
    """The part hanging off the support still has to fit somewhere.

    `tau < 1` lets a footprint hang off the region, and the region is the only thing the
    overhang path used to look at: the ceiling came from `min` over the SUPPORTED cells,
    and the overhanging cells are outside the region by definition, so nothing ever asked
    what was over there. A deck with a pillar beside it rising above deck level then
    reported centres whose body runs straight through the pillar.

    The CoG bracket does not cover this — it asks whether the object tips, not whether
    there is room for the part sticking out.
    """
    scene = _deck_beside_a_pillar()
    scope = ((-1.0, -1.0, 0.95), (1.0, 1.0, 3.0))
    obj = (0.4, 0.4, 0.4)
    res = detect_support_regions(scene, *scope, 0.05, obj, tau=0.5, smooth=False)

    deck = [r for r in res.regions if abs(r.support_z - 1.0) < 1e-6]
    assert deck, "the deck itself must still be a support region"
    ox = res.origin[0]
    right = max(ox + (c[0] + 0.5) * 0.05 for r in deck for c in r.cells)
    assert right + obj[0] / 2.0 <= 0.1 + 1e-9, (
        f"a centre at x={right:.3f} puts the object's edge at "
        f"{right + obj[0] / 2.0:.3f}, inside the pillar at x>=0.100")


def test_cuda_partial_support_matches_cpu_pillar_clearance():
    """Phase 4 must reject the same body-through-pillar centres as the CPU oracle."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    scene = _deck_beside_a_pillar()
    args = ((-1.0, -1.0, 0.95), (1.0, 1.0, 3.0), 0.05, (0.4, 0.4, 0.4))
    cpu = detect_support_regions(scene, *args, tau=0.5, smooth=False, device="cpu")
    cuda = detect_support_regions(scene, *args, tau=0.5, smooth=False, device="cuda:0")
    assert _canonical_regions(cuda) == _canonical_regions(cpu)


def test_partial_support_warp_cpu_matches_oracle_on_deck_and_pillar():
    """CPU Warp executes the realistic tau<1, CoG, and clearance kernel path.

    This deliberately bypasses the CUDA-availability gate: Warp's CPU backend must
    agree with the Python oracle for a deck whose legal overhang is curtailed by an
    adjacent pillar.  It covers the same support-fraction, CoG-bracket, and body
    clearance semantics used by the production partial-support path.
    """
    scene = _deck_beside_a_pillar()
    smin = np.asarray((-1.0, -1.0, 0.95), dtype=np.float64)
    smax = np.asarray((1.0, 1.0, 3.0), dtype=np.float64)
    cell_size, object_height, half_footprint = 0.05, 0.4, 4
    mesh = support_module.DeviceMesh(scene, device="cpu")
    spans = support_module.build_spans_raster(
        mesh, smin, smax, cell_size, object_height,
        merge_gap=support_module.DEFAULT_MERGE_GAP,
        slope_mode="terrain", slope_threshold_deg=20.0, vertical_cutoff_deg=80.0,
    )
    raw_regions = detect_regions(
        spans, object_height,
        bottom_tolerance=min(0.1, object_height / 3.0),
        top_tolerance=min(0.2, 2.0 * object_height / 3.0),
    )
    deck = next(region for region in raw_regions if max(member[2] for member in region) < 1.1)
    bottom_top = {
        (int(ix), int(iy)): (float(bottom), float(top))
        for ix, iy, bottom, top in deck
    }
    region_cells = set(bottom_top)
    candidates = support_module._dilate(
        region_cells, half_footprint, half_footprint
    )
    oracle = support_module._partial_support_centers(
        region_cells, bottom_top, half_footprint, half_footprint, 0.5, 1,
        object_height, support_module._column_probe(spans), candidates,
    )
    device_spans = support_module._partial_support_device_spans(spans, device="cpu")
    warp_centres = support_module._partial_support_centers_warp(
        region_cells, bottom_top, spans, half_footprint, half_footprint, 0.5, 1,
        object_height, candidates, device="cpu", device_spans=device_spans,
    )

    assert warp_centres == oracle


def test_overhang_still_reaches_further_than_full_support():
    """Fixing the clearance hole must not quietly turn `--tau 0.5` into `--tau 1`.

    Over open air the overhang is legitimate and has to survive, otherwise the gate is
    just an expensive way to disable the feature.
    """
    scene = _deck_beside_a_pillar()
    scope = ((-1.0, -1.0, 0.95), (1.0, 1.0, 3.0))
    obj = (0.4, 0.4, 0.4)

    def right_edge(tau):
        res = detect_support_regions(scene, *scope, 0.05, obj, tau=tau, smooth=False)
        cells = [c for r in res.regions if abs(r.support_z - 1.0) < 1e-6 for c in r.cells]
        return max(res.origin[0] + (c[0] + 0.5) * 0.05 for c in cells)

    assert right_edge(0.5) > right_edge(1.0) + 1e-9, "overhang gained nothing over full support"


def test_overhang_over_a_column_of_open_air_is_allowed():
    """A column with no geometry at all is free, not blocked.

    The span grid emits NO free span for an empty column (it has no rest surface), which
    is the same zero a fully solid column reports. Reading only that count would refuse
    every overhang over a void — the case the feature exists for. `solid_count` is what
    tells them apart.
    """
    pad = scene_from_boxes([([-0.15, -0.15, -0.5], [0.15, 0.15, 0.0])], names=["pad"])
    res = detect_support_regions(pad, (-1.0, -1.0, -0.05), (1.0, 1.0, 2.0), CS, OBJ,
                                 tau=0.3, smooth=False)
    assert res.regions, "a pad surrounded by empty air must still allow overhang"


def test_merge_gap_only_matters_below_the_object_height():
    """The tolerance fuses solid layers; the free-span filter drops short spans.

    Those two together mean a merged gap can only ever have been reachable by an
    object shorter than the gap, so for anything taller the value is a no-op — which
    is why it can carry a fixed default at all. Below it, the merge deletes a real
    resting surface, which is why it has to be settable.
    """
    plates = scene_from_boxes([
        ([-0.5, -0.5, 0.0000], [0.5, 0.5, 0.0020]),   # lower, top at 2.0 mm
        ([-0.5, -0.5, 0.0029], [0.5, 0.5, 0.0049]),   # upper, bottom at 2.9 mm
    ], names=["lower", "upper"])
    scope = ((-0.5, -0.5, -0.001), (0.5, 0.5, 0.02))

    def levels(obj, **kw):
        res = detect_support_regions(plates, *scope, 0.01, obj, smooth=False, **kw)
        return sorted({round(r.support_z * 1000.0, 3) for r in res.regions})

    # 0.5 mm object: it fits in the 0.9 mm gap, and the default fuses that gap shut.
    assert levels((0.05, 0.05, 0.0005)) == [4.9]
    assert levels((0.05, 0.05, 0.0005), merge_gap=0.0) == [2.0, 4.9]
    # 5 mm object: taller than any gap the tolerance can fuse, so it cannot see it.
    tall = (0.05, 0.05, 0.005)
    assert levels(tall) == levels(tall, merge_gap=0.0) == levels(tall, merge_gap=1.0e-6)
