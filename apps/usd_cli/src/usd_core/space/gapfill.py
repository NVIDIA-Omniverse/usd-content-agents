# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge slat gaps from the SIDES — the missing half of the sliding-window pass.

Upstream's `smooth` bridges a slatted deck exactly the right way: it takes a gap
column and lifts its floor to the level the neighbours agree on, over a window of the
object's own footprint, and only when **both** opposing sides match. Nothing about that
rule needs to know what is underneath the gap — the answer is a function of the left and
right heights alone.

The limitation is not in the rule, it is in what the pass is allowed to touch. Its
docstring is explicit that it "never adds, drops, or reorders spans": it can only lift a
floor that already exists. That was complete for the raycast span builder, where a
down-ray always hits something and every column therefore has a span. Under the
rasterised heightfield a column containing no geometry at all gets no span (free spans
are built as the gaps *above* solid layers, and there is no layer), so the pass has
nothing to lift and the gap silently stays a hole:

    slat gap over the pallet's bottom deck board  -> floor at 30 mm  -> lifted
    slat gap over open air                        -> no span at all  -> untouched

Both occur on the same EUR pallet deck, so two identical 55 mm gaps were bridged and two
were not. Each unbridged gap is a one-cell hole, and the footprint erosion in
`detect_support_regions` then widens every hole by the object's half-footprint — which
is what tore one continuous deck into disconnected strips.

This pass runs first and gives those columns a span to lift, using the same both-sides
rule. Crucially it synthesizes the span **at the neighbours' agreed height**, never at
the scope floor: making the span builder emit a full-height span for empty columns
instead (tried, reverted) floors every empty cell at the bottom of the scope, and the
region grouper then joins them into a large phantom resting surface wherever the scene
is simply empty.

Conservative by construction:

* a synthetic level must be fully contained in free air already measured in its column
  (or the column must be genuinely empty).  This permits an upper slatted deck above a
  lower continuous floor without ever adding a span through a wall or other solid;
* the window is the object's own footprint, so a gap wider than the object is never
  bridged;
* support must exist on **both** sides along an axis and agree in height — a deck's
  outer edge has support on one side only and is left alone, so nothing is ever floated
  out over a void;
* every free layer is considered. A rack can have a ground-level span, an intermediate
  bay, and a top-deck span in one neighbour column; selecting only slot zero bridges
  the wrong level and drops the upper deck;
* each synthesized span rests on the higher of the two neighbouring floors and is capped
  by the lower of their ceilings. It is emitted only when that common interval still
  fits the requested object, so it never claims headroom its neighbours lack.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import SpaceOutOfRange


# The maps themselves are vectorised, but their known-floor lookup and the general
# synthesis proof both compare level pairs. Keep that worst-case request below a
# daemon-scale latency rather than allowing a legal grid to spend minutes walking every
# open column and every free-level pair.
MAX_GAPFILL_WORK = 50_000_000


def _fill_single_level_vectorized(
    spans: dict,
    *,
    wx: int,
    wy: int,
    min_height: float,
    bottom_tolerance: float,
) -> dict | None:
    """Fast path for the overwhelmingly common one-free-level heightfield.

    A pallet deck over open air has one usable free band in each slat column.  The
    general multi-level algorithm below must retain Python lists because it can add
    several differently-heighted bands to one column.  Doing that bookkeeping for
    every cell in this one-level case used to dominate CUDA queries *after* their
    Warp raster and smoothing passes had completed.

    Here every target can receive at most one synthetic level, so the exact same
    opposing-side rule is expressible as array operations.  This intentionally stays
    on the host: ``build_spans_raster`` has already copied the span arrays back for
    region extraction, and uploading four nearest-neighbour maps solely for this
    operation costs more than the vectorised scan.  A future device-resident compact
    heightfield will move gapfill together with extraction, not introduce a ping-pong
    transfer for this isolated stage.

    Returns ``None`` when more than one measured free level occurs anywhere; the
    caller then uses the general, multi-level-safe implementation.
    """
    count = spans["span_count"]
    if int(np.max(count)) != 1:
        return None

    measured = count == 1
    if not np.any(measured):
        return spans
    # A single occupied slot does not necessarily mean a single *level*: a lower
    # floor plus an upper slatted shelf may have one band in every column but at
    # different heights.  That case can synthesize two valid levels in a gap and
    # must retain the general path.  The bulk path is only the genuinely uniform
    # deck case.
    # Match the scalar proof's Python-float comparisons exactly. The stored values
    # are float32, but subtracting them in float32 can round a just-over-tolerance
    # step down to the tolerance and incorrectly select this fast path.
    measured_floor = spans["span_min"][:, :, 0][measured].astype(np.float64)
    measured_ceiling = spans["span_max"][:, :, 0][measured].astype(np.float64)
    if (
        float(measured_floor.max() - measured_floor.min()) > bottom_tolerance
        or float(measured_ceiling.max() - measured_ceiling.min()) > bottom_tolerance
    ):
        return None

    nx, ny = int(spans["nx"]), int(spans["ny"])
    capacity = int(spans["span_min"].shape[2])
    if capacity < 1:
        return spans
    solid_count = spans["solid_count"]
    # A no-span column is bridgeable only when it contains no solid geometry.  This is
    # the same ``is_open`` test as the general path; a fully-solid column is never
    # turned into a virtual support surface.
    open_cell = (count == 0) & (solid_count == 0)
    if not np.any(open_cell):
        return spans

    left, right, below, above = _nearest_support_maps(count, 1)
    floors = spans["span_min"][:, :, 0].astype(np.float64)
    ceilings = spans["span_max"][:, :, 0].astype(np.float64)
    grid_x, grid_y = np.indices((nx, ny), dtype=np.int32)
    best_floor = np.full((nx, ny), np.inf, dtype=np.float64)
    best_ceiling = np.full((nx, ny), -np.inf, dtype=np.float64)

    def consider(lo: np.ndarray, hi: np.ndarray, along_x: bool) -> None:
        """Merge one axis's matching-side witness into the chosen level.

        The generic path appends X then Y witnesses, sorts by floor, and, when the
        fixed one-level capacity is reached, keeps the lowest floor.  Selecting the
        lower candidate here reproduces that policy.  Equal levels are intersected
        (higher floor/lower ceiling), exactly as its ``merged`` step does.
        """
        valid = open_cell & (lo >= 0) & (hi >= 0)
        if along_x:
            valid &= grid_x - lo <= wx
            valid &= hi - grid_x <= wx
            lo_floor, lo_ceil = (
                floors[lo.clip(min=0), grid_y],
                ceilings[lo.clip(min=0), grid_y],
            )
            hi_floor, hi_ceil = (
                floors[hi.clip(min=0), grid_y],
                ceilings[hi.clip(min=0), grid_y],
            )
        else:
            valid &= grid_y - lo <= wy
            valid &= hi - grid_y <= wy
            lo_floor, lo_ceil = (
                floors[grid_x, lo.clip(min=0)],
                ceilings[grid_x, lo.clip(min=0)],
            )
            hi_floor, hi_ceil = (
                floors[grid_x, hi.clip(min=0)],
                ceilings[grid_x, hi.clip(min=0)],
            )
        floor = np.maximum(lo_floor, hi_floor)
        ceiling = np.minimum(lo_ceil, hi_ceil)
        valid &= np.abs(lo_floor - hi_floor) <= bottom_tolerance
        valid &= ceiling - floor >= min_height
        # X is considered first, matching the general candidate order.  A lower Y
        # floor replaces it; an equal level intersects it.
        replace = valid & (
            (~np.isfinite(best_floor)) | (floor < best_floor - bottom_tolerance)
        )
        equal = valid & ~replace & (np.abs(floor - best_floor) <= bottom_tolerance)
        best_floor[replace] = floor[replace]
        best_ceiling[replace] = ceiling[replace]
        best_floor[equal] = np.maximum(best_floor[equal], floor[equal])
        best_ceiling[equal] = np.minimum(best_ceiling[equal], ceiling[equal])

    consider(left[0], right[0], True)
    consider(below[0], above[0], False)

    accepted = (
        open_cell & np.isfinite(best_floor) & (best_ceiling - best_floor >= min_height)
    )
    if not np.any(accepted):
        return spans
    out_min = spans["span_min"].copy()
    out_max = spans["span_max"].copy()
    out_count = count.copy()
    out_min[:, :, 0][accepted] = best_floor[accepted]
    out_max[:, :, 0][accepted] = best_ceiling[accepted]
    out_count[accepted] = 1
    bridged = dict(spans)
    bridged["span_min"] = out_min
    bridged["span_max"] = out_max
    bridged["span_count"] = out_count
    bridged["bridged_cells"] = int(spans.get("bridged_cells", 0)) + int(
        np.count_nonzero(accepted)
    )
    bridged["gapfill_overflow_cells"] = int(spans.get("gapfill_overflow_cells", 0))
    return bridged


def _nearest_support_maps(
    count: np.ndarray, max_spans: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest support maps for every free-span level and direction.

    A lower free band does not prove an upper deck exists.  Therefore level ``k`` only
    treats a column as support when that column has an original span at slot ``k``;
    intermediate lower-only slat gaps cannot shadow a farther matching upper slat.
    Each level/direction map takes one linear sweep, replacing the old per-cell
    ``for step in range(window)`` search whose work grew with footprint radius.
    Each value is the coordinate on that axis or ``-1`` when none exists.
    """
    nx, ny = count.shape
    shape = (int(max_spans), nx, ny)
    left = np.full(shape, -1, dtype=np.int32)
    right = np.full(shape, -1, dtype=np.int32)
    below = np.full(shape, -1, dtype=np.int32)
    above = np.full(shape, -1, dtype=np.int32)

    for level in range(int(max_spans)):
        supported = count > level
        last_x = np.full(ny, -1, dtype=np.int32)
        for ix in range(nx):
            left[level, ix, :] = last_x
            last_x[supported[ix, :]] = ix
        last_x.fill(-1)
        for ix in range(nx - 1, -1, -1):
            right[level, ix, :] = last_x
            last_x[supported[ix, :]] = ix

        last_y = np.full(nx, -1, dtype=np.int32)
        for iy in range(ny):
            below[level, :, iy] = last_y
            last_y[supported[:, iy]] = iy
        last_y.fill(-1)
        for iy in range(ny - 1, -1, -1):
            above[level, :, iy] = last_y
            last_y[supported[:, iy]] = iy
    return left, right, below, above


def _candidate_cells(
    spans: dict,
    *,
    left: np.ndarray,
    right: np.ndarray,
    below: np.ndarray,
    above: np.ndarray,
    wx: int,
    wy: int,
    min_height: float,
    bottom_tolerance: float,
    max_spans: int,
) -> np.ndarray:
    """Return columns which could gain a synthetic free level.

    The exact multi-level proof below still handles every selected column.  This is
    only a necessary-condition filter: it skips the broad continuous bays where the
    target already owns the same level as both neighbours, and must never skip a
    possible new level or a reportable capacity overflow.
    """
    count = spans["span_count"]
    smin = spans["span_min"][:, :, :max_spans]
    smax = spans["span_max"][:, :, :max_spans]
    solid_count = spans["solid_count"]
    nx, ny = count.shape
    grid_x, grid_y = np.indices((nx, ny), dtype=np.int32)
    allowed = (count > 0) | (solid_count == 0)
    needs = np.zeros((nx, ny), dtype=bool)

    def consider(lo: np.ndarray, hi: np.ndarray, level: int, along_x: bool) -> None:
        valid = allowed & (lo >= 0) & (hi >= 0)
        if along_x:
            valid &= (grid_x - lo <= wx) & (hi - grid_x <= wx)
            lo_floor, lo_ceiling = (
                smin[lo.clip(min=0), grid_y, level],
                smax[lo.clip(min=0), grid_y, level],
            )
            hi_floor, hi_ceiling = (
                smin[hi.clip(min=0), grid_y, level],
                smax[hi.clip(min=0), grid_y, level],
            )
        else:
            valid &= (grid_y - lo <= wy) & (hi - grid_y <= wy)
            lo_floor, lo_ceiling = (
                smin[grid_x, lo.clip(min=0), level],
                smax[grid_x, lo.clip(min=0), level],
            )
            hi_floor, hi_ceiling = (
                smin[grid_x, hi.clip(min=0), level],
                smax[grid_x, hi.clip(min=0), level],
            )
        # The exact proof widens the float32 storage through ``float(...)`` before
        # every height comparison.  Do the same before subtracting here: a float32
        # subtraction can round a clearance just above the object height down and
        # incorrectly prune a bridge that the proof would accept.
        lo_floor = lo_floor.astype(np.float64)
        hi_floor = hi_floor.astype(np.float64)
        lo_ceiling = lo_ceiling.astype(np.float64)
        hi_ceiling = hi_ceiling.astype(np.float64)
        floor = np.maximum(lo_floor, hi_floor)
        ceiling = np.minimum(lo_ceiling, hi_ceiling)
        valid &= np.abs(lo_floor - hi_floor) <= bottom_tolerance
        valid &= ceiling - floor >= min_height
        # This is a deliberately conservative *same-slot* fast rejection.  A match
        # proves the target already has this level; a match in another slot merely
        # leaves the cell for the exact proof below.  That preserves correctness while
        # avoiding a whole ``grid x levels`` comparison for every witness level.
        known_floor = (
            (count > level)
            & (np.abs(smin[:, :, level].astype(np.float64) - floor)
               <= bottom_tolerance)
        )
        needs[:] |= valid & ~known_floor

    for level in range(max_spans):
        consider(left[level], right[level], level, True)
        consider(below[level], above[level], level, False)
    return needs


def _exact_proof_work(
    candidate_cells: np.ndarray,
    count: np.ndarray,
    *,
    left: np.ndarray,
    right: np.ndarray,
    below: np.ndarray,
    above: np.ndarray,
    wx: int,
    wy: int,
) -> int:
    """Conservative Python-work bound for the selected multi-level proof.

    Each candidate can only inspect a witness where the corresponding nearest-map
    pair exists inside the footprint window.  Charging every selected column as if
    it had every X and Y witness (and every possible merge) rejects sparse racks
    purely because another column has a deep span stack.  Count the witnesses that
    can actually enter ``candidates`` instead. The exact proof also traverses every
    observed slot and can scan existing spans twice per synthesized level, so charge
    those Python loops explicitly.
    """
    nx, ny = candidate_cells.shape
    grid_x = np.arange(nx, dtype=np.int32)[None, :, None]
    grid_y = np.arange(ny, dtype=np.int32)[None, None, :]
    witness_count = np.zeros((nx, ny), dtype=np.int64)
    if wx > 0:
        horizontal = (
            (left >= 0) & (right >= 0)
            & (grid_x - left <= wx) & (right - grid_x <= wx)
        )
        witness_count += np.sum(horizontal, axis=0, dtype=np.int64)
    if wy > 0:
        vertical = (
            (below >= 0) & (above >= 0)
            & (grid_y - below <= wy) & (above - grid_y <= wy)
        )
        witness_count += np.sum(vertical, axis=0, dtype=np.int64)
    witnesses = witness_count[candidate_cells]
    existing = count[candidate_cells].astype(np.int64)
    # The proof visits every observed slot before forming ``witnesses``. It then
    # builds ``original`` once, scans witnesses for pairing, and merges them against
    # prior levels. For each merged level, the two ``any`` predicates can each scan
    # every original span (floor identity, then interval containment). Later
    # geometric predicates only reduce this bound.
    levels = int(left.shape[0])
    return int(np.sum(
        levels + existing + 2 * witnesses + witnesses * witnesses + 2 * witnesses * existing
    ))


def fill_empty_columns(
    spans: dict,
    x_length_threshold: float,
    y_length_threshold: float,
    section_height_threshold: float,
    bottom_tolerance: float = 0.02,
) -> dict:
    """Give empty columns all supported free layers from their neighbourhood.

    Args:
        spans: a `build_spans_raster` result. Requires the `solid_count` field.
        x_length_threshold, y_length_threshold: the object footprint in metres; the
            per-axis half-window is `ceil((length / 2) / cell_size)` cells, matching
            `smooth.smooth_spans`.
        section_height_threshold: minimum free height for a synthesized span to be
            worth emitting (the object height).
        bottom_tolerance: how closely the two sides' support heights must agree to
            count as one continuous surface rather than a step.

    Returns a new spans dict (inputs are not mutated). A no-op — returning `spans`
    unchanged — when there is nothing to bridge or `solid_count` is unavailable, so it
    is safe to call unconditionally.
    """
    solid_count = spans.get("solid_count")
    if solid_count is None:
        return spans  # older heightfield result: nothing we can do safely

    nx, ny = int(spans["nx"]), int(spans["ny"])
    cell = float(spans["cell_size"])
    if cell <= 0:
        return spans
    wx = int(math.ceil((float(x_length_threshold) / 2.0) / cell))
    wy = int(math.ceil((float(y_length_threshold) / 2.0) / cell))
    if wx <= 0 and wy <= 0:
        return spans  # object smaller than a cell — nothing to bridge across

    capacity = int(spans["span_min"].shape[2])
    count = spans["span_count"]
    min_height = float(section_height_threshold)
    # Only allocate maps for levels that actually occur. The raster storage has a fixed
    # capacity, but charging all unused slots would reject ordinary large single-level
    # scenes solely because that capacity is generous.
    max_spans = min(capacity, int(np.max(count)))
    if max_spans <= 0:
        return spans
    # The vectorised prefilter is linear in grid cells and observed levels.  It is
    # intentionally budgeted separately from the Python exact proof below: treating
    # array element operations as Python candidate work would reject ordinary scoped
    # rack queries solely because one remote column has several free bands.
    # Preserve the former accepted-query envelope for this vectorised prefilter.  It
    # is NumPy work, not the Python candidate proof guarded below; the precise
    # per-candidate bound is calculated only after it identifies candidates.
    prefilter_work = nx * ny * (6 * max_spans + 1)
    if prefilter_work > MAX_GAPFILL_WORK:
        raise SpaceOutOfRange(
            f"gap bridging would perform up to {prefilter_work:,} bounded prefilter operations, over the "
            f"{MAX_GAPFILL_WORK:,} work budget. Use a larger --cell, a smaller "
            f"--scope, or --no-smooth."
        )

    # Most pallet/grating queries have one usable free band per column.  Keep that
    # hot path bulk-vectorised; multi-level racks deliberately take the general path
    # below because a cell can then need several independently justified additions.
    if max_spans == 1:
        accelerated = _fill_single_level_vectorized(
            spans,
            wx=wx,
            wy=wy,
            min_height=min_height,
            bottom_tolerance=bottom_tolerance,
        )
        if accelerated is not None:
            return accelerated

    left, right, below, above = _nearest_support_maps(count, max_spans)
    candidate_cells = _candidate_cells(
        spans,
        left=left,
        right=right,
        below=below,
        above=above,
        wx=wx,
        wy=wy,
        min_height=min_height,
        bottom_tolerance=bottom_tolerance,
        max_spans=max_spans,
    )
    if not np.any(candidate_cells):
        return spans

    # Only selected cells enter the exact Python proof. Charge its actual possible
    # nearest-map witnesses instead of a global maximum number of levels squared.
    exact_work = _exact_proof_work(
        candidate_cells, count, left=left, right=right, below=below, above=above,
        wx=wx, wy=wy)
    if exact_work > MAX_GAPFILL_WORK:
        raise SpaceOutOfRange(
            f"gap bridging would perform up to {exact_work:,} exact candidate operations, over the "
            f"{MAX_GAPFILL_WORK:,} work budget. Use a larger --cell, a smaller "
            f"--scope, or --no-smooth."
        )

    span_min = spans["span_min"].copy()
    span_max = spans["span_max"].copy()
    out_count = count.copy()
    filled = 0
    overflow = 0

    def support_at(ix: int, iy: int):
        """All real free layers in that cell as ``(floor, ceiling)`` pairs.

        Reads the ORIGINAL counts, not the working copy: bridging must be decided
        against real measured support, or one bridged cell becomes the evidence for
        bridging its neighbour and a gap wider than the object creeps across.
        """
        if not (0 <= ix < nx and 0 <= iy < ny):
            return []
        return [
            (float(spans["span_min"][ix, iy, k]), float(spans["span_max"][ix, iy, k]))
            for k in range(int(count[ix, iy]))
        ]

    for ix, iy in np.argwhere(candidate_cells):
        ix, iy = int(ix), int(iy)
        # Keep the established proof body nested without duplicating it: candidates
        # are a filtered traversal, not a semantic rewrite of the proof itself.
        for _candidate in (None,):
            # The original spans are the evidence, and must never be overwritten.
            # An empty column is open at every height; a non-empty one may still be
            # open through an upper slat gap while retaining a lower free level.
            original = support_at(ix, iy)
            is_open = not original and int(solid_count[ix, iy]) == 0
            if not original and not is_open:
                continue
            candidates = []
            for level in range(max_spans):
                if wx > 0:
                    lo, hi = int(left[level, ix, iy]), int(right[level, ix, iy])
                    if lo >= 0 and hi >= 0 and ix - lo <= wx and hi - ix <= wx:
                        candidates.append(
                            (
                                float(spans["span_min"][lo, iy, level]),
                                float(spans["span_max"][lo, iy, level]),
                                float(spans["span_min"][hi, iy, level]),
                                float(spans["span_max"][hi, iy, level]),
                            )
                        )
                if wy > 0:
                    lo, hi = int(below[level, ix, iy]), int(above[level, ix, iy])
                    if lo >= 0 and hi >= 0 and iy - lo <= wy and hi - iy <= wy:
                        candidates.append(
                            (
                                float(spans["span_min"][ix, lo, level]),
                                float(spans["span_max"][ix, lo, level]),
                                float(spans["span_min"][ix, hi, level]),
                                float(spans["span_max"][ix, hi, level]),
                            )
                        )

            # A nearest-map entry exists only where both columns have this exact
            # free-span level.  Comparing those matching slots is both the geometric
            # rule (a lower bay cannot certify a higher deck) and a hard performance
            # bound: at most two candidates per level, rather than every left/right
            # layer pair. Span slots are ordered by floor in the heightfield.
            paired = []
            for lo_floor, lo_ceil, hi_floor, hi_ceil in candidates:
                if abs(lo_floor - hi_floor) > bottom_tolerance:
                    continue
                floor, ceiling = max(lo_floor, hi_floor), min(lo_ceil, hi_ceil)
                if ceiling - floor >= min_height:
                    paired.append((floor, ceiling))

            # The same level can be found along X and Y.  Collapse those witnesses into
            # one stricter interval: higher floor + lower ceiling.  This avoids duplicate
            # synthetic levels while retaining the evidence that is safe on both axes.
            merged = []
            for floor, ceiling in sorted(paired):
                for i, (old_floor, old_ceiling) in enumerate(merged):
                    if abs(floor - old_floor) <= bottom_tolerance:
                        merged[i] = (max(floor, old_floor), min(ceiling, old_ceiling))
                        break
                else:
                    merged.append((floor, ceiling))
            merged = [
                (floor, ceiling)
                for floor, ceiling in merged
                if ceiling - floor >= min_height
            ]

            additions = []
            for floor, ceiling in merged:
                # An existing floor already represents this support level.  Otherwise
                # a non-empty column must prove that the entire virtual upper level is
                # free; this is what distinguishes a slat gap from a wall.
                if any(
                    abs(old_floor - floor) <= bottom_tolerance
                    for old_floor, _ in original
                ):
                    continue
                if is_open or any(
                    old_floor <= floor and old_ceiling >= ceiling
                    for old_floor, old_ceiling in original
                ):
                    additions.append((floor, ceiling))
            if not additions:
                continue
            if len(original) + len(additions) > capacity:
                # Do not evict a measured span just to add a synthetic one.  Losing the
                # virtual level is conservative, and the caller must be told about it.
                # A genuinely open column has no measured span to protect, so retain
                # the established lower-level-first policy for its synthetic levels.
                overflow += 1
                if original:
                    continue
                additions = additions[:capacity]
            levels = sorted([*original, *additions])
            for k, (floor, ceiling) in enumerate(levels):
                span_min[ix, iy, k], span_max[ix, iy, k] = floor, ceiling
            out_count[ix, iy] = len(levels)
            filled += 1

    if not filled and not overflow:
        return spans
    bridged = dict(spans)
    if filled:
        bridged["span_min"] = span_min
        bridged["span_max"] = span_max
        bridged["span_count"] = out_count
        bridged["bridged_cells"] = int(spans.get("bridged_cells", 0)) + filled
    bridged["gapfill_overflow_cells"] = (
        int(spans.get("gapfill_overflow_cells", 0)) + overflow
    )
    return bridged
