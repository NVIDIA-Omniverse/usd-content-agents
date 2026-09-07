# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy Warp CPU/CUDA coverage-mask scoring for usd-cli camera analysis.

Geometry and visibility remain owned by Newton.  This module owns the usd-cli-specific
parallel work called out by the camera-rig contract: packed coverage bitsets, overlap
histograms, mask reduction, per-camera marginal contribution, and deterministic greedy
marginal-gain selection.  Warp is imported only when one of these functions executes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usd_core.camera_analysis.cancellation import check_cancelled


@dataclass(frozen=True)
class CoverageMaskScores:
    counts: np.ndarray
    covered: np.ndarray
    accessible_count: int
    covered_count: int
    histogram: dict[int, int]
    visible_counts: np.ndarray
    marginal_counts: np.ndarray


@dataclass(frozen=True)
class GreedyMaskScores:
    selected: tuple[int, ...]
    marginal_counts: tuple[int, ...]
    counts: np.ndarray
    covered: np.ndarray
    accessible_count: int
    covered_count: int
    visible_counts: np.ndarray
    stop_reason: str


def _lazy_runtime():
    try:
        import warp as wp
    except ImportError as exc:
        from usd_core.camera_analysis.newton_backend import CameraAnalysisUnavailable

        raise CameraAnalysisUnavailable(
            "camera scoring requires the optional Warp backend; install "
            "`usd-cli[camera-analysis]`"
        ) from exc
    from usd_core.camera_analysis import _warp_scoring_kernels as kernels

    return wp, kernels


def _validated_masks(
    masks: np.ndarray, accessible: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int]:
    mask_array = np.asarray(masks)
    accessible_array = np.asarray(accessible)
    if mask_array.ndim < 2:
        raise ValueError("camera masks must have shape (cameras, ...cells)")
    camera_count = int(mask_array.shape[0])
    cell_shape = tuple(int(value) for value in mask_array.shape[1:])
    if camera_count < 1 or any(value < 1 for value in cell_shape):
        raise ValueError("camera masks must contain at least one camera and one cell")
    if accessible_array.shape != cell_shape:
        raise ValueError(
            "accessible mask shape must match the camera-mask cell dimensions"
        )
    if mask_array.dtype.kind not in "biu" or accessible_array.dtype.kind not in "biu":
        raise ValueError("camera and accessible masks must be boolean/integer arrays")
    return (
        np.ascontiguousarray(mask_array.reshape(camera_count, -1) != 0),
        np.ascontiguousarray(accessible_array.reshape(-1) != 0),
        cell_shape,
        camera_count,
        int(np.prod(cell_shape, dtype=np.int64)),
    )


def _device(wp, requested):
    try:
        return wp.get_device(requested)
    except Exception as exc:  # noqa: BLE001 - normalize public diagnostics
        raise ValueError(
            f"unknown or unavailable Warp device {requested!r}: {exc}"
        ) from exc


def _packed_inputs(masks: np.ndarray, accessible: np.ndarray, device):
    check_cancelled()
    wp, kernels = _lazy_runtime()
    masks, accessible, cell_shape, camera_count, cell_count = _validated_masks(
        masks, accessible
    )
    resolved_device = _device(wp, device)
    word_count = (cell_count + kernels.WORD_BITS - 1) // kernels.WORD_BITS
    wp_masks = wp.array(
        masks.reshape(-1).astype(np.uint8, copy=False),
        dtype=wp.uint8,
        device=resolved_device,
    )
    wp_accessible = wp.array(
        accessible.astype(np.uint8, copy=False),
        dtype=wp.uint8,
        device=resolved_device,
    )
    packed = wp.empty(
        camera_count * word_count, dtype=wp.uint32, device=resolved_device
    )
    wp.launch(
        kernels.pack_mask_bits,
        dim=camera_count * word_count,
        inputs=[wp_masks, cell_count, word_count, packed],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    return (
        wp,
        kernels,
        resolved_device,
        packed,
        wp_accessible,
        masks,
        cell_shape,
        camera_count,
        cell_count,
        word_count,
    )


def score_coverage_masks(
    masks: np.ndarray,
    accessible: np.ndarray,
    *,
    per_cell: int,
    device="cpu",
) -> CoverageMaskScores:
    """Reduce existing-camera coverage using packed Warp bitsets."""

    check_cancelled()
    if isinstance(per_cell, bool) or not isinstance(per_cell, int | np.integer):
        raise ValueError("per-cell redundancy must be an integer")
    if per_cell < 1:
        raise ValueError("per-cell redundancy must be positive")
    (
        wp,
        kernels,
        resolved_device,
        packed,
        wp_accessible,
        _mask_values,
        cell_shape,
        camera_count,
        cell_count,
        word_count,
    ) = _packed_inputs(masks, accessible, device)
    counts = wp.empty(cell_count, dtype=wp.int32, device=resolved_device)
    covered = wp.empty(cell_count, dtype=wp.uint8, device=resolved_device)
    histogram = wp.zeros(camera_count + 1, dtype=wp.int32, device=resolved_device)
    totals = wp.zeros(2, dtype=wp.int32, device=resolved_device)
    visible_counts = wp.zeros(camera_count, dtype=wp.int32, device=resolved_device)
    marginal_counts = wp.zeros(camera_count, dtype=wp.int32, device=resolved_device)
    check_cancelled()
    wp.launch(
        kernels.reduce_coverage,
        dim=cell_count,
        inputs=[
            packed,
            wp_accessible,
            camera_count,
            cell_count,
            word_count,
            int(per_cell),
            counts,
            covered,
            histogram,
            totals,
        ],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    wp.launch(
        kernels.reduce_camera_contributions,
        dim=camera_count * cell_count,
        inputs=[
            packed,
            wp_accessible,
            counts,
            cell_count,
            word_count,
            int(per_cell),
            visible_counts,
            marginal_counts,
        ],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    host_totals = totals.numpy()
    host_histogram = histogram.numpy()
    return CoverageMaskScores(
        counts=counts.numpy().astype(np.int16, copy=False).reshape(cell_shape),
        covered=(covered.numpy() != 0).reshape(cell_shape),
        accessible_count=int(host_totals[0]),
        covered_count=int(host_totals[1]),
        histogram={
            index: int(value)
            for index, value in enumerate(host_histogram)
            if int(value) > 0
        },
        visible_counts=visible_counts.numpy().astype(np.int64, copy=False),
        marginal_counts=marginal_counts.numpy().astype(np.int64, copy=False),
    )


def greedy_select_masks(
    masks: np.ndarray,
    accessible: np.ndarray,
    *,
    per_cell: int,
    max_cameras: int,
    target_coverage: float,
    minimum_gain: float,
    device="cpu",
) -> GreedyMaskScores:
    """Select deterministic lexicographic-gain masks using Warp reductions.

    Candidates maximize ``(newly_satisfied, progress, -input_index)``. ``progress``
    counts visible accessible cells whose current count is below ``per_cell``;
    ``newly_satisfied`` is the subset whose count is exactly ``per_cell - 1``.
    ``minimum_gain`` remains a progress fraction of all accessible cells and acts as
    an eligibility threshold before lexicographic selection. ``marginal_counts``
    likewise retains its historical progress-count meaning. Only compact score/count
    scalars cross back to the host per iteration.
    """

    check_cancelled()
    if isinstance(per_cell, bool) or not isinstance(per_cell, int | np.integer):
        raise ValueError("per-cell redundancy must be an integer")
    if per_cell < 1:
        raise ValueError("per-cell redundancy must be positive")
    if isinstance(max_cameras, bool) or not isinstance(max_cameras, int | np.integer):
        raise ValueError("max cameras must be an integer")
    if max_cameras < 1:
        raise ValueError("max cameras must be positive")
    if not np.isfinite(target_coverage) or not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target coverage must be between 0 and 1")
    if not np.isfinite(minimum_gain) or not 0.0 <= minimum_gain <= 1.0:
        raise ValueError("minimum gain must be between 0 and 1")

    (
        wp,
        kernels,
        resolved_device,
        packed,
        wp_accessible,
        _mask_values,
        cell_shape,
        camera_count,
        cell_count,
        word_count,
    ) = _packed_inputs(masks, accessible, device)
    counts = wp.zeros(cell_count, dtype=wp.int32, device=resolved_device)
    accessible_total_buffer = wp.zeros(1, dtype=wp.int32, device=resolved_device)
    visible_counts_buffer = wp.zeros(
        camera_count, dtype=wp.int32, device=resolved_device
    )
    unused_marginal_counts = wp.zeros(
        camera_count, dtype=wp.int32, device=resolved_device
    )
    check_cancelled()
    wp.launch(
        kernels.count_accessible_cells,
        dim=cell_count,
        inputs=[wp_accessible, accessible_total_buffer],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    wp.launch(
        kernels.reduce_camera_contributions,
        dim=camera_count * cell_count,
        inputs=[
            packed,
            wp_accessible,
            counts,
            cell_count,
            word_count,
            int(per_cell),
            visible_counts_buffer,
            unused_marginal_counts,
        ],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    accessible_total = int(accessible_total_buffer.numpy()[0])
    if accessible_total < 1:
        raise ValueError("accessible mask contains no cells")
    visible_counts = visible_counts_buffer.numpy().astype(np.int64, copy=False)
    remaining = list(range(camera_count))
    selected: list[int] = []
    marginal_counts: list[int] = []
    stop_reason = "camera_limit"
    covered_count = 0

    while len(selected) < max_cameras:
        check_cancelled()
        covered_total = wp.zeros(1, dtype=wp.int32, device=resolved_device)
        wp.launch(
            kernels.count_covered_cells,
            dim=cell_count,
            inputs=[wp_accessible, counts, int(per_cell), covered_total],
            device=resolved_device,
        )
        wp.synchronize_device(resolved_device)
        check_cancelled()
        covered_count = int(covered_total.numpy()[0])
        ratio = covered_count / accessible_total
        if ratio + 1.0e-12 >= target_coverage:
            stop_reason = "target_met"
            break
        if not remaining:
            stop_reason = "no_valid_candidate"
            break

        wp_remaining = wp.array(
            np.asarray(remaining, dtype=np.int32),
            dtype=wp.int32,
            device=resolved_device,
        )
        newly_satisfied_gains = wp.zeros(
            len(remaining), dtype=wp.int32, device=resolved_device
        )
        progress_gains = wp.zeros(
            len(remaining), dtype=wp.int32, device=resolved_device
        )
        check_cancelled()
        wp.launch(
            kernels.score_remaining_gains,
            dim=len(remaining) * cell_count,
            inputs=[
                packed,
                wp_accessible,
                counts,
                wp_remaining,
                cell_count,
                word_count,
                int(per_cell),
                newly_satisfied_gains,
                progress_gains,
            ],
            device=resolved_device,
        )
        wp.synchronize_device(resolved_device)
        check_cancelled()
        host_newly_satisfied = newly_satisfied_gains.numpy()
        host_progress = progress_gains.numpy()
        positive_slots = [
            slot for slot in range(len(remaining)) if int(host_progress[slot]) > 0
        ]
        if not positive_slots:
            stop_reason = "no_valid_candidate"
            break
        eligible_slots = [
            slot
            for slot in positive_slots
            if int(host_progress[slot]) / accessible_total + 1.0e-12 >= minimum_gain
        ]
        if not eligible_slots:
            stop_reason = "minimum_gain"
            break
        best_slot = max(
            eligible_slots,
            key=lambda slot: (
                int(host_newly_satisfied[slot]),
                int(host_progress[slot]),
                -remaining[slot],
            ),
        )
        best = remaining[best_slot]
        gain = int(host_progress[best_slot])
        selected.append(best)
        marginal_counts.append(gain)
        wp.launch(
            kernels.add_selected_mask,
            dim=cell_count,
            inputs=[
                packed,
                wp_accessible,
                best,
                word_count,
                counts,
            ],
            device=resolved_device,
        )
        wp.synchronize_device(resolved_device)
        check_cancelled()
        remaining.pop(best_slot)
    else:
        check_cancelled()
        covered_total = wp.zeros(1, dtype=wp.int32, device=resolved_device)
        wp.launch(
            kernels.count_covered_cells,
            dim=cell_count,
            inputs=[wp_accessible, counts, int(per_cell), covered_total],
            device=resolved_device,
        )
        wp.synchronize_device(resolved_device)
        check_cancelled()
        covered_count = int(covered_total.numpy()[0])
        stop_reason = (
            "target_met"
            if covered_count / accessible_total + 1.0e-12 >= target_coverage
            else "camera_limit"
        )

    covered = wp.empty(cell_count, dtype=wp.uint8, device=resolved_device)
    check_cancelled()
    wp.launch(
        kernels.materialize_covered_mask,
        dim=cell_count,
        inputs=[wp_accessible, counts, int(per_cell), covered],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)
    check_cancelled()
    host_counts = counts.numpy().astype(np.int16, copy=False).reshape(cell_shape)
    host_covered = (covered.numpy() != 0).reshape(cell_shape)
    return GreedyMaskScores(
        selected=tuple(selected),
        marginal_counts=tuple(marginal_counts),
        counts=host_counts,
        covered=host_covered,
        accessible_count=accessible_total,
        covered_count=covered_count,
        visible_counts=visible_counts,
        stop_reason=stop_reason,
    )
