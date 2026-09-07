# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for camera mask packing and coverage scoring.

This module intentionally imports Warp at module scope.  It is imported only by
``warp_scoring._lazy_runtime()`` after a camera-analysis operation has selected a
Newton/Warp device; ordinary usd-cli imports and help never load this module.
"""

from __future__ import annotations

import warp as wp

WORD_BITS = 32


@wp.kernel
def pack_mask_bits(
    masks: wp.array(dtype=wp.uint8),
    cell_count: int,
    word_count: int,
    packed: wp.array(dtype=wp.uint32),
) -> None:
    packed_index = wp.tid()
    camera_index = packed_index // word_count
    word_index = packed_index - camera_index * word_count
    first_cell = word_index * WORD_BITS
    value = wp.uint32(0)
    for offset in range(WORD_BITS):
        cell_index = first_cell + offset
        if cell_index < cell_count:
            if masks[camera_index * cell_count + cell_index] != wp.uint8(0):
                value = value | (wp.uint32(1) << wp.uint32(offset))
    packed[packed_index] = value


@wp.kernel
def reduce_coverage(
    packed: wp.array(dtype=wp.uint32),
    accessible: wp.array(dtype=wp.uint8),
    camera_count: int,
    cell_count: int,
    word_count: int,
    per_cell: int,
    counts: wp.array(dtype=wp.int32),
    covered: wp.array(dtype=wp.uint8),
    histogram: wp.array(dtype=wp.int32),
    totals: wp.array(dtype=wp.int32),
) -> None:
    cell_index = wp.tid()
    counts[cell_index] = 0
    covered[cell_index] = wp.uint8(0)
    word_index = cell_index // WORD_BITS
    bit_index = wp.uint32(cell_index - word_index * WORD_BITS)
    count = int(0)
    for camera_index in range(camera_count):
        word = packed[camera_index * word_count + word_index]
        count += int((word >> bit_index) & wp.uint32(1))

    counts[cell_index] = count
    if accessible[cell_index] == wp.uint8(0):
        return
    wp.atomic_add(histogram, count, 1)
    wp.atomic_add(totals, 0, 1)
    if count >= per_cell:
        covered[cell_index] = wp.uint8(1)
        wp.atomic_add(totals, 1, 1)


@wp.kernel
def reduce_camera_contributions(
    packed: wp.array(dtype=wp.uint32),
    accessible: wp.array(dtype=wp.uint8),
    counts: wp.array(dtype=wp.int32),
    cell_count: int,
    word_count: int,
    per_cell: int,
    visible_counts: wp.array(dtype=wp.int32),
    marginal_counts: wp.array(dtype=wp.int32),
) -> None:
    index = wp.tid()
    camera_index = index // cell_count
    cell_index = index - camera_index * cell_count
    if accessible[cell_index] == wp.uint8(0):
        return

    word_index = cell_index // WORD_BITS
    bit_index = wp.uint32(cell_index - word_index * WORD_BITS)
    word = packed[camera_index * word_count + word_index]
    if ((word >> bit_index) & wp.uint32(1)) == wp.uint32(0):
        return

    wp.atomic_add(visible_counts, camera_index, 1)
    # Removing this camera makes a currently covered cell fail exactly when the
    # cell has the requested redundancy and this camera contributes one view.
    if counts[cell_index] == per_cell:
        wp.atomic_add(marginal_counts, camera_index, 1)


@wp.kernel
def count_covered_cells(
    accessible: wp.array(dtype=wp.uint8),
    counts: wp.array(dtype=wp.int32),
    per_cell: int,
    covered_total: wp.array(dtype=wp.int32),
) -> None:
    cell_index = wp.tid()
    if accessible[cell_index] != wp.uint8(0) and counts[cell_index] >= per_cell:
        wp.atomic_add(covered_total, 0, 1)


@wp.kernel
def count_accessible_cells(
    accessible: wp.array(dtype=wp.uint8),
    accessible_total: wp.array(dtype=wp.int32),
) -> None:
    cell_index = wp.tid()
    if accessible[cell_index] != wp.uint8(0):
        wp.atomic_add(accessible_total, 0, 1)


@wp.kernel
def score_remaining_gains(
    packed: wp.array(dtype=wp.uint32),
    accessible: wp.array(dtype=wp.uint8),
    counts: wp.array(dtype=wp.int32),
    remaining: wp.array(dtype=wp.int32),
    cell_count: int,
    word_count: int,
    per_cell: int,
    newly_satisfied_gains: wp.array(dtype=wp.int32),
    progress_gains: wp.array(dtype=wp.int32),
) -> None:
    index = wp.tid()
    remaining_index = index // cell_count
    cell_index = index - remaining_index * cell_count
    if accessible[cell_index] == wp.uint8(0) or counts[cell_index] >= per_cell:
        return

    camera_index = remaining[remaining_index]
    word_index = cell_index // WORD_BITS
    bit_index = wp.uint32(cell_index - word_index * WORD_BITS)
    word = packed[camera_index * word_count + word_index]
    if ((word >> bit_index) & wp.uint32(1)) != wp.uint32(0):
        # Progress counts every still-unmet cell advanced by this camera. Newly
        # satisfied is the stricter primary objective: cells that this camera
        # takes across the requested redundancy threshold on this iteration.
        wp.atomic_add(progress_gains, remaining_index, 1)
        if counts[cell_index] == per_cell - 1:
            wp.atomic_add(newly_satisfied_gains, remaining_index, 1)


@wp.kernel
def add_selected_mask(
    packed: wp.array(dtype=wp.uint32),
    accessible: wp.array(dtype=wp.uint8),
    camera_index: int,
    word_count: int,
    counts: wp.array(dtype=wp.int32),
) -> None:
    cell_index = wp.tid()
    if accessible[cell_index] == wp.uint8(0):
        return
    word_index = cell_index // WORD_BITS
    bit_index = wp.uint32(cell_index - word_index * WORD_BITS)
    word = packed[camera_index * word_count + word_index]
    if ((word >> bit_index) & wp.uint32(1)) != wp.uint32(0):
        counts[cell_index] += 1


@wp.kernel
def materialize_covered_mask(
    accessible: wp.array(dtype=wp.uint8),
    counts: wp.array(dtype=wp.int32),
    per_cell: int,
    covered: wp.array(dtype=wp.uint8),
) -> None:
    cell_index = wp.tid()
    covered[cell_index] = wp.uint8(0)
    if accessible[cell_index] != wp.uint8(0) and counts[cell_index] >= per_cell:
        covered[cell_index] = wp.uint8(1)
