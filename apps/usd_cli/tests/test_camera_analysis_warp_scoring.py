# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic Warp coverage-bitset and marginal-gain scoring tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from usd_core.camera_analysis.warp_scoring import (
    greedy_select_masks,
    score_coverage_masks,
)


def _mask_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Five masks crossing 32-bit word boundaries, including an exact tie."""

    cell_count = 70
    flat = np.zeros((5, cell_count), dtype=bool)
    flat[0, :26] = True
    flat[1] = flat[0]  # tie: the lower candidate index must win
    flat[2, 20:51] = True
    flat[3, 45:] = True
    flat[4, ::2] = True
    # Exercise both sides of every packed-word boundary explicitly.
    flat[:, [31, 32, 63, 64]] ^= np.asarray([[1], [1], [0], [1], [0]], dtype=bool)
    accessible = np.ones(cell_count, dtype=bool)
    accessible[::11] = False
    flat[:, ~accessible] = False
    return flat.reshape(5, 7, 10), accessible.reshape(7, 10)


def _coverage_oracle(masks: np.ndarray, accessible: np.ndarray, per_cell: int) -> dict:
    counts = masks.sum(axis=0, dtype=np.int16)
    covered = accessible & (counts >= per_cell)
    accessible_counts = counts[accessible]
    histogram = {
        int(value): int(np.count_nonzero(accessible_counts == value))
        for value in np.unique(accessible_counts)
    }
    visible = np.count_nonzero(masks & accessible[None, ...], axis=(1, 2))
    marginal = np.asarray(
        [np.count_nonzero(covered & mask & (counts == per_cell)) for mask in masks],
        dtype=np.int64,
    )
    return {
        "counts": counts,
        "covered": covered,
        "accessible_count": int(accessible.sum()),
        "covered_count": int(covered.sum()),
        "histogram": histogram,
        "visible": visible,
        "marginal": marginal,
    }


def _greedy_oracle(
    masks: np.ndarray,
    accessible: np.ndarray,
    *,
    per_cell: int,
    max_cameras: int,
    target_coverage: float,
    minimum_gain: float,
) -> dict:
    counts = np.zeros(accessible.shape, dtype=np.int16)
    accessible_count = int(accessible.sum())
    remaining = list(range(len(masks)))
    selected: list[int] = []
    marginal: list[int] = []
    stop_reason = "camera_limit"
    while len(selected) < max_cameras:
        covered = accessible & (counts >= per_cell)
        if covered.sum() / accessible_count + 1.0e-12 >= target_coverage:
            stop_reason = "target_met"
            break
        if not remaining:
            stop_reason = "no_valid_candidate"
            break
        unmet = accessible & (counts < per_cell)
        threshold = accessible & (counts == per_cell - 1)
        progress = [int(np.count_nonzero(masks[index] & unmet)) for index in remaining]
        newly_satisfied = [
            int(np.count_nonzero(masks[index] & threshold)) for index in remaining
        ]
        positive_slots = [index for index, gain in enumerate(progress) if gain > 0]
        if not positive_slots:
            stop_reason = "no_valid_candidate"
            break
        eligible_slots = [
            index
            for index in positive_slots
            if progress[index] / accessible_count + 1.0e-12 >= minimum_gain
        ]
        if not eligible_slots:
            stop_reason = "minimum_gain"
            break
        slot = max(
            eligible_slots,
            key=lambda index: (
                newly_satisfied[index],
                progress[index],
                -remaining[index],
            ),
        )
        gain = progress[slot]
        chosen = remaining.pop(slot)
        selected.append(chosen)
        marginal.append(gain)
        counts += masks[chosen]
    else:
        covered = accessible & (counts >= per_cell)
        stop_reason = (
            "target_met"
            if covered.sum() / accessible_count + 1.0e-12 >= target_coverage
            else "camera_limit"
        )
    covered = accessible & (counts >= per_cell)
    return {
        "selected": tuple(selected),
        "marginal": tuple(marginal),
        "counts": counts,
        "covered": covered,
        "covered_count": int(covered.sum()),
        "stop_reason": stop_reason,
    }


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_warp_cpu_coverage_scores_match_numpy_oracle_exactly() -> None:
    masks, accessible = _mask_fixture()
    expected = _coverage_oracle(masks, accessible, per_cell=2)

    actual = score_coverage_masks(masks, accessible, per_cell=2, device="cpu")

    assert actual.counts.dtype == np.int16
    assert np.array_equal(actual.counts, expected["counts"])
    assert np.array_equal(actual.covered, expected["covered"])
    assert actual.accessible_count == expected["accessible_count"]
    assert actual.covered_count == expected["covered_count"]
    assert actual.histogram == expected["histogram"]
    assert np.array_equal(actual.visible_counts, expected["visible"])
    assert np.array_equal(actual.marginal_counts, expected["marginal"])


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_warp_cpu_greedy_scoring_is_exact_deterministic_and_stable_on_ties() -> None:
    masks, accessible = _mask_fixture()
    options = {
        "per_cell": 1,
        "max_cameras": 4,
        "target_coverage": 0.95,
        "minimum_gain": 0.0,
    }
    expected = _greedy_oracle(masks, accessible, **options)

    first = greedy_select_masks(masks, accessible, device="cpu", **options)
    second = greedy_select_masks(masks, accessible, device="cpu", **options)

    assert first.selected == expected["selected"]
    assert second.selected == first.selected
    assert 0 in first.selected
    assert 1 not in first.selected
    assert first.marginal_counts == expected["marginal"]
    assert second.marginal_counts == first.marginal_counts
    assert np.array_equal(first.counts, expected["counts"])
    assert np.array_equal(second.counts, first.counts)
    assert np.array_equal(first.covered, expected["covered"])
    assert np.array_equal(second.covered, first.covered)
    assert first.covered_count == expected["covered_count"]
    assert first.stop_reason == expected["stop_reason"]


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_redundant_greedy_prefers_newly_satisfied_cells_over_raw_progress() -> None:
    accessible = np.ones(200, dtype=bool)
    masks = np.zeros((3, 200), dtype=bool)
    masks[0, :100] = True
    masks[1, :50] = True
    masks[2, 100:] = True
    options = {
        "per_cell": 2,
        "max_cameras": 2,
        "target_coverage": 0.25,
        "minimum_gain": 0.0,
    }

    expected = _greedy_oracle(masks, accessible, **options)
    actual = greedy_select_masks(masks, accessible, device="cpu", **options)

    assert expected["selected"] == (0, 1)
    assert actual.selected == expected["selected"]
    assert actual.marginal_counts == (100, 50)
    assert actual.covered_count == 50
    assert actual.stop_reason == "target_met"


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_minimum_gain_filters_on_progress_before_redundancy_score() -> None:
    accessible = np.ones(200, dtype=bool)
    masks = np.zeros((3, 200), dtype=bool)
    masks[0, :100] = True
    masks[1, :50] = True
    masks[2, 100:] = True
    options = {
        "per_cell": 2,
        "max_cameras": 2,
        "target_coverage": 0.25,
        "minimum_gain": 0.30,
    }

    expected = _greedy_oracle(masks, accessible, **options)
    actual = greedy_select_masks(masks, accessible, device="cpu", **options)

    # Candidate 1 would satisfy 50 cells but advances only 25% of the grid, so
    # it is ineligible. Candidate 2 still clears the 30% progress threshold.
    assert expected["selected"] == (0, 2)
    assert actual.selected == expected["selected"]
    assert actual.marginal_counts == (100, 100)
    assert actual.covered_count == 0
    assert actual.stop_reason == "camera_limit"

    stopped = greedy_select_masks(
        masks,
        accessible,
        device="cpu",
        **{**options, "minimum_gain": 0.51},
    )
    assert stopped.selected == ()
    assert stopped.stop_reason == "minimum_gain"


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_redundant_greedy_breaks_full_score_ties_by_lowest_input_index() -> None:
    accessible = np.ones(8, dtype=bool)
    masks = np.zeros((3, 8), dtype=bool)
    masks[0, :4] = True
    masks[1, :4] = True
    masks[2, :4] = True
    options = {
        "per_cell": 2,
        "max_cameras": 2,
        "target_coverage": 0.5,
        "minimum_gain": 0.0,
    }

    actual = greedy_select_masks(masks, accessible, device="cpu", **options)

    assert actual.selected == (0, 1)
    assert actual.covered_count == 4


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_impossible_redundancy_and_large_camera_cap_preserve_public_behavior() -> None:
    masks, accessible = _mask_fixture()
    per_cell = len(masks) + 1

    coverage = score_coverage_masks(masks, accessible, per_cell=per_cell, device="cpu")
    assert coverage.covered_count == 0
    assert not coverage.covered.any()

    options = {
        "per_cell": per_cell,
        "max_cameras": len(masks) + 3,
        "target_coverage": 0.5,
        "minimum_gain": 0.0,
    }
    expected = _greedy_oracle(masks, accessible, **options)
    actual = greedy_select_masks(masks, accessible, device="cpu", **options)

    assert actual.selected == expected["selected"]
    assert actual.marginal_counts == expected["marginal"]
    assert np.array_equal(actual.counts, expected["counts"])
    assert not actual.covered.any()
    assert actual.covered_count == 0
    assert actual.stop_reason == "no_valid_candidate"


@pytest.mark.usefixtures("qualified_warp_runtime")
def test_warp_cuda_uses_the_same_scoring_contract_when_available() -> None:
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("Warp CUDA device is unavailable")
    masks, accessible = _mask_fixture()
    cpu = score_coverage_masks(masks, accessible, per_cell=2, device="cpu")
    cuda = score_coverage_masks(masks, accessible, per_cell=2, device="cuda:0")

    assert np.array_equal(cuda.counts, cpu.counts)
    assert np.array_equal(cuda.covered, cpu.covered)
    assert cuda.histogram == cpu.histogram
    assert np.array_equal(cuda.visible_counts, cpu.visible_counts)
    assert np.array_equal(cuda.marginal_counts, cpu.marginal_counts)

    options = {
        "per_cell": 1,
        "max_cameras": 4,
        "target_coverage": 0.95,
        "minimum_gain": 0.0,
    }
    cpu_greedy = greedy_select_masks(masks, accessible, device="cpu", **options)
    cuda_greedy = greedy_select_masks(masks, accessible, device="cuda:0", **options)
    assert cuda_greedy.selected == cpu_greedy.selected
    assert cuda_greedy.marginal_counts == cpu_greedy.marginal_counts
    assert np.array_equal(cuda_greedy.counts, cpu_greedy.counts)
    assert np.array_equal(cuda_greedy.covered, cpu_greedy.covered)

    redundant_accessible = np.ones(200, dtype=bool)
    redundant_masks = np.zeros((3, 200), dtype=bool)
    redundant_masks[0, :100] = True
    redundant_masks[1, :50] = True
    redundant_masks[2, 100:] = True
    redundant_options = {
        "per_cell": 2,
        "max_cameras": 2,
        "target_coverage": 0.25,
        "minimum_gain": 0.0,
    }
    cpu_redundant = greedy_select_masks(
        redundant_masks,
        redundant_accessible,
        device="cpu",
        **redundant_options,
    )
    cuda_redundant = greedy_select_masks(
        redundant_masks,
        redundant_accessible,
        device="cuda:0",
        **redundant_options,
    )
    assert cuda_redundant.selected == cpu_redundant.selected == (0, 1)
    assert cuda_redundant.marginal_counts == cpu_redundant.marginal_counts
    assert cuda_redundant.covered_count == cpu_redundant.covered_count == 50


def test_importing_scoring_contract_does_not_import_or_initialize_warp() -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(source), environment.get("PYTHONPATH", "")) if item
    )
    code = (
        "import sys; import usd_core.camera_analysis.warp_scoring; "
        "assert 'warp' not in sys.modules; "
        "assert 'usd_core.camera_analysis._warp_scoring_kernels' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], env=environment, check=True)
