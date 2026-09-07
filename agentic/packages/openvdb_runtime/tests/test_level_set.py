# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openvdb_runtime import (
    ExecutionLimits,
    ResourceLimitError,
    difference,
    intersection,
    mean_filter,
    normalize,
    offset,
    rebuild,
    resample_to_match,
    scalar_mean_filter,
    union,
)


@pytest.mark.parametrize(
    ("operation", "call_name", "result_label"),
    [
        (union, "csg_union", "union"),
        (intersection, "csg_intersection", "intersection"),
        (difference, "csg_difference", "difference"),
    ],
)
def test_csg_operations_delegate_to_copy_producing_tools(
    fake_openvdb, operation, call_name, result_label
):
    left = fake_openvdb.FloatGrid("left")
    right = fake_openvdb.FloatGrid("right")

    result = operation(left, right, thread_count=4)

    assert result.label == result_label
    call = next(call for call in fake_openvdb.calls if call[0] == call_name)
    assert call[1] == (left, right)
    assert call[2] == {"thread_count": 4, "max_active_voxels": 50_000_000}


def test_csg_threads_caller_active_voxel_limit_into_native_tool(fake_openvdb):
    left = fake_openvdb.FloatGrid("left", active_voxels=8)
    right = fake_openvdb.FloatGrid("right", active_voxels=8)

    union(left, right, limits=ExecutionLimits(max_active_voxels=16))

    call = next(call for call in fake_openvdb.calls if call[0] == "csg_union")
    assert call[2]["max_active_voxels"] == 16


def test_offset_delegates_signed_world_distance(fake_openvdb):
    grid = fake_openvdb.FloatGrid()

    result = offset(grid, -0.25, thread_count=2)

    assert result.label == "offset"
    call = next(call for call in fake_openvdb.calls if call[0] == "level_set_offset")
    assert call[1] == (grid, -0.25)
    assert call[2] == {"thread_count": 2}


def test_mean_filter_validates_and_delegates(fake_openvdb):
    grid = fake_openvdb.FloatGrid()

    result = mean_filter(grid, width=2, iterations=3, thread_count=4)

    assert result.label == "mean"
    call = next(call for call in fake_openvdb.calls if call[0] == "level_set_mean")
    assert call[2] == {"width": 2, "iterations": 3, "thread_count": 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0}, "width"),
        ({"width": 1.5}, "width"),
        ({"iterations": False}, "iterations"),
        ({"iterations": -1}, "iterations"),
    ],
)
def test_mean_filter_rejects_invalid_controls(fake_openvdb, kwargs, message):
    with pytest.raises(ValueError, match=message):
        mean_filter(fake_openvdb.FloatGrid(), **kwargs)


def test_scalar_mean_filter_accepts_non_level_set_float_grid(fake_openvdb):
    unsigned_distance = fake_openvdb.FloatGrid("unsigned-grid")

    result = scalar_mean_filter(
        unsigned_distance,
        width=2,
        iterations=3,
        thread_count=4,
    )

    assert result.label == "scalar-mean"
    call = next(call for call in fake_openvdb.calls if call[0] == "scalar_mean")
    assert call[1] == (unsigned_distance,)
    assert call[2] == {"width": 2, "iterations": 3, "thread_count": 4}


def test_scalar_mean_filter_enforces_filter_limits(fake_openvdb):
    with pytest.raises(ResourceLimitError, match="width"):
        scalar_mean_filter(
            fake_openvdb.FloatGrid(),
            width=3,
            limits=ExecutionLimits(max_filter_width=2),
        )
    with pytest.raises(ResourceLimitError, match="iterations"):
        scalar_mean_filter(
            fake_openvdb.FloatGrid(),
            iterations=3,
            limits=ExecutionLimits(max_filter_iterations=2),
        )


@pytest.mark.parametrize("operation", [mean_filter, scalar_mean_filter])
def test_mean_filters_enforce_aggregate_work_limit(fake_openvdb, operation):
    grid = fake_openvdb.FloatGrid(active_voxels=1_000_000)

    with pytest.raises(ResourceLimitError, match="work units"):
        operation(grid, width=256, iterations=1024)


def test_mean_filter_accepts_a_stricter_caller_work_limit(fake_openvdb):
    with pytest.raises(ResourceLimitError, match="limit is 10"):
        mean_filter(
            fake_openvdb.FloatGrid(active_voxels=8),
            limits=ExecutionLimits(max_filter_work=10),
        )


def test_offset_rejects_nonfinite_distance(fake_openvdb):
    with pytest.raises(ValueError, match="finite"):
        offset(fake_openvdb.FloatGrid(), float("nan"))


def test_offset_checks_voxel_distance_before_native_call(fake_openvdb):
    grid = fake_openvdb.FloatGrid()
    grid.transform = SimpleNamespace(voxelSize=lambda: (0.1, 0.1, 0.1))

    with pytest.raises(ResourceLimitError, match="max_band_width_voxels"):
        offset(
            grid,
            1.0,
            limits=ExecutionLimits(max_band_width_voxels=8.0),
        )

    assert not any(call[0] == "level_set_offset" for call in fake_openvdb.calls)


@pytest.mark.parametrize(
    ("operation", "kwargs", "label"),
    [
        (offset, {"distance": True}, "distance"),
        (rebuild, {"isovalue": False}, "isovalue"),
        (rebuild, {"exterior_width": True}, "exterior_width"),
    ],
)
def test_level_set_numeric_controls_reject_booleans(fake_openvdb, operation, kwargs, label):
    with pytest.raises(TypeError, match=rf"{label} must be a real number"):
        operation(fake_openvdb.FloatGrid(), **kwargs)


def test_normalize_is_copy_producing_native_call(fake_openvdb):
    source = fake_openvdb.FloatGrid("source")

    result = normalize(source, thread_count=2)

    assert result.label == "normalized"
    call = next(call for call in fake_openvdb.calls if call[0] == "level_set_normalize")
    assert call[1] == (source,)
    assert call[2] == {"thread_count": 2}


def test_rebuild_exposes_isovalue_and_band_widths(fake_openvdb):
    source = fake_openvdb.FloatGrid("source")

    result = rebuild(
        source,
        isovalue=0.25,
        exterior_width=4.0,
        interior_width=5.0,
        thread_count=3,
    )

    assert result.label == "rebuilt"
    call = next(call for call in fake_openvdb.calls if call[0] == "level_set_rebuild")
    assert call[1] == (source,)
    assert call[2] == {
        "isovalue": 0.25,
        "exterior_width": 4.0,
        "interior_width": 5.0,
        "thread_count": 3,
    }


def test_rebuild_rejects_nonfinite_or_oversized_parameters(fake_openvdb):
    grid = fake_openvdb.FloatGrid()
    with pytest.raises(ValueError, match="isovalue"):
        rebuild(grid, isovalue=float("nan"))
    with pytest.raises(ResourceLimitError, match="exterior_width"):
        rebuild(
            grid,
            exterior_width=9.0,
            limits=ExecutionLimits(max_band_width_voxels=8.0),
        )


def test_rebuild_checks_float32_background_range(fake_openvdb):
    grid = fake_openvdb.FloatGrid()
    grid.transform = SimpleNamespace(voxelSize=lambda: (1e38, 1e38, 1e38))

    with pytest.raises(ResourceLimitError, match="float32 background range"):
        rebuild(grid, exterior_width=4.0, interior_width=4.0)

    assert not any(call[0] == "level_set_rebuild" for call in fake_openvdb.calls)


def test_resample_to_match_has_explicit_interpolation(fake_openvdb):
    source = fake_openvdb.FloatGrid("source")
    reference = fake_openvdb.FloatGrid("reference")

    result = resample_to_match(source, reference, interpolation="linear", thread_count=2)

    assert result.label == "resampled"
    call = next(call for call in fake_openvdb.calls if call[0] == "resample_to_match")
    assert call[1] == (source, reference)
    assert call[2] == {"interpolation": "linear", "thread_count": 2}


def test_resample_rejects_unknown_interpolation(fake_openvdb):
    with pytest.raises(ValueError, match="nearest, linear, quadratic"):
        resample_to_match(
            fake_openvdb.FloatGrid(),
            fake_openvdb.FloatGrid(),
            interpolation="cubic",
        )


def test_resample_checks_reference_grid_extent_before_native_call(fake_openvdb):
    reference = fake_openvdb.FloatGrid(active_dimensions=(11, 4, 4))

    with pytest.raises(ResourceLimitError, match="active dimensions"):
        resample_to_match(
            fake_openvdb.FloatGrid(),
            reference,
            limits=ExecutionLimits(max_voxel_extent=10),
        )

    assert not any(call[0] == "resample_to_match" for call in fake_openvdb.calls)


def test_resample_checks_transformed_output_domain_before_native_call(fake_openvdb):
    source = fake_openvdb.FloatGrid(active_voxels=1, active_dimensions=(1, 1, 1))
    source.evalActiveVoxelBoundingBox = lambda: ((0, 0, 0), (0, 0, 0))
    source.transform = SimpleNamespace(indexToWorld=lambda point: point)
    reference = fake_openvdb.FloatGrid(active_voxels=0, active_dimensions=(0, 0, 0))
    reference.transform = SimpleNamespace(
        worldToIndex=lambda point: tuple(value * 1_000 for value in point)
    )

    with pytest.raises(ResourceLimitError, match="active voxels limit"):
        resample_to_match(source, reference, interpolation="nearest")

    assert not any(call[0] == "resample_to_match" for call in fake_openvdb.calls)
