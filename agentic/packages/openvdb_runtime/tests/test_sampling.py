# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from openvdb_runtime import (
    ExecutionLimits,
    InvalidGeometryError,
    NativeOperationError,
    ResourceLimitError,
    sample_gradients,
    sample_values,
)

POINTS = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float64)


def _call(module, name):
    return next(call for call in module.calls if call[0] == name)


def test_sample_values_canonicalizes_world_points(fake_openvdb):
    grid = fake_openvdb.FloatGrid()

    result = sample_values(grid, POINTS, interpolation="nearest", thread_count=2)

    np.testing.assert_array_equal(result, [-0.5, 0.25])
    assert result.dtype == np.float32
    _, args, kwargs = _call(fake_openvdb, "sample_values")
    assert args[0] is grid
    assert args[1].dtype == np.float32
    assert args[1].flags.c_contiguous
    assert kwargs == {"interpolation": "nearest", "thread_count": 2}


def test_sample_values_copies_unaligned_world_points(fake_openvdb):
    storage = np.zeros(2 * 3 * 4 + 1, dtype=np.uint8)
    points = np.ndarray((2, 3), dtype=np.float32, buffer=storage, offset=1)
    points[:] = ((0, 0, 0), (1, 2, 3))
    assert not points.flags.aligned

    sample_values(fake_openvdb.FloatGrid(), points)

    _, args, _kwargs = _call(fake_openvdb, "sample_values")
    native_points = args[1]
    assert native_points.flags.aligned and native_points.flags.owndata
    assert not np.shares_memory(native_points, points)


def test_sample_gradients_returns_n_by_three_float_array(fake_openvdb):
    result = sample_gradients(fake_openvdb.FloatGrid(), POINTS, interpolation="linear")

    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 1, 0]])
    assert result.shape == (2, 3)
    _, _, kwargs = _call(fake_openvdb, "sample_gradients")
    assert kwargs == {"interpolation": "linear", "thread_count": 1}


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.zeros((2, 2)), "shape"),
        (np.array([[0, 0, np.nan]]), "finite"),
        (np.array([["x", "y", "z"]]), "numeric"),
        (np.array([[0.0 + 1.0j, 0.0, 0.0]]), "real numeric"),
    ],
)
def test_sampling_rejects_invalid_points(fake_openvdb, points, message):
    with pytest.raises(InvalidGeometryError, match=message):
        sample_values(fake_openvdb.FloatGrid(), points)


def test_sampling_checks_batch_limit_before_native_call(fake_openvdb):
    with pytest.raises(ResourceLimitError, match="sample batch"):
        sample_values(
            fake_openvdb.FloatGrid(),
            POINTS,
            limits=ExecutionLimits(max_sample_points=1),
        )

    assert not any(call[0] == "sample_values" for call in fake_openvdb.calls)


def test_sampling_checks_batch_limit_before_array_conversion(fake_openvdb):
    class OversizedPoints:
        def __len__(self) -> int:
            return 2

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("oversized sample batch must not be copied")

    with pytest.raises(ResourceLimitError, match="sample batch"):
        sample_values(
            fake_openvdb.FloatGrid(),
            OversizedPoints(),
            limits=ExecutionLimits(max_sample_points=1),
        )


def test_sampling_rejects_deceptive_array_like_without_materializing(fake_openvdb):
    class DeceptivePoints:
        def __len__(self) -> int:
            return 1

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("untrusted __array__ must not be invoked")

    with pytest.raises(InvalidGeometryError, match="NumPy array"):
        sample_values(fake_openvdb.FloatGrid(), DeceptivePoints())

    assert not any(call[0] == "sample_values" for call in fake_openvdb.calls)


def test_sampling_rejects_oversized_ragged_rows_before_inspection(fake_openvdb):
    class UntouchedRow:
        def __len__(self) -> int:
            raise AssertionError("oversized rows must not be inspected")

    with pytest.raises(ResourceLimitError, match="2 points"):
        sample_values(
            fake_openvdb.FloatGrid(),
            [UntouchedRow(), UntouchedRow()],
            limits=ExecutionLimits(max_sample_points=1),
        )

    assert not any(call[0] == "sample_values" for call in fake_openvdb.calls)


@pytest.mark.parametrize(
    ("operation", "tool_name"),
    [(sample_values, "sample_values"), (sample_gradients, "sample_gradients")],
)
def test_sampling_checks_grid_limits_before_native_call(fake_openvdb, operation, tool_name):
    grid = fake_openvdb.FloatGrid(active_voxels=2)

    with pytest.raises(ResourceLimitError, match="active voxels"):
        operation(
            grid,
            POINTS,
            limits=ExecutionLimits(max_active_voxels=1),
        )

    assert not any(call[0] == tool_name for call in fake_openvdb.calls)


def test_sampling_validates_native_result_shape(fake_openvdb):
    fake_openvdb.tools.sample_values = lambda *args, **kwargs: np.zeros((2, 1))

    with pytest.raises(InvalidGeometryError, match="returned shape"):
        sample_values(fake_openvdb.FloatGrid(), POINTS)


def test_sampling_validates_native_result_finiteness(fake_openvdb):
    fake_openvdb.tools.sample_gradients = lambda *args, **kwargs: np.full((2, 3), np.nan)

    with pytest.raises(InvalidGeometryError, match="nonfinite"):
        sample_gradients(fake_openvdb.FloatGrid(), POINTS)


def test_sampling_wraps_native_binding_errors(fake_openvdb):
    def fail(*args, **kwargs):
        raise ValueError("native hard limit")

    fake_openvdb.tools.sample_values = fail

    with pytest.raises(NativeOperationError, match="sample_values.*native hard limit"):
        sample_values(fake_openvdb.FloatGrid(), POINTS)


def test_sampling_validates_native_result_type(fake_openvdb):
    fake_openvdb.tools.sample_values = lambda *args, **kwargs: object()

    with pytest.raises(InvalidGeometryError, match="nonnumeric"):
        sample_values(fake_openvdb.FloatGrid(), POINTS)
