# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openvdb_runtime import (
    ExecutionLimits,
    InvalidGeometryError,
    ResourceLimitError,
    active_value_mask,
    extract_enclosed_region,
    topology_to_level_set,
)


def _call(module, name):
    return next(call for call in module.calls if call[0] == name)


def test_active_value_mask_exposes_inclusive_value_window(fake_openvdb):
    source = fake_openvdb.FloatGrid("distance")

    result = active_value_mask(
        source,
        min_value=0.0,
        max_value=0.25,
        thread_count=2,
    )

    assert result.label == "mask"
    call = _call(fake_openvdb, "active_value_mask")
    assert call[1] == (source,)
    assert call[2] == {"min_value": 0.0, "max_value": 0.25, "thread_count": 2}


def test_active_value_mask_rejects_invalid_window(fake_openvdb):
    with pytest.raises(ValueError, match="must not exceed"):
        active_value_mask(fake_openvdb.FloatGrid(), min_value=2.0, max_value=1.0)
    with pytest.raises(ValueError, match="finite"):
        active_value_mask(fake_openvdb.FloatGrid(), max_value=float("inf"))


@pytest.mark.parametrize("kwargs", [{"min_value": True}, {"max_value": False}])
def test_active_value_mask_rejects_boolean_bounds(fake_openvdb, kwargs):
    with pytest.raises(TypeError, match="must be a real number"):
        active_value_mask(fake_openvdb.FloatGrid(), **kwargs)


def test_topology_to_level_set_exposes_morphology_steps(fake_openvdb):
    mask = fake_openvdb.FloatGrid("mask")

    result = topology_to_level_set(
        mask,
        half_width=4,
        closing_steps=2,
        dilation=1,
        smoothing_steps=3,
        thread_count=2,
    )

    assert result.label == "topology-level-set"
    call = _call(fake_openvdb, "topology_to_level_set")
    assert call[1] == (mask,)
    assert call[2] == {
        "half_width": 4,
        "closing_steps": 2,
        "dilation": 1,
        "smoothing_steps": 3,
        "thread_count": 2,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"half_width": 0}, "half_width"),
        ({"half_width": 1.5}, "half_width"),
        ({"closing_steps": -1}, "closing_steps"),
        ({"dilation": False}, "dilation"),
        ({"smoothing_steps": 1.5}, "smoothing_steps"),
    ],
)
def test_topology_to_level_set_rejects_invalid_controls(fake_openvdb, kwargs, message):
    with pytest.raises(ValueError, match=message):
        topology_to_level_set(fake_openvdb.FloatGrid(), **kwargs)


def test_topology_step_limit_is_enforced(fake_openvdb):
    with pytest.raises(ResourceLimitError, match="max_topology_steps"):
        topology_to_level_set(
            fake_openvdb.FloatGrid(),
            closing_steps=3,
            limits=ExecutionLimits(max_topology_steps=2),
        )


def test_topology_checks_float32_background_range(fake_openvdb):
    grid = fake_openvdb.FloatGrid()
    grid.transform = SimpleNamespace(voxelSize=lambda: (1e38, 1e38, 1e38))

    with pytest.raises(ResourceLimitError, match="float32 background range"):
        topology_to_level_set(grid, half_width=4)

    assert not any(call[0] == "topology_to_level_set" for call in fake_openvdb.calls)


def test_topology_rejects_int32_boundary_before_native_call(fake_openvdb):
    grid = fake_openvdb.FloatGrid()
    grid.evalActiveVoxelBoundingBox = lambda: ((2**31 - 1, 0, 0), (2**31 - 1, 0, 0))

    with pytest.raises(ResourceLimitError, match="int32 index boundary"):
        topology_to_level_set(grid, dilation=1)

    assert not any(call[0] == "topology_to_level_set" for call in fake_openvdb.calls)


def test_nonempty_wrapped_active_bbox_is_rejected(fake_openvdb):
    grid = fake_openvdb.FloatGrid(active_voxels=1)
    grid.evalActiveVoxelBoundingBox = lambda: ((2**31 - 1, 0, 0), (-(2**31), 0, 0))

    with pytest.raises(InvalidGeometryError, match="empty active bounding box"):
        active_value_mask(grid)

    assert not any(call[0] == "active_value_mask" for call in fake_openvdb.calls)


def test_extract_enclosed_region_is_copy_producing(fake_openvdb):
    level_set = fake_openvdb.FloatGrid("level-set")

    result = extract_enclosed_region(level_set, thread_count=2)

    assert result is not level_set
    assert result.label == "enclosed"
    assert level_set.label == "level-set"
    call = _call(fake_openvdb, "extract_enclosed_region")
    assert call[1] == (level_set,)
    assert call[2] == {"thread_count": 2}
