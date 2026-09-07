# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy-producing level-set operations exposed by ``openvdb.tools``."""

from __future__ import annotations

import math
import numbers
from typing import Any

from .errors import CapabilityUnavailableError, InvalidGeometryError, ResourceLimitError
from .mesh import _check_output_limit, _coerce_real, _validate_band_width, _validate_thread_count
from .runtime import _call_native, _tools_api_version, require_runtime
from .types import (
    DEFAULT_LIMITS,
    Capability,
    ExecutionLimits,
    Interpolation,
    _coerce_interpolation,
)


def _tool(module: Any, name: str, capability: Capability) -> Any:
    value = getattr(getattr(module, "tools", None), name, None)
    if not callable(value):
        raise CapabilityUnavailableError((capability.value,))
    return value


def _voxel_size_for_preflight(grid: Any) -> float | None:
    transform = getattr(grid, "transform", None)
    evaluator = getattr(transform, "voxelSize", None)
    if not callable(evaluator):
        return None
    try:
        values = tuple(float(value) for value in evaluator())
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 3 or not all(math.isfinite(value) and value > 0 for value in values):
        return None
    return min(values)


def _check_resample_domain(
    grid: Any,
    reference_grid: Any,
    interpolation: Interpolation,
    limits: ExecutionLimits,
) -> None:
    bbox_evaluator = getattr(grid, "evalActiveVoxelBoundingBox", None)
    source_transform = getattr(grid, "transform", None)
    reference_transform = getattr(reference_grid, "transform", None)
    index_to_world = getattr(source_transform, "indexToWorld", None)
    world_to_index = getattr(reference_transform, "worldToIndex", None)
    if not callable(bbox_evaluator) or not callable(index_to_world) or not callable(world_to_index):
        return
    try:
        raw_bbox = bbox_evaluator()
        minimum = tuple(int(value) for value in raw_bbox[0])
        maximum = tuple(int(value) for value in raw_bbox[1])
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidGeometryError("grid returned a malformed active bounding box") from exc
    if len(minimum) != 3 or len(maximum) != 3:
        raise InvalidGeometryError("grid returned a malformed active bounding box")
    if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
        return

    target_points: list[tuple[float, float, float]] = []
    try:
        for x in (minimum[0], maximum[0] + 1):
            for y in (minimum[1], maximum[1] + 1):
                for z in (minimum[2], maximum[2] + 1):
                    world = index_to_world((x, y, z))
                    target = tuple(float(value) for value in world_to_index(world))
                    if len(target) != 3 or not all(math.isfinite(value) for value in target):
                        raise ValueError
                    target_points.append(target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidGeometryError(
            "grid transforms do not define a finite resampling domain"
        ) from exc

    sampler_radius = 0 if interpolation is Interpolation.NEAREST else 1
    safety_margin = 4
    output_minimum = tuple(
        math.floor(min(point[axis] for point in target_points)) - sampler_radius
        for axis in range(3)
    )
    output_maximum = tuple(
        math.ceil(max(point[axis] for point in target_points)) + sampler_radius for axis in range(3)
    )
    coord_minimum = -(2**31) + safety_margin
    coord_maximum = 2**31 - 1 - safety_margin
    if any(
        lower < coord_minimum or upper > coord_maximum
        for lower, upper in zip(output_minimum, output_maximum, strict=True)
    ):
        raise ResourceLimitError("resampled grid maps outside the supported int32 index range")
    dimensions = tuple(
        upper - lower + 1 + 2 * safety_margin
        for lower, upper in zip(output_minimum, output_maximum, strict=True)
    )
    if any(value > limits.max_voxel_extent for value in dimensions):
        raise ResourceLimitError("resampled grid domain exceeds max_voxel_extent")
    dense_voxel_count = math.prod(dimensions)
    if dense_voxel_count > limits.max_active_voxels:
        raise ResourceLimitError(
            f"resampled grid domain can contain {dense_voxel_count} voxels; "
            f"active voxels limit is {limits.max_active_voxels}"
        )


def _csg(
    name: str,
    left: Any,
    right: Any,
    *,
    thread_count: int,
    limits: ExecutionLimits,
) -> Any:
    _check_output_limit(left, limits)
    _check_output_limit(right, limits)
    module = require_runtime((Capability.CSG,))
    total_active_voxels = 0
    for grid in (left, right):
        counter = getattr(grid, "activeVoxelCount", None)
        if not callable(counter):
            continue
        try:
            count = int(counter())
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidGeometryError("grid returned a malformed active-voxel count") from exc
        if count < 0:
            raise InvalidGeometryError("grid returned a negative active-voxel count")
        total_active_voxels += count
        if total_active_voxels > limits.max_active_voxels:
            raise ResourceLimitError(
                f"combined CSG inputs have {total_active_voxels} active voxels; "
                f"limit is {limits.max_active_voxels}"
            )
    kwargs = {"thread_count": _validate_thread_count(thread_count, limits)}
    if _tools_api_version(module) >= 3:
        kwargs["max_active_voxels"] = limits.max_active_voxels
    result = _call_native(
        name,
        _tool(module, name, Capability.CSG),
        left,
        right,
        **kwargs,
    )
    _check_output_limit(result, limits)
    return result


def union(
    left: Any,
    right: Any,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return the level-set union without modifying either input grid."""

    return _csg("csg_union", left, right, thread_count=thread_count, limits=limits)


def intersection(
    left: Any,
    right: Any,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return the level-set intersection without modifying either input grid."""

    return _csg("csg_intersection", left, right, thread_count=thread_count, limits=limits)


def difference(
    left: Any,
    right: Any,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return ``left - right`` without modifying either input grid."""

    return _csg("csg_difference", left, right, thread_count=thread_count, limits=limits)


def offset(
    grid: Any,
    distance: float,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a level set offset by a signed world-space distance."""

    distance = _coerce_real(distance, label="distance")
    if not math.isfinite(distance):
        raise ValueError("distance must be finite")
    voxel_size = _voxel_size_for_preflight(grid)
    if voxel_size is None:
        _check_output_limit(grid, limits)
    else:
        distance_voxels = abs(distance) / voxel_size
        if not math.isfinite(distance_voxels) or distance_voxels > limits.max_band_width_voxels:
            raise ResourceLimitError(
                "distance exceeds max_band_width_voxels at the grid voxel size"
            )
        _check_output_limit(
            grid,
            limits,
            margin=math.ceil(distance_voxels) + 4,
            require_dense_bound=True,
        )
    module = require_runtime((Capability.LEVEL_SET_OFFSET,))
    result = _call_native(
        "level_set_offset",
        _tool(module, "level_set_offset", Capability.LEVEL_SET_OFFSET),
        grid,
        float(distance),
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def mean_filter(
    grid: Any,
    *,
    width: int = 1,
    iterations: int = 1,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a mean-filtered copy of a level set."""

    width, iterations = _filter_controls(width, iterations, limits)
    _check_output_limit(grid, limits, margin=width * iterations)
    _check_filter_work(grid, width, iterations, limits)
    module = require_runtime((Capability.LEVEL_SET_FILTER,))
    result = _call_native(
        "level_set_mean",
        _tool(module, "level_set_mean", Capability.LEVEL_SET_FILTER),
        grid,
        width=width,
        iterations=iterations,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def _filter_controls(width: int, iterations: int, limits: ExecutionLimits) -> tuple[int, int]:
    if isinstance(width, bool) or not isinstance(width, numbers.Integral) or width <= 0:
        raise ValueError("width must be a positive integer")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, numbers.Integral)
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    if width > limits.max_filter_width:
        raise ResourceLimitError(f"width is {width}; limit is {limits.max_filter_width}")
    if iterations > limits.max_filter_iterations:
        raise ResourceLimitError(
            f"iterations is {iterations}; limit is {limits.max_filter_iterations}"
        )
    return int(width), int(iterations)


def _check_filter_work(
    grid: Any,
    width: int,
    iterations: int,
    limits: ExecutionLimits,
) -> None:
    counter = getattr(grid, "activeVoxelCount", None)
    if not callable(counter):
        return
    try:
        active_voxels = int(counter())
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidGeometryError("grid returned a malformed active-voxel count") from exc
    work = active_voxels * (2 * width + 1) * 3 * iterations
    if work > limits.max_filter_work:
        raise ResourceLimitError(
            f"filter request requires {work} work units; limit is {limits.max_filter_work}"
        )


def scalar_mean_filter(
    grid: Any,
    *,
    width: int = 1,
    iterations: int = 1,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a mean-filtered scalar-grid copy without requiring level-set class."""

    width, iterations = _filter_controls(width, iterations, limits)
    _check_output_limit(grid, limits, margin=width * iterations)
    _check_filter_work(grid, width, iterations, limits)
    module = require_runtime((Capability.SCALAR_MEAN_FILTER,))
    result = _call_native(
        "scalar_mean",
        _tool(module, "scalar_mean", Capability.SCALAR_MEAN_FILTER),
        grid,
        width=width,
        iterations=iterations,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def normalize(
    grid: Any,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a normalized level-set copy with restored signed-distance values."""

    _check_output_limit(grid, limits, margin=4)
    module = require_runtime((Capability.LEVEL_SET_NORMALIZE,))
    result = _call_native(
        "level_set_normalize",
        _tool(module, "level_set_normalize", Capability.LEVEL_SET_NORMALIZE),
        grid,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def rebuild(
    grid: Any,
    *,
    isovalue: float = 0.0,
    exterior_width: float = 3.0,
    interior_width: float = 3.0,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a rebuilt narrow-band level set around ``isovalue``.

    Band widths are expressed in voxels.
    """

    isovalue = _coerce_real(isovalue, label="isovalue")
    if not math.isfinite(isovalue):
        raise ValueError("isovalue must be finite")
    exterior_width = _validate_band_width(exterior_width, label="exterior_width", limits=limits)
    interior_width = _validate_band_width(interior_width, label="interior_width", limits=limits)
    voxel_size = _voxel_size_for_preflight(grid)
    if voxel_size is not None and voxel_size * max(exterior_width, interior_width) > 3.4028235e38:
        raise ResourceLimitError("rebuild band exceeds the finite float32 background range")
    expansion = math.ceil(max(exterior_width, interior_width)) + 4
    _check_output_limit(
        grid,
        limits,
        margin=expansion,
        require_dense_bound=True,
    )
    module = require_runtime((Capability.LEVEL_SET_REBUILD,))
    result = _call_native(
        "level_set_rebuild",
        _tool(module, "level_set_rebuild", Capability.LEVEL_SET_REBUILD),
        grid,
        isovalue=float(isovalue),
        exterior_width=exterior_width,
        interior_width=interior_width,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def resample_to_match(
    grid: Any,
    reference_grid: Any,
    *,
    interpolation: Interpolation | str = Interpolation.QUADRATIC,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return ``grid`` resampled into ``reference_grid`` index space."""

    interpolation = _coerce_interpolation(interpolation)
    _check_output_limit(grid, limits)
    _check_output_limit(reference_grid, limits)
    _check_resample_domain(grid, reference_grid, interpolation, limits)
    module = require_runtime((Capability.RESAMPLE_TO_MATCH,))
    result = _call_native(
        "resample_to_match",
        _tool(module, "resample_to_match", Capability.RESAMPLE_TO_MATCH),
        grid,
        reference_grid,
        interpolation=interpolation.value,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result
