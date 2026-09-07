# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded world-space sampling through ``openvdb.tools``."""

from __future__ import annotations

from typing import Any

from .errors import CapabilityUnavailableError, InvalidGeometryError, ResourceLimitError
from .mesh import _check_output_limit, _validate_thread_count
from .runtime import _call_native, require_runtime
from .types import (
    DEFAULT_LIMITS,
    Capability,
    ExecutionLimits,
    Float32Array,
    Interpolation,
    _canonical_world_points,
    _coerce_interpolation,
    np,
)


def _bounded_points(values: object, limits: ExecutionLimits) -> Float32Array:
    try:
        point_count = len(values)  # type: ignore[arg-type]
    except TypeError:
        point_count = None
    if point_count is not None and point_count > limits.max_sample_points:
        raise ResourceLimitError(
            f"sample batch has {point_count} points; limit is {limits.max_sample_points}"
        )
    return _canonical_world_points(values, maximum=limits.max_sample_points)


def _sample(
    tool_name: str,
    capability: Capability,
    grid: Any,
    points: object,
    *,
    interpolation: Interpolation | str,
    thread_count: int,
    limits: ExecutionLimits,
) -> Float32Array:
    points_array = _bounded_points(points, limits)
    interpolation = _coerce_interpolation(interpolation)
    _check_output_limit(grid, limits)
    module = require_runtime((capability,))
    tool = getattr(getattr(module, "tools", None), tool_name, None)
    if not callable(tool):
        raise CapabilityUnavailableError((capability.value,))
    raw_result = _call_native(
        tool_name,
        tool,
        grid,
        points_array,
        interpolation=interpolation.value,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    try:
        result = np.ascontiguousarray(raw_result, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryError(f"{tool_name} returned a nonnumeric result") from exc
    expected_shape = (
        (len(points_array), 3)
        if capability is Capability.SAMPLE_GRADIENTS
        else (len(points_array),)
    )
    if result.shape != expected_shape:
        raise InvalidGeometryError(
            f"{tool_name} returned shape {result.shape}; expected {expected_shape}"
        )
    if not np.isfinite(result).all():
        raise InvalidGeometryError(f"{tool_name} returned nonfinite values")
    return result


def sample_values(
    grid: Any,
    points: object,
    *,
    interpolation: Interpolation | str = Interpolation.QUADRATIC,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Float32Array:
    """Sample scalar grid values at an ``(N, 3)`` world-space point array."""

    return _sample(
        "sample_values",
        Capability.SAMPLE_VALUES,
        grid,
        points,
        interpolation=interpolation,
        thread_count=thread_count,
        limits=limits,
    )


def sample_gradients(
    grid: Any,
    points: object,
    *,
    interpolation: Interpolation | str = Interpolation.QUADRATIC,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Float32Array:
    """Sample gradients in world coordinates, expressed per world unit."""

    return _sample(
        "sample_gradients",
        Capability.SAMPLE_GRADIENTS,
        grid,
        points,
        interpolation=interpolation,
        thread_count=thread_count,
        limits=limits,
    )
