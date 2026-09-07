# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy-producing topology operations through ``openvdb.tools``."""

from __future__ import annotations

import math
import numbers
from typing import Any

from .errors import ResourceLimitError
from .level_set import _tool, _voxel_size_for_preflight
from .mesh import _check_output_limit, _coerce_real, _validate_thread_count
from .runtime import _call_native, require_runtime
from .types import DEFAULT_LIMITS, Capability, ExecutionLimits


def _optional_finite(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    value = _coerce_real(value, label=label)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite when supplied")
    return float(value)


def _topology_steps(value: int, label: str, limits: ExecutionLimits) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    if value > limits.max_topology_steps:
        raise ResourceLimitError(
            f"{label} exceeds max_topology_steps ({limits.max_topology_steps})"
        )
    return int(value)


def active_value_mask(
    grid: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a Boolean mask for active values within optional inclusive bounds."""

    min_value = _optional_finite(min_value, "min_value")
    max_value = _optional_finite(max_value, "max_value")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError("min_value must not exceed max_value")
    _check_output_limit(grid, limits)
    module = require_runtime((Capability.ACTIVE_VALUE_MASK,))
    result = _call_native(
        "active_value_mask",
        _tool(module, "active_value_mask", Capability.ACTIVE_VALUE_MASK),
        grid,
        min_value=min_value,
        max_value=max_value,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def topology_to_level_set(
    grid: Any,
    *,
    half_width: int = 3,
    closing_steps: int = 0,
    dilation: int = 0,
    smoothing_steps: int = 0,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a level set from active topology with explicit morphology steps."""

    if (
        isinstance(half_width, bool)
        or not isinstance(half_width, numbers.Integral)
        or half_width <= 0
    ):
        raise ValueError("half_width must be a positive integer")
    if half_width > limits.max_band_width_voxels:
        raise ResourceLimitError(
            f"half_width exceeds max_band_width_voxels ({limits.max_band_width_voxels})"
        )
    closing_steps = _topology_steps(closing_steps, "closing_steps", limits)
    dilation = _topology_steps(dilation, "dilation", limits)
    smoothing_steps = _topology_steps(smoothing_steps, "smoothing_steps", limits)
    voxel_size = _voxel_size_for_preflight(grid)
    if voxel_size is not None and voxel_size * int(half_width) > 3.4028235e38:
        raise ResourceLimitError("topology band exceeds the finite float32 background range")
    expansion = int(half_width) + closing_steps + dilation + smoothing_steps + 4
    _check_output_limit(
        grid,
        limits,
        margin=expansion,
        require_dense_bound=True,
    )
    module = require_runtime((Capability.TOPOLOGY_TO_LEVEL_SET,))
    result = _call_native(
        "topology_to_level_set",
        _tool(module, "topology_to_level_set", Capability.TOPOLOGY_TO_LEVEL_SET),
        grid,
        half_width=int(half_width),
        closing_steps=closing_steps,
        dilation=dilation,
        smoothing_steps=smoothing_steps,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result


def extract_enclosed_region(
    grid: Any,
    *,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Return a Boolean grid containing the region enclosed by a level set."""

    _check_output_limit(grid, limits, margin=1, require_dense_bound=True)
    module = require_runtime((Capability.EXTRACT_ENCLOSED_REGION,))
    result = _call_native(
        "extract_enclosed_region",
        _tool(module, "extract_enclosed_region", Capability.EXTRACT_ENCLOSED_REGION),
        grid,
        thread_count=_validate_thread_count(thread_count, limits),
    )
    _check_output_limit(result, limits)
    return result
