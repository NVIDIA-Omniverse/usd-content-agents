# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, in-process OpenVDB 13 operations for trusted application code."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORT_MODULES = {
    "DEFAULT_LIMITS": ".types",
    "Capability": ".types",
    "CapabilityUnavailableError": ".errors",
    "ExecutionLimits": ".types",
    "Interpolation": ".types",
    "InvalidGeometryError": ".errors",
    "Mesh": ".types",
    "NativeOperationError": ".errors",
    "OpenVDBRuntimeError": ".errors",
    "ResourceLimitError": ".errors",
    "RuntimeInfo": ".types",
    "RuntimeUnavailableError": ".errors",
    "RuntimeVersionError": ".errors",
    "VDBContents": ".types",
    "active_value_mask": ".topology",
    "detect_capabilities": ".runtime",
    "difference": ".level_set",
    "extract_enclosed_region": ".topology",
    "inspect_runtime": ".runtime",
    "intersection": ".level_set",
    "is_available": ".runtime",
    "mean_filter": ".level_set",
    "mesh_to_level_set": ".mesh",
    "mesh_to_unsigned_distance_field": ".mesh",
    "normalize": ".level_set",
    "offset": ".level_set",
    "read_all": ".io",
    "read_grid": ".io",
    "require_runtime": ".runtime",
    "rebuild": ".level_set",
    "resample_to_match": ".level_set",
    "sample_gradients": ".sampling",
    "sample_values": ".sampling",
    "scalar_mean_filter": ".level_set",
    "topology_to_level_set": ".topology",
    "union": ".level_set",
    "volume_to_mesh": ".mesh",
    "write": ".io",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


if TYPE_CHECKING:
    from .errors import (
        CapabilityUnavailableError,
        InvalidGeometryError,
        NativeOperationError,
        OpenVDBRuntimeError,
        ResourceLimitError,
        RuntimeUnavailableError,
        RuntimeVersionError,
    )
    from .io import read_all, read_grid, write
    from .level_set import (
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
    from .mesh import mesh_to_level_set, mesh_to_unsigned_distance_field, volume_to_mesh
    from .runtime import detect_capabilities, inspect_runtime, is_available, require_runtime
    from .sampling import sample_gradients, sample_values
    from .topology import active_value_mask, extract_enclosed_region, topology_to_level_set
    from .types import (
        DEFAULT_LIMITS,
        Capability,
        ExecutionLimits,
        Interpolation,
        Mesh,
        RuntimeInfo,
        VDBContents,
    )
