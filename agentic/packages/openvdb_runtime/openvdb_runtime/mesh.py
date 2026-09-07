# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh and level-set conversion through the in-process OpenVDB module."""

from __future__ import annotations

import math
import numbers
from typing import Any

from .errors import CapabilityUnavailableError, InvalidGeometryError, ResourceLimitError
from .runtime import _call_native, _tools_api_version, require_runtime
from .types import DEFAULT_LIMITS, Capability, ExecutionLimits, Mesh, _is_bool_scalar, np


def _coerce_real(value: object, *, label: str) -> float:
    if _is_bool_scalar(value) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real number")
    return float(value)


def _validate_thread_count(thread_count: int, limits: ExecutionLimits) -> int:
    if (
        isinstance(thread_count, bool)
        or not isinstance(thread_count, numbers.Integral)
        or thread_count <= 0
    ):
        raise ValueError("thread_count must be a positive integer")
    if thread_count > limits.max_threads:
        raise ResourceLimitError(f"thread_count is {thread_count}; limit is {limits.max_threads}")
    return int(thread_count)


def _check_input_limits(
    mesh: Mesh,
    voxel_size: float,
    half_width: float,
    limits: ExecutionLimits,
) -> None:
    if voxel_size * half_width > float(np.finfo(np.float32).max):
        raise ResourceLimitError("mesh band exceeds the finite float32 background range")
    if len(mesh.vertices) > limits.max_vertices:
        raise ResourceLimitError(
            f"mesh has {len(mesh.vertices)} vertices; limit is {limits.max_vertices}"
        )
    if mesh.face_count > limits.max_faces:
        raise ResourceLimitError(f"mesh has {mesh.face_count} faces; limit is {limits.max_faces}")
    vertices = mesh.vertices.astype(np.float64)
    margin = math.ceil(half_width) + 4
    voxel_minimum = np.floor(np.min(vertices, axis=0) / voxel_size)
    voxel_maximum = np.ceil(np.max(vertices, axis=0) / voxel_size)
    if not np.isfinite(voxel_minimum).all() or not np.isfinite(voxel_maximum).all():
        raise ResourceLimitError(
            "mesh coordinates at the requested voxel size are not representable"
        )
    coord_info = np.iinfo(np.int32)
    if np.any(voxel_minimum < coord_info.min + margin) or np.any(
        voxel_maximum > coord_info.max - margin
    ):
        raise ResourceLimitError("mesh maps outside the supported int32 voxel range")
    voxel_dimensions = voxel_maximum - voxel_minimum + 1 + 2 * margin
    if any(int(value) > limits.max_voxel_extent for value in voxel_dimensions):
        raise ResourceLimitError("mesh extent at the requested voxel size exceeds max_voxel_extent")
    dense_voxel_count = math.prod(int(value) for value in voxel_dimensions)
    if dense_voxel_count > limits.max_active_voxels:
        raise ResourceLimitError(
            f"mesh voxel domain can contain {dense_voxel_count} voxels; "
            f"active voxels limit is {limits.max_active_voxels}"
        )


def _check_mesh_cardinality(mesh: Mesh, limits: ExecutionLimits) -> None:
    if len(mesh.vertices) > limits.max_vertices:
        raise ResourceLimitError(
            f"mesh has {len(mesh.vertices)} vertices; limit is {limits.max_vertices}"
        )
    if mesh.face_count > limits.max_faces:
        raise ResourceLimitError(f"mesh has {mesh.face_count} faces; limit is {limits.max_faces}")


def _grid_memory_bytes(grid: Any) -> int:
    evaluator = getattr(grid, "memUsage", None)
    if not callable(evaluator):
        raise InvalidGeometryError("VDB field does not report its in-memory size")
    try:
        value = evaluator()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise InvalidGeometryError("VDB field memory usage could not be inspected") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidGeometryError("VDB field returned malformed memory usage")
    return value


def _inspect_output_limits(
    grid: Any,
    limits: ExecutionLimits,
    *,
    margin: int = 0,
    require_dense_bound: bool = False,
) -> tuple[int | None, int]:
    count: int | None = None
    counter = getattr(grid, "activeVoxelCount", None)
    if callable(counter):
        try:
            raw_count = counter()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidGeometryError("grid returned a malformed active-voxel count") from exc
        if isinstance(raw_count, bool) or not isinstance(raw_count, numbers.Integral):
            raise InvalidGeometryError("grid returned a malformed active-voxel count")
        count = int(raw_count)
        if count < 0:
            raise InvalidGeometryError("grid returned a negative active-voxel count")
        if count > limits.max_active_voxels:
            raise ResourceLimitError(
                f"grid has {count} active voxels; limit is {limits.max_active_voxels}"
            )
    dimensions: tuple[int, int, int] | None = None
    bbox_evaluator = getattr(grid, "evalActiveVoxelBoundingBox", None)
    if callable(bbox_evaluator):
        try:
            raw_bbox = bbox_evaluator()
            if len(raw_bbox) != 2 or any(len(corner) != 3 for corner in raw_bbox):
                raise ValueError
            minimum = tuple(int(value) for value in raw_bbox[0])
            maximum = tuple(int(value) for value in raw_bbox[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidGeometryError("grid returned a malformed active bounding box") from exc
        if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
            if count not in (None, 0):
                raise InvalidGeometryError("nonempty grid returned an empty active bounding box")
            dimensions = (0, 0, 0)
        else:
            coord_info = np.iinfo(np.int32)
            if any(
                lower < coord_info.min + margin or upper > coord_info.max - margin
                for lower, upper in zip(minimum, maximum, strict=True)
            ):
                raise ResourceLimitError("grid is too close to the int32 index boundary")
            dimensions = tuple(
                upper - lower + 1 + 2 * margin
                for lower, upper in zip(minimum, maximum, strict=True)
            )
    else:
        dimension_evaluator = getattr(grid, "evalActiveVoxelDim", None)
        if callable(dimension_evaluator):
            try:
                dimensions = tuple(int(value) for value in dimension_evaluator())
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidGeometryError("grid returned malformed active dimensions") from exc
            if len(dimensions) != 3 or any(value < 0 for value in dimensions):
                raise InvalidGeometryError("grid returned malformed active dimensions")

    if dimensions is not None:
        if any(value > limits.max_voxel_extent for value in dimensions):
            raise ResourceLimitError(
                f"grid active dimensions {dimensions} exceed max_voxel_extent "
                f"({limits.max_voxel_extent})"
            )
        dense_voxel_count = math.prod(dimensions)
        if require_dense_bound and dense_voxel_count > limits.max_active_voxels:
            raise ResourceLimitError(
                f"expanded grid domain can contain {dense_voxel_count} voxels; "
                f"active voxels limit is {limits.max_active_voxels}"
            )

    memory_bytes = _grid_memory_bytes(grid)
    if memory_bytes > limits.max_field_memory_bytes:
        raise ResourceLimitError(
            f"VDB field requires {memory_bytes} bytes; limit is {limits.max_field_memory_bytes}"
        )
    return count, memory_bytes


def _check_output_limit(
    grid: Any,
    limits: ExecutionLimits,
    *,
    margin: int = 0,
    require_dense_bound: bool = False,
) -> int:
    _, memory_bytes = _inspect_output_limits(
        grid,
        limits,
        margin=margin,
        require_dense_bound=require_dense_bound,
    )
    return memory_bytes


def _validate_band_width(width: float, *, label: str, limits: ExecutionLimits) -> float:
    width = _coerce_real(width, label=label)
    if not math.isfinite(width) or width <= 0:
        raise InvalidGeometryError(f"{label} must be finite and positive")
    if width > limits.max_band_width_voxels:
        raise ResourceLimitError(
            f"{label} is {width}; limit is {limits.max_band_width_voxels} voxels"
        )
    return float(width)


def _validate_grid_name(name: str | None) -> str | None:
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("name must be a nonempty string when supplied")
    return name


def _snapshot_mesh(mesh: Mesh, limits: ExecutionLimits) -> Mesh:
    """Copy then revalidate caller-owned arrays before entering native code."""

    return Mesh._from_bounded_buffers(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        quads=mesh.quads,
        max_vertices=limits.max_vertices,
        max_faces=limits.max_faces,
    )


def mesh_to_level_set(
    mesh: Mesh,
    *,
    voxel_size: float,
    half_width: float = 3.0,
    name: str | None = None,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Convert a closed world-space mesh to a narrow-band ``FloatGrid``.

    The repository extension is preferred because it provides explicit thread
    control. A stock OpenVDB 13 module remains usable through
    ``FloatGrid.createLevelSetFromPolygons``.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an openvdb_runtime.Mesh")
    voxel_size = _coerce_real(voxel_size, label="voxel_size")
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise InvalidGeometryError("voxel_size must be finite and positive")
    name = _validate_grid_name(name)
    half_width = _validate_band_width(half_width, label="half_width", limits=limits)
    thread_count = _validate_thread_count(thread_count, limits)
    _check_mesh_cardinality(mesh, limits)
    mesh = _snapshot_mesh(mesh, limits)
    _check_input_limits(mesh, voxel_size, half_width, limits)
    module = require_runtime((Capability.TRANSFORMS, Capability.MESH_TO_LEVEL_SET))
    tools = getattr(module, "tools", None)
    native_tool = getattr(tools, "mesh_to_level_set", None)
    if callable(native_tool):
        native_faces = mesh.triangulated_faces(maximum=limits.max_faces)
        native_kwargs = {
            "voxel_size": float(voxel_size),
            "half_width": float(half_width),
            "thread_count": thread_count,
        }
        if _tools_api_version(module) >= 3:
            native_kwargs.update(
                max_vertices=limits.max_vertices,
                max_faces=limits.max_faces,
                max_active_voxels=limits.max_active_voxels,
            )
        grid = _call_native(
            "mesh_to_level_set",
            native_tool,
            mesh.vertices,
            native_faces,
            **native_kwargs,
        )
    else:
        transform = _call_native(
            "createLinearTransform", module.createLinearTransform, float(voxel_size)
        )
        grid = _call_native(
            "createLevelSetFromPolygons",
            module.FloatGrid.createLevelSetFromPolygons,
            mesh.vertices,
            triangles=mesh.triangles if len(mesh.triangles) else None,
            quads=mesh.quads if len(mesh.quads) else None,
            transform=transform,
            halfWidth=float(half_width),
        )
    if name is not None:
        grid.name = name
    _check_output_limit(grid, limits)
    return grid


def mesh_to_unsigned_distance_field(
    mesh: Mesh,
    *,
    voxel_size: float,
    half_width: float = 3.0,
    name: str | None = None,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Any:
    """Convert a world-space mesh to an unsigned narrow-band distance grid."""

    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be an openvdb_runtime.Mesh")
    voxel_size = _coerce_real(voxel_size, label="voxel_size")
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise InvalidGeometryError("voxel_size must be finite and positive")
    name = _validate_grid_name(name)
    half_width = _validate_band_width(half_width, label="half_width", limits=limits)
    thread_count = _validate_thread_count(thread_count, limits)
    _check_mesh_cardinality(mesh, limits)
    mesh = _snapshot_mesh(mesh, limits)
    _check_input_limits(mesh, voxel_size, half_width, limits)
    module = require_runtime((Capability.MESH_TO_UNSIGNED_DISTANCE_FIELD,))
    tool = getattr(getattr(module, "tools", None), "mesh_to_unsigned_distance_field", None)
    if not callable(tool):
        raise CapabilityUnavailableError((Capability.MESH_TO_UNSIGNED_DISTANCE_FIELD.value,))
    native_faces = mesh.triangulated_faces(maximum=limits.max_faces)
    native_kwargs = {
        "voxel_size": float(voxel_size),
        "half_width": half_width,
        "thread_count": thread_count,
    }
    if _tools_api_version(module) >= 3:
        native_kwargs.update(
            max_vertices=limits.max_vertices,
            max_faces=limits.max_faces,
            max_active_voxels=limits.max_active_voxels,
        )
    grid = _call_native(
        "mesh_to_unsigned_distance_field",
        tool,
        mesh.vertices,
        native_faces,
        **native_kwargs,
    )
    if name is not None:
        grid.name = name
    _check_output_limit(grid, limits)
    return grid


def volume_to_mesh(
    grid: Any,
    *,
    isovalue: float = 0.0,
    adaptivity: float = 0.0,
    relax_disoriented_triangles: bool = True,
    thread_count: int = 1,
    limits: ExecutionLimits = DEFAULT_LIMITS,
) -> Mesh:
    """Extract a world-space polygon mesh from a scalar grid."""

    isovalue = _coerce_real(isovalue, label="isovalue")
    if not math.isfinite(isovalue):
        raise InvalidGeometryError("isovalue must be finite")
    adaptivity = _coerce_real(adaptivity, label="adaptivity")
    if not math.isfinite(adaptivity) or not 0.0 <= adaptivity <= 1.0:
        raise InvalidGeometryError("adaptivity must be finite and between 0 and 1")
    if not isinstance(relax_disoriented_triangles, bool):
        raise TypeError("relax_disoriented_triangles must be a bool")
    thread_count = _validate_thread_count(thread_count, limits)
    _check_output_limit(grid, limits, margin=2)
    module = require_runtime((Capability.VOLUME_TO_MESH,))
    tools = getattr(module, "tools", None)
    native_tool = getattr(tools, "volume_to_mesh", None)
    if callable(native_tool):
        kwargs = {
            "isovalue": float(isovalue),
            "adaptivity": float(adaptivity),
            "thread_count": thread_count,
        }
        if _tools_api_version(module) >= 2:
            kwargs["relax_disoriented_triangles"] = relax_disoriented_triangles
        elif not relax_disoriented_triangles:
            raise CapabilityUnavailableError((Capability.EXTENDED_VOLUME_TO_MESH.value,))
        if _tools_api_version(module) >= 3:
            kwargs.update(
                max_vertices=limits.max_vertices,
                max_faces=limits.max_faces,
                max_active_voxels=limits.max_active_voxels,
            )
        elif (
            limits.max_vertices < DEFAULT_LIMITS.max_vertices
            or limits.max_faces < DEFAULT_LIMITS.max_faces
            or limits.max_active_voxels < DEFAULT_LIMITS.max_active_voxels
        ):
            raise CapabilityUnavailableError(("bounded_volume_to_mesh",))
        vertices, triangles, quads = _call_native("volume_to_mesh", native_tool, grid, **kwargs)
    else:
        if not relax_disoriented_triangles:
            raise CapabilityUnavailableError((Capability.EXTENDED_VOLUME_TO_MESH.value,))
        converter = getattr(grid, "convertToPolygons", None)
        if not callable(converter):
            raise InvalidGeometryError("grid does not support scalar volume meshing")
        if (
            limits.max_vertices < DEFAULT_LIMITS.max_vertices
            or limits.max_faces < DEFAULT_LIMITS.max_faces
        ):
            raise CapabilityUnavailableError(("bounded_volume_to_mesh",))
        vertices, triangles, quads = _call_native(
            "convertToPolygons",
            converter,
            isovalue=float(isovalue),
            adaptivity=float(adaptivity),
        )
    mesh = Mesh(vertices=vertices, triangles=triangles, quads=quads)
    if len(mesh.vertices) > limits.max_vertices or mesh.face_count > limits.max_faces:
        raise ResourceLimitError("meshed volume exceeds configured mesh limits")
    return mesh
