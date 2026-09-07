# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public value types for the OpenVDB runtime facade."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Array = NDArray[Any]
    Float32Array = NDArray[np.float32]
    Int32Array = NDArray[np.int32]
else:
    Array = Any
    Float32Array = Any
    Int32Array = Any

from .errors import InvalidGeometryError, ResourceLimitError, RuntimeUnavailableError


class _LazyNumpy:
    """Load application-owned NumPy only when a numerical operation needs it."""

    _module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        module = self._module
        if module is None:
            try:
                module = import_module("numpy")
            except ImportError as exc:
                raise RuntimeUnavailableError(
                    "OpenVDB numerical operations require NumPy supplied by the application"
                ) from exc
            self._module = module
        return getattr(module, name)


if not TYPE_CHECKING:
    np = _LazyNumpy()


def _is_bool_scalar(value: object) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__.partition(".")[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    )


class Capability(StrEnum):
    """Operations discoverable on the loaded OpenVDB module."""

    TRANSFORMS = "transforms"
    VDB_IO = "vdb_io"
    NUMPY_TRANSFER = "numpy_transfer"
    MESH_TO_LEVEL_SET = "mesh_to_level_set"
    MESH_TO_UNSIGNED_DISTANCE_FIELD = "mesh_to_unsigned_distance_field"
    VOLUME_TO_MESH = "volume_to_mesh"
    EXTENDED_VOLUME_TO_MESH = "extended_volume_to_mesh"
    CSG = "csg"
    LEVEL_SET_OFFSET = "level_set_offset"
    LEVEL_SET_FILTER = "level_set_filter"
    SCALAR_MEAN_FILTER = "scalar_mean_filter"
    LEVEL_SET_NORMALIZE = "level_set_normalize"
    LEVEL_SET_REBUILD = "level_set_rebuild"
    RESAMPLE_TO_MATCH = "resample_to_match"
    SAMPLE_VALUES = "sample_values"
    SAMPLE_GRADIENTS = "sample_gradients"
    ACTIVE_VALUE_MASK = "active_value_mask"
    TOPOLOGY_TO_LEVEL_SET = "topology_to_level_set"
    EXTRACT_ENCLOSED_REGION = "extract_enclosed_region"


class Interpolation(StrEnum):
    """Interpolation kernels supported by resampling and world sampling."""

    NEAREST = "nearest"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Identity and available operations for the loaded native module."""

    library_version: tuple[int, int, int]
    distribution_version: str | None
    file_format_version: int | None
    module_path: str | None
    module_sha256: str | None
    source_lock_path: str | None
    source_lock_sha256: str | None
    source_distribution_version: str | None
    policy_schema: str
    source_commit: str | None
    capabilities: frozenset[Capability]

    def supports(self, *capabilities: Capability) -> bool:
        """Return whether every requested capability is available."""

        return set(capabilities).issubset(self.capabilities)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible identity record."""

        return {
            "library_version": list(self.library_version),
            "distribution_version": self.distribution_version,
            "file_format_version": self.file_format_version,
            "module_path": self.module_path,
            "module_sha256": self.module_sha256,
            "source_lock_path": self.source_lock_path,
            "source_lock_sha256": self.source_lock_sha256,
            "source_distribution_version": self.source_distribution_version,
            "policy_schema": self.policy_schema,
            "source_commit": self.source_commit,
            "capabilities": sorted(capability.value for capability in self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Caller-visible bounds applied before and after native operations."""

    max_vertices: int = 10_000_000
    max_faces: int = 20_000_000
    max_active_voxels: int = 50_000_000
    max_voxel_extent: int = 1_000_000
    max_threads: int = 32
    max_sample_points: int = 10_000_000
    max_topology_steps: int = 1_024
    max_band_width_voxels: float = 1_024.0
    max_filter_width: int = 256
    max_filter_iterations: int = 1_024
    max_filter_work: int = 1_000_000_000
    max_file_bytes: int = 8_589_934_592
    max_fields: int = 1_024
    max_field_memory_bytes: int = 1_073_741_824
    max_total_field_memory_bytes: int = 2_147_483_648
    max_metadata_entries: int = 1_024
    max_metadata_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "max_vertices",
            "max_faces",
            "max_active_voxels",
            "max_voxel_extent",
            "max_threads",
            "max_sample_points",
            "max_topology_steps",
            "max_filter_width",
            "max_filter_iterations",
            "max_filter_work",
            "max_file_bytes",
            "max_fields",
            "max_field_memory_bytes",
            "max_total_field_memory_bytes",
            "max_metadata_entries",
            "max_metadata_bytes",
        ):
            value = getattr(self, name)
            if _is_bool_scalar(value) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        value = self.max_band_width_voxels
        if _is_bool_scalar(value) or not isinstance(value, numbers.Real):
            raise TypeError("max_band_width_voxels must be a real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("max_band_width_voxels must be finite and positive")


DEFAULT_LIMITS = ExecutionLimits()


def _empty_faces(width: int) -> Int32Array:
    return np.empty((0, width), dtype=np.int32)


def _ndarray_view(values: object) -> Array | None:
    if not isinstance(values, np.ndarray):
        return None
    # Bypass ndarray-subclass overrides of ``shape`` and ``__array__``.
    return np.ndarray.view(values, np.ndarray)


def _declared_row_count(values: object, *, label: str, maximum: int) -> int | None:
    array = _ndarray_view(values)
    if array is not None:
        if array.ndim != 2:
            raise InvalidGeometryError(f"{label} must be a two-dimensional array")
        count = int(array.shape[0])
    elif type(values) in (list, tuple):
        count = len(values)  # Built-in container length cannot under-report its storage.
    else:
        try:
            count = len(values)  # type: ignore[arg-type]
        except (OverflowError, TypeError, ValueError):
            return None
    if count > maximum:
        if label == "vertices":
            raise ResourceLimitError(f"mesh has {count} vertices; limit is {maximum}")
        if label == "points":
            raise ResourceLimitError(f"sample batch has {count} points; limit is {maximum}")
        raise ResourceLimitError(f"{label} has {count} rows; limit is {maximum}")
    return count


def _row_values(row: object, width: int, *, label: str) -> tuple[object, ...]:
    array = _ndarray_view(row)
    if array is not None:
        if array.ndim != 1 or array.shape != (width,):
            raise InvalidGeometryError(f"{label} rows must contain exactly {width} values")
        return tuple(array[index] for index in range(width))
    if type(row) not in (list, tuple) or len(row) != width:
        raise InvalidGeometryError(f"{label} rows must contain exactly {width} values")
    return tuple(row)


def _bounded_numeric_rows(
    values: object,
    *,
    width: int,
    label: str,
    maximum: int,
    integer: bool,
) -> Array:
    declared = _declared_row_count(values, label=label, maximum=maximum)
    array = _ndarray_view(values)
    if array is not None:
        if array.shape[1:] != (width,):
            raise InvalidGeometryError(f"{label} must have shape (N, {width})")
        if integer:
            if not np.issubdtype(array.dtype, np.integer) or array.dtype.kind == "b":
                raise InvalidGeometryError(f"{label} must contain integer indices")
            if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int32).max)):
                raise InvalidGeometryError(f"{label} indices must fit in nonnegative int32")
            return np.array(array, dtype=np.int32, order="C", copy=True)
        if array.dtype.kind not in "iuf":
            raise InvalidGeometryError(f"{label} must contain real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(array, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(result).all():
            raise InvalidGeometryError(f"{label} must contain only finite values")
        return result

    # Never invoke an arbitrary ``__array__`` hook. Only built-in outer row
    # containers are materialized, and their exact cardinality was checked first.
    if type(values) not in (list, tuple):
        if declared is not None and declared > maximum:
            raise ResourceLimitError(f"{label} has {declared} rows; limit is {maximum}")
        raise InvalidGeometryError(
            f"{label} must be a NumPy array or a finite built-in row sequence"
        )
    result = np.empty((len(values), width), dtype=np.int32 if integer else np.float32)
    for row_index, row in enumerate(values):
        items = _row_values(row, width, label=label)
        for column, item in enumerate(items):
            if integer:
                if _is_bool_scalar(item) or not isinstance(item, numbers.Integral):
                    raise InvalidGeometryError(f"{label} must contain integer indices")
                value = int(item)
                if value < 0 or value > np.iinfo(np.int32).max:
                    raise InvalidGeometryError(f"{label} indices must fit in nonnegative int32")
            else:
                if _is_bool_scalar(item) or not isinstance(item, numbers.Real):
                    raise InvalidGeometryError(f"{label} must contain real numeric values")
                value = float(item)
                if not math.isfinite(value) or abs(value) > np.finfo(np.float32).max:
                    raise InvalidGeometryError(f"{label} must contain only finite float32 values")
            result[row_index, column] = value
    return result


def _canonical_vertices(
    values: object, *, maximum: int = DEFAULT_LIMITS.max_vertices
) -> Float32Array:
    return _bounded_numeric_rows(values, width=3, label="vertices", maximum=maximum, integer=False)


def _canonical_world_points(
    values: object, *, maximum: int = DEFAULT_LIMITS.max_sample_points
) -> Float32Array:
    return _bounded_numeric_rows(values, width=3, label="points", maximum=maximum, integer=False)


def _coerce_interpolation(value: Interpolation | str) -> Interpolation:
    try:
        return Interpolation(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in Interpolation)
        raise ValueError(f"interpolation must be one of: {choices}") from exc


def _canonical_faces(
    values: object,
    width: int,
    label: str,
    *,
    maximum: int = DEFAULT_LIMITS.max_faces,
) -> Int32Array:
    result = _bounded_numeric_rows(values, width=width, label=label, maximum=maximum, integer=True)
    if any(
        np.any(result[:, left] == result[:, right])
        for left in range(width)
        for right in range(left + 1, width)
    ):
        raise InvalidGeometryError(f"{label} must not repeat a vertex within a face")
    return result


def _validated_mesh_arrays(
    vertices: object,
    triangles: object,
    quads: object,
    *,
    max_vertices: int,
    max_faces: int,
) -> tuple[Float32Array, Int32Array, Int32Array]:
    owned_vertices = _canonical_vertices(vertices, maximum=max_vertices)
    triangle_count = _declared_row_count(triangles, label="triangles", maximum=max_faces)
    quad_count = _declared_row_count(quads, label="quads", maximum=max_faces)
    if triangle_count is not None and quad_count is not None:
        face_count = triangle_count + quad_count
        if face_count > max_faces:
            raise ResourceLimitError(f"mesh has {face_count} faces; limit is {max_faces}")
    owned_triangles = _canonical_faces(triangles, 3, "triangles", maximum=max_faces)
    owned_quads = _canonical_faces(quads, 4, "quads", maximum=max_faces - len(owned_triangles))
    if not len(owned_vertices):
        raise InvalidGeometryError("mesh must contain at least one vertex")
    if not len(owned_triangles) and not len(owned_quads):
        raise InvalidGeometryError("mesh must contain at least one face")
    for label, faces in (("triangles", owned_triangles), ("quads", owned_quads)):
        if faces.size and int(faces.max()) >= len(owned_vertices):
            raise InvalidGeometryError(f"{label} reference a vertex outside the vertex array")
    return owned_vertices, owned_triangles, owned_quads


@dataclass(frozen=True, slots=True)
class Mesh:
    """Owned, validated world-space polygon mesh arrays."""

    vertices: Float32Array
    triangles: Int32Array = field(default_factory=lambda: _empty_faces(3))
    quads: Int32Array = field(default_factory=lambda: _empty_faces(4))

    def __post_init__(self) -> None:
        vertices, triangles, quads = _validated_mesh_arrays(
            self.vertices,
            self.triangles,
            self.quads,
            max_vertices=DEFAULT_LIMITS.max_vertices,
            max_faces=DEFAULT_LIMITS.max_faces,
        )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "quads", quads)

    @classmethod
    def _from_bounded_buffers(
        cls,
        *,
        vertices: object,
        triangles: object = (),
        quads: object = (),
        max_vertices: int,
        max_faces: int,
    ) -> Mesh:
        """Construct from caller buffers without any unbounded array conversion."""

        owned_vertices, owned_triangles, owned_quads = _validated_mesh_arrays(
            vertices,
            triangles,
            quads,
            max_vertices=max_vertices,
            max_faces=max_faces,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "vertices", owned_vertices)
        object.__setattr__(result, "triangles", owned_triangles)
        object.__setattr__(result, "quads", owned_quads)
        return result

    @property
    def face_count(self) -> int:
        return len(self.triangles) + len(self.quads)

    def triangulated_faces(self, *, maximum: int = DEFAULT_LIMITS.max_faces) -> Int32Array:
        """Return triangles, splitting each quad along its 0-2 diagonal."""

        face_count = len(self.triangles) + 2 * len(self.quads)
        if face_count > maximum:
            raise ResourceLimitError(
                f"triangulated mesh has {face_count} faces; limit is {maximum}"
            )
        if not len(self.quads):
            return self.triangles
        quad_triangles = np.empty((len(self.quads) * 2, 3), dtype=np.int32)
        quad_triangles[0::2] = self.quads[:, (0, 1, 2)]
        quad_triangles[1::2] = self.quads[:, (0, 2, 3)]
        return np.ascontiguousarray(
            np.concatenate((self.triangles, quad_triangles), axis=0), dtype=np.int32
        )


@dataclass(frozen=True, slots=True)
class VDBContents:
    """All grids and file-level metadata read from a VDB file."""

    grids: tuple[Any, ...]
    metadata: dict[str, Any]
