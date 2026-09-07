# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public value types for backend-neutral SDF operations."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .errors import BackendMismatchError, InvalidGeometryError


class Operation(StrEnum):
    """Semantic operations that an SDF backend may implement."""

    READ_FIELDS = "read_fields"
    WRITE_FIELDS = "write_fields"
    MESH_TO_SDF = "mesh_to_sdf"
    MESH_TO_UDF = "mesh_to_udf"
    FIELD_TO_MESH = "field_to_mesh"
    UNION = "union"
    INTERSECTION = "intersection"
    DIFFERENCE = "difference"
    OFFSET = "offset"
    SMOOTH_SDF = "smooth_sdf"
    SMOOTH_SCALAR = "smooth_scalar"
    NORMALIZE_SDF = "normalize_sdf"
    REBUILD_SDF = "rebuild_sdf"
    RESAMPLE_TO_MATCH = "resample_to_match"
    SAMPLE_VALUES = "sample_values"
    SAMPLE_GRADIENTS = "sample_gradients"
    ACTIVE_VALUE_MASK = "active_value_mask"
    TOPOLOGY_TO_SDF = "topology_to_sdf"
    EXTRACT_ENCLOSED_REGION = "extract_enclosed_region"


class FieldKind(StrEnum):
    """Semantic meaning of values stored in a backend field."""

    SIGNED_DISTANCE = "signed_distance"
    UNSIGNED_DISTANCE = "unsigned_distance"
    SCALAR = "scalar"
    MASK = "mask"
    UNKNOWN = "unknown"


class Interpolation(StrEnum):
    """Sampling and resampling kernels shared by admitted backends."""

    NEAREST = "nearest"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


def _empty_faces() -> tuple[tuple[int, ...], ...]:
    return ()


def _shape(values: object) -> tuple[int, ...] | None:
    raw_shape = getattr(values, "shape", None)
    if raw_shape is None:
        return None
    try:
        return tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidGeometryError("mesh array has a malformed shape") from exc


def _validate_width(values: object, width: int, label: str) -> None:
    shape = _shape(values)
    if shape is not None and (len(shape) != 2 or shape[1] != width):
        raise InvalidGeometryError(f"{label} must have shape (N, {width})")


@dataclass(frozen=True, slots=True)
class Mesh:
    """Backend-neutral world-space polygon mesh buffers.

    The core package intentionally has no numerical-runtime dependency. Drivers
    snapshot, canonicalize, and fully validate these array-like buffers before
    native execution.
    """

    vertices: Any
    triangles: Any = field(default_factory=_empty_faces)
    quads: Any = field(default_factory=_empty_faces)

    def __post_init__(self) -> None:
        _validate_width(self.vertices, 3, "vertices")
        _validate_width(self.triangles, 3, "triangles")
        _validate_width(self.quads, 4, "quads")
        if not len(self.vertices):
            raise InvalidGeometryError("mesh must contain at least one vertex")
        if not len(self.triangles) and not len(self.quads):
            raise InvalidGeometryError("mesh must contain at least one face")

    @property
    def face_count(self) -> int:
        return len(self.triangles) + len(self.quads)


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Caller-visible bounds applied by every admitted backend."""

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
        integer_fields = (
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
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        width = self.max_band_width_voxels
        if isinstance(width, bool) or not isinstance(width, numbers.Real):
            raise TypeError("max_band_width_voxels must be a real number")
        if not math.isfinite(width) or width <= 0:
            raise ValueError("max_band_width_voxels must be finite and positive")


DEFAULT_LIMITS = ExecutionLimits()


@dataclass(frozen=True, slots=True)
class Field:
    """Opaque field owned by one backend implementation."""

    backend_id: str
    kind: FieldKind
    _payload: Any = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)

    def _is_owned_by(self, owner: object) -> bool:
        return self._owner is owner

    def _payload_for(self, backend_id: str, owner: object) -> Any:
        """Return the opaque payload to its owning implementation driver."""

        if backend_id != self.backend_id or not self._is_owned_by(owner):
            raise BackendMismatchError(
                f"field does not belong to this {backend_id!r} backend implementation"
            )
        return self._payload


@dataclass(frozen=True, slots=True)
class FieldContents:
    """Fields and file metadata returned by an explicit format reader."""

    fields: tuple[Field, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identity and provenance for one selected backend implementation."""

    backend_id: str
    implementation_version: str
    operations: frozenset[Operation]
    execution_mode: str
    read_formats: frozenset[str]
    write_formats: frozenset[str]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("read_formats", "write_formats"):
            formats = getattr(self, name)
            if not isinstance(formats, frozenset):
                raise TypeError(f"{name} must be a frozenset")
            if any(not isinstance(value, str) or not value for value in formats):
                raise ValueError(f"{name} must contain nonempty string identifiers")
        if self.read_formats and Operation.READ_FIELDS not in self.operations:
            raise ValueError("read_formats require the read_fields operation")
        if self.write_formats and Operation.WRITE_FIELDS not in self.operations:
            raise ValueError("write_formats require the write_fields operation")
        try:
            canonical = json.loads(
                json.dumps(
                    _thaw_json(self.provenance),
                    allow_nan=False,
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("backend provenance must be finite JSON data") from exc
        if not isinstance(canonical, dict):
            raise TypeError("backend provenance must be a JSON object")
        object.__setattr__(self, "provenance", _freeze_json(canonical))

    @property
    def supported_formats(self) -> frozenset[str]:
        """Compatibility view of every readable or writable artifact format."""

        return self.read_formats | self.write_formats

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "implementation_version": self.implementation_version,
            "operations": sorted(operation.value for operation in self.operations),
            "execution_mode": self.execution_mode,
            "read_formats": sorted(self.read_formats),
            "write_formats": sorted(self.write_formats),
            "supported_formats": sorted(self.supported_formats),
            "provenance": _thaw_json(self.provenance),
        }


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
