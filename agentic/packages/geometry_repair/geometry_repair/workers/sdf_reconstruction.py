# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process backend-neutral SDF reconstruction over owned NumPy mesh arrays."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

import numpy as np
import sdf_tools
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ..sdf_backend_qualification import (
    DEFAULT_SDF_BACKEND_ID,
    GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS,
    get_sdf_backend_qualification,
)

_RECONSTRUCTION_OPERATIONS = GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS
SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION = "geometry-repair.sdf-execution-evidence.v1"
_SDF_EXECUTION_EVIDENCE_ATTRIBUTE = "geometry_repair_sdf_execution_evidence"


class SdfReconstructionControls(BaseModel):
    """Validated controls and resource bounds for one reconstruction."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)

    backend_id: str = Field(default=DEFAULT_SDF_BACKEND_ID, pattern=r"^[a-z][a-z0-9-]*$")
    mode: Literal["signed", "unsigned_offset"] = "signed"
    voxel_size: float = Field(gt=0.0, le=1.0e100)
    half_width: float = Field(default=3.0, ge=2.0, le=16.0)
    offset_voxels: float = Field(default=1.5, gt=0.0, le=15.0)
    adaptivity: float = Field(default=0.0, ge=0.0, le=1.0)
    closing_steps: int = Field(default=2, ge=0, le=2)
    smoothing_steps: Literal[1] = 1
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    max_grid_dimension: int = Field(default=256, ge=32, le=512)
    max_input_vertices: int = Field(default=5_000_000, ge=4, le=10_000_000)
    max_input_faces: int = Field(default=5_000_000, ge=4, le=10_000_000)
    max_active_voxels: int = Field(default=50_000_000, ge=1_000, le=50_000_000)
    max_output_faces: int = Field(default=1_000_000, ge=1_000, le=5_000_000)

    @field_validator("smoothing_steps", mode="before")
    @classmethod
    def _validate_smoothing_steps_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("smoothing_steps must be the integer 1")
        return value

    @field_validator("backend_id")
    @classmethod
    def _require_explicit_backend(cls, value: str) -> str:
        if value == "auto":
            raise ValueError(
                "Geometry Repair requires an explicit qualified SDF backend; "
                "'auto' is reserved for exploratory sdf_tools use"
            )
        return value

    @model_validator(mode="after")
    def _validate_band(self) -> Self:
        if self.offset_voxels >= self.half_width:
            raise ValueError("offset_voxels must be below half_width")
        return self


class _StrictEvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)

    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in type(self).model_fields else default


class SdfArrayDigestEvidence(_StrictEvidenceModel):
    """Digest and canonical representation of one owned mesh array."""

    dtype: Literal["float32", "int32"]
    shape: tuple[int, int]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SdfMeshDigestEvidence(_StrictEvidenceModel):
    """Integrity records for one canonical triangle mesh."""

    vertices: SdfArrayDigestEvidence
    triangles: SdfArrayDigestEvidence


class SdfBackendSelectionRejectionEvidence(_StrictEvidenceModel):
    """One backend rejected while establishing the supplied session."""

    backend_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    reason: str = Field(min_length=1)


class SdfExecutionLimitsEvidence(_StrictEvidenceModel):
    """Exact portable limits applied to every backend call."""

    max_vertices: int = Field(gt=0)
    max_faces: int = Field(gt=0)
    max_active_voxels: int = Field(gt=0)
    max_voxel_extent: int = Field(gt=0)
    max_threads: int = Field(gt=0)
    max_sample_points: int = Field(gt=0)
    max_topology_steps: int = Field(gt=0)
    max_band_width_voxels: float = Field(gt=0.0)
    max_filter_width: int = Field(gt=0)
    max_filter_iterations: int = Field(gt=0)
    max_filter_work: int = Field(gt=0)
    max_file_bytes: int = Field(gt=0)
    max_fields: int = Field(gt=0)
    max_field_memory_bytes: int = Field(gt=0)
    max_total_field_memory_bytes: int = Field(gt=0)
    max_metadata_entries: int = Field(gt=0)
    max_metadata_bytes: int = Field(gt=0)


class SdfResourceBoundsEvidence(_StrictEvidenceModel):
    """Geometry Repair bounds plus the exact facade limits."""

    max_active_voxels: int = Field(gt=0)
    max_grid_dimension: int = Field(gt=0)
    max_input_faces: int = Field(gt=0)
    max_input_vertices: int = Field(gt=0)
    max_output_faces: int = Field(gt=0)
    max_output_vertices: int = Field(gt=0)
    execution_limits: SdfExecutionLimitsEvidence


class SdfAlgorithmEvidence(_StrictEvidenceModel):
    """Deterministic reconstruction route and its numeric controls."""

    route: Literal["signed_level_set", "signed_topology_closing", "unsigned_offset"]
    signed_level_set: bool
    fallback_used: Literal[False] = False
    explicit_closing_operator_applied: bool
    unsigned_distance_sampling: bool
    unsigned_offset_surface: bool
    gap_closing_budget_voxels: int = Field(ge=0)
    smoothing_steps: Literal[1]
    filter_operator: Literal["smooth_sdf", "smooth_scalar"]
    adaptivity: float = Field(ge=0.0, le=1.0)
    half_width: float = Field(gt=0.0)
    offset_voxels: float = Field(gt=0.0)
    voxel_size: float = Field(gt=0.0)
    isovalue: float
    repair_orientation: Literal[True] = True
    deterministic_seed: int = Field(ge=0)


class SdfResourceUsageEvidence(_StrictEvidenceModel):
    """Measured input use and optional accepted-candidate use."""

    estimated_grid_dimensions: tuple[int, int, int]
    input_array_bytes: int = Field(ge=0)
    input_boundary_edges: int = Field(ge=0)
    input_duplicate_faces: int = Field(ge=0)
    input_faces: int = Field(ge=0)
    input_non_manifold_edges: int = Field(ge=0)
    input_topology_defects: int = Field(ge=0)
    input_vertices: int = Field(ge=0)
    output_array_bytes: int | None = Field(default=None, ge=0)
    output_faces: int | None = Field(default=None, ge=0)
    output_vertices: int | None = Field(default=None, ge=0)
    grid_limits_enforced_by_driver: bool


class SdfGeometryEvidence(_StrictEvidenceModel):
    """Measurements over a candidate accepted by reconstruction validation."""

    output_surface_area: float = Field(gt=0.0)
    output_signed_volume: float = Field(gt=0.0)


class SdfDeterminismEvidence(_StrictEvidenceModel):
    canonical_vertex_order: Literal[True] = True
    canonical_face_order: Literal[True] = True
    normalized_signed_zero: Literal[True] = True
    shortest_quad_diagonal: Literal[True] = True
    single_thread: Literal[True] = True


class SdfExecutionEvidence(_StrictEvidenceModel):
    """Versioned backend execution and candidate-validation evidence."""

    schema_version: Literal["geometry-repair.sdf-execution-evidence.v1"] = (
        SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION
    )
    requested_backend_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    selected_backend_identity: dict[str, JsonValue] | None
    backend_qualification_id: str | None = Field(default=None, min_length=1)
    required_operations: tuple[str, ...] = Field(min_length=1)
    limits: SdfResourceBoundsEvidence
    backend_call_status: Literal["not_started", "succeeded", "failed", "unavailable"]
    candidate_validation_status: Literal["accepted", "rejected", "not_evaluated"]
    source_digests: SdfMeshDigestEvidence
    output_digests: SdfMeshDigestEvidence | None
    selection_rejections: tuple[SdfBackendSelectionRejectionEvidence, ...] = ()
    algorithm: SdfAlgorithmEvidence
    resource_usage: SdfResourceUsageEvidence
    geometry: SdfGeometryEvidence | None
    determinism: SdfDeterminismEvidence | None

    @field_validator("required_operations")
    @classmethod
    def _validate_required_operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("required_operations must be sorted and unique")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if self.selected_backend_identity is not None:
            selected_backend_id = self.selected_backend_identity.get("backend_id")
            if selected_backend_id != self.requested_backend_id:
                raise ValueError("selected backend identity must match the requested backend")
        if self.backend_call_status in {"succeeded", "failed"} and (
            self.selected_backend_identity is None
        ):
            raise ValueError("an attempted backend call requires a selected backend identity")
        if self.backend_call_status in {"succeeded", "failed"} and (
            self.backend_qualification_id is None
        ):
            raise ValueError("an attempted backend call requires a backend qualification ID")
        if self.backend_call_status != "succeeded" and (
            self.candidate_validation_status != "not_evaluated"
        ):
            raise ValueError("candidate validation cannot run before backend execution succeeds")
        accepted = self.candidate_validation_status == "accepted"
        if accepted != all(
            item is not None for item in (self.output_digests, self.geometry, self.determinism)
        ):
            raise ValueError(
                "accepted candidate evidence requires output digests, geometry, and determinism"
            )
        if accepted and self.backend_call_status != "succeeded":
            raise ValueError("an accepted candidate requires successful backend execution")
        output_usage = (
            self.resource_usage.output_array_bytes,
            self.resource_usage.output_faces,
            self.resource_usage.output_vertices,
        )
        if accepted != all(value is not None for value in output_usage):
            raise ValueError("accepted candidate evidence requires complete output resource usage")
        return self


@dataclass(frozen=True, slots=True)
class SdfReconstructionResult:
    """Canonical generated surface and its JSON-compatible execution evidence."""

    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]
    evidence: SdfExecutionEvidence


@dataclass(frozen=True, slots=True)
class _CanonicalSurface:
    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]
    surface_area: float
    signed_volume: float


def _owned_vertices(
    values: NDArray[Any], controls: SdfReconstructionControls
) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise sdf_tools.InvalidGeometryError("vertices must have shape (N, 3)")
    if array.dtype.kind not in "iuf":
        raise sdf_tools.InvalidGeometryError("vertices must be real numeric values")
    if len(array) < 4:
        raise sdf_tools.InvalidGeometryError("input must contain at least four vertices")
    if len(array) > controls.max_input_vertices:
        raise sdf_tools.ResourceLimitError("input exceeds max_input_vertices")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise sdf_tools.InvalidGeometryError("vertices must contain only finite float32 values")
    result[result == 0.0] = np.float32(0.0)
    return result


def _owned_triangles(
    values: NDArray[Any],
    vertex_count: int,
    controls: SdfReconstructionControls,
) -> NDArray[np.int32]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise sdf_tools.InvalidGeometryError("triangles must have shape (N, 3)")
    if array.dtype.kind not in "iu":
        raise sdf_tools.InvalidGeometryError("triangles must contain integer indices")
    if len(array) < 4:
        raise sdf_tools.InvalidGeometryError("input must contain at least four faces")
    if len(array) > controls.max_input_faces:
        raise sdf_tools.ResourceLimitError("input exceeds max_input_faces")
    if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int32).max)):
        raise sdf_tools.InvalidGeometryError("triangle indices must fit in nonnegative int32")
    result = np.array(array, dtype=np.int32, order="C", copy=True)
    if result.size and int(result.max()) >= vertex_count:
        raise sdf_tools.InvalidGeometryError(
            "triangles reference a vertex outside the vertex array"
        )
    repeated = (
        (result[:, 0] == result[:, 1])
        | (result[:, 1] == result[:, 2])
        | (result[:, 2] == result[:, 0])
    )
    if np.any(repeated):
        raise sdf_tools.InvalidGeometryError("input face repeats a vertex index")
    return result


def _execution_limits(controls: SdfReconstructionControls) -> sdf_tools.ExecutionLimits:
    return sdf_tools.ExecutionLimits(
        max_vertices=max(controls.max_input_vertices, _max_output_vertices(controls)),
        max_faces=max(controls.max_input_faces, controls.max_output_faces),
        max_active_voxels=controls.max_active_voxels,
        max_voxel_extent=controls.max_grid_dimension,
        max_threads=1,
        max_sample_points=1,
        max_topology_steps=max(1, controls.closing_steps),
        max_band_width_voxels=float(math.ceil(controls.half_width)),
        max_filter_width=1,
        max_filter_iterations=controls.smoothing_steps,
        max_filter_work=1_000_000_000,
    )


def _max_output_vertices(controls: SdfReconstructionControls) -> int:
    return min(3 * controls.max_output_faces, sdf_tools.DEFAULT_LIMITS.max_vertices)


def _grid_margin(
    controls: SdfReconstructionControls,
    *,
    topology_closing: bool,
) -> int:
    margin = math.ceil(controls.half_width) + 4
    if topology_closing:
        margin += math.ceil(controls.offset_voxels) + controls.closing_steps
    return margin


def _estimated_grid_dimensions(
    vertices: NDArray[np.float32],
    controls: SdfReconstructionControls,
    *,
    topology_closing: bool,
) -> tuple[int, int, int]:
    margin = _grid_margin(controls, topology_closing=topology_closing)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        voxel_minimum = np.floor(np.min(vertices, axis=0).astype(np.float64) / controls.voxel_size)
        voxel_maximum = np.ceil(np.max(vertices, axis=0).astype(np.float64) / controls.voxel_size)
    if not np.isfinite(voxel_minimum).all() or not np.isfinite(voxel_maximum).all():
        raise sdf_tools.ResourceLimitError(
            "mesh extent at the requested voxel size is not representable"
        )
    dimensions = voxel_maximum - voxel_minimum + 1 + 2 * margin
    if np.any(dimensions > controls.max_grid_dimension):
        raise sdf_tools.ResourceLimitError("estimated grid exceeds max_grid_dimension")
    return tuple(int(value) for value in dimensions)  # type: ignore[return-value]


def _topology_defect_counts(triangles: NDArray[np.int32]) -> tuple[int, int, int]:
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    canonical_faces = np.sort(triangles, axis=1)
    _unique_faces, face_counts = np.unique(canonical_faces, axis=0, return_counts=True)
    return (
        int(np.count_nonzero(counts == 1)),
        int(np.count_nonzero(counts > 2)),
        int(np.sum(face_counts - 1)),
    )


_SIGNED_OPERATIONS = frozenset(
    {
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.SMOOTH_SDF,
        sdf_tools.Operation.FIELD_TO_MESH,
    }
)

_SIGNED_TOPOLOGY_OPERATIONS = frozenset(
    {
        sdf_tools.Operation.MESH_TO_UDF,
        sdf_tools.Operation.ACTIVE_VALUE_MASK,
        sdf_tools.Operation.TOPOLOGY_TO_SDF,
        sdf_tools.Operation.EXTRACT_ENCLOSED_REGION,
        sdf_tools.Operation.SMOOTH_SDF,
        sdf_tools.Operation.FIELD_TO_MESH,
    }
)

_UNSIGNED_OFFSET_OPERATIONS = frozenset(
    {
        sdf_tools.Operation.MESH_TO_UDF,
        sdf_tools.Operation.SMOOTH_SCALAR,
        sdf_tools.Operation.FIELD_TO_MESH,
    }
)

if (
    _SIGNED_OPERATIONS | _SIGNED_TOPOLOGY_OPERATIONS | _UNSIGNED_OFFSET_OPERATIONS
) != GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS:
    raise RuntimeError("Geometry Repair SDF routes and backend qualification policy disagree")


def _route_operations(
    controls: SdfReconstructionControls,
    *,
    topology_defects: int,
) -> frozenset[sdf_tools.Operation]:
    if controls.mode == "unsigned_offset":
        return _UNSIGNED_OFFSET_OPERATIONS
    return _SIGNED_TOPOLOGY_OPERATIONS if topology_defects else _SIGNED_OPERATIONS


def _output_vertices(values: object) -> NDArray[np.float32]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.dtype.kind not in "iuf":
        raise sdf_tools.InvalidGeometryError(
            "SDF backend output vertices must be a numeric (N, 3) array"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise sdf_tools.InvalidGeometryError("SDF backend output contains a non-finite vertex")
    return result


def _output_faces(values: object, width: int, label: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1:] != (width,) or array.dtype.kind not in "iu":
        raise sdf_tools.InvalidGeometryError(
            f"SDF backend output {label} must be an integer (N, {width}) array"
        )
    if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
        raise sdf_tools.InvalidGeometryError(
            f"SDF backend output {label} contain an invalid vertex index"
        )
    return np.array(array, dtype=np.int64, order="C", copy=True)


def _triangulate_faces(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int64],
    quads: NDArray[np.int64],
) -> NDArray[np.int64]:
    vertex_count = len(vertices)
    for label, faces in (("triangle", triangles), ("quad", quads)):
        if faces.size and int(faces.max()) >= vertex_count:
            raise sdf_tools.InvalidGeometryError(
                f"SDF backend output {label} index is out of range"
            )
    if not len(quads):
        return np.array(triangles, dtype=np.int64, order="C", copy=True)

    with np.errstate(over="ignore", invalid="ignore"):
        diagonal_02 = vertices[quads[:, 0]] - vertices[quads[:, 2]]
        diagonal_13 = vertices[quads[:, 1]] - vertices[quads[:, 3]]
        length_02 = _length_squared_float32(diagonal_02)
        length_13 = _length_squared_float32(diagonal_13)
    use_02 = length_02 <= length_13
    first = np.where(use_02[:, None], quads[:, (0, 1, 2)], quads[:, (0, 1, 3)])
    second = np.where(use_02[:, None], quads[:, (0, 2, 3)], quads[:, (1, 2, 3)])
    quad_triangles = np.empty((2 * len(quads), 3), dtype=np.int64)
    quad_triangles[0::2] = first
    quad_triangles[1::2] = second
    return np.ascontiguousarray(np.concatenate((triangles, quad_triangles), axis=0))


def _rotate_minimum_first(triangles: NDArray[np.int64]) -> NDArray[np.int64]:
    offsets = np.argmin(triangles, axis=1)
    columns = (offsets[:, None] + np.arange(3, dtype=np.int64)) % 3
    return np.ascontiguousarray(np.take_along_axis(triangles, columns, axis=1))


def _cross_float32(left: NDArray[np.float32], right: NDArray[np.float32]) -> NDArray[np.float32]:
    result = np.empty_like(left, dtype=np.float32)
    result[:, 0] = left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1]
    result[:, 1] = left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2]
    result[:, 2] = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    return result


def _length_squared_float32(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.asarray(
        (vectors[:, 0] * vectors[:, 0] + vectors[:, 1] * vectors[:, 1])
        + vectors[:, 2] * vectors[:, 2],
        dtype=np.float32,
    )


def _sequential_float64_sum(initial: float, values: NDArray[np.float64]) -> float:
    accumulation = np.empty(len(values) + 1, dtype=np.float64)
    accumulation[0] = initial
    accumulation[1:] = values
    return float(np.cumsum(accumulation, dtype=np.float64)[-1])


def _signed_volume(vertices: NDArray[np.float32], triangles: NDArray[np.int64]) -> float:
    if not len(triangles):
        return 0.0

    canonical_triangles = _rotate_minimum_first(triangles)
    order = np.lexsort(
        (
            canonical_triangles[:, 2],
            canonical_triangles[:, 1],
            canonical_triangles[:, 0],
        )
    )
    canonical_triangles = canonical_triangles[order]
    origin = vertices[int(np.min(canonical_triangles))].astype(np.float64)
    volume = 0.0
    for start in range(0, len(canonical_triangles), 100_000):
        coordinates = vertices[canonical_triangles[start : start + 100_000]].astype(np.float64)
        coordinates -= origin
        with np.errstate(over="ignore", invalid="ignore"):
            cross = np.cross(coordinates[:, 1], coordinates[:, 2])
            contributions = np.einsum("ij,ij->i", coordinates[:, 0], cross)
            terms = contributions / 6.0
        volume = _sequential_float64_sum(volume, terms)
    return volume


def _surface_area(vertices: NDArray[np.float32], triangles: NDArray[np.int64]) -> float:
    area = 0.0
    for start in range(0, len(triangles), 100_000):
        coordinates = vertices[triangles[start : start + 100_000]]
        with np.errstate(over="ignore", invalid="ignore"):
            left = coordinates[:, 1] - coordinates[:, 0]
            right = coordinates[:, 2] - coordinates[:, 0]
            cross = _cross_float32(left, right)
            squared_lengths = _length_squared_float32(cross)
            lengths = np.sqrt(squared_lengths.astype(np.float64)).astype(np.float32)
            triangle_areas = lengths.astype(np.float64) * 0.5
        if not np.isfinite(triangle_areas).all() or np.any(triangle_areas <= 0.0):
            raise sdf_tools.InvalidGeometryError(
                "SDF backend output contains a degenerate triangle"
            )
        area = _sequential_float64_sum(area, triangle_areas)
    if not math.isfinite(area):
        raise sdf_tools.InvalidGeometryError("SDF backend output has non-finite surface area")
    return area


def _canonical_surface(mesh: object, controls: SdfReconstructionControls) -> _CanonicalSurface:
    vertices = _output_vertices(getattr(mesh, "vertices", None))
    triangles = _output_faces(getattr(mesh, "triangles", None), 3, "triangles")
    quads = _output_faces(getattr(mesh, "quads", None), 4, "quads")
    if not len(vertices) or (not len(triangles) and not len(quads)):
        raise sdf_tools.InvalidGeometryError("SDF backend produced an empty surface")
    output_face_count = len(triangles) + 2 * len(quads)
    if output_face_count > controls.max_output_faces:
        raise sdf_tools.ResourceLimitError("SDF backend output exceeds max_output_faces")
    if len(vertices) > _max_output_vertices(controls):
        raise sdf_tools.ResourceLimitError("SDF backend output exceeds max_output_vertices")

    triangles = _triangulate_faces(vertices, triangles, quads)
    original_indices = np.arange(len(vertices), dtype=np.int64)
    order = np.lexsort((original_indices, vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.empty(len(vertices), dtype=np.int64)
    remap[order] = original_indices
    vertices = np.array(vertices[order], dtype=np.float32, order="C", copy=True)
    vertices[vertices == 0.0] = np.float32(0.0)

    triangles = remap[triangles]
    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    )
    if np.any(repeated):
        raise sdf_tools.InvalidGeometryError("SDF backend output face repeats a vertex index")
    triangles = _rotate_minimum_first(triangles)

    volume = _signed_volume(vertices, triangles)
    if not math.isfinite(volume) or abs(volume) <= np.finfo(np.float64).eps:
        raise sdf_tools.InvalidGeometryError("SDF backend output has no stable enclosed volume")
    if volume < 0.0:
        triangles[:, (1, 2)] = triangles[:, (2, 1)]
        triangles = _rotate_minimum_first(triangles)
        volume = -volume

    order = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
    triangles = np.ascontiguousarray(triangles[order])
    if len(triangles) > 1 and np.any(np.all(triangles[1:] == triangles[:-1], axis=1)):
        raise sdf_tools.InvalidGeometryError("SDF backend output contains duplicate oriented faces")
    area = _surface_area(vertices, triangles)
    return _CanonicalSurface(
        vertices=np.array(vertices, dtype=np.float32, order="C", copy=True),
        triangles=np.array(triangles, dtype=np.int32, order="C", copy=True),
        surface_area=area,
        signed_volume=volume,
    )


def _array_evidence(array: NDArray[Any]) -> SdfArrayDigestEvidence:
    digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return SdfArrayDigestEvidence(
        dtype=array.dtype.name,
        shape=tuple(int(value) for value in array.shape),
        sha256=digest,
    )


def _mesh_digest_evidence(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.int32],
) -> SdfMeshDigestEvidence:
    return SdfMeshDigestEvidence(
        vertices=_array_evidence(vertices),
        triangles=_array_evidence(triangles),
    )


def _algorithm_evidence(
    controls: SdfReconstructionControls,
    *,
    topology_closing: bool,
) -> SdfAlgorithmEvidence:
    shell_radius = float(controls.offset_voxels * controls.voxel_size)
    if controls.mode == "unsigned_offset":
        route: Literal["signed_level_set", "signed_topology_closing", "unsigned_offset"] = (
            "unsigned_offset"
        )
        filter_operator: Literal["smooth_sdf", "smooth_scalar"] = "smooth_scalar"
        isovalue = shell_radius
    elif topology_closing:
        route = "signed_topology_closing"
        filter_operator = "smooth_sdf"
        isovalue = -shell_radius
    else:
        route = "signed_level_set"
        filter_operator = "smooth_sdf"
        isovalue = 0.0
    return SdfAlgorithmEvidence(
        route=route,
        signed_level_set=controls.mode == "signed",
        fallback_used=False,
        explicit_closing_operator_applied=topology_closing,
        unsigned_distance_sampling=topology_closing,
        unsigned_offset_surface=controls.mode == "unsigned_offset",
        gap_closing_budget_voxels=controls.closing_steps,
        smoothing_steps=controls.smoothing_steps,
        filter_operator=filter_operator,
        adaptivity=controls.adaptivity,
        half_width=controls.half_width,
        offset_voxels=controls.offset_voxels,
        voxel_size=controls.voxel_size,
        isovalue=isovalue,
        repair_orientation=True,
        deterministic_seed=controls.deterministic_seed,
    )


def _selection_rejection_evidence(
    session: sdf_tools.SdfSession,
) -> tuple[SdfBackendSelectionRejectionEvidence, ...]:
    return tuple(
        SdfBackendSelectionRejectionEvidence(
            backend_id=rejection.backend_id,
            reason=rejection.reason,
        )
        for rejection in getattr(session, "selection_rejections", ())
    )


def _execution_evidence(
    *,
    controls: SdfReconstructionControls,
    qualification_id: str | None,
    required_operations: frozenset[sdf_tools.Operation],
    limits: sdf_tools.ExecutionLimits,
    selected_backend_identity: dict[str, Any] | None,
    selection_rejections: tuple[SdfBackendSelectionRejectionEvidence, ...],
    algorithm: SdfAlgorithmEvidence,
    estimated_dimensions: tuple[int, int, int],
    source_vertices: NDArray[np.float32],
    source_triangles: NDArray[np.int32],
    boundary_edges: int,
    non_manifold_edges: int,
    duplicate_faces: int,
    backend_call_status: Literal["not_started", "succeeded", "failed", "unavailable"],
    candidate_validation_status: Literal["accepted", "rejected", "not_evaluated"],
    surface: _CanonicalSurface | None = None,
) -> SdfExecutionEvidence:
    topology_defects = boundary_edges + non_manifold_edges + duplicate_faces
    return SdfExecutionEvidence(
        requested_backend_id=controls.backend_id,
        selected_backend_identity=selected_backend_identity,
        backend_qualification_id=qualification_id,
        required_operations=tuple(sorted(operation.value for operation in required_operations)),
        limits=SdfResourceBoundsEvidence(
            max_active_voxels=controls.max_active_voxels,
            max_grid_dimension=controls.max_grid_dimension,
            max_input_faces=controls.max_input_faces,
            max_input_vertices=controls.max_input_vertices,
            max_output_faces=controls.max_output_faces,
            max_output_vertices=_max_output_vertices(controls),
            execution_limits=SdfExecutionLimitsEvidence(**asdict(limits)),
        ),
        backend_call_status=backend_call_status,
        candidate_validation_status=candidate_validation_status,
        source_digests=_mesh_digest_evidence(source_vertices, source_triangles),
        output_digests=(
            _mesh_digest_evidence(surface.vertices, surface.triangles)
            if surface is not None
            else None
        ),
        selection_rejections=selection_rejections,
        algorithm=algorithm,
        resource_usage=SdfResourceUsageEvidence(
            estimated_grid_dimensions=estimated_dimensions,
            input_array_bytes=source_vertices.nbytes + source_triangles.nbytes,
            input_boundary_edges=boundary_edges,
            input_duplicate_faces=duplicate_faces,
            input_faces=len(source_triangles),
            input_non_manifold_edges=non_manifold_edges,
            input_topology_defects=topology_defects,
            input_vertices=len(source_vertices),
            output_array_bytes=(
                surface.vertices.nbytes + surface.triangles.nbytes if surface is not None else None
            ),
            output_faces=len(surface.triangles) if surface is not None else None,
            output_vertices=len(surface.vertices) if surface is not None else None,
            grid_limits_enforced_by_driver=selected_backend_identity is not None,
        ),
        geometry=(
            SdfGeometryEvidence(
                output_surface_area=surface.surface_area,
                output_signed_volume=surface.signed_volume,
            )
            if surface is not None
            else None
        ),
        determinism=SdfDeterminismEvidence() if surface is not None else None,
    )


def _attach_execution_evidence(error: Exception, evidence: SdfExecutionEvidence) -> None:
    try:
        setattr(error, _SDF_EXECUTION_EVIDENCE_ATTRIBUTE, evidence)
    except (AttributeError, TypeError):
        return


def sdf_execution_evidence_from_exception(error: BaseException) -> SdfExecutionEvidence | None:
    """Return strict execution evidence attached to a reconstruction failure."""

    evidence = getattr(error, _SDF_EXECUTION_EVIDENCE_ATTRIBUTE, None)
    return evidence if isinstance(evidence, SdfExecutionEvidence) else None


def sdf_execution_failure_evidence(
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    controls: SdfReconstructionControls,
    *,
    backend_call_status: Literal["failed", "unavailable"],
    selected_backend_identity: dict[str, Any] | None = None,
    selection_rejections: tuple[SdfBackendSelectionRejectionEvidence, ...] = (),
) -> SdfExecutionEvidence:
    """Build a strict non-returning execution record at an outer worker boundary."""

    if not isinstance(controls, SdfReconstructionControls):
        raise TypeError("controls must be SdfReconstructionControls")
    owned_vertices = _owned_vertices(vertices, controls)
    owned_triangles = _owned_triangles(triangles, len(owned_vertices), controls)
    boundary_edges, non_manifold_edges, duplicate_faces = _topology_defect_counts(owned_triangles)
    topology_defects = boundary_edges + non_manifold_edges + duplicate_faces
    topology_closing = controls.mode == "signed" and topology_defects > 0
    qualification_id: str | None = None
    required_operations = _RECONSTRUCTION_OPERATIONS
    try:
        qualification = get_sdf_backend_qualification(controls.backend_id)
    except sdf_tools.BackendUnavailableError:
        pass
    else:
        qualification_id = qualification.qualification_id
        required_operations = qualification.required_operations
    return _execution_evidence(
        controls=controls,
        qualification_id=qualification_id,
        required_operations=required_operations,
        limits=_execution_limits(controls),
        selected_backend_identity=selected_backend_identity,
        selection_rejections=selection_rejections,
        algorithm=_algorithm_evidence(controls, topology_closing=topology_closing),
        estimated_dimensions=_estimated_grid_dimensions(
            owned_vertices,
            controls,
            topology_closing=topology_closing,
        ),
        source_vertices=owned_vertices,
        source_triangles=owned_triangles,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        duplicate_faces=duplicate_faces,
        backend_call_status=backend_call_status,
        candidate_validation_status="not_evaluated",
    )


def unavailable_sdf_execution_evidence(
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    controls: SdfReconstructionControls,
) -> SdfExecutionEvidence:
    """Build the canonical unavailable record when selection fails outside reconstruction."""

    return sdf_execution_failure_evidence(
        vertices,
        triangles,
        controls,
        backend_call_status="unavailable",
    )


def reconstruct_sdf_mesh(
    vertices: NDArray[Any],
    triangles: NDArray[Any],
    controls: SdfReconstructionControls,
    *,
    session: sdf_tools.SdfSession | None = None,
) -> SdfReconstructionResult:
    """Reconstruct one mesh through the backend-neutral SDF session."""

    if not isinstance(controls, SdfReconstructionControls):
        raise TypeError("controls must be SdfReconstructionControls")
    owned_vertices = _owned_vertices(vertices, controls)
    owned_triangles = _owned_triangles(triangles, len(owned_vertices), controls)
    boundary_edges, non_manifold_edges, duplicate_faces = _topology_defect_counts(owned_triangles)
    topology_defects = boundary_edges + non_manifold_edges + duplicate_faces
    topology_closing = controls.mode == "signed" and topology_defects > 0
    estimated_dimensions = _estimated_grid_dimensions(
        owned_vertices,
        controls,
        topology_closing=topology_closing,
    )
    limits = _execution_limits(controls)
    algorithm = _algorithm_evidence(controls, topology_closing=topology_closing)
    try:
        qualification = get_sdf_backend_qualification(controls.backend_id)
    except Exception as exc:
        _attach_execution_evidence(
            exc,
            _execution_evidence(
                controls=controls,
                qualification_id=None,
                required_operations=_RECONSTRUCTION_OPERATIONS,
                limits=limits,
                selected_backend_identity=None,
                selection_rejections=(),
                algorithm=algorithm,
                estimated_dimensions=estimated_dimensions,
                source_vertices=owned_vertices,
                source_triangles=owned_triangles,
                boundary_edges=boundary_edges,
                non_manifold_edges=non_manifold_edges,
                duplicate_faces=duplicate_faces,
                backend_call_status="unavailable",
                candidate_validation_status="not_evaluated",
            ),
        )
        raise
    required_operations = qualification.required_operations
    selected_backend_identity: dict[str, Any] | None = None
    selection_rejections: tuple[SdfBackendSelectionRejectionEvidence, ...] = ()

    def execution_evidence(
        *,
        backend_call_status: Literal["not_started", "succeeded", "failed", "unavailable"],
        candidate_validation_status: Literal["accepted", "rejected", "not_evaluated"],
        surface: _CanonicalSurface | None = None,
    ) -> SdfExecutionEvidence:
        return _execution_evidence(
            controls=controls,
            qualification_id=qualification.qualification_id,
            required_operations=required_operations,
            limits=limits,
            selected_backend_identity=selected_backend_identity,
            selection_rejections=selection_rejections,
            algorithm=algorithm,
            estimated_dimensions=estimated_dimensions,
            source_vertices=owned_vertices,
            source_triangles=owned_triangles,
            boundary_edges=boundary_edges,
            non_manifold_edges=non_manifold_edges,
            duplicate_faces=duplicate_faces,
            backend_call_status=backend_call_status,
            candidate_validation_status=candidate_validation_status,
            surface=surface,
        )

    try:
        if session is None:
            session = sdf_tools.create_session(
                backend=controls.backend_id,
                require=required_operations,
                limits=limits,
            )
        else:
            if not isinstance(session, sdf_tools.SdfSession):
                raise TypeError("session must be an sdf_tools.SdfSession")
            if session.backend_id != controls.backend_id:
                raise sdf_tools.BackendMismatchError(
                    f"session uses backend {session.backend_id!r}; controls require "
                    f"{controls.backend_id!r}"
                )
            if session.execution_limits != limits:
                raise ValueError("session execution limits do not match reconstruction controls")
        selected_backend_identity = session.backend_info.as_dict()
        selection_rejections = _selection_rejection_evidence(session)
        qualification.validate(selected_backend_identity)
    except Exception as exc:
        unavailable = isinstance(
            exc,
            sdf_tools.BackendUnavailableError | sdf_tools.CapabilityUnavailableError,
        )
        _attach_execution_evidence(
            exc,
            execution_evidence(
                backend_call_status="unavailable" if unavailable else "not_started",
                candidate_validation_status="not_evaluated",
            ),
        )
        raise

    backend_identity = selected_backend_identity
    assert backend_identity is not None
    mesh = sdf_tools.Mesh(vertices=owned_vertices, triangles=owned_triangles)

    shell_radius = float(controls.offset_voxels * controls.voxel_size)
    topology_half_width = math.ceil(controls.half_width)
    try:
        if controls.mode == "signed" and not topology_closing:
            grid = session.mesh_to_sdf(
                mesh,
                voxel_size=controls.voxel_size,
                half_width=controls.half_width,
                thread_count=1,
            )
            grid = session.smooth(
                grid,
                width=1,
                iterations=controls.smoothing_steps,
                thread_count=1,
            )
        elif topology_closing:
            unsigned_distance = session.mesh_to_udf(
                mesh,
                voxel_size=controls.voxel_size,
                half_width=controls.half_width,
                thread_count=1,
            )
            shell_mask = session.active_value_mask(
                unsigned_distance,
                max_value=shell_radius,
                thread_count=1,
            )
            shell_level_set = session.topology_to_sdf(
                shell_mask,
                half_width=topology_half_width,
                closing_steps=controls.closing_steps,
                dilation=0,
                smoothing_steps=0,
                thread_count=1,
            )
            filled_region = session.extract_enclosed_region(
                shell_level_set,
                thread_count=1,
            )
            grid = session.topology_to_sdf(
                filled_region,
                half_width=topology_half_width,
                closing_steps=0,
                dilation=0,
                smoothing_steps=0,
                thread_count=1,
            )
            grid = session.smooth(
                grid,
                width=1,
                iterations=controls.smoothing_steps,
                thread_count=1,
            )
        else:
            grid = session.mesh_to_udf(
                mesh,
                voxel_size=controls.voxel_size,
                half_width=controls.half_width,
                thread_count=1,
            )
            grid = session.smooth(
                grid,
                width=1,
                iterations=controls.smoothing_steps,
                thread_count=1,
            )

        native_mesh = session.field_to_mesh(
            grid,
            isovalue=algorithm.isovalue,
            adaptivity=controls.adaptivity,
            repair_orientation=True,
            thread_count=1,
        )
    except Exception as exc:
        unavailable = isinstance(
            exc,
            sdf_tools.BackendUnavailableError | sdf_tools.CapabilityUnavailableError,
        )
        _attach_execution_evidence(
            exc,
            execution_evidence(
                backend_call_status="unavailable" if unavailable else "failed",
                candidate_validation_status="not_evaluated",
            ),
        )
        raise

    try:
        surface = _canonical_surface(native_mesh, controls)
    except Exception as exc:
        _attach_execution_evidence(
            exc,
            execution_evidence(
                backend_call_status="succeeded",
                candidate_validation_status="rejected",
            ),
        )
        raise

    evidence = execution_evidence(
        backend_call_status="succeeded",
        candidate_validation_status="accepted",
        surface=surface,
    )
    return SdfReconstructionResult(
        vertices=surface.vertices,
        triangles=surface.triangles,
        evidence=evidence,
    )


__all__ = [
    "SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION",
    "SdfExecutionEvidence",
    "SdfReconstructionControls",
    "SdfReconstructionResult",
    "reconstruct_sdf_mesh",
    "sdf_execution_evidence_from_exception",
    "sdf_execution_failure_evidence",
    "unavailable_sdf_execution_evidence",
]
