# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic routing policy for semantic mesh segmentation.

This module only decides which producer or consumer path owns the next step. It
does not consume a completed run, split a mesh, invoke an agent, or export USD.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SEGMENTATION_ROUTING_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-segmentation-routing.v1"
)

SegmentationRoute = Literal[
    "consume_completed_run",
    "not_requested",
    "reuse_source_identity",
    "deterministic_shell_split",
    "agentic_semantic_segmentation",
    "rejected",
    "undetermined",
]
SemanticIdentitySource = Literal[
    "named_mesh_prim",
    "geom_subset",
    "semantic_part_metadata",
]

_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_MAX_SOURCE_FILE_BYTES = 1 * 1024 * 1024 * 1024
_MAX_INPUT_PATH_LENGTH = 4096
_MAX_PRIM_PATH_LENGTH = 2048
_MAX_PRIM_COUNT = 100_000
_MAX_MESH_PRIM_COUNT = 1024
_MAX_GEOM_SUBSET_COUNT = 8192
_MAX_POINTS_PER_MESH = 2_000_000
_MAX_FACES_PER_MESH = 750_000
_MAX_FACE_VERTEX_INDICES_PER_MESH = 3_000_000
_MAX_TOTAL_POINTS = 4_000_000
_MAX_TOTAL_FACES = 1_250_000
_MAX_TOTAL_FACE_VERTEX_INDICES = 5_000_000
_MAX_TOTAL_GEOM_SUBSET_INDICES = 20_000_000
# Extremely large n-gons make edge expansion and downstream triangulation
# disproportionately expensive. Require pathological faces to be pre-triangulated.
_MAX_VERTICES_PER_FACE = 10_000
_MAX_SEMANTIC_NAMES = 4096
# A valid USD identifier can occupy almost the complete bounded prim path. Keep
# source-authored identity intact while retaining a hard serialization bound.
_MAX_SEMANTIC_NAME_LENGTH = _MAX_PRIM_PATH_LENGTH
_MAX_TOPOLOGY_WORKING_BYTES = 384 * 1024 * 1024

# Conservative CPython working-set model for the compact buffers and packed
# edge table below. It intentionally overestimates each unique edge because
# dictionary entry sizes vary by Python build and allocator state.
_NORMALIZED_POINT_BYTES = 3 * 8
_FACE_BUFFER_BYTES = 2 * 4
_INDEX_BUFFER_BYTES = 4
_PACKED_EDGE_TABLE_BYTES = 160


class SegmentationSemanticNameEvidence(BaseModel):
    """One source-authored name retained without upgrading it to geometry proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=_MAX_SEMANTIC_NAME_LENGTH)
    sources: list[SemanticIdentitySource] = Field(default_factory=list)
    prim_paths: list[str] = Field(default_factory=list)
    geometry_semantics_proven: Literal[False] = False


class GeometrySegmentationRoutingMetrics(BaseModel):
    """Bounded metrics produced by source inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file_bytes: int | None = Field(default=None, ge=0)
    inspected_prim_count: int = Field(default=0, ge=0)
    unloaded_payload_count: int = Field(default=0, ge=0)
    mesh_prim_count: int = Field(default=0, ge=0)
    named_mesh_prim_count: int = Field(default=0, ge=0)
    geom_subset_count: int = Field(default=0, ge=0)
    geom_subset_face_index_count: int = Field(default=0, ge=0)
    supplied_semantic_part_count: int = Field(default=0, ge=0)
    point_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    face_vertex_index_count: int = Field(default=0, ge=0)
    triangular_face_count: int = Field(default=0, ge=0)
    non_triangular_face_count: int = Field(default=0, ge=0)
    unique_edge_count: int = Field(default=0, ge=0)
    non_manifold_edge_count: int = Field(default=0, ge=0)
    edge_connected_shell_count: int | None = Field(default=None, ge=0)
    shell_face_counts: list[int] = Field(default_factory=list)
    malformed_mesh_count: int = Field(default=0, ge=0)
    topology_inspected: bool = False
    all_faces_triangular: bool | None = None
    mesh_prim_paths: list[str] = Field(default_factory=list)


class GeometrySegmentationRoutingDecision(BaseModel):
    """Typed top-level decision for semantic mesh segmentation ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SEGMENTATION_ROUTING_SCHEMA_VERSION
    route: SegmentationRoute
    requested: bool
    source_usd_path: str | None = Field(
        default=None,
        max_length=_MAX_INPUT_PATH_LENGTH,
    )
    completed_run_reference: str | None = Field(
        default=None,
        max_length=_MAX_INPUT_PATH_LENGTH,
    )
    required_semantic_names: list[str] = Field(default_factory=list)
    observed_source_semantic_names: list[str] = Field(default_factory=list)
    missing_required_semantic_names: list[str] = Field(default_factory=list)
    semantic_name_evidence: list[SegmentationSemanticNameEvidence] = Field(
        default_factory=list
    )
    should_invoke_agentic_segmentation: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(min_length=1)
    metrics: GeometrySegmentationRoutingMetrics = Field(
        default_factory=GeometrySegmentationRoutingMetrics
    )


# Short aliases keep the contract convenient for callers outside Geometry.
SegmentationRoutingDecision = GeometrySegmentationRoutingDecision
SegmentationRoutingMetrics = GeometrySegmentationRoutingMetrics


@dataclass(frozen=True)
class _IdentityRecord:
    name: str
    source: SemanticIdentitySource
    prim_path: str | None = None


@dataclass(frozen=True)
class _MeshTopology:
    prim_path: str
    prim_name: str
    face_count: int
    all_faces_triangular: bool
    shell_face_counts: tuple[int, ...]


@dataclass
class _MetricsBuilder:
    source_file_bytes: int | None = None
    inspected_prim_count: int = 0
    unloaded_payload_count: int = 0
    mesh_prim_count: int = 0
    named_mesh_prim_count: int = 0
    geom_subset_count: int = 0
    geom_subset_face_index_count: int = 0
    supplied_semantic_part_count: int = 0
    point_count: int = 0
    face_count: int = 0
    face_vertex_index_count: int = 0
    triangular_face_count: int = 0
    non_triangular_face_count: int = 0
    unique_edge_count: int = 0
    non_manifold_edge_count: int = 0
    shell_face_counts: list[int] = field(default_factory=list)
    malformed_mesh_count: int = 0
    topology_inspected: bool = False
    mesh_prim_paths: list[str] = field(default_factory=list)

    def model(self) -> GeometrySegmentationRoutingMetrics:
        shell_counts = sorted(self.shell_face_counts, reverse=True)
        return GeometrySegmentationRoutingMetrics(
            source_file_bytes=self.source_file_bytes,
            inspected_prim_count=self.inspected_prim_count,
            unloaded_payload_count=self.unloaded_payload_count,
            mesh_prim_count=self.mesh_prim_count,
            named_mesh_prim_count=self.named_mesh_prim_count,
            geom_subset_count=self.geom_subset_count,
            geom_subset_face_index_count=self.geom_subset_face_index_count,
            supplied_semantic_part_count=self.supplied_semantic_part_count,
            point_count=self.point_count,
            face_count=self.face_count,
            face_vertex_index_count=self.face_vertex_index_count,
            triangular_face_count=self.triangular_face_count,
            non_triangular_face_count=self.non_triangular_face_count,
            unique_edge_count=self.unique_edge_count,
            non_manifold_edge_count=self.non_manifold_edge_count,
            edge_connected_shell_count=(
                len(shell_counts) if self.topology_inspected else None
            ),
            shell_face_counts=shell_counts,
            malformed_mesh_count=self.malformed_mesh_count,
            topology_inspected=self.topology_inspected,
            all_faces_triangular=(
                self.non_triangular_face_count == 0 if self.topology_inspected else None
            ),
            mesh_prim_paths=sorted(self.mesh_prim_paths),
        )


class _InspectionFailure(ValueError):
    def __init__(
        self,
        *,
        route: Literal["rejected", "undetermined"],
        code: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.route = route
        self.code = code
        self.reason = reason


def _plain_semantic_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    name = value.strip()
    if (
        len(name) > _MAX_SEMANTIC_NAME_LENGTH
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError(f"{label} must be a bounded plain semantic name")
    return name


def _source_semantic_name(value: Any, *, label: str) -> str:
    try:
        return _plain_semantic_name(value, label=label)
    except ValueError as exc:
        raise _InspectionFailure(
            route="rejected",
            code="malformed_source_semantic_name",
            reason=str(exc),
        ) from exc


def _required_names(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str | bytes):
        raise ValueError("required_semantic_names must be a sequence of names")
    if len(values) > _MAX_SEMANTIC_NAMES:
        raise ValueError(
            f"required_semantic_names exceeds the {_MAX_SEMANTIC_NAMES} name limit"
        )
    return sorted(
        {
            _plain_semantic_name(value, label="required semantic name")
            for value in values
        }
    )


def _metadata_identity_records(
    values: Sequence[str | Mapping[str, Any]] | None,
    *,
    strict: bool,
) -> list[_IdentityRecord]:
    if values is None:
        return []
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        if strict:
            raise ValueError("supplied_semantic_parts must be a sequence")
        return []
    if len(values) > _MAX_SEMANTIC_NAMES:
        if strict:
            raise ValueError(
                f"supplied_semantic_parts exceeds the {_MAX_SEMANTIC_NAMES} part limit"
            )
        return []

    records: list[_IdentityRecord] = []
    for index, item in enumerate(values):
        raw_name = (
            item
            if isinstance(item, str)
            else item.get("name")
            if isinstance(item, Mapping)
            else None
        )
        try:
            name = _plain_semantic_name(
                raw_name,
                label=f"supplied semantic part {index} name",
            )
        except (AttributeError, ValueError):
            if strict:
                raise ValueError(
                    f"supplied semantic part {index} must provide a valid name"
                ) from None
            continue
        records.append(_IdentityRecord(name=name, source="semantic_part_metadata"))
    return records


def _bounded_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text or len(text) > _MAX_INPUT_PATH_LENGTH:
        return None
    return text


def _semantic_name_evidence(
    records: Sequence[_IdentityRecord],
) -> tuple[list[str], list[SegmentationSemanticNameEvidence]]:
    sources_by_name: dict[str, set[SemanticIdentitySource]] = defaultdict(set)
    paths_by_name: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sources_by_name[record.name].add(record.source)
        if record.prim_path is not None:
            paths_by_name[record.name].add(record.prim_path)

    source_order = {
        "named_mesh_prim": 0,
        "geom_subset": 1,
        "semantic_part_metadata": 2,
    }
    names = sorted(sources_by_name)
    evidence = [
        SegmentationSemanticNameEvidence(
            name=name,
            sources=sorted(sources_by_name[name], key=source_order.__getitem__),
            prim_paths=sorted(paths_by_name[name]),
        )
        for name in names
    ]
    return names, evidence


def _decision(
    *,
    route: SegmentationRoute,
    requested: bool,
    required_names: list[str],
    identity_records: Sequence[_IdentityRecord],
    reason_codes: Sequence[str],
    reasons: Sequence[str],
    metrics: _MetricsBuilder,
    source_usd_path: str | None = None,
    completed_run_reference: str | None = None,
) -> GeometrySegmentationRoutingDecision:
    observed_names, evidence = _semantic_name_evidence(identity_records)
    missing_names = sorted(set(required_names).difference(observed_names))
    final_codes = list(reason_codes)
    final_reasons = list(reasons)
    if missing_names:
        final_codes.append("required_semantic_names_missing_from_source_identity")
        final_reasons.append(
            f"{len(missing_names)} required semantic name(s) are absent from "
            "source identity evidence."
        )
    if evidence and route == "reuse_source_identity":
        final_codes.append("source_names_are_not_geometry_proof")
        final_reasons.append(
            "Source-authored names are identity assertions only; this routing "
            "decision does not prove their semantic geometry."
        )
    return GeometrySegmentationRoutingDecision(
        route=route,
        requested=requested,
        source_usd_path=source_usd_path,
        completed_run_reference=completed_run_reference,
        required_semantic_names=required_names,
        observed_source_semantic_names=observed_names,
        missing_required_semantic_names=missing_names,
        semantic_name_evidence=evidence,
        should_invoke_agentic_segmentation=(route == "agentic_semantic_segmentation"),
        reason_codes=final_codes,
        reasons=final_reasons,
        metrics=metrics.model(),
    )


def _fail_malformed(metrics: _MetricsBuilder, reason: str) -> None:
    metrics.malformed_mesh_count += 1
    raise _InspectionFailure(
        route="rejected",
        code="malformed_mesh_topology",
        reason=reason,
    )


def _bounded_attribute_value(
    attribute: Any,
    *,
    label: str,
    metrics: _MetricsBuilder,
) -> Any:
    if not attribute or not attribute.IsValid():
        _fail_malformed(metrics, f"Mesh {label} attribute is missing.")
    if attribute.ValueMightBeTimeVarying():
        _fail_malformed(
            metrics,
            f"Time-varying mesh {label} is unsupported for deterministic routing.",
        )
    try:
        value = attribute.Get()
    except Exception:
        _fail_malformed(metrics, f"Mesh {label} could not be read with OpenUSD.")
    if value is None:
        _fail_malformed(metrics, f"Mesh {label} has no default value.")
    return value


def _point_coordinates(
    point: Any, metrics: _MetricsBuilder
) -> tuple[float, float, float]:
    try:
        coordinates = tuple(float(value) for value in point)
    except (TypeError, ValueError, OverflowError):
        _fail_malformed(metrics, "Mesh points contain a non-numeric value.")
    if len(coordinates) != 3 or not all(math.isfinite(value) for value in coordinates):
        _fail_malformed(metrics, "Mesh points must contain finite 3D coordinates.")
    return coordinates[0], coordinates[1], coordinates[2]


def _triangle_has_finite_area(
    points: Sequence[float],
    face: Sequence[int],
) -> bool:
    a_offset = face[0] * 3
    b_offset = face[1] * 3
    c_offset = face[2] * 3
    a = points[a_offset], points[a_offset + 1], points[a_offset + 2]
    b = points[b_offset], points[b_offset + 1], points[b_offset + 2]
    c = points[c_offset], points[c_offset + 1], points[c_offset + 2]
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    area_squared = sum(value * value for value in cross)
    return math.isfinite(area_squared) and area_squared > 0.0


def _estimated_topology_working_bytes(
    *,
    point_count: int,
    face_count: int,
    index_count: int,
) -> int:
    """Conservatively bound peak memory before allocating topology helpers."""

    return (
        point_count * _NORMALIZED_POINT_BYTES
        + face_count * _FACE_BUFFER_BYTES
        + index_count * (_INDEX_BUFFER_BYTES + _PACKED_EDGE_TABLE_BYTES)
    )


def _shell_face_counts(
    face_counts: Sequence[int],
    face_indices: Sequence[int],
) -> tuple[tuple[int, ...], int, int]:
    parents = array("I", range(len(face_counts)))

    def find(face_id: int) -> int:
        while parents[face_id] != face_id:
            parents[face_id] = parents[parents[face_id]]
            face_id = parents[face_id]
        return face_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parents[right_root] = left_root
        else:
            parents[left_root] = right_root

    # Each edge key packs two uint32 vertex ids. The value packs the first
    # owning face plus a use count saturated at three, which is sufficient to
    # identify non-manifold edges without retaining a second dictionary.
    edge_state: dict[int, int] = {}
    cursor = 0
    for face_id, face_size in enumerate(face_counts):
        first = face_indices[cursor]
        start = first
        for offset in range(1, face_size):
            end = face_indices[cursor + offset]
            low, high = (start, end) if start < end else (end, start)
            edge = (low << 32) | high
            state = edge_state.get(edge)
            if state is None:
                edge_state[edge] = (face_id << 2) | 1
            else:
                owner = state >> 2
                union(owner, face_id)
                edge_state[edge] = (owner << 2) | min((state & 3) + 1, 3)
            start = end
        low, high = (start, first) if start < first else (first, start)
        edge = (low << 32) | high
        state = edge_state.get(edge)
        if state is None:
            edge_state[edge] = (face_id << 2) | 1
        else:
            owner = state >> 2
            union(owner, face_id)
            edge_state[edge] = (owner << 2) | min((state & 3) + 1, 3)
        cursor += face_size

    counts = Counter(find(face_id) for face_id in range(len(face_counts)))
    return (
        tuple(sorted(counts.values(), reverse=True)),
        len(edge_state),
        sum((state & 3) == 3 for state in edge_state.values()),
    )


def _inspect_mesh(mesh_prim: Any, metrics: _MetricsBuilder) -> _MeshTopology:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    path = str(mesh_prim.GetPath())
    points = _bounded_attribute_value(
        mesh.GetPointsAttr(),
        label="points",
        metrics=metrics,
    )
    counts = _bounded_attribute_value(
        mesh.GetFaceVertexCountsAttr(),
        label="faceVertexCounts",
        metrics=metrics,
    )
    indices = _bounded_attribute_value(
        mesh.GetFaceVertexIndicesAttr(),
        label="faceVertexIndices",
        metrics=metrics,
    )

    point_count = len(points)
    face_count = len(counts)
    index_count = len(indices)
    if point_count < 3 or face_count < 1 or index_count < 3:
        _fail_malformed(metrics, f"Mesh {path} has empty or incomplete topology.")
    if point_count > _MAX_POINTS_PER_MESH:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=f"Mesh point count exceeds {_MAX_POINTS_PER_MESH}.",
        )
    if face_count > _MAX_FACES_PER_MESH:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=f"Mesh face count exceeds {_MAX_FACES_PER_MESH}.",
        )
    if index_count > _MAX_FACE_VERTEX_INDICES_PER_MESH:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=(
                "Mesh face-vertex index count exceeds "
                f"{_MAX_FACE_VERTEX_INDICES_PER_MESH}."
            ),
        )
    if metrics.point_count + point_count > _MAX_TOTAL_POINTS:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=f"Total mesh point count exceeds {_MAX_TOTAL_POINTS}.",
        )
    if metrics.face_count + face_count > _MAX_TOTAL_FACES:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=f"Total mesh face count exceeds {_MAX_TOTAL_FACES}.",
        )
    if metrics.face_vertex_index_count + index_count > _MAX_TOTAL_FACE_VERTEX_INDICES:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=(
                "Total mesh face-vertex index count exceeds "
                f"{_MAX_TOTAL_FACE_VERTEX_INDICES}."
            ),
        )

    estimated_working_bytes = _estimated_topology_working_bytes(
        point_count=point_count,
        face_count=face_count,
        index_count=index_count,
    )
    if estimated_working_bytes > _MAX_TOPOLOGY_WORKING_BYTES:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_memory_budget_exceeded",
            reason=(
                "Estimated mesh topology inspection working set exceeds "
                f"{_MAX_TOPOLOGY_WORKING_BYTES} bytes. Pre-split or simplify the "
                "source before semantic routing."
            ),
        )

    normalized_points = array("d")
    for point in points:
        normalized_points.extend(_point_coordinates(point, metrics))

    expected_index_count = 0
    normalized_counts = array("I")
    for raw_count in counts:
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            _fail_malformed(metrics, f"Mesh {path} has a non-integer face count.")
        count = int(raw_count)
        if count < 3 or count > _MAX_VERTICES_PER_FACE:
            _fail_malformed(metrics, f"Mesh {path} has an invalid face size.")
        expected_index_count += count
        normalized_counts.append(count)
    if expected_index_count != index_count:
        _fail_malformed(
            metrics,
            f"Mesh {path} face counts do not match its face-vertex indices.",
        )

    normalized_indices = array("I")
    for raw_index in indices:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            _fail_malformed(metrics, f"Mesh {path} has a non-integer vertex index.")
        index = int(raw_index)
        if index < 0 or index >= point_count:
            _fail_malformed(metrics, f"Mesh {path} has an out-of-range vertex index.")
        normalized_indices.append(index)

    cursor = 0
    for count in normalized_counts:
        face = normalized_indices[cursor : cursor + count]
        cursor += count
        if len(set(face)) != len(face):
            _fail_malformed(metrics, f"Mesh {path} has a repeated vertex in a face.")
        if count == 3 and not _triangle_has_finite_area(normalized_points, face):
            _fail_malformed(metrics, f"Mesh {path} has a degenerate triangle.")

    shell_counts, unique_edges, non_manifold_edges = _shell_face_counts(
        normalized_counts,
        normalized_indices,
    )
    triangular_faces = sum(count == 3 for count in normalized_counts)
    metrics.point_count += point_count
    metrics.face_count += face_count
    metrics.face_vertex_index_count += index_count
    metrics.triangular_face_count += triangular_faces
    metrics.non_triangular_face_count += face_count - triangular_faces
    metrics.unique_edge_count += unique_edges
    metrics.non_manifold_edge_count += non_manifold_edges
    metrics.shell_face_counts.extend(shell_counts)
    return _MeshTopology(
        prim_path=path,
        prim_name=str(mesh_prim.GetName()),
        face_count=face_count,
        all_faces_triangular=triangular_faces == face_count,
        shell_face_counts=shell_counts,
    )


def _inspect_geom_subsets(
    subset_prims: Sequence[Any],
    meshes_by_path: Mapping[str, _MeshTopology],
    metrics: _MetricsBuilder,
) -> list[_IdentityRecord]:
    from pxr import UsdGeom

    records: list[_IdentityRecord] = []
    for prim in subset_prims:
        subset = UsdGeom.Subset(prim)
        element_type = subset.GetElementTypeAttr().Get()
        if str(element_type) != str(UsdGeom.Tokens.face):
            continue
        parent_path = str(prim.GetParent().GetPath())
        parent_mesh = meshes_by_path.get(parent_path)
        if parent_mesh is None:
            raise _InspectionFailure(
                route="rejected",
                code="malformed_geom_subset",
                reason="A face GeomSubset is not a direct child of an inspected mesh.",
            )
        indices_attr = subset.GetIndicesAttr()
        if indices_attr.ValueMightBeTimeVarying():
            raise _InspectionFailure(
                route="rejected",
                code="malformed_geom_subset",
                reason="Time-varying GeomSubset indices are unsupported.",
            )
        indices = indices_attr.Get()
        if indices is None or not indices:
            raise _InspectionFailure(
                route="rejected",
                code="malformed_geom_subset",
                reason="A face GeomSubset must contain at least one face index.",
            )
        if len(indices) > _MAX_FACES_PER_MESH:
            raise _InspectionFailure(
                route="rejected",
                code="inspection_limit_exceeded",
                reason="GeomSubset face count exceeds the inspection limit.",
            )
        if (
            metrics.geom_subset_face_index_count + len(indices)
            > _MAX_TOTAL_GEOM_SUBSET_INDICES
        ):
            raise _InspectionFailure(
                route="rejected",
                code="inspection_limit_exceeded",
                reason=(
                    "Total GeomSubset face-index count exceeds "
                    f"{_MAX_TOTAL_GEOM_SUBSET_INDICES}."
                ),
            )
        normalized: list[int] = []
        for raw_index in indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise _InspectionFailure(
                    route="rejected",
                    code="malformed_geom_subset",
                    reason="GeomSubset indices must be integers.",
                )
            index = int(raw_index)
            if index < 0 or index >= parent_mesh.face_count:
                raise _InspectionFailure(
                    route="rejected",
                    code="malformed_geom_subset",
                    reason="GeomSubset contains an out-of-range face index.",
                )
            normalized.append(index)
        if len(set(normalized)) != len(normalized):
            raise _InspectionFailure(
                route="rejected",
                code="malformed_geom_subset",
                reason="GeomSubset contains duplicate face indices.",
            )
        metrics.geom_subset_face_index_count += len(normalized)
        records.append(
            _IdentityRecord(
                name=_source_semantic_name(
                    str(prim.GetName()),
                    label="GeomSubset prim name",
                ),
                source="geom_subset",
                prim_path=str(prim.GetPath()),
            )
        )
    metrics.geom_subset_count = len(records)
    return records


def _inspect_usd(
    source_path: Path,
    metrics: _MetricsBuilder,
) -> tuple[list[_MeshTopology], list[_IdentityRecord]]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise _InspectionFailure(
            route="undetermined",
            code="openusd_unavailable",
            reason="OpenUSD is unavailable, so the source cannot be inspected.",
        ) from exc

    if source_path.suffix.lower() not in _USD_SUFFIXES:
        raise _InspectionFailure(
            route="rejected",
            code="unsupported_source_format",
            reason="Semantic mesh routing only inspects USD source formats.",
        )
    try:
        file_stat = source_path.stat()
    except OSError as exc:
        raise _InspectionFailure(
            route="undetermined",
            code="source_usd_uninspectable",
            reason="The source USD does not exist or cannot be inspected.",
        ) from exc
    if not source_path.is_file():
        raise _InspectionFailure(
            route="undetermined",
            code="source_usd_uninspectable",
            reason="The source USD is not a regular file.",
        )
    metrics.source_file_bytes = file_stat.st_size
    if file_stat.st_size > _MAX_SOURCE_FILE_BYTES:
        raise _InspectionFailure(
            route="rejected",
            code="inspection_limit_exceeded",
            reason=f"Source USD exceeds {_MAX_SOURCE_FILE_BYTES} bytes.",
        )

    try:
        stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadNone)
    except Exception as exc:
        raise _InspectionFailure(
            route="undetermined",
            code="source_usd_uninspectable",
            reason="OpenUSD could not open the source stage.",
        ) from exc
    if stage is None:
        raise _InspectionFailure(
            route="undetermined",
            code="source_usd_uninspectable",
            reason="OpenUSD could not open the source stage.",
        )

    # The default traversal predicate omits unloaded payload prims entirely.
    # Scan all composed prim specs first so a partial stage can never look
    # complete merely because its deferred geometry was not traversed.
    payload_scan_count = 0
    for prim in stage.TraverseAll():
        payload_scan_count += 1
        if payload_scan_count > _MAX_PRIM_COUNT:
            raise _InspectionFailure(
                route="rejected",
                code="inspection_limit_exceeded",
                reason=f"USD prim count exceeds {_MAX_PRIM_COUNT}.",
            )
        if prim.HasPayload() and not prim.IsLoaded():
            metrics.unloaded_payload_count += 1

    mesh_prims: list[Any] = []
    subset_prims: list[Any] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        metrics.inspected_prim_count += 1
        if metrics.inspected_prim_count > _MAX_PRIM_COUNT:
            raise _InspectionFailure(
                route="rejected",
                code="inspection_limit_exceeded",
                reason=f"USD prim count exceeds {_MAX_PRIM_COUNT}.",
            )
        prim_path = str(prim.GetPath())
        if len(prim_path) > _MAX_PRIM_PATH_LENGTH:
            raise _InspectionFailure(
                route="rejected",
                code="inspection_limit_exceeded",
                reason=f"USD prim path exceeds {_MAX_PRIM_PATH_LENGTH} characters.",
            )
        if prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim)
            if len(mesh_prims) > _MAX_MESH_PRIM_COUNT:
                raise _InspectionFailure(
                    route="rejected",
                    code="inspection_limit_exceeded",
                    reason=f"Mesh prim count exceeds {_MAX_MESH_PRIM_COUNT}.",
                )
        elif prim.IsA(UsdGeom.Subset):
            subset_prims.append(prim)
            if len(subset_prims) > _MAX_GEOM_SUBSET_COUNT:
                raise _InspectionFailure(
                    route="rejected",
                    code="inspection_limit_exceeded",
                    reason=(f"GeomSubset prim count exceeds {_MAX_GEOM_SUBSET_COUNT}."),
                )

    mesh_prims.sort(key=lambda prim: str(prim.GetPath()))
    subset_prims.sort(key=lambda prim: str(prim.GetPath()))
    metrics.mesh_prim_count = len(mesh_prims)
    metrics.named_mesh_prim_count = sum(bool(prim.GetName()) for prim in mesh_prims)
    metrics.mesh_prim_paths = [str(prim.GetPath()) for prim in mesh_prims]

    meshes = [_inspect_mesh(prim, metrics) for prim in mesh_prims]
    metrics.topology_inspected = bool(meshes)
    meshes_by_path = {mesh.prim_path: mesh for mesh in meshes}
    subset_records = _inspect_geom_subsets(
        subset_prims,
        meshes_by_path,
        metrics,
    )
    return meshes, subset_records


def route_segmentation(
    *,
    requested: bool = False,
    source_usd_path: str | Path | None = None,
    completed_run_reference: str | Path | None = None,
    required_semantic_names: Sequence[str] | None = None,
    supplied_semantic_parts: Sequence[str | Mapping[str, Any]] | None = None,
) -> GeometrySegmentationRoutingDecision:
    """Choose the deterministic next step for semantic mesh segmentation.

    Routing is side-effect free. In particular, a ``consume_completed_run``
    result delegates validation and consumption to the existing handoff
    consumer; the function does not inspect or execute that run.
    """

    metrics = _MetricsBuilder()
    safe_source_text = _bounded_path_text(source_usd_path)
    try:
        required_names = _required_names(required_semantic_names)
    except (TypeError, ValueError) as exc:
        return _decision(
            route="rejected",
            requested=requested,
            required_names=[],
            identity_records=(),
            reason_codes=("invalid_required_semantic_names",),
            reasons=(str(exc),),
            metrics=metrics,
            source_usd_path=safe_source_text,
        )

    if completed_run_reference is not None:
        completed_text = _bounded_path_text(completed_run_reference)
        if completed_text is None:
            return _decision(
                route="rejected",
                requested=requested,
                required_names=required_names,
                identity_records=(),
                reason_codes=("invalid_completed_run_reference",),
                reasons=("The completed run reference is empty or too long.",),
                metrics=metrics,
                source_usd_path=safe_source_text,
            )
        return _decision(
            route="consume_completed_run",
            requested=requested,
            required_names=required_names,
            identity_records=(),
            reason_codes=("completed_run_reference_provided",),
            reasons=(
                "An explicit completed run reference takes precedence and must "
                "be validated by the existing segmentation handoff consumer.",
            ),
            metrics=metrics,
            source_usd_path=safe_source_text,
            completed_run_reference=completed_text,
        )

    if not requested:
        metadata_records = _metadata_identity_records(
            supplied_semantic_parts,
            strict=False,
        )
        metrics.supplied_semantic_part_count = len(metadata_records)
        return _decision(
            route="not_requested",
            requested=False,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=("segmentation_not_requested",),
            reasons=("Semantic mesh segmentation was not requested.",),
            metrics=metrics,
            source_usd_path=safe_source_text,
        )

    try:
        metadata_records = _metadata_identity_records(
            supplied_semantic_parts,
            strict=True,
        )
    except ValueError as exc:
        return _decision(
            route="rejected",
            requested=True,
            required_names=required_names,
            identity_records=(),
            reason_codes=("malformed_semantic_part_metadata",),
            reasons=(str(exc),),
            metrics=metrics,
            source_usd_path=safe_source_text,
        )
    metrics.supplied_semantic_part_count = len(metadata_records)

    if source_usd_path is None:
        if metadata_records:
            return _decision(
                route="reuse_source_identity",
                requested=True,
                required_names=required_names,
                identity_records=metadata_records,
                reason_codes=("supplied_semantic_part_metadata",),
                reasons=(
                    "Supplied semantic-part metadata already provides source "
                    "identity; no mesh segmentation is selected.",
                ),
                metrics=metrics,
            )
        return _decision(
            route="undetermined",
            requested=True,
            required_names=required_names,
            identity_records=(),
            reason_codes=("source_usd_not_provided",),
            reasons=("No source USD was provided for deterministic inspection.",),
            metrics=metrics,
        )

    if safe_source_text is None:
        return _decision(
            route="rejected",
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=("source_path_limit_exceeded",),
            reasons=("The source USD path is empty or exceeds the path limit.",),
            metrics=metrics,
        )

    try:
        source_path = Path(source_usd_path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return _decision(
            route="undetermined",
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=("source_usd_uninspectable",),
            reasons=("The source USD path could not be resolved.",),
            metrics=metrics,
            source_usd_path=safe_source_text,
        )
    resolved_source_text = _bounded_path_text(source_path)
    if resolved_source_text is None:
        return _decision(
            route="rejected",
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=("source_path_limit_exceeded",),
            reasons=("The resolved source USD path exceeds the path limit.",),
            metrics=metrics,
        )

    try:
        meshes, subset_records = _inspect_usd(source_path, metrics)
    except _InspectionFailure as exc:
        return _decision(
            route=exc.route,
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=(exc.code,),
            reasons=(exc.reason,),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )

    if metrics.unloaded_payload_count:
        return _decision(
            route="undetermined",
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=("unloaded_payloads",),
            reasons=(
                "The source stage contains unloaded payloads, so its geometry and "
                "semantic inventory is incomplete.",
            ),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )

    try:
        mesh_identity_records = (
            [
                _IdentityRecord(
                    name=_source_semantic_name(
                        mesh.prim_name,
                        label="mesh prim name",
                    ),
                    source="named_mesh_prim",
                    prim_path=mesh.prim_path,
                )
                for mesh in meshes
            ]
            if len(meshes) > 1
            else []
        )
    except _InspectionFailure as exc:
        return _decision(
            route=exc.route,
            requested=True,
            required_names=required_names,
            identity_records=metadata_records,
            reason_codes=(exc.code,),
            reasons=(exc.reason,),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )
    # Source-preparation metadata may contain expected or descriptive part names
    # without any face or prim binding. Keep it as guidance, but never let an
    # unbound name suppress segmentation of a fused mesh.
    identity_records = [*mesh_identity_records, *subset_records]
    identity_codes: list[str] = []
    identity_reasons: list[str] = []
    if metadata_records:
        identity_codes.append("unbound_semantic_part_metadata_is_guidance_only")
        identity_reasons.append(
            "Supplied semantic-part metadata has no verified source prim or face "
            "binding, so it is retained as guidance rather than geometry identity."
        )
    if mesh_identity_records:
        identity_codes.append("multiple_named_mesh_prims")
        identity_reasons.append("Multiple named mesh prims provide source identity.")
    if subset_records:
        identity_codes.append("face_geom_subsets")
        identity_reasons.append("Face GeomSubsets provide source identity.")
    if identity_records:
        identity_reasons.append(
            "Source identity takes precedence over derived shell or agentic routing."
        )
        return _decision(
            route="reuse_source_identity",
            requested=True,
            required_names=required_names,
            identity_records=identity_records,
            reason_codes=identity_codes,
            reasons=identity_reasons,
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )

    if not meshes:
        return _decision(
            route="rejected",
            requested=True,
            required_names=required_names,
            identity_records=(),
            reason_codes=(*identity_codes, "no_mesh_prims"),
            reasons=(*identity_reasons, "The source USD contains no mesh prims."),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )

    # Multiple valid meshes necessarily have source prim names and were handled above.
    mesh = meshes[0]
    shell_count = len(mesh.shell_face_counts)
    if shell_count > 1:
        return _decision(
            route="deterministic_shell_split",
            requested=True,
            required_names=required_names,
            identity_records=(),
            reason_codes=(*identity_codes, "multiple_edge_connected_shells"),
            reasons=(
                *identity_reasons,
                "The single mesh contains multiple edge-connected shells and "
                "can be split deterministically without semantic inference.",
            ),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )
    if shell_count == 1 and mesh.all_faces_triangular:
        return _decision(
            route="agentic_semantic_segmentation",
            requested=True,
            required_names=required_names,
            identity_records=(),
            reason_codes=(
                *identity_codes,
                "single_fused_connected_triangular_mesh",
            ),
            reasons=(
                *identity_reasons,
                "The source is one edge-connected triangular mesh with no "
                "reusable source identity, so semantic segmentation is required.",
            ),
            metrics=metrics,
            source_usd_path=resolved_source_text,
        )
    return _decision(
        route="rejected",
        requested=True,
        required_names=required_names,
        identity_records=(),
        reason_codes=(*identity_codes, "unsupported_mesh_configuration"),
        reasons=(
            *identity_reasons,
            "The source mesh is connected but not fully triangular, which is "
            "unsupported by the semantic segmentation route.",
        ),
        metrics=metrics,
        source_usd_path=resolved_source_text,
    )


# Compatibility names for integration call sites while the workflow owns wiring.
decide_segmentation_route = route_segmentation
route_semantic_mesh_segmentation = route_segmentation
