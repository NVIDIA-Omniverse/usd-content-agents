# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow Manifold merge-vector analysis for lost property-vertex seams."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..artifacts import atomic_write_json, file_sha256
from ..mesh_io import load_meshes
from ..models import RepairOperation
from .base import WorkerResult

MANIFOLD_SEAM_ANALYSIS_SCHEMA_VERSION = "geometry-repair.manifold-seam-analysis.v1"
SUPPORTED_MANIFOLD3D_VERSIONS = frozenset({"3.5.2"})


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifoldSeamPartAnalysis(_StrictModel):
    """Exact merge-vector evidence for one source mesh."""

    path: str
    status: Literal["identity", "merge_vectors", "refused", "unavailable"]
    backend_version: str | None = None
    vertex_count: int = Field(ge=0)
    face_count: int = Field(ge=0)
    property_channel_count: int = Field(ge=0)
    indexed_boundary_edge_count: int = Field(ge=0)
    positional_boundary_edge_count: int = Field(ge=0)
    exact_duplicate_position_count: int = Field(ge=0)
    merge_from_vert: list[int] = Field(default_factory=list)
    merge_to_vert: list[int] = Field(default_factory=list)
    positions_preserved: bool = False
    properties_preserved: bool = False
    surface_preserved: bool = False
    face_ids_preserved: bool = False
    run_original_ids_preserved: bool = False
    source_surface_sha256: str | None = None
    roundtrip_surface_sha256: str | None = None
    property_rows_sha256: str | None = None
    roundtrip_property_rows_sha256: str | None = None
    refusal_reasons: list[str] = Field(default_factory=list)


class ManifoldSeamAnalysis(_StrictModel):
    """Aggregate source artifact; this worker never rewrites render geometry."""

    schema_version: Literal["geometry-repair.manifold-seam-analysis.v1"] = (
        MANIFOLD_SEAM_ANALYSIS_SCHEMA_VERSION
    )
    source_path: str
    source_sha256: str
    status: Literal["pass", "refused", "unavailable"]
    operation: Literal["manifold_restore_merge_vectors"] = "manifold_restore_merge_vectors"
    backend_version: str | None = None
    changed_geometry: Literal[False] = False
    parts: list[ManifoldSeamPartAnalysis] = Field(default_factory=list)
    refusal_reasons: list[str] = Field(default_factory=list)


def _load_manifold_backend() -> tuple[Any | None, str | None, str | None]:
    try:
        import manifold3d

        version = metadata.version("manifold3d")
    except Exception as exc:
        return None, None, f"manifold3d unavailable: {type(exc).__name__}: {exc}"
    if version not in SUPPORTED_MANIFOLD3D_VERSIONS:
        return (
            None,
            version,
            f"manifold3d {version} is not the audited version; expected one of "
            f"{sorted(SUPPORTED_MANIFOLD3D_VERSIONS)}",
        )
    required = ("Mesh64", "Manifold", "Error")
    missing = [name for name in required if not hasattr(manifold3d, name)]
    if missing:
        return None, version, "manifold3d is missing required capabilities: " + ", ".join(missing)
    return manifold3d, version, None


def _boundary_edge_count(triangles: np.ndarray) -> int:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(triangles, dtype=np.int64):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge = tuple(sorted((int(start), int(end))))
            counts[edge] = counts.get(edge, 0) + 1
    return sum(count == 1 for count in counts.values())


def _canonical_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values)
    if not len(rows):
        return rows.reshape((0, rows.shape[-1] if rows.ndim else 0))
    keys = tuple(rows[:, index] for index in reversed(range(rows.shape[1])))
    return rows[np.lexsort(keys)]


def _canonical_surface_rows(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    canonical_triangles = []
    for triangle in coordinates:
        canonical_triangles.append(_canonical_rows(triangle).reshape(-1))
    return _canonical_rows(np.asarray(canonical_triangles, dtype=np.float64))


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def analyze_manifold_seams(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    properties: np.ndarray | None = None,
    part_ids: list[str] | None = None,
    path: str = "/Mesh",
    _backend: Any | None = None,
    _backend_version: str | None = None,
) -> ManifoldSeamPartAnalysis:
    """Return exact merge vectors or refuse; never close a positional boundary."""

    points = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    faces = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    channels = (
        np.asarray(properties, dtype=np.float64).reshape((len(points), -1))
        if properties is not None
        else np.empty((len(points), 0), dtype=np.float64)
    )
    reasons: list[str] = []
    if not len(points) or not len(faces):
        reasons.append("seam analysis requires non-empty vertices and triangles")
    if not np.isfinite(points).all() or not np.isfinite(channels).all():
        reasons.append("positions and properties must be finite")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        reasons.append("triangle indices are out of range")
    if part_ids is not None and len(part_ids) != len(points):
        reasons.append("part_ids must contain one value per property vertex")
    if reasons:
        return ManifoldSeamPartAnalysis(
            path=path,
            status="refused",
            backend_version=_backend_version,
            vertex_count=len(points),
            face_count=len(faces),
            property_channel_count=channels.shape[1],
            indexed_boundary_edge_count=0,
            positional_boundary_edge_count=0,
            exact_duplicate_position_count=0,
            refusal_reasons=reasons,
        )

    unique_positions, positional_ids = np.unique(points, axis=0, return_inverse=True)
    positional_faces = positional_ids[faces]
    indexed_boundary = _boundary_edge_count(faces)
    positional_boundary = _boundary_edge_count(positional_faces)
    duplicate_count = len(points) - len(unique_positions)
    if np.any(
        (positional_faces[:, 0] == positional_faces[:, 1])
        | (positional_faces[:, 1] == positional_faces[:, 2])
        | (positional_faces[:, 2] == positional_faces[:, 0])
    ):
        reasons.append("exact positional merging would collapse one or more triangles")
    if positional_boundary:
        reasons.append(
            f"{positional_boundary} positional boundary edges remain; they may be real openings"
        )
    if reasons:
        return ManifoldSeamPartAnalysis(
            path=path,
            status="refused",
            backend_version=_backend_version,
            vertex_count=len(points),
            face_count=len(faces),
            property_channel_count=channels.shape[1],
            indexed_boundary_edge_count=indexed_boundary,
            positional_boundary_edge_count=positional_boundary,
            exact_duplicate_position_count=duplicate_count,
            refusal_reasons=reasons,
        )
    backend = _backend
    backend_version = _backend_version
    if backend is None:
        backend, backend_version, capability_error = _load_manifold_backend()
        if capability_error:
            return ManifoldSeamPartAnalysis(
                path=path,
                status="unavailable",
                backend_version=backend_version,
                vertex_count=len(points),
                face_count=len(faces),
                property_channel_count=channels.shape[1],
                indexed_boundary_edge_count=indexed_boundary,
                positional_boundary_edge_count=positional_boundary,
                exact_duplicate_position_count=duplicate_count,
                refusal_reasons=[capability_error],
            )
    # The source-index sentinel proves that every arbitrary property row survives
    # independently even when several property vertices share one position.
    source_ids = np.arange(len(points), dtype=np.float64)[:, None]
    property_rows = np.column_stack((points, channels, source_ids))
    manifold_mesh = backend.Mesh64(
        np.array(property_rows, dtype=np.float64, copy=True, order="C"),
        np.array(faces, dtype=np.uint64, copy=True, order="C"),
        run_index=np.array([0], dtype=np.uint64),
        run_original_id=np.array([1], dtype=np.uint32),
        face_id=np.arange(len(faces), dtype=np.uint64),
        tolerance=0.0,
    )
    merged = bool(manifold_mesh.merge())
    merge_from = [int(value) for value in manifold_mesh.merge_from_vert]
    merge_to = [int(value) for value in manifold_mesh.merge_to_vert]
    if len(merge_from) != len(merge_to):
        reasons.append("Manifold returned mismatched merge-vector lengths")
    for source_id, target_id in zip(merge_from, merge_to, strict=False):
        if source_id >= len(points) or target_id >= len(points):
            reasons.append("Manifold returned an out-of-range merge-vector index")
            continue
        if not np.array_equal(points[source_id], points[target_id]):
            reasons.append(
                f"merge vector {source_id}->{target_id} is not exactly position-coincident"
            )
        if part_ids is not None and part_ids[source_id] != part_ids[target_id]:
            reasons.append(f"merge vector {source_id}->{target_id} crosses source part identity")
    if indexed_boundary and not merged:
        reasons.append("indexed boundaries exist but Manifold produced no exact merge vectors")
    if not indexed_boundary and merged:
        reasons.append("Manifold proposed merges for an input without indexed boundary seams")

    manifold = backend.Manifold(manifold_mesh)
    if str(manifold.status()) != "Error.NoError":
        reasons.append(
            f"Manifold construction failed after merge-vector analysis: {manifold.status()}"
        )
    if reasons:
        return ManifoldSeamPartAnalysis(
            path=path,
            status="refused",
            backend_version=backend_version,
            vertex_count=len(points),
            face_count=len(faces),
            property_channel_count=channels.shape[1],
            indexed_boundary_edge_count=indexed_boundary,
            positional_boundary_edge_count=positional_boundary,
            exact_duplicate_position_count=duplicate_count,
            merge_from_vert=merge_from,
            merge_to_vert=merge_to,
            refusal_reasons=sorted(set(reasons)),
        )

    roundtrip = manifold.to_mesh64()
    output_rows = np.asarray(roundtrip.vert_properties, dtype=np.float64)
    output_faces = np.asarray(roundtrip.tri_verts, dtype=np.int64)
    source_surface = _canonical_surface_rows(points, faces)
    roundtrip_surface = _canonical_surface_rows(output_rows[:, :3], output_faces)
    canonical_properties = _canonical_rows(property_rows)
    canonical_roundtrip_properties = _canonical_rows(output_rows)
    positions_preserved = np.array_equal(
        _canonical_rows(points),
        _canonical_rows(output_rows[:, :3]),
    )
    properties_preserved = np.array_equal(
        canonical_properties,
        canonical_roundtrip_properties,
    )
    surface_preserved = np.array_equal(source_surface, roundtrip_surface)
    face_ids_preserved = sorted(int(value) for value in roundtrip.face_id) == list(
        range(len(faces))
    )
    run_original_ids_preserved = {int(value) for value in roundtrip.run_original_id} == {1}
    if not positions_preserved:
        reasons.append("round-trip changed the exact property-vertex position set")
    if not properties_preserved:
        reasons.append("round-trip changed or dropped arbitrary property rows")
    if not surface_preserved:
        reasons.append("round-trip changed the exact world-space triangle surface")
    if not face_ids_preserved:
        reasons.append("round-trip did not preserve every source faceID")
    if not run_original_ids_preserved:
        reasons.append("round-trip did not preserve runOriginalID")
    status: Literal["identity", "merge_vectors", "refused", "unavailable"] = (
        "refused" if reasons else "merge_vectors" if merged else "identity"
    )
    return ManifoldSeamPartAnalysis(
        path=path,
        status=status,
        backend_version=backend_version,
        vertex_count=len(points),
        face_count=len(faces),
        property_channel_count=channels.shape[1],
        indexed_boundary_edge_count=indexed_boundary,
        positional_boundary_edge_count=positional_boundary,
        exact_duplicate_position_count=duplicate_count,
        merge_from_vert=merge_from,
        merge_to_vert=merge_to,
        positions_preserved=positions_preserved,
        properties_preserved=properties_preserved,
        surface_preserved=surface_preserved,
        face_ids_preserved=face_ids_preserved,
        run_original_ids_preserved=run_original_ids_preserved,
        source_surface_sha256=_array_sha256(source_surface),
        roundtrip_surface_sha256=_array_sha256(roundtrip_surface),
        property_rows_sha256=_array_sha256(canonical_properties),
        roundtrip_property_rows_sha256=_array_sha256(canonical_roundtrip_properties),
        refusal_reasons=sorted(set(reasons)),
    )


class ManifoldSeamWorker:
    """Emit merge vectors for exact lost seams; never perform generic repair."""

    name = "manifold_restore_merge_vectors"
    operations = frozenset({"manifold_restore_merge_vectors"})

    def available(self) -> tuple[bool, str | None]:
        _backend, _version, error = _load_manifold_backend()
        return error is None, error

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        requested_operation = operation.parameters.get("operation")
        if requested_operation != self.name:
            return WorkerResult(
                status="unavailable",
                failures=[
                    "ManifoldSeamWorker accepts only the typed "
                    "manifold_restore_merge_vectors operation"
                ],
            )
        backend, version, capability_error = _load_manifold_backend()
        if capability_error:
            return WorkerResult(status="unavailable", failures=[capability_error])
        try:
            meshes, _metadata = load_meshes(source)
        except Exception as exc:
            return WorkerResult(
                status="unavailable",
                failures=[f"source mesh intake failed: {type(exc).__name__}: {exc}"],
            )
        parts = []
        for mesh in sorted(meshes, key=lambda item: item.path):
            if len(mesh.source_face_counts) != len(mesh.triangles) or np.any(
                mesh.source_face_counts != 3
            ):
                parts.append(
                    ManifoldSeamPartAnalysis(
                        path=mesh.path,
                        status="refused",
                        backend_version=version,
                        vertex_count=len(mesh.local_vertices),
                        face_count=len(mesh.triangles),
                        property_channel_count=0,
                        indexed_boundary_edge_count=0,
                        positional_boundary_edge_count=0,
                        exact_duplicate_position_count=0,
                        refusal_reasons=[
                            "source n-gons require an explicit corner correspondence adapter"
                        ],
                    )
                )
                continue
            parts.append(
                analyze_manifold_seams(
                    mesh.local_vertices,
                    mesh.triangles,
                    path=mesh.path,
                    _backend=backend,
                    _backend_version=version,
                )
            )
        refusal_reasons = sorted(
            {
                f"{part.path}: {reason}"
                for part in parts
                if part.status in {"refused", "unavailable"}
                for reason in part.refusal_reasons
            }
        )
        status: Literal["pass", "refused", "unavailable"]
        if not parts:
            status = "refused"
            refusal_reasons = ["source contains no render mesh eligible for seam analysis"]
        elif any(part.status == "unavailable" for part in parts):
            status = "unavailable"
        elif refusal_reasons:
            status = "refused"
        else:
            status = "pass"
        report = ManifoldSeamAnalysis(
            source_path=str(source.resolve()),
            source_sha256=file_sha256(source),
            status=status,
            backend_version=version,
            parts=parts,
            refusal_reasons=refusal_reasons,
        )
        report_path = atomic_write_json(output, report)
        if status != "pass":
            return WorkerResult(
                status="unavailable",
                output_path=str(source.resolve()),
                changed=False,
                failures=refusal_reasons,
                metadata={"merge_vector_report_path": str(report_path)},
            )
        merge_count = sum(len(part.merge_from_vert) for part in parts)
        return WorkerResult(
            status="completed",
            output_path=str(source.resolve()),
            output_sha256=file_sha256(source),
            changed=False,
            operations=["analyze_exact_property_vertex_seams"],
            warnings=(
                ["source was already manifold; no merge vectors were required"]
                if merge_count == 0
                else []
            ),
            metadata={
                "merge_vector_report_path": str(report_path),
                "merge_vector_count": merge_count,
                "source_geometry_mutated": False,
            },
        )
