# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed source correspondence and attribute-transfer evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import atomic_write_json, file_sha256
from .mesh_io import USD_SUFFIXES, load_meshes

CORRESPONDENCE_SCHEMA_VERSION = "geometry-repair.correspondence.v2"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexMapping(_StrictModel):
    """One explicit output-to-source index relation."""

    output_id: int = Field(ge=0)
    source_id: int = Field(ge=0)


class IndexMappingSpan(_StrictModel):
    """Compact contiguous output-to-source identity relation."""

    output_start: int = Field(ge=0)
    source_start: int = Field(ge=0)
    count: int = Field(ge=1)


class CornerMapping(_StrictModel):
    """Map an output face-corner exactly or barycentrically to one source face."""

    output_face: int = Field(ge=0)
    output_corner: int = Field(ge=0)
    source_face: int = Field(ge=0)
    source_corner: int | None = Field(default=None, ge=0)
    barycentric: list[float] | None = None

    @model_validator(mode="after")
    def _validate_source_relation(self) -> CornerMapping:
        if (self.source_corner is None) == (self.barycentric is None):
            raise ValueError("corner mapping requires exactly one of source_corner or barycentric")
        if self.barycentric is not None:
            if len(self.barycentric) != 3:
                raise ValueError("barycentric coordinates must contain three values")
            values = [float(value) for value in self.barycentric]
            if any(not math.isfinite(value) or value < -1e-12 for value in values):
                raise ValueError("barycentric coordinates must be finite and non-negative")
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("barycentric coordinates must sum to one")
        return self


class FaceMapping(_StrictModel):
    """Classify one output face relative to contributing source faces."""

    output_face: int = Field(ge=0)
    source_faces: list[int] = Field(default_factory=list)
    classification: Literal["unchanged", "split", "merged", "generated"]

    @model_validator(mode="after")
    def _validate_classification(self) -> FaceMapping:
        if any(value < 0 for value in self.source_faces):
            raise ValueError("source face IDs must be non-negative")
        if len(set(self.source_faces)) != len(self.source_faces):
            raise ValueError("source face IDs must not contain duplicates")
        if self.classification == "generated" and self.source_faces:
            raise ValueError("generated faces cannot claim source face identity")
        if self.classification in {"unchanged", "split"} and len(self.source_faces) != 1:
            raise ValueError(f"{self.classification} faces require exactly one source face")
        if self.classification == "merged" and len(self.source_faces) < 2:
            raise ValueError("merged faces require at least two source faces")
        return self


class EntityMapping(_StrictModel):
    """Prim or semantic-part identity relation."""

    source_id: str
    output_id: str | None
    status: Literal["unchanged", "renamed", "split", "merged", "removed", "generated"]
    contributing_source_ids: list[str] = Field(default_factory=list)


class CategoricalResolution(_StrictModel):
    """Deterministic agreement or refusal for categorical source values."""

    attribute_name: str
    status: Literal["assigned", "unassigned", "conflict"]
    value: Any = None
    contributing_values: list[Any] = Field(default_factory=list)
    reason: str | None = None


class AttributeTransfer(_StrictModel):
    """Coverage and semantics for one transferred or explicitly reauthored attribute."""

    attribute_name: str
    domain: Literal["vertex", "corner", "face", "part", "prim"]
    value_kind: Literal["continuous", "categorical"]
    source_interpolation: str | None = None
    output_interpolation: str | None = None
    method: Literal[
        "identity",
        "exact_source_index",
        "exact_source_corner",
        "barycentric_source_face",
        "mixed_source_face_corner",
        "agreed_categorical",
        "generated_unassigned",
        "recomputed",
        "reauthored",
        "refused",
    ]
    status: Literal["pass", "conditional", "refused"]
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    generated_value_count: int = Field(default=0, ge=0)
    ambiguous_value_count: int = Field(default=0, ge=0)
    reauthored: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_transfer_semantics(self) -> AttributeTransfer:
        if self.source_interpolation == "faceVarying" and self.method == "exact_source_index":
            raise ValueError("face-varying data must map by source face-corner, not vertex index")
        if self.value_kind == "categorical" and self.method in {
            "barycentric_source_face",
            "mixed_source_face_corner",
        }:
            raise ValueError("categorical values cannot be barycentrically interpolated")
        if self.method in {"recomputed", "reauthored"} and not self.reauthored:
            raise ValueError("recomputed or reauthored attributes must set reauthored=true")
        if self.status == "pass" and self.method == "refused":
            raise ValueError("a refused transfer cannot pass")
        return self


class CorrespondenceEvidence(_StrictModel):
    """Serializable authority for every source-preserving topology mutation."""

    schema_version: Literal["geometry-repair.correspondence.v2"] = CORRESPONDENCE_SCHEMA_VERSION
    status: Literal["pass", "conditional", "refused"]
    operation: str
    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    source_counts: dict[str, int] = Field(default_factory=dict)
    output_counts: dict[str, int] = Field(default_factory=dict)
    vertex_mappings: list[IndexMapping] = Field(default_factory=list)
    vertex_mapping_spans: list[IndexMappingSpan] = Field(default_factory=list)
    corner_mappings: list[CornerMapping] = Field(default_factory=list)
    corner_mapping_spans: list[IndexMappingSpan] = Field(default_factory=list)
    face_mappings: list[FaceMapping] = Field(default_factory=list)
    face_mapping_spans: list[IndexMappingSpan] = Field(default_factory=list)
    part_mappings: list[EntityMapping] = Field(default_factory=list)
    prim_mappings: list[EntityMapping] = Field(default_factory=list)
    changed_source_regions: dict[str, list[int]] = Field(default_factory=dict)
    changed_output_regions: dict[str, list[int]] = Field(default_factory=dict)
    generated_output_regions: dict[str, list[int]] = Field(default_factory=dict)
    categorical_resolutions: list[CategoricalResolution] = Field(default_factory=list)
    attribute_transfers: list[AttributeTransfer] = Field(default_factory=list)
    part_mapping_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    material_mapping_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    source_face_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_face_coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguous_part_mapping_count: int = Field(default=0, ge=0)
    face_varying_uv_semantics: Literal["source_face_corner"] = "source_face_corner"
    operation_version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    deterministic_seed: int | None = Field(default=None, ge=0)
    resource_use: dict[str, Any] = Field(default_factory=dict)
    log_paths: list[str] = Field(default_factory=list)
    refusal_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_sha256: str | None = None

    @model_validator(mode="after")
    def _validate_fail_closed_contract(self) -> CorrespondenceEvidence:
        for digest in (self.source_sha256, self.output_sha256):
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError("source/output SHA-256 values must be lowercase hex digests")
        conflicts = [
            item.attribute_name
            for item in self.categorical_resolutions
            if item.status == "conflict"
        ]
        refused_attributes = [
            item.attribute_name for item in self.attribute_transfers if item.status == "refused"
        ]
        if self.status == "pass" and (self.refusal_reasons or conflicts or refused_attributes):
            raise ValueError(
                "passing correspondence cannot contain refusal or categorical conflict"
            )
        if conflicts and self.status != "refused":
            raise ValueError("categorical source conflicts require deterministic refusal")
        if self.status == "refused" and not self.refusal_reasons:
            raise ValueError("refused correspondence requires at least one deterministic reason")
        generated_faces = set(self.generated_output_regions.get("faces", []))
        generated_faces.update(
            mapping.output_face
            for mapping in self.face_mappings
            if mapping.classification == "generated"
        )
        mapped_faces = {mapping.output_face for mapping in self.face_mappings}
        changed_faces = set(self.changed_output_regions.get("faces", []))
        unmapped_changed_faces = {
            face
            for face in changed_faces - mapped_faces - generated_faces
            if not any(
                span.output_start <= face < span.output_start + span.count
                for span in self.face_mapping_spans
            )
        }
        if unmapped_changed_faces and self.status != "refused":
            raise ValueError(
                "changed output faces must be mapped or explicitly classified as generated"
            )
        if len(set(self.generated_output_regions.get("faces", []))) != len(
            self.generated_output_regions.get("faces", [])
        ):
            raise ValueError("generated face IDs must not contain duplicates")
        return self


class FaceVaryingTransferResult(_StrictModel):
    """Concrete face-corner transfer result with deterministic refusal evidence."""

    status: Literal["pass", "refused"]
    values: list[list[list[float]]] | None = None
    transfer: AttributeTransfer
    refusal_reasons: list[str] = Field(default_factory=list)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_digest(evidence: CorrespondenceEvidence) -> str:
    payload = evidence.model_dump(mode="json", exclude={"evidence_sha256"})
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _mapping_spans(pairs: list[tuple[int, int]]) -> list[IndexMappingSpan]:
    """Encode exact integer pairs as deterministic contiguous spans."""

    if not pairs:
        return []
    ordered = sorted(pairs)
    if len({output_id for output_id, _source_id in ordered}) != len(ordered):
        raise ValueError("one output index cannot have multiple exact source mappings")
    spans: list[IndexMappingSpan] = []
    output_start, source_start = ordered[0]
    previous_output, previous_source = ordered[0]
    count = 1
    for output_id, source_id in ordered[1:]:
        if output_id == previous_output + 1 and source_id == previous_source + 1:
            count += 1
        else:
            spans.append(
                IndexMappingSpan(
                    output_start=output_start,
                    source_start=source_start,
                    count=count,
                )
            )
            output_start, source_start, count = output_id, source_id, 1
        previous_output, previous_source = output_id, source_id
    spans.append(
        IndexMappingSpan(
            output_start=output_start,
            source_start=source_start,
            count=count,
        )
    )
    return spans


def finalize_correspondence(evidence: CorrespondenceEvidence) -> CorrespondenceEvidence:
    """Attach a digest over the complete serializable evidence payload."""

    return evidence.model_copy(update={"evidence_sha256": _payload_digest(evidence)})


def write_correspondence_evidence(
    path: str | Path,
    evidence: CorrespondenceEvidence,
) -> Path:
    """Write stable JSON and verify an existing evidence digest when supplied."""

    expected = _payload_digest(evidence)
    if evidence.evidence_sha256 is not None and evidence.evidence_sha256 != expected:
        raise ValueError("correspondence evidence digest does not match its payload")
    finalized = evidence.model_copy(update={"evidence_sha256": expected})
    return atomic_write_json(path, finalized)


def resolve_categorical_assignment(
    attribute_name: str,
    values: list[Any],
) -> CategoricalResolution:
    """Assign categorical data only when every contributing source value agrees."""

    if not values:
        return CategoricalResolution(
            attribute_name=attribute_name,
            status="unassigned",
            reason="no contributing source values",
        )
    canonical = {_canonical_json(value): value for value in values}
    ordered = [canonical[key] for key in sorted(canonical)]
    if len(ordered) != 1:
        return CategoricalResolution(
            attribute_name=attribute_name,
            status="conflict",
            contributing_values=ordered,
            reason="contributing source faces disagree; split the output region or refuse",
        )
    return CategoricalResolution(
        attribute_name=attribute_name,
        status="assigned",
        value=ordered[0],
        contributing_values=ordered,
    )


def resolve_generated_patch_categorical_assignment(
    attribute_name: str,
    boundary_values: list[Any],
    *,
    boundary_complete: bool,
) -> CategoricalResolution:
    """Inherit generated-patch categories only from one complete, agreeing boundary."""

    if not boundary_complete:
        return CategoricalResolution(
            attribute_name=attribute_name,
            status="unassigned",
            contributing_values=[
                value
                for _key, value in sorted(
                    {_canonical_json(value): value for value in boundary_values}.items()
                )
            ],
            reason="generated patch boundary is incomplete; categorical inheritance is unsafe",
        )
    return resolve_categorical_assignment(attribute_name, boundary_values)


def identity_correspondence(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    vertex_count: int,
    face_count: int,
    corner_count: int | None = None,
    part_ids: list[str] | None = None,
    prim_paths: list[str] | None = None,
    attributes: dict[str, tuple[str, str | None]] | None = None,
) -> CorrespondenceEvidence:
    """Build a compact exact no-op map, or refuse when file identity is false."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve() if output_path else source
    source_digest = file_sha256(source)
    output_digest = file_sha256(output)
    corners = corner_count if corner_count is not None else face_count * 3
    if source_digest != output_digest:
        return refused_correspondence(
            source,
            output,
            operation="identity",
            reasons=["identity/no-op correspondence requires byte-identical source and output"],
        )
    attribute_transfers = []
    for name, (domain, interpolation) in sorted((attributes or {}).items()):
        method = "exact_source_corner" if interpolation == "faceVarying" else "identity"
        attribute_transfers.append(
            AttributeTransfer(
                attribute_name=name,
                domain=domain,  # type: ignore[arg-type]
                value_kind="continuous",
                source_interpolation=interpolation,
                output_interpolation=interpolation,
                method=method,
                status="pass",
                coverage_ratio=1.0,
            )
        )
    part_mappings = [
        EntityMapping(source_id=value, output_id=value, status="unchanged")
        for value in sorted(part_ids or [])
    ]
    prim_mappings = [
        EntityMapping(source_id=value, output_id=value, status="unchanged")
        for value in sorted(prim_paths or [])
    ]
    return finalize_correspondence(
        CorrespondenceEvidence(
            status="pass",
            operation="identity",
            source_path=str(source),
            output_path=str(output),
            source_sha256=source_digest,
            output_sha256=output_digest,
            source_counts={"vertices": vertex_count, "faces": face_count, "corners": corners},
            output_counts={"vertices": vertex_count, "faces": face_count, "corners": corners},
            vertex_mapping_spans=(
                [IndexMappingSpan(output_start=0, source_start=0, count=vertex_count)]
                if vertex_count
                else []
            ),
            corner_mapping_spans=(
                [IndexMappingSpan(output_start=0, source_start=0, count=corners)] if corners else []
            ),
            face_mapping_spans=(
                [IndexMappingSpan(output_start=0, source_start=0, count=face_count)]
                if face_count
                else []
            ),
            part_mappings=part_mappings,
            prim_mappings=prim_mappings,
            attribute_transfers=attribute_transfers,
            part_mapping_coverage_ratio=1.0,
            material_mapping_coverage_ratio=1.0,
            source_face_coverage_ratio=1.0,
            candidate_face_coverage_ratio=1.0,
        )
    )


def refused_correspondence(
    source_path: str | Path,
    output_path: str | Path,
    *,
    operation: str,
    reasons: list[str],
    warnings: list[str] | None = None,
) -> CorrespondenceEvidence:
    """Create stable refusal evidence rather than dropping unsupported attributes."""

    if not reasons:
        raise ValueError("deterministic correspondence refusal requires at least one reason")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    return finalize_correspondence(
        CorrespondenceEvidence(
            status="refused",
            operation=operation,
            source_path=str(source),
            output_path=str(output),
            source_sha256=file_sha256(source),
            output_sha256=file_sha256(output),
            refusal_reasons=sorted(set(reasons)),
            warnings=sorted(set(warnings or [])),
        )
    )


def _generated_patch_maps(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    if path.suffix.lower() not in USD_SUFFIXES:
        return {}
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return {}
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        encoded = prim.GetCustomDataByKey("geometryRepairGeneratedPatchMap")
        if not isinstance(encoded, str):
            continue
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if payload.get("schema_version") != "geometry-repair.generated-patch-map.v1":
            continue
        records: dict[int, dict[str, Any]] = {}
        for raw_face, raw_record in (payload.get("faces") or {}).items():
            try:
                face_id = int(raw_face)
                sources = [int(value) for value in raw_record["source_faces"]]
                corners = [[int(value[0]), int(value[1])] for value in raw_record["source_corners"]]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if len(corners) != 3 or not sources:
                continue
            records[face_id] = {
                "source_faces": sources,
                "source_corners": corners,
                "normals_reauthored": bool(payload.get("normals_reauthored")),
            }
        if records:
            result[str(prim.GetPath())] = records
    return result


def build_mesh_correspondence(
    source_path: str | Path,
    output_path: str | Path,
    *,
    operation: str,
    operation_version: str | None = None,
    parameters: dict[str, Any] | None = None,
    deterministic_seed: int | None = None,
    explicit_mapping_limit: int = 250_000,
) -> CorrespondenceEvidence:
    """Build exact same-part mesh correspondence and refuse unsupported semantics.

    This adapter is intentionally narrower than nearest-surface fidelity. It maps
    only position-coincident source vertices and exact source triangles. Faces
    without an exact source triangle are classified as generated, never guessed.
    """

    if explicit_mapping_limit < 1:
        raise ValueError("explicit_mapping_limit must be positive")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    source_meshes, _ = load_meshes(source, include_guide_purpose=True)
    output_meshes, _ = load_meshes(output, include_guide_purpose=True)
    source_by_path = {mesh.path: mesh for mesh in source_meshes}
    output_by_path = {mesh.path: mesh for mesh in output_meshes}
    patch_maps = _generated_patch_maps(output)
    refusal_reasons: list[str] = []
    warnings: list[str] = []

    def below_root(path: str) -> str:
        components = path.strip("/").split("/")
        return "/" + "/".join(components[1:]) if len(components) > 1 else "/"

    path_pairs: list[tuple[str, str]] = [
        (path, path) for path in sorted(set(source_by_path) & set(output_by_path))
    ]
    unmatched_source = set(source_by_path) - {source_path for source_path, _ in path_pairs}
    unmatched_output = set(output_by_path) - {output_path for _, output_path in path_pairs}
    output_by_relative: dict[str, list[str]] = {}
    for path in sorted(unmatched_output):
        output_by_relative.setdefault(below_root(path), []).append(path)
    for source_path in sorted(unmatched_source):
        candidates = output_by_relative.get(below_root(source_path), [])
        if len(candidates) == 1:
            output_path = candidates[0]
            path_pairs.append((source_path, output_path))
            unmatched_output.remove(output_path)
    paired_sources = {source_path for source_path, _ in path_pairs}
    paired_outputs = {output_path for _, output_path in path_pairs}
    missing_paths = sorted(set(source_by_path) - paired_sources)
    generated_paths = sorted(set(output_by_path) - paired_outputs)
    if missing_paths or generated_paths:
        refusal_reasons.extend(
            [
                *(
                    f"source mesh prim was removed without an explicit part map: {path}"
                    for path in missing_paths
                ),
                *(
                    f"output mesh prim was generated without an explicit source part map: {path}"
                    for path in generated_paths
                ),
            ]
        )

    source_vertex_offsets: dict[str, int] = {}
    source_face_offsets: dict[str, int] = {}
    output_vertex_offsets: dict[str, int] = {}
    output_face_offsets: dict[str, int] = {}
    source_vertex_count = source_face_count = output_vertex_count = output_face_count = 0
    for path, mesh in sorted(source_by_path.items()):
        source_vertex_offsets[path] = source_vertex_count
        source_face_offsets[path] = source_face_count
        source_vertex_count += len(mesh.world_vertices_m)
        source_face_count += len(mesh.triangles)
    for path, mesh in sorted(output_by_path.items()):
        output_vertex_offsets[path] = output_vertex_count
        output_face_offsets[path] = output_face_count
        output_vertex_count += len(mesh.world_vertices_m)
        output_face_count += len(mesh.triangles)
    if max(source_vertex_count, source_face_count, output_vertex_count, output_face_count) > (
        explicit_mapping_limit
    ):
        return refused_correspondence(
            source,
            output,
            operation=operation,
            reasons=[
                "explicit correspondence exceeds the bounded mapping limit "
                f"of {explicit_mapping_limit}; a compact worker-native mapping is required"
            ],
        )

    all_points = [
        mesh.world_vertices_m
        for mesh in (*source_meshes, *output_meshes)
        if len(mesh.world_vertices_m)
    ]
    diagonal = (
        float(np.linalg.norm(np.ptp(np.concatenate(all_points, axis=0), axis=0)))
        if all_points
        else 0.0
    )
    tolerance = max(diagonal * 1e-7, 1e-12)
    vertex_mapping_pairs: list[tuple[int, int]] = []
    corner_mapping_pairs: list[tuple[int, int]] = []
    face_mappings: list[FaceMapping] = []
    face_mapping_pairs: list[tuple[int, int]] = []
    changed_source_faces: list[int] = []
    changed_output_faces: list[int] = []
    generated_output_faces: list[int] = []
    attributed_generated_faces: set[int] = set()
    generated_normals_reauthored = False
    attributed_generated_paths: list[str] = []

    for source_path, output_path in sorted(path_pairs):
        source_mesh = source_by_path[source_path]
        output_mesh = output_by_path[output_path]
        generated_before = len(generated_output_faces)
        source_points = np.asarray(source_mesh.world_vertices_m, dtype=np.float64)
        output_points = np.asarray(output_mesh.world_vertices_m, dtype=np.float64)
        local_vertex_map: dict[int, int] = {}
        if len(source_points) and len(output_points):
            if source_points.shape == output_points.shape and np.array_equal(
                source_points,
                output_points,
            ):
                local_vertex_map = dict(enumerate(range(len(output_points))))
            else:
                from scipy.spatial import cKDTree

                tree = cKDTree(source_points)
                distances, indices = tree.query(output_points, k=1)
                for output_id, (distance, source_id) in enumerate(
                    zip(np.asarray(distances), np.asarray(indices), strict=True)
                ):
                    if float(distance) <= tolerance:
                        local_vertex_map[output_id] = int(source_id)
            vertex_mapping_pairs.extend(
                (
                    output_vertex_offsets[output_path] + output_id,
                    source_vertex_offsets[source_path] + source_id,
                )
                for output_id, source_id in local_vertex_map.items()
            )

        source_face_keys: dict[tuple[int, int, int], list[int]] = {}
        for source_face_id, face in enumerate(np.asarray(source_mesh.triangles, dtype=np.int64)):
            source_face_keys.setdefault(tuple(sorted(int(value) for value in face)), []).append(
                source_face_id
            )
        used_source_faces: set[int] = set()
        for output_face_id, face in enumerate(np.asarray(output_mesh.triangles, dtype=np.int64)):
            global_output_face = output_face_offsets[output_path] + output_face_id
            mapped_vertices = [local_vertex_map.get(int(value)) for value in face]
            candidates = (
                source_face_keys.get(tuple(sorted(int(value) for value in mapped_vertices)), [])
                if all(value is not None for value in mapped_vertices)
                else []
            )
            source_face_id = next(
                (value for value in candidates if value not in used_source_faces),
                candidates[0] if candidates else None,
            )
            if source_face_id is None:
                generated_output_faces.append(global_output_face)
                changed_output_faces.append(global_output_face)
                face_mappings.append(
                    FaceMapping(
                        output_face=global_output_face,
                        classification="generated",
                    )
                )
                patch_record = patch_maps.get(output_path, {}).get(output_face_id)
                if patch_record is not None:
                    persisted_corners: list[CornerMapping] = []
                    declared_source_faces = set(patch_record["source_faces"])
                    valid_record = bool(declared_source_faces)
                    if any(
                        source_face < 0 or source_face >= len(source_mesh.triangles)
                        for source_face in declared_source_faces
                    ):
                        valid_record = False
                    for output_corner, (local_source_face, source_corner) in enumerate(
                        patch_record["source_corners"]
                    ):
                        if (
                            local_source_face < 0
                            or local_source_face >= len(source_mesh.triangles)
                            or source_corner < 0
                            or source_corner > 2
                            or local_source_face not in declared_source_faces
                        ):
                            valid_record = False
                            break
                        source_vertex = int(source_mesh.triangles[local_source_face][source_corner])
                        if local_vertex_map.get(int(face[output_corner])) != source_vertex:
                            valid_record = False
                            break
                        persisted_corners.append(
                            CornerMapping(
                                output_face=global_output_face,
                                output_corner=output_corner,
                                source_face=(source_face_offsets[source_path] + local_source_face),
                                source_corner=source_corner,
                            )
                        )
                    if valid_record:
                        corner_mapping_pairs.extend(
                            (
                                item.output_face * 3 + item.output_corner,
                                item.source_face * 3 + int(item.source_corner),
                            )
                            for item in persisted_corners
                        )
                        attributed_generated_faces.add(global_output_face)
                        generated_normals_reauthored = generated_normals_reauthored or bool(
                            patch_record.get("normals_reauthored")
                        )
                    else:
                        refusal_reasons.append(
                            f"{output_path}: generated face {output_face_id} has invalid "
                            "persisted corner correspondence"
                        )
                continue
            used_source_faces.add(source_face_id)
            global_source_face = source_face_offsets[source_path] + source_face_id
            face_mapping_pairs.append((global_output_face, global_source_face))
            source_face = np.asarray(source_mesh.triangles[source_face_id], dtype=np.int64)
            for output_corner, output_vertex in enumerate(face):
                mapped_vertex = local_vertex_map.get(int(output_vertex))
                matching_corners = np.flatnonzero(source_face == mapped_vertex)
                if len(matching_corners) != 1:
                    refusal_reasons.append(
                        f"{source_path}: output face {output_face_id} corner {output_corner} "
                        "does not have one exact source face-corner"
                    )
                    continue
                corner_mapping_pairs.append(
                    (
                        global_output_face * 3 + output_corner,
                        global_source_face * 3 + int(matching_corners[0]),
                    )
                )
        changed_source_faces.extend(
            source_face_offsets[source_path] + face_id
            for face_id in range(len(source_mesh.triangles))
            if face_id not in used_source_faces
        )
        local_generated = set(generated_output_faces[generated_before:])
        if local_generated - attributed_generated_faces and (
            source_mesh.has_face_varying_data
            or source_mesh.authored_uv_count
            or source_mesh.authored_normal_count
            or source_mesh.material_subset_count
        ):
            attributed_generated_paths.append(source_path)

    if attributed_generated_paths:
        refusal_reasons.append(
            "generated faces lack complete face-corner or categorical transfer evidence on: "
            + ", ".join(sorted(set(attributed_generated_paths)))
        )
    mapped_corner_count = len(corner_mapping_pairs)
    expected_mapped_corner_count = 3 * (
        output_face_count - len(generated_output_faces) + len(attributed_generated_faces)
    )
    if mapped_corner_count != expected_mapped_corner_count:
        refusal_reasons.append(
            "exact source face-corner coverage is incomplete: "
            f"{mapped_corner_count}/{expected_mapped_corner_count}"
        )
    if len(vertex_mapping_pairs) != output_vertex_count and not generated_output_faces:
        refusal_reasons.append(
            "exact output vertex coverage is incomplete: "
            f"{len(vertex_mapping_pairs)}/{output_vertex_count}"
        )

    topology_changed = bool(changed_source_faces or generated_output_faces)
    if topology_changed:
        warnings.append(
            "topology changed only in the explicitly listed removed/generated face regions"
        )
    part_mappings = [
        EntityMapping(
            source_id=source_path,
            output_id=output_path,
            status="unchanged" if source_path == output_path else "renamed",
        )
        for source_path, output_path in sorted(path_pairs)
    ]
    part_mappings.extend(
        EntityMapping(source_id=path, output_id=None, status="removed") for path in missing_paths
    )
    part_mappings.extend(
        EntityMapping(source_id="", output_id=path, status="generated") for path in generated_paths
    )
    attribute_transfers: list[AttributeTransfer] = []
    generated_attribute_coverage_complete = (
        set(generated_output_faces) <= attributed_generated_faces
    )
    if any(mesh.has_face_varying_data or mesh.authored_uv_count for mesh in source_meshes):
        attribute_transfers.append(
            AttributeTransfer(
                attribute_name="source_face_varying_primvars",
                domain="corner",
                value_kind="continuous",
                source_interpolation="faceVarying",
                output_interpolation="faceVarying",
                method=(
                    "exact_source_corner" if generated_attribute_coverage_complete else "refused"
                ),
                status="pass" if generated_attribute_coverage_complete else "refused",
                coverage_ratio=(
                    1.0
                    if generated_attribute_coverage_complete
                    else (
                        (output_face_count - len(generated_output_faces)) / output_face_count
                        if output_face_count
                        else 1.0
                    )
                ),
            )
        )
    if any(mesh.authored_normal_count for mesh in source_meshes):
        normals_status: Literal["pass", "conditional", "refused"]
        normals_method: Literal["exact_source_corner", "reauthored", "refused"]
        normals_notes = (
            ["generated patch normals were recomputed from generated face geometry"]
            if generated_normals_reauthored
            else []
        )
        if not generated_attribute_coverage_complete:
            normals_status = "refused"
            normals_method = "refused"
            normals_notes.append("generated faces lack complete normal-transfer evidence")
        elif generated_normals_reauthored:
            normals_status = "conditional"
            normals_method = "reauthored"
        else:
            normals_status = "pass"
            normals_method = "exact_source_corner"
        attribute_transfers.append(
            AttributeTransfer(
                attribute_name="source_normals",
                domain="corner",
                value_kind="continuous",
                source_interpolation="faceVarying",
                output_interpolation="faceVarying",
                method=normals_method,
                status=normals_status,
                coverage_ratio=(
                    1.0
                    if generated_attribute_coverage_complete
                    else (
                        (
                            output_face_count
                            - len(generated_output_faces)
                            + len(attributed_generated_faces)
                        )
                        / output_face_count
                        if output_face_count
                        else 1.0
                    )
                ),
                generated_value_count=(
                    len(attributed_generated_faces) * 3 if generated_normals_reauthored else 0
                ),
                reauthored=generated_normals_reauthored,
                notes=normals_notes,
            )
        )
    if any(mesh.material_subset_count for mesh in source_meshes):
        attribute_transfers.append(
            AttributeTransfer(
                attribute_name="material_subset_assignment",
                domain="face",
                value_kind="categorical",
                method=("agreed_categorical" if generated_output_faces else "identity")
                if generated_attribute_coverage_complete
                else "refused",
                status="pass" if generated_attribute_coverage_complete else "refused",
                coverage_ratio=(
                    1.0
                    if generated_attribute_coverage_complete
                    else (
                        (output_face_count - len(generated_output_faces)) / output_face_count
                        if output_face_count
                        else 1.0
                    )
                ),
            )
        )
    if any(mesh.material_binding_count for mesh in source_meshes):
        attribute_transfers.append(
            AttributeTransfer(
                attribute_name="prim_material_binding",
                domain="prim",
                value_kind="categorical",
                method="identity",
                status="pass",
                coverage_ratio=1.0,
            )
        )
    status: Literal["pass", "conditional", "refused"] = (
        "refused" if refusal_reasons else "conditional" if warnings else "pass"
    )
    evidence = CorrespondenceEvidence(
        status=status,
        operation=operation,
        source_path=str(source),
        output_path=str(output),
        source_sha256=file_sha256(source),
        output_sha256=file_sha256(output),
        source_counts={
            "vertices": source_vertex_count,
            "faces": source_face_count,
            "corners": source_face_count * 3,
            "parts": len(source_meshes),
        },
        output_counts={
            "vertices": output_vertex_count,
            "faces": output_face_count,
            "corners": output_face_count * 3,
            "parts": len(output_meshes),
        },
        vertex_mapping_spans=_mapping_spans(vertex_mapping_pairs),
        corner_mapping_spans=_mapping_spans(corner_mapping_pairs),
        face_mappings=face_mappings,
        face_mapping_spans=_mapping_spans(face_mapping_pairs),
        part_mappings=part_mappings,
        prim_mappings=list(part_mappings),
        changed_source_regions={"faces": sorted(changed_source_faces)}
        if changed_source_faces
        else {},
        changed_output_regions={"faces": sorted(changed_output_faces)}
        if changed_output_faces
        else {},
        generated_output_regions={"faces": sorted(generated_output_faces)}
        if generated_output_faces
        else {},
        attribute_transfers=attribute_transfers,
        part_mapping_coverage_ratio=(
            len(path_pairs) / len(source_meshes) if source_meshes else 1.0
        ),
        material_mapping_coverage_ratio=(1.0 if generated_attribute_coverage_complete else 0.0),
        source_face_coverage_ratio=(
            (source_face_count - len(changed_source_faces)) / source_face_count
            if source_face_count
            else 1.0
        ),
        candidate_face_coverage_ratio=(
            1.0
            if generated_attribute_coverage_complete
            else (
                (output_face_count - len(generated_output_faces)) / output_face_count
                if output_face_count
                else 1.0
            )
        ),
        ambiguous_part_mapping_count=len(missing_paths) + len(generated_paths),
        operation_version=operation_version,
        parameters=parameters or {},
        deterministic_seed=deterministic_seed,
        refusal_reasons=sorted(set(refusal_reasons)),
        warnings=sorted(set(warnings)),
    )
    return finalize_correspondence(evidence)


def transfer_face_varying_values(
    attribute_name: str,
    source_values: np.ndarray,
    mappings: list[CornerMapping],
    *,
    output_face_count: int,
    output_corner_count: int = 3,
) -> FaceVaryingTransferResult:
    """Transfer continuous face-varying values by source face-corner semantics."""

    values = np.asarray(source_values, dtype=np.float64)
    reasons: list[str] = []
    if values.ndim != 3 or values.shape[1] != 3:
        reasons.append("source face-varying values must have shape (faces, 3, channels)")
    if output_corner_count != 3:
        reasons.append("only triangular output face-corners are currently supported")
    expected_count = output_face_count * output_corner_count
    key_counts: dict[tuple[int, int], int] = {}
    for mapping in mappings:
        key = (mapping.output_face, mapping.output_corner)
        key_counts[key] = key_counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in key_counts.items() if count != 1)
    if duplicates:
        reasons.append(f"duplicate output face-corner mappings: {duplicates}")
    expected_keys = {
        (face, corner) for face in range(output_face_count) for corner in range(output_corner_count)
    }
    missing = sorted(expected_keys - set(key_counts))
    extras = sorted(set(key_counts) - expected_keys)
    if missing:
        reasons.append(f"missing output face-corner mappings: {missing}")
    if extras:
        reasons.append(f"out-of-range output face-corner mappings: {extras}")
    if reasons:
        return FaceVaryingTransferResult(
            status="refused",
            transfer=AttributeTransfer(
                attribute_name=attribute_name,
                domain="corner",
                value_kind="continuous",
                source_interpolation="faceVarying",
                output_interpolation="faceVarying",
                method="refused",
                status="refused",
                coverage_ratio=0.0,
                notes=reasons,
            ),
            refusal_reasons=reasons,
        )
    output = np.empty(
        (output_face_count, output_corner_count, values.shape[2]),
        dtype=np.float64,
    )
    used_exact = False
    used_barycentric = False
    for mapping in sorted(mappings, key=lambda item: (item.output_face, item.output_corner)):
        if mapping.source_face >= len(values):
            reasons.append(f"source face {mapping.source_face} is out of range")
            continue
        if mapping.source_corner is not None:
            if mapping.source_corner >= 3:
                reasons.append(
                    f"source corner {mapping.source_corner} is out of range for face "
                    f"{mapping.source_face}"
                )
                continue
            output[mapping.output_face, mapping.output_corner] = values[
                mapping.source_face,
                mapping.source_corner,
            ]
            used_exact = True
        else:
            barycentric = np.asarray(mapping.barycentric, dtype=np.float64)
            output[mapping.output_face, mapping.output_corner] = (
                barycentric @ values[mapping.source_face]
            )
            used_barycentric = True
    if reasons:
        return FaceVaryingTransferResult(
            status="refused",
            transfer=AttributeTransfer(
                attribute_name=attribute_name,
                domain="corner",
                value_kind="continuous",
                source_interpolation="faceVarying",
                output_interpolation="faceVarying",
                method="refused",
                status="refused",
                coverage_ratio=max(0.0, 1.0 - len(reasons) / max(expected_count, 1)),
                notes=reasons,
            ),
            refusal_reasons=reasons,
        )
    method = (
        "mixed_source_face_corner"
        if used_exact and used_barycentric
        else "barycentric_source_face"
        if used_barycentric
        else "exact_source_corner"
    )
    return FaceVaryingTransferResult(
        status="pass",
        values=output.tolist(),
        transfer=AttributeTransfer(
            attribute_name=attribute_name,
            domain="corner",
            value_kind="continuous",
            source_interpolation="faceVarying",
            output_interpolation="faceVarying",
            method=method,
            status="pass",
            coverage_ratio=1.0,
        ),
    )
