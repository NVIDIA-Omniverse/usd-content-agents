# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded PMP adapter for one classified hole or generated patch."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifacts import file_sha256
from ..attributed_rewrite import (
    GeneratedPatchTopologyRewrite,
    rewrite_usd_triangle_meshes_with_generated_patches,
)
from ..hard_mesh_policy import (
    PMP_BUILD_ID,
    PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
    PMP_SOURCE_COMMIT,
    PMP_SOURCE_TREE_SHA256,
    HardMeshCapabilities,
    HardMeshExecutableSpec,
    HardMeshResult,
    discover_hard_mesh_executable,
    run_hard_mesh_executable,
    verify_approved_executable_digest,
)
from ..mesh_io import USD_SUFFIXES, MeshData, load_meshes
from ..models import RepairOperation
from .base import WorkerResult

_PMP_SPEC = HardMeshExecutableSpec(
    worker="pmp_patch",
    environment_variable="GEOMETRY_REPAIR_PMP_EXECUTABLE",
    executable_names=("geometry_repair_pmp_patch",),
    build_id=PMP_BUILD_ID,
    operations=frozenset({"pmp_fill_classified_hole"}),
    required_capabilities=frozenset(
        {
            "changed_region_report",
            "deterministic_single_thread",
            "freeze_vertices",
            "generated_face_classification",
            "independently_approved_executable_digest",
            "pinned_backend_source",
            "protect_edges",
            "source_face_correspondence",
        }
    ),
)


class _StrictParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PmpFillParameters(_StrictParameters):
    operation: Literal["pmp_fill_classified_hole"]
    target_mesh_path: str = Field(min_length=1)
    region_intent: Literal["classified_accidental_hole"]
    intent_evidence_id: str = Field(min_length=1)
    boundary_loop_vertex_ids: list[int] = Field(min_length=3, max_length=4096)
    frozen_boundary_vertex_ids: list[int] = Field(min_length=3, max_length=4096)
    protected_edge_vertex_pairs: list[tuple[int, int]] = Field(min_length=3, max_length=4096)
    max_loop_perimeter_ratio: float = Field(default=0.25, gt=0.0, le=0.5)
    max_patch_area_ratio: float = Field(default=0.05, gt=0.0, le=0.1)
    max_nonplanarity_ratio: float = Field(default=0.005, ge=0.0, le=0.02)
    max_boundary_turn_radians: float = Field(default=math.pi, gt=0.0, le=math.pi)
    max_envelope_ratio: float = Field(default=0.001, gt=0.0, le=0.005)
    max_new_vertices: int = Field(default=8192, ge=1, le=100_000)
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    timeout_s: float = Field(default=120.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _validate_boundary_protection(self) -> _PmpFillParameters:
        loop = self.boundary_loop_vertex_ids
        if len(set(loop)) != len(loop):
            raise ValueError("boundary_loop_vertex_ids must be unique")
        frozen = self.frozen_boundary_vertex_ids
        if len(set(frozen)) != len(frozen) or not set(loop) <= set(frozen):
            raise ValueError("every boundary-loop vertex must be uniquely frozen")
        expected_edges = {
            tuple(sorted((start, end)))
            for start, end in zip(loop, (*loop[1:], loop[0]), strict=True)
        }
        protected_edges = {tuple(sorted(edge)) for edge in self.protected_edge_vertex_pairs}
        if any(start == end for start, end in self.protected_edge_vertex_pairs):
            raise ValueError("protected edges must have two distinct endpoints")
        if not expected_edges <= protected_edges:
            raise ValueError("every boundary-loop edge must be explicitly protected")
        return self


def _refused(operation: str | None, message: str) -> HardMeshResult:
    return HardMeshResult(status="refused", operation=operation, failures=[message])


def _target_mesh(source: Path, target_path: str) -> tuple[MeshData | None, str | None]:
    try:
        meshes, _ = load_meshes(source)
    except Exception as exc:
        return None, f"PMP could not inspect the canonical source: {type(exc).__name__}: {exc}"
    matches = [mesh for mesh in meshes if mesh.path == target_path and mesh.role == "render"]
    if len(matches) != 1:
        return None, f"PMP target {target_path!r} must identify exactly one render mesh"
    mesh = matches[0]
    unsupported = []
    if mesh.is_instance_proxy:
        unsupported.append("instance proxy")
    if len(mesh.source_face_counts) != len(mesh.triangles) or any(
        int(count) != 3 for count in mesh.source_face_counts
    ):
        unsupported.append("non-triangular source faces")
    if mesh.transform_non_finite or mesh.transform_singular:
        unsupported.append("invalid source transform")
    if mesh.transform_non_uniform or mesh.transform_sheared or mesh.transform_reflected:
        unsupported.append("non-rigid or reflected source transform")
    if unsupported:
        return None, (
            f"PMP refused {target_path!r} because exact transfer is unavailable for: "
            + ", ".join(unsupported)
        )
    return mesh, None


def _boundary_source_faces(
    mesh: MeshData,
    loop: list[int],
) -> tuple[int, ...]:
    edge_owners: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(np.asarray(mesh.triangles, dtype=np.int64)):
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge_owners.setdefault(tuple(sorted((int(start), int(end)))), []).append(face_id)
    owners: set[int] = set()
    for start, end in zip(loop, (*loop[1:], loop[0]), strict=True):
        edge = tuple(sorted((start, end)))
        if len(edge_owners.get(edge, [])) != 1:
            raise ValueError("classified patch boundary does not have one source-face owner")
        owners.add(edge_owners[edge][0])
    if not owners:
        raise ValueError("classified patch boundary has no source-face attribution evidence")
    return tuple(sorted(owners))


def _validate_vertex_ids(mesh: MeshData, vertex_ids: list[int], label: str) -> str | None:
    if any(index < 0 or index >= len(mesh.local_vertices) for index in vertex_ids):
        return f"{label} contains an out-of-range source vertex identifier"
    return None


def _validate_fill_geometry(mesh: MeshData, parameters: _PmpFillParameters) -> str | None:
    for values, label in (
        (parameters.boundary_loop_vertex_ids, "boundary loop"),
        (parameters.frozen_boundary_vertex_ids, "frozen boundary"),
        (
            [value for edge in parameters.protected_edge_vertex_pairs for value in edge],
            "protected edges",
        ),
    ):
        failure = _validate_vertex_ids(mesh, values, label)
        if failure:
            return failure
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in mesh.triangles:
        for start, end in zip(face, np.roll(face, -1), strict=True):
            edge_counts[tuple(sorted((int(start), int(end))))] += 1
    loop = parameters.boundary_loop_vertex_ids
    loop_edges = [
        tuple(sorted((start, end))) for start, end in zip(loop, (*loop[1:], loop[0]), strict=True)
    ]
    if any(edge_counts[edge] != 1 for edge in loop_edges):
        return "classified hole loop does not match one indexed source boundary"
    points = np.asarray(mesh.local_vertices[loop], dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(mesh.local_vertices, axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 1e-12:
        return "PMP target mesh has no finite repair scale"
    perimeter = float(
        sum(
            np.linalg.norm(points[(index + 1) % len(points)] - points[index])
            for index in range(len(points))
        )
    )
    if perimeter / diagonal > parameters.max_loop_perimeter_ratio:
        return "classified hole perimeter exceeds its bbox-relative policy bound"
    centered = points - points.mean(axis=0)
    _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    nonplanarity = float(np.max(np.abs(centered @ normal), initial=0.0)) / diagonal
    if nonplanarity > parameters.max_nonplanarity_ratio:
        return "classified hole nonplanarity exceeds its policy bound"
    area_vector = np.zeros(3, dtype=np.float64)
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
        area_vector += np.cross(start, end)
    area_ratio = 0.5 * float(np.linalg.norm(area_vector)) / (diagonal * diagonal)
    if area_ratio <= 1e-12:
        return "classified hole boundary has no stable projected area"
    if area_ratio > parameters.max_patch_area_ratio:
        return "classified hole projected area exceeds its policy bound"
    edge_vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edge_vectors, axis=1)
    if np.any(lengths <= diagonal * 1e-12):
        return "classified hole contains a zero-length boundary edge"
    unit_edges = edge_vectors / lengths[:, None]
    turns = np.arccos(
        np.clip(np.sum(-np.roll(unit_edges, 1, axis=0) * unit_edges, axis=1), -1.0, 1.0)
    )
    if float(np.max(turns, initial=0.0)) > parameters.max_boundary_turn_radians:
        return "classified hole boundary curvature exceeds its policy bound"
    return None


def _verified_pmp_identity(
    executable: Path,
    capabilities: HardMeshCapabilities,
) -> tuple[str | None, str | None]:
    digest, reason = verify_approved_executable_digest(
        executable,
        digest_environment_variable=PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
    )
    if reason or digest is None:
        return None, reason or "PMP executable digest could not be verified"
    if capabilities.executable_sha256 != digest:
        return (
            None,
            "PMP capability executable digest does not match the independently hashed binary",
        )
    if capabilities.backend_source_commit != PMP_SOURCE_COMMIT:
        return None, "PMP capability source commit does not match the pinned backend"
    if capabilities.backend_source_tree_sha256 != PMP_SOURCE_TREE_SHA256:
        return None, "PMP capability source-tree digest does not match the pinned backend"
    return digest, None


def _write_neutral_obj(mesh: MeshData, path: Path) -> Path:
    """Write the exact local-space triangle arrays accepted by the native boundary worker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# geometry-repair PMP neutral bridge v1\n")
        for x, y, z in np.asarray(mesh.local_vertices, dtype=np.float64):
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for first, second, third in np.asarray(mesh.triangles, dtype=np.int64):
            stream.write(f"f {first + 1} {second + 1} {third + 1}\n")
    return path


def _read_neutral_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("PMP neutral output exceeds the 512 MiB adapter limit")
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        if fields[0] == "v" and len(fields) == 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f" and len(fields) == 4:
            indices = tuple(int(value) - 1 for value in fields[1:4])
            triangles.append(indices)
        else:
            raise ValueError(f"PMP neutral output contains unsupported OBJ record {fields[0]!r}")
    vertex_array = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if not len(vertex_array) or not len(triangle_array):
        raise ValueError("PMP neutral output is empty")
    if not np.isfinite(vertex_array).all():
        raise ValueError("PMP neutral output contains non-finite vertices")
    if np.any(triangle_array < 0) or np.any(triangle_array >= len(vertex_array)):
        raise ValueError("PMP neutral output contains out-of-range face indices")
    return vertex_array, triangle_array


def _validate_native_fill_output(
    *,
    mesh: MeshData,
    parameters: _PmpFillParameters,
    result: HardMeshResult,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if result.report is None or result.output_path is None:
        return None, None, "PMP fill returned no native output evidence"
    try:
        vertices, triangles = _read_neutral_obj(Path(result.output_path))
    except (OSError, UnicodeError, ValueError) as exc:
        return None, None, f"PMP neutral output is invalid: {type(exc).__name__}: {exc}"
    source_vertices = np.asarray(mesh.local_vertices, dtype=np.float64)
    source_triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) < len(source_vertices) or len(triangles) <= len(source_triangles):
        return None, None, "PMP fill did not retain all source geometry and add a patch"
    tolerance = max(float(np.linalg.norm(np.ptp(source_vertices, axis=0))) * 1e-7, 1e-9)
    if not np.allclose(vertices[: len(source_vertices)], source_vertices, rtol=0.0, atol=tolerance):
        return None, None, "PMP fill changed a source vertex outside the generated patch"
    if not np.array_equal(triangles[: len(source_triangles)], source_triangles):
        return None, None, "PMP fill changed or reordered a source face outside the patch"
    if len(vertices) - len(source_vertices) > parameters.max_new_vertices:
        return None, None, "PMP fill exceeded max_new_vertices"
    generated_ids = list(range(len(source_triangles), len(triangles)))
    if result.report.generated_face_ids != generated_ids:
        return None, None, "PMP generated-face evidence does not match the neutral output"
    if result.report.changed_face_ids != generated_ids:
        return None, None, "PMP changed-face evidence includes geometry outside the named patch"
    frozen = np.asarray(parameters.frozen_boundary_vertex_ids, dtype=np.int64)
    if not np.allclose(vertices[frozen], source_vertices[frozen], rtol=0.0, atol=tolerance):
        return None, None, "PMP fill moved a frozen source vertex"
    output_edges = {
        tuple(sorted((int(start), int(end))))
        for face in triangles
        for start, end in zip(face, np.roll(face, -1), strict=True)
    }
    protected = {tuple(sorted(edge)) for edge in parameters.protected_edge_vertex_pairs}
    if not protected <= output_edges:
        return None, None, "PMP fill removed a protected edge"
    try:
        correspondence_path = Path(result.report.correspondence_path or "")
        correspondence = json.loads(correspondence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, None, f"PMP correspondence evidence is invalid: {type(exc).__name__}: {exc}"
    expected_mapping = [
        {"output_face_id": index, "source_face_id": index} for index in range(len(source_triangles))
    ]
    if (
        correspondence.get("schema_version") != "geometry-repair.pmp-correspondence.v1"
        or correspondence.get("source_face_count") != len(source_triangles)
        or correspondence.get("output_face_count") != len(triangles)
        or correspondence.get("source_faces") != expected_mapping
    ):
        return None, None, "PMP correspondence evidence is incomplete or non-identity"
    try:
        attribute_path = Path(result.report.attribute_transfer_path or "")
        attribute_transfer = json.loads(attribute_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, None, f"PMP attribute evidence is invalid: {type(exc).__name__}: {exc}"
    if attribute_transfer != {
        "schema_version": "geometry-repair.pmp-attribute-transfer.v1",
        "policy": "preserve_source_faces_generated_patch_inherits_mesh_binding",
        "source_face_attributes_preserved": True,
        "generated_face_attributes": "whole_mesh_binding_only",
    }:
        return None, None, "PMP attribute evidence does not match the adapter transfer policy"
    return vertices, triangles, None


class PmpPatchWorker:
    """Generate one PMP candidate without inferring a hole or editable region."""

    name = "pmp_patch"
    operations = _PMP_SPEC.operations

    def available(self) -> tuple[bool, str | None]:
        executable, capabilities, reason = discover_hard_mesh_executable(_PMP_SPEC)
        if executable is None or capabilities is None:
            return False, reason
        _digest, reason = _verified_pmp_identity(executable, capabilities)
        return reason is None, reason

    def execute_typed(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> HardMeshResult:
        if operation.worker != self.name:
            return _refused(None, f"PMP operation targets unexpected worker {operation.worker!r}")
        operation_name = operation.parameters.get("operation")
        try:
            if operation_name == "pmp_fill_classified_hole":
                parameters = _PmpFillParameters.model_validate(operation.parameters)
            else:
                return _refused(None, f"unsupported PMP operation {operation_name!r}")
        except Exception as exc:
            return _refused(
                str(operation_name) if operation_name else None, f"invalid PMP intent: {exc}"
            )
        if parameters.intent_evidence_id not in operation.issue_ids:
            return _refused(
                parameters.operation,
                "PMP intent_evidence_id must name one diagnosed issue in the repair operation",
            )
        source_path = source.expanduser().resolve()
        if source_path.suffix.lower() not in USD_SUFFIXES:
            return _refused(parameters.operation, "PMP requires a canonical USD working copy")
        mesh, failure = _target_mesh(source_path, parameters.target_mesh_path)
        if failure or mesh is None:
            return _refused(parameters.operation, failure or "PMP target is unavailable")
        failure = _validate_fill_geometry(mesh, parameters)
        if failure:
            return _refused(parameters.operation, failure)
        executable, capabilities, reason = discover_hard_mesh_executable(_PMP_SPEC)
        if executable is None or capabilities is None:
            return HardMeshResult(
                status="unavailable",
                operation=parameters.operation,
                failures=[reason or "PMP executable is unavailable"],
            )
        approved_digest, reason = _verified_pmp_identity(executable, capabilities)
        if approved_digest is None:
            return HardMeshResult(
                status="unavailable",
                operation=parameters.operation,
                failures=[reason or "PMP executable identity is not approved"],
            )
        bridge_dir = output.expanduser().resolve().parent / f"{output.name}.pmp_bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        native_source = _write_neutral_obj(mesh, bridge_dir / "source.obj")
        native_output = bridge_dir / "candidate.obj"
        result = run_hard_mesh_executable(
            spec=_PMP_SPEC,
            executable=executable,
            capabilities=capabilities,
            source=native_source,
            output=native_output,
            request_id=operation.operation_id,
            operation=parameters.operation,
            parameters=parameters.model_dump(mode="json"),
            deterministic_seed=parameters.deterministic_seed,
            timeout_s=parameters.timeout_s,
            approved_executable_sha256=approved_digest,
            expected_backend_source_commit=PMP_SOURCE_COMMIT,
            expected_backend_source_tree_sha256=PMP_SOURCE_TREE_SHA256,
        )
        if result.status != "success" or result.report is None:
            return result
        report = result.report
        required_artifacts = (
            report.changed_region_path,
            report.correspondence_path,
            report.attribute_transfer_path,
        )
        semantic_failure: str | None = None
        if not all(required_artifacts):
            semantic_failure = (
                "PMP success omitted changed-region/correspondence/attribute evidence"
            )
        elif report.correspondence_coverage_ratio != 1.0:
            semantic_failure = "PMP success did not provide complete source-face correspondence"
        elif report.preserved_frozen_vertices is not True:
            semantic_failure = "PMP success did not prove frozen vertices were preserved"
        elif report.preserved_protected_edges is not True:
            semantic_failure = "PMP success did not prove protected edges were preserved"
        elif report.maximum_envelope_ratio is None or (
            report.maximum_envelope_ratio > parameters.max_envelope_ratio
        ):
            semantic_failure = "PMP success exceeded or omitted the requested envelope bound"
        elif not report.generated_face_ids:
            semantic_failure = "PMP hole fill did not identify generated patch faces"
        if semantic_failure:
            if result.output_path:
                Path(result.output_path).unlink(missing_ok=True)
            return HardMeshResult(
                status="failed",
                operation=parameters.operation,
                implementation_version=result.implementation_version,
                build_id=result.build_id,
                report_path=result.report_path,
                report=report,
                failures=[semantic_failure],
            )
        vertices, triangles, semantic_failure = _validate_native_fill_output(
            mesh=mesh,
            parameters=parameters,
            result=result,
        )
        if semantic_failure or vertices is None or triangles is None:
            output.expanduser().resolve().unlink(missing_ok=True)
            return HardMeshResult(
                status="failed",
                operation=parameters.operation,
                implementation_version=result.implementation_version,
                build_id=result.build_id,
                report_path=result.report_path,
                report=report,
                failures=[semantic_failure or "PMP fill output validation failed"],
            )
        output_path = output.expanduser().resolve()
        try:
            source_vertex_count = len(mesh.local_vertices)
            if len(vertices) == source_vertex_count:
                boundary_sources = _boundary_source_faces(
                    mesh,
                    parameters.boundary_loop_vertex_ids,
                )
                generated_face_sources = dict.fromkeys(
                    range(len(mesh.triangles), len(triangles)),
                    boundary_sources,
                )
                rewrite_usd_triangle_meshes_with_generated_patches(
                    source_path,
                    output_path,
                    {
                        parameters.target_mesh_path: GeneratedPatchTopologyRewrite(
                            vertices=np.asarray(mesh.local_vertices, dtype=np.float64),
                            triangles=triangles,
                            generated_face_sources=generated_face_sources,
                        )
                    },
                )
                attribute_policy = "exact_boundary_agreement_or_refuse"
            else:
                raise ValueError(
                    "PMP introduced patch vertices; this local-surgery route permits generated "
                    "faces over frozen source-boundary vertices only"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            output_path.unlink(missing_ok=True)
            return HardMeshResult(
                status="failed",
                operation=parameters.operation,
                implementation_version=result.implementation_version,
                build_id=result.build_id,
                report_path=result.report_path,
                report=report,
                failures=[f"PMP USD patch authoring failed: {type(exc).__name__}: {exc}"],
            )
        result = result.model_copy(
            update={
                "output_path": str(output_path),
                "output_sha256": file_sha256(output_path),
                "metadata": {
                    **result.metadata,
                    "native_neutral_output_path": result.output_path,
                    "native_neutral_output_sha256": result.output_sha256,
                    "usd_target_mesh_path": parameters.target_mesh_path,
                    "attribute_policy": attribute_policy,
                },
            }
        )
        return result

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        return self.execute_typed(
            source=source, output=output, operation=operation
        ).to_worker_result()
