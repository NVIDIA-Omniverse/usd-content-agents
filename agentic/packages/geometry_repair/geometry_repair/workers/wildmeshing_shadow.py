# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shadow-only Wildmeshing adapter for one explicitly named local patch."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..hard_mesh_policy import (
    WILDMESHING_BUILD_ID,
    HardMeshExecutableSpec,
    HardMeshResult,
    discover_hard_mesh_executable,
    run_hard_mesh_executable,
)
from ..mesh_io import USD_SUFFIXES, MeshData, load_meshes
from ..models import RepairOperation
from .base import WorkerResult

_REQUIRED_INVARIANTS = frozenset(
    {
        "frozen_vertices",
        "manifoldness",
        "material_boundaries",
        "orientation",
        "part_boundaries",
        "protected_edges",
        "surface_envelope",
        "uv_seams",
    }
)

_WILDMESHING_SPEC = HardMeshExecutableSpec(
    worker="wildmeshing_shadow",
    environment_variable="GEOMETRY_REPAIR_WILDMESHING_EXECUTABLE",
    executable_names=("geometry_repair_wildmeshing_shadow",),
    build_id=WILDMESHING_BUILD_ID,
    operations=frozenset({"wildmeshing_remesh_named_patch"}),
    required_capabilities=frozenset(
        {
            "attribute_update_rules",
            "changed_region_report",
            "deterministic_single_thread",
            "invariant_callbacks",
            "operation_rollback",
            "source_face_barycentric_correspondence",
            "surface_envelope",
        }
    ),
    shadow_only=True,
)


class _WildmeshingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["wildmeshing_remesh_named_patch"]
    target_mesh_path: str = Field(min_length=1)
    region_intent: Literal["named_local_remesh"]
    intent_evidence_id: str = Field(min_length=1)
    region_face_ids: list[int] = Field(min_length=1, max_length=250_000)
    transition_face_ids: list[int] = Field(default_factory=list, max_length=250_000)
    frozen_vertex_ids: list[int] = Field(min_length=1, max_length=1_000_000)
    protected_edge_vertex_pairs: list[tuple[int, int]] = Field(min_length=1, max_length=250_000)
    required_invariants: list[
        Literal[
            "frozen_vertices",
            "manifoldness",
            "material_boundaries",
            "orientation",
            "part_boundaries",
            "protected_edges",
            "surface_envelope",
            "uv_seams",
        ]
    ] = Field(min_length=8, max_length=8)
    correspondence_mode: Literal["source_face_barycentric"]
    attribute_policy: Literal["preserve_or_refuse"]
    rollback_on_invariant_failure: Literal[True]
    target_edge_length_ratio: float = Field(gt=0.0, le=0.05)
    max_envelope_ratio: float = Field(gt=0.0, le=0.005)
    max_operations: int = Field(default=100_000, ge=1, le=1_000_000)
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    timeout_s: float = Field(default=120.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _validate_contract(self) -> _WildmeshingParameters:
        if set(self.required_invariants) != _REQUIRED_INVARIANTS:
            raise ValueError("Wildmeshing requires the complete invariant set")
        if len(set(self.region_face_ids)) != len(self.region_face_ids):
            raise ValueError("region_face_ids must be unique")
        if len(set(self.transition_face_ids)) != len(self.transition_face_ids):
            raise ValueError("transition_face_ids must be unique")
        if set(self.region_face_ids) & set(self.transition_face_ids):
            raise ValueError("region and transition face sets must be disjoint")
        if len(set(self.frozen_vertex_ids)) != len(self.frozen_vertex_ids):
            raise ValueError("frozen_vertex_ids must be unique")
        if any(start == end for start, end in self.protected_edge_vertex_pairs):
            raise ValueError("protected edges must have distinct endpoints")
        return self


def _refused(operation: str | None, message: str) -> HardMeshResult:
    return HardMeshResult(status="refused", operation=operation, failures=[message])


def _target_mesh(source: Path, target_path: str) -> tuple[MeshData | None, str | None]:
    try:
        meshes, _ = load_meshes(source)
    except Exception as exc:
        return None, f"Wildmeshing could not inspect source: {type(exc).__name__}: {exc}"
    matches = [mesh for mesh in meshes if mesh.path == target_path and mesh.role == "render"]
    if len(matches) != 1:
        return None, f"Wildmeshing target {target_path!r} must identify one render mesh"
    mesh = matches[0]
    unsupported = []
    if mesh.is_instance_proxy:
        unsupported.append("instance proxy")
    if mesh.material_subset_count:
        unsupported.append("face material subsets")
    if mesh.has_face_varying_data:
        unsupported.append("face-varying data")
    if mesh.authored_uv_count:
        unsupported.append("authored UVs")
    if mesh.authored_normal_count:
        unsupported.append("authored normals")
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
            f"Wildmeshing shadow refused {target_path!r} because exact transfer is unavailable for: "
            + ", ".join(unsupported)
        )
    return mesh, None


def _validate_source_ids(mesh: MeshData, parameters: _WildmeshingParameters) -> str | None:
    face_count = len(mesh.triangles)
    if any(
        face_id < 0 or face_id >= face_count
        for face_id in (*parameters.region_face_ids, *parameters.transition_face_ids)
    ):
        return "Wildmeshing face scope contains an out-of-range source identifier"
    vertex_count = len(mesh.local_vertices)
    vertex_ids = [
        *parameters.frozen_vertex_ids,
        *(value for edge in parameters.protected_edge_vertex_pairs for value in edge),
    ]
    if any(vertex_id < 0 or vertex_id >= vertex_count for vertex_id in vertex_ids):
        return "Wildmeshing protection scope contains an out-of-range source vertex identifier"
    return None


class WildmeshingShadowWorker:
    """Run Wildmeshing for comparison evidence, never as an accepting route."""

    name = "wildmeshing_shadow"

    def available(self) -> tuple[bool, str | None]:
        executable, _capabilities, reason = discover_hard_mesh_executable(_WILDMESHING_SPEC)
        return executable is not None, reason

    def execute_typed(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> HardMeshResult:
        if operation.worker != self.name:
            return _refused(None, f"Wildmeshing operation targets {operation.worker!r}")
        try:
            parameters = _WildmeshingParameters.model_validate(operation.parameters)
        except Exception as exc:
            return _refused(None, f"invalid Wildmeshing intent/invariants: {exc}")
        if parameters.intent_evidence_id not in operation.issue_ids:
            return _refused(
                parameters.operation,
                "Wildmeshing intent_evidence_id must name a diagnosed issue",
            )
        source_path = source.expanduser().resolve()
        if source_path.suffix.lower() not in USD_SUFFIXES:
            return _refused(parameters.operation, "Wildmeshing requires a canonical USD source")
        mesh, failure = _target_mesh(source_path, parameters.target_mesh_path)
        if failure or mesh is None:
            return _refused(parameters.operation, failure or "Wildmeshing target unavailable")
        failure = _validate_source_ids(mesh, parameters)
        if failure:
            return _refused(parameters.operation, failure)
        executable, capabilities, reason = discover_hard_mesh_executable(_WILDMESHING_SPEC)
        if executable is None or capabilities is None:
            return HardMeshResult(
                status="unavailable",
                operation=parameters.operation,
                failures=[reason or "Wildmeshing executable is unavailable"],
                metadata={"shadow_only": True},
            )
        result = run_hard_mesh_executable(
            spec=_WILDMESHING_SPEC,
            executable=executable,
            capabilities=capabilities,
            source=source_path,
            output=output,
            request_id=operation.operation_id,
            operation=parameters.operation,
            parameters=parameters.model_dump(mode="json"),
            deterministic_seed=parameters.deterministic_seed,
            timeout_s=parameters.timeout_s,
        )
        if result.status != "success" or result.report is None:
            return result
        report = result.report
        semantic_failure: str | None = None
        if not (
            report.changed_region_path
            and report.correspondence_path
            and report.attribute_transfer_path
        ):
            semantic_failure = "Wildmeshing omitted required changed-region/correspondence evidence"
        elif report.correspondence_coverage_ratio != 1.0:
            semantic_failure = "Wildmeshing correspondence coverage was not complete"
        elif report.preserved_frozen_vertices is not True:
            semantic_failure = "Wildmeshing did not prove frozen vertices were preserved"
        elif report.preserved_protected_edges is not True:
            semantic_failure = "Wildmeshing did not prove protected edges were preserved"
        elif report.maximum_envelope_ratio is None or (
            report.maximum_envelope_ratio > parameters.max_envelope_ratio
        ):
            semantic_failure = "Wildmeshing exceeded or omitted its surface envelope"
        elif not _REQUIRED_INVARIANTS <= set(report.invariants_satisfied):
            semantic_failure = "Wildmeshing did not prove every requested invariant"
        elif not report.changed_face_ids:
            semantic_failure = "Wildmeshing did not identify its changed source region"
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
                metadata={"shadow_only": True},
            )
        result.metadata["shadow_only"] = True
        return result

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        typed = self.execute_typed(source=source, output=output, operation=operation)
        projected = typed.to_worker_result()
        if typed.status != "success":
            return projected
        return WorkerResult(
            status="unavailable",
            changed=False,
            warnings=[*projected.warnings, "Wildmeshing result is shadow evidence only"],
            failures=["shadow-only Wildmeshing output cannot satisfy a repair operation"],
            metadata={
                **projected.metadata,
                "shadow_candidate_path": typed.output_path,
                "shadow_candidate_sha256": typed.output_sha256,
                "shadow_only": True,
            },
        )
