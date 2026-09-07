# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shadow-only MCUT partition adapter for exactly two named source parts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..hard_mesh_policy import (
    MCUT_BUILD_ID,
    HardMeshExecutableSpec,
    HardMeshResult,
    discover_hard_mesh_executable,
    run_hard_mesh_executable,
)
from ..mesh_io import USD_SUFFIXES, MeshData, load_meshes
from ..models import RepairOperation
from .base import WorkerResult

_MCUT_SPEC = HardMeshExecutableSpec(
    worker="mcut_shadow",
    environment_variable="GEOMETRY_REPAIR_MCUT_EXECUTABLE",
    executable_names=("geometry_repair_mcut_shadow",),
    build_id=MCUT_BUILD_ID,
    operations=frozenset({"mcut_partition_source_parts"}),
    required_capabilities=frozenset(
        {
            "deterministic_single_thread",
            "exactly_two_input_meshes",
            "no_fragment_selection",
            "return_all_fragments",
            "return_intersection_curves",
            "source_face_correspondence",
        }
    ),
    shadow_only=True,
)


class _McutParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["mcut_partition_source_parts"]
    part_paths: tuple[str, str]
    region_intent: Literal["diagnosed_two_part_intersection"]
    intent_evidence_id: str = Field(min_length=1)
    protected_feature_ids: list[str] = Field(min_length=1, max_length=1024)
    fragment_selection: Literal["none"]
    selected_fragment_ids: list[str] = Field(default_factory=list, max_length=0)
    deleted_fragment_ids: list[str] = Field(default_factory=list, max_length=0)
    combine_fragments: Literal[False]
    max_fragments: int = Field(default=256, ge=2, le=4096)
    max_intersection_curves: int = Field(default=4096, ge=1, le=100_000)
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    timeout_s: float = Field(default=120.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _validate_parts(self) -> _McutParameters:
        if any(not path for path in self.part_paths):
            raise ValueError("MCUT part paths must be non-empty")
        if self.part_paths[0] == self.part_paths[1]:
            raise ValueError("MCUT requires two distinct named source parts")
        if len(set(self.protected_feature_ids)) != len(self.protected_feature_ids):
            raise ValueError("protected_feature_ids must be unique")
        return self


def _refused(operation: str | None, message: str) -> HardMeshResult:
    return HardMeshResult(status="refused", operation=operation, failures=[message])


def _target_parts(source: Path, part_paths: tuple[str, str]) -> tuple[list[MeshData], str | None]:
    try:
        meshes, _ = load_meshes(source)
    except Exception as exc:
        return [], f"MCUT could not inspect source: {type(exc).__name__}: {exc}"
    by_path = {mesh.path: mesh for mesh in meshes if mesh.role == "render"}
    if any(path not in by_path for path in part_paths):
        return [], "MCUT part_paths must identify exactly two existing render mesh prims"
    selected = [by_path[path] for path in part_paths]
    unsupported: list[str] = []
    for mesh in selected:
        attributes = []
        if mesh.is_instance_proxy:
            attributes.append("instance proxy")
        if mesh.material_subset_count:
            attributes.append("face material subsets")
        if mesh.has_face_varying_data:
            attributes.append("face-varying data")
        if mesh.authored_uv_count:
            attributes.append("authored UVs")
        if mesh.authored_normal_count:
            attributes.append("authored normals")
        if len(mesh.source_face_counts) != len(mesh.triangles) or any(
            int(count) != 3 for count in mesh.source_face_counts
        ):
            attributes.append("non-triangular source faces")
        if mesh.transform_non_finite or mesh.transform_singular:
            attributes.append("invalid source transform")
        if mesh.transform_non_uniform or mesh.transform_sheared or mesh.transform_reflected:
            attributes.append("non-rigid or reflected source transform")
        if attributes:
            unsupported.append(f"{mesh.path}: {', '.join(attributes)}")
    if unsupported:
        return [], (
            "MCUT shadow refused attributed parts without exact transfer support: "
            + "; ".join(unsupported)
        )
    return selected, None


class McutShadowWorker:
    """Return every MCUT fragment and curve without choosing asset intent."""

    name = "mcut_shadow"

    def available(self) -> tuple[bool, str | None]:
        executable, _capabilities, reason = discover_hard_mesh_executable(_MCUT_SPEC)
        return executable is not None, reason

    def execute_typed(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> HardMeshResult:
        if operation.worker != self.name:
            return _refused(None, f"MCUT operation targets {operation.worker!r}")
        try:
            parameters = _McutParameters.model_validate(operation.parameters)
        except Exception as exc:
            return _refused(None, f"invalid MCUT two-part intent: {exc}")
        if parameters.intent_evidence_id not in operation.issue_ids:
            return _refused(
                parameters.operation,
                "MCUT intent_evidence_id must name one diagnosed intersection issue",
            )
        source_path = source.expanduser().resolve()
        if source_path.suffix.lower() not in USD_SUFFIXES:
            return _refused(parameters.operation, "MCUT requires a canonical USD source")
        _parts, failure = _target_parts(source_path, parameters.part_paths)
        if failure:
            return _refused(parameters.operation, failure)
        executable, capabilities, reason = discover_hard_mesh_executable(_MCUT_SPEC)
        if executable is None or capabilities is None:
            return HardMeshResult(
                status="unavailable",
                operation=parameters.operation,
                failures=[reason or "MCUT executable is unavailable"],
                metadata={"shadow_only": True, "fragment_selection": "none"},
            )
        result = run_hard_mesh_executable(
            spec=_MCUT_SPEC,
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
        source_paths = {fragment.source_part_path for fragment in report.fragments}
        semantic_failure: str | None = None
        if report.fragment_selection_performed or report.deleted_fragment_ids:
            semantic_failure = "MCUT selected or deleted fragments, which this adapter forbids"
        elif not report.fragments or source_paths != set(parameters.part_paths):
            semantic_failure = "MCUT did not return all fragments associated with both source parts"
        elif len(report.fragments) > parameters.max_fragments:
            semantic_failure = "MCUT fragment count exceeded the request bound"
        elif not report.intersection_curves:
            semantic_failure = "MCUT did not return intersection curves"
        elif len(report.intersection_curves) > parameters.max_intersection_curves:
            semantic_failure = "MCUT intersection-curve count exceeded the request bound"
        elif not report.changed_region_path or not report.correspondence_path:
            semantic_failure = "MCUT omitted changed-region or source-face correspondence evidence"
        elif report.correspondence_coverage_ratio != 1.0:
            semantic_failure = "MCUT correspondence coverage was not complete"
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
                metadata={"shadow_only": True, "fragment_selection": "none"},
            )
        result.metadata.update(
            {
                "shadow_only": True,
                "fragment_selection": "none",
                "fragment_count": len(report.fragments),
                "intersection_curve_count": len(report.intersection_curves),
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
        typed = self.execute_typed(source=source, output=output, operation=operation)
        projected = typed.to_worker_result()
        if typed.status != "success":
            return projected
        return WorkerResult(
            status="unavailable",
            changed=False,
            warnings=[*projected.warnings, "MCUT result is unclassified shadow evidence only"],
            failures=["shadow-only MCUT output cannot satisfy a repair operation"],
            metadata={
                **projected.metadata,
                "shadow_candidate_path": typed.output_path,
                "shadow_candidate_sha256": typed.output_sha256,
                "shadow_only": True,
                "fragment_selection": "none",
            },
        )
