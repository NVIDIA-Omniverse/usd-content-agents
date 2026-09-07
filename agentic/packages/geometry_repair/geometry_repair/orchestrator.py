# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded repair planning, execution, rollback, and certification."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from .advanced_profiles import (
    AdvancedProfileEvidenceReport,
    AdvancedProfileRequest,
    collision_unavailable_advanced_profile_report,
    evaluate_advanced_profile,
    write_advanced_profile_report,
)
from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    classify_unresolved_dependencies,
    file_sha256,
    preserve_source,
)
from .brep import BREP_SUFFIXES
from .collision import compose_asset
from .collision_audit import SourceCollisionAudit
from .correspondence import (
    CorrespondenceEvidence,
    build_mesh_correspondence,
    identity_correspondence,
    write_correspondence_evidence,
)
from .dependency_localization import (
    DependencyLocalizationReport,
    localize_usd_dependencies,
)
from .diagnosis import diagnose_asset
from .fidelity import compare_geometry
from .format_validation import validate_source_format
from .mesh_io import (
    MESH_SUFFIXES,
    USD_SUFFIXES,
    copy_usd_stage,
    external_mesh_to_usd_stage,
    flatten_usd_stage,
    separate_non_render_geometry,
)
from .models import (
    AttemptRecord,
    CollisionReport,
    Diagnosis,
    FidelityReport,
    ProtectedFeature,
    RepairBudgets,
    RepairCertificate,
    RepairOperation,
    RepairPlan,
    RepairProfile,
    RepairRequest,
    RepairResult,
    SourceFormatValidationReport,
)
from .policy import assert_worker_enabled, assert_worker_operations_enabled
from .process_limits import (
    ADDRESS_SPACE_LIMIT_MODE_HARD,
    ADDRESS_SPACE_LIMIT_MODE_SOFT,
    ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
    OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
    OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
    bounded_process_environment,
)
from .protected_features import (
    ProtectedFeatureCandidateReport,
    candidate_to_protected_feature,
    compare_protected_feature_candidates,
    detect_protected_feature_candidates,
)
from .repair_intent import (
    bind_deterministic_repair_intents,
    rank_operations_from_proposed_intents,
    validate_proposed_repair_intents,
)
from .roles import build_diagnosis_role_validation, build_final_role_validation
from .router import ranked_workers
from .scalable_audit import ScalableAuditBudget
from .sdf_backend_qualification import (
    DEFAULT_SDF_BACKEND_ID,
    SDF_REBUILD_IMPLEMENTATION_VERSION,
)
from .usd_intake import UsdIntakeReport, inventory_usd_stage
from .worker_ids import SDF_COLLISION_REBUILD_WORKER
from .workers import (
    CollisionGeometryWorker,
    GeogramLocalRepairWorker,
    OcpShapeHealWorker,
    PmpPatchWorker,
    SceneOptimizerDeinstanceWorker,
    SdfRebuildWorker,
    TrimeshBoundedHoleFillWorker,
    TrimeshCleanupWorker,
    UsdStructureRepairWorker,
)
from .workers.base import RepairWorker, WorkerResult
from .workers.manifold_seam import ManifoldSeamAnalysis, ManifoldSeamWorker

logger = logging.getLogger(__name__)

_WORKERS: dict[str, RepairWorker] = {
    TrimeshCleanupWorker.name: TrimeshCleanupWorker(),
    TrimeshBoundedHoleFillWorker.name: TrimeshBoundedHoleFillWorker(),
    OcpShapeHealWorker.name: OcpShapeHealWorker(),
    UsdStructureRepairWorker.name: UsdStructureRepairWorker(),
    GeogramLocalRepairWorker.name: GeogramLocalRepairWorker(),
    SceneOptimizerDeinstanceWorker.name: SceneOptimizerDeinstanceWorker(),
    SdfRebuildWorker.name: SdfRebuildWorker(),
    ManifoldSeamWorker.name: ManifoldSeamWorker(),
    PmpPatchWorker.name: PmpPatchWorker(),
}
_COLLISION_WORKERS = {"coacd_collision", SDF_COLLISION_REBUILD_WORKER}
_DEFAULT_WORKERS = (*_WORKERS, *_COLLISION_WORKERS)
_TRIMESH_REPAIRABLE = {
    "mesh:degenerate_faces",
    "mesh:duplicate_faces",
    "mesh:inconsistent_orientation",
    "mesh:inverted_shells",
}
_USD_STRUCTURE_REPAIRABLE = {
    "asset:default_prim_missing",
    "asset:noncanonical_up_axis",
    "asset:noncanonical_stage_units",
}
_GEOGRAM_REPAIRABLE = {
    "mesh:over_connected_edges",
    "mesh:non_manifold_vertices",
    "mesh:self_intersections",
    "mesh:self_intersections_not_evaluated",
    "mesh:coplanar_overlaps",
    "mesh:coplanar_overlaps_not_evaluated",
    "mesh:needle_triangles",
    "mesh:inverted_shells_not_evaluated",
}
_WORKER_VERIFIABLE_ISSUE_IDS = frozenset(
    {
        "mesh:coplanar_overlaps_not_evaluated",
        "mesh:self_intersections_not_evaluated",
    }
)
_SDF_REPAIRABLE = {
    "mesh:boundary_edges",
    "mesh:coplanar_overlaps",
    "mesh:coplanar_overlaps_not_evaluated",
    "mesh:degenerate_faces",
    "mesh:duplicate_faces",
    "mesh:inconsistent_orientation",
    "mesh:inverted_shells",
    "mesh:inverted_shells_not_evaluated",
    "mesh:needle_triangles",
    "mesh:non_manifold_vertices",
    "mesh:over_connected_edges",
    "mesh:self_intersections",
    "mesh:self_intersections_not_evaluated",
}


def _admitted_worker_verified_issue_ids(
    reported_issue_ids: list[str],
    blocking_issue_ids: list[str],
) -> list[str]:
    """Admit only independent-audit gaps that a bounded worker may close."""

    return sorted(set(reported_issue_ids) & set(blocking_issue_ids) & _WORKER_VERIFIABLE_ISSUE_IDS)


def _normalized_copy(
    source: Path,
    output_dir: Path,
    *,
    source_meters_per_unit: float | None = None,
    source_up_axis: str | None = None,
) -> Path:
    normalized_dir = output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    target_suffix = ".usd" if suffix in USD_SUFFIXES | MESH_SUFFIXES else suffix
    target = normalized_dir / f"source{target_suffix}"
    if suffix in USD_SUFFIXES:
        return copy_usd_stage(source, target)
    if suffix in MESH_SUFFIXES:
        try:
            return external_mesh_to_usd_stage(
                source,
                target,
                meters_per_unit=source_meters_per_unit,
                up_axis=source_up_axis,
            )
        except ValueError as exc:
            if "physical normalization requires" not in str(exc):
                raise
            # Preserve a diagnosable raw working copy when the caller has not
            # declared the physical frame. Repair planning must remain
            # fail-closed instead of pretending an uncreated USD layer exists.
            target = normalized_dir / f"source{suffix}"
    shutil.copy2(source, target)
    return target


def _enabled_workers(request: RepairRequest) -> set[str]:
    requested = set(request.enabled_workers or _DEFAULT_WORKERS)
    unknown = requested - (set(_WORKERS) | _COLLISION_WORKERS)
    if unknown:
        raise ValueError(f"Unknown or unapproved repair workers: {', '.join(sorted(unknown))}")
    for name in sorted(requested):
        assert_worker_enabled(name)
    return requested


def _plan_repair(
    request: RepairRequest,
    diagnosis: Diagnosis,
    normalized_source: Path,
    plan_path: Path,
) -> RepairPlan:
    assert_worker_operations_enabled("noop", "immutable_copy")
    enabled = _enabled_workers(request)
    ranked, routing_evidence = ranked_workers(set(diagnosis.blocking_issue_ids), enabled)
    operations = [
        RepairOperation(
            operation_id="attempt-00-noop",
            worker="noop",
            implementation="geometry_repair.orchestrator.copy_source",
            issue_ids=[],
            drift_band="identity",
            source_checkpoint=str(normalized_source),
            expected_changes=[],
        )
    ]
    if request.mode == "auto" and diagnosis.blocking_issue_ids:
        blocking = set(diagnosis.blocking_issue_ids)
        cleanup_blockers = _TRIMESH_REPAIRABLE | {"mesh:over_connected_edges"}
        duplicate_induced_nonmanifold = "mesh:duplicate_faces" in blocking and bool(
            blocking & cleanup_blockers
        )
        mesh_blocking = blocking & cleanup_blockers
        structure_blocking = blocking & _USD_STRUCTURE_REPAIRABLE
        geogram_blocking = blocking & _GEOGRAM_REPAIRABLE
        sdf_blocking = blocking & _SDF_REPAIRABLE
        bounded_hole_blocking = blocking & {"mesh:boundary_edges"}
        if (
            diagnosis.metrics.instance_proxy_mesh_count
            and (mesh_blocking or geogram_blocking or sdf_blocking)
            and SceneOptimizerDeinstanceWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            operations.append(
                RepairOperation(
                    operation_id="attempt-01-scene-optimizer-deinstance",
                    worker=SceneOptimizerDeinstanceWorker.name,
                    implementation=(
                        "world_understanding.functions.graphics."
                        "scene_optimizer_local.optimize_usd_local"
                    ),
                    parameters={"timeout_s": min(request.budgets.timeout_s, 300.0)},
                    issue_ids=[],
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "materialize read-only USD instances without merging source parts"
                    ],
                )
            )
        candidates: dict[str, list[RepairOperation]] = {}
        if (
            mesh_blocking
            and (mesh_blocking <= _TRIMESH_REPAIRABLE or duplicate_induced_nonmanifold)
            and TrimeshCleanupWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            candidates[TrimeshCleanupWorker.name] = [
                RepairOperation(
                    operation_id="candidate-trimesh-cleanup",
                    worker=TrimeshCleanupWorker.name,
                    implementation="geometry_repair.workers.trimesh_cleanup",
                    issue_ids=sorted(mesh_blocking),
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "remove exact duplicate or degenerate triangles",
                        "correct inconsistent manifold face winding",
                    ],
                )
            ]
        if (
            structure_blocking
            and UsdStructureRepairWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            candidates[UsdStructureRepairWorker.name] = [
                RepairOperation(
                    operation_id="candidate-usd-structure",
                    worker=UsdStructureRepairWorker.name,
                    implementation="geometry_repair.workers.usd_structure",
                    issue_ids=sorted(structure_blocking),
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "assign an unambiguous default prim",
                        "normalize physical units and up axis through a root transform",
                    ],
                )
            ]
        if (
            geogram_blocking
            and GeogramLocalRepairWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            candidates[GeogramLocalRepairWorker.name] = [
                RepairOperation(
                    operation_id="candidate-geogram-local-repair",
                    worker=GeogramLocalRepairWorker.name,
                    implementation=(f"Geogram.vorpalite-{GeogramLocalRepairWorker.version}"),
                    parameters={"epsilon_ratio": 1e-8},
                    issue_ids=sorted(geogram_blocking),
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "repair local mesh topology without hole filling or component removal",
                        "remove triangle self-intersections when diagnosed",
                    ],
                )
            ]
        if (
            bounded_hole_blocking
            and request.budgets.allow_reconstructive
            and TrimeshBoundedHoleFillWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            candidates[TrimeshBoundedHoleFillWorker.name] = [
                RepairOperation(
                    operation_id="candidate-trimesh-bounded-hole-fill",
                    worker=TrimeshBoundedHoleFillWorker.name,
                    implementation="geometry_repair.workers.trimesh_hole_fill",
                    parameters={
                        "max_loop_vertices": 4,
                        "max_loop_count": 8,
                        "max_perimeter_ratio": 0.2,
                        "max_added_area_ratio": 0.02,
                        "planarity_ratio": 1e-4,
                    },
                    issue_ids=sorted(bounded_hole_blocking),
                    drift_band="reconstructive",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "close only triangular or planar quadrilateral manifold boundary loops",
                        "preserve every existing source vertex, face, part path, and uniform material",
                    ],
                )
            ]
        if (
            bounded_hole_blocking
            and request.classified_holes
            and PmpPatchWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            candidates[PmpPatchWorker.name] = [
                RepairOperation(
                    operation_id=f"candidate-pmp-classified-hole-{index:02d}",
                    worker=PmpPatchWorker.name,
                    implementation="geometry_repair.workers.pmp_patch+PMP",
                    parameters={
                        "operation": "pmp_fill_classified_hole",
                        "target_mesh_path": intent.target_mesh_path,
                        "region_intent": "classified_accidental_hole",
                        "intent_evidence_id": intent.intent_evidence_id,
                        "boundary_loop_vertex_ids": intent.boundary_loop_vertex_ids,
                        "frozen_boundary_vertex_ids": intent.frozen_boundary_vertex_ids,
                        "protected_edge_vertex_pairs": intent.protected_edge_vertex_pairs,
                        "max_loop_perimeter_ratio": intent.max_loop_perimeter_ratio,
                        "max_patch_area_ratio": intent.max_patch_area_ratio,
                        "max_nonplanarity_ratio": intent.max_nonplanarity_ratio,
                        "max_boundary_turn_radians": intent.max_boundary_turn_radians,
                        "max_envelope_ratio": intent.max_envelope_ratio,
                        "max_new_vertices": intent.max_new_vertices,
                        "deterministic_seed": request.deterministic_seed,
                        "timeout_s": intent.timeout_s,
                    },
                    issue_ids=sorted({intent.issue_id, intent.intent_evidence_id}),
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "fill only the explicitly classified accidental boundary loop",
                        "freeze source vertices and preserve every protected boundary edge",
                        "emit changed-region, source-face correspondence, and attribute evidence",
                    ],
                )
                for index, intent in enumerate(request.classified_holes)
                if intent.issue_id in blocking
            ]
        if (
            sdf_blocking
            and request.budgets.allow_reconstructive
            and SdfRebuildWorker.name in enabled
            and normalized_source.suffix.lower() in USD_SUFFIXES
        ):
            feature_scales = [
                value
                for feature in request.protected_features
                if feature.required
                for value in (
                    feature.minimum_size_m,
                    feature.minimum_clearance_m,
                    2.0 * feature.probe.radius_m
                    if feature.probe is not None and feature.probe.radius_m is not None
                    else None,
                )
                if value is not None
            ]
            # Unsigned offsets thicken sheets and can invent a second surface. Keep
            # automatic production routing on signed reconstruction; the unsigned
            # mode remains available only to explicit evaluation plans.
            mode = "signed"
            maximum_grid = request.budgets.max_reconstruction_grid_dimension
            grid_ladder: list[int] = []
            for ratio in (1.0, 0.75, 0.625, 0.5):
                dimension = max(32, int(round(maximum_grid * ratio)))
                if dimension not in grid_ladder:
                    grid_ladder.append(dimension)
            # Each resolution is an immutable candidate measured against the
            # original source. The highest-resolution candidate that clears all
            # gates wins; lower grids can close discretization-scale defects that
            # remain open in a finer signed rasterization.
            candidates[SdfRebuildWorker.name] = [
                RepairOperation(
                    operation_id=f"candidate-sdf-rebuild-grid-{grid_dimension}",
                    worker=SdfRebuildWorker.name,
                    implementation=(
                        f"geometry_repair.workers.sdf_rebuild+{SDF_REBUILD_IMPLEMENTATION_VERSION}"
                    ),
                    parameters={
                        "backend_id": DEFAULT_SDF_BACKEND_ID,
                        "mode": mode,
                        "max_grid_dimension": grid_dimension,
                        "max_output_faces": request.budgets.max_reconstruction_output_faces,
                        "feature_voxels": request.budgets.reconstruction_feature_voxels,
                        "minimum_feature_m": min(feature_scales) if feature_scales else None,
                        "half_width": 4.0 if mode == "unsigned_offset" else 3.0,
                        "offset_voxels": 0.75,
                        "adaptivity": 0.05,
                        "closing_steps": 2,
                        "smoothing_steps": 1,
                        "timeout_s": request.budgets.timeout_s,
                    },
                    issue_ids=sorted(sdf_blocking),
                    drift_band="reconstructive",
                    source_checkpoint=str(normalized_source),
                    expected_changes=[
                        "replace each eligible source mesh with a bounded per-part level-set surface",
                        "preserve USD part paths while allowing explicitly reconstructive topology",
                        "close discretization-scale source holes by at most 2 topology voxels",
                        "apply one bounded mean-filter step to suppress topology-grid ripples",
                        f"bound the reconstruction grid to {grid_dimension} cells",
                    ],
                )
                for grid_dimension in grid_ladder
            ]
        if (
            blocking == {"brep:invalid_shape"}
            and OcpShapeHealWorker.name in enabled
            and normalized_source.suffix.lower() in BREP_SUFFIXES
        ):
            diagonal = 1.0
            bounds_min = diagnosis.metrics.brep.bbox_min_source_units
            bounds_max = diagnosis.metrics.brep.bbox_max_source_units
            if bounds_min and bounds_max:
                diagonal = max(
                    sum(
                        (float(maximum) - float(minimum)) ** 2
                        for minimum, maximum in zip(bounds_min, bounds_max, strict=True)
                    )
                    ** 0.5,
                    1e-9,
                )
            precision = diagonal * 1e-9
            candidates[OcpShapeHealWorker.name] = [
                RepairOperation(
                    operation_id="candidate-ocp-shape-heal",
                    worker=OcpShapeHealWorker.name,
                    implementation="OCP.ShapeFix.ShapeFix_Shape",
                    parameters={
                        "precision": precision,
                        "maximum_tolerance": precision * 10.0,
                        "sew": True,
                    },
                    issue_ids=sorted(blocking),
                    drift_band="conservative",
                    source_checkpoint=str(normalized_source),
                    expected_changes=["heal invalid native B-rep topology within tolerance budget"],
                )
            ]
        for worker_name in ranked:
            worker_candidates = candidates.get(worker_name)
            if worker_candidates is None:
                continue
            for candidate in worker_candidates:
                operations.append(
                    candidate.model_copy(
                        update={
                            "operation_id": (
                                f"attempt-{len(operations):02d}-"
                                f"{candidate.operation_id.removeprefix('candidate-')}"
                            )
                        }
                    )
                )
    rationale = (
        "Diagnose-only mode evaluates the immutable normalized source."
        if request.mode == "diagnose"
        else "Try the no-op candidate first, then a bounded sequence of least-destructive approved workers; advance only candidates that strictly reduce blockers and pass source-relative fidelity."
    )
    bound_operations, all_deterministic_intents = bind_deterministic_repair_intents(
        operations,
        diagnosis,
        request.protected_features,
    )
    proposed_intents = validate_proposed_repair_intents(
        request.proposed_intents,
        diagnosis,
        enabled_workers=enabled,
        protected_features=request.protected_features,
    )
    duplicate_intent_ids = {intent.intent_id for intent in all_deterministic_intents} & {
        intent.intent_id for intent in proposed_intents
    }
    if duplicate_intent_ids:
        raise ValueError(
            "proposed repair intent IDs collide with deterministic plan IDs: "
            + ", ".join(sorted(duplicate_intent_ids))
        )
    ranking_evidence: list[dict[str, object]] = []
    if request.use_proposed_intent_ranking:
        bound_operations, ranking_evidence = rank_operations_from_proposed_intents(
            bound_operations,
            proposed_intents,
        )
    bounded_operations = bound_operations[: request.budgets.max_attempts]
    retained_intent_ids = {
        operation.intent_id for operation in bounded_operations if operation.intent_id
    }
    deterministic_intents = [
        intent for intent in all_deterministic_intents if intent.intent_id in retained_intent_ids
    ]
    routing_evidence = {
        **routing_evidence,
        "proposed_intent_ids": [intent.intent_id for intent in proposed_intents],
        "proposal_authority": "ranking_only_no_gate_authority",
        "proposal_ranking_enabled": request.use_proposed_intent_ranking,
        "proposal_ranking": ranking_evidence,
    }
    plan = RepairPlan(
        source_sha256=diagnosis.source_sha256,
        profile=request.profile,
        operations=bounded_operations,
        intents=[*deterministic_intents, *proposed_intents],
        protected_features=request.protected_features,
        budgets=request.budgets,
        deterministic_seed=request.deterministic_seed,
        routing_policy_id=str(routing_evidence["policy_id"]),
        routing_evidence=routing_evidence,
        rationale=rationale,
        plan_path=str(plan_path),
    )
    atomic_write_json(plan_path, plan)
    return plan


def _copy_noop(source: Path, target: Path) -> WorkerResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    try:
        implementation_version = metadata.version("geometry-repair")
    except metadata.PackageNotFoundError:
        implementation_version = "source-tree"
    return WorkerResult(
        status="completed",
        output_path=str(target),
        output_sha256=file_sha256(target),
        changed=False,
        warnings=["No geometry change was made."],
        metadata={
            "execution_scope": "in_process",
            "implementation_version": implementation_version,
        },
    )


def _verified_worker_report(attempt: AttemptRecord) -> tuple[Path, WorkerResult]:
    """Load one attempt report only after its durable digest is rechecked."""

    if attempt.worker_report_path is None or attempt.worker_report_sha256 is None:
        raise RuntimeError(f"{attempt.attempt_id} has no digest-bound worker report")
    report_path = Path(attempt.worker_report_path).expanduser().resolve()
    if not report_path.is_file():
        raise RuntimeError(f"{attempt.attempt_id} worker report is missing: {report_path}")
    try:
        report_bytes = report_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{attempt.attempt_id} worker report cannot be read") from exc
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if report_sha256 != attempt.worker_report_sha256:
        raise RuntimeError(f"{attempt.attempt_id} worker report SHA-256 does not match its ledger")
    try:
        worker_result = WorkerResult.model_validate_json(report_bytes)
    except ValueError as exc:
        raise RuntimeError(
            f"{attempt.attempt_id} worker report is not a valid WorkerResult"
        ) from exc
    return report_path, worker_result


def _accepted_backend_provenance(
    accepted_attempt: AttemptRecord | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return backend provenance only from the accepted, digest-verified report."""

    if accepted_attempt is None:
        return None, None
    _, worker_result = _verified_worker_report(accepted_attempt)
    if worker_result.status != "completed":
        raise RuntimeError("accepted attempt does not have a completed worker report")
    if accepted_attempt.output_path is None or worker_result.output_path is None:
        raise RuntimeError("accepted attempt has no report-bound output")
    accepted_output = Path(accepted_attempt.output_path).expanduser().resolve()
    reported_output = Path(worker_result.output_path).expanduser().resolve()
    if reported_output != accepted_output or not accepted_output.is_file():
        raise RuntimeError("accepted attempt output differs from its worker report")
    try:
        accepted_output_sha256 = file_sha256(accepted_output)
    except OSError as exc:
        raise RuntimeError(
            "accepted attempt output cannot be read for SHA-256 verification"
        ) from exc
    if worker_result.output_sha256 != accepted_output_sha256:
        raise RuntimeError("accepted attempt output SHA-256 differs from its worker report")
    identity = worker_result.metadata.get("sdf_backend")
    qualification_id = worker_result.metadata.get("backend_qualification_id")
    if identity is None and qualification_id is None:
        if accepted_attempt.operation.worker == SdfRebuildWorker.name:
            raise RuntimeError("accepted SDF attempt has no qualified backend provenance")
        return None, None
    if (
        not isinstance(identity, dict)
        or not isinstance(qualification_id, str)
        or not qualification_id
    ):
        raise RuntimeError("accepted backend identity and qualification ID are incomplete")
    backend_id = identity.get("backend_id")
    if not isinstance(backend_id, str) or not backend_id:
        raise RuntimeError("accepted backend identity has no backend_id")
    requested_backend_id = accepted_attempt.operation.parameters.get("backend_id")
    if requested_backend_id is not None and backend_id != requested_backend_id:
        raise RuntimeError("accepted backend identity differs from the requested backend")
    return dict(identity), qualification_id


def _verified_operation_report_artifacts(
    attempts: list[AttemptRecord],
) -> list[dict[str, str]]:
    """Build evidence references from ledger digests after rechecking every report."""

    artifacts: list[dict[str, str]] = []
    for attempt in attempts:
        report_path, _ = _verified_worker_report(attempt)
        report_sha256 = attempt.worker_report_sha256
        if report_sha256 is None:  # pragma: no cover - guarded by the verifier
            raise RuntimeError(f"{attempt.attempt_id} has no worker report SHA-256")
        artifacts.append(
            {
                "kind": "operation_report",
                "path": str(report_path),
                "sha256": report_sha256,
            }
        )
    return artifacts


def _pin_scene_optimizer_dependency_roots(
    operation: RepairOperation,
    *,
    checkpoint_path: Path,
    source_path: Path,
    requested_dependency_roots: list[Path],
    localized_source_path: Path | None,
    resolved_snapshot_path: Path | None,
) -> RepairOperation:
    """Replace plan-provided roots with provenance-approved runtime roots."""
    if operation.worker != SceneOptimizerDeinstanceWorker.name:
        return operation

    requested_roots = [root.expanduser().resolve() for root in requested_dependency_roots]
    for requested_root in requested_roots:
        if not requested_root.is_dir():
            logger.warning(
                "Requested Scene Optimizer dependency root will not be forwarded "
                "because it is not an existing directory: %s",
                requested_root,
            )

    candidates = [
        checkpoint_path.resolve().parent,
        source_path.resolve().parent,
        *requested_roots,
    ]
    if localized_source_path is not None:
        candidates.append(localized_source_path.resolve().parent)
    if resolved_snapshot_path is not None:
        candidates.append(resolved_snapshot_path.resolve().parent)

    canonical_candidates = dict.fromkeys(candidates)
    roots = [
        str(candidate)
        for candidate in canonical_candidates
        if candidate.is_dir() and candidate.parent != candidate
    ]
    parameters = dict(operation.parameters)
    parameters["approved_dependency_roots"] = roots
    return operation.model_copy(update={"parameters": parameters})


def _run_worker_subprocess(
    *,
    operation: RepairOperation,
    source: Path,
    output: Path,
    attempt_dir: Path,
    memory_mb: int,
    timeout_s: float,
) -> WorkerResult:
    operation_path = attempt_dir / "operation_request.json"
    result_path = attempt_dir / "worker_result.json"
    result_path.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    atomic_write_json(operation_path, operation)
    command = [
        sys.executable,
        "-m",
        "geometry_repair.worker_runner",
        "--worker",
        operation.worker,
        "--source",
        str(source),
        "--output",
        str(output),
        "--operation",
        str(operation_path),
        "--result",
        str(result_path),
        "--memory-mb",
        str(memory_mb),
    ]
    worker_environment = bounded_process_environment(
        deterministic_seed=int(operation.parameters.get("deterministic_seed", 0))
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=worker_environment,
    )
    try:
        stdout, stderr = process.communicate(timeout=max(timeout_s, 0.001))
    except subprocess.TimeoutExpired:
        _kill_worker_process_group(process)
        result_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        return WorkerResult(
            status="failed",
            failures=[f"worker exceeded remaining wall-clock budget of {timeout_s:.3f}s"],
            metadata={
                "execution_scope": "isolated_subprocess",
                "memory_limit_mb": memory_mb,
            },
        )
    assert process.returncode is not None
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if result_path.is_file():
        result = WorkerResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    else:
        result = WorkerResult(
            status="failed",
            failures=[f"worker exited with code {completed.returncode} without a result artifact"],
        )
    if completed.returncode not in {0, 2}:
        stderr = completed.stderr.strip()
        result.failures.append(
            f"isolated worker process exited with code {completed.returncode}"
            + (f": {stderr[-1000:]}" if stderr else "")
        )
        result.status = "failed"
    return result


def _failed_collision_worker_report(
    *,
    render_path: Path,
    output_path: Path,
    report_path: Path,
    source_collision_audit_path: Path,
    source_render_sha256_before: str,
    runtime_engine: Literal["skip", "fake", "ovphysx"],
    failure: str,
) -> CollisionReport:
    """Write fail-closed collision evidence after an isolated-worker failure."""

    output_path.unlink(missing_ok=True)
    source_collision_audit_path.unlink(missing_ok=True)
    source_render_sha256_after = file_sha256(render_path)
    source_render_unchanged = source_render_sha256_after == source_render_sha256_before
    failures = [failure]
    if not source_render_unchanged:
        failures.append("collision planning altered the immutable render geometry source")
    report = CollisionReport(
        status="fail",
        representation="none",
        source_render_path=str(render_path),
        source_render_sha256=source_render_sha256_before,
        source_render_sha256_after=source_render_sha256_after,
        source_render_unchanged=source_render_unchanged,
        failures=failures,
        runtime_engine=runtime_engine,
        report_path=str(report_path),
    )
    atomic_write_json(report_path, report)
    return report


def _run_collision_geometry_subprocess(
    *,
    render_path: Path,
    output_path: Path,
    profile: RepairProfile,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature],
    deterministic_seed: int,
    coacd_enabled: bool,
    sdf_collision_rebuild_enabled: bool,
    timeout_s: float,
    runtime_engine: Literal["skip", "fake", "ovphysx"],
    report_path: Path,
    source_collision_audit_path: Path,
    attempt_dir: Path,
) -> CollisionReport:
    """Run complete collision planning inside the generic isolated worker."""

    render_path = render_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    source_collision_audit_path = source_collision_audit_path.expanduser().resolve()
    requested_attempt_dir = attempt_dir.expanduser()
    source_render_sha256_before = file_sha256(render_path)
    output_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    source_collision_audit_path.unlink(missing_ok=True)
    try:
        if requested_attempt_dir.is_symlink():
            raise ValueError("collision worker attempt directory must not be a symbolic link")
        requested_attempt_dir.mkdir(parents=True, exist_ok=True)
        if not requested_attempt_dir.is_dir():
            raise ValueError("collision worker attempt path must be a directory")
        attempt_dir = requested_attempt_dir.resolve()
        for artifact_name in ("operation_request.json", "worker_result.json"):
            artifact_path = attempt_dir / artifact_name
            if artifact_path.is_symlink():
                raise ValueError(
                    f"collision worker attempt artifact must not be a symbolic link: {artifact_name}"
                )
            if artifact_path.exists() and not artifact_path.is_file():
                raise ValueError(
                    f"collision worker attempt artifact must be a file: {artifact_name}"
                )
            artifact_path.unlink(missing_ok=True)
    except Exception as exc:
        return _failed_collision_worker_report(
            render_path=render_path,
            output_path=output_path,
            report_path=report_path,
            source_collision_audit_path=source_collision_audit_path,
            source_render_sha256_before=source_render_sha256_before,
            runtime_engine=runtime_engine,
            failure=f"collision geometry worker failed: {type(exc).__name__}: {exc}",
        )
    startup_allowance_s = min(5.0, timeout_s * 0.5)
    worker_timeout_s = max(timeout_s - startup_allowance_s, 0.001)
    operation = RepairOperation(
        operation_id="build-collision-geometry",
        worker=CollisionGeometryWorker.name,
        implementation="geometry_repair.workers.collision_geometry",
        parameters={
            "operation": "build_collision_geometry",
            "profile": profile,
            "budgets": budgets.model_dump(mode="json"),
            "protected_features": [
                feature.model_dump(mode="json") for feature in protected_features
            ],
            "deterministic_seed": deterministic_seed,
            "coacd_enabled": coacd_enabled,
            "sdf_collision_rebuild_enabled": sdf_collision_rebuild_enabled,
            "timeout_s": worker_timeout_s,
            "runtime_engine": runtime_engine,
            "report_path": str(report_path),
            "source_collision_audit_path": str(source_collision_audit_path),
        },
        drift_band="reconstructive",
        source_checkpoint=str(render_path),
        expected_changes=["author a separate collision geometry layer"],
        target_role="collision",
    )

    try:
        worker_result = _run_worker_subprocess(
            operation=operation,
            source=render_path,
            output=output_path,
            attempt_dir=attempt_dir,
            memory_mb=budgets.max_memory_mb,
            timeout_s=timeout_s,
        )
        if worker_result.status != "completed":
            details = "; ".join(worker_result.failures) or worker_result.status
            raise RuntimeError(f"isolated collision worker returned {details}")
        if worker_result.operations != ["build_collision_geometry"]:
            raise RuntimeError("isolated collision worker returned unexpected operations")
        if worker_result.metadata.get("execution_scope") != "isolated_subprocess":
            raise RuntimeError("isolated collision worker omitted its execution scope")
        if worker_result.metadata.get("memory_budget_mb") != budgets.max_memory_mb:
            raise RuntimeError("isolated collision worker reported an unexpected memory budget")
        if worker_result.metadata.get("memory_limit_mb") != budgets.max_memory_mb:
            raise RuntimeError("isolated collision worker reported an unexpected memory limit")
        uses_ovphysx_daemon = runtime_engine == "ovphysx" and profile in {
            "rigid_pick_place",
            "static_environment",
        }
        expected_limit_mode = (
            ADDRESS_SPACE_LIMIT_MODE_SOFT if uses_ovphysx_daemon else ADDRESS_SPACE_LIMIT_MODE_HARD
        )
        expected_limit_scope = (
            OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE
            if uses_ovphysx_daemon
            else ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE
        )
        expected_limit_exemption = (
            OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION if uses_ovphysx_daemon else None
        )
        if worker_result.metadata.get("address_space_limit_mode") != expected_limit_mode:
            raise RuntimeError(
                "isolated collision worker reported an unexpected address-space limit mode"
            )
        if worker_result.metadata.get("address_space_limit_scope") != expected_limit_scope:
            raise RuntimeError(
                "isolated collision worker reported an unexpected address-space limit scope"
            )
        if worker_result.metadata.get("memory_limit_exemption") != expected_limit_exemption:
            raise RuntimeError(
                "isolated collision worker reported an unexpected memory-limit exemption"
            )
        if not report_path.is_file():
            raise RuntimeError("isolated collision worker did not produce its collision report")
        if (
            Path(str(worker_result.metadata.get("collision_report_path", ""))).resolve()
            != report_path
        ):
            raise RuntimeError("isolated collision worker reported an invalid report path")
        report_sha256 = file_sha256(report_path)
        if worker_result.metadata.get("collision_report_sha256") != report_sha256:
            raise RuntimeError("isolated collision worker report digest does not match")

        report = CollisionReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if worker_result.metadata.get("collision_status") != report.status:
            raise RuntimeError("isolated collision worker status evidence does not match")
        if report.runtime_engine != runtime_engine:
            raise RuntimeError("collision report runtime engine does not match the request")
        if (
            report.source_render_path is None
            or Path(report.source_render_path).resolve() != render_path
        ):
            raise RuntimeError("collision report points at an unexpected render source")
        if report.source_render_sha256 != source_render_sha256_before:
            raise RuntimeError("collision report source digest does not match")
        source_render_sha256_after = file_sha256(render_path)
        if report.source_render_sha256_after != source_render_sha256_after:
            raise RuntimeError("collision report final source digest does not match")
        if (
            source_render_sha256_after != source_render_sha256_before
            or report.source_render_unchanged is not True
        ):
            raise RuntimeError("collision worker did not preserve the immutable render source")
        if report.report_path is not None and Path(report.report_path).resolve() != report_path:
            raise RuntimeError("collision report contains an invalid report path")
        if report.source_collision_audit_path is not None:
            claimed_audit = Path(report.source_collision_audit_path).resolve()
            if claimed_audit != source_collision_audit_path or not claimed_audit.is_file():
                raise RuntimeError("collision report contains invalid source-audit evidence")
            reported_audit_path = worker_result.metadata.get("source_collision_audit_path")
            if not isinstance(reported_audit_path, str):
                raise RuntimeError("collision worker omitted its source-audit path")
            if Path(reported_audit_path).resolve() != claimed_audit:
                raise RuntimeError("collision worker returned an unexpected source-audit path")
            audit_sha256 = file_sha256(claimed_audit)
            if worker_result.metadata.get("source_collision_audit_sha256") != audit_sha256:
                raise RuntimeError("collision worker source-audit digest does not match")
        else:
            if source_collision_audit_path.exists():
                raise RuntimeError("collision worker left an unreported source-audit artifact")
            if (
                "source_collision_audit_path" in worker_result.metadata
                or "source_collision_audit_sha256" in worker_result.metadata
            ):
                raise RuntimeError("collision worker returned unclaimed source-audit metadata")

        if report.collision_path is None:
            if worker_result.output_path is not None or worker_result.output_sha256 is not None:
                raise RuntimeError("collision worker returned an output absent from its report")
            if worker_result.changed or output_path.exists():
                raise RuntimeError("collision worker left an unreported collision artifact")
        else:
            collision_path = Path(report.collision_path).resolve()
            if collision_path != output_path or not output_path.is_file():
                raise RuntimeError("collision report contains an invalid collision artifact path")
            collision_sha256 = file_sha256(output_path)
            if report.collision_sha256 != collision_sha256:
                raise RuntimeError("collision report digest does not match its artifact")
            if worker_result.output_path is None:
                raise RuntimeError("collision worker omitted its output path")
            if Path(worker_result.output_path).resolve() != output_path:
                raise RuntimeError("collision worker returned an unexpected output path")
            if worker_result.output_sha256 != collision_sha256 or not worker_result.changed:
                raise RuntimeError("collision worker output evidence does not match")
        return report
    except Exception as exc:
        return _failed_collision_worker_report(
            render_path=render_path,
            output_path=output_path,
            report_path=report_path,
            source_collision_audit_path=source_collision_audit_path,
            source_render_sha256_before=source_render_sha256_before,
            runtime_engine=runtime_engine,
            failure=f"collision geometry worker failed: {type(exc).__name__}: {exc}",
        )


def _kill_worker_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the isolated worker and every child process in its process group."""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.communicate()


def _brep_fidelity(
    source: Path,
    candidate: Path,
    *,
    operation: RepairOperation,
    budgets: RepairBudgets,
    protected_features: list[ProtectedFeature],
    output_path: Path,
) -> FidelityReport:
    source_diagnosis = diagnose_asset(
        source,
        profile="visual_only",
        run_scalable_audit=False,
    )
    candidate_diagnosis = diagnose_asset(
        candidate,
        profile="visual_only",
        run_scalable_audit=False,
    )
    before = source_diagnosis.metrics.brep
    after = candidate_diagnosis.metrics.brep
    failures: list[str] = []
    warnings: list[str] = []
    volume_drift = None
    area_drift = None
    if before.volume_source_units3 not in {None, 0.0} and after.volume_source_units3 is not None:
        volume_drift = abs(after.volume_source_units3 - before.volume_source_units3) / abs(
            before.volume_source_units3
        )
        if volume_drift > budgets.conservative_volume_drift:
            failures.append(
                f"native B-rep volume drift exceeds {budgets.conservative_volume_drift:.4f}"
            )
    if (
        before.surface_area_source_units2 not in {None, 0.0}
        and after.surface_area_source_units2 is not None
    ):
        area_drift = abs(
            after.surface_area_source_units2 - before.surface_area_source_units2
        ) / abs(before.surface_area_source_units2)
        if area_drift > budgets.conservative_volume_drift:
            failures.append(
                f"native B-rep surface-area drift exceeds {budgets.conservative_volume_drift:.4f}"
            )
    if before.solid_count != after.solid_count:
        failures.append("native B-rep solid count changed")
    diagonal = 0.0
    if before.bbox_min_source_units and before.bbox_max_source_units:
        diagonal = max(
            sum(
                (float(maximum) - float(minimum)) ** 2
                for minimum, maximum in zip(
                    before.bbox_min_source_units,
                    before.bbox_max_source_units,
                    strict=True,
                )
            )
            ** 0.5,
            1e-12,
        )
    bbox_drift_ratio = None
    if (
        diagonal
        and before.bbox_min_source_units
        and before.bbox_max_source_units
        and after.bbox_min_source_units
        and after.bbox_max_source_units
    ):
        bbox_drift_ratio = (
            max(
                abs(float(candidate_value) - float(source_value))
                for source_values, candidate_values in (
                    (before.bbox_min_source_units, after.bbox_min_source_units),
                    (before.bbox_max_source_units, after.bbox_max_source_units),
                )
                for source_value, candidate_value in zip(
                    source_values,
                    candidate_values,
                    strict=True,
                )
            )
            / diagonal
        )
        if bbox_drift_ratio > budgets.conservative_p99_ratio:
            failures.append(
                "native B-rep bbox drift exceeds "
                f"{budgets.conservative_p99_ratio:.4f} of source diagonal"
            )
    centroid_drift_ratio = None
    if diagonal and before.center_of_mass_source_units and after.center_of_mass_source_units:
        centroid_drift_ratio = (
            sum(
                (float(candidate) - float(source_value)) ** 2
                for source_value, candidate in zip(
                    before.center_of_mass_source_units,
                    after.center_of_mass_source_units,
                    strict=True,
                )
            )
            ** 0.5
            / diagonal
        )
        if centroid_drift_ratio > budgets.conservative_p99_ratio:
            failures.append(
                "native B-rep centroid drift exceeds "
                f"{budgets.conservative_p99_ratio:.4f} of source diagonal"
            )
    tolerance_limit = max(
        float(operation.parameters.get("maximum_tolerance", 0.0)),
        before.maximum_tolerance_source_units or 0.0,
    )
    if (
        after.maximum_tolerance_source_units is not None
        and after.maximum_tolerance_source_units > tolerance_limit * (1.0 + 1e-6)
    ):
        failures.append("native B-rep maximum tolerance grew beyond the declared limit")
    unmeasured_features = [feature.name for feature in protected_features if feature.required]
    if unmeasured_features:
        warnings.append(
            "native B-rep protected features require post-tessellation measurement: "
            + ", ".join(sorted(unmeasured_features))
        )
    identity_deltas = {
        name: getattr(after, name) - getattr(before, name)
        for name in ("solid_count", "shell_count", "face_count", "wire_count", "edge_count")
    }
    report = FidelityReport(
        source_path=str(source),
        candidate_path=str(candidate),
        drift_band=operation.drift_band,
        status="fail" if failures else "conditional" if warnings else "pass",
        bbox_max_drift_ratio=bbox_drift_ratio,
        centroid_drift_ratio=centroid_drift_ratio,
        volume_drift_ratio=volume_drift,
        surface_area_drift_ratio=area_drift,
        part_count_delta=after.solid_count - before.solid_count,
        brep_identity_deltas=identity_deltas,
        protected_features_unmeasured=unmeasured_features,
        failures=failures,
        warnings=warnings,
        report_path=str(output_path),
    )
    atomic_write_json(output_path, report)
    return report


def _report_markdown(
    request: RepairRequest,
    diagnosis: Diagnosis,
    certificate: RepairCertificate,
    attempts: list[AttemptRecord],
) -> str:
    lines = [
        "# Geometry Repair Report",
        "",
        f"- Profile: `{request.profile}`",
        f"- Mode: `{request.mode}`",
        f"- Outcome: `{certificate.outcome}`",
        f"- Claim scope: `{certificate.claim_scope}`",
        f"- Source SHA-256: `{certificate.source_sha256}`",
        f"- Initial diagnosis: `{diagnosis.status}`",
        "",
        "## Attempts",
        "",
    ]
    for attempt in attempts:
        lines.append(
            f"- `{attempt.attempt_id}`: `{attempt.status}` via `{attempt.operation.worker}`"
        )
        lines.extend(f"  - {reason}" for reason in attempt.reasons)
    lines.extend(["", "## Remaining Warnings", ""])
    lines.extend(f"- {warning}" for warning in certificate.remaining_warnings)
    if not certificate.remaining_warnings:
        lines.append("- None.")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {blocker}" for blocker in certificate.blockers)
    if not certificate.blockers:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This certificate covers geometry repair only. Materials, physical properties, joints, controls, and final SimReady/runtime acceptance remain downstream-owned.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_final_usd_package(
    render_path: Path,
    collision_path: Path | None,
    composed_path: Path,
    output_path: Path,
) -> dict:
    """Fail closed when final USD layers cannot be opened or resolve dependencies."""

    from pxr import Usd, UsdPhysics, UsdUtils

    def active_collision_paths(stage: Any, prefix: str | None = None) -> list[str]:
        paths: list[str] = []
        for prim in stage.Traverse():
            path_text = str(prim.GetPath())
            if prefix is not None and not (
                path_text == prefix or path_text.startswith(f"{prefix}/")
            ):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                paths.append(path_text)
        return sorted(paths)

    failures: list[str] = []
    warnings: list[str] = []
    layers = [render_path, composed_path, *([collision_path] if collision_path else [])]
    records = []
    for layer in layers:
        stage = Usd.Stage.Open(str(layer))
        if stage is None:
            failures.append(f"OpenUSD could not open final layer {layer}")
            records.append({"path": str(layer), "status": "fail"})
            continue
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(layer))
        unresolved_paths = sorted(str(item) for item in unresolved)
        unresolved_geometry, unresolved_material = classify_unresolved_dependencies(
            unresolved_paths
        )
        if unresolved_geometry:
            failures.append(
                f"final layer {layer} has unresolved geometry dependencies: "
                + ", ".join(unresolved_geometry)
            )
        if unresolved_material:
            warnings.append(
                f"final layer {layer} has unresolved material dependencies: "
                + ", ".join(unresolved_material)
            )
        if layer in {render_path, composed_path} and not stage.GetDefaultPrim():
            failures.append(f"final layer {layer} has no default prim")
        layer_status = (
            "fail" if unresolved_geometry else "warning" if unresolved_material else "pass"
        )
        records.append(
            {
                "path": str(layer),
                "status": layer_status,
                "default_prim": str(stage.GetDefaultPrim().GetPath())
                if stage.GetDefaultPrim()
                else None,
                "unresolved_dependencies": unresolved_paths,
                "unresolved_geometry_dependencies": unresolved_geometry,
                "unresolved_material_dependencies": unresolved_material,
            }
        )
    composition: dict[str, Any] = {"status": "not_evaluated"}
    composition_failures: list[str] = []

    def fail_composition(message: str) -> None:
        composition_failures.append(message)
        failures.append(message)

    render_stage = Usd.Stage.Open(str(render_path))
    composed_stage = Usd.Stage.Open(str(composed_path))
    if render_stage is not None and composed_stage is not None:
        source_default = render_stage.GetDefaultPrim()
        composed_render = composed_stage.GetPrimAtPath("/GeometryRepairAsset/Render")
        source_active = active_collision_paths(render_stage)
        composed_render_active = active_collision_paths(
            composed_stage,
            "/GeometryRepairAsset/Render",
        )
        composition = {
            "status": "pass",
            "source_default_prim_type": source_default.GetTypeName() if source_default else None,
            "composed_render_prim_type": (
                composed_render.GetTypeName() if composed_render else None
            ),
            "source_active_collision_count": len(source_active),
            "composed_render_active_collision_count": len(composed_render_active),
            "replacement_active_collision_count": 0,
            "composed_collision_active_count": 0,
        }
        if not source_default or not composed_render:
            fail_composition("composed asset does not expose the referenced render default prim")
        elif source_default.GetTypeName() != composed_render.GetTypeName():
            fail_composition(
                "composed render root schema differs from the referenced source default prim"
            )
        if collision_path is None:
            if len(source_active) != len(composed_render_active):
                fail_composition(
                    "composed asset did not preserve active source collision when no "
                    "replacement collision layer was supplied"
                )
            if composed_stage.GetPrimAtPath("/GeometryRepairAsset/Collision"):
                fail_composition(
                    "composed asset authored a collision representation without a collision layer"
                )
        else:
            collision_stage = Usd.Stage.Open(str(collision_path))
            composed_collision = composed_stage.GetPrimAtPath("/GeometryRepairAsset/Collision")
            replacement_active = (
                active_collision_paths(collision_stage) if collision_stage is not None else []
            )
            composed_collision_active = active_collision_paths(
                composed_stage,
                "/GeometryRepairAsset/Collision",
            )
            composition.update(
                {
                    "replacement_active_collision_count": len(replacement_active),
                    "composed_collision_active_count": len(composed_collision_active),
                }
            )
            if collision_stage is None or not collision_stage.GetDefaultPrim():
                fail_composition("replacement collision layer has no readable default prim")
            elif not composed_collision:
                fail_composition("composed asset does not expose the replacement collision root")
            elif collision_stage.GetDefaultPrim().GetTypeName() != composed_collision.GetTypeName():
                fail_composition(
                    "composed collision root schema differs from the replacement default prim"
                )
            if composed_render_active:
                fail_composition(
                    "composed render representation retains active source collision alongside "
                    "a replacement"
                )
            if not replacement_active or len(replacement_active) != len(composed_collision_active):
                fail_composition(
                    "composed asset did not preserve every active replacement collision prim"
                )
        if composition_failures:
            composition["status"] = "fail"
            composition["failures"] = composition_failures

    payload = {
        "schema_version": "geometry-repair.usd-package-validation.v1",
        "status": "fail" if failures else "warning" if warnings else "pass",
        "layers": records,
        "composition": composition,
        "failures": failures,
        "warnings": warnings,
    }
    atomic_write_json(output_path, payload)
    return payload


def _write_correspondence(
    source_path: Path,
    candidate_path: Path,
    operation: RepairOperation,
    worker_result: WorkerResult,
    source_diagnosis: Diagnosis,
    candidate_diagnosis: Diagnosis,
    output_path: Path,
) -> tuple[Path, CorrespondenceEvidence]:
    """Write typed exact correspondence or an explicit refusal artifact."""

    if not worker_result.changed and file_sha256(source_path) == file_sha256(candidate_path):
        attributes: dict[str, tuple[str, str | None]] = {}
        if source_diagnosis.metrics.mesh.authored_uv_count:
            attributes["source_uv_primvars"] = ("corner", "faceVarying")
        if source_diagnosis.metrics.mesh.authored_normal_count:
            attributes["source_normals"] = ("corner", "faceVarying")
        evidence = identity_correspondence(
            source_path,
            candidate_path,
            vertex_count=source_diagnosis.metrics.mesh.point_count,
            face_count=source_diagnosis.metrics.mesh.triangle_count,
            part_ids=source_diagnosis.metrics.source_part_paths,
            prim_paths=source_diagnosis.metrics.source_part_paths,
            attributes=attributes,
        )
    else:
        evidence = build_mesh_correspondence(
            source_path,
            candidate_path,
            operation=operation.worker,
            operation_version=str(worker_result.metadata.get("implementation_version") or "")
            or None,
            parameters=operation.parameters,
            deterministic_seed=None,
        )
    path = write_correspondence_evidence(output_path, evidence)
    return path, evidence


_MATERIAL_ONLY_USD_CHECKERS = {
    "MaterialBindingAPIAppliedChecker",
    "ShaderPropertyTypeConformanceChecker",
}


def _geometry_scoped_source_format_status(
    report: SourceFormatValidationReport,
    *,
    unresolved_material_dependencies: list[str],
) -> tuple[str, list[str], list[str]]:
    """Separate raw format conformance from geometry-owned blockers."""

    if (
        report.status == "not_evaluated"
        and report.metadata.get("excluded_material_rule_names")
        and not report.errors
    ):
        return "pass_with_material_warnings", [], []
    if report.status != "fail":
        return report.status, [], []
    material_errors: list[str] = []
    geometry_errors: list[str] = []
    for error in report.errors:
        material_checker = any(
            f"(fails '{checker}')" in error for checker in _MATERIAL_ONLY_USD_CHECKERS
        )
        exact_material_path = any(
            dependency and dependency in error for dependency in unresolved_material_dependencies
        )
        (material_errors if material_checker or exact_material_path else geometry_errors).append(
            error
        )
    if geometry_errors or not material_errors:
        return "fail", geometry_errors or list(report.errors), material_errors
    return "pass_with_material_warnings", [], material_errors


def run_geometry_repair(request: RepairRequest) -> RepairResult:
    """Execute a bounded deterministic repair job and emit an honest certificate."""

    started = time.monotonic()
    source = request.source_path.expanduser().resolve()
    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_format_validation_path = output_dir / "source_format_validation.json"
    source_format_validation = validate_source_format(
        source,
        source_format_validation_path,
        timeout_s=min(request.budgets.timeout_s, 120.0),
    )
    usd_intake_path: Path | None = None
    usd_intake: UsdIntakeReport | None = None
    usd_semantic_facts: dict[str, list[str]] | None = None
    dependency_localization_path: Path | None = None
    dependency_localization: DependencyLocalizationReport | None = None
    localized_source_path: Path | None = None
    dependency_paths: list[str] | None = None
    unresolved_dependencies: list[str] | None = None
    if source.suffix.lower() in USD_SUFFIXES:
        usd_intake = inventory_usd_stage(
            source,
            allowed_dependency_roots=request.dependency_roots,
            dependency_remap_manifest=request.dependency_remap_manifest,
            max_dependencies=request.budgets.max_dependency_count,
            max_hash_bytes=request.budgets.max_dependency_hash_bytes,
        )
        usd_intake_path = output_dir / "usd_intake.json"
        atomic_write_json(usd_intake_path, usd_intake)
        usd_semantic_facts = {
            "joint_prim_paths": usd_intake.joint_prim_paths,
            "rigid_body_prim_paths": usd_intake.rigid_body_prim_paths,
            "time_varying_prim_paths": usd_intake.time_varying_prim_paths,
            "geometry_time_varying_prim_paths": (usd_intake.geometry_time_varying_prim_paths),
            "skeleton_prim_paths": usd_intake.skeleton_prim_paths,
            "animation_prim_paths": usd_intake.animation_prim_paths,
            "brep_prim_paths": usd_intake.brep_prim_paths,
        }
        if source.suffix.lower() != ".usdz":
            approved_roots = list(
                dict.fromkeys(
                    [
                        source.parent,
                        *(path.expanduser().resolve() for path in request.dependency_roots),
                    ]
                )
            )
            dependency_localization = localize_usd_dependencies(
                source,
                output_dir / "dependency_bundle",
                approved_roots=approved_roots,
                remap_manifest=request.dependency_remap_manifest,
                max_dependencies=request.budgets.max_dependency_count,
                max_hash_bytes=request.budgets.max_dependency_hash_bytes,
            )
            dependency_localization_path = Path(dependency_localization.evidence_path)
            dependency_paths = sorted(
                {mapping.source_path for mapping in dependency_localization.mappings}
            )
            unresolved_dependencies = sorted(
                {item.authored_path for item in dependency_localization.unresolved}
            )
            if dependency_localization.portable_package_complete:
                localized_source_path = Path(dependency_localization.localized_source_path)
        else:
            dependency_paths = sorted(
                {
                    record.local_path
                    for record in usd_intake.dependencies
                    if record.status in {"resolved_local", "resolved_package"}
                    and record.local_path is not None
                }
            )
            unresolved_dependencies = usd_intake.unresolved_dependencies
    source_package = preserve_source(
        source,
        output_dir,
        source_uri=request.source_uri,
        source_license=request.source_license,
        source_provenance=request.source_provenance,
        dependency_paths=dependency_paths,
        unresolved_dependencies=unresolved_dependencies,
        create_resolved_snapshot=usd_intake.composition_complete if usd_intake else True,
    )
    if source_package.resolved_snapshot_path:
        normalized_source = _normalized_copy(
            Path(source_package.resolved_snapshot_path),
            output_dir,
            source_meters_per_unit=request.source_meters_per_unit,
            source_up_axis=request.source_up_axis,
        )
    elif localized_source_path is not None:
        normalized_dir = output_dir / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_source = flatten_usd_stage(
            localized_source_path,
            normalized_dir / "source.usd",
        )
    else:
        normalized_source = _normalized_copy(
            Path(source_package.source.preserved_path),
            output_dir,
            source_meters_per_unit=request.source_meters_per_unit,
            source_up_axis=request.source_up_axis,
        )
    enabled_workers = _enabled_workers(request)
    protected_feature_report: ProtectedFeatureCandidateReport | None = None
    protected_feature_candidates_path: Path | None = None
    if normalized_source.suffix.lower() not in BREP_SUFFIXES:
        try:
            protected_feature_report = detect_protected_feature_candidates(
                normalized_source,
                candidate_limit=request.budgets.max_protected_feature_candidates,
            )
            protected_feature_candidates_path = output_dir / "protected_feature_candidates.json"
            atomic_write_json(protected_feature_candidates_path, protected_feature_report)
        except (OSError, RuntimeError, ValueError):
            protected_feature_report = None
    manifold_seam_analysis_path: Path | None = None
    if (
        ManifoldSeamWorker.name in enabled_workers
        and normalized_source.suffix.lower() not in BREP_SUFFIXES
    ):
        analysis_dir = output_dir / "analysis" / "manifold_seams"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        requested_report_path = analysis_dir / "manifold_seam_analysis.json"
        analysis_operation = RepairOperation(
            operation_id="analysis-manifold-restore-merge-vectors",
            worker=ManifoldSeamWorker.name,
            implementation="manifold3d.Mesh64.merge",
            parameters={"operation": ManifoldSeamWorker.name},
            issue_ids=[],
            drift_band="identity",
            source_checkpoint=str(normalized_source),
            expected_changes=["emit exact merge-vector evidence without mutating geometry"],
        )
        seam_result = _run_worker_subprocess(
            operation=analysis_operation,
            source=normalized_source,
            output=requested_report_path,
            attempt_dir=analysis_dir,
            memory_mb=request.budgets.max_memory_mb,
            timeout_s=min(request.budgets.timeout_s, 120.0),
        )
        reported_path = seam_result.metadata.get("merge_vector_report_path")
        candidate_report_path = Path(reported_path) if isinstance(reported_path, str) else None
        if candidate_report_path is not None and candidate_report_path.is_file():
            manifold_seam_analysis_path = candidate_report_path.resolve()
    scalable_audit_budget = ScalableAuditBudget(
        broad_pair_limit=request.budgets.audit_broad_pair_limit,
        exact_test_limit=request.budgets.audit_exact_test_limit,
        wall_time_s=min(request.budgets.audit_wall_time_s, request.budgets.timeout_s),
        memory_limit_bytes=request.budgets.max_memory_mb * 1024 * 1024,
        chunk_size=request.budgets.audit_chunk_size,
    )
    diagnosis_path = output_dir / "diagnosis.json"
    diagnosis = diagnose_asset(
        normalized_source,
        profile=request.profile,
        output_path=diagnosis_path,
        unresolved_dependency_count=len(source_package.unresolved_dependencies),
        unresolved_geometry_dependency_count=len(source_package.unresolved_geometry_dependencies),
        unresolved_material_dependency_count=len(source_package.unresolved_material_dependencies),
        scalable_audit_budget=scalable_audit_budget,
        usd_semantic_facts=usd_semantic_facts,
    )
    plan_path = output_dir / "repair_plan.json"
    plan = _plan_repair(request, diagnosis, normalized_source, plan_path)
    attempts: list[AttemptRecord] = []
    accepted_path: Path | None = None
    accepted_attempt: AttemptRecord | None = None
    accepted_diagnosis: Diagnosis | None = None
    accepted_fidelity: FidelityReport | None = None
    checkpoint_path = normalized_source
    checkpoint_blockers = set(diagnosis.blocking_issue_ids)
    checkpoint_instance_proxy_count = diagnosis.metrics.instance_proxy_mesh_count
    cumulative_expected_issue_ids: set[str] = set()
    applied_operations: list[str] = []
    deleted_entities: list[str] = []
    merged_entities: list[str] = []
    attempts_dir = output_dir / "attempts"
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    budget_exhausted = False
    unavailable_workers: dict[str, list[str]] = {}

    for index, operation in enumerate(plan.operations):
        if time.monotonic() - started > request.budgets.timeout_s:
            budget_exhausted = True
            break
        attempt_id = f"attempt-{index:02d}"
        attempt_dir = attempts_dir / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        suffix = normalized_source.suffix.lower()
        candidate_path = attempt_dir / f"geometry{suffix}"
        runtime_operation = operation.model_copy(update={"source_checkpoint": str(checkpoint_path)})
        runtime_operation = _pin_scene_optimizer_dependency_roots(
            runtime_operation,
            checkpoint_path=checkpoint_path,
            source_path=source,
            requested_dependency_roots=request.dependency_roots,
            localized_source_path=localized_source_path,
            resolved_snapshot_path=(
                Path(source_package.resolved_snapshot_path)
                if source_package.resolved_snapshot_path
                else None
            ),
        )
        if runtime_operation.worker in unavailable_workers:
            worker_result = WorkerResult(
                status="unavailable",
                failures=[
                    f"skipped parameter variant because {runtime_operation.worker} already "
                    "reported an invariant capability refusal",
                    *unavailable_workers[runtime_operation.worker],
                ],
                metadata={"capability_memoized": True},
            )
        elif runtime_operation.worker == "noop":
            worker_result = _copy_noop(normalized_source, candidate_path)
        else:
            remaining = request.budgets.timeout_s - (time.monotonic() - started)
            worker_result = _run_worker_subprocess(
                operation=runtime_operation,
                source=checkpoint_path,
                output=candidate_path,
                attempt_dir=attempt_dir,
                memory_mb=request.budgets.max_memory_mb,
                timeout_s=remaining,
            )
        worker_report_path = attempt_dir / "operation.json"
        atomic_write_json(worker_report_path, worker_result)
        worker_report_sha256 = file_sha256(worker_report_path)
        if worker_result.status != "completed" or not worker_result.output_path:
            if worker_result.status == "unavailable":
                unavailable_workers.setdefault(
                    runtime_operation.worker,
                    [*worker_result.failures, *worker_result.warnings],
                )
            attempts.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    operation=runtime_operation,
                    status="unavailable" if worker_result.status == "unavailable" else "failed",
                    worker_report_path=str(worker_report_path),
                    worker_report_sha256=worker_report_sha256,
                    reasons=[*worker_result.failures, *worker_result.warnings],
                )
            )
            continue
        candidate = Path(worker_result.output_path).resolve()
        allowed_outputs = {
            normalized_source.resolve(),
            checkpoint_path.resolve(),
            candidate_path.resolve(),
        }
        if candidate not in allowed_outputs or not candidate.is_file():
            attempts.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    operation=runtime_operation,
                    status="failed",
                    worker_report_path=str(worker_report_path),
                    worker_report_sha256=worker_report_sha256,
                    reasons=["worker returned an unapproved or missing output path"],
                )
            )
            continue
        candidate_sha256 = file_sha256(candidate)
        if worker_result.output_sha256 != candidate_sha256:
            attempts.append(
                AttemptRecord(
                    attempt_id=attempt_id,
                    operation=runtime_operation,
                    status="failed",
                    worker_report_path=str(worker_report_path),
                    worker_report_sha256=worker_report_sha256,
                    reasons=["worker output digest is missing or does not match the candidate"],
                )
            )
            continue
        candidate_diagnosis_path = attempt_dir / "diagnosis.json"
        candidate_diagnosis = diagnose_asset(
            candidate,
            profile=request.profile,
            output_path=candidate_diagnosis_path,
            scalable_audit_budget=scalable_audit_budget,
            usd_semantic_facts=usd_semantic_facts,
        )
        verified_issue_ids = _admitted_worker_verified_issue_ids(
            worker_result.verified_issue_ids,
            candidate_diagnosis.blocking_issue_ids,
        )
        if verified_issue_ids:
            remaining_blockers = sorted(
                set(candidate_diagnosis.blocking_issue_ids) - set(verified_issue_ids)
            )
            remaining_warnings = [
                issue
                for issue in candidate_diagnosis.issues
                if issue.severity == "warning" and issue.issue_id not in verified_issue_ids
            ]
            candidate_diagnosis = candidate_diagnosis.model_copy(
                update={
                    "blocking_issue_ids": remaining_blockers,
                    "verified_issue_ids": verified_issue_ids,
                    "role_validation": build_diagnosis_role_validation(
                        profile=request.profile,
                        metrics=candidate_diagnosis.metrics,
                        issues=candidate_diagnosis.issues,
                        diagnosis_path=str(candidate_diagnosis_path),
                        blocking_issue_ids=remaining_blockers,
                        verified_issue_ids=verified_issue_ids,
                    ),
                    "status": (
                        "fail"
                        if remaining_blockers
                        else "conditional"
                        if remaining_warnings
                        else "pass"
                    ),
                }
            )
            atomic_write_json(candidate_diagnosis_path, candidate_diagnosis)
        atomic_write_json(attempt_dir / "geometry_metrics.json", candidate_diagnosis.metrics)
        fidelity_path = attempt_dir / "fidelity.json"
        if candidate.suffix.lower() in BREP_SUFFIXES:
            fidelity = _brep_fidelity(
                normalized_source,
                candidate,
                operation=runtime_operation,
                budgets=request.budgets,
                protected_features=request.protected_features,
                output_path=fidelity_path,
            )
        else:
            fidelity = compare_geometry(
                normalized_source,
                candidate,
                drift_band=runtime_operation.drift_band,
                budgets=request.budgets,
                protected_features=request.protected_features,
                expected_issue_ids=sorted(
                    cumulative_expected_issue_ids | set(runtime_operation.issue_ids)
                ),
                generated_patch_area_ratio_limit=(
                    float(runtime_operation.parameters["max_patch_area_ratio"])
                    if runtime_operation.parameters.get("operation") == "pmp_fill_classified_hole"
                    and "max_patch_area_ratio" in runtime_operation.parameters
                    else None
                ),
                output_path=fidelity_path,
            )
        correspondence_path: Path | None = None
        correspondence: CorrespondenceEvidence | None = None
        correspondence_failures: list[str] = []
        correspondence_warnings: list[str] = []
        if candidate.suffix.lower() not in BREP_SUFFIXES:
            try:
                correspondence_path, correspondence = _write_correspondence(
                    normalized_source,
                    candidate,
                    runtime_operation,
                    worker_result,
                    diagnosis,
                    candidate_diagnosis,
                    attempt_dir / "correspondence.json",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                correspondence_failures.append(
                    f"typed correspondence could not be established: {type(exc).__name__}: {exc}"
                )
        if correspondence is not None and correspondence.status == "refused":
            correspondence_failures.extend(correspondence.refusal_reasons)
        if correspondence is not None:
            correspondence_warnings.extend(correspondence.warnings)
        protected_comparison_path: Path | None = None
        protected_comparison_failures: list[str] = []
        protected_comparison_warnings: list[str] = []
        if worker_result.changed and protected_feature_report is not None:
            authorized_hole_loops = {
                (intent.target_mesh_path, frozenset(intent.boundary_loop_vertex_ids))
                for intent in request.classified_holes
                if runtime_operation.parameters.get("intent_evidence_id")
                == intent.intent_evidence_id
            }
            comparison_candidates = [
                candidate
                for candidate in protected_feature_report.candidates
                if (
                    candidate.scope_path,
                    frozenset(candidate.evidence.get("loop_vertex_ids", [])),
                )
                not in authorized_hole_loops
            ]
            protected_comparison = compare_protected_feature_candidates(
                normalized_source,
                candidate,
                comparison_candidates,
            )
            protected_comparison_path = attempt_dir / "protected_feature_comparison.json"
            atomic_write_json(protected_comparison_path, protected_comparison)
            protected_comparison_failures.extend(protected_comparison.failures)
            protected_comparison_warnings.extend(protected_comparison.warnings)
        reasons = [
            *candidate_diagnosis.blocking_issue_ids,
            *fidelity.failures,
            *fidelity.warnings,
            *correspondence_failures,
            *correspondence_warnings,
            *protected_comparison_failures,
            *protected_comparison_warnings,
            *worker_result.warnings,
        ]
        candidate_blockers = set(candidate_diagnosis.blocking_issue_ids)
        accepted = (
            not candidate_blockers
            and fidelity.status != "fail"
            and not correspondence_failures
            and not protected_comparison_failures
        )
        deinstance_prerequisite_advanced = (
            runtime_operation.worker == SceneOptimizerDeinstanceWorker.name
            and checkpoint_instance_proxy_count > 0
            and candidate_diagnosis.metrics.instance_proxy_mesh_count == 0
        )
        advanced = (
            not accepted
            and fidelity.status != "fail"
            and not correspondence_failures
            and not protected_comparison_failures
            and (
                candidate_blockers < checkpoint_blockers
                or (deinstance_prerequisite_advanced and candidate_blockers <= checkpoint_blockers)
            )
        )
        attempt = AttemptRecord(
            attempt_id=attempt_id,
            operation=runtime_operation,
            status="accepted" if accepted else "advanced" if advanced else "rejected",
            output_path=str(candidate),
            output_sha256=candidate_sha256,
            diagnosis_path=str(candidate_diagnosis_path),
            fidelity_path=str(fidelity_path),
            worker_report_path=str(worker_report_path),
            worker_report_sha256=worker_report_sha256,
            correspondence_path=str(correspondence_path) if correspondence_path else None,
            protected_feature_comparison_path=(
                str(protected_comparison_path) if protected_comparison_path else None
            ),
            reasons=reasons,
        )
        attempts.append(attempt)
        if advanced or accepted:
            checkpoint_path = candidate
            checkpoint_blockers = candidate_blockers
            checkpoint_instance_proxy_count = candidate_diagnosis.metrics.instance_proxy_mesh_count
            cumulative_expected_issue_ids.update(runtime_operation.issue_ids)
            applied_operations.extend(worker_result.operations)
            deleted_entities.extend(worker_result.deleted_entities)
            merged_entities.extend(worker_result.merged_entities)
        if advanced:
            continue
        if accepted:
            accepted_path = candidate
            accepted_attempt = attempt
            accepted_diagnosis = candidate_diagnosis
            accepted_fidelity = fidelity
            break

    attempts_path = attempts_dir / "attempts.json"
    atomic_write_json(
        attempts_path,
        {
            "schema_version": "geometry-repair.attempt-ledger.v1",
            "attempts": [attempt.model_dump(mode="json") for attempt in attempts],
        },
    )
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    render_path: Path | None = None
    collision_path: Path | None = None
    composed_path: Path | None = None
    collision_report_path = final_dir / "collision_report.json"
    collision_report = None
    source_collision_audit_path: Path | None = None
    advanced_profile_report: AdvancedProfileEvidenceReport | None = None
    advanced_profile_path: Path | None = None
    usd_package_validation = None
    correspondence_path: Path | None = None
    separated_source_roles: dict[str, list[str]] | None = None
    blockers: list[str] = []
    warnings: list[str] = list(source_format_validation.warnings)
    (
        source_format_geometry_status,
        source_format_geometry_errors,
        source_format_material_errors,
    ) = _geometry_scoped_source_format_status(
        source_format_validation,
        unresolved_material_dependencies=source_package.unresolved_material_dependencies,
    )
    blockers.extend(source_format_geometry_errors)
    warnings.extend(source_format_material_errors)
    if source_format_validation.status == "not_evaluated":
        warnings.extend(source_format_validation.errors)
    render_geometry_changed = False
    source_rights_status = (
        "not_required"
        if not request.production_use
        else "pass"
        if request.source_uri
        and request.source_uri.strip()
        and request.source_license
        and request.source_license.strip()
        and request.source_provenance
        else "missing"
    )
    if not request.profile_confirmed:
        warnings.append(
            "The repair profile was inferred and must be confirmed before certification."
        )
    if source_rights_status == "missing":
        warnings.append(
            "Production certification requires source URI, license, and provenance evidence."
        )
    if accepted_path is None:
        outcome = "rejected"
        blockers.extend(sorted(checkpoint_blockers))
        if budget_exhausted:
            blockers.append(
                f"repair job exceeded wall-clock budget of {request.budgets.timeout_s:.3f}s"
            )
        blockers.extend(
            f"{attempt.attempt_id}: {reason}"
            for attempt in attempts
            if attempt.status in {"failed", "unavailable"}
            for reason in attempt.reasons
        )
        attempted_issue_ids = {
            issue_id for operation in plan.operations for issue_id in operation.issue_ids
        }
        blockers.extend(
            f"no approved automatic worker targeted {issue_id}"
            for issue_id in diagnosis.blocking_issue_ids
            if issue_id not in attempted_issue_ids
        )
    elif accepted_path.suffix.lower() in USD_SUFFIXES:
        render_path = flatten_usd_stage(accepted_path, final_dir / "render.usd")
        separated_source_roles = separate_non_render_geometry(render_path)
        if accepted_attempt is not None and accepted_attempt.correspondence_path:
            correspondence_source = Path(accepted_attempt.correspondence_path)
            if correspondence_source.is_file():
                correspondence_path = final_dir / "geometry_correspondence.json"
                shutil.copy2(correspondence_source, correspondence_path)
                accepted_correspondence = CorrespondenceEvidence.model_validate_json(
                    correspondence_source.read_text(encoding="utf-8")
                )
                warnings.extend(accepted_correspondence.warnings)
        classified_hole_loops = {
            (intent.target_mesh_path, frozenset(intent.boundary_loop_vertex_ids))
            for intent in request.classified_holes
        }
        inferred_protections = [
            candidate_to_protected_feature(candidate)
            for candidate in (
                protected_feature_report.candidates if protected_feature_report else []
            )
            if (
                candidate.scope_path,
                frozenset(candidate.evidence.get("loop_vertex_ids", [])),
            )
            not in classified_hole_loops
        ]
        collision_report = _run_collision_geometry_subprocess(
            render_path=render_path,
            output_path=final_dir / "collision.usda",
            profile=request.profile,
            budgets=request.budgets,
            protected_features=[*request.protected_features, *inferred_protections],
            deterministic_seed=request.deterministic_seed,
            coacd_enabled="coacd_collision" in enabled_workers,
            sdf_collision_rebuild_enabled=(SDF_COLLISION_REBUILD_WORKER in enabled_workers),
            timeout_s=max(
                0.001,
                request.budgets.timeout_s - (time.monotonic() - started),
            ),
            runtime_engine=request.collision_runtime_engine,
            report_path=collision_report_path,
            source_collision_audit_path=final_dir / "source_collision_audit.json",
            attempt_dir=final_dir / "collision_worker",
        )
        collision_path = (
            Path(collision_report.collision_path) if collision_report.collision_path else None
        )
        source_collision_audit_path = (
            Path(collision_report.source_collision_audit_path)
            if collision_report.source_collision_audit_path
            else None
        )
        composed_path = compose_asset(render_path, collision_path, final_dir / "asset.usda")
        usd_package_validation = _validate_final_usd_package(
            render_path,
            collision_path,
            composed_path,
            final_dir / "usd_package_validation.json",
        )
        if request.profile in {"articulated_rigid", "contact_rich"}:
            advanced_request = AdvancedProfileRequest.model_validate(
                request.advanced_profile or {"profile": request.profile}
            )
            advanced_profile_path = final_dir / "advanced_profile_evidence.json"
            if collision_path is not None:
                advanced_profile_report = evaluate_advanced_profile(
                    render_path,
                    collision_path,
                    advanced_request,
                )
            else:
                advanced_profile_report = collision_unavailable_advanced_profile_report(
                    advanced_request,
                    reason="accepted collision geometry is unavailable",
                )
            write_advanced_profile_report(advanced_profile_report, advanced_profile_path)
            if advanced_profile_report.disposition == "failed":
                blockers.extend(advanced_profile_report.blockers)
            elif advanced_profile_report.disposition == "conditional":
                warnings.extend(advanced_profile_report.blockers)
        warnings.extend(collision_report.warnings)
        warnings.extend(usd_package_validation["warnings"])
        if accepted_diagnosis is not None:
            warnings.extend(
                issue.summary for issue in accepted_diagnosis.issues if issue.severity == "warning"
            )
        warnings.extend(
            issue.summary
            for issue in diagnosis.issues
            if issue.issue_id == "asset:unresolved_material_dependencies"
        )
        if accepted_fidelity is not None:
            warnings.extend(accepted_fidelity.warnings)
        render_geometry_changed = bool(
            accepted_fidelity is not None and not accepted_fidelity.exact_world_geometry_match
        )
        if render_geometry_changed:
            warnings.append(
                "Changed render geometry requires accepted OVRTX evidence or explicit human review."
            )
        blockers.extend(collision_report.failures)
        blockers.extend(usd_package_validation["failures"])
        if (
            collision_report.status == "fail"
            or usd_package_validation["status"] == "fail"
            or source_format_geometry_status == "fail"
            or (
                advanced_profile_report is not None
                and advanced_profile_report.disposition == "failed"
            )
        ):
            outcome = "rejected"
        elif (
            collision_report.status == "conditional"
            or (accepted_diagnosis is not None and accepted_diagnosis.status == "conditional")
            or accepted_fidelity is None
            or accepted_fidelity.status == "conditional"
            or not request.profile_confirmed
            or source_rights_status == "missing"
            or render_geometry_changed
            or source_format_geometry_status == "not_evaluated"
            or advanced_profile_report is not None
        ):
            outcome = "conditional"
        else:
            outcome = "certified"
    else:
        warnings.append(
            "Native CAD healing passed, but conversion and USD profile validation remain required."
        )
        outcome = "conditional"
    claim_scope = (
        f"geometry_repair.{request.profile}.geometry_only"
        if request.profile in {"articulated_rigid", "contact_rich"}
        else f"geometry_repair.{request.profile}"
    )
    final_diagnosis = accepted_diagnosis or diagnosis
    usd_package_status = (
        str(usd_package_validation["status"])
        if usd_package_validation is not None
        else "not_evaluated"
    )
    role_validation = build_final_role_validation(
        profile=request.profile,
        diagnosis_role_validation=final_diagnosis.role_validation,
        source_format=final_diagnosis.metrics.source_format,
        render_path=str(render_path) if render_path else None,
        collision_path=str(collision_path) if collision_path else None,
        source_format_status=source_format_geometry_status,
        fidelity_status=accepted_fidelity.status if accepted_fidelity else "not_evaluated",
        visual_review_required=render_geometry_changed,
        collision_status=collision_report.status if collision_report else "not_evaluated",
        collision_runtime_status=(
            collision_report.runtime_status if collision_report else "not_evaluated"
        ),
        usd_package_status=usd_package_status,
        advanced_profile_status=(
            advanced_profile_report.status
            if advanced_profile_report is not None
            else "not_evaluated"
        ),
        evidence_paths_by_role={
            "render": [
                str(diagnosis_path),
                str(source_format_validation_path),
                *(
                    [accepted_attempt.fidelity_path]
                    if accepted_attempt and accepted_attempt.fidelity_path
                    else []
                ),
                *([str(correspondence_path)] if correspondence_path else []),
                *(
                    [str(final_dir / "usd_package_validation.json")]
                    if usd_package_validation is not None
                    else []
                ),
            ],
            "collision": [
                *([str(collision_report_path)] if collision_report is not None else []),
                *([str(source_collision_audit_path)] if source_collision_audit_path else []),
                *(
                    [str(final_dir / "usd_package_validation.json")]
                    if usd_package_validation is not None
                    else []
                ),
            ],
            "brep_source": [
                str(diagnosis_path),
                *(
                    [accepted_attempt.fidelity_path]
                    if accepted_attempt and accepted_attempt.fidelity_path
                    else []
                ),
            ],
            "helper": [*([str(advanced_profile_path)] if advanced_profile_path else [])],
        },
    )
    failed_required_roles = [
        role
        for role, role_evidence in role_validation.items()
        if role_evidence.required and role_evidence.status == "fail"
    ]
    incomplete_required_roles = [
        role
        for role, role_evidence in role_validation.items()
        if role_evidence.required and role_evidence.status != "pass"
    ]
    if failed_required_roles:
        outcome = "rejected"
        blockers.extend(f"required geometry role {role!r} failed" for role in failed_required_roles)
    elif incomplete_required_roles and outcome == "certified":
        outcome = "conditional"
        warnings.extend(
            f"required geometry role {role!r} is not fully passed"
            for role in incomplete_required_roles
        )
    operation_report_artifacts: list[dict[str, str]] = []
    accepted_backend_identity: dict[str, Any] | None = None
    accepted_backend_qualification_id: str | None = None
    operation_evidence_error: str | None = None
    try:
        operation_report_artifacts = _verified_operation_report_artifacts(attempts)
        (
            accepted_backend_identity,
            accepted_backend_qualification_id,
        ) = _accepted_backend_provenance(accepted_attempt)
    except RuntimeError as exc:
        operation_evidence_error = f"final operation evidence verification failed: {exc}"
        outcome = "rejected"
        blockers.append(operation_evidence_error)
    certificate_path = final_dir / "repair_certificate.json"
    certificate = RepairCertificate(
        outcome=outcome,
        claim_scope=claim_scope,
        source_sha256=source_package.source.sha256,
        normalized_source_sha256=diagnosis.source_sha256,
        output_sha256=file_sha256(render_path) if render_path else None,
        profile=request.profile,
        profile_confirmed=request.profile_confirmed,
        production_use=request.production_use,
        source_rights_status=source_rights_status,
        render_geometry_changed=render_geometry_changed,
        visual_review_status="required" if render_geometry_changed else "not_required",
        deterministic_seed=request.deterministic_seed,
        routing_policy_id=plan.routing_policy_id,
        accepted_attempt_id=accepted_attempt.attempt_id if accepted_attempt else None,
        accepted_backend_identity=accepted_backend_identity,
        accepted_backend_qualification_id=accepted_backend_qualification_id,
        operations_applied=applied_operations,
        deleted_entities=deleted_entities,
        merged_entities=merged_entities,
        diagnosis_path=str(diagnosis_path),
        source_format_validation_path=str(source_format_validation_path),
        repair_plan_path=str(plan_path),
        fidelity_path=accepted_attempt.fidelity_path if accepted_attempt else None,
        collision_report_path=str(collision_report_path) if collision_report is not None else None,
        source_collision_audit_path=(
            str(source_collision_audit_path) if source_collision_audit_path else None
        ),
        dependency_localization_path=(
            str(dependency_localization_path) if dependency_localization_path else None
        ),
        protected_feature_candidates_path=(
            str(protected_feature_candidates_path) if protected_feature_candidates_path else None
        ),
        manifold_seam_analysis_path=(
            str(manifold_seam_analysis_path) if manifold_seam_analysis_path else None
        ),
        advanced_profile_evidence_path=str(advanced_profile_path)
        if advanced_profile_path
        else None,
        validation_results={
            "source_format": source_format_geometry_status,
            "source_format_raw": source_format_validation.status,
            "diagnosis": accepted_diagnosis.status
            if accepted_diagnosis is not None
            else diagnosis.status,
            "fidelity": accepted_fidelity.status if accepted_fidelity else "not_evaluated",
            "source_rights": source_rights_status,
            "visual_review": "required" if render_geometry_changed else "not_required",
            "collision": collision_report.status if collision_report else "not_evaluated",
            "collision_runtime": (
                collision_report.runtime_status if collision_report is not None else "not_evaluated"
            ),
            "usd_package": (
                usd_package_validation["status"]
                if usd_package_validation is not None
                else "not_evaluated"
            ),
            "correspondence": (
                CorrespondenceEvidence.model_validate_json(
                    correspondence_path.read_text(encoding="utf-8")
                ).status
                if correspondence_path
                else "not_evaluated"
            ),
            "protected_feature_candidates": (
                "pass" if protected_feature_candidates_path else "not_evaluated"
            ),
            "manifold_seam_analysis": (
                ManifoldSeamAnalysis.model_validate_json(
                    manifold_seam_analysis_path.read_text(encoding="utf-8")
                ).status
                if manifold_seam_analysis_path
                else "not_evaluated"
            ),
            "source_collision_audit": (
                SourceCollisionAudit.model_validate_json(
                    source_collision_audit_path.read_text(encoding="utf-8")
                ).status
                if source_collision_audit_path
                else "not_evaluated"
            ),
            "usd_intake": (
                "pass"
                if usd_intake is not None
                and usd_intake.source_unchanged
                and usd_intake.stage_readable
                and not usd_intake.unresolved_dependencies
                else "conditional"
                if usd_intake is not None and usd_intake.source_unchanged
                else "not_evaluated"
            ),
            "dependency_localization": (
                "pass"
                if dependency_localization is not None
                and dependency_localization.portable_package_complete
                else "conditional"
                if dependency_localization is not None
                else "not_evaluated"
            ),
            "scalable_audit": (
                "pass"
                if diagnosis.metrics.mesh.self_intersection_status == "pass"
                and diagnosis.metrics.mesh.coplanar_overlap_status == "pass"
                else "fail"
                if diagnosis.metrics.mesh.self_intersection_status == "fail"
                or diagnosis.metrics.mesh.coplanar_overlap_status == "fail"
                else "not_evaluated"
            ),
            "advanced_profile_geometry": (
                advanced_profile_report.status
                if advanced_profile_report is not None
                else "not_evaluated"
            ),
            "operation_report_provenance": "fail" if operation_evidence_error else "pass",
            **{f"role.{role}": evidence.status for role, evidence in role_validation.items()},
        },
        role_validation=role_validation,
        remaining_warnings=sorted(set(warnings)),
        blockers=sorted(set(blockers)),
        downstream_owners=[
            "content-workflow-material",
            "content-workflow-physics",
            "content-workflow-articulation",
            "content-workflow-runtime-validation",
            "content-workflow-simready",
        ],
        certificate_path=str(certificate_path),
    )
    atomic_write_json(certificate_path, certificate)
    evidence_path = final_dir / "geometry_validation_evidence.json"
    evidence_artifacts = [
        {
            "kind": kind,
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for kind, path in (
            ("source_format_validation", source_format_validation_path),
            ("geometry_diagnosis", diagnosis_path),
            *([("usd_intake", usd_intake_path)] if usd_intake_path else []),
            *(
                [("dependency_localization", dependency_localization_path)]
                if dependency_localization_path
                else []
            ),
            *(
                [("scalable_geometry_audit", Path(diagnosis.scalable_audit_path))]
                if diagnosis.scalable_audit_path
                else []
            ),
            ("repair_plan", plan_path),
            ("attempt_ledger", attempts_path),
            ("repair_certificate", certificate_path),
            *(
                [("geometry_fidelity", Path(accepted_attempt.fidelity_path))]
                if accepted_attempt and accepted_attempt.fidelity_path
                else []
            ),
            *(
                [
                    (
                        "protected_feature_comparison",
                        Path(accepted_attempt.protected_feature_comparison_path),
                    )
                ]
                if accepted_attempt and accepted_attempt.protected_feature_comparison_path
                else []
            ),
            *(
                [("collision_report", collision_report_path)]
                if collision_report is not None
                else []
            ),
            *(
                [("source_collision_audit", source_collision_audit_path)]
                if source_collision_audit_path is not None
                else []
            ),
            *(
                [("protected_feature_candidates", protected_feature_candidates_path)]
                if protected_feature_candidates_path is not None
                else []
            ),
            *(
                [("manifold_seam_analysis", manifold_seam_analysis_path)]
                if manifold_seam_analysis_path is not None
                else []
            ),
            *(
                [("advanced_profile_geometry", advanced_profile_path)]
                if advanced_profile_path is not None
                else []
            ),
            *(
                [("geometry_correspondence", correspondence_path)]
                if correspondence_path is not None
                else []
            ),
            *(
                [("usd_package_validation", final_dir / "usd_package_validation.json")]
                if usd_package_validation is not None
                else []
            ),
        )
        if path.is_file()
    ]
    evidence_artifacts.extend(operation_report_artifacts)
    atomic_write_json(
        evidence_path,
        {
            "schema_version": "content-agent-workflows.validation-evidence.v1",
            "workflow": "geometry_repair",
            "asset": str(composed_path or render_path or normalized_source),
            "target_runtime": "profile-defined",
            "validation_tier": "T1_basic_stability",
            "checks": [
                {
                    "name": "geometry_repair_certificate",
                    "status": "pass"
                    if outcome == "certified"
                    else "warning"
                    if outcome == "conditional"
                    else "fail",
                    "summary": f"Geometry repair outcome: {outcome}.",
                    "evidence_artifacts": evidence_artifacts,
                    "failures": certificate.blockers,
                    "warnings": certificate.remaining_warnings,
                    "repair_hints": [],
                    "metadata": {"claim_scope": claim_scope},
                },
                *(
                    {
                        "name": f"geometry_role_{role}",
                        "status": (
                            "pass"
                            if role_evidence.status in {"pass", "not_applicable"}
                            else "fail"
                            if role_evidence.status == "fail"
                            else "warning"
                        ),
                        "summary": (
                            f"{role} geometry role: {role_evidence.status}; "
                            f"required={str(role_evidence.required).lower()}."
                        ),
                        "evidence_artifacts": [],
                        "failures": role_evidence.blockers,
                        "warnings": role_evidence.warnings,
                        "repair_hints": [],
                        "metadata": role_evidence.model_dump(mode="json"),
                    }
                    for role, role_evidence in role_validation.items()
                ),
            ],
            "evidence_artifacts": evidence_artifacts,
            "failures": certificate.blockers,
            "warnings": certificate.remaining_warnings,
            "unresolved_issues": [
                *certificate.blockers,
                *certificate.remaining_warnings,
            ],
            "repair_hints": [],
            "sim_ready_status": "not_evaluated",
            "metadata": {
                "geometry_repair_outcome": outcome,
                "role_validation": {
                    role: role_evidence.model_dump(mode="json")
                    for role, role_evidence in role_validation.items()
                },
            },
        },
    )
    manifest_path = final_dir / "content_agents_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "geometry-repair.handoff.v1",
            "workflow": "geometry_repair",
            "claim_scope": claim_scope,
            "outcome": outcome,
            "source": source_package.model_dump(mode="json"),
            "source_format_validation": str(source_format_validation_path),
            "usd_intake": str(usd_intake_path) if usd_intake_path else None,
            "dependency_localization": (
                str(dependency_localization_path) if dependency_localization_path else None
            ),
            "normalized_source": str(normalized_source),
            "deterministic_seed": request.deterministic_seed,
            "render_usd": str(render_path) if render_path else None,
            "collision_usd": str(collision_path) if collision_path else None,
            "collision_report": str(collision_report_path)
            if collision_report is not None
            else None,
            "source_collision_audit": (
                str(source_collision_audit_path) if source_collision_audit_path else None
            ),
            "protected_feature_candidates": (
                str(protected_feature_candidates_path)
                if protected_feature_candidates_path
                else None
            ),
            "manifold_seam_analysis": (
                str(manifold_seam_analysis_path) if manifold_seam_analysis_path else None
            ),
            "advanced_profile_evidence": (
                str(advanced_profile_path) if advanced_profile_path else None
            ),
            "usd_package_validation": (
                str(final_dir / "usd_package_validation.json")
                if usd_package_validation is not None
                else None
            ),
            "geometry_correspondence": (str(correspondence_path) if correspondence_path else None),
            "protected_feature_comparison": (
                accepted_attempt.protected_feature_comparison_path if accepted_attempt else None
            ),
            "separated_source_roles": (separated_source_roles if render_path is not None else None),
            "asset_usd": str(composed_path) if composed_path else None,
            "diagnosis": str(diagnosis_path),
            "scalable_audit": diagnosis.scalable_audit_path,
            "repair_plan": str(plan_path),
            "attempt_ledger": str(attempts_path),
            "attempt_ledger_sha256": file_sha256(attempts_path),
            "repair_certificate": str(certificate_path),
            "repair_certificate_sha256": file_sha256(certificate_path),
            "geometry_validation_evidence": str(evidence_path),
            "geometry_validation_evidence_sha256": file_sha256(evidence_path),
            "accepted_backend_identity": certificate.accepted_backend_identity,
            "accepted_backend_qualification_id": (certificate.accepted_backend_qualification_id),
            "downstream_owners": certificate.downstream_owners,
            "role_validation": {
                role: evidence.model_dump(mode="json") for role, evidence in role_validation.items()
            },
        },
    )
    report_path = final_dir / "repair_report.md"
    atomic_write_text(
        report_path,
        _report_markdown(request, diagnosis, certificate, attempts),
    )
    result_path = final_dir / "repair_result.json"
    result = RepairResult(
        outcome=outcome,
        claim_scope=claim_scope,
        output_dir=str(output_dir),
        source_package_path=source_package.manifest_path,
        source_format_validation_path=str(source_format_validation_path),
        usd_intake_path=str(usd_intake_path) if usd_intake_path else None,
        dependency_localization_path=(
            str(dependency_localization_path) if dependency_localization_path else None
        ),
        normalized_source_path=str(normalized_source),
        render_usd_path=str(render_path) if render_path else None,
        collision_usd_path=str(collision_path) if collision_path else None,
        composed_usd_path=str(composed_path) if composed_path else None,
        diagnosis_path=str(diagnosis_path),
        scalable_audit_path=diagnosis.scalable_audit_path,
        repair_plan_path=str(plan_path),
        attempts_path=str(attempts_path),
        collision_report_path=str(collision_report_path) if collision_report is not None else None,
        correspondence_path=str(correspondence_path) if correspondence_path else None,
        source_collision_audit_path=(
            str(source_collision_audit_path) if source_collision_audit_path else None
        ),
        protected_feature_candidates_path=(
            str(protected_feature_candidates_path) if protected_feature_candidates_path else None
        ),
        manifold_seam_analysis_path=(
            str(manifold_seam_analysis_path) if manifold_seam_analysis_path else None
        ),
        advanced_profile_evidence_path=str(advanced_profile_path)
        if advanced_profile_path
        else None,
        certificate_path=str(certificate_path),
        manifest_path=str(manifest_path),
        geometry_validation_evidence_path=str(evidence_path),
        report_path=str(report_path),
        attempts=attempts,
        role_validation=role_validation,
        error=operation_evidence_error,
    )
    atomic_write_json(result_path, result)
    return result
