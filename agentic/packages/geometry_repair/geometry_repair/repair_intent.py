# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed repair-intent validation and deterministic operation binding."""

from __future__ import annotations

from pathlib import Path

from .models import (
    Diagnosis,
    GeometryRole,
    ProtectedFeature,
    RepairIntent,
    RepairOperation,
    TopologyEffect,
)
from .policy import assert_worker_enabled
from .worker_ids import SDF_COLLISION_REBUILD_WORKER, SDF_REBUILD_WORKER

_BREP_WORKERS = {"ocp_shape_heal"}
_COLLISION_WORKERS = {"coacd_collision", SDF_COLLISION_REBUILD_WORKER}

_TOPOLOGY_EFFECTS_BY_WORKER: dict[str, list[TopologyEffect]] = {
    "noop": ["none"],
    "scene_optimizer_deinstance": ["none"],
    "usd_structure_repair": ["none"],
    "trimesh_conservative_cleanup": [
        "delete_source_faces",
        "reverse_source_faces",
    ],
    "trimesh_bounded_hole_fill": ["generate_local_faces"],
    "pmp_patch": ["generate_local_faces"],
    "geogram_local_repair": ["split_source_faces", "generate_local_faces"],
    SDF_REBUILD_WORKER: ["replace_target_surface"],
    SDF_COLLISION_REBUILD_WORKER: ["generate_collision_representation"],
    "coacd_collision": ["generate_collision_representation"],
    "ocp_shape_heal": ["heal_brep_topology"],
}


def operation_target_role(operation: RepairOperation, diagnosis: Diagnosis) -> GeometryRole:
    if operation.worker in _BREP_WORKERS:
        return "brep_source"
    if operation.worker in _COLLISION_WORKERS:
        return "collision"
    if operation.worker == "noop" and any(
        "brep_source" in issue.affected_roles for issue in diagnosis.issues
    ):
        return "brep_source"
    return "render"


def _operation_target_paths(
    operation: RepairOperation,
    diagnosis: Diagnosis,
) -> list[str]:
    paths = {
        path
        for issue in diagnosis.issues
        if issue.issue_id in operation.issue_ids
        for path in issue.affected_prim_paths
    }
    target_mesh_path = operation.parameters.get("target_mesh_path")
    if isinstance(target_mesh_path, str) and target_mesh_path:
        paths.add(target_mesh_path)
    return sorted(paths)


def bind_deterministic_repair_intents(
    operations: list[RepairOperation],
    diagnosis: Diagnosis,
    protected_features: list[ProtectedFeature],
) -> tuple[list[RepairOperation], list[RepairIntent]]:
    """Give every operation an auditable intent without changing route order."""

    bound: list[RepairOperation] = []
    intents: list[RepairIntent] = []
    protected_names = sorted(feature.name for feature in protected_features if feature.required)
    evidence_paths = [diagnosis.report_path] if diagnosis.report_path else []
    for operation in operations:
        intent_id = f"repair-intent-{operation.operation_id}"
        target_role = operation_target_role(operation, diagnosis)
        target_paths = _operation_target_paths(operation, diagnosis)
        effects = _TOPOLOGY_EFFECTS_BY_WORKER.get(operation.worker)
        if effects is None:
            raise ValueError(f"No typed topology-effect policy for worker {operation.worker!r}")
        intent = RepairIntent(
            intent_id=intent_id,
            authority="deterministic",
            target_role=target_role,
            target_prim_paths=target_paths,
            issue_ids=sorted(set(operation.issue_ids)),
            defect_class=(operation.issue_ids[0] if operation.issue_ids else "identity_or_noop"),
            protected_feature_names=protected_names,
            expected_topology_effects=effects,
            candidate_workers=[operation.worker],
            evidence_paths=evidence_paths,
        )
        intents.append(intent)
        bound.append(
            operation.model_copy(
                update={
                    "intent_id": intent_id,
                    "target_role": target_role,
                    "target_prim_paths": target_paths,
                }
            )
        )
    return bound, intents


def validate_proposed_repair_intents(
    intents: list[RepairIntent],
    diagnosis: Diagnosis,
    *,
    enabled_workers: set[str],
    protected_features: list[ProtectedFeature],
) -> list[RepairIntent]:
    """Reject proposals that are not grounded in measured, policy-approved evidence."""

    known_issues = {issue.issue_id: issue for issue in diagnosis.issues}
    known_paths = {
        *diagnosis.metrics.source_part_paths,
        *diagnosis.metrics.source_collision_paths,
        *diagnosis.metrics.source_helper_paths,
        *(path for issue in diagnosis.issues for path in issue.affected_prim_paths),
    }
    protected_names = {feature.name for feature in protected_features}
    validated: list[RepairIntent] = []
    for intent in intents:
        if intent.authority == "deterministic":
            raise ValueError(
                f"proposed repair intent {intent.intent_id!r} cannot claim deterministic "
                "authority; deterministic intents are planner-authored"
            )
        unknown_issues = set(intent.issue_ids) - known_issues.keys()
        if unknown_issues:
            raise ValueError(
                f"repair intent {intent.intent_id!r} cites unknown issue_ids: "
                + ", ".join(sorted(unknown_issues))
            )
        incompatible = [
            issue_id
            for issue_id in intent.issue_ids
            if intent.target_role not in known_issues[issue_id].affected_roles
        ]
        if incompatible:
            raise ValueError(
                f"repair intent {intent.intent_id!r} target role {intent.target_role!r} "
                "does not match issue role evidence: " + ", ".join(sorted(incompatible))
            )
        if known_paths:
            unknown_paths = set(intent.target_prim_paths) - known_paths
            if unknown_paths:
                raise ValueError(
                    f"repair intent {intent.intent_id!r} targets unknown prim paths: "
                    + ", ".join(sorted(unknown_paths))
                )
        unknown_features = set(intent.protected_feature_names) - protected_names
        if unknown_features:
            raise ValueError(
                f"repair intent {intent.intent_id!r} cites unknown protected features: "
                + ", ".join(sorted(unknown_features))
            )
        unknown_workers = set(intent.candidate_workers) - enabled_workers
        if unknown_workers:
            raise ValueError(
                f"repair intent {intent.intent_id!r} cites disabled workers: "
                + ", ".join(sorted(unknown_workers))
            )
        for worker in intent.candidate_workers:
            assert_worker_enabled(worker)
            expected_effects = _TOPOLOGY_EFFECTS_BY_WORKER.get(worker)
            if expected_effects is None:
                raise ValueError(
                    f"repair intent {intent.intent_id!r} has no topology policy for {worker!r}"
                )
            unsupported_effects = set(intent.expected_topology_effects) - set(expected_effects)
            if unsupported_effects:
                raise ValueError(
                    f"repair intent {intent.intent_id!r} misstates topology effects for "
                    f"{worker!r}: " + ", ".join(sorted(unsupported_effects))
                )
            expected_role: GeometryRole = (
                "brep_source"
                if worker in _BREP_WORKERS
                else "collision"
                if worker in _COLLISION_WORKERS
                else "render"
            )
            if intent.target_role != expected_role:
                raise ValueError(
                    f"repair intent {intent.intent_id!r} targets {intent.target_role!r} with "
                    f"{worker!r}, whose policy role is {expected_role!r}"
                )
        missing_evidence = [path for path in intent.evidence_paths if not Path(path).is_file()]
        if missing_evidence:
            raise ValueError(
                f"repair intent {intent.intent_id!r} cites missing evidence paths: "
                + ", ".join(sorted(missing_evidence))
            )
        validated.append(intent)
    return validated


def rank_operations_from_proposed_intents(
    operations: list[RepairOperation],
    intents: list[RepairIntent],
) -> tuple[list[RepairOperation], list[dict[str, object]]]:
    """Reorder approved candidates without changing their content or baseline."""

    if not intents or len(operations) < 2:
        return list(operations), []
    baseline = [operation for operation in operations if operation.worker == "noop"]
    candidates = [operation for operation in operations if operation.worker != "noop"]
    if len(baseline) > 1:
        raise ValueError("repair planning permits at most one no-op baseline")
    scored: list[tuple[float, int, RepairOperation, list[str]]] = []
    for index, operation in enumerate(candidates):
        matches = []
        for intent in intents:
            if operation.worker not in intent.candidate_workers:
                continue
            if intent.target_role != operation.target_role:
                continue
            if not set(intent.issue_ids) & set(operation.issue_ids):
                continue
            if (
                intent.target_prim_paths
                and operation.target_prim_paths
                and not set(intent.target_prim_paths) & set(operation.target_prim_paths)
            ):
                continue
            matches.append(intent)
        score = max((intent.confidence for intent in matches), default=-1.0)
        scored.append((score, index, operation, sorted(intent.intent_id for intent in matches)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    evidence = [
        {
            "operation_id": operation.operation_id,
            "worker": operation.worker,
            "proposal_score": score if score >= 0.0 else None,
            "matching_intent_ids": intent_ids,
            "deterministic_fallback_index": index,
        }
        for score, index, operation, intent_ids in scored
    ]
    return [*baseline, *(item[2] for item in scored)], evidence


__all__ = [
    "bind_deterministic_repair_intents",
    "operation_target_role",
    "rank_operations_from_proposed_intents",
    "validate_proposed_repair_intents",
]
