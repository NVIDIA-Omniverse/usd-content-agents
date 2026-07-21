# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Promote a preflighted Stage 2 document to the first-class contract."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    identify_usd_artifact,
)

from joint_agent.functions import candidate_edge_authoring
from joint_agent.functions.articulation_candidates import Stage2ArticulationCandidate
from joint_agent.functions.articulation_contract import (
    ARTICULATION_CONTRACT_SCHEMA_VERSION,
    ArticulationContractV1,
    BodyAuthoring,
    ContractSummaryV1,
    JointRecordV1,
    LinkRecordV1,
    PrimRecordV1,
)
from joint_agent.functions.consistency import (
    canonical_link_instance_id,
    is_model_supplied_link_instance_id,
)
from joint_agent.functions.joint_rigger_core_bridge import (
    NoReadyJointCandidatesError,
    _close_sealed_candidate_binding,
    _create_sealed_candidate_binding,
    _require_candidate_path_authority,
    _require_sealed_candidate_binding,
    _write_sealed_candidate_snapshot,
    build_stage2_candidate_edges_input,
)
from joint_agent.functions.stage1_schema import unwrap_stage1_prediction_payload

_DERIVATION = "preflighted_stage2_v0_to_articulation_contract_v1"
_AGGREGATE_BODY_DERIVATION = "deterministic_flat_member_aggregate_target_v1"
_FIXED_BODY_DERIVATION = "stage1_fixed_body_membership_projection_v1"
_FIXED_ALIAS_DERIVATION = "stage1_fixed_body_alias_canonicalization_v1"
_SUPPORTED_STAGE2_TYPES = frozenset({"prismatic", "revolute"})
_UNRESOLVED_ROLES = frozenset({"", "unknown", "none", "null", "n/a", "na"})
_FIXED_JOINT_HINTS = frozenset({"none", "fixed"})
_MAX_PREDICTIONS_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class _FixedAssemblyProjection:
    artifact: ArtifactIdentityV1
    link_id: str
    parent_path: str
    member_paths: tuple[str, ...]

    @property
    def aliases(self) -> frozenset[str]:
        return frozenset((*self.member_paths, self.parent_path))


@dataclass(frozen=True)
class _Stage1Prediction:
    prim_path: str
    payload: Mapping[str, Any]


def build_articulation_contract_from_stage2(
    *,
    input_usd_path: str | Path,
    articulation_candidates_path: str | Path,
    predictions_path: str | Path | None = None,
    expected_articulation_candidates_sha256: str | None = None,
    allow_ready_subset: bool = False,
) -> ArticulationContractV1:
    """Build a first-class contract from one exact, ready Stage 2 artifact.

    The existing Stage 2 bridge remains the authority for source binding,
    candidate readiness, graph checks, stage-frame axes, and limit policy. This
    producer consumes that preflighted request instead of independently
    reimplementing those rules. Review-required candidates are rejected as a
    whole unless the caller explicitly requests a ready-subset projection. The
    subset remains evidence-bound to the complete Stage 2 artifact.
    """

    source_path = Path(input_usd_path)
    candidates_path = Path(articulation_candidates_path)
    fixed_assembly: _FixedAssemblyProjection | None = None
    prediction_artifact: ArtifactIdentityV1 | None = None
    stage1_predictions: tuple[_Stage1Prediction, ...] | None = None
    if predictions_path is not None:
        stage1_predictions, prediction_artifact = _load_stage1_predictions(
            Path(predictions_path),
        )
    candidate_binding: Any | None = None
    primary_error: BaseException | None = None
    try:
        candidate_binding = _create_sealed_candidate_binding(
            candidates_path,
            expected_sha256=expected_articulation_candidates_sha256,
        )
        candidates_sha256 = candidate_binding.sha256
        with tempfile.TemporaryDirectory(
            prefix="joint-agent-contract-stage2-"
        ) as snapshot_dir:
            candidate_snapshot = Path(snapshot_dir) / "candidates.json"
            _write_sealed_candidate_snapshot(candidate_binding, candidate_snapshot)
            candidate_bytes = _read_candidate_bytes(
                candidate_snapshot,
                display_path=candidates_path,
            )
            if hashlib.sha256(candidate_bytes).hexdigest() != candidates_sha256:
                raise JointRiggerContractError(
                    "stage2_snapshot_mutated",
                    "private Stage 2 candidate snapshot changed before parsing",
                )
            candidates = _load_candidates(
                candidate_bytes,
                path=candidates_path,
            )
            review_required = sorted(
                candidate.candidate_id
                for candidate in candidates
                if candidate.review_status != "ready_for_rigger_input"
            )
            no_ready = not candidates or len(review_required) == len(candidates)
            if no_ready:
                preflight_bytes = candidate_bytes
            else:
                preflight_bytes = _topology_preflight_bytes(
                    candidate_bytes,
                    candidates,
                )
            preflight_sha256 = hashlib.sha256(preflight_bytes).hexdigest()
            preflight_snapshot = candidate_snapshot
            if preflight_sha256 != candidates_sha256:
                preflight_snapshot = (
                    Path(snapshot_dir) / "topology-preflight-candidates.json"
                )
                _write_private_candidate_snapshot(
                    preflight_snapshot,
                    preflight_bytes,
                )
            request: JointRiggerInputV1 | None = None
            try:
                request = _build_preflight_request(
                    source_path=source_path,
                    candidate_snapshot=preflight_snapshot,
                    expected_candidates_sha256=preflight_sha256,
                )
            except NoReadyJointCandidatesError:
                if not no_ready:
                    raise
            try:
                _require_sealed_candidate_binding(candidate_binding)
                _require_candidate_path_authority(candidates_path, candidate_binding)
            except JointRiggerContractError:
                raise
            except Exception as exc:
                raise JointRiggerContractError(
                    "stage2_artifact_mutated",
                    "Stage 2 candidate document changed while the contract was built",
                ) from exc
            if no_ready:
                if request is not None:  # pragma: no cover - preflight invariant
                    raise JointRiggerContractError(
                        "stage2_preflight_projection_mismatch",
                        "candidate readiness and preflighted topology differ",
                    )
                if not candidates:
                    raise JointRiggerContractError(
                        "stage2_candidates_empty",
                        "a first-class articulation contract requires at least one "
                        "candidate",
                    )
                raise JointRiggerContractError(
                    "stage2_candidates_no_ready",
                    "first-class promotion found no ready candidates; "
                    "review-required candidates: " + ", ".join(review_required),
                )
            assert request is not None
            if review_required and not allow_ready_subset:
                raise JointRiggerContractError(
                    "stage2_candidates_require_review",
                    "first-class promotion is all-or-nothing; review-required "
                    "candidates: " + ", ".join(review_required),
                )
            ready_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.review_status == "ready_for_rigger_input"
            )
            candidate_by_key = _index_candidates(ready_candidates)
            _require_unique_child_topologies(candidate_by_key)
            link_candidate_by_body = _link_candidates_for_ready_subset(
                candidates=candidates,
                ready_candidate_by_key=candidate_by_key,
            )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if candidate_binding is not None:
            cleanup_errors = _close_sealed_candidate_binding(candidate_binding)
            if cleanup_errors:
                detail = "; ".join(str(error) for error in cleanup_errors)
                if primary_error is not None:
                    primary_error.add_note(
                        "Bound Stage 2 candidate cleanup also failed: " + detail
                    )
                elif len(cleanup_errors) == 1:
                    raise cleanup_errors[0]
                else:
                    raise ExceptionGroup(
                        "Bound Stage 2 candidate cleanup failed",
                        cleanup_errors,
                    )

    if stage1_predictions is not None:
        assert prediction_artifact is not None
        fixed_assembly = _build_fixed_assembly_projection(
            stage1_predictions,
            prediction_artifact,
        )

    candidate_artifact = ArtifactIdentityV1(
        uri=str(candidates_path),
        root_sha256=candidates_sha256,
    )
    if fixed_assembly is not None:
        _validate_fixed_assembly_source_members(
            source_path,
            fixed_assembly,
            expected_source=request.source_asset,
        )
    plan_by_key = _index_plans(request.plan.joints)
    if set(candidate_by_key) != set(plan_by_key):
        missing = sorted(set(candidate_by_key) - set(plan_by_key), key=str)
        extra = sorted(set(plan_by_key) - set(candidate_by_key), key=str)
        raise JointRiggerContractError(
            "stage2_preflight_projection_mismatch",
            f"candidate and preflighted topology differ; missing={missing}, extra={extra}",
        )

    canonical_body0_by_key = _canonical_parent_links(
        link_candidate_by_body=link_candidate_by_body,
        plan_by_key=plan_by_key,
        fixed_assembly=fixed_assembly,
    )

    candidate_for_body = dict(link_candidate_by_body)
    incoming_axis: dict[str, tuple[float, float, float] | None] = {}
    members_by_body: dict[str, set[str]] = defaultdict(set)
    for key, candidate in candidate_by_key.items():
        topology = plan_by_key[key].topology
        body0_link = canonical_body0_by_key[key]
        incoming_axis[topology.body1] = topology.axis_stage
        if fixed_assembly is not None and body0_link == fixed_assembly.link_id:
            members_by_body[body0_link].update(fixed_assembly.member_paths)
        else:
            parent_candidate = candidate_for_body.get(body0_link)
            if parent_candidate is None:
                members_by_body[body0_link].add(body0_link)
            else:
                members_by_body[body0_link].update(parent_candidate.moving_part_prims)
        members_by_body[topology.body1].update(candidate.moving_part_prims)

    moving_aggregate_members: set[str] = set()
    for body_path, body_members in members_by_body.items():
        if fixed_assembly is not None and body_path == fixed_assembly.link_id:
            continue
        member_paths = _minimal_member_roots(body_members)
        body_authoring, _ = _body_authoring(
            link_id=body_path,
            member_paths=member_paths,
        )
        if body_authoring == "aggregate":
            moving_aggregate_members.update(member_paths)
    if moving_aggregate_members:
        _validate_moving_aggregate_source_members(
            source_path,
            tuple(sorted(moving_aggregate_members)),
            expected_source=request.source_asset,
        )

    records: list[PrimRecordV1 | LinkRecordV1 | JointRecordV1] = []
    for body_path in sorted(members_by_body):
        member_paths = _minimal_member_roots(members_by_body[body_path])
        body_authoring, authored_body_path = _body_authoring(
            link_id=body_path,
            member_paths=member_paths,
        )
        body_candidate = candidate_for_body.get(body_path)
        is_projected_fixed_body = (
            fixed_assembly is not None and body_path == fixed_assembly.link_id
        )
        role = (
            "body"
            if is_projected_fixed_body
            else "fixed_parent"
            if body_candidate is None
            else body_candidate.role
        )
        if role.strip().lower() in _UNRESOLVED_ROLES:
            raise JointRiggerContractError(
                "stage2_link_role_unresolved",
                f"ready Stage 2 body {body_path} has no resolved first-class role",
            )
        body_source = None
        if not is_projected_fixed_body:
            body_source = body_candidate or _first_parent_candidate(
                body_path,
                candidate_by_key.values(),
            )
        axis = incoming_axis.get(body_path)
        if is_projected_fixed_body:
            assert fixed_assembly is not None
            representative = fixed_assembly.member_paths[0]
            link_evidence = {
                "body_prim_path": _prediction_evidence(
                    fixed_assembly.artifact,
                    prim_path=representative,
                    properties=("role", "is_articulation_candidate", "joint_type_hint"),
                    field="body_prim_path",
                    derivation=(
                        _AGGREGATE_BODY_DERIVATION
                        if body_authoring == "aggregate"
                        else _FIXED_BODY_DERIVATION
                    ),
                ),
                "role": _prediction_evidence(
                    fixed_assembly.artifact,
                    prim_path=representative,
                    properties=("role", "is_articulation_candidate", "joint_type_hint"),
                    field="role",
                ),
            }
        else:
            assert body_source is not None
            link_evidence = {
                "body_prim_path": _evidence(
                    candidate_artifact,
                    body_source,
                    prim_path=body_path,
                    properties=(
                        ("moving_part_prims",)
                        if body_candidate is not None
                        else ("fixed_parent_prim",)
                    ),
                    field="body_prim_path",
                    derivation=(
                        _AGGREGATE_BODY_DERIVATION
                        if body_authoring == "aggregate"
                        else _DERIVATION
                    ),
                ),
                "role": _evidence(
                    candidate_artifact,
                    body_source,
                    prim_path=body_path,
                    properties=(
                        ("role",)
                        if body_candidate is not None
                        else ("fixed_parent_prim",)
                    ),
                    field="role",
                ),
            }
        if axis is not None:
            assert body_source is not None
            link_evidence["axis_stage"] = _evidence(
                candidate_artifact,
                body_source,
                prim_path=body_path,
                properties=("motion_axis_world",),
                field="axis_stage",
            )
        records.append(
            LinkRecordV1(
                kind="link",
                link_id=body_path,
                body_prim_path=authored_body_path,
                body_authoring=body_authoring,
                role=role,
                axis_stage=axis,
                field_evidence=link_evidence,
                review_status="ready_for_rigger_input",
            )
        )
        for member_path in member_paths:
            membership_evidence = (
                _prediction_evidence(
                    fixed_assembly.artifact,
                    prim_path=member_path,
                    properties=("role", "is_articulation_candidate", "joint_type_hint"),
                    field="link_id",
                )
                if is_projected_fixed_body and fixed_assembly is not None
                else _evidence(
                    candidate_artifact,
                    cast(Stage2ArticulationCandidate, body_source),
                    prim_path=member_path,
                    properties=("moving_part_prims", "fixed_parent_prim"),
                    field="link_id",
                )
            )
            records.append(
                PrimRecordV1(
                    kind="prim",
                    prim_path=member_path,
                    link_id=body_path,
                    membership_evidence=membership_evidence,
                )
            )

    for key in sorted(candidate_by_key, key=str):
        candidate = candidate_by_key[key]
        plan = plan_by_key[key]
        topology = plan.topology
        body0_link = canonical_body0_by_key[key]
        field_evidence = {
            "motion_type": _evidence(
                candidate_artifact,
                candidate,
                prim_path=topology.body1,
                properties=("motion_type",),
                field="motion_type",
            ),
            "body0_link": _evidence(
                candidate_artifact,
                candidate,
                prim_path=topology.body0,
                properties=("fixed_parent_prim", "connectivity_evidence"),
                field="body0_link",
                derivation=(
                    _FIXED_ALIAS_DERIVATION
                    if body0_link != topology.body0
                    else _DERIVATION
                ),
            ),
            "body1_link": _evidence(
                candidate_artifact,
                candidate,
                prim_path=topology.body1,
                properties=("moving_part_prims", "connectivity_evidence"),
                field="body1_link",
            ),
        }
        if topology.axis_stage is None:
            raise JointRiggerContractError(
                "stage2_preflight_axis_unresolved",
                f"preflighted joint {topology.joint_id!r} has no stage-frame axis",
            )
        field_evidence["axis_stage"] = _evidence(
            candidate_artifact,
            candidate,
            prim_path=topology.body1,
            properties=("motion_axis_world", "axis_evidence"),
            field="axis_stage",
        )
        records.append(
            JointRecordV1(
                kind="joint",
                joint_id=topology.joint_id,
                body0_link=body0_link,
                body1_link=topology.body1,
                motion_type=topology.joint_type,
                axis_stage=topology.axis_stage,
                limit=_rebind_limit(
                    plan.limit,
                    artifact=candidate_artifact,
                    candidate=candidate,
                    prim_path=topology.body1,
                ),
                field_evidence=field_evidence,
                review_status="ready_for_rigger_input",
            )
        )

    prim_count = sum(isinstance(record, PrimRecordV1) for record in records)
    link_count = sum(isinstance(record, LinkRecordV1) for record in records)
    joint_count = sum(isinstance(record, JointRecordV1) for record in records)
    child_link_ids = {plan.topology.body1 for plan in plan_by_key.values()}
    source_identities = [request.source_asset, candidate_artifact]
    if prediction_artifact is not None:
        source_identities.append(prediction_artifact)
    return ArticulationContractV1(
        schema_version=ARTICULATION_CONTRACT_SCHEMA_VERSION,
        status="ready_for_rigger_input",
        articulation_roots=tuple(sorted(set(members_by_body) - child_link_ids)),
        source_identities=tuple(source_identities),
        records=tuple(records),
        summary=ContractSummaryV1(
            prim_count=prim_count,
            link_count=link_count,
            joint_count=joint_count,
            review_required_link_count=0,
            review_required_joint_count=0,
            diagnostic_count=0,
        ),
    )


def _load_stage1_predictions(
    path: Path,
) -> tuple[tuple[_Stage1Prediction, ...], ArtifactIdentityV1]:
    prediction_bytes = _read_stable_predictions_bytes(path)
    prediction_artifact = ArtifactIdentityV1(
        uri=str(path),
        root_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
    )
    predictions: list[_Stage1Prediction] = []
    seen_ids: set[str] = set()
    try:
        lines = prediction_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise JointRiggerContractError(
            "stage1_predictions_invalid",
            f"cannot decode Stage 1 predictions document {path}: {exc}",
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"invalid JSON in Stage 1 predictions document {path} at "
                f"line {line_number}: {exc}",
            ) from exc
        if not isinstance(row, dict):
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 prediction at line {line_number} must be an object",
            )
        prim_path = row.get("id")
        if not isinstance(prim_path, str) or not _is_absolute_prim_path(prim_path):
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 prediction at line {line_number} has an invalid id",
            )
        if prim_path in seen_ids:
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 predictions contain duplicate id {prim_path!r}",
            )
        seen_ids.add(prim_path)
        try:
            payload = unwrap_stage1_prediction_payload(row)
        except Exception as exc:
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"cannot unwrap Stage 1 prediction {prim_path!r}: {exc}",
            ) from exc
        raw_role = payload.get("role")
        if raw_role is not None and not isinstance(raw_role, str):
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 prediction {prim_path!r} has a non-string role",
            )
        predictions.append(_Stage1Prediction(prim_path=prim_path, payload=payload))

    return tuple(predictions), prediction_artifact


def _build_fixed_assembly_projection(
    predictions: Iterable[_Stage1Prediction],
    prediction_artifact: ArtifactIdentityV1,
) -> _FixedAssemblyProjection | None:
    member_paths: list[str] = []
    member_instance_ids: set[str] = set()
    for prediction in predictions:
        prim_path = prediction.prim_path
        payload = prediction.payload
        raw_role = payload.get("role")
        if not isinstance(raw_role, str) or raw_role.strip().lower() != "body":
            continue
        joint_type_hint = payload.get("joint_type_hint")
        normalized_joint_type = (
            joint_type_hint.strip().lower() if isinstance(joint_type_hint, str) else ""
        )
        if (
            payload.get("is_articulation_candidate") is not False
            or normalized_joint_type not in _FIXED_JOINT_HINTS
        ):
            raise JointRiggerContractError(
                "stage1_fixed_body_prediction_conflict",
                f"role=body prediction {prim_path!r} is not coherently fixed",
            )
        instance_id = canonical_link_instance_id(payload.get("instance_id"))
        if (
            not is_model_supplied_link_instance_id(payload)
            or not instance_id
            or instance_id in _UNRESOLVED_ROLES
        ):
            raise JointRiggerContractError(
                "stage1_fixed_body_membership_unresolved",
                f"role=body prediction {prim_path!r} lacks an explicit "
                "model-supplied physical-link instance_id",
            )
        member_instance_ids.add(instance_id)
        member_paths.append(prim_path)

    if not member_paths:
        # Stage 2 already carries an exact fixed parent for every accepted edge.
        # Predictions refine that parent only when they provide a coherent fixed
        # assembly; moving-only predictions remain valid bound source evidence.
        return None
    if len(member_instance_ids) != 1:
        raise JointRiggerContractError(
            "stage1_fixed_body_membership_ambiguous",
            "role=body predictions identify multiple physical links: "
            + ", ".join(sorted(member_instance_ids)),
        )
    parents = {_direct_parent(path) for path in member_paths}
    if len(parents) != 1:
        raise JointRiggerContractError(
            "stage1_fixed_assemblies_ambiguous",
            "role=body predictions span multiple fixed assemblies: "
            + ", ".join(sorted(parents)),
        )
    parent_path = next(iter(parents))
    if len(member_paths) > 1 and not _is_absolute_prim_path(parent_path):
        raise JointRiggerContractError(
            "stage1_fixed_assemblies_ambiguous",
            "multi-member role=body predictions must share a concrete "
            "direct-parent prim",
        )
    ordered_members = tuple(sorted(member_paths))
    return _FixedAssemblyProjection(
        artifact=prediction_artifact,
        # One fixed member is already a valid existing rigid body. Multiple
        # siblings need one deterministic aggregate link rooted at their
        # shared direct parent.
        link_id=ordered_members[0] if len(ordered_members) == 1 else parent_path,
        parent_path=parent_path,
        member_paths=ordered_members,
    )


def _read_stable_predictions_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise JointRiggerContractError(
            "stage1_predictions_invalid",
            f"cannot open Stage 1 predictions document {path}: {exc}",
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 predictions document {path} must be a regular file",
            )
        if before.st_size > _MAX_PREDICTIONS_BYTES:
            raise JointRiggerContractError(
                "stage1_predictions_invalid",
                f"Stage 1 predictions document {path} exceeds "
                f"{_MAX_PREDICTIONS_BYTES} bytes",
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                raise JointRiggerContractError(
                    "stage1_predictions_mutated",
                    f"Stage 1 predictions document {path} changed while being read",
                )
            chunks.append(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            raise JointRiggerContractError(
                "stage1_predictions_mutated",
                f"Stage 1 predictions document {path} grew while being read",
            )
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_after != identity_before:
            raise JointRiggerContractError(
                "stage1_predictions_mutated",
                f"Stage 1 predictions document {path} changed while being read",
            )
        return b"".join(chunks)
    except JointRiggerContractError:
        raise
    except OSError as exc:
        raise JointRiggerContractError(
            "stage1_predictions_invalid",
            f"cannot read Stage 1 predictions document {path}: {exc}",
        ) from exc
    finally:
        os.close(descriptor)


def _validate_fixed_assembly_source_members(
    source_path: Path,
    fixed_assembly: _FixedAssemblyProjection,
    *,
    expected_source: ArtifactIdentityV1,
) -> None:
    """Prove projected fixed members are authorable prims in the bound source."""

    _validate_source_member_paths(
        source_path,
        fixed_assembly.member_paths,
        expected_source=expected_source,
        error_prefix="stage1_fixed_body",
        member_label="projected fixed-body member",
        require_root_authorship=len(fixed_assembly.member_paths) > 1,
    )


def _validate_moving_aggregate_source_members(
    source_path: Path,
    member_paths: tuple[str, ...],
    *,
    expected_source: ArtifactIdentityV1,
) -> None:
    """Prove every moving aggregate member is authorable in the bound source."""

    _validate_source_member_paths(
        source_path,
        member_paths,
        expected_source=expected_source,
        error_prefix="stage2_aggregate",
        member_label="moving aggregate member",
    )


def _validate_source_member_paths(
    source_path: Path,
    member_paths: tuple[str, ...],
    *,
    expected_source: ArtifactIdentityV1,
    error_prefix: str,
    member_label: str,
    require_root_authorship: bool = True,
) -> None:
    """Validate source-bound prims against their authoring prerequisites."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - preflight already needs OpenUSD
        raise JointRiggerContractError(
            "openusd_unavailable",
            "OpenUSD bindings are required to validate aggregate source members",
        ) from exc

    try:
        stage = Usd.Stage.Open(str(source_path))
    except Exception as exc:
        raise JointRiggerContractError(
            f"{error_prefix}_source_invalid",
            f"cannot open source stage {source_path}: {exc}",
        ) from exc
    if stage is None:
        raise JointRiggerContractError(
            f"{error_prefix}_source_invalid",
            f"cannot open source stage {source_path}",
        )

    root_layer = stage.GetRootLayer()
    try:
        for member_path in member_paths:
            prim = stage.GetPrimAtPath(member_path)
            if (
                not prim
                or not prim.IsValid()
                or not prim.IsActive()
                or not prim.IsDefined()
            ):
                raise JointRiggerContractError(
                    f"{error_prefix}_member_missing",
                    f"{member_label} must be active and defined in "
                    f"the bound source: {member_path}",
                )
            aggregate_incompatible = require_root_authorship and (
                prim.IsInstance() or prim.IsInstanceable()
            )
            if (
                aggregate_incompatible
                or prim.IsInstanceProxy()
                or prim.IsPrototype()
                or prim.IsInPrototype()
            ):
                raise JointRiggerContractError(
                    f"{error_prefix}_member_instance_unsupported",
                    f"{member_label} cannot use an unsupported instance or "
                    f"prototype prim: {member_path}",
                )
            if require_root_authorship:
                prim_stack = tuple(prim.GetPrimStack())
                if (
                    len(prim_stack) != 1
                    or prim_stack[0].layer.identifier != root_layer.identifier
                ):
                    raise JointRiggerContractError(
                        f"{error_prefix}_member_authorship_unsupported",
                        f"{member_label} must have exactly one root-layer "
                        f"PrimSpec: {member_path}",
                    )
                if prim_stack[0].path.ContainsPrimVariantSelection():
                    raise JointRiggerContractError(
                        f"{error_prefix}_member_variant_unsupported",
                        f"{member_label} cannot be authored inside a "
                        f"variant: {member_path}",
                    )
    finally:
        del stage

    current_source = identify_usd_artifact(source_path, uri=str(source_path))
    if current_source != expected_source:
        raise JointRiggerContractError(
            f"{error_prefix}_source_mutated",
            f"source USD changed while {member_label} membership was validated",
        )


def _canonical_parent_links(
    *,
    link_candidate_by_body: Mapping[str, Stage2ArticulationCandidate],
    plan_by_key: Mapping[
        tuple[str, str, str, tuple[float, float, float] | None],
        JointPlanV1,
    ],
    fixed_assembly: _FixedAssemblyProjection | None,
) -> dict[
    tuple[str, str, str, tuple[float, float, float] | None],
    str,
]:
    if fixed_assembly is None:
        return {key: plan.topology.body0 for key, plan in plan_by_key.items()}

    moving_link_ids = set(link_candidate_by_body)
    moving_members = {
        member
        for candidate in link_candidate_by_body.values()
        for member in candidate.moving_part_prims
    }
    overlap = sorted(set(fixed_assembly.member_paths) & moving_members)
    if fixed_assembly.link_id in moving_link_ids or overlap:
        detail = ", ".join(overlap or [fixed_assembly.link_id])
        raise JointRiggerContractError(
            "stage1_fixed_body_membership_conflict",
            "fixed body membership overlaps moving Stage 2 topology: " + detail,
        )

    result: dict[
        tuple[str, str, str, tuple[float, float, float] | None],
        str,
    ] = {}
    for key, plan in plan_by_key.items():
        body0 = plan.topology.body0
        if body0 in fixed_assembly.aliases:
            result[key] = fixed_assembly.link_id
        elif body0 in moving_link_ids:
            result[key] = body0
        else:
            raise JointRiggerContractError(
                "stage1_fixed_parent_alias_conflict",
                f"Stage 2 body0 {body0!r} is neither a fixed body alias nor "
                "an articulated parent link",
            )
    return result


def _direct_parent(path: str) -> str:
    return path.rsplit("/", 1)[0] or "/"


def _is_absolute_prim_path(value: str) -> bool:
    return (
        value.startswith("/")
        and value != "/"
        and not value.endswith("/")
        and "//" not in value
    )


def _read_candidate_bytes(
    path: Path,
    *,
    display_path: Path | None = None,
) -> bytes:
    displayed = path if display_path is None else display_path
    try:
        return path.read_bytes()
    except OSError as exc:
        raise JointRiggerContractError(
            "stage2_artifact_invalid",
            f"cannot read Stage 2 candidate document {displayed}: {exc}",
        ) from exc


def _load_candidates(
    candidate_bytes: bytes,
    *,
    path: Path,
) -> tuple[Stage2ArticulationCandidate, ...]:
    try:
        document = candidate_edge_authoring._parse_and_validate_document(
            candidate_bytes,
            path=path,
        )
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        raise JointRiggerContractError(
            "stage2_artifact_invalid",
            f"cannot validate Stage 2 candidate document {path}: {exc}",
        ) from exc
    return tuple(document.candidates)


def _build_preflight_request(
    *,
    source_path: Path,
    candidate_snapshot: Path,
    expected_candidates_sha256: str,
) -> JointRiggerInputV1:
    """Preflight the exact bytes parsed by this producer."""

    try:
        return cast(
            JointRiggerInputV1,
            build_stage2_candidate_edges_input(
                input_usd_path=source_path,
                articulation_candidates_path=candidate_snapshot,
                expected_articulation_candidates_sha256=expected_candidates_sha256,
            ),
        )
    except NoReadyJointCandidatesError:
        raise
    except JointRiggerContractError:
        raise
    except Exception as exc:
        raise JointRiggerContractError(
            "stage2_preflight_failed",
            f"Stage 2 candidate preflight failed: {exc}",
        ) from exc


def _write_private_candidate_snapshot(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            written = stream.write(payload)
            if written != len(payload):  # pragma: no cover - regular-file invariant
                raise OSError("short write while creating candidate snapshot")
    except OSError as exc:
        raise JointRiggerContractError(
            "stage2_snapshot_failed",
            f"cannot create private Stage 2 candidate snapshot: {exc}",
        ) from exc


def _topology_preflight_bytes(
    candidate_bytes: bytes,
    candidates: tuple[Stage2ArticulationCandidate, ...],
) -> bytes:
    if all(
        candidate.review_status != "ready_for_rigger_input"
        or len(candidate.moving_part_prims) == 1
        for candidate in candidates
    ):
        return candidate_bytes

    raw = json.loads(candidate_bytes)
    raw_candidates = raw["candidates"]
    projected_candidates: list[dict[str, Any]] = []
    for raw_candidate, candidate in zip(raw_candidates, candidates, strict=True):
        if (
            not isinstance(raw_candidate, dict)
            or raw_candidate.get("candidate_id") != candidate.candidate_id
        ):
            raise JointRiggerContractError(
                "stage2_artifact_invalid",
                "validated Stage 2 candidate order does not match its JSON document",
            )
        projected = dict(raw_candidate)
        if (
            candidate.review_status == "ready_for_rigger_input"
            and candidate.moving_part_prims
        ):
            projected["moving_part_prims"] = [candidate.moving_part_prims[0]]
        projected_candidates.append(projected)
    projected_document = dict(raw)
    projected_document["candidates"] = projected_candidates
    return json.dumps(
        projected_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _minimal_member_roots(paths: set[str]) -> tuple[str, ...]:
    ordered = sorted(paths, key=lambda path: (len(path.split("/")), path))
    roots: list[str] = []
    for path in ordered:
        if any(path == root or path.startswith(f"{root}/") for root in roots):
            continue
        roots.append(path)
    return tuple(sorted(roots))


def _body_authoring(
    *,
    link_id: str,
    member_paths: tuple[str, ...],
) -> tuple[BodyAuthoring, str]:
    if member_paths == (link_id,):
        return "existing", link_id
    parents = {path.rsplit("/", 1)[0] or "/" for path in member_paths}
    if len(parents) != 1:
        raise JointRiggerContractError(
            "stage2_aggregate_members_not_flat",
            f"link {link_id} requires aggregate members with one direct parent: "
            + ", ".join(member_paths),
        )
    parent = next(iter(parents))
    suffix = hashlib.sha256(link_id.encode("utf-8")).hexdigest()[:16]
    name = f"__JointAgent_{suffix}"
    body_path = f"/{name}" if parent == "/" else f"{parent}/{name}"
    if any(
        body_path == member
        or body_path.startswith(f"{member}/")
        or member.startswith(f"{body_path}/")
        for member in member_paths
    ):
        raise JointRiggerContractError(
            "stage2_aggregate_target_collision",
            f"deterministic aggregate target {body_path} overlaps a source member",
        )
    return "aggregate", body_path


def _topology_key(
    *,
    motion_type: str,
    body0: str,
    body1: str,
    axis: tuple[float, float, float] | None,
) -> tuple[str, str, str, tuple[float, float, float] | None]:
    return (motion_type, body0, body1, axis)


def _index_candidates(
    candidates: tuple[Stage2ArticulationCandidate, ...],
) -> Mapping[
    tuple[str, str, str, tuple[float, float, float] | None],
    Stage2ArticulationCandidate,
]:
    result: dict[
        tuple[str, str, str, tuple[float, float, float] | None],
        Stage2ArticulationCandidate,
    ] = {}
    for candidate in candidates:
        if candidate.motion_type not in _SUPPORTED_STAGE2_TYPES:
            raise JointRiggerContractError(
                "stage2_joint_type_unsupported",
                f"candidate {candidate.candidate_id!r} uses unsupported legacy "
                f"Stage 2 joint type {candidate.motion_type!r}",
            )
        if candidate.fixed_parent_prim is None or not candidate.moving_part_prims:
            raise JointRiggerContractError(
                "stage2_topology_incomplete",
                f"candidate {candidate.candidate_id!r} has unresolved endpoints",
            )
        if candidate.motion_axis_world is None:
            raise JointRiggerContractError(
                "stage2_axis_unresolved",
                f"candidate {candidate.candidate_id!r} has no stage-frame axis",
            )
        axis = (
            float(candidate.motion_axis_world[0]),
            float(candidate.motion_axis_world[1]),
            float(candidate.motion_axis_world[2]),
        )
        primary_moving_prim = candidate.moving_part_prims[0]
        key = _topology_key(
            motion_type=candidate.motion_type,
            body0=candidate.fixed_parent_prim,
            body1=primary_moving_prim,
            axis=axis,
        )
        if key in result:
            raise JointRiggerContractError(
                "stage2_topology_duplicate",
                f"candidates {result[key].candidate_id!r} and "
                f"{candidate.candidate_id!r} resolve to the same topology",
            )
        result[key] = candidate
    return result


def _index_plans(
    joints: tuple[JointPlanV1, ...],
) -> Mapping[
    tuple[str, str, str, tuple[float, float, float] | None],
    JointPlanV1,
]:
    result: dict[
        tuple[str, str, str, tuple[float, float, float] | None],
        JointPlanV1,
    ] = {}
    for joint in joints:
        topology = joint.topology
        key = _topology_key(
            motion_type=topology.joint_type,
            body0=topology.body0,
            body1=topology.body1,
            axis=topology.axis_stage,
        )
        if key in result:
            raise JointRiggerContractError(
                "stage2_preflight_topology_duplicate",
                f"preflight produced duplicate topology {key}",
            )
        result[key] = joint
    return result


def _require_unique_child_topologies(
    candidates: Mapping[
        tuple[str, str, str, tuple[float, float, float] | None],
        Stage2ArticulationCandidate,
    ],
) -> None:
    """Reject distinct topologies that would overwrite one child link."""

    candidate_for_body: dict[str, Stage2ArticulationCandidate] = {}
    for key, candidate in candidates.items():
        body1 = key[2]
        previous = candidate_for_body.get(body1)
        if previous is not None:
            raise JointRiggerContractError(
                "stage2_child_topology_ambiguous",
                f"candidates {previous.candidate_id!r} and "
                f"{candidate.candidate_id!r} target child body {body1}",
            )
        candidate_for_body[body1] = candidate


def _link_candidates_for_ready_subset(
    *,
    candidates: tuple[Stage2ArticulationCandidate, ...],
    ready_candidate_by_key: Mapping[
        tuple[str, str, str, tuple[float, float, float] | None],
        Stage2ArticulationCandidate,
    ],
) -> Mapping[str, Stage2ArticulationCandidate]:
    """Require every candidate-supplied parent link to be independently ready."""

    result = {key[2]: candidate for key, candidate in ready_candidate_by_key.items()}
    candidates_by_primary_body: dict[str, list[Stage2ArticulationCandidate]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        if candidate.moving_part_prims:
            candidates_by_primary_body[candidate.moving_part_prims[0]].append(candidate)

    for parent_body in sorted(key[1] for key in ready_candidate_by_key):
        matches = candidates_by_primary_body.get(parent_body, [])
        if len(matches) > 1:
            candidate_ids = ", ".join(
                repr(candidate.candidate_id)
                for candidate in sorted(matches, key=lambda item: item.candidate_id)
            )
            raise JointRiggerContractError(
                "stage2_parent_link_ambiguous",
                f"candidates {candidate_ids} all supply parent link {parent_body}",
            )
        if matches:
            parent_candidate = matches[0]
            if (
                parent_candidate.review_status != "ready_for_rigger_input"
                or parent_candidate.role.strip().lower() in _UNRESOLVED_ROLES
            ):
                detail = ", ".join(
                    sorted(parent_candidate.unresolved_reason_codes)
                    or [
                        "unresolved_role"
                        if parent_candidate.review_status == "ready_for_rigger_input"
                        else "review_status_not_ready"
                    ]
                )
                raise JointRiggerContractError(
                    "stage2_parent_link_requires_review",
                    f"parent link {parent_body!r} depends on candidate "
                    f"{parent_candidate.candidate_id!r}, which is not independently "
                    f"ready: {detail}",
                )
            if parent_body not in result:  # pragma: no cover - readiness invariant
                raise JointRiggerContractError(
                    "stage2_parent_link_requires_review",
                    f"parent link {parent_body!r} depends on candidate "
                    f"{parent_candidate.candidate_id!r}, which was not admitted to "
                    "the ready subset",
                )
    return result


def _first_parent_candidate(
    body_path: str,
    candidates: Iterable[Stage2ArticulationCandidate],
) -> Stage2ArticulationCandidate:
    matches = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.fixed_parent_prim == body_path
        ),
        key=lambda candidate: candidate.candidate_id,
    )
    if not matches:
        raise JointRiggerContractError(
            "stage2_parent_evidence_missing",
            f"no Stage 2 candidate supplies parent evidence for {body_path}",
        )
    return matches[0]


def _evidence(
    artifact: ArtifactIdentityV1,
    candidate: Stage2ArticulationCandidate,
    *,
    prim_path: str,
    properties: tuple[str, ...],
    field: str,
    derivation: str = _DERIVATION,
) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="accepted_manifest",
        artifact=artifact,
        prim_path=prim_path,
        properties=properties,
        derivation=derivation,
        evidence=(
            f"Validated Stage 2 candidate {candidate.candidate_id!r} supplies "
            f"first-class {field}."
        ),
    )


def _prediction_evidence(
    artifact: ArtifactIdentityV1,
    *,
    prim_path: str,
    properties: tuple[str, ...],
    field: str,
    derivation: str = _FIXED_BODY_DERIVATION,
) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="accepted_manifest",
        artifact=artifact,
        prim_path=prim_path,
        properties=properties,
        derivation=derivation,
        evidence=(
            f"Coherent Stage 1 role=body prediction supplies first-class {field}."
        ),
    )


def _rebind_limit(
    limit: JointLimitV1 | None,
    *,
    artifact: ArtifactIdentityV1,
    candidate: Stage2ArticulationCandidate,
    prim_path: str,
) -> JointLimitV1 | None:
    if limit is None:
        return None
    return JointLimitV1(
        lower=limit.lower,
        upper=limit.upper,
        unit=limit.unit,
        provenance=_evidence(
            artifact,
            candidate,
            prim_path=prim_path,
            properties=("lower_limit", "upper_limit", "limit_unit"),
            field="limit",
        ),
    )


__all__ = ["build_articulation_contract_from_stage2"]
