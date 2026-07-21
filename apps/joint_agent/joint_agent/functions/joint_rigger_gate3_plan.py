# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit complete evidence-backed static Gate 3 physics plans.

Topology is rebuilt from the first-class contract. Physics plans must cover the
same links and joints and supply rigid bodies, colliders, mass/inertia, state,
and control facts required by the frozen static profiles. Joint anchors remain
optional because neither frozen Gate 3 profile requires an authored anchor.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from world_understanding.functions.physics.joint_rigger import (
    INPUT_SCHEMA_VERSION_V2,
    PLAN_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION_V2,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    FieldProvenanceV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerInputV2,
    JointRiggerPlanV1,
    JointRiggerPlanV2,
    RigidBodyPlanV1,
    canonical_sha256,
)

from joint_agent.functions.articulation_contract import (
    ArticulationContractV1,
    JointRecordV1,
    LinkRecordV1,
)
from joint_agent.functions.joint_rigger_contract_bridge import (
    build_joint_rigger_input_from_contract,
)

_ROOT_DERIVATION = "unique_first_class_link_graph_root"
_MULTI_ROOT_DERIVATION = "first_class_link_graph_component_root"
_VALUE_TOLERANCE = 1e-6
GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION: Literal[
    "joint-agent-gate3-physics-plan-envelope-v1"
] = "joint-agent-gate3-physics-plan-envelope-v1"


class Gate3PhysicsPlanEnvelopeV1(BaseModel):
    """Exact source and contract revision authorized for one physics plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["joint-agent-gate3-physics-plan-envelope-v1"]
    source_asset: ArtifactIdentityV1
    contract_artifact: ArtifactIdentityV1
    plan: JointRiggerPlanV1


type JointRiggerGate3Plan = JointRiggerPlanV1 | JointRiggerPlanV2


def build_gate3_joint_rigger_input_from_contract(
    contract: ArticulationContractV1,
    *,
    contract_artifact: ArtifactIdentityV1,
    source_asset: ArtifactIdentityV1,
    physics_plan: Gate3PhysicsPlanEnvelopeV1,
    physics_plan_artifact: ArtifactIdentityV1,
) -> JointRiggerInputV1 | JointRiggerInputV2:
    """Join contract-owned topology with complete, separately bound physics facts.

    The topology always comes from the first-class contract. ``physics_plan`` is
    an evidence container for rigid bodies, colliders, mass/inertia, joint
    state/control, limits, anchors, mimic relations, and an optional root. It is
    admitted only when its canonical identity, topology projection, coverage,
    Gate 3-required facts, and evidence lineage all match.
    """

    if not isinstance(contract, ArticulationContractV1):
        raise TypeError("contract must be an ArticulationContractV1")
    if not isinstance(contract_artifact, ArtifactIdentityV1):
        raise TypeError("contract_artifact must be an ArtifactIdentityV1")
    if not isinstance(source_asset, ArtifactIdentityV1):
        raise TypeError("source_asset must be an ArtifactIdentityV1")
    if not isinstance(physics_plan, Gate3PhysicsPlanEnvelopeV1):
        raise TypeError("physics_plan must be a Gate3PhysicsPlanEnvelopeV1")
    if not isinstance(physics_plan_artifact, ArtifactIdentityV1):
        raise TypeError("physics_plan_artifact must be an ArtifactIdentityV1")
    if physics_plan_artifact.root_sha256 != canonical_sha256(physics_plan):
        raise JointRiggerContractError(
            "physics_plan_identity_mismatch",
            "physics_plan_artifact root_sha256 does not match the canonical "
            "source/contract-bound plan envelope",
        )
    if physics_plan.source_asset != source_asset:
        raise JointRiggerContractError(
            "physics_plan_source_identity_mismatch",
            "physics plan envelope is not bound to the current source asset",
        )
    if physics_plan.contract_artifact != contract_artifact:
        raise JointRiggerContractError(
            "physics_plan_contract_identity_mismatch",
            "physics plan envelope is not bound to the current contract artifact",
        )
    supplied_plan = physics_plan.plan

    topology_request = build_joint_rigger_input_from_contract(
        contract,
        contract_artifact=contract_artifact,
        source_asset=source_asset,
    )
    topology_joints = {
        joint.topology.joint_id: joint for joint in topology_request.plan.joints
    }
    physics_joints = {joint.topology.joint_id: joint for joint in supplied_plan.joints}
    _require_exact_keys(
        expected=set(topology_joints),
        actual=set(physics_joints),
        code="physics_joint_coverage_mismatch",
        label="joint IDs",
    )

    merged_joints = tuple(
        _merge_joint_plan(topology_joints[joint_id], physics_joints[joint_id])
        for joint_id in sorted(topology_joints)
    )
    _require_gate3_mimic_references(merged_joints)
    links = {
        record.link_id: record
        for record in contract.records
        if isinstance(record, LinkRecordV1)
    }
    expected_body_paths = {link.body_prim_path for link in links.values()}
    bodies = {body.prim_path: body for body in supplied_plan.rigid_bodies}
    _require_exact_keys(
        expected=expected_body_paths,
        actual=set(bodies),
        code="physics_body_coverage_mismatch",
        label="link body paths",
    )
    for path in sorted(bodies):
        _require_gate3_body_facts(bodies[path])

    try:
        if isinstance(topology_request, JointRiggerInputV2):
            if supplied_plan.articulation_root is not None:
                raise JointRiggerContractError(
                    "articulation_root_conflict",
                    "a singleton physics-plan articulation_root cannot represent a "
                    "multi-root articulation contract",
                )
            roots = _resolve_articulation_roots(
                contract,
                links=links,
                contract_artifact=contract_artifact,
            )
            merged_plan: JointRiggerGate3Plan = JointRiggerPlanV2(
                schema_version=PLAN_SCHEMA_VERSION_V2,
                joints=merged_joints,
                rigid_bodies=supplied_plan.rigid_bodies,
                articulation_roots=roots,
            )
        else:
            root = _resolve_articulation_root(
                contract,
                links=links,
                proposed=supplied_plan.articulation_root,
                contract_artifact=contract_artifact,
            )
            merged_plan = JointRiggerPlanV1(
                schema_version=PLAN_SCHEMA_VERSION,
                joints=merged_joints,
                rigid_bodies=tuple(bodies.values()),
                articulation_root=root,
            )
    except ValidationError as exc:
        raise JointRiggerContractError(
            "physics_plan_merge_invalid",
            f"merged Gate 3 physics plan is invalid: {exc}",
        ) from exc
    _validate_physics_evidence_lineage(
        merged_plan,
        allowed_artifacts={
            *contract.source_identities,
            contract_artifact,
        },
        source_asset=source_asset,
    )
    if isinstance(topology_request, JointRiggerInputV2):
        assert isinstance(merged_plan, JointRiggerPlanV2)
        return JointRiggerInputV2(
            schema_version=INPUT_SCHEMA_VERSION_V2,
            source_asset=topology_request.source_asset,
            plan=merged_plan,
            rigid_links=topology_request.rigid_links,
            legacy_component_names=None,
            conflict_policy="error",
        )
    assert isinstance(merged_plan, JointRiggerPlanV1)
    return JointRiggerInputV1(
        schema_version=topology_request.schema_version,
        source_asset=topology_request.source_asset,
        plan=merged_plan,
        legacy_component_names=None,
        conflict_policy="error",
    )


def _merge_joint_plan(
    topology_joint: JointPlanV1,
    physics_joint: JointPlanV1,
) -> JointPlanV1:
    expected = topology_joint.topology
    actual = physics_joint.topology
    expected_facts = (
        expected.joint_type,
        expected.body0,
        expected.body1,
        expected.axis_stage,
    )
    actual_facts = (
        actual.joint_type,
        actual.body0,
        actual.body1,
        actual.axis_stage,
    )
    if actual_facts != expected_facts:
        raise JointRiggerContractError(
            "physics_topology_mismatch",
            f"physics plan topology conflicts with contract joint "
            f"{expected.joint_id!r}",
        )

    if expected.joint_type == "spherical":
        scalar_fields = tuple(
            field
            for field in (
                "limit",
                "joint_friction",
                "drive",
                "state",
                "mimic",
            )
            if getattr(physics_joint, field) is not None
        )
        if scalar_fields:
            raise JointRiggerContractError(
                "spherical_scalar_physics_unsupported",
                f"spherical joint {expected.joint_id!r} cannot carry scalar "
                f"physics fields: {', '.join(scalar_fields)}",
            )

    # Contract evidence is authoritative when both plans carry the same limit;
    # the physics copy is checked for fact equality below and is not used as a
    # second independently editable source of truth.
    limit = topology_joint.limit or physics_joint.limit
    if (
        topology_joint.limit is not None
        and physics_joint.limit is not None
        and _limit_facts(topology_joint.limit) != _limit_facts(physics_joint.limit)
    ):
        raise JointRiggerContractError(
            "physics_limit_conflict",
            f"physics plan limit conflicts with contract joint {expected.joint_id!r}",
        )
    if topology_joint.limit is None and physics_joint.limit is not None:
        _require_source_backed_optional_fact(
            physics_joint.limit.provenance,
            joint_id=expected.joint_id,
            field="limit",
        )
    if physics_joint.anchor is not None:
        _require_source_backed_optional_fact(
            physics_joint.anchor.provenance,
            joint_id=expected.joint_id,
            field="anchor",
        )
    if expected.joint_type != "spherical":
        if physics_joint.state is None:
            raise JointRiggerContractError(
                "joint_state_evidence_missing",
                f"joint {expected.joint_id!r} requires explicit state evidence",
            )
        if physics_joint.state.position != 0.0 or physics_joint.state.velocity != 0.0:
            raise JointRiggerContractError(
                "unsafe_new_joint_state",
                f"joint {expected.joint_id!r} requires an explicit zero rest state",
            )
        _require_position_inside_limit(
            physics_joint.state.position,
            limit=limit,
            joint_id=expected.joint_id,
            code="joint_state_outside_limits",
        )
        if physics_joint.mimic is not None:
            _require_source_backed_optional_fact(
                physics_joint.mimic.provenance,
                joint_id=expected.joint_id,
                field="mimic",
            )
        if physics_joint.drive is None and physics_joint.mimic is None:
            raise JointRiggerContractError(
                "joint_control_evidence_missing",
                f"joint {expected.joint_id!r} requires drive or mimic evidence",
            )
        if physics_joint.drive is not None and any(
            value not in (None, 0.0)
            for value in (
                physics_joint.drive.target_position,
                physics_joint.drive.target_velocity,
            )
        ):
            raise JointRiggerContractError(
                "unsafe_new_drive_target",
                f"joint {expected.joint_id!r} requires zero or absent drive targets",
            )
        if (
            physics_joint.drive is not None
            and physics_joint.drive.target_position is not None
        ):
            _require_position_inside_limit(
                physics_joint.drive.target_position,
                limit=limit,
                joint_id=expected.joint_id,
                code="drive_target_outside_limits",
            )
        if physics_joint.mimic is not None and physics_joint.joint_friction is not None:
            raise JointRiggerContractError(
                "mimic_schema_conflict",
                f"mimic joint {expected.joint_id!r} cannot carry joint friction",
            )

    return JointPlanV1(
        topology=expected,
        limit=limit,
        anchor=physics_joint.anchor,
        joint_friction=physics_joint.joint_friction,
        drive=physics_joint.drive,
        state=physics_joint.state,
        mimic=physics_joint.mimic,
    )


def _require_gate3_mimic_references(joints: tuple[JointPlanV1, ...]) -> None:
    by_id = {joint.topology.joint_id: joint for joint in joints}
    for joint_id, joint in by_id.items():
        if joint.mimic is None:
            continue
        reference_id = joint.mimic.reference_joint_id
        reference = by_id.get(reference_id)
        if reference is None:
            raise JointRiggerContractError(
                "mimic_reference_missing",
                f"joint {joint_id!r} references absent mimic joint {reference_id!r}",
            )
        if joint.topology.joint_type != "revolute" or (
            reference.topology.joint_type != "revolute"
        ):
            raise JointRiggerContractError(
                "mimic_not_applicable",
                f"mimic joint {joint_id!r} and reference {reference_id!r} must be revolute",
            )
        if reference.mimic is not None:
            raise JointRiggerContractError(
                "mimic_chain_unsupported",
                f"mimic reference {reference_id!r} cannot itself be a mimic",
            )
        axis = joint.topology.axis_stage
        reference_axis = reference.topology.axis_stage
        if (
            axis is None
            or reference_axis is None
            or math.fsum(
                left * right for left, right in zip(axis, reference_axis, strict=True)
            )
            < (1.0 - _VALUE_TOLERANCE)
        ):
            raise JointRiggerContractError(
                "mimic_axis_mismatch",
                f"mimic and reference signed axes differ at {joint_id!r}",
            )
        _require_complete_zero_spanning_limit(joint)
        _require_complete_zero_spanning_limit(reference)


def _require_complete_zero_spanning_limit(joint: JointPlanV1) -> None:
    limit = joint.limit
    joint_id = joint.topology.joint_id
    if limit is None or limit.lower is None or limit.upper is None:
        raise JointRiggerContractError(
            "mimic_limits_incomplete",
            f"mimic requires complete finite limits at {joint_id!r}",
        )
    if limit.lower > 0.0 or limit.upper < 0.0 or limit.lower == limit.upper:
        raise JointRiggerContractError(
            "mimic_limits_incompatible",
            f"mimic limits must span the zero rest pose at {joint_id!r}",
        )


def _require_position_inside_limit(
    position: float,
    *,
    limit: JointLimitV1 | None,
    joint_id: str,
    code: str,
) -> None:
    if limit is None:
        return
    if limit.lower is not None and position < limit.lower - _VALUE_TOLERANCE:
        raise JointRiggerContractError(
            code,
            f"position {position} is below {limit.lower} at {joint_id!r}",
        )
    if limit.upper is not None and position > limit.upper + _VALUE_TOLERANCE:
        raise JointRiggerContractError(
            code,
            f"position {position} is above {limit.upper} at {joint_id!r}",
        )


def _require_source_backed_optional_fact(
    provenance: FieldProvenanceV1,
    *,
    joint_id: str,
    field: str,
) -> None:
    if provenance.source != "accepted_manifest" or provenance.artifact is None:
        raise JointRiggerContractError(
            "optional_physics_fact_not_source_backed",
            f"joint {joint_id!r} physics-only {field} requires source-backed evidence",
        )


def _limit_facts(limit: JointLimitV1) -> tuple[float | None, float | None, str]:
    return (limit.lower, limit.upper, limit.unit)


def _require_gate3_body_facts(body: RigidBodyPlanV1) -> None:
    if body.mass is None:
        raise JointRiggerContractError(
            "mass_evidence_missing",
            f"rigid body {body.prim_path} requires explicit mass and inertia",
        )
    if body.mass.principal_axes is None:
        raise JointRiggerContractError(
            "principal_axes_evidence_missing",
            f"rigid body {body.prim_path} requires principal axes with inertia",
        )
    if not body.colliders:
        raise JointRiggerContractError(
            "collider_evidence_missing",
            f"rigid body {body.prim_path} requires at least one collider",
        )


def _resolve_articulation_root(
    contract: ArticulationContractV1,
    *,
    links: Mapping[str, LinkRecordV1],
    proposed: ArticulationRootPlanV1 | None,
    contract_artifact: ArtifactIdentityV1,
) -> ArticulationRootPlanV1:
    child_link_ids = {
        record.body1_link
        for record in contract.records
        if isinstance(record, JointRecordV1)
    }
    root_link_ids = sorted(set(links) - child_link_ids)
    if len(root_link_ids) != 1:
        raise JointRiggerContractError(
            "articulation_root_ambiguous",
            "the first-class link graph must have exactly one root; got "
            f"{root_link_ids}",
        )
    root_link = links[root_link_ids[0]]
    if proposed is not None:
        if proposed.prim_path != root_link.body_prim_path:
            raise JointRiggerContractError(
                "articulation_root_conflict",
                f"physics plan root {proposed.prim_path} conflicts with unique "
                f"contract root {root_link.body_prim_path}",
            )
        return proposed
    return ArticulationRootPlanV1(
        prim_path=root_link.body_prim_path,
        provenance=FieldProvenanceV1(
            source="accepted_manifest",
            artifact=contract_artifact,
            prim_path=root_link.body_prim_path,
            properties=(
                f"link:{root_link.link_id}.body_prim_path",
                "joint_graph.unique_root",
            ),
            derivation=_ROOT_DERIVATION,
            evidence=(
                f"Link {root_link.link_id!r} is the unique node with no "
                "incoming first-class joint."
            ),
        ),
    )


def _resolve_articulation_roots(
    contract: ArticulationContractV1,
    *,
    links: Mapping[str, LinkRecordV1],
    contract_artifact: ArtifactIdentityV1,
) -> tuple[ArticulationRootPlanV1, ...]:
    return tuple(
        ArticulationRootPlanV1(
            prim_path=links[link_id].body_prim_path,
            provenance=FieldProvenanceV1(
                source="accepted_manifest",
                artifact=contract_artifact,
                prim_path=links[link_id].body_prim_path,
                properties=(
                    f"link:{link_id}.body_prim_path",
                    "joint_graph.component_root",
                ),
                derivation=_MULTI_ROOT_DERIVATION,
                evidence=(
                    f"Link {link_id!r} is a declared component root with no "
                    "incoming first-class joint."
                ),
            ),
        )
        for link_id in contract.articulation_roots
    )


def _validate_physics_evidence_lineage(
    plan: JointRiggerGate3Plan,
    *,
    allowed_artifacts: set[ArtifactIdentityV1],
    source_asset: ArtifactIdentityV1,
) -> None:
    # Topology provenance is intentionally excluded: the merged topology was
    # freshly rebuilt by build_joint_rigger_input_from_contract and is already
    # bound to contract_artifact. Only separately supplied physics facts are
    # audited here.
    for label, provenance in _iter_physics_provenance(plan):
        if provenance.source == "template_default":
            raise JointRiggerContractError(
                "unapproved_physics_evidence",
                f"{label} uses template_default evidence",
            )
        if provenance.artifact is None:
            raise JointRiggerContractError(
                "physics_evidence_artifact_missing",
                f"{label} does not bind its evidence to a current artifact",
            )
        if provenance.artifact not in allowed_artifacts:
            raise JointRiggerContractError(
                "undeclared_physics_evidence_artifact",
                f"{label} references undeclared artifact {provenance.artifact.uri}",
            )
        if (
            provenance.source == "owner_approved_plan"
            and provenance.artifact != source_asset
        ):
            raise JointRiggerContractError(
                "physics_evidence_source_identity_mismatch",
                f"{label} is not bound to the current source asset",
            )


def _iter_physics_provenance(
    plan: JointRiggerGate3Plan,
) -> Iterable[tuple[str, FieldProvenanceV1]]:
    if isinstance(plan, JointRiggerPlanV1):
        if plan.articulation_root is not None:
            yield "articulation_root", plan.articulation_root.provenance
    else:
        for root in plan.articulation_roots:
            yield f"articulation_roots[{root.prim_path}]", root.provenance
    for body in plan.rigid_bodies:
        prefix = f"rigid_bodies[{body.prim_path}]"
        yield prefix, body.provenance
        if body.mass is not None:
            yield f"{prefix}.mass", body.mass.provenance
        for collider in body.colliders:
            yield f"{prefix}.colliders[{collider.prim_path}]", collider.provenance
    for joint in plan.joints:
        prefix = f"joints[{joint.topology.joint_id}]"
        for field in (
            "limit",
            "anchor",
            "joint_friction",
            "drive",
            "state",
            "mimic",
        ):
            value = getattr(joint, field)
            if value is not None:
                yield f"{prefix}.{field}", value.provenance


def _require_exact_keys(
    *,
    expected: set[str],
    actual: set[str],
    code: str,
    label: str,
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise JointRiggerContractError(
        code,
        f"{label} differ; missing={missing}, extra={extra}",
    )


__all__ = [
    "GATE3_PHYSICS_PLAN_ENVELOPE_SCHEMA_VERSION",
    "Gate3PhysicsPlanEnvelopeV1",
    "build_gate3_joint_rigger_input_from_contract",
]
