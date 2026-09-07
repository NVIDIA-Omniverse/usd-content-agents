# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from content_agent_workflows.articulation import (
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationDecisionPatch,
    EmbeddedArticulationError,
    EmbeddedArticulationGroup,
    EmbeddedArticulationHumanReviewPolicy,
    EmbeddedArticulationJoint,
    EmbeddedArticulationMembership,
    select_embedded_articulation_human_review_policy,
)
from content_agent_workflows.articulation.embedded_decision import (
    _validate_canonical_graph,
)
from content_agent_workflows.articulation.models import ArticulationRunState
from content_agent_workflows.common.embedded_domain_decision import (
    EmbeddedDomainEvidence,
    ProviderNeutralEvidenceRecord,
)

_MISSING_MEMBERSHIP_ROWS = object()


def _human_policy(**gates: bool) -> EmbeddedArticulationHumanReviewPolicy:
    selected = {
        "task_policy_requires_human": False,
        "ambiguity": False,
        "unsupported_facts": False,
        "contradictory_evidence": False,
    }
    selected.update(gates)
    return select_embedded_articulation_human_review_policy(
        task_policy_requires_human=selected["task_policy_requires_human"],
        ambiguity=selected["ambiguity"],
        unsupported_facts=selected["unsupported_facts"],
        contradictory_evidence=selected["contradictory_evidence"],
    )


def _graph() -> EmbeddedArticulationCanonicalGraph:
    return EmbeddedArticulationCanonicalGraph(
        graph_id="current-policy-graph",
        source_sha256="a" * 64,
        source_dependency_bundle_sha256="b" * 64,
        source_member_prims=("/World/Base", "/World/Drawer"),
        authoritative_owner_prims=("/World/Base", "/World/Drawer"),
        candidate_ids=("drawer_joint",),
        groups=(
            EmbeddedArticulationGroup(
                group_id="base",
                authoritative_owner_prim="/World/Base",
                member_prims=("/World/Base",),
                role="fixed base",
                role_state="source_backed",
                evidence_ids=("joint-authoritative-owner-inspection",),
            ),
            EmbeddedArticulationGroup(
                group_id="drawer",
                authoritative_owner_prim="/World/Drawer",
                member_prims=("/World/Drawer",),
                role="moving drawer",
                role_state="source_backed",
                evidence_ids=("joint-authoritative-owner-inspection",),
            ),
        ),
        memberships=(
            EmbeddedArticulationMembership(
                member_prim="/World/Base",
                authoritative_owner_prim="/World/Base",
                group_id="base",
                disposition="independent_motion",
                state="source_backed",
                evidence_ids=("joint-source-member-inspection",),
            ),
            EmbeddedArticulationMembership(
                member_prim="/World/Drawer",
                authoritative_owner_prim="/World/Drawer",
                group_id="drawer",
                disposition="independent_motion",
                state="source_backed",
                evidence_ids=("joint-source-member-inspection",),
            ),
        ),
        joints=(
            EmbeddedArticulationJoint(
                joint_id="drawer_joint",
                body0_owner_prim="/World/Base",
                body1_owner_prim="/World/Drawer",
                body0_role="fixed base",
                body1_role="moving drawer",
                role_state="source_backed",
                joint_type="prismatic",
                endpoint_state="source_backed",
                type_state="source_backed",
                axis="x",
                axis_state="source_backed",
                lower_limit=0.0,
                upper_limit=0.5,
                limit_unit="meters",
                limit_state="source_backed",
                frame_policy="body1_world_origin",
                frame_state="source_backed",
                evidence_ids=("joint-scene-inspection",),
            ),
        ),
    )


def _inspection_evidence(
    membership_rows: object = _MISSING_MEMBERSHIP_ROWS,
    *,
    include_membership_policy: bool | None = None,
    source_member_prims: tuple[str, ...] = ("/World/Base", "/World/Drawer"),
    authoritative_owner_prims: tuple[str, ...] = ("/World/Base", "/World/Drawer"),
) -> EmbeddedDomainEvidence:
    ownership_facts: dict[str, JsonValue] = {
        "authoritative_owner_prims": list(authoritative_owner_prims)
    }
    if membership_rows is not _MISSING_MEMBERSHIP_ROWS:
        ownership_facts["membership_rows"] = cast(JsonValue, membership_rows)
    if include_membership_policy is True or (
        include_membership_policy is None
        and membership_rows is not _MISSING_MEMBERSHIP_ROWS
    ):
        ownership_facts["membership_policy"] = "retained-explicit-membership-v1"
    return EmbeddedDomainEvidence.model_construct(
        records=(
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-source-member-inspection",
                evidence_type="inspection",
                status="available",
                summary="Exact source members.",
                facts={"source_member_prims": list(source_member_prims)},
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-authoritative-owner-inspection",
                evidence_type="inspection",
                status="available",
                summary="Exact ownership rows.",
                facts=ownership_facts,
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-scene-inspection",
                evidence_type="inspection",
                status="available",
                summary="Exact scene evidence.",
                facts={"complete": True},
            ),
        )
    )


def test_current_policy_selects_no_human_when_no_live_gate_exists() -> None:
    assert _human_policy() == (
        EmbeddedArticulationHumanReviewPolicy(status="not_requested")
    )


def test_current_policy_fails_closed_for_every_supported_live_gate() -> None:
    policy = _human_policy(
        task_policy_requires_human=True,
        ambiguity=True,
        unsupported_facts=True,
        contradictory_evidence=True,
    )
    assert policy.status == "human_required"
    assert policy.reasons == (
        "task_policy",
        "ambiguity",
        "unsupported_facts",
        "contradictory_evidence",
    )


@pytest.mark.parametrize(
    ("argument", "reason"),
    (
        ("task_policy_requires_human", "task_policy"),
        ("ambiguity", "ambiguity"),
        ("unsupported_facts", "unsupported_facts"),
        ("contradictory_evidence", "contradictory_evidence"),
    ),
)
def test_current_policy_preserves_each_individual_live_gate(
    argument: str,
    reason: str,
) -> None:
    policy = _human_policy(**{argument: True})
    assert policy.status == "human_required"
    assert policy.reasons == (reason,)


def test_current_policy_does_not_change_legacy_patch_compatibility() -> None:
    legacy = EmbeddedArticulationDecisionPatch(
        schema_version="content-agent-workflows.embedded-articulation-decision-patch.v2",
        identity_digest="c" * 64,
        evidence_digest="d" * 64,
        canonical_graph=_graph(),
        rationale="Preserve the frozen legacy human gate.",
    )
    assert legacy.effective_outer_review_disposition == "accept"
    assert legacy.effective_human_review == EmbeddedArticulationHumanReviewPolicy(
        status="human_required",
        reasons=("legacy_compatibility",),
    )

    current = EmbeddedArticulationDecisionPatch(
        identity_digest="c" * 64,
        evidence_digest="d" * 64,
        canonical_graph=_graph(),
        outer_review_disposition="accept",
        human_review=_human_policy(),
        rationale="No live task policy, ambiguity, or conflict requires a human gate.",
    )
    assert current.effective_human_review.status == "not_requested"

    mixed = current.model_dump(mode="json")
    mixed["schema_version"] = (
        "content-agent-workflows.embedded-articulation-decision-patch.v2"
    )
    with pytest.raises(ValidationError, match="v1/v2 articulation patches"):
        EmbeddedArticulationDecisionPatch.model_validate(mixed)


def test_current_graph_must_match_exact_inspected_membership_rows() -> None:
    evidence = _inspection_evidence(
        [
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Base",
                "disposition": "independent_motion",
            },
            {
                "member_prim": "/World/Drawer",
                "authoritative_owner_prim": "/World/Drawer",
                "disposition": "independent_motion",
            },
        ]
    )
    state = ArticulationRunState.model_construct(
        source_sha256="a" * 64,
        source_dependency_bundle_sha256="b" * 64,
    )
    graph = _graph()
    _validate_canonical_graph(
        graph,
        state=state,
        evidence=evidence,
        capabilities=EmbeddedArticulationCapabilityLimits(),
    )

    forged = graph.model_copy(
        update={
            "memberships": (
                graph.memberships[0].model_copy(
                    update={"disposition": "explicit_fixed"}
                ),
                graph.memberships[1],
            )
        }
    )
    with pytest.raises(
        EmbeddedArticulationError,
        match="membership differs from exact inspection rows",
    ):
        _validate_canonical_graph(
            forged,
            state=state,
            evidence=evidence,
            capabilities=EmbeddedArticulationCapabilityLimits(),
        )


def test_legacy_inspection_without_membership_rows_remains_compatible() -> None:
    _validate_canonical_graph(
        _graph(),
        state=ArticulationRunState.model_construct(
            source_sha256="a" * 64,
            source_dependency_bundle_sha256="b" * 64,
        ),
        evidence=_inspection_evidence(),
        capabilities=EmbeddedArticulationCapabilityLimits(),
    )


@pytest.mark.parametrize(
    "evidence",
    (
        _inspection_evidence(
            [
                {
                    "member_prim": "/World/Base",
                    "authoritative_owner_prim": "/World/Base",
                    "disposition": "independent_motion",
                },
                {
                    "member_prim": "/World/Drawer",
                    "authoritative_owner_prim": "/World/Drawer",
                    "disposition": "independent_motion",
                },
            ],
            include_membership_policy=False,
        ),
        _inspection_evidence(include_membership_policy=True),
    ),
)
def test_current_membership_policy_and_rows_cannot_be_separated(
    evidence: EmbeddedDomainEvidence,
) -> None:
    with pytest.raises(EmbeddedArticulationError, match="present together"):
        _validate_canonical_graph(
            _graph(),
            state=ArticulationRunState.model_construct(
                source_sha256="a" * 64,
                source_dependency_bundle_sha256="b" * 64,
            ),
            evidence=evidence,
            capabilities=EmbeddedArticulationCapabilityLimits(),
        )


def test_exact_membership_rows_reject_invalid_independent_ownership() -> None:
    evidence = _inspection_evidence(
        [
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Drawer",
                "disposition": "independent_motion",
            },
            {
                "member_prim": "/World/Drawer",
                "authoritative_owner_prim": "/World/Base",
                "disposition": "independent_motion",
            },
        ]
    )
    with pytest.raises(EmbeddedArticulationError, match="ownership semantics"):
        _validate_canonical_graph(
            _graph(),
            state=ArticulationRunState.model_construct(
                source_sha256="a" * 64,
                source_dependency_bundle_sha256="b" * 64,
            ),
            evidence=evidence,
            capabilities=EmbeddedArticulationCapabilityLimits(),
        )


def test_exact_membership_rows_reject_transitively_owned_owner() -> None:
    root = "/World/Root"
    owned = f"{root}/Owned"
    child = f"{owned}/Child"
    evidence = _inspection_evidence(
        [
            {
                "member_prim": root,
                "authoritative_owner_prim": root,
                "disposition": "independent_motion",
            },
            {
                "member_prim": owned,
                "authoritative_owner_prim": root,
                "disposition": "co_rigid",
            },
            {
                "member_prim": child,
                "authoritative_owner_prim": owned,
                "disposition": "co_rigid",
            },
        ],
        source_member_prims=(root, owned, child),
        authoritative_owner_prims=(root, owned),
    )
    with pytest.raises(EmbeddedArticulationError, match="transitively owned"):
        _validate_canonical_graph(
            _graph(),
            state=ArticulationRunState.model_construct(
                source_sha256="a" * 64,
                source_dependency_bundle_sha256="b" * 64,
            ),
            evidence=evidence,
            capabilities=EmbeddedArticulationCapabilityLimits(),
        )


@pytest.mark.parametrize(
    "membership_rows",
    (
        "not-a-sequence",
        [None],
        [
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Base",
                "disposition": "unsupported",
            }
        ],
        [
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Base",
                "disposition": "independent_motion",
            },
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Drawer",
                "disposition": "co_rigid",
            },
        ],
        [
            {
                "member_prim": "/World/Base",
                "authoritative_owner_prim": "/World/Base",
                "disposition": "independent_motion",
            }
        ],
    ),
)
def test_current_inspection_rejects_malformed_membership_rows(
    membership_rows: object,
) -> None:
    with pytest.raises(EmbeddedArticulationError, match="membership rows"):
        _validate_canonical_graph(
            _graph(),
            state=ArticulationRunState.model_construct(
                source_sha256="a" * 64,
                source_dependency_bundle_sha256="b" * 64,
            ),
            evidence=_inspection_evidence(membership_rows),
            capabilities=EmbeddedArticulationCapabilityLimits(),
        )
