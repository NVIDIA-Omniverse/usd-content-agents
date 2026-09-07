# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outer-owned embedded articulation decisions and graph-only authoring.

Joint inference and Scene collection remain non-mutating evidence/proposal
providers.  Only an exact canonical graph explicitly accepted by the outer
coordinator, plus a human decision when frozen policy requires one, can become
input to the deterministic Joint authoring adapter.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EmbeddedDecisionArtifactStore,
)
from content_agent_workflows.common.embedded_domain_decision import (
    AcceptedSemanticDecision,
    BoundedExecutionAuthorization,
    ContractArtifact,
    DomainProposalPayload,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
    PersistedExecutionLineage,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    accepted_semantic_decision_digest,
    artifact_reference,
    authorize_bounded_execution,
    build_decision_receipt,
    canonical_json_digest,
    validate_bounded_execution_result,
    validate_coordinator_review,
    validate_decision_receipt,
)

from .client import (
    ArticulationAuthoringClient,
    ArticulationWorkflowClient,
    CancelChecker,
)
from .finalizer import write_articulation_workflow_summary
from .models import (
    ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationFinalizationResult,
    ArticulationInferenceResult,
    ArticulationRunState,
    ArticulationStateTransition,
    ArticulationValidationResult,
    ArticulationWorkflowMode,
    ArticulationWorkflowPhase,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    Stage2ArticulationCandidate,
    Stage2CandidateDocument,
    Stage2CandidateSummary,
    Stage2EvidenceItem,
)
from .output_evidence import (
    EmbeddedArticulationOutputEvidence,
    EmbeddedArticulationTerminalReceipt,
    build_embedded_articulation_output_evidence,
    validate_embedded_articulation_output_evidence,
)
from .scene_evidence import ArticulationSceneEvidenceCollector

EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-graph.v1"
] = "content-agent-workflows.embedded-articulation-graph.v1"
EMBEDDED_ARTICULATION_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-decision-patch.v3"
] = "content-agent-workflows.embedded-articulation-decision-patch.v3"
EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-decision-patch.v1"
] = "content-agent-workflows.embedded-articulation-decision-patch.v1"
EMBEDDED_ARTICULATION_OPTIONAL_PROPOSAL_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-decision-patch.v2"
] = "content-agent-workflows.embedded-articulation-decision-patch.v2"
EMBEDDED_ARTICULATION_OUTER_REVIEW_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-outer-review.v1"
] = "content-agent-workflows.embedded-articulation-outer-review.v1"
EMBEDDED_ARTICULATION_GRAPH_REVISION_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-graph-revision-patch.v1"
] = "content-agent-workflows.embedded-articulation-graph-revision-patch.v1"
EMBEDDED_ARTICULATION_GRAPH_REVISION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-graph-revision.v1"
] = "content-agent-workflows.embedded-articulation-graph-revision.v1"
EMBEDDED_ARTICULATION_PREPARATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-preparation.v1"
] = "content-agent-workflows.embedded-articulation-preparation.v1"
EMBEDDED_ARTICULATION_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-readback.v1"
] = "content-agent-workflows.embedded-articulation-readback.v1"
EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-post-review.v2"
] = "content-agent-workflows.embedded-articulation-post-review.v2"
EMBEDDED_ARTICULATION_LEGACY_POST_REVIEW_SCHEMA_VERSION: Literal[
    "content-agent-workflows.embedded-articulation-post-review.v1"
] = "content-agent-workflows.embedded-articulation-post-review.v1"
EMBEDDED_ARTICULATION_IMPLEMENTATION_MANIFEST: dict[str, dict[str, Any]] = {
    "embedded_articulation_controller": {
        "schema_version": "content-agent-workflows.embedded-articulation-controller.v1",
        "authority": "outer_canonical_graph_only",
        "authorization": "commit_before_executor",
        "readback": "saved_stage_exact_graph_and_membership",
        "post_review": "required_before_receipt",
        "graph_revision": "typed_immutable_parent_child",
    },
    "joint_agent_graph_adapter": {
        "schema_version": "content-agent-workflows.joint-graph-adapter.v1",
        "backend": "owned_core",
        "predictions_authority": False,
        "proposal_authority": False,
    },
    "asset_coordinator": {
        "schema_version": "content-workflow-asset.embedded-articulation.v1",
        "semantic_owner": "asset_coordinator",
        "outer_review": "required_exact_canonical_graph_digest",
        "human_review": "explicit_policy",
        "human_revise": "supersede_then_complete_re_review",
    },
    "canonical_output_evidence": {
        "schema_version": (
            "content-agent-workflows.embedded-articulation-output-evidence.v1"
        ),
        "producer": "shared_canonical_visual_leaf",
        "semantic_owner": "asset_coordinator",
        "terminal_binding": "required",
    },
}
_AXES = {
    "x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

FactState = Literal["known", "source_backed", "unknown"]
MembershipDisposition = Literal[
    "independent_motion", "co_rigid", "explicit_fixed", "unresolved"
]
OptionalProposalStatus = Literal["not_requested", "not_evaluated", "available"]
HumanReviewStatus = Literal["not_requested", "human_required"]
HumanReviewReason = Literal[
    "task_policy",
    "ambiguity",
    "unsupported_facts",
    "contradictory_evidence",
    "legacy_compatibility",
]
GraphRevisionField = Literal[
    "joint_type",
    "axis",
    "lower_limit",
    "upper_limit",
    "limit_unit",
    "frame_policy",
]


class EmbeddedArticulationError(RuntimeError):
    """Raised when embedded articulation authority is incomplete or stale."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp() -> str:
    return _now().isoformat().replace("+00:00", "Z")


def _unique(name: str, values: Sequence[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")


def _prim_path(value: str) -> bool:
    return bool(
        value.startswith("/")
        and value != "/"
        and all(part.isidentifier() for part in value[1:].split("/"))
    )


def _prim_is_within(member_prim: str, owner_prim: str) -> bool:
    """Return whether one canonical prim path is the owner or its descendant."""

    return member_prim == owner_prim or member_prim.startswith(f"{owner_prim}/")


class EmbeddedArticulationCapabilityLimits(_FrozenModel):
    supported_joint_types: tuple[Literal["revolute", "prismatic"], ...] = (
        "revolute",
        "prismatic",
    )
    supported_frame_policies: tuple[Literal["body1_world_origin"], ...] = (
        "body1_world_origin",
    )
    fixed_joint_authoring_supported: Literal[False] = False
    co_rigid_preservation_supported: Literal[True] = True
    rigid_link_body_membership_authoring_supported: bool = False
    raw_predictions_authority: Literal[False] = False
    provider_proposals_authority: Literal[False] = False
    canonical_output_evidence_required: bool = False

    @model_serializer(mode="wrap")
    def _omit_legacy_default_output_evidence_requirement(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Keep pre-output-evidence capability facts stable on Pydantic 2.11."""

        serialized = cast(dict[str, Any], handler(self))
        if not self.canonical_output_evidence_required:
            serialized.pop("canonical_output_evidence_required", None)
        if not self.rigid_link_body_membership_authoring_supported:
            serialized.pop("rigid_link_body_membership_authoring_supported", None)
        return serialized


class EmbeddedArticulationRigidLinkOperation(_FrozenModel):
    """One accepted promotion of an existing member into a rigid-body owner."""

    operation_id: str = Field(min_length=1)
    operation: Literal["apply_rigid_body_membership"] = "apply_rigid_body_membership"
    body_prim_path: str
    previous_authoritative_owner_prim: str
    disposition: Literal["independent_motion"] = "independent_motion"
    state: FactState
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        if (
            not _prim_path(self.body_prim_path)
            or not _prim_path(self.previous_authoritative_owner_prim)
            or self.body_prim_path == self.previous_authoritative_owner_prim
        ):
            raise ValueError(
                "rigid-link operation requires distinct canonical body and prior owner"
            )
        _unique("rigid-link operation evidence_ids", self.evidence_ids)
        return self


class EmbeddedArticulationMembership(_FrozenModel):
    member_prim: str
    authoritative_owner_prim: str
    group_id: str = Field(min_length=1)
    disposition: MembershipDisposition
    state: FactState
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        if not _prim_path(self.member_prim) or not _prim_path(
            self.authoritative_owner_prim
        ):
            raise ValueError("membership paths must be canonical absolute prim paths")
        _unique("membership evidence_ids", self.evidence_ids)
        return self


class EmbeddedArticulationGroup(_FrozenModel):
    group_id: str = Field(min_length=1)
    authoritative_owner_prim: str
    member_prims: tuple[str, ...] = Field(min_length=1)
    role: str = Field(min_length=1)
    role_state: FactState
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if not _prim_path(self.authoritative_owner_prim) or any(
            not _prim_path(item) for item in self.member_prims
        ):
            raise ValueError("group paths must be canonical absolute prim paths")
        _unique("group member_prims", self.member_prims)
        _unique("group evidence_ids", self.evidence_ids)
        return self


class EmbeddedArticulationJoint(_FrozenModel):
    joint_id: str = Field(min_length=1)
    body0_owner_prim: str
    body1_owner_prim: str
    body0_role: str = Field(min_length=1)
    body1_role: str = Field(min_length=1)
    role_state: FactState
    joint_type: Literal["revolute", "prismatic"]
    endpoint_state: FactState
    type_state: FactState
    axis: Literal["x", "-x", "y", "-y", "z", "-z"]
    axis_state: FactState
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: Literal["degrees", "meters"]
    limit_state: FactState
    frame_policy: Literal["body1_world_origin"]
    frame_state: FactState
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_joint(self) -> Self:
        if (
            not _prim_path(self.body0_owner_prim)
            or not _prim_path(self.body1_owner_prim)
            or self.body0_owner_prim == self.body1_owner_prim
        ):
            raise ValueError("joint endpoints must be distinct canonical prim paths")
        if self.lower_limit is not None and self.upper_limit is not None:
            if self.lower_limit > self.upper_limit:
                raise ValueError("joint lower_limit must not exceed upper_limit")
        expected_unit = "degrees" if self.joint_type == "revolute" else "meters"
        if self.limit_unit != expected_unit:
            raise ValueError(f"{self.joint_type} limits require {expected_unit}")
        _unique("joint evidence_ids", self.evidence_ids)
        return self


class EmbeddedArticulationCanonicalGraph(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-graph.v1"
    ] = EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION
    graph_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_member_prims: tuple[str, ...] = Field(min_length=1)
    authoritative_owner_prims: tuple[str, ...] = Field(min_length=1)
    candidate_ids: tuple[str, ...] = Field(min_length=1)
    groups: tuple[EmbeddedArticulationGroup, ...] = Field(min_length=1)
    memberships: tuple[EmbeddedArticulationMembership, ...] = Field(min_length=1)
    joints: tuple[EmbeddedArticulationJoint, ...] = Field(min_length=1)
    rigid_link_operations: tuple[EmbeddedArticulationRigidLinkOperation, ...] = ()
    required_fact_blockers: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_legacy_default_rigid_link_operations(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Keep existing graph-v1 digests stable when no promotion is selected."""

        serialized = cast(dict[str, Any], handler(self))
        if not self.rigid_link_operations:
            serialized.pop("rigid_link_operations", None)
        return serialized

    @model_validator(mode="after")
    def validate_local_graph(self) -> Self:
        for name, values in (
            ("source_member_prims", self.source_member_prims),
            ("authoritative_owner_prims", self.authoritative_owner_prims),
            ("candidate_ids", self.candidate_ids),
        ):
            _unique(name, values)
        if any(not _prim_path(item) for item in self.source_member_prims):
            raise ValueError("source members must be canonical prim paths")
        if any(not _prim_path(item) for item in self.authoritative_owner_prims):
            raise ValueError("authoritative owners must be canonical prim paths")
        _unique("group IDs", tuple(item.group_id for item in self.groups))
        _unique(
            "membership members", tuple(item.member_prim for item in self.memberships)
        )
        _unique("joint IDs", tuple(item.joint_id for item in self.joints))
        _unique(
            "rigid-link operation IDs",
            tuple(item.operation_id for item in self.rigid_link_operations),
        )
        _unique(
            "rigid-link operation bodies",
            tuple(item.body_prim_path for item in self.rigid_link_operations),
        )
        if self.candidate_ids != tuple(item.joint_id for item in self.joints):
            raise ValueError(
                "candidate_ids must exactly preserve canonical joint order"
            )
        return self


class EmbeddedArticulationGraphChange(_FrozenModel):
    """One exact supported field change on one stable canonical joint ID."""

    candidate_id: str = Field(min_length=1)
    field: GraphRevisionField
    previous_value: str | float | None
    revised_value: str | float | None

    @model_validator(mode="after")
    def validate_changed_value(self) -> Self:
        if self.previous_value == self.revised_value:
            raise ValueError("Articulation graph change must change its value")
        numeric = self.field in {"lower_limit", "upper_limit"}
        for label, value in (
            ("previous_value", self.previous_value),
            ("revised_value", self.revised_value),
        ):
            if numeric:
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int | float)
                ):
                    raise ValueError(f"{self.field} {label} must be numeric or null")
            elif not isinstance(value, str):
                raise ValueError(f"{self.field} {label} must be a string")
        return self


class EmbeddedArticulationGraphRevisionPatch(_FrozenModel):
    """Outer-authored request to supersede one human-revised canonical graph."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-graph-revision-patch.v1"
    ] = EMBEDDED_ARTICULATION_GRAPH_REVISION_PATCH_SCHEMA_VERSION
    expected_state_revision: int = Field(ge=0)
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_canonical_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_decisions: ExecutionArtifactBinding
    reviewer: str = Field(min_length=1)
    revision_reason: str = Field(min_length=1)
    requested_at: datetime
    changes: tuple[EmbeddedArticulationGraphChange, ...] = Field(min_length=1)
    revised_graph: EmbeddedArticulationCanonicalGraph
    revised_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_revision_patch(self) -> Self:
        if self.requested_at.tzinfo is None or self.requested_at.utcoffset() is None:
            raise ValueError(
                "Articulation graph revision timestamp must be timezone-aware"
            )
        if self.revised_graph_digest != canonical_json_digest(self.revised_graph):
            raise ValueError("Revised Articulation graph digest is stale")
        change_keys = tuple((item.candidate_id, item.field) for item in self.changes)
        if len(change_keys) != len(set(change_keys)):
            raise ValueError("Articulation graph revision changes must be unique")
        return self


class EmbeddedArticulationGraphRevision(_FrozenModel):
    """Immutable parent/child provenance for one canonical graph supersession."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-graph-revision.v1"
    ] = EMBEDDED_ARTICULATION_GRAPH_REVISION_SCHEMA_VERSION
    revision: int = Field(ge=2)
    parent_state_revision: int = Field(ge=0)
    requested_at: datetime
    created_at: datetime
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    revision_patch_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_revision: ArtifactBinding | None = None
    parent_canonical_graph: ArtifactBinding
    parent_canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_outer_review: ArtifactBinding
    parent_coordinator_decision: ArtifactBinding
    human_decision: ArtifactBinding
    human_decisions: ExecutionArtifactBinding
    reviewer: str = Field(min_length=1)
    revision_reason: str = Field(min_length=1)
    changes: tuple[EmbeddedArticulationGraphChange, ...] = Field(min_length=1)
    revised_canonical_graph: ArtifactBinding
    revised_canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    revised_outer_review: ArtifactBinding
    revised_coordinator_decision: ArtifactBinding

    @model_validator(mode="after")
    def validate_revision_record(self) -> Self:
        if any(
            timestamp.tzinfo is None or timestamp.utcoffset() is None
            for timestamp in (self.requested_at, self.created_at)
        ):
            raise ValueError(
                "Articulation graph revision timestamps must be timezone-aware"
            )
        if self.requested_at > self.created_at:
            raise ValueError(
                "Articulation graph revision request cannot postdate its commit"
            )
        if self.parent_canonical_graph == self.revised_canonical_graph:
            raise ValueError(
                "Articulation graph revision must bind distinct graph bytes"
            )
        return self


class EmbeddedArticulationProposalCapability(_FrozenModel):
    """Exact selected capability and transport for one advisory proposal."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-proposal-capability.v1"
    ] = "content-agent-workflows.embedded-articulation-proposal-capability.v1"
    provider_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    provider_alias: str = Field(min_length=1)
    adapter_id: Literal["artifact-json", "http-json"]
    adapter_implementation: str = Field(min_length=1)
    adapter_implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fallback_allowed: Literal[False] = False
    classic_controller_allowed: Literal[False] = False
    joint_agent_local_client_allowed: Literal[False] = False


class EmbeddedArticulationProviderProposal(_FrozenModel):
    """Optional provider proposal paired with its exact producer provenance."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-provider-proposal.v1",
        "content-agent-workflows.embedded-articulation-provider-proposal.v2",
    ] = "content-agent-workflows.embedded-articulation-provider-proposal.v1"
    producer: ProducerIdentity
    payload: DomainProposalPayload
    preparation_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evidence_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    provider_request_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    capability: EmbeddedArticulationProposalCapability | None = None
    provider_request: ExecutionArtifactBinding | None = None
    native_payload: ExecutionArtifactBinding | None = None

    @model_validator(mode="after")
    def validate_proposal_provider(self) -> Self:
        if self.producer.role != "proposal_provider":
            raise ValueError("Articulation proposal producer must be proposal_provider")
        v2 = (
            self.schema_version
            == "content-agent-workflows.embedded-articulation-provider-proposal.v2"
        )
        bound_fields = (
            self.preparation_digest,
            self.evidence_digest,
            self.provider_request_digest,
            self.capability,
            self.provider_request,
            self.native_payload,
        )
        if v2 and any(item is None for item in bound_fields):
            raise ValueError(
                "v2 Articulation proposals require preparation, evidence, request, "
                "and capability provenance"
            )
        if not v2 and any(item is not None for item in bound_fields):
            raise ValueError("v1 Articulation proposals cannot carry v2 provenance")
        if self.capability is not None:
            if self.capability.provider_id != self.producer.producer_id:
                raise ValueError(
                    "Articulation proposal capability names another provider"
                )
            if (
                self.capability.adapter_implementation_sha256
                != self.producer.implementation_digest
            ):
                raise ValueError(
                    "Articulation proposal capability implementation is stale"
                )
        return self


class EmbeddedArticulationHumanReviewPolicy(_FrozenModel):
    """Explicit provenance for the optional human-review leaf."""

    status: HumanReviewStatus
    reasons: tuple[HumanReviewReason, ...] = ()

    @model_validator(mode="after")
    def validate_human_review_policy(self) -> Self:
        _unique("Articulation human review reasons", self.reasons)
        if self.status == "not_requested" and self.reasons:
            raise ValueError("not_requested human review cannot retain reasons")
        if self.status == "human_required" and not self.reasons:
            raise ValueError("human_required review requires an explicit reason")
        return self


def select_embedded_articulation_human_review_policy(
    *,
    task_policy_requires_human: bool,
    ambiguity: bool,
    unsupported_facts: bool,
    contradictory_evidence: bool,
) -> EmbeddedArticulationHumanReviewPolicy:
    """Select the current outer human gate from live, explicit policy facts.

    This helper is intentionally outside the frozen v1/v2 patch serializers.
    Current v3 callers can omit the human leaf when no live requirement exists;
    any supported gate reason deterministically fails closed to human review.
    """

    reasons: list[HumanReviewReason] = []
    for selected, reason in (
        (task_policy_requires_human, "task_policy"),
        (ambiguity, "ambiguity"),
        (unsupported_facts, "unsupported_facts"),
        (contradictory_evidence, "contradictory_evidence"),
    ):
        if selected:
            reasons.append(cast(HumanReviewReason, reason))
    if not reasons:
        return EmbeddedArticulationHumanReviewPolicy(status="not_requested")
    return EmbeddedArticulationHumanReviewPolicy(
        status="human_required",
        reasons=tuple(reasons),
    )


class EmbeddedArticulationPreparation(_FrozenModel):
    """Provider-neutral deterministic evidence accepted before outer reasoning."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-preparation.v1"
    ] = EMBEDDED_ARTICULATION_PREPARATION_SCHEMA_VERSION
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_provider: ProducerIdentity
    source_hierarchy: ProviderNeutralEvidenceRecord
    source_members: ProviderNeutralEvidenceRecord
    authoritative_owners: ProviderNeutralEvidenceRecord
    capabilities: ProviderNeutralEvidenceRecord
    renders: ProviderNeutralEvidenceRecord
    scene: ProviderNeutralEvidenceRecord
    additional_evidence: tuple[ProviderNeutralEvidenceRecord, ...] = ()
    proposal_status: OptionalProposalStatus = "not_requested"
    proposal: EmbeddedArticulationProviderProposal | None = None

    @model_validator(mode="after")
    def validate_preparation_contract(self) -> Self:
        if self.evidence_provider.role != "evidence_provider":
            raise ValueError(
                "Articulation preparation producer must be evidence_provider"
            )
        expected = (
            (self.source_hierarchy, "source-hierarchy-inspection", "inspection"),
            (self.source_members, "joint-source-member-inspection", "inspection"),
            (
                self.authoritative_owners,
                "joint-authoritative-owner-inspection",
                "inspection",
            ),
            (self.capabilities, "joint-authoring-capabilities", "capability"),
            (self.renders, "joint-render-inspection", "render"),
            (self.scene, "joint-scene-inspection", "inspection"),
        )
        for record, evidence_id, evidence_type in expected:
            if record.evidence_id != evidence_id:
                raise ValueError(
                    f"Articulation preparation requires evidence_id {evidence_id!r}"
                )
            if record.evidence_type != evidence_type:
                raise ValueError(
                    f"Articulation evidence {evidence_id!r} must be {evidence_type}"
                )
        records = self.evidence_records
        evidence_ids = tuple(item.evidence_id for item in records)
        _unique("Articulation preparation evidence IDs", evidence_ids)
        reserved_ids = {"source-identity", "articulation-preparation"}
        if reserved_ids.intersection(evidence_ids):
            raise ValueError(
                "Articulation preparation evidence uses a reserved evidence ID"
            )
        if (self.proposal is None) != (self.proposal_status != "available"):
            raise ValueError(
                "Articulation proposal status must distinguish an available "
                "proposal from an unselected or unevaluated optional leaf"
            )
        if (
            self.proposal is not None
            and self.proposal.schema_version
            == "content-agent-workflows.embedded-articulation-provider-proposal.v2"
        ):
            unbound = self.model_copy(
                update={"proposal_status": "not_evaluated", "proposal": None}
            )
            if self.proposal.preparation_digest != canonical_json_digest(unbound):
                raise ValueError(
                    "Articulation proposal binds another provider-neutral preparation"
                )
            evidence_digest = canonical_json_digest(
                {
                    "records": [
                        item.model_dump(mode="json") for item in self.evidence_records
                    ]
                }
            )
            if self.proposal.evidence_digest != evidence_digest:
                raise ValueError(
                    "Articulation proposal binds another evidence record set"
                )
        return self

    @property
    def evidence_records(self) -> tuple[ProviderNeutralEvidenceRecord, ...]:
        return (
            self.source_hierarchy,
            self.source_members,
            self.authoritative_owners,
            self.capabilities,
            self.renders,
            self.scene,
            *self.additional_evidence,
        )


class EmbeddedArticulationDecisionPatch(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-decision-patch.v1",
        "content-agent-workflows.embedded-articulation-decision-patch.v2",
        "content-agent-workflows.embedded-articulation-decision-patch.v3",
    ] = EMBEDDED_ARTICULATION_PATCH_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    canonical_graph: EmbeddedArticulationCanonicalGraph
    outer_review_disposition: Literal["accept", "reject", "revise"] | None = None
    human_review: EmbeddedArticulationHumanReviewPolicy | None = None
    rationale: str = Field(min_length=1)
    revision_requests: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_versioned_review(self) -> Self:
        if (
            self.schema_version == EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION
            and self.proposal_digest is None
        ):
            raise ValueError("v1 articulation decision patches require a proposal")
        legacy = self.schema_version in {
            EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION,
            EMBEDDED_ARTICULATION_OPTIONAL_PROPOSAL_PATCH_SCHEMA_VERSION,
        }
        if legacy and (
            self.outer_review_disposition is not None
            or self.human_review is not None
            or self.revision_requests
        ):
            raise ValueError(
                "v1/v2 articulation patches use human_required compatibility"
            )
        if not legacy:
            if self.outer_review_disposition is None or self.human_review is None:
                raise ValueError(
                    "v3 articulation patches require explicit outer and human review policy"
                )
            if (
                self.outer_review_disposition != "accept"
                and self.human_review.status == "human_required"
            ):
                raise ValueError(
                    "human_required applies only to an outer-accepted graph"
                )
            if self.outer_review_disposition == "revise" and not self.revision_requests:
                raise ValueError("outer revise requires revision_requests")
            if self.outer_review_disposition != "revise" and self.revision_requests:
                raise ValueError("revision_requests require outer revise")
        return self

    @property
    def effective_outer_review_disposition(
        self,
    ) -> Literal["accept", "reject", "revise"]:
        if self.schema_version in {
            EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION,
            EMBEDDED_ARTICULATION_OPTIONAL_PROPOSAL_PATCH_SCHEMA_VERSION,
        }:
            return "accept"
        if self.outer_review_disposition is None:
            raise ValueError("v3 articulation patches require an outer review")
        return self.outer_review_disposition

    @property
    def effective_human_review(self) -> EmbeddedArticulationHumanReviewPolicy:
        if self.schema_version in {
            EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION,
            EMBEDDED_ARTICULATION_OPTIONAL_PROPOSAL_PATCH_SCHEMA_VERSION,
        }:
            return EmbeddedArticulationHumanReviewPolicy(
                status="human_required",
                reasons=("legacy_compatibility",),
            )
        if self.human_review is None:
            raise ValueError("v3 articulation patches require human review policy")
        return self.human_review


class EmbeddedArticulationOuterReview(_FrozenModel):
    """Persisted exact outer accept/reject/revise review of one canonical graph."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-outer-review.v1"
    ] = EMBEDDED_ARTICULATION_OUTER_REVIEW_SCHEMA_VERSION
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: ProducerIdentity
    evidence_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    canonical_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["accept", "reject", "revise"]
    human_review: EmbeddedArticulationHumanReviewPolicy
    rationale: str = Field(min_length=1)
    revision_requests: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_outer_review(self) -> Self:
        if self.reviewer.role != "outer_coordinator":
            raise ValueError("Articulation outer reviewer must be outer_coordinator")
        if self.disposition == "revise" and not self.revision_requests:
            raise ValueError("outer revise requires revision_requests")
        if self.disposition != "revise" and self.revision_requests:
            raise ValueError("revision_requests require outer revise")
        if (
            self.disposition != "accept"
            and self.human_review.status == "human_required"
        ):
            raise ValueError("human_required applies only to an outer-accepted graph")
        return self


class EmbeddedArticulationPostReviewPatch(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-post-review.v1",
        "content-agent-workflows.embedded-articulation-post-review.v2",
    ] = EMBEDDED_ARTICULATION_LEGACY_POST_REVIEW_SCHEMA_VERSION
    execution_result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_evidence_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    inspected_render_image_sha256s: tuple[str, ...] = ()
    disposition: Literal["accept", "reject", "revise"]
    findings: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_versioned_output_review(self) -> Self:
        if self.schema_version == EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION:
            if self.output_evidence_digest is None:
                raise ValueError("v2 post review requires output evidence digest")
            if not self.inspected_render_image_sha256s:
                raise ValueError("v2 post review requires inspected render images")
            _unique(
                "post-review inspected image digests",
                self.inspected_render_image_sha256s,
            )
            if any(
                len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
                for item in self.inspected_render_image_sha256s
            ):
                raise ValueError("post-review image digests must be SHA-256 values")
        elif self.output_evidence_digest is not None or (
            self.inspected_render_image_sha256s
        ):
            raise ValueError("v1 post review cannot carry v2 output evidence")
        return self


class EmbeddedArticulationReadback(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-readback.v1"
    ] = EMBEDDED_ARTICULATION_READBACK_SCHEMA_VERSION
    canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset_path: str = Field(min_length=1)
    output_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    joint_ids: tuple[str, ...] = Field(min_length=1)
    topology: tuple[dict[str, Any], ...] = Field(min_length=1)
    groups: tuple[dict[str, Any], ...] = Field(min_length=1)
    memberships: tuple[dict[str, Any], ...] = Field(min_length=1)
    exact_topology_match: Literal[True] = True
    exact_membership_match: Literal[True] = True
    exact_co_rigid_disposition_match: Literal[True] = True


class EmbeddedArticulationRuntime(_FrozenModel):
    identity: EmbeddedDecisionIdentity
    evidence_provider: ProducerIdentity
    proposal_provider: ProducerIdentity | None = None
    outer_coordinator: ProducerIdentity
    executor: ProducerIdentity
    capabilities: EmbeddedArticulationCapabilityLimits

    @model_validator(mode="after")
    def validate_runtime_producers(self) -> Self:
        if self.evidence_provider.role != "evidence_provider":
            raise ValueError("Articulation evidence provider has the wrong role")
        if (
            self.proposal_provider is not None
            and self.proposal_provider.role != "proposal_provider"
        ):
            raise ValueError("Articulation proposal provider has the wrong role")
        if self.outer_coordinator.role != "outer_coordinator":
            raise ValueError("Articulation outer coordinator has the wrong role")
        if self.executor.role != "executor":
            raise ValueError("Articulation executor has the wrong role")
        return self


class EmbeddedArticulationHumanAcceptance(_FrozenModel):
    canonical_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions_path: str = Field(min_length=1)
    decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    decisions: dict[str, Literal["accept", "reject", "revise"]]


HumanAcceptanceLoader = Callable[
    [Path, ArtifactBinding], EmbeddedArticulationHumanAcceptance | None
]


def _binding(path: Path) -> ArtifactBinding:
    return ArtifactBinding(path=str(path.resolve()), sha256=file_sha256(path))


def _execution_binding(path: str | Path) -> ExecutionArtifactBinding:
    resolved = Path(path).expanduser().resolve()
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _store_binding(
    store: EmbeddedDecisionArtifactStore, relative_path: str
) -> ArtifactBinding:
    return _binding(store.store_root / relative_path)


def _load_bound[ModelT: BaseModel](
    binding: ArtifactBinding | None,
    model_type: type[ModelT],
) -> ModelT:
    if binding is None:
        raise EmbeddedArticulationError(
            f"Required {model_type.__name__} artifact binding is missing"
        )
    path = Path(binding.path).expanduser().resolve()
    if file_sha256(path) != binding.sha256:
        raise EmbeddedArticulationError(f"Bound artifact changed: {path}")
    return model_type.model_validate_json(path.read_bytes())


def _persist_state(
    output_dir: Path, state: ArticulationRunState
) -> ArticulationRunState:
    updated = state.model_copy(update={"revision": state.revision + 1})
    atomic_write_json(output_dir / "checkpoint.json", updated)
    return updated


def _transition(
    state: ArticulationRunState,
    phase: ArticulationWorkflowPhase,
    reason: str,
    **updates: Any,
) -> ArticulationRunState:
    transition = ArticulationStateTransition(
        timestamp=_timestamp(),
        from_phase=state.phase,
        to_phase=phase,
        reason=reason,
    )
    return state.model_copy(
        update={
            "phase": phase,
            "transitions": (*state.transitions, transition),
            **updates,
        }
    )


def _state(output_dir: Path) -> ArticulationRunState:
    return ArticulationRunState.model_validate_json(
        (output_dir / "checkpoint.json").read_bytes()
    )


def _load_store_artifact[ModelT: ContractArtifact](
    store: EmbeddedDecisionArtifactStore,
    binding: ArtifactBinding | None,
    model_type: type[ModelT],
) -> ModelT:
    parsed = _load_bound(binding, model_type)
    return store.load_typed(artifact_reference(parsed), model_type)


def _load_optional_proposals(
    store: EmbeddedDecisionArtifactStore,
    state: ArticulationRunState,
) -> tuple[EmbeddedDomainProposal, ...]:
    if state.embedded_proposal is None:
        return ()
    proposal = _load_store_artifact(
        store,
        state.embedded_proposal,
        EmbeddedDomainProposal,
    )
    return (proposal,)


def _required_accepted_decision_digest(
    decision: EmbeddedCoordinatorDecision,
) -> str:
    digest = decision.accepted_decision_digest
    if digest is None:
        raise EmbeddedArticulationError(
            "Accepted articulation decision is missing its semantic digest"
        )
    return digest


def _validate_outer_review_chain(
    state: ArticulationRunState,
    *,
    runtime: EmbeddedArticulationRuntime,
    evidence: EmbeddedDomainEvidence,
    proposals: Sequence[EmbeddedDomainProposal],
    decision: EmbeddedCoordinatorDecision,
    graph: EmbeddedArticulationCanonicalGraph,
) -> EmbeddedArticulationOuterReview | None:
    if state.embedded_outer_review is None:
        if (
            state.schema_version == "content-agent-workflows.articulation-run-state.v2"
            and decision.human_decision_required
        ):
            return None
        raise EmbeddedArticulationError(
            "Embedded Articulation mutation requires an exact outer graph review"
        )
    review = _load_bound(
        state.embedded_outer_review,
        EmbeddedArticulationOuterReview,
    )
    assert isinstance(review, EmbeddedArticulationOuterReview)
    graph_binding = state.embedded_canonical_graph
    if graph_binding is None:
        raise EmbeddedArticulationError(
            "Embedded Articulation outer review lacks its canonical graph binding"
        )
    expected_proposal_digest = (
        artifact_reference(proposals[-1]).sha256 if proposals else None
    )
    if (
        review.identity_digest != canonical_json_digest(runtime.identity)
        or review.reviewer != runtime.outer_coordinator
        or review.reviewer != decision.producer
        or review.evidence_digest != artifact_reference(evidence).sha256
        or review.proposal_digest != expected_proposal_digest
        or review.canonical_graph_sha256 != graph_binding.sha256
        or review.disposition != decision.disposition
        or decision.human_decision_required
        != (review.human_review.status == "human_required")
        or review.rationale != decision.rationale
        or review.revision_requests != decision.revision_requests
    ):
        raise EmbeddedArticulationError(
            "Embedded Articulation outer graph review is stale or inconsistent"
        )
    if decision.disposition == "accept":
        expected_semantic = AcceptedSemanticDecision(
            schema_version=EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION,
            values={
                "canonical_graph": graph.model_dump(mode="json"),
                "human_review": review.human_review.model_dump(mode="json"),
            },
        )
        if (
            decision.accepted_decision != expected_semantic
            or decision.accepted_decision_digest
            != accepted_semantic_decision_digest(runtime.identity, expected_semantic)
        ):
            raise EmbeddedArticulationError(
                "Outer-accepted Articulation semantics differ from the reviewed graph"
            )
    return review


def _provider_backed_preparation(
    state: ArticulationRunState,
    inference: ArticulationInferenceResult,
    runtime: EmbeddedArticulationRuntime,
) -> EmbeddedArticulationPreparation:
    if state.candidate_document is None or state.inference_result is None:
        raise EmbeddedArticulationError(
            "Joint inference artifacts are not checkpointed"
        )
    candidate_artifact = _execution_binding(state.candidate_document.path)
    inference_artifact = _execution_binding(state.inference_result.path)
    membership = inference.membership_disposition_document
    inspected_members = (
        tuple(item.member_prim for item in membership.dispositions)
        if membership is not None
        else ()
    )
    owners = tuple(
        dict.fromkeys(
            [
                item.physical_owner_prim
                for item in (membership.dispositions if membership is not None else ())
                if item.physical_owner_prim is not None
            ]
            + [
                path
                for candidate in inference.candidate_document.candidates
                for path in (
                    candidate.fixed_parent_prim,
                    candidate.moving_part_prims[0]
                    if candidate.moving_part_prims
                    else None,
                )
                if path is not None
            ]
        )
    )
    source_members = tuple(dict.fromkeys([*inspected_members, *owners]))
    visual_status: Literal["available", "unavailable"] = (
        "available" if state.scene_evidence is not None else "unavailable"
    )
    visual_artifacts = (
        (_execution_binding(state.scene_evidence.path),)
        if state.scene_evidence is not None
        else ()
    )
    visual_facts = (
        {"manifest_sha256": state.scene_evidence.sha256}
        if state.scene_evidence is not None
        else {}
    )
    proposal_provider = runtime.proposal_provider
    if proposal_provider is None:
        raise EmbeddedArticulationError(
            "Provider-backed preparation requires proposal producer provenance"
        )
    return EmbeddedArticulationPreparation(
        source_sha256=state.source_sha256,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        configuration_sha256=state.backend_configuration_sha256,
        evidence_provider=runtime.evidence_provider,
        source_hierarchy=ProviderNeutralEvidenceRecord(
            evidence_id="source-hierarchy-inspection",
            evidence_type="inspection",
            status="available",
            summary="Source hierarchy and provider topology facts; never authority.",
            artifacts=(candidate_artifact, inference_artifact),
            facts={
                "candidate_ids": list(inference.candidate_document.candidate_ids),
                "candidate_count": len(inference.candidate_document.candidates),
            },
        ),
        source_members=ProviderNeutralEvidenceRecord(
            evidence_id="joint-source-member-inspection",
            evidence_type="inspection",
            status="available"
            if membership is not None and source_members
            else "error",
            summary="Complete deterministic source-member inspection.",
            artifacts=(inference_artifact,),
            facts={"source_member_prims": list(source_members)}
            if membership is not None and source_members
            else {},
        ),
        authoritative_owners=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoritative-owner-inspection",
            evidence_type="inspection",
            status="available" if membership is not None and owners else "error",
            summary="Complete deterministic authoritative-owner inspection.",
            artifacts=(inference_artifact,),
            facts={"authoritative_owner_prims": list(owners)}
            if membership is not None and owners
            else {},
        ),
        capabilities=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoring-capabilities",
            evidence_type="capability",
            status="available",
            summary="Provider-neutral limits of the embedded graph authoring adapter.",
            facts=runtime.capabilities.model_dump(mode="json"),
        ),
        renders=ProviderNeutralEvidenceRecord(
            evidence_id="joint-render-inspection",
            evidence_type="render",
            status=visual_status,
            summary=(
                "Source-bound deterministic render evidence."
                if visual_status == "available"
                else "Required render evidence was not checkpointed."
            ),
            artifacts=visual_artifacts,
            facts=visual_facts,
        ),
        scene=ProviderNeutralEvidenceRecord(
            evidence_id="joint-scene-inspection",
            evidence_type="inspection",
            status=visual_status,
            summary=(
                "Source-bound Scene inspection manifest."
                if visual_status == "available"
                else "Required Scene evidence was not checkpointed."
            ),
            artifacts=visual_artifacts,
            facts=visual_facts,
        ),
        additional_evidence=(
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-topology-inspection",
                evidence_type="inspection",
                status="available",
                summary=(
                    "Legacy-compatible Joint Agent topology evidence; never authority."
                ),
                artifacts=(candidate_artifact, inference_artifact),
                facts={
                    "candidate_ids": list(inference.candidate_document.candidate_ids),
                    "candidate_count": len(inference.candidate_document.candidates),
                },
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-membership-inspection",
                evidence_type="inspection",
                status=(
                    "available"
                    if membership is not None and source_members and owners
                    else "error"
                ),
                summary="Legacy-compatible combined member and owner inspection.",
                artifacts=(inference_artifact,),
                facts={
                    "source_member_prims": list(source_members),
                    "authoritative_owner_prims": list(owners),
                }
                if membership is not None and source_members and owners
                else {},
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="joint-visual-inspection",
                evidence_type="render",
                status=visual_status,
                summary="Legacy-compatible source-bound visual evidence.",
                artifacts=visual_artifacts,
                facts=visual_facts,
            ),
        ),
        proposal_status="available",
        proposal=EmbeddedArticulationProviderProposal(
            producer=proposal_provider,
            payload=DomainProposalPayload(
                schema_version=(
                    "content-agent-workflows.articulation-provider-proposal.v1"
                ),
                values={
                    "stage2_candidate_document": (
                        inference.candidate_document.model_dump(mode="json")
                    ),
                    "membership_disposition_document": (
                        inference.membership_disposition_document.model_dump(
                            mode="json"
                        )
                        if inference.membership_disposition_document is not None
                        else None
                    ),
                    "predictions_path": inference.predictions_path,
                    "predictions_sha256": inference.predictions_sha256,
                },
            ),
        ),
    )


def state_to_source_binding(state: ArticulationRunState) -> ExecutionArtifactBinding:
    source = Path(state.source_asset).expanduser().resolve()
    return ExecutionArtifactBinding(
        path=str(source), sha256=state.source_sha256, size_bytes=source.stat().st_size
    )


def _verify_execution_binding(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> None:
    """Preserve frozen caller-owned source/evidence compatibility semantics."""

    expanded = Path(binding.path).expanduser()
    path = expanded.resolve()
    if expanded.is_symlink() or not path.is_file():
        raise EmbeddedArticulationError(f"{label} is not a regular file: {path}")
    if path.stat().st_size != binding.size_bytes or file_sha256(path) != binding.sha256:
        raise EmbeddedArticulationError(f"{label} changed after deterministic capture")


def _verify_strict_execution_binding(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
) -> None:
    _read_execution_binding(binding, label=label)


def _read_execution_binding(
    binding: ExecutionArtifactBinding,
    *,
    label: str,
    max_bytes: int | None = None,
    capture_bytes: bool = False,
) -> bytes | None:
    absolute = Path(os.path.abspath(Path(binding.path).expanduser()))
    try:
        captured = read_contained_artifact(
            absolute.parent,
            absolute.name,
            max_bytes=max_bytes,
            capture_bytes=capture_bytes,
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(f"{label} is unsafe: {exc}") from exc
    observed = ExecutionArtifactBinding(
        path=str(captured.path),
        sha256=captured.sha256,
        size_bytes=captured.size_bytes,
    )
    if observed != binding:
        raise EmbeddedArticulationError(f"{label} changed after deterministic capture")
    return captured.data


def validate_embedded_articulation_provider_proposal(
    preparation: EmbeddedArticulationPreparation,
) -> None:
    """Revalidate a replacement proposal's typed request and native payload."""

    proposal = preparation.proposal
    if proposal is None or proposal.capability is None:
        return
    capability = proposal.capability
    if proposal.provider_request is None or proposal.native_payload is None:
        raise EmbeddedArticulationError(
            "Replacement proposal lacks typed request or payload provenance"
        )
    try:
        request_bytes = _read_execution_binding(
            proposal.provider_request,
            label="Articulation proposal provider request",
            max_bytes=128 * 1024 * 1024,
            capture_bytes=True,
        )
        native_payload_bytes = _read_execution_binding(
            proposal.native_payload,
            label="Articulation proposal native payload",
            max_bytes=128 * 1024 * 1024,
            capture_bytes=True,
        )
        if request_bytes is None or native_payload_bytes is None:
            raise ValueError("proposal provenance bytes were not captured")
        request_document = json.loads(request_bytes)
        native_payload = DomainProposalPayload.model_validate_json(native_payload_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EmbeddedArticulationError(
            "Replacement proposal provenance is malformed"
        ) from exc
    if not isinstance(request_document, dict):
        raise EmbeddedArticulationError("Replacement proposal provenance is malformed")
    request_schema_version = request_document.get("schema_version")
    try:
        # Imported lazily to keep the provider implementation layered above the
        # embedded decision models while reusing its exact frozen wire schemas.
        from .proposal_provider import (
            ArticulationProposalProviderRequest,
            ArticulationProposalProviderRequestContract,
            ArticulationProposalProviderRequestV2,
        )

        request_payload: ArticulationProposalProviderRequestContract
        if request_schema_version == (
            "content-agent-workflows.articulation-proposal-provider-request.v2"
        ):
            request_payload = ArticulationProposalProviderRequestV2.model_validate(
                request_document
            )
            provider_inputs = request_payload.provider_inputs
        elif request_schema_version == (
            "content-agent-workflows.articulation-proposal-provider-request.v1"
        ):
            request_payload = ArticulationProposalProviderRequest.model_validate(
                request_document
            )
            provider_inputs = ()
        else:
            raise ValueError("unsupported provider request schema")
    except (ImportError, ValueError) as exc:
        raise EmbeddedArticulationError(
            "Replacement proposal request schema is malformed or unsupported"
        ) from exc
    for provider_input in provider_inputs:
        _verify_strict_execution_binding(
            provider_input,
            label="Articulation proposal provider input",
        )
    original_preparation = preparation.model_copy(
        update={"proposal_status": "not_evaluated", "proposal": None}
    )
    try:
        request_preparation_bytes = _read_execution_binding(
            request_payload.preparation,
            label="Articulation proposal request preparation",
            max_bytes=64 * 1024 * 1024,
            capture_bytes=True,
        )
        if request_preparation_bytes is None:
            raise ValueError("request preparation bytes were not captured")
        request_preparation = EmbeddedArticulationPreparation.model_validate_json(
            request_preparation_bytes
        )
    except (EmbeddedArticulationError, ValueError) as exc:
        raise EmbeddedArticulationError(
            "Replacement proposal request preparation is malformed"
        ) from exc
    wrapper_values = dict(proposal.payload.model_dump(mode="json")["values"])
    try:
        wrapped_native_payload = DomainProposalPayload.model_validate(
            wrapper_values.get("native_payload")
        )
    except ValueError as exc:
        raise EmbeddedArticulationError(
            "Replacement proposal wrapped payload is malformed"
        ) from exc
    if (
        canonical_json_digest(request_document) != proposal.provider_request_digest
        or request_document.get("preparation_digest") != proposal.preparation_digest
        or request_document.get("evidence_digest") != proposal.evidence_digest
        or request_document.get("source_sha256") != preparation.source_sha256
        or request_document.get("source_dependency_bundle_sha256")
        != preparation.source_dependency_bundle_sha256
        or request_document.get("configuration_sha256")
        != preparation.configuration_sha256
        or request_document.get("capability") != capability.model_dump(mode="json")
        or request_document.get("preparation_payload")
        != original_preparation.model_dump(mode="json")
        or request_preparation != original_preparation
        or canonical_json_digest(request_preparation)
        != request_payload.preparation_digest
        or proposal.payload.schema_version
        != "content-agent-workflows.articulation-provider-proposal-payload.v2"
        or set(wrapper_values)
        != {
            "provider_request_digest",
            "preparation_digest",
            "evidence_digest",
            "provider_capability",
            "native_payload",
            "native_payload_binding",
        }
        or wrapper_values.get("provider_request_digest")
        != proposal.provider_request_digest
        or wrapper_values.get("preparation_digest") != proposal.preparation_digest
        or wrapper_values.get("evidence_digest") != proposal.evidence_digest
        or wrapper_values.get("provider_capability")
        != capability.model_dump(mode="json")
        or wrapped_native_payload != native_payload
        or wrapper_values.get("native_payload_binding")
        != proposal.native_payload.model_dump(mode="json")
    ):
        raise EmbeddedArticulationError(
            "Replacement proposal request or native payload provenance is stale"
        )


def _validate_preparation(
    state: ArticulationRunState,
    preparation: EmbeddedArticulationPreparation,
    runtime: EmbeddedArticulationRuntime,
) -> None:
    if (
        preparation.source_sha256 != state.source_sha256
        or preparation.source_dependency_bundle_sha256
        != state.source_dependency_bundle_sha256
    ):
        raise EmbeddedArticulationError(
            "Embedded Articulation preparation source identity is stale"
        )
    if preparation.configuration_sha256 != state.backend_configuration_sha256:
        raise EmbeddedArticulationError(
            "Embedded Articulation preparation configuration is stale"
        )
    if preparation.evidence_provider != runtime.evidence_provider:
        raise EmbeddedArticulationError(
            "Embedded Articulation evidence producer identity is stale"
        )
    proposal_provider = (
        preparation.proposal.producer if preparation.proposal is not None else None
    )
    if proposal_provider != runtime.proposal_provider:
        raise EmbeddedArticulationError(
            "Embedded Articulation proposal producer identity is stale"
        )
    if preparation.proposal is not None and preparation.proposal.capability:
        proposal = preparation.proposal
        capability = proposal.capability
        if capability is None:  # pragma: no cover - truthiness guard above
            raise EmbeddedArticulationError(
                "Embedded Articulation proposal capability is missing"
            )
        if canonical_json_digest(capability) not in set(
            runtime.identity.digests.capabilities.values()
        ):
            raise EmbeddedArticulationError(
                "Embedded Articulation proposal capability is not identity-bound"
            )
        if capability.provider_configuration_sha256 not in set(
            runtime.identity.digests.configuration.values()
        ):
            raise EmbeddedArticulationError(
                "Embedded Articulation proposal configuration is not identity-bound"
            )
        if capability.adapter_implementation_sha256 not in set(
            runtime.identity.digests.implementations.values()
        ):
            raise EmbeddedArticulationError(
                "Embedded Articulation proposal implementation is not identity-bound"
            )
        validate_embedded_articulation_provider_proposal(preparation)
    if preparation.capabilities.status == "available" and canonical_json_digest(
        dict(preparation.capabilities.facts)
    ) != canonical_json_digest(runtime.capabilities):
        raise EmbeddedArticulationError(
            "Embedded Articulation capability evidence is stale"
        )
    _verify_execution_binding(state_to_source_binding(state), label="source asset")
    for record in preparation.evidence_records:
        for artifact in record.artifacts:
            _verify_execution_binding(
                artifact,
                label=f"Articulation evidence {record.evidence_id}",
            )


def _artifact_created_at(state: ArticulationRunState) -> datetime:
    if not state.transitions:
        raise EmbeddedArticulationError(
            "Embedded Articulation evidence requires a durable phase transition"
        )
    return datetime.fromisoformat(
        state.transitions[-1].timestamp.replace("Z", "+00:00")
    )


def _initialize_contract_artifacts(
    output_dir: Path,
    state: ArticulationRunState,
    runtime: EmbeddedArticulationRuntime,
    preparation: EmbeddedArticulationPreparation,
    preparation_binding: ArtifactBinding,
) -> ArticulationRunState:
    _validate_preparation(state, preparation, runtime)
    store = EmbeddedDecisionArtifactStore(output_dir)
    created_at = _artifact_created_at(state)
    evidence = EmbeddedDomainEvidence(
        artifact_id="articulation-inspection",
        identity=runtime.identity,
        producer=runtime.evidence_provider,
        parent_artifact=runtime.identity.coordinator_plan,
        created_at=created_at,
        records=(
            ProviderNeutralEvidenceRecord(
                evidence_id="source-identity",
                evidence_type="artifact",
                status="available",
                summary=(
                    "Exact source and dependency identity for the active outer stage."
                ),
                artifacts=(state_to_source_binding(state),),
                facts={
                    "source_sha256": state.source_sha256,
                    "source_dependency_bundle_sha256": (
                        state.source_dependency_bundle_sha256
                    ),
                },
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="articulation-preparation",
                evidence_type="artifact",
                status="available",
                summary="Exact public provider-neutral preparation contract.",
                artifacts=(_execution_binding(preparation_binding.path),),
                facts={
                    "schema_version": preparation.schema_version,
                    "proposal_status": preparation.proposal_status,
                },
            ),
            *preparation.evidence_records,
        ),
    )
    evidence_commit = store.append(evidence)
    proposal_binding = None
    if preparation.proposal is not None:
        proposal = EmbeddedDomainProposal(
            artifact_id="articulation-provider-proposal",
            identity=runtime.identity,
            producer=preparation.proposal.producer,
            parent_artifact=artifact_reference(evidence),
            created_at=created_at,
            evidence_artifacts=(artifact_reference(evidence),),
            proposal=preparation.proposal.payload,
            proposal_digest=canonical_json_digest(preparation.proposal.payload),
        )
        proposal_commit = store.append(proposal)
        proposal_binding = _store_binding(store, proposal_commit.relative_path)
    updated = state.model_copy(
        update={
            "schema_version": ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
            "embedded_evidence": _store_binding(store, evidence_commit.relative_path),
            "embedded_proposal": proposal_binding,
            "review_required_candidate_ids": state.candidate_ids,
            "auto_accepted_candidate_ids": (),
            "accepted_candidate_ids": (),
            "rejected_candidate_ids": (),
            "unresolved_candidate_ids": (),
            "review_required_membership_disposition_ids": (),
            "unresolved_membership_disposition_ids": (),
            "error": None,
        }
    )
    return _persist_state(output_dir, updated)


def _inspection_coverage(
    evidence: EmbeddedDomainEvidence,
) -> tuple[set[str], set[str], dict[str, tuple[str, str]] | None]:
    for record in evidence.records:
        for artifact in record.artifacts:
            _verify_execution_binding(
                artifact,
                label=f"Articulation evidence {record.evidence_id}",
            )
    unavailable = [
        record.evidence_id
        for record in evidence.records
        if record.required and record.status != "available"
    ]
    if unavailable:
        raise EmbeddedArticulationError(
            "Required embedded articulation evidence is unavailable: "
            + ", ".join(unavailable)
        )
    source_members = next(
        (
            item
            for item in evidence.records
            if item.evidence_id == "joint-source-member-inspection"
        ),
        None,
    )
    authoritative_owners = next(
        (
            item
            for item in evidence.records
            if item.evidence_id == "joint-authoritative-owner-inspection"
        ),
        None,
    )
    if source_members is None or authoritative_owners is None:
        legacy_membership = next(
            (
                item
                for item in evidence.records
                if item.evidence_id == "joint-membership-inspection"
            ),
            None,
        )
        if legacy_membership is not None:
            source_members = source_members or legacy_membership
            authoritative_owners = authoritative_owners or legacy_membership
    if source_members is None or authoritative_owners is None:
        raise EmbeddedArticulationError(
            "Source-member or authoritative-owner coverage evidence is missing"
        )
    members = source_members.facts.get("source_member_prims")
    owners = authoritative_owners.facts.get("authoritative_owner_prims")
    if not isinstance(members, Sequence) or isinstance(members, str | bytes):
        raise EmbeddedArticulationError("Source member coverage facts are malformed")
    if not isinstance(owners, Sequence) or isinstance(owners, str | bytes):
        raise EmbeddedArticulationError("Authoritative owner facts are malformed")
    expected_members = {str(item) for item in members}
    expected_owners = {str(item) for item in owners}
    raw_rows = authoritative_owners.facts.get("membership_rows")
    membership_policy = authoritative_owners.facts.get("membership_policy")
    if (raw_rows is None) != (membership_policy is None):
        raise EmbeddedArticulationError(
            "Current exact membership rows and policy must be present together"
        )
    if membership_policy is not None and (
        membership_policy != "retained-explicit-membership-v1"
    ):
        raise EmbeddedArticulationError("Exact membership policy is unsupported")
    membership_rows: dict[str, tuple[str, str]] | None = None
    if raw_rows is not None:
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, str | bytes):
            raise EmbeddedArticulationError("Exact membership rows are malformed")
        membership_rows = {}
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise EmbeddedArticulationError("Exact membership rows are malformed")
            member = raw_row.get("member_prim")
            owner = raw_row.get("authoritative_owner_prim")
            disposition = raw_row.get("disposition")
            if (
                not isinstance(member, str)
                or not isinstance(owner, str)
                or not isinstance(disposition, str)
                or disposition
                not in {"independent_motion", "co_rigid", "explicit_fixed"}
                or member in membership_rows
            ):
                raise EmbeddedArticulationError("Exact membership rows are malformed")
            membership_rows[member] = (owner, str(disposition))
        if (
            set(membership_rows) != expected_members
            or {item[0] for item in membership_rows.values()} != expected_owners
        ):
            raise EmbeddedArticulationError(
                "Exact membership rows differ from inspection coverage"
            )
        if any(
            (disposition == "independent_motion" and member != owner)
            or (
                disposition in {"co_rigid", "explicit_fixed"}
                and not _prim_is_within(member, owner)
            )
            for member, (owner, disposition) in membership_rows.items()
        ):
            raise EmbeddedArticulationError(
                "Exact membership rows contain invalid ownership semantics"
            )
        if any(
            owner not in membership_rows or membership_rows[owner][0] != owner
            for owner, _disposition in membership_rows.values()
        ):
            raise EmbeddedArticulationError(
                "Exact membership rows name a transitively owned authoritative owner"
            )
    return expected_members, expected_owners, membership_rows


def _articulation_capabilities_from_evidence(
    evidence: EmbeddedDomainEvidence,
) -> EmbeddedArticulationCapabilityLimits:
    capability_facts = next(
        (
            record.facts
            for record in evidence.records
            if record.evidence_id == "joint-authoring-capabilities"
        ),
        None,
    )
    if capability_facts is None:
        raise ValueError(
            "Completed embedded articulation evidence lacks capability facts"
        )
    return EmbeddedArticulationCapabilityLimits.model_validate(capability_facts)


def _validate_canonical_graph(
    graph: EmbeddedArticulationCanonicalGraph,
    *,
    state: ArticulationRunState,
    evidence: EmbeddedDomainEvidence,
    capabilities: EmbeddedArticulationCapabilityLimits,
    authoring_required: bool = True,
    allow_preserved_explicit_fixed_membership: bool = False,
) -> None:
    if graph.source_sha256 != state.source_sha256 or (
        graph.source_dependency_bundle_sha256 != state.source_dependency_bundle_sha256
    ):
        raise EmbeddedArticulationError("Canonical graph source identity is stale")
    if authoring_required and graph.required_fact_blockers:
        raise EmbeddedArticulationError(
            "Canonical graph retains required fact blockers"
        )
    expected_members, expected_owners, expected_memberships = _inspection_coverage(
        evidence
    )
    operations = graph.rigid_link_operations
    if operations and not capabilities.rigid_link_body_membership_authoring_supported:
        raise EmbeddedArticulationError(
            "Rigid-link/body-membership operation exceeds embedded capabilities"
        )
    promoted_owners = {item.body_prim_path for item in operations}
    effective_owners = expected_owners | promoted_owners
    record_ids = {item.evidence_id for item in evidence.records}
    for operation in operations:
        if operation.body_prim_path not in expected_members:
            raise EmbeddedArticulationError(
                "Rigid-link operation body is not an inspected source member"
            )
        if operation.body_prim_path in expected_owners:
            raise EmbeddedArticulationError(
                "Rigid-link operation cannot re-author an existing body owner"
            )
        if operation.previous_authoritative_owner_prim not in expected_owners:
            raise EmbeddedArticulationError(
                "Rigid-link operation prior owner is not source-backed"
            )
        if (
            operation.state == "unknown"
            or not set(operation.evidence_ids) <= record_ids
        ):
            raise EmbeddedArticulationError(
                "Rigid-link operation lacks accepted source-bound evidence"
            )
    if set(graph.source_member_prims) != expected_members:
        raise EmbeddedArticulationError(
            "Canonical graph lacks complete source-member coverage"
        )
    if set(graph.authoritative_owner_prims) != effective_owners:
        raise EmbeddedArticulationError(
            "Canonical graph lacks authoritative-owner coverage"
        )
    if {item.member_prim for item in graph.memberships} != expected_members:
        raise EmbeddedArticulationError(
            "Membership disposition does not cover every source member"
        )
    if {item.authoritative_owner_prim for item in graph.memberships} - effective_owners:
        raise EmbeddedArticulationError("Membership names a non-authoritative owner")
    observed_memberships = {
        item.member_prim: (item.authoritative_owner_prim, item.disposition)
        for item in graph.memberships
    }
    for operation in operations:
        promoted_membership = observed_memberships.get(operation.body_prim_path)
        if (
            promoted_membership is None
            or promoted_membership[0] != operation.body_prim_path
        ):
            raise EmbeddedArticulationError(
                "Rigid-link operation promoted body must own itself"
            )
    if expected_memberships is not None:
        adjusted_memberships = dict(expected_memberships)
        for operation in operations:
            previous = adjusted_memberships.get(operation.body_prim_path)
            if previous is None or previous[0] != (
                operation.previous_authoritative_owner_prim
            ):
                raise EmbeddedArticulationError(
                    "Rigid-link operation does not bind exact prior membership"
                )
            adjusted_memberships[operation.body_prim_path] = (
                operation.body_prim_path,
                operation.disposition,
            )
        if observed_memberships != adjusted_memberships:
            raise EmbeddedArticulationError(
                "Canonical graph membership differs from exact inspection rows"
            )
    group_by_id = {item.group_id: item for item in graph.groups}
    if {item.authoritative_owner_prim for item in graph.groups} != effective_owners:
        raise EmbeddedArticulationError("Groups do not cover every authoritative owner")
    for membership in graph.memberships:
        group = group_by_id.get(membership.group_id)
        if (
            group is None
            or membership.member_prim not in group.member_prims
            or (membership.authoritative_owner_prim != group.authoritative_owner_prim)
        ):
            raise EmbeddedArticulationError(
                "Membership grouping is incomplete or inconsistent"
            )
        if authoring_required and (
            membership.disposition == "unresolved"
            or (
                membership.disposition == "explicit_fixed"
                and not allow_preserved_explicit_fixed_membership
            )
        ):
            raise EmbeddedArticulationError(
                "Unresolved or explicit-fixed membership cannot enter v1 authoring"
            )
        if authoring_required and membership.state == "unknown":
            raise EmbeddedArticulationError(
                "Membership contains an unknown required fact"
            )
        if not set(membership.evidence_ids) <= record_ids:
            raise EmbeddedArticulationError("Membership cites unknown evidence")
        if membership.disposition == "co_rigid" and not _prim_is_within(
            membership.member_prim,
            membership.authoritative_owner_prim,
        ):
            raise EmbeddedArticulationError(
                "Co-rigid member is not contained by its authoritative owner"
            )
    for group in graph.groups:
        expected_group_members = {
            item.member_prim
            for item in graph.memberships
            if item.group_id == group.group_id
        }
        if set(group.member_prims) != expected_group_members:
            raise EmbeddedArticulationError(
                "Group membership disposition is incomplete"
            )
        if authoring_required and group.role_state == "unknown":
            raise EmbeddedArticulationError(
                "Group role is unknown or cites unknown evidence"
            )
        if not set(group.evidence_ids) <= record_ids:
            raise EmbeddedArticulationError(
                "Group role is unknown or cites unknown evidence"
            )
    moving_owners: set[str] = set()
    for joint in graph.joints:
        states = (
            joint.role_state,
            joint.endpoint_state,
            joint.type_state,
            joint.axis_state,
            joint.limit_state,
            joint.frame_state,
        )
        if authoring_required and "unknown" in states:
            raise EmbeddedArticulationError("Joint contains an unknown required fact")
        if joint.body0_owner_prim not in effective_owners or (
            joint.body1_owner_prim not in effective_owners
        ):
            raise EmbeddedArticulationError(
                "Joint endpoints are not authoritative owners"
            )
        if joint.body1_owner_prim in moving_owners:
            raise EmbeddedArticulationError(
                "Canonical graph drives one moving owner from multiple joints"
            )
        moving_owners.add(joint.body1_owner_prim)
        if (
            authoring_required
            and joint.joint_type not in capabilities.supported_joint_types
        ):
            raise EmbeddedArticulationError("Joint type exceeds embedded capabilities")
        if (
            authoring_required
            and joint.frame_policy not in capabilities.supported_frame_policies
        ):
            raise EmbeddedArticulationError(
                "Joint frame policy exceeds embedded capabilities"
            )
        if not set(joint.evidence_ids) <= record_ids:
            raise EmbeddedArticulationError("Joint cites unknown evidence")


def apply_embedded_articulation_decision_patch(
    output_dir: str | Path,
    patch: EmbeddedArticulationDecisionPatch,
    *,
    runtime: EmbeddedArticulationRuntime,
) -> ArticulationRunState:
    """Persist an exact outer graph review before any optional human review."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if state.embedded_evidence is None:
        raise EmbeddedArticulationError("Embedded evidence must be persisted first")
    if patch.identity_digest != canonical_json_digest(runtime.identity):
        raise EmbeddedArticulationError("Decision patch identity is stale")
    store = EmbeddedDecisionArtifactStore(root)
    evidence = _load_store_artifact(
        store, state.embedded_evidence, EmbeddedDomainEvidence
    )
    proposals = _load_optional_proposals(store, state)
    assert isinstance(evidence, EmbeddedDomainEvidence)
    if patch.evidence_digest != artifact_reference(evidence).sha256:
        raise EmbeddedArticulationError("Decision patch evidence digest is stale")
    expected_proposal_digest = (
        artifact_reference(proposals[-1]).sha256 if proposals else None
    )
    if patch.proposal_digest != expected_proposal_digest:
        raise EmbeddedArticulationError("Decision patch proposal digest is stale")
    if state.embedded_coordinator_decision is not None:
        graph_binding = state.embedded_canonical_graph
        if graph_binding is None:
            raise EmbeddedArticulationError(
                "Persisted Articulation decision lacks its canonical graph binding"
            )
        existing_graph = _load_bound(
            graph_binding,
            EmbeddedArticulationCanonicalGraph,
        )
        if existing_graph != patch.canonical_graph:
            raise EmbeddedArticulationError(
                "A different canonical graph is already persisted"
            )
        if state.embedded_outer_review is not None:
            existing_outer_review = _load_bound(
                state.embedded_outer_review,
                EmbeddedArticulationOuterReview,
            )
            expected_outer_review = EmbeddedArticulationOuterReview(
                identity_digest=patch.identity_digest,
                reviewer=runtime.outer_coordinator,
                evidence_digest=patch.evidence_digest,
                proposal_digest=patch.proposal_digest,
                canonical_graph_sha256=graph_binding.sha256,
                disposition=patch.effective_outer_review_disposition,
                human_review=patch.effective_human_review,
                rationale=patch.rationale,
                revision_requests=patch.revision_requests,
            )
            if existing_outer_review != expected_outer_review:
                raise EmbeddedArticulationError(
                    "A different outer graph review is already persisted"
                )
        elif state.schema_version in {
            "content-agent-workflows.articulation-run-state.v3",
            ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
        }:
            raise EmbeddedArticulationError(
                "Current embedded Articulation decision lacks its outer review"
            )
        existing_decision = _load_store_artifact(
            store,
            state.embedded_coordinator_decision,
            EmbeddedCoordinatorDecision,
        )
        if (
            existing_decision.identity != runtime.identity
            or existing_decision.producer != runtime.outer_coordinator
            or existing_decision.disposition != patch.effective_outer_review_disposition
            or existing_decision.evidence_artifacts != (artifact_reference(evidence),)
            or existing_decision.proposal_artifacts
            != tuple(artifact_reference(item) for item in proposals)
            or existing_decision.human_decision_required
            != (patch.effective_human_review.status == "human_required")
            or existing_decision.rationale != patch.rationale
            or existing_decision.revision_requests != patch.revision_requests
        ):
            raise EmbeddedArticulationError(
                "Existing outer Articulation decision differs from the exact patch"
            )
        return state
    outer_disposition = patch.effective_outer_review_disposition
    human_review = patch.effective_human_review
    _validate_canonical_graph(
        patch.canonical_graph,
        state=state,
        evidence=evidence,
        capabilities=runtime.capabilities,
        authoring_required=outer_disposition == "accept",
    )
    graph_path = root / "canonical_articulation_graph.json"
    if graph_path.exists():
        existing = EmbeddedArticulationCanonicalGraph.model_validate_json(
            graph_path.read_bytes()
        )
        if existing != patch.canonical_graph:
            raise EmbeddedArticulationError(
                "Canonical graph path contains different bytes"
            )
    else:
        atomic_write_json(graph_path, patch.canonical_graph)
    graph_binding = _binding(graph_path)
    outer_review = EmbeddedArticulationOuterReview(
        identity_digest=patch.identity_digest,
        reviewer=runtime.outer_coordinator,
        evidence_digest=patch.evidence_digest,
        proposal_digest=patch.proposal_digest,
        canonical_graph_sha256=graph_binding.sha256,
        disposition=outer_disposition,
        human_review=human_review,
        rationale=patch.rationale,
        revision_requests=patch.revision_requests,
    )
    outer_review_path = root / "embedded_articulation_outer_review.json"
    if outer_review_path.exists():
        existing_review = EmbeddedArticulationOuterReview.model_validate_json(
            outer_review_path.read_bytes()
        )
        if existing_review != outer_review:
            raise EmbeddedArticulationError(
                "Outer graph review path contains different bytes"
            )
    else:
        atomic_write_json(outer_review_path, outer_review)
    outer_review_binding = _binding(outer_review_path)
    semantic = (
        AcceptedSemanticDecision(
            schema_version=EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION,
            values={
                "canonical_graph": patch.canonical_graph.model_dump(mode="json"),
                "human_review": human_review.model_dump(mode="json"),
            },
        )
        if outer_disposition == "accept"
        else None
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id="articulation-canonical-graph-decision",
        identity=runtime.identity,
        producer=runtime.outer_coordinator,
        parent_artifact=(
            artifact_reference(proposals[-1])
            if proposals
            else artifact_reference(evidence)
        ),
        created_at=_now(),
        disposition=outer_disposition,
        evidence_artifacts=(artifact_reference(evidence),),
        proposal_artifacts=tuple(artifact_reference(item) for item in proposals),
        accepted_decision=semantic,
        accepted_decision_digest=(
            accepted_semantic_decision_digest(runtime.identity, semantic)
            if semantic is not None
            else None
        ),
        human_decision_required=(
            outer_disposition == "accept" and human_review.status == "human_required"
        ),
        rationale=patch.rationale,
        revision_requests=patch.revision_requests,
    )
    commit = store.append(decision)
    state = state.model_copy(
        update={
            "embedded_canonical_graph": graph_binding,
            "embedded_outer_review": outer_review_binding,
            "embedded_coordinator_decision": _store_binding(
                store, commit.relative_path
            ),
            "candidate_ids": patch.canonical_graph.candidate_ids,
            "review_required_candidate_ids": (
                patch.canonical_graph.candidate_ids
                if decision.human_decision_required
                else ()
            ),
            "accepted_candidate_ids": (),
            "rejected_candidate_ids": (),
            "unresolved_candidate_ids": (),
            "error": None,
        }
    )
    if outer_disposition != "accept":
        state = _transition(
            state,
            "conditional",
            f"Outer canonical graph review {outer_disposition} blocks authoring.",
            rejected_candidate_ids=(
                patch.canonical_graph.candidate_ids
                if outer_disposition == "reject"
                else ()
            ),
            unresolved_candidate_ids=(
                patch.canonical_graph.candidate_ids
                if outer_disposition == "revise"
                else ()
            ),
        )
    elif decision.human_decision_required and state.phase == "awaiting_decision":
        state = _transition(
            state,
            "needs_review",
            "Exact outer-accepted canonical graph requires policy-selected human review.",
        )
    return _persist_state(root, state)


_GRAPH_REVISION_FIELDS: tuple[GraphRevisionField, ...] = (
    "joint_type",
    "axis",
    "lower_limit",
    "upper_limit",
    "limit_unit",
    "frame_policy",
)


def _canonical_graph_changes(
    parent: EmbeddedArticulationCanonicalGraph,
    revised: EmbeddedArticulationCanonicalGraph,
) -> tuple[EmbeddedArticulationGraphChange, ...]:
    """Return the exact supported semantic delta or reject structural drift."""

    immutable_graph_fields = (
        "schema_version",
        "graph_id",
        "source_sha256",
        "source_dependency_bundle_sha256",
        "source_member_prims",
        "authoritative_owner_prims",
        "candidate_ids",
        "groups",
        "memberships",
        "required_fact_blockers",
    )
    for field_name in immutable_graph_fields:
        if getattr(parent, field_name) != getattr(revised, field_name):
            raise EmbeddedArticulationError(
                "Canonical graph revision contains unsupported graph drift: "
                f"{field_name}"
            )
    if len(parent.joints) != len(revised.joints):
        raise EmbeddedArticulationError(
            "Canonical graph revision cannot change joint coverage"
        )
    changes: list[EmbeddedArticulationGraphChange] = []
    supported = set(_GRAPH_REVISION_FIELDS)
    for parent_joint, revised_joint in zip(
        parent.joints,
        revised.joints,
        strict=True,
    ):
        if parent_joint.joint_id != revised_joint.joint_id:
            raise EmbeddedArticulationError(
                "Canonical graph revision cannot change candidate identity or order"
            )
        parent_values = parent_joint.model_dump(mode="json")
        revised_values = revised_joint.model_dump(mode="json")
        changed_fields = {
            field_name
            for field_name in parent_values
            if parent_values[field_name] != revised_values[field_name]
        }
        unsupported = sorted(changed_fields - supported)
        if unsupported:
            raise EmbeddedArticulationError(
                "Canonical graph revision contains unsupported joint drift for "
                f"{parent_joint.joint_id}: {unsupported}"
            )
        for field_name in _GRAPH_REVISION_FIELDS:
            if field_name not in changed_fields:
                continue
            changes.append(
                EmbeddedArticulationGraphChange(
                    candidate_id=parent_joint.joint_id,
                    field=field_name,
                    previous_value=parent_values[field_name],
                    revised_value=revised_values[field_name],
                )
            )
    if not changes:
        raise EmbeddedArticulationError(
            "Canonical graph revision must contain at least one supported change"
        )
    return tuple(changes)


def _load_revision_human_decisions(
    binding: ExecutionArtifactBinding,
    *,
    graph: EmbeddedArticulationCanonicalGraph,
) -> dict[str, Literal["accept", "reject", "revise"]]:
    actual = _execution_binding(binding.path)
    if actual != binding:
        raise EmbeddedArticulationError(
            "Canonical graph revision human decision bytes are stale"
        )
    try:
        payload = json.loads(Path(binding.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddedArticulationError(
            "Canonical graph revision human decisions are not valid JSON"
        ) from exc
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), dict):
        payload = payload["decisions"]
    if not isinstance(payload, dict) or set(payload) != set(graph.candidate_ids):
        raise EmbeddedArticulationError(
            "Canonical graph revision human decisions do not cover the parent graph"
        )
    allowed = {"accept", "reject", "revise"}
    if any(value not in allowed for value in payload.values()):
        raise EmbeddedArticulationError(
            "Canonical graph revision contains an invalid human disposition"
        )
    return cast(
        dict[str, Literal["accept", "reject", "revise"]],
        {candidate_id: payload[candidate_id] for candidate_id in graph.candidate_ids},
    )


def canonical_articulation_graph_changes(
    parent: EmbeddedArticulationCanonicalGraph,
    revised: EmbeddedArticulationCanonicalGraph,
) -> tuple[EmbeddedArticulationGraphChange, ...]:
    """Return the supported exact delta between two canonical graphs."""

    return _canonical_graph_changes(parent, revised)


def load_articulation_revision_human_decisions(
    binding: ExecutionArtifactBinding,
    *,
    graph: EmbeddedArticulationCanonicalGraph,
) -> dict[str, Literal["accept", "reject", "revise"]]:
    """Load one complete digest-bound human decision map for graph revision."""

    return _load_revision_human_decisions(binding, graph=graph)


def _revision_record_matches_patch(
    record: EmbeddedArticulationGraphRevision,
    patch: EmbeddedArticulationGraphRevisionPatch,
) -> bool:
    return (
        record.parent_state_revision == patch.expected_state_revision
        and record.identity_digest == patch.identity_digest
        and record.evidence_digest == patch.evidence_digest
        and record.proposal_digest == patch.proposal_digest
        and record.revision_patch_digest == canonical_json_digest(patch)
        and record.parent_canonical_graph.sha256 == patch.parent_canonical_graph_sha256
        and record.parent_canonical_graph_digest == patch.parent_canonical_graph_digest
        and record.human_decision.sha256 == patch.human_decision_sha256
        and record.human_decisions == patch.human_decisions
        and record.reviewer == patch.reviewer
        and record.revision_reason == patch.revision_reason
        and record.requested_at == patch.requested_at
        and record.changes == patch.changes
        and record.revised_canonical_graph_digest == patch.revised_graph_digest
    )


def apply_embedded_articulation_graph_revision(
    output_dir: str | Path,
    patch: EmbeddedArticulationGraphRevisionPatch,
    *,
    runtime: EmbeddedArticulationRuntime,
) -> ArticulationRunState:
    """Supersede one human-revised graph without weakening the exact review gate."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    latest_revision = (
        _load_bound(
            state.embedded_graph_revision,
            EmbeddedArticulationGraphRevision,
        )
        if state.embedded_graph_revision is not None
        else None
    )
    if latest_revision is not None and _revision_record_matches_patch(
        latest_revision,
        patch,
    ):
        if (
            state.revision != patch.expected_state_revision + 1
            or state.embedded_canonical_graph != latest_revision.revised_canonical_graph
            or state.embedded_outer_review != latest_revision.revised_outer_review
            or state.embedded_coordinator_decision
            != latest_revision.revised_coordinator_decision
            or state.embedded_human_decision is not None
            or state.phase != "needs_review"
        ):
            raise EmbeddedArticulationError(
                "Persisted graph revision is inconsistent with the current checkpoint"
            )
        return state
    if state.revision != patch.expected_state_revision:
        raise EmbeddedArticulationError("Canonical graph revision checkpoint is stale")
    if patch.identity_digest != canonical_json_digest(runtime.identity):
        raise EmbeddedArticulationError("Canonical graph revision identity is stale")
    if state.phase not in {"needs_review", "conditional"}:
        raise EmbeddedArticulationError(
            "Canonical graph revision requires a pending human revise disposition"
        )
    mutation_bindings = (
        state.approved_candidate_document,
        state.authoring_request,
        state.authoring_result,
        state.validation_result,
        state.embedded_execution_authorization,
        state.embedded_readback,
        state.embedded_execution_result,
        state.embedded_output_evidence,
        state.embedded_coordinator_review,
        state.embedded_decision_receipt,
        state.embedded_terminal_receipt,
    )
    if any(binding is not None for binding in mutation_bindings):
        raise EmbeddedArticulationError(
            "Canonical graph revision cannot supersede started authoring or review"
        )
    if (
        state.embedded_evidence is None
        or state.embedded_canonical_graph is None
        or state.embedded_outer_review is None
        or state.embedded_coordinator_decision is None
        or state.embedded_human_decision is None
    ):
        raise EmbeddedArticulationError(
            "Canonical graph revision requires the complete prior review chain"
        )
    store = EmbeddedDecisionArtifactStore(root)
    evidence = _load_store_artifact(
        store,
        state.embedded_evidence,
        EmbeddedDomainEvidence,
    )
    proposals = _load_optional_proposals(store, state)
    parent_graph = _load_bound(
        state.embedded_canonical_graph,
        EmbeddedArticulationCanonicalGraph,
    )
    parent_outer_review = _load_bound(
        state.embedded_outer_review,
        EmbeddedArticulationOuterReview,
    )
    parent_decision = _load_store_artifact(
        store,
        state.embedded_coordinator_decision,
        EmbeddedCoordinatorDecision,
    )
    human = _load_store_artifact(
        store,
        state.embedded_human_decision,
        EmbeddedHumanDecision,
    )
    assert isinstance(evidence, EmbeddedDomainEvidence)
    assert isinstance(parent_graph, EmbeddedArticulationCanonicalGraph)
    assert isinstance(parent_outer_review, EmbeddedArticulationOuterReview)
    assert isinstance(parent_decision, EmbeddedCoordinatorDecision)
    assert isinstance(human, EmbeddedHumanDecision)
    if human.disposition != "revise":
        raise EmbeddedArticulationError(
            "Canonical graph supersession requires an exact human revise decision"
        )
    if human.producer.producer_id != patch.reviewer:
        raise EmbeddedArticulationError(
            "Canonical graph revision reviewer differs from the human decision"
        )
    if patch.requested_at < human.created_at:
        raise EmbeddedArticulationError(
            "Canonical graph revision predates the human revise decision"
        )
    observed_at = _now()
    if patch.requested_at > observed_at:
        raise EmbeddedArticulationError(
            "Canonical graph revision request timestamp is in the future"
        )
    if patch.evidence_digest != artifact_reference(evidence).sha256:
        raise EmbeddedArticulationError("Canonical graph revision evidence is stale")
    expected_proposal_digest = (
        artifact_reference(proposals[-1]).sha256 if proposals else None
    )
    if patch.proposal_digest != expected_proposal_digest:
        raise EmbeddedArticulationError("Canonical graph revision proposal is stale")
    if (
        patch.parent_canonical_graph_sha256 != state.embedded_canonical_graph.sha256
        or patch.parent_canonical_graph_digest != canonical_json_digest(parent_graph)
        or patch.human_decision_sha256 != state.embedded_human_decision.sha256
    ):
        raise EmbeddedArticulationError(
            "Canonical graph revision does not bind the exact parent review"
        )
    _validate_outer_review_chain(
        state,
        runtime=runtime,
        evidence=evidence,
        proposals=proposals,
        decision=parent_decision,
        graph=parent_graph,
    )
    human_decisions = _load_revision_human_decisions(
        patch.human_decisions,
        graph=parent_graph,
    )
    if "reject" in human_decisions.values():
        raise EmbeddedArticulationError(
            "Rejected human candidates cannot enter graph supersession"
        )
    derived_changes = _canonical_graph_changes(parent_graph, patch.revised_graph)
    if derived_changes != patch.changes:
        raise EmbeddedArticulationError(
            "Declared graph revision changes differ from the exact graph delta"
        )
    changed_candidates = {item.candidate_id for item in derived_changes}
    revise_candidates = {
        candidate_id
        for candidate_id, disposition in human_decisions.items()
        if disposition == "revise"
    }
    if changed_candidates != revise_candidates:
        raise EmbeddedArticulationError(
            "Graph revision changes must exactly match human-revised candidates"
        )
    _validate_canonical_graph(
        patch.revised_graph,
        state=state,
        evidence=evidence,
        capabilities=runtime.capabilities,
        authoring_required=True,
    )
    revision_number = latest_revision.revision + 1 if latest_revision is not None else 2
    revision_dir = root / "graph_revisions" / f"revision-{revision_number:03d}"
    revision_dir.mkdir(parents=True, exist_ok=True)
    revised_graph_path = revision_dir / "canonical_articulation_graph.json"
    if revised_graph_path.exists():
        existing_graph = EmbeddedArticulationCanonicalGraph.model_validate_json(
            revised_graph_path.read_bytes()
        )
        if existing_graph != patch.revised_graph:
            raise EmbeddedArticulationError(
                "Canonical graph revision path contains different bytes"
            )
    else:
        atomic_write_json(revised_graph_path, patch.revised_graph)
    revised_graph_binding = _binding(revised_graph_path)
    human_review = parent_outer_review.human_review
    if human_review.status != "human_required":
        raise EmbeddedArticulationError(
            "Canonical graph revision cannot invent a new human review policy"
        )
    outer_review = EmbeddedArticulationOuterReview(
        identity_digest=patch.identity_digest,
        reviewer=runtime.outer_coordinator,
        evidence_digest=patch.evidence_digest,
        proposal_digest=patch.proposal_digest,
        canonical_graph_sha256=revised_graph_binding.sha256,
        disposition="accept",
        human_review=human_review,
        rationale=patch.revision_reason,
    )
    outer_review_path = revision_dir / "embedded_articulation_outer_review.json"
    if outer_review_path.exists():
        existing_outer_review = EmbeddedArticulationOuterReview.model_validate_json(
            outer_review_path.read_bytes()
        )
        if existing_outer_review != outer_review:
            raise EmbeddedArticulationError(
                "Graph revision outer review path contains different bytes"
            )
    else:
        atomic_write_json(outer_review_path, outer_review)
    outer_review_binding = _binding(outer_review_path)
    semantic = AcceptedSemanticDecision(
        schema_version=EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION,
        values={
            "canonical_graph": patch.revised_graph.model_dump(mode="json"),
            "human_review": human_review.model_dump(mode="json"),
        },
    )
    decision_artifact_id = (
        f"articulation-canonical-graph-decision-r{revision_number:03d}"
    )
    existing_decision: EmbeddedCoordinatorDecision | None = None
    for entry in store.journal().entries:
        if (
            entry.reference.artifact_kind == "coordinator_decision"
            and entry.reference.artifact_id == decision_artifact_id
        ):
            existing_decision = store.load_typed(
                entry.reference,
                EmbeddedCoordinatorDecision,
            )
            break
    commit_time = (
        existing_decision.created_at if existing_decision is not None else observed_at
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id=decision_artifact_id,
        identity=runtime.identity,
        producer=runtime.outer_coordinator,
        parent_artifact=(
            artifact_reference(proposals[-1])
            if proposals
            else artifact_reference(evidence)
        ),
        created_at=commit_time,
        disposition="accept",
        evidence_artifacts=(artifact_reference(evidence),),
        proposal_artifacts=tuple(artifact_reference(item) for item in proposals),
        accepted_decision=semantic,
        accepted_decision_digest=accepted_semantic_decision_digest(
            runtime.identity,
            semantic,
        ),
        human_decision_required=True,
        rationale=patch.revision_reason,
    )
    if existing_decision is not None and existing_decision != decision:
        raise EmbeddedArticulationError(
            "Persisted revised graph decision differs from the exact request"
        )
    decision_commit = store.append(decision)
    decision_binding = _store_binding(store, decision_commit.relative_path)
    revision = EmbeddedArticulationGraphRevision(
        revision=revision_number,
        parent_state_revision=patch.expected_state_revision,
        requested_at=patch.requested_at,
        created_at=commit_time,
        identity_digest=patch.identity_digest,
        evidence_digest=patch.evidence_digest,
        proposal_digest=patch.proposal_digest,
        revision_patch_digest=canonical_json_digest(patch),
        parent_revision=state.embedded_graph_revision,
        parent_canonical_graph=state.embedded_canonical_graph,
        parent_canonical_graph_digest=patch.parent_canonical_graph_digest,
        parent_outer_review=state.embedded_outer_review,
        parent_coordinator_decision=state.embedded_coordinator_decision,
        human_decision=state.embedded_human_decision,
        human_decisions=patch.human_decisions,
        reviewer=patch.reviewer,
        revision_reason=patch.revision_reason,
        changes=patch.changes,
        revised_canonical_graph=revised_graph_binding,
        revised_canonical_graph_digest=patch.revised_graph_digest,
        revised_outer_review=outer_review_binding,
        revised_coordinator_decision=decision_binding,
    )
    revision_path = revision_dir / "graph_revision.json"
    if revision_path.exists():
        existing_revision = EmbeddedArticulationGraphRevision.model_validate_json(
            revision_path.read_bytes()
        )
        if existing_revision != revision:
            raise EmbeddedArticulationError(
                "Canonical graph revision receipt contains different bytes"
            )
    else:
        atomic_write_json(revision_path, revision)
    revision_binding = _binding(revision_path)
    state = _transition(
        state.model_copy(
            update={
                "embedded_canonical_graph": revised_graph_binding,
                "embedded_graph_revision": revision_binding,
                "embedded_outer_review": outer_review_binding,
                "embedded_coordinator_decision": decision_binding,
                "embedded_human_decision": None,
                "candidate_ids": patch.revised_graph.candidate_ids,
                "review_required_candidate_ids": patch.revised_graph.candidate_ids,
                "accepted_candidate_ids": (),
                "rejected_candidate_ids": (),
                "unresolved_candidate_ids": (),
                "error": None,
            }
        ),
        "needs_review",
        (
            "Human-revised canonical graph was superseded in immutable revision "
            f"{revision_number}; exact revised-digest review is required."
        ),
    )
    return _persist_state(root, state)


def _validate_graph_revision_chain(
    state: ArticulationRunState,
    *,
    runtime: EmbeddedArticulationRuntime,
) -> None:
    """Revalidate every immutable parent graph, review, and exact change record."""

    if state.embedded_graph_revision is None:
        return
    if (
        state.embedded_evidence is None
        or state.embedded_canonical_graph is None
        or state.embedded_outer_review is None
        or state.embedded_coordinator_decision is None
    ):
        raise EmbeddedArticulationError(
            "Canonical graph revision chain lacks its active review bindings"
        )
    root = Path(state.request.path).expanduser().resolve().parent
    store = EmbeddedDecisionArtifactStore(root)
    evidence = _load_store_artifact(
        store,
        state.embedded_evidence,
        EmbeddedDomainEvidence,
    )
    proposals = _load_optional_proposals(store, state)
    assert isinstance(evidence, EmbeddedDomainEvidence)
    expected_evidence_digest = artifact_reference(evidence).sha256
    expected_proposal_digest = (
        artifact_reference(proposals[-1]).sha256 if proposals else None
    )
    current_revision_binding = state.embedded_graph_revision
    expected_revised_graph = state.embedded_canonical_graph
    expected_revised_outer_review = state.embedded_outer_review
    expected_revised_decision = state.embedded_coordinator_decision
    seen: set[str] = set()
    while current_revision_binding is not None:
        if current_revision_binding.sha256 in seen:
            raise EmbeddedArticulationError("Canonical graph revision chain is cyclic")
        seen.add(current_revision_binding.sha256)
        record = _load_bound(
            current_revision_binding,
            EmbeddedArticulationGraphRevision,
        )
        assert isinstance(record, EmbeddedArticulationGraphRevision)
        if (
            record.identity_digest != canonical_json_digest(runtime.identity)
            or record.revised_canonical_graph != expected_revised_graph
            or record.revised_outer_review != expected_revised_outer_review
            or record.revised_coordinator_decision != expected_revised_decision
        ):
            raise EmbeddedArticulationError(
                "Canonical graph revision chain is stale or disconnected"
            )
        parent_graph = _load_bound(
            record.parent_canonical_graph,
            EmbeddedArticulationCanonicalGraph,
        )
        revised_graph = _load_bound(
            record.revised_canonical_graph,
            EmbeddedArticulationCanonicalGraph,
        )
        parent_outer = _load_bound(
            record.parent_outer_review,
            EmbeddedArticulationOuterReview,
        )
        revised_outer = _load_bound(
            record.revised_outer_review,
            EmbeddedArticulationOuterReview,
        )
        parent_decision = _load_store_artifact(
            store,
            record.parent_coordinator_decision,
            EmbeddedCoordinatorDecision,
        )
        revised_decision = _load_store_artifact(
            store,
            record.revised_coordinator_decision,
            EmbeddedCoordinatorDecision,
        )
        human = _load_store_artifact(
            store,
            record.human_decision,
            EmbeddedHumanDecision,
        )
        assert isinstance(parent_graph, EmbeddedArticulationCanonicalGraph)
        assert isinstance(revised_graph, EmbeddedArticulationCanonicalGraph)
        assert isinstance(parent_outer, EmbeddedArticulationOuterReview)
        assert isinstance(revised_outer, EmbeddedArticulationOuterReview)
        assert isinstance(parent_decision, EmbeddedCoordinatorDecision)
        assert isinstance(revised_decision, EmbeddedCoordinatorDecision)
        assert isinstance(human, EmbeddedHumanDecision)
        reconstructed_patch = EmbeddedArticulationGraphRevisionPatch(
            expected_state_revision=record.parent_state_revision,
            identity_digest=record.identity_digest,
            evidence_digest=record.evidence_digest,
            proposal_digest=record.proposal_digest,
            parent_canonical_graph_sha256=record.parent_canonical_graph.sha256,
            parent_canonical_graph_digest=record.parent_canonical_graph_digest,
            human_decision_sha256=record.human_decision.sha256,
            human_decisions=record.human_decisions,
            reviewer=record.reviewer,
            revision_reason=record.revision_reason,
            requested_at=record.requested_at,
            changes=record.changes,
            revised_graph=revised_graph,
            revised_graph_digest=record.revised_canonical_graph_digest,
        )
        if (
            record.parent_canonical_graph_digest != canonical_json_digest(parent_graph)
            or record.revised_canonical_graph_digest
            != canonical_json_digest(revised_graph)
            or record.requested_at < human.created_at
            or record.created_at < record.requested_at
            or human.disposition != "revise"
            or human.producer.producer_id != record.reviewer
            or record.evidence_digest != expected_evidence_digest
            or record.proposal_digest != expected_proposal_digest
            or record.revision_patch_digest
            != canonical_json_digest(reconstructed_patch)
            or human.coordinator_decision != artifact_reference(parent_decision)
            or parent_outer.canonical_graph_sha256
            != record.parent_canonical_graph.sha256
            or revised_outer.canonical_graph_sha256
            != record.revised_canonical_graph.sha256
            or revised_outer.disposition != "accept"
            or revised_outer.human_review.status != "human_required"
            or revised_decision.disposition != "accept"
            or revised_decision.created_at != record.created_at
            or not revised_decision.human_decision_required
        ):
            raise EmbeddedArticulationError(
                "Canonical graph revision provenance is stale or incomplete"
            )
        parent_semantic = AcceptedSemanticDecision(
            schema_version=EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION,
            values={
                "canonical_graph": parent_graph.model_dump(mode="json"),
                "human_review": parent_outer.human_review.model_dump(mode="json"),
            },
        )
        revised_semantic = AcceptedSemanticDecision(
            schema_version=EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION,
            values={
                "canonical_graph": revised_graph.model_dump(mode="json"),
                "human_review": revised_outer.human_review.model_dump(mode="json"),
            },
        )
        if (
            parent_decision.accepted_decision != parent_semantic
            or parent_decision.accepted_decision_digest
            != accepted_semantic_decision_digest(runtime.identity, parent_semantic)
            or revised_decision.accepted_decision != revised_semantic
            or revised_decision.accepted_decision_digest
            != accepted_semantic_decision_digest(runtime.identity, revised_semantic)
            or _canonical_graph_changes(parent_graph, revised_graph) != record.changes
        ):
            raise EmbeddedArticulationError(
                "Canonical graph revision semantics differ from the recorded delta"
            )
        decisions = _load_revision_human_decisions(
            record.human_decisions,
            graph=parent_graph,
        )
        changed_candidates = {item.candidate_id for item in record.changes}
        revise_candidates = {
            candidate_id
            for candidate_id, disposition in decisions.items()
            if disposition == "revise"
        }
        if "reject" in decisions.values() or changed_candidates != revise_candidates:
            raise EmbeddedArticulationError(
                "Canonical graph revision no longer matches the exact human review"
            )
        if record.parent_revision is None:
            break
        parent_record = _load_bound(
            record.parent_revision,
            EmbeddedArticulationGraphRevision,
        )
        assert isinstance(parent_record, EmbeddedArticulationGraphRevision)
        if (
            parent_record.revision + 1 != record.revision
            or parent_record.revised_canonical_graph != record.parent_canonical_graph
            or parent_record.revised_outer_review != record.parent_outer_review
            or parent_record.revised_coordinator_decision
            != record.parent_coordinator_decision
        ):
            raise EmbeddedArticulationError(
                "Canonical graph revision parent link is stale"
            )
        current_revision_binding = record.parent_revision
        expected_revised_graph = record.parent_canonical_graph
        expected_revised_outer_review = record.parent_outer_review
        expected_revised_decision = record.parent_coordinator_decision


def _human_disposition(
    graph: EmbeddedArticulationCanonicalGraph,
    acceptance: EmbeddedArticulationHumanAcceptance,
) -> tuple[Literal["accept", "reject", "revise"], tuple[str, ...]]:
    if acceptance.canonical_graph_sha256 != canonical_json_digest(graph):
        raise EmbeddedArticulationError("Human gate canonical graph digest is stale")
    if tuple(acceptance.decisions) != graph.candidate_ids or set(
        acceptance.decisions
    ) != set(graph.candidate_ids):
        raise EmbeddedArticulationError("Human decisions do not cover the exact graph")
    decisions = tuple(acceptance.decisions[item] for item in graph.candidate_ids)
    if "revise" in decisions:
        return "revise", ("Human review requested canonical graph revision.",)
    if "reject" in decisions:
        return "reject", ()
    return "accept", ()


def _persist_human_decision(
    root: Path,
    state: ArticulationRunState,
    runtime: EmbeddedArticulationRuntime,
    acceptance: EmbeddedArticulationHumanAcceptance,
) -> tuple[ArticulationRunState, EmbeddedHumanDecision]:
    if (
        state.embedded_canonical_graph is None
        or state.embedded_coordinator_decision is None
    ):
        raise EmbeddedArticulationError(
            "Human review requires a persisted canonical graph"
        )
    graph = _load_bound(
        state.embedded_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    decision = _load_bound(
        state.embedded_coordinator_decision, EmbeddedCoordinatorDecision
    )
    assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
    assert isinstance(decision, EmbeddedCoordinatorDecision)
    disposition, revisions = _human_disposition(graph, acceptance)
    store = EmbeddedDecisionArtifactStore(root)
    revision_record = (
        _load_bound(
            state.embedded_graph_revision,
            EmbeddedArticulationGraphRevision,
        )
        if state.embedded_graph_revision is not None
        else None
    )
    revision_number = revision_record.revision if revision_record is not None else 1
    artifact_id = (
        "articulation-human-graph-decision"
        if revision_number == 1
        else f"articulation-human-graph-decision-r{revision_number:03d}"
    )
    producer = ProducerIdentity(
        producer_id=acceptance.reviewer,
        role="human_reviewer",
        implementation="asset-composition-human-gate",
    )
    parent = artifact_reference(decision)
    expected_fields = {
        "identity": runtime.identity,
        "producer": producer,
        "parent_artifact": parent,
        "coordinator_decision": parent,
        "reviewed_decision_digest": _required_accepted_decision_digest(decision),
        "disposition": disposition,
        "rationale": f"Exact canonical graph reviewed by {acceptance.reviewer}.",
        "revision_requests": revisions,
    }
    existing_human: EmbeddedHumanDecision | None = None
    for entry in store.journal().entries:
        if (
            entry.reference.artifact_kind == "human_decision"
            and entry.reference.artifact_id == artifact_id
        ):
            existing_human = store.load_typed(
                entry.reference,
                EmbeddedHumanDecision,
            )
            break
    if existing_human is not None:
        if any(
            getattr(existing_human, field_name) != expected_value
            for field_name, expected_value in expected_fields.items()
        ):
            raise EmbeddedArticulationError(
                "Persisted human graph decision differs from the exact review"
            )
        human = existing_human
    else:
        human = EmbeddedHumanDecision(
            artifact_id=artifact_id,
            identity=runtime.identity,
            producer=producer,
            parent_artifact=parent,
            created_at=_now(),
            coordinator_decision=parent,
            reviewed_decision_digest=_required_accepted_decision_digest(decision),
            disposition=disposition,
            rationale=f"Exact canonical graph reviewed by {acceptance.reviewer}.",
            revision_requests=revisions,
        )
    commit = store.append(human)
    state = _persist_state(
        root,
        state.model_copy(
            update={
                "embedded_human_decision": _store_binding(store, commit.relative_path)
            }
        ),
    )
    return state, human


def _graph_candidate_document(
    graph: EmbeddedArticulationCanonicalGraph,
) -> Stage2CandidateDocument:
    candidates: list[Stage2ArticulationCandidate] = []
    for joint in graph.joints:
        axis = _AXES[joint.axis]
        common_evidence = (
            Stage2EvidenceItem(
                source="accepted_manifest",
                description=(
                    "Exact outer-accepted canonical graph; human acceptance is "
                    "also bound when frozen policy requires it."
                ),
                value=joint.axis,
                prim_paths=(joint.body0_owner_prim, joint.body1_owner_prim),
            ),
        )
        connectivity = (
            Stage2EvidenceItem(
                source="accepted_manifest",
                description="Canonical authoritative-owner topology edge.",
                value=joint.body0_owner_prim,
                prim_paths=(joint.body0_owner_prim, joint.body1_owner_prim),
                connectivity_role="body0_body1_edge",
            ),
            Stage2EvidenceItem(
                source="accepted_manifest",
                description="Canonical moving authoritative owner.",
                value=joint.body1_owner_prim,
                prim_paths=(joint.body1_owner_prim,),
                connectivity_role="body1_ownership",
            ),
        )
        has_limits = joint.lower_limit is not None or joint.upper_limit is not None
        candidates.append(
            Stage2ArticulationCandidate(
                candidate_id=joint.joint_id,
                motion_type=joint.joint_type,
                moving_part_prims=(joint.body1_owner_prim,),
                fixed_parent_prim=joint.body0_owner_prim,
                parent_resolution_source="accepted_manifest",
                joint_type_hint=joint.joint_type,
                axis_hint=joint.axis,
                motion_axis_world=axis,
                confidence="high",
                parent_hint=joint.body0_role,
                child_hint=joint.body1_role,
                component_name=joint.body1_role,
                component_type="articulated_owner",
                role=joint.body1_role,
                source_prediction_ids=(),
                evidence=(
                    "Exact outer-accepted canonical graph with policy-bound review."
                ),
                field_sources={
                    "motion_type": "accepted_manifest",
                    "axis_hint": "accepted_manifest",
                    "motion_axis_world": "accepted_manifest",
                    "fixed_parent_prim": "accepted_manifest",
                },
                axis_evidence=common_evidence,
                connectivity_evidence=connectivity,
                lower_limit=joint.lower_limit,
                upper_limit=joint.upper_limit,
                limit_unit=joint.limit_unit if has_limits else "unknown",
                limit_source="accepted_manifest" if has_limits else "unknown",
                limit_readiness="source_backed" if has_limits else "not_provided",
                limit_evidence=common_evidence if has_limits else (),
                review_status="ready_for_rigger_input",
            )
        )
    return Stage2CandidateDocument(
        summary=Stage2CandidateSummary(
            candidate_count=len(candidates),
            ready_candidate_count=len(candidates),
            review_required_candidate_count=0,
        ),
        candidates=tuple(candidates),
    )


def _authoring_key(
    state: ArticulationRunState,
    graph_digest: str,
    output_dir: Path,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.embedded-articulation-authoring-key.v1",
            "source_sha256": state.source_sha256,
            "dependency_sha256": state.source_dependency_bundle_sha256,
            "canonical_graph_digest": graph_digest,
            "output_dir": str(output_dir),
            "predictions_path": None,
        }
    )


def _validate_saved_membership(
    graph: EmbeddedArticulationCanonicalGraph,
    output_asset: Path,
) -> Literal[True]:
    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise EmbeddedArticulationError(
            "OpenUSD is required for saved membership readback"
        ) from exc
    stage = Usd.Stage.Open(str(output_asset))
    if stage is None:
        raise EmbeddedArticulationError(
            "Saved articulation stage could not be reopened"
        )
    missing = sorted(
        path
        for path in {*graph.source_member_prims, *graph.authoritative_owner_prims}
        if not stage.GetPrimAtPath(path).IsValid()
    )
    if missing:
        raise EmbeddedArticulationError(
            "Saved stage lost canonical members or owners: " + ", ".join(missing)
        )
    for membership in graph.memberships:
        if membership.disposition != "co_rigid":
            continue
        member = stage.GetPrimAtPath(membership.member_prim)
        owner = stage.GetPrimAtPath(membership.authoritative_owner_prim)
        if not member.GetPath().HasPrefix(owner.GetPath()):
            raise EmbeddedArticulationError(
                "Saved stage lost the accepted co-rigid owner containment"
            )
    for operation in graph.rigid_link_operations:
        body = stage.GetPrimAtPath(operation.body_prim_path)
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise EmbeddedArticulationError(
                "Saved stage lacks the accepted rigid-link body membership: "
                f"{operation.body_prim_path}"
            )
    return True


def _readback_topology(
    graph: EmbeddedArticulationCanonicalGraph,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "joint_id": item.joint_id,
            "body0_owner_prim": item.body0_owner_prim,
            "body1_owner_prim": item.body1_owner_prim,
            "joint_type": item.joint_type,
            "axis": item.axis,
            "lower_limit": item.lower_limit,
            "upper_limit": item.upper_limit,
            "limit_unit": item.limit_unit,
            "frame_policy": item.frame_policy,
            "body0_role": item.body0_role,
            "body1_role": item.body1_role,
        }
        for item in graph.joints
    )


def _readback_groups(
    graph: EmbeddedArticulationCanonicalGraph,
) -> tuple[dict[str, Any], ...]:
    return tuple(item.model_dump(mode="json") for item in graph.groups)


def _readback_memberships(
    graph: EmbeddedArticulationCanonicalGraph,
) -> tuple[dict[str, Any], ...]:
    return tuple(item.model_dump(mode="json") for item in graph.memberships)


def _build_readback(
    root: Path,
    graph: EmbeddedArticulationCanonicalGraph,
    decision: EmbeddedCoordinatorDecision,
    authoring: ArticulationAuthoringResult,
    validation: ArticulationValidationResult,
    validation_binding: ArtifactBinding,
) -> EmbeddedArticulationReadback:
    if (
        validation.status != "pass"
        or validation.expected_candidate_ids != graph.candidate_ids
    ):
        raise EmbeddedArticulationError("Saved-stage exact topology readback failed")
    if authoring.authored_candidate_ids != graph.candidate_ids:
        raise EmbeddedArticulationError(
            "Authoring result differs from the accepted graph"
        )
    exact_membership_match = _validate_saved_membership(
        graph, Path(authoring.output_asset_path)
    )
    return EmbeddedArticulationReadback(
        canonical_graph_digest=canonical_json_digest(graph),
        accepted_decision_digest=_required_accepted_decision_digest(decision),
        output_asset_path=authoring.output_asset_path,
        output_asset_sha256=authoring.output_asset_sha256,
        validation_sha256=validation_binding.sha256,
        joint_ids=graph.candidate_ids,
        topology=_readback_topology(graph),
        groups=_readback_groups(graph),
        memberships=_readback_memberships(graph),
        exact_topology_match=True,
        exact_membership_match=exact_membership_match,
        exact_co_rigid_disposition_match=exact_membership_match,
    )


def _execute_graph(
    root: Path,
    state: ArticulationRunState,
    runtime: EmbeddedArticulationRuntime,
    human: EmbeddedHumanDecision | None,
    *,
    client: ArticulationAuthoringClient,
    cancel_checker: CancelChecker | None,
) -> tuple[ArticulationRunState, ArticulationAuthoringResult]:
    store = EmbeddedDecisionArtifactStore(root)
    evidence = _load_store_artifact(
        store, state.embedded_evidence, EmbeddedDomainEvidence
    )
    proposals = _load_optional_proposals(store, state)
    decision = _load_store_artifact(
        store,
        state.embedded_coordinator_decision,
        EmbeddedCoordinatorDecision,
    )
    graph = _load_bound(
        state.embedded_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    assert isinstance(evidence, EmbeddedDomainEvidence)
    assert isinstance(decision, EmbeddedCoordinatorDecision)
    assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
    _validate_outer_review_chain(
        state,
        runtime=runtime,
        evidence=evidence,
        proposals=proposals,
        decision=decision,
        graph=graph,
    )
    _validate_graph_revision_chain(state, runtime=runtime)
    approved = _graph_candidate_document(graph)
    approved_path = root / "approved_articulation_candidates.json"
    atomic_write_json(approved_path, approved)
    approved_binding = _binding(approved_path)
    authoring_request = ArticulationAuthoringRequest(
        source_asset=state.source_asset,
        source_sha256=state.source_sha256,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        candidate_document_path=approved_binding.path,
        candidate_document_sha256=approved_binding.sha256,
        accepted_candidate_ids=graph.candidate_ids,
        idempotency_key=_authoring_key(state, canonical_json_digest(graph), root),
        predictions_path=None,
        predictions_sha256=None,
        output_dir=root,
    )
    if authoring_request.predictions_path is not None:
        raise EmbeddedArticulationError("Embedded graph authoring forbids predictions")
    request_path = root / "authoring_request.json"
    atomic_write_json(request_path, authoring_request)
    request_binding = _binding(request_path)
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=runtime.identity,
        evidence=(evidence,),
        proposals=proposals,
        existing_authorizations=(),
        executor=runtime.executor,
        execution_effect="mutation",
        human_decision=human,
    )

    def mutate(
        _authorization: BoundedExecutionAuthorization,
    ) -> tuple[
        ArticulationAuthoringResult, ArticulationValidationResult, ArtifactBinding
    ]:
        authored = client.author(authoring_request, cancel_checker=cancel_checker)
        author_path = root / "authoring_result.json"
        atomic_write_json(author_path, authored)
        validation = client.validate(
            authored,
            expected_candidate_ids=graph.candidate_ids,
            cancel_checker=cancel_checker,
        )
        validation_path = root / "validation_evidence.json"
        atomic_write_json(validation_path, validation)
        validation_binding = _binding(validation_path)
        readback = _build_readback(
            root, graph, decision, authored, validation, validation_binding
        )
        readback_path = root / "embedded_articulation_readback.json"
        atomic_write_json(readback_path, readback)
        return authored, validation, _binding(readback_path)

    authored, validation, readback_binding = store.invoke_after_authorization_commit(
        authorization, mutate
    )
    authorization_ref = artifact_reference(authorization)
    authorization_binding = _binding(
        store.store_root
        / "artifacts"
        / authorization_ref.artifact_kind
        / f"{authorization_ref.sha256}.json"
    )
    author_binding = _binding(root / "authoring_result.json")
    validation_binding = _binding(root / "validation_evidence.json")
    outputs = (
        _execution_binding(authored.output_asset_path),
        _execution_binding(author_binding.path),
        _execution_binding(validation_binding.path),
        _execution_binding(readback_binding.path),
    )
    result = EmbeddedBoundedExecutionResult(
        artifact_id="articulation-graph-execution-result",
        identity=runtime.identity,
        producer=runtime.executor,
        parent_artifact=authorization_ref,
        created_at=_now(),
        accepted_decision=artifact_reference(decision),
        accepted_decision_digest=_required_accepted_decision_digest(decision),
        execution_effect="mutation",
        operation_id=authorization.operation_id,
        mutation_id=authorization.mutation_id,
        attempt=authorization.attempt,
        status="succeeded",
        mutation_state="verified",
        outputs=outputs,
        evidence=(
            ProviderNeutralEvidenceRecord(
                evidence_id="saved-articulation-readback",
                evidence_type="validation",
                status="available",
                summary="Exact saved topology, frames, limits, membership, and co-rigid disposition.",
                artifacts=outputs,
                facts={"canonical_graph_digest": canonical_json_digest(graph)},
            ),
        ),
    )
    validate_bounded_execution_result(result, authorization)
    result_commit = store.append_result(result)
    state = state.model_copy(
        update={
            "approved_candidate_document": approved_binding,
            "authoring_request": request_binding,
            "authoring_result": author_binding,
            "validation_result": validation_binding,
            "embedded_execution_authorization": authorization_binding,
            "embedded_readback": readback_binding,
            "embedded_execution_result": _store_binding(
                store, result_commit.relative_path
            ),
            "accepted_candidate_ids": graph.candidate_ids,
            "review_required_candidate_ids": graph.candidate_ids,
            "error": None,
        }
    )
    state = _transition(
        state,
        "awaiting_post_review",
        "Exact saved-stage readback requires outer-coordinator post review.",
    )
    return _persist_state(root, state), authored


def _validate_runtime_identity(
    state: ArticulationRunState,
    request: ArticulationWorkflowRequest,
    runtime: EmbeddedArticulationRuntime,
) -> None:
    if runtime.identity.execution_context != request.execution_context:
        raise EmbeddedArticulationError(
            "Embedded Articulation runtime context is stale"
        )
    if runtime.identity.source != state_to_source_binding(state):
        raise EmbeddedArticulationError(
            "Embedded Articulation runtime source identity is stale"
        )
    if state.backend_configuration_sha256 not in set(
        runtime.identity.digests.configuration.values()
    ):
        raise EmbeddedArticulationError(
            "Embedded Articulation configuration is not identity-bound"
        )
    capability_digest = canonical_json_digest(runtime.capabilities)
    if capability_digest not in set(runtime.identity.digests.capabilities.values()):
        raise EmbeddedArticulationError(
            "Embedded Articulation capabilities are not identity-bound"
        )


def prepare_embedded_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    mode: ArticulationWorkflowMode,
    runtime: EmbeddedArticulationRuntime,
    preparation: EmbeddedArticulationPreparation,
) -> ArticulationFinalizationResult:
    """Initialize one embedded run from deterministic provider-neutral evidence."""

    if (
        request.execution_context is None
        or request.execution_context.mode != "embedded"
    ):
        raise EmbeddedArticulationError(
            "Provider-neutral Articulation preparation requires embedded context"
        )
    from .workflow import _normalize_request, _source_identity, _write_once_json

    normalized_request = _normalize_request(request)
    root = normalized_request.output_dir
    root.mkdir(parents=True, exist_ok=True)
    request_binding = _write_once_json(
        root / "request.json",
        normalized_request,
        label="articulation request",
    )
    source_sha256, dependency_sha256 = _source_identity(normalized_request.source_asset)
    if (
        preparation.source_sha256 != source_sha256
        or preparation.source_dependency_bundle_sha256 != dependency_sha256
    ):
        raise EmbeddedArticulationError(
            "Provider-neutral Articulation preparation source identity is stale"
        )
    checkpoint = root / "checkpoint.json"
    if checkpoint.exists():
        state = _state(root)
        if (
            state.mode != mode
            or state.request != request_binding
            or state.source_asset != normalized_request.source_asset
            or state.source_sha256 != source_sha256
            or state.source_dependency_bundle_sha256 != dependency_sha256
            or state.backend_configuration_sha256 != preparation.configuration_sha256
        ):
            raise EmbeddedArticulationError(
                "Provider-neutral Articulation preparation differs from checkpoint"
            )
    else:
        state = ArticulationRunState(
            schema_version=ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
            mode=mode,
            request=request_binding,
            source_asset=normalized_request.source_asset,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=dependency_sha256,
            backend_configuration_sha256=preparation.configuration_sha256,
            scene_evidence_configuration_sha256=canonical_json_digest(
                preparation.scene
            ),
        )
        state = _persist_state(root, state)
    _validate_runtime_identity(state, normalized_request, runtime)
    preparation_binding = _write_once_json(
        root / "embedded_articulation_preparation.json",
        preparation,
        label="embedded Articulation preparation",
    )
    if state.embedded_evidence is None:
        if state.phase == "initialized":
            state = _transition(
                state,
                "awaiting_decision",
                "Deterministic Articulation evidence awaits the outer canonical graph.",
            )
            state = _persist_state(root, state)
        if state.phase not in {"awaiting_decision", "needs_review"}:
            raise EmbeddedArticulationError(
                "Provider-neutral preparation cannot replace active workflow state"
            )
        state = _initialize_contract_artifacts(
            root,
            state,
            runtime,
            preparation,
            preparation_binding,
        )
    else:
        _validate_preparation(state, preparation, runtime)
    authoring = (
        _load_bound(state.authoring_result, ArticulationAuthoringResult)
        if state.phase == "completed" and state.authoring_result is not None
        else None
    )
    return write_articulation_workflow_summary(
        state,
        output_dir=root,
        authoring=(
            authoring if isinstance(authoring, ArticulationAuthoringResult) else None
        ),
    )


def run_embedded_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    mode: ArticulationWorkflowMode,
    client: ArticulationAuthoringClient | None,
    runtime: EmbeddedArticulationRuntime,
    human_acceptance_loader: HumanAcceptanceLoader,
    preparation: EmbeddedArticulationPreparation | None = None,
    scene_evidence_collector: ArticulationSceneEvidenceCollector | None = None,
    cancel_checker: CancelChecker | None = None,
    phase_boundary_hook: Callable[[str], None] | None = None,
) -> ArticulationFinalizationResult:
    """Advance only the embedded articulation controller on the real client path."""

    if (
        request.execution_context is None
        or request.execution_context.mode != "embedded"
    ):
        raise EmbeddedArticulationError(
            "Embedded articulation requires embedded context"
        )
    root = request.output_dir.expanduser().resolve()
    checkpoint = root / "checkpoint.json"
    if not checkpoint.exists():
        if preparation is not None:
            prepare_embedded_articulation_workflow(
                request,
                mode=mode,
                runtime=runtime,
                preparation=preparation,
            )
        else:
            if client is None:
                raise EmbeddedArticulationError(
                    "Provider-backed preparation requires an inference client"
                )
            from .workflow import run_articulation_workflow

            run_articulation_workflow(
                request,
                mode=mode,
                client=cast(ArticulationWorkflowClient, client),
                scene_evidence_collector=scene_evidence_collector,
                cancel_checker=cancel_checker,
                phase_boundary_hook=phase_boundary_hook,
            )
    state = _state(root)
    if preparation is not None:
        # Revalidate the public preparation, runtime identity, and immutable
        # persisted bytes on every retry before reading any prior decision.
        prepare_embedded_articulation_workflow(
            request,
            mode=mode,
            runtime=runtime,
            preparation=preparation,
        )
        state = _state(root)
    if state.embedded_evidence is None:
        if preparation is not None:
            return prepare_embedded_articulation_workflow(
                request,
                mode=mode,
                runtime=runtime,
                preparation=preparation,
            )
        if state.phase != "needs_review":
            return write_articulation_workflow_summary(state, output_dir=root)
        inference = _load_bound(state.inference_result, ArticulationInferenceResult)
        assert isinstance(inference, ArticulationInferenceResult)
        provider_preparation = _provider_backed_preparation(state, inference, runtime)
        from .workflow import _write_once_json

        preparation_binding = _write_once_json(
            root / "embedded_articulation_preparation.json",
            provider_preparation,
            label="embedded Articulation preparation",
        )
        state = _initialize_contract_artifacts(
            root,
            state,
            runtime,
            provider_preparation,
            preparation_binding,
        )
    if state.phase in {
        "completed",
        "not_articulated",
        "conditional",
        "cancelled",
        "failed",
    }:
        authoring = (
            _load_bound(state.authoring_result, ArticulationAuthoringResult)
            if state.authoring_result is not None
            else None
        )
        return write_articulation_workflow_summary(
            state,
            output_dir=root,
            authoring=authoring
            if isinstance(authoring, ArticulationAuthoringResult)
            else None,
        )
    if state.embedded_coordinator_decision is None:
        return write_articulation_workflow_summary(state, output_dir=root)
    if state.embedded_canonical_graph is None:
        raise EmbeddedArticulationError(
            "Embedded coordinator decision lacks its canonical graph binding"
        )
    if state.embedded_execution_result is not None:
        authoring = _load_bound(state.authoring_result, ArticulationAuthoringResult)
        assert isinstance(authoring, ArticulationAuthoringResult)
        return write_articulation_workflow_summary(
            state, output_dir=root, authoring=authoring
        )
    decision = _load_bound(
        state.embedded_coordinator_decision,
        EmbeddedCoordinatorDecision,
    )
    assert isinstance(decision, EmbeddedCoordinatorDecision)
    if decision.disposition != "accept":
        raise EmbeddedArticulationError(
            "Only an outer-accepted Articulation graph can authorize mutation"
        )
    human: EmbeddedHumanDecision | None = None
    if decision.human_decision_required:
        if state.embedded_human_decision is None:
            acceptance = human_acceptance_loader(root, state.embedded_canonical_graph)
            if acceptance is None:
                return write_articulation_workflow_summary(state, output_dir=root)
            state, human = _persist_human_decision(root, state, runtime, acceptance)
            if human.disposition != "accept":
                graph = _load_bound(
                    state.embedded_canonical_graph,
                    EmbeddedArticulationCanonicalGraph,
                )
                assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
                state = _transition(
                    state,
                    "conditional",
                    f"Human {human.disposition} blocks embedded articulation authoring.",
                    rejected_candidate_ids=(
                        graph.candidate_ids if human.disposition == "reject" else ()
                    ),
                    unresolved_candidate_ids=(
                        graph.candidate_ids if human.disposition == "revise" else ()
                    ),
                    review_required_candidate_ids=(),
                )
                state = _persist_state(root, state)
                return write_articulation_workflow_summary(state, output_dir=root)
        else:
            human = _load_bound(
                state.embedded_human_decision,
                EmbeddedHumanDecision,
            )
            assert isinstance(human, EmbeddedHumanDecision)
            if human.disposition != "accept":
                raise EmbeddedArticulationError(
                    "Non-accepted human decision cannot resume"
                )
    elif state.embedded_human_decision is not None:
        raise EmbeddedArticulationError(
            "Outer-only Articulation review cannot invent human authority"
        )
    if client is None:
        raise EmbeddedArticulationError(
            "Outer-accepted Articulation graph requires a deterministic authorer"
        )
    state, authoring = _execute_graph(
        root,
        state,
        runtime,
        human,
        client=client,
        cancel_checker=cancel_checker,
    )
    return write_articulation_workflow_summary(
        state, output_dir=root, authoring=authoring
    )


def bind_embedded_articulation_output_evidence(
    output_dir: str | Path,
    canonical_visual_envelope_path: str | Path,
) -> ArticulationRunState:
    """Bind #1153 canonical OVRTX evidence before exact outer post review."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if state.phase != "awaiting_post_review":
        raise EmbeddedArticulationError(
            "Output evidence binding requires awaiting_post_review"
        )
    store = EmbeddedDecisionArtifactStore(root)
    domain_evidence = _load_store_artifact(
        store,
        state.embedded_evidence,
        EmbeddedDomainEvidence,
    )
    assert isinstance(domain_evidence, EmbeddedDomainEvidence)
    if not _articulation_capabilities_from_evidence(
        domain_evidence
    ).canonical_output_evidence_required:
        raise EmbeddedArticulationError(
            "Canonical output evidence was not selected for this compatibility run"
        )
    authoring = _load_bound(state.authoring_result, ArticulationAuthoringResult)
    readback = _load_bound(state.embedded_readback, EmbeddedArticulationReadback)
    assert isinstance(authoring, ArticulationAuthoringResult)
    assert isinstance(readback, EmbeddedArticulationReadback)
    authored_output = _execution_binding(authoring.output_asset_path)
    if (
        authored_output.sha256 != authoring.output_asset_sha256
        or authored_output.sha256 != readback.output_asset_sha256
        or authored_output.path
        != str(Path(authoring.output_asset_path).expanduser().resolve())
    ):
        raise EmbeddedArticulationError(
            "Canonical visual evidence requires exact authored readback bytes"
        )
    evidence = build_embedded_articulation_output_evidence(
        canonical_visual_envelope_path,
        expected_source=state_to_source_binding(state),
        expected_output=authored_output,
    )
    from .workflow import _write_once_json

    binding = _write_once_json(
        root / "embedded_articulation_output_evidence.json",
        evidence,
        label="embedded Articulation output evidence",
    )
    if state.embedded_output_evidence is not None:
        if state.embedded_output_evidence != binding:
            raise EmbeddedArticulationError(
                "Embedded Articulation output evidence conflicts with checkpoint"
            )
        return state
    state = state.model_copy(update={"embedded_output_evidence": binding})
    return _persist_state(root, state)


def apply_embedded_articulation_post_review(
    output_dir: str | Path,
    patch: EmbeddedArticulationPostReviewPatch,
    *,
    runtime: EmbeddedArticulationRuntime,
) -> ArticulationRunState:
    """Persist post-readback outer review, then and only then publish a receipt."""

    root = Path(output_dir).expanduser().resolve()
    state = _state(root)
    if state.phase != "awaiting_post_review":
        raise EmbeddedArticulationError("Post review requires awaiting_post_review")
    store = EmbeddedDecisionArtifactStore(root)
    evidence = _load_store_artifact(
        store, state.embedded_evidence, EmbeddedDomainEvidence
    )
    proposals = _load_optional_proposals(store, state)
    decision = _load_store_artifact(
        store, state.embedded_coordinator_decision, EmbeddedCoordinatorDecision
    )
    human = (
        _load_store_artifact(
            store,
            state.embedded_human_decision,
            EmbeddedHumanDecision,
        )
        if state.embedded_human_decision is not None
        else None
    )
    authorization = _load_store_artifact(
        store, state.embedded_execution_authorization, BoundedExecutionAuthorization
    )
    result = _load_store_artifact(
        store, state.embedded_execution_result, EmbeddedBoundedExecutionResult
    )
    graph = _load_bound(
        state.embedded_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    authoring = _load_bound(state.authoring_result, ArticulationAuthoringResult)
    validation = _load_bound(state.validation_result, ArticulationValidationResult)
    readback = _load_bound(state.embedded_readback, EmbeddedArticulationReadback)
    assert isinstance(evidence, EmbeddedDomainEvidence)
    assert isinstance(decision, EmbeddedCoordinatorDecision)
    assert human is None or isinstance(human, EmbeddedHumanDecision)
    assert isinstance(authorization, BoundedExecutionAuthorization)
    assert isinstance(result, EmbeddedBoundedExecutionResult)
    assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
    assert isinstance(authoring, ArticulationAuthoringResult)
    assert isinstance(validation, ArticulationValidationResult)
    assert isinstance(readback, EmbeddedArticulationReadback)
    persisted_capabilities = _articulation_capabilities_from_evidence(evidence)
    if runtime.capabilities != persisted_capabilities:
        raise EmbeddedArticulationError(
            "Post review runtime capabilities differ from persisted evidence"
        )
    output_evidence_required = persisted_capabilities.canonical_output_evidence_required
    output_evidence: EmbeddedArticulationOutputEvidence | None = None
    if output_evidence_required:
        output_evidence = _load_bound(
            state.embedded_output_evidence,
            EmbeddedArticulationOutputEvidence,
        )
        assert isinstance(output_evidence, EmbeddedArticulationOutputEvidence)
        validate_embedded_articulation_output_evidence(output_evidence)
    _validate_outer_review_chain(
        state,
        runtime=runtime,
        evidence=evidence,
        proposals=proposals,
        decision=decision,
        graph=graph,
    )
    _validate_graph_revision_chain(state, runtime=runtime)
    validation_binding = state.validation_result
    assert validation_binding is not None
    if (
        validation.status != "pass"
        or validation.expected_candidate_ids != graph.candidate_ids
        or validation.validated_candidate_ids != graph.candidate_ids
        or validation.output_asset_path != authoring.output_asset_path
        or validation.expected_output_asset_sha256 != authoring.output_asset_sha256
        or readback.canonical_graph_digest != canonical_json_digest(graph)
        or readback.accepted_decision_digest
        != _required_accepted_decision_digest(decision)
        or readback.validation_sha256 != validation_binding.sha256
        or readback.output_asset_sha256 != authoring.output_asset_sha256
        or readback.joint_ids != graph.candidate_ids
        or readback.topology != _readback_topology(graph)
        or readback.groups != _readback_groups(graph)
        or readback.memberships != _readback_memberships(graph)
    ):
        raise EmbeddedArticulationError(
            "Post review requires exact persisted saved-stage readback"
        )
    if patch.execution_result_digest != artifact_reference(result).sha256:
        raise EmbeddedArticulationError("Post review execution result digest is stale")
    if patch.accepted_decision_digest != decision.accepted_decision_digest:
        raise EmbeddedArticulationError("Post review accepted decision digest is stale")
    if output_evidence_required:
        assert output_evidence is not None
        if patch.schema_version != EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION:
            raise EmbeddedArticulationError(
                "Current embedded Articulation requires output-bound post review v2"
            )
        output_binding = state.embedded_output_evidence
        assert output_binding is not None
        if (
            patch.output_evidence_digest != output_binding.sha256
            or patch.inspected_render_image_sha256s
            != tuple(item.sha256 for item in output_evidence.images)
            or output_evidence.source != state_to_source_binding(state)
            or output_evidence.post_mutation_output
            != _execution_binding(authoring.output_asset_path)
        ):
            raise EmbeddedArticulationError(
                "Post review binds stale output or OVRTX image evidence"
            )
    elif (
        patch.schema_version != EMBEDDED_ARTICULATION_LEGACY_POST_REVIEW_SCHEMA_VERSION
    ):
        raise EmbeddedArticulationError(
            "Legacy embedded Articulation cannot invent output-evidence authority"
        )
    from .workflow import _write_once_json

    patch_binding = _write_once_json(
        root / "embedded_articulation_post_review_patch.json",
        patch,
        label="embedded Articulation post-review patch",
    )
    review = EmbeddedCoordinatorReview(
        artifact_id="articulation-post-readback-review",
        identity=runtime.identity,
        producer=runtime.outer_coordinator,
        parent_artifact=artifact_reference(result),
        created_at=_now(),
        execution_result=artifact_reference(result),
        semantic_decision_owner=runtime.outer_coordinator,
        accepted_decision_digest=decision.accepted_decision_digest,
        disposition=patch.disposition,
        outputs=result.outputs,
        findings=patch.findings,
    )
    validate_coordinator_review(review, result)
    review_commit = store.append_review(review)
    updates: dict[str, Any] = {
        "embedded_coordinator_review": _store_binding(
            store, review_commit.relative_path
        )
    }
    if patch.disposition == "accept":
        receipt = build_decision_receipt(
            artifact_id="articulation-embedded-decision-receipt",
            decision=decision,
            evidence=(evidence,),
            proposals=proposals,
            authorization=authorization,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
        )
        receipt_commit = store.append_receipt(receipt)
        decision_receipt_binding = _store_binding(store, receipt_commit.relative_path)
        updates["embedded_decision_receipt"] = decision_receipt_binding
        if output_evidence_required:
            assert output_evidence is not None
            output_evidence_binding = state.embedded_output_evidence
            assert output_evidence_binding is not None
            coordinator_review_binding = updates["embedded_coordinator_review"]
            terminal = EmbeddedArticulationTerminalReceipt(
                accepted_decision_digest=_required_accepted_decision_digest(decision),
                canonical_graph_digest=canonical_json_digest(graph),
                post_review_patch=patch_binding,
                output_evidence=output_evidence_binding,
                shared_coordinator_review=coordinator_review_binding,
                shared_decision_receipt=decision_receipt_binding,
                output_asset=output_evidence.post_mutation_output,
                dependencies=output_evidence.dependencies,
                dependency_closure_sha256=(output_evidence.dependency_closure_sha256),
                render_report=output_evidence.render_report,
                images=output_evidence.images,
                renderer_backend_alias=output_evidence.backend_alias,
            )
            terminal_binding = _write_once_json(
                root / "embedded_articulation_terminal_receipt.json",
                terminal,
                label="embedded Articulation terminal receipt",
            )
            updates["embedded_terminal_receipt"] = terminal_binding
        updates["review_required_candidate_ids"] = ()
        state = _transition(
            state.model_copy(update=updates),
            "completed",
            "Outer post-readback review accepted the exact embedded articulation graph.",
        )
    else:
        graph = _load_bound(
            state.embedded_canonical_graph, EmbeddedArticulationCanonicalGraph
        )
        assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
        state = _transition(
            state.model_copy(update=updates),
            "conditional",
            f"Outer post-readback review {patch.disposition} blocks stage completion.",
            unresolved_candidate_ids=graph.candidate_ids,
        )
    return _persist_state(root, state)


def validate_completed_embedded_articulation_checkpoint(
    state: ArticulationRunState,
    *,
    authoring: ArticulationAuthoringResult | None,
) -> None:
    """Revalidate the exact shared chain used by an embedded completion index."""

    required = (
        state.embedded_evidence,
        state.embedded_canonical_graph,
        state.embedded_coordinator_decision,
        state.embedded_execution_authorization,
        state.embedded_readback,
        state.embedded_execution_result,
        state.embedded_output_evidence,
        state.embedded_coordinator_review,
        state.embedded_decision_receipt,
        state.embedded_terminal_receipt,
    )
    compatibility_required = required[:6] + required[7:9]
    if any(item is None for item in compatibility_required):
        raise ValueError("Completed embedded articulation checkpoint is incomplete")
    if (
        state.schema_version
        in {
            "content-agent-workflows.articulation-run-state.v3",
            ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
        }
        and state.embedded_outer_review is None
    ):
        raise ValueError(
            "Completed embedded articulation checkpoint lacks outer review"
        )
    root = Path(state.request.path).expanduser().resolve().parent
    store = EmbeddedDecisionArtifactStore(root)
    result = _load_store_artifact(
        store, state.embedded_execution_result, EmbeddedBoundedExecutionResult
    )
    evidence = _load_store_artifact(
        store, state.embedded_evidence, EmbeddedDomainEvidence
    )
    proposals = _load_optional_proposals(store, state)
    decision = _load_store_artifact(
        store, state.embedded_coordinator_decision, EmbeddedCoordinatorDecision
    )
    human = (
        _load_store_artifact(
            store,
            state.embedded_human_decision,
            EmbeddedHumanDecision,
        )
        if state.embedded_human_decision is not None
        else None
    )
    review = _load_store_artifact(
        store, state.embedded_coordinator_review, EmbeddedCoordinatorReview
    )
    receipt = _load_store_artifact(
        store, state.embedded_decision_receipt, EmbeddedDecisionReceipt
    )
    authorization = _load_store_artifact(
        store, state.embedded_execution_authorization, BoundedExecutionAuthorization
    )
    graph = _load_bound(
        state.embedded_canonical_graph, EmbeddedArticulationCanonicalGraph
    )
    validation = _load_bound(state.validation_result, ArticulationValidationResult)
    readback = _load_bound(state.embedded_readback, EmbeddedArticulationReadback)
    assert isinstance(result, EmbeddedBoundedExecutionResult)
    assert isinstance(evidence, EmbeddedDomainEvidence)
    assert isinstance(decision, EmbeddedCoordinatorDecision)
    assert human is None or isinstance(human, EmbeddedHumanDecision)
    assert isinstance(review, EmbeddedCoordinatorReview)
    assert isinstance(receipt, EmbeddedDecisionReceipt)
    assert isinstance(authorization, BoundedExecutionAuthorization)
    assert isinstance(graph, EmbeddedArticulationCanonicalGraph)
    assert isinstance(validation, ArticulationValidationResult)
    assert isinstance(readback, EmbeddedArticulationReadback)
    runtime = EmbeddedArticulationRuntime(
        identity=decision.identity,
        evidence_provider=evidence.producer,
        proposal_provider=(proposals[-1].producer if proposals else None),
        outer_coordinator=decision.producer,
        executor=result.producer,
        capabilities=_articulation_capabilities_from_evidence(evidence),
    )
    _validate_graph_revision_chain(state, runtime=runtime)
    if runtime.capabilities.canonical_output_evidence_required:
        if state.embedded_output_evidence is None or (
            state.embedded_terminal_receipt is None
        ):
            raise ValueError(
                "Completed embedded articulation lacks canonical output evidence"
            )
        output_evidence = _load_bound(
            state.embedded_output_evidence,
            EmbeddedArticulationOutputEvidence,
        )
        terminal = _load_bound(
            state.embedded_terminal_receipt,
            EmbeddedArticulationTerminalReceipt,
        )
        assert isinstance(output_evidence, EmbeddedArticulationOutputEvidence)
        assert isinstance(terminal, EmbeddedArticulationTerminalReceipt)
        validate_embedded_articulation_output_evidence(output_evidence)
        post_review_patch = _load_bound(
            terminal.post_review_patch,
            EmbeddedArticulationPostReviewPatch,
        )
        assert isinstance(post_review_patch, EmbeddedArticulationPostReviewPatch)
        if (
            post_review_patch.schema_version
            != EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION
            or post_review_patch.disposition != "accept"
            or post_review_patch.output_evidence_digest
            != state.embedded_output_evidence.sha256
            or post_review_patch.inspected_render_image_sha256s
            != tuple(item.sha256 for item in output_evidence.images)
            or terminal.output_evidence != state.embedded_output_evidence
            or terminal.shared_coordinator_review != state.embedded_coordinator_review
            or terminal.shared_decision_receipt != state.embedded_decision_receipt
            or terminal.accepted_decision_digest
            != _required_accepted_decision_digest(decision)
            or terminal.canonical_graph_digest != canonical_json_digest(graph)
            or terminal.output_asset != output_evidence.post_mutation_output
            or terminal.dependencies != output_evidence.dependencies
            or terminal.dependency_closure_sha256
            != output_evidence.dependency_closure_sha256
            or terminal.render_report != output_evidence.render_report
            or terminal.images != output_evidence.images
            or terminal.renderer_backend_alias != output_evidence.backend_alias
        ):
            raise ValueError("Completed embedded articulation output evidence is stale")
    _validate_outer_review_chain(
        state,
        runtime=runtime,
        evidence=evidence,
        proposals=proposals,
        decision=decision,
        graph=graph,
    )
    validate_bounded_execution_result(result, authorization)
    validate_coordinator_review(review, result)
    validate_decision_receipt(
        receipt,
        decision=decision,
        evidence=(evidence,),
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
    )
    if receipt.receipt_status != "completed" or receipt.review_disposition != "accept":
        raise ValueError("Embedded articulation receipt is not completed and accepted")
    authoring_request = _load_bound(
        state.authoring_request, ArticulationAuthoringRequest
    )
    bound_authoring = _load_bound(state.authoring_result, ArticulationAuthoringResult)
    if not isinstance(authoring_request, ArticulationAuthoringRequest) or (
        authoring_request.predictions_path is not None
        or authoring_request.predictions_sha256 is not None
    ):
        raise ValueError(
            "Embedded articulation authoring contains prediction authority"
        )
    if (
        authoring is None
        or authoring != bound_authoring
        or authoring.authored_candidate_ids != state.accepted_candidate_ids
    ):
        raise ValueError(
            "Embedded articulation completion lacks exact authoring evidence"
        )
    validation_binding = state.validation_result
    assert validation_binding is not None
    if (
        validation.status != "pass"
        or validation.expected_candidate_ids != graph.candidate_ids
        or validation.validated_candidate_ids != graph.candidate_ids
        or validation.output_asset_path != authoring.output_asset_path
        or validation.expected_output_asset_sha256 != authoring.output_asset_sha256
        or readback.joint_ids != state.accepted_candidate_ids
        or readback.topology != _readback_topology(graph)
        or readback.groups != _readback_groups(graph)
        or readback.memberships != _readback_memberships(graph)
        or readback.canonical_graph_digest != canonical_json_digest(graph)
        or readback.accepted_decision_digest
        != _required_accepted_decision_digest(decision)
        or readback.validation_sha256 != validation_binding.sha256
        or authoring is None
        or readback.output_asset_sha256 != authoring.output_asset_sha256
    ):
        raise ValueError("Embedded articulation completion readback is stale")


__all__ = [
    "EMBEDDED_ARTICULATION_GRAPH_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_GRAPH_REVISION_PATCH_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_GRAPH_REVISION_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_IMPLEMENTATION_MANIFEST",
    "EMBEDDED_ARTICULATION_LEGACY_PATCH_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_LEGACY_POST_REVIEW_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_OPTIONAL_PROPOSAL_PATCH_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_OUTER_REVIEW_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_PATCH_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_PREPARATION_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_READBACK_SCHEMA_VERSION",
    "EmbeddedArticulationCanonicalGraph",
    "EmbeddedArticulationCapabilityLimits",
    "EmbeddedArticulationDecisionPatch",
    "EmbeddedArticulationError",
    "EmbeddedArticulationGroup",
    "EmbeddedArticulationGraphChange",
    "EmbeddedArticulationGraphRevision",
    "EmbeddedArticulationGraphRevisionPatch",
    "EmbeddedArticulationHumanAcceptance",
    "EmbeddedArticulationHumanReviewPolicy",
    "EmbeddedArticulationJoint",
    "EmbeddedArticulationMembership",
    "EmbeddedArticulationRigidLinkOperation",
    "EmbeddedArticulationOuterReview",
    "EmbeddedArticulationOutputEvidence",
    "EmbeddedArticulationPostReviewPatch",
    "EmbeddedArticulationPreparation",
    "EmbeddedArticulationProposalCapability",
    "EmbeddedArticulationProviderProposal",
    "EmbeddedArticulationReadback",
    "EmbeddedArticulationRuntime",
    "EmbeddedArticulationTerminalReceipt",
    "apply_embedded_articulation_decision_patch",
    "apply_embedded_articulation_graph_revision",
    "apply_embedded_articulation_post_review",
    "bind_embedded_articulation_output_evidence",
    "canonical_articulation_graph_changes",
    "load_articulation_revision_human_decisions",
    "prepare_embedded_articulation_workflow",
    "run_embedded_articulation_workflow",
    "select_embedded_articulation_human_review_policy",
    "state_to_source_binding",
    "validate_completed_embedded_articulation_checkpoint",
    "validate_embedded_articulation_provider_proposal",
]
