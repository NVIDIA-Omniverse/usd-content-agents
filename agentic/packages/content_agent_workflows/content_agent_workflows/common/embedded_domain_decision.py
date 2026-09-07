# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral decision ownership for embedded domain execution.

This module is intentionally additive.  Standalone and compatibility-pipeline
requests continue to use :mod:`domain_execution`; only an explicitly embedded
context can construct this decision chain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    model_validator,
)

from .domain_execution import DomainExecutionContext, ExecutionArtifactBinding

EMBEDDED_DECISION_EVIDENCE_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-domain-evidence.v1"
)
EMBEDDED_DECISION_PROPOSAL_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-domain-proposal.v1"
)
EMBEDDED_COORDINATOR_DECISION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-coordinator-decision.v1"
)
EMBEDDED_HUMAN_DECISION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-human-decision.v1"
)
EMBEDDED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-execution-authorization.v1"
)
EMBEDDED_EXECUTION_RESULT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-execution-result.v1"
)
EMBEDDED_COORDINATOR_REVIEW_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-coordinator-review.v1"
)
EMBEDDED_DECISION_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-domain-decision-receipt.v1"
)

Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactKind = Literal[
    "outer_request",
    "coordinator_plan",
    "evidence",
    "proposal",
    "coordinator_decision",
    "human_decision",
    "execution_authorization",
    "execution_result",
    "coordinator_review",
    "decision_receipt",
]
ProducerRole = Literal[
    "outer_coordinator",
    "evidence_provider",
    "proposal_provider",
    "human_reviewer",
    "executor",
]
EvidenceType = Literal[
    "inspection",
    "render",
    "validation",
    "artifact",
    "measurement",
    "critique",
    "capability",
]
EvidenceStatus = Literal["available", "unavailable", "error", "unsupported"]
CoordinatorDecisionDisposition = Literal["accept", "reject", "revise"]
HumanDecisionDisposition = Literal["accept", "reject", "revise"]
ExecutionStatus = Literal["succeeded", "failed", "interrupted", "cancelled"]
ExecutionEffect = Literal["mutation", "non_mutating"]
MutationState = Literal["not_started", "applied", "verified", "not_applicable"]
CoordinatorReviewDisposition = Literal[
    "accept",
    "reject",
    "revise",
    "retry",
    "stop",
    "cancelled",
]
ReceiptStatus = Literal[
    "completed",
    "rejected",
    "revision_required",
    "retry_required",
    "interrupted",
    "cancelled",
    "stopped",
]


def _freeze_nested_container(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_nested_container(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_nested_container(item) for item in value)
    return value


def _thaw_nested_container(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_nested_container(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_nested_container(item) for item in value]
    return value


def _freeze_mapping[ValueT](value: Mapping[str, ValueT]) -> Mapping[str, ValueT]:
    return cast(
        Mapping[str, ValueT],
        _freeze_nested_container(value),
    )


def _serialize_frozen_mapping(value: Mapping[str, object]) -> dict[str, object]:
    thawed = _thaw_nested_container(value)
    if not isinstance(thawed, dict):  # pragma: no cover - mapping input invariant
        raise TypeError("frozen mapping serializer requires a mapping")
    return thawed


def _freeze_sequence[ValueT](value: Sequence[ValueT]) -> Sequence[ValueT]:
    return tuple(value)


type FrozenMapping[ValueT] = Annotated[
    Mapping[str, ValueT],
    AfterValidator(_freeze_mapping),
    PlainSerializer(_serialize_frozen_mapping, return_type=dict[str, object]),
]
type FrozenSequence[ValueT] = Annotated[
    Sequence[ValueT],
    AfterValidator(_freeze_sequence),
]


class EmbeddedDecisionContractError(RuntimeError):
    """Raised when an embedded decision chain is incomplete or stale."""


def canonical_json_digest(value: BaseModel | dict[str, JsonValue]) -> str:
    """Return a stable SHA-256 for one JSON-compatible contract value."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProducerIdentity(BaseModel):
    """Stable identity and implementation of one contract participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    producer_id: str = Field(min_length=1)
    role: ProducerRole
    implementation: str = Field(min_length=1)
    implementation_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_implementation_identity(self) -> Self:
        if self.role != "human_reviewer" and self.implementation_digest is None:
            raise ValueError(f"{self.role} requires an implementation_digest")
        return self


class NamedDecisionDigests(BaseModel):
    """Exact mutable inputs whose drift invalidates an accepted decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration: FrozenMapping[Sha256Digest] = Field(min_length=1)
    prompt: FrozenMapping[Sha256Digest] = Field(min_length=1)
    references: FrozenMapping[Sha256Digest] = Field(default_factory=dict)
    capabilities: FrozenMapping[Sha256Digest] = Field(min_length=1)
    implementations: FrozenMapping[Sha256Digest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        for group_name in (
            "configuration",
            "prompt",
            "references",
            "capabilities",
            "implementations",
        ):
            group = getattr(self, group_name)
            if any(not name.strip() for name in group):
                raise ValueError(f"{group_name} digest names must not be empty")
        return self


class ContractArtifactReference(BaseModel):
    """Content identity for a persisted parent or linked contract artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: ArtifactKind
    artifact_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    sha256: Sha256Digest


class EmbeddedDecisionIdentity(BaseModel):
    """Frozen request, run, source, and dependency identity for one attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_context: DomainExecutionContext
    source: ExecutionArtifactBinding
    coordinator_plan: ContractArtifactReference
    digests: NamedDecisionDigests

    @model_validator(mode="after")
    def validate_embedded_identity(self) -> Self:
        context = self.execution_context
        if context.mode != "embedded" or context.embedded_stage is None:
            raise ValueError(
                "embedded decision artifacts require an embedded execution context"
            )
        if context.reasoning_loop_owner != "asset_coordinator":
            raise ValueError(
                "embedded decision artifacts require asset_coordinator ownership"
            )
        if self.source != context.embedded_stage.input_asset:
            raise ValueError(
                "embedded decision source differs from the active stage input"
            )
        if (
            self.coordinator_plan.artifact_kind != "coordinator_plan"
            or self.coordinator_plan.sha256
            != context.embedded_stage.coordinator_plan.sha256
        ):
            raise ValueError(
                "embedded decision coordinator plan differs from the active stage"
            )
        return self


def outer_stage_attempt_seal_key(identity: EmbeddedDecisionIdentity) -> str:
    """Return the stable outer-owned key that seals one embedded stage attempt."""

    stage = identity.execution_context.embedded_stage
    if stage is None:  # pragma: no cover - EmbeddedDecisionIdentity invariant
        raise ValueError("embedded decision identity lacks an outer stage binding")
    return canonical_json_digest(
        {
            "schema_version": (
                "content-agent-workflows.outer-stage-attempt-seal-key.v1"
            ),
            "outer_run_id": stage.outer_run_id,
            "domain": identity.execution_context.domain,
            "stage": stage.stage,
            "stage_attempt": stage.stage_attempt,
        }
    )


class ContractArtifact(BaseModel):
    """Fields shared by every append-only embedded decision artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: ArtifactKind
    artifact_id: str = Field(min_length=1)
    identity: EmbeddedDecisionIdentity
    producer: ProducerIdentity
    parent_artifact: ContractArtifactReference
    created_at: datetime

    @model_validator(mode="after")
    def validate_common_artifact_identity(self) -> Self:
        if self.parent_artifact.artifact_id == self.artifact_id:
            raise ValueError("contract artifact cannot name itself as its parent")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("contract artifact timestamp must be timezone-aware")
        if (
            self.producer.role != "human_reviewer"
            and self.producer.implementation_digest
            not in set(self.identity.digests.implementations.values())
        ):
            raise ValueError(
                "contract artifact producer implementation is not identity-bound"
            )
        return self


def artifact_reference(artifact: ContractArtifact) -> ContractArtifactReference:
    """Return the exact content identity of one typed contract artifact."""

    schema_version = getattr(artifact, "schema_version", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise TypeError("Contract artifact has no schema_version")
    return ContractArtifactReference(
        artifact_kind=artifact.artifact_kind,
        artifact_id=artifact.artifact_id,
        schema_version=schema_version,
        sha256=canonical_json_digest(artifact),
    )


def _validate_execution_effect(
    identity: EmbeddedDecisionIdentity,
    execution_effect: ExecutionEffect,
) -> None:
    if (
        identity.execution_context.domain == "validation"
        and execution_effect != "non_mutating"
    ):
        raise ValueError("validation execution must be non_mutating")


class ProviderNeutralEvidenceRecord(BaseModel):
    """Inspection, render, validation, or artifact evidence without a backend ABI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    status: EvidenceStatus
    required: bool = True
    summary: str = Field(min_length=1)
    artifacts: FrozenSequence[ExecutionArtifactBinding] = ()
    facts: FrozenMapping[JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_available_payload(self) -> Self:
        if self.status == "available" and not self.artifacts and not self.facts:
            raise ValueError("available evidence requires artifacts or factual data")
        if self.status != "available" and self.facts:
            raise ValueError(
                "unavailable, error, or unsupported evidence cannot assert facts"
            )
        return self


class EmbeddedDomainEvidence(ContractArtifact):
    """Digest-bound factual evidence produced by a replaceable provider."""

    artifact_kind: Literal["evidence"] = "evidence"
    schema_version: Literal["content-agent-workflows.embedded-domain-evidence.v1"] = (
        EMBEDDED_DECISION_EVIDENCE_SCHEMA_VERSION
    )
    records: FrozenSequence[ProviderNeutralEvidenceRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence_provider(self) -> Self:
        if self.producer.role != "evidence_provider":
            raise ValueError("evidence artifact producer must be an evidence_provider")
        stage = self.identity.execution_context.embedded_stage
        if stage is None:  # pragma: no cover - EmbeddedDecisionIdentity invariant
            raise ValueError("evidence identity lacks an embedded stage")
        if self.parent_artifact != self.identity.coordinator_plan:
            raise ValueError(
                "evidence parent must be the exact active coordinator plan"
            )
        return self


class DomainProposalPayload(BaseModel):
    """Domain-versioned proposal values that have not been accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    values: FrozenMapping[JsonValue] = Field(min_length=1)


class EmbeddedDomainProposal(ContractArtifact):
    """Optional provider proposal that is never itself mutation authority."""

    artifact_kind: Literal["proposal"] = "proposal"
    schema_version: Literal["content-agent-workflows.embedded-domain-proposal.v1"] = (
        EMBEDDED_DECISION_PROPOSAL_SCHEMA_VERSION
    )
    evidence_artifacts: FrozenSequence[ContractArtifactReference] = Field(min_length=1)
    proposal: DomainProposalPayload
    proposal_digest: Sha256Digest

    @model_validator(mode="after")
    def validate_proposal(self) -> Self:
        if self.producer.role != "proposal_provider":
            raise ValueError("proposal artifact producer must be a proposal_provider")
        if any(ref.artifact_kind != "evidence" for ref in self.evidence_artifacts):
            raise ValueError(
                "proposal evidence references must name evidence artifacts"
            )
        if self.parent_artifact not in self.evidence_artifacts:
            raise ValueError("proposal parent must be one of its evidence artifacts")
        if self.proposal_digest != canonical_json_digest(self.proposal):
            raise ValueError("proposal_digest does not match the proposal payload")
        return self


class AcceptedSemanticDecision(BaseModel):
    """Domain-versioned semantic values authored by the outer coordinator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    values: FrozenMapping[JsonValue] = Field(min_length=1)


def accepted_semantic_decision_digest(
    identity: EmbeddedDecisionIdentity,
    decision: AcceptedSemanticDecision,
) -> str:
    """Digest semantic values together with every frozen decision input."""

    return canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.accepted-semantic-decision.v1",
            "identity": identity.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
        }
    )


class EmbeddedCoordinatorDecision(ContractArtifact):
    """Canonical accept, reject, or revise decision from the outer coordinator."""

    artifact_kind: Literal["coordinator_decision"] = "coordinator_decision"
    schema_version: Literal[
        "content-agent-workflows.embedded-coordinator-decision.v1"
    ] = EMBEDDED_COORDINATOR_DECISION_SCHEMA_VERSION
    disposition: CoordinatorDecisionDisposition
    evidence_artifacts: FrozenSequence[ContractArtifactReference] = ()
    proposal_artifacts: FrozenSequence[ContractArtifactReference] = ()
    accepted_decision: AcceptedSemanticDecision | None = None
    accepted_decision_digest: Sha256Digest | None = None
    human_decision_required: bool = False
    rationale: str = Field(min_length=1)
    revision_requests: FrozenSequence[str] = ()

    @model_validator(mode="after")
    def validate_coordinator_decision(self) -> Self:
        if self.producer.role != "outer_coordinator":
            raise ValueError(
                "coordinator decision producer must be the outer_coordinator"
            )
        if any(ref.artifact_kind != "evidence" for ref in self.evidence_artifacts):
            raise ValueError(
                "decision evidence references must name evidence artifacts"
            )
        if any(ref.artifact_kind != "proposal" for ref in self.proposal_artifacts):
            raise ValueError(
                "decision proposal references must name proposal artifacts"
            )
        inputs = [*self.evidence_artifacts, *self.proposal_artifacts]
        if not inputs:
            raise ValueError("coordinator decision requires evidence or proposals")
        if not self.evidence_artifacts:
            raise ValueError("coordinator decision requires evidence artifacts")
        if self.parent_artifact not in inputs:
            raise ValueError("coordinator decision parent must be a reviewed input")
        if self.disposition == "accept":
            if self.accepted_decision is None or self.accepted_decision_digest is None:
                raise ValueError(
                    "accepted coordinator decision requires typed values and digest"
                )
            expected = accepted_semantic_decision_digest(
                self.identity,
                self.accepted_decision,
            )
            if self.accepted_decision_digest != expected:
                raise ValueError(
                    "accepted_decision_digest does not match identity and values"
                )
            if self.revision_requests:
                raise ValueError("accepted decision cannot retain revision requests")
        else:
            if self.accepted_decision is not None or self.accepted_decision_digest:
                raise ValueError(
                    "rejected or revision decision cannot publish accepted values"
                )
            if self.disposition == "revise" and not self.revision_requests:
                raise ValueError("revision decision requires revision_requests")
        return self


class EmbeddedHumanDecision(ContractArtifact):
    """Exact human disposition for a coordinator decision that requires it."""

    artifact_kind: Literal["human_decision"] = "human_decision"
    schema_version: Literal["content-agent-workflows.embedded-human-decision.v1"] = (
        EMBEDDED_HUMAN_DECISION_SCHEMA_VERSION
    )
    coordinator_decision: ContractArtifactReference
    reviewed_decision_digest: Sha256Digest
    disposition: HumanDecisionDisposition
    rationale: str = Field(min_length=1)
    revision_requests: FrozenSequence[str] = ()

    @model_validator(mode="after")
    def validate_human_decision(self) -> Self:
        if self.producer.role != "human_reviewer":
            raise ValueError("human decision producer must be a human_reviewer")
        if self.coordinator_decision.artifact_kind != "coordinator_decision":
            raise ValueError("human decision must bind a coordinator decision")
        if self.parent_artifact != self.coordinator_decision:
            raise ValueError("human decision parent must be its coordinator decision")
        if self.disposition == "revise" and not self.revision_requests:
            raise ValueError("human revision requires revision_requests")
        if self.disposition == "accept" and self.revision_requests:
            raise ValueError("accepted human decision cannot retain revision requests")
        return self


class BoundedExecutionAuthorization(ContractArtifact):
    """Persistable, idempotency-bound authority for one executor attempt."""

    artifact_kind: Literal["execution_authorization"] = "execution_authorization"
    schema_version: Literal[
        "content-agent-workflows.embedded-execution-authorization.v1"
    ] = EMBEDDED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION
    accepted_decision: ContractArtifactReference
    accepted_decision_digest: Sha256Digest
    human_decision: ContractArtifactReference | None = None
    executor: ProducerIdentity
    execution_effect: ExecutionEffect
    outer_stage_attempt_seal_key: Sha256Digest
    operation_id: Sha256Digest
    mutation_id: Sha256Digest | None = None
    attempt: int = Field(ge=1)
    resume_of: ContractArtifactReference | None = None
    resume_review: ContractArtifactReference | None = None

    @model_validator(mode="after")
    def validate_authorization_claim(self) -> Self:
        _validate_execution_effect(self.identity, self.execution_effect)
        if self.producer.role != "outer_coordinator":
            raise ValueError("execution authorization requires outer_coordinator")
        if self.accepted_decision.artifact_kind != "coordinator_decision":
            raise ValueError("authorization must bind a coordinator decision")
        if self.executor.role != "executor":
            raise ValueError("authorization executor must have executor role")
        if self.executor.implementation_digest not in set(
            self.identity.digests.implementations.values()
        ):
            raise ValueError(
                "authorization executor implementation is not identity-bound"
            )
        if self.attempt == 1 and (
            self.resume_of is not None or self.resume_review is not None
        ):
            raise ValueError("first authorization attempt cannot carry resume links")
        if self.attempt > 1 and (self.resume_of is None or self.resume_review is None):
            raise ValueError(
                "authorization attempts after the first require both resume links"
            )
        if self.resume_of is not None and self.resume_of.artifact_kind != (
            "execution_result"
        ):
            raise ValueError("authorization resume_of must name an execution result")
        if self.resume_review is not None and self.resume_review.artifact_kind != (
            "coordinator_review"
        ):
            raise ValueError(
                "authorization resume_review must name a coordinator review"
            )
        expected_parent = (
            self.resume_review or self.human_decision or self.accepted_decision
        )
        if self.parent_artifact != expected_parent:
            raise ValueError("authorization parent differs from its exact authority")
        expected_operation_id = _operation_id(
            self.identity,
            accepted_decision_digest=self.accepted_decision_digest,
            execution_effect=self.execution_effect,
        )
        if self.operation_id != expected_operation_id:
            raise ValueError(
                "authorization operation_id does not match its exact decision"
            )
        if self.outer_stage_attempt_seal_key != outer_stage_attempt_seal_key(
            self.identity
        ):
            raise ValueError(
                "authorization stage-attempt seal does not match outer ownership"
            )
        if self.execution_effect == "mutation":
            if self.mutation_id != expected_operation_id:
                raise ValueError(
                    "mutating authorization requires mutation_id equal to operation_id"
                )
        elif self.mutation_id is not None:
            raise ValueError("non-mutating authorization cannot invent a mutation_id")
        return self


def _unavailable_required_evidence(
    evidence: Sequence[ProviderNeutralEvidenceRecord],
) -> list[str]:
    return [
        record.evidence_id
        for record in evidence
        if record.required and record.status != "available"
    ]


class EmbeddedBoundedExecutionResult(ContractArtifact):
    """Result of one exact accepted decision and authorization claim."""

    artifact_kind: Literal["execution_result"] = "execution_result"
    schema_version: Literal["content-agent-workflows.embedded-execution-result.v1"] = (
        EMBEDDED_EXECUTION_RESULT_SCHEMA_VERSION
    )
    accepted_decision: ContractArtifactReference
    accepted_decision_digest: Sha256Digest
    execution_effect: ExecutionEffect
    operation_id: Sha256Digest
    mutation_id: Sha256Digest | None = None
    attempt: int = Field(ge=1)
    resume_of: ContractArtifactReference | None = None
    status: ExecutionStatus
    mutation_state: MutationState
    outputs: FrozenSequence[ExecutionArtifactBinding] = ()
    evidence: FrozenSequence[ProviderNeutralEvidenceRecord] = ()
    error: str | None = None

    @model_validator(mode="after")
    def validate_execution_result(self) -> Self:
        _validate_execution_effect(self.identity, self.execution_effect)
        if self.producer.role != "executor":
            raise ValueError("execution result producer must be an executor")
        if self.accepted_decision.artifact_kind != "coordinator_decision":
            raise ValueError("execution result must bind a coordinator decision")
        if self.resume_of is not None and self.resume_of.artifact_kind != (
            "execution_result"
        ):
            raise ValueError("resume_of must name an execution result")
        if self.execution_effect == "mutation":
            if self.mutation_id != self.operation_id:
                raise ValueError(
                    "mutating result requires mutation_id equal to operation_id"
                )
            if self.mutation_state == "not_applicable":
                raise ValueError("mutating result requires a mutation state")
            if self.mutation_state == "not_started" and (self.outputs or self.evidence):
                raise ValueError(
                    "mutating not_started result cannot retain outputs or evidence"
                )
        else:
            if self.mutation_id is not None:
                raise ValueError("non-mutating result cannot invent a mutation_id")
            if self.mutation_state in {"applied", "verified"}:
                raise ValueError(
                    "non-mutating result cannot claim an applied or verified mutation"
                )
        if self.status == "succeeded":
            if self.execution_effect == "mutation" and (
                self.mutation_state != "verified" or not self.outputs
            ):
                raise ValueError(
                    "successful mutation requires verified state and at least one "
                    "exact output binding"
                )
            if self.execution_effect != "mutation" and (
                self.mutation_state != "not_applicable"
                or not (self.outputs or self.evidence)
            ):
                raise ValueError(
                    "successful non-mutating execution requires not_applicable state "
                    "and retained outputs or evidence"
                )
            if self.error is not None:
                raise ValueError("successful execution cannot contain an error")
            unavailable = _unavailable_required_evidence(self.evidence)
            if unavailable:
                raise ValueError(
                    "successful execution has unavailable required evidence: "
                    + ", ".join(unavailable)
                )
        else:
            if not self.error:
                raise ValueError(f"{self.status} execution requires an error or reason")
            if self.mutation_state != "not_started" and not self.outputs:
                if not self.evidence:
                    raise ValueError(
                        "a started non-success result must retain outputs or evidence"
                    )
        return self


class EmbeddedCoordinatorReview(ContractArtifact):
    """Outer-coordinator disposition of exact bounded execution evidence."""

    artifact_kind: Literal["coordinator_review"] = "coordinator_review"
    schema_version: Literal[
        "content-agent-workflows.embedded-coordinator-review.v1"
    ] = EMBEDDED_COORDINATOR_REVIEW_SCHEMA_VERSION
    execution_result: ContractArtifactReference
    semantic_decision_owner: ProducerIdentity
    accepted_decision_digest: Sha256Digest
    disposition: CoordinatorReviewDisposition
    outputs: FrozenSequence[ExecutionArtifactBinding] = ()
    findings: FrozenSequence[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_coordinator_review(self) -> Self:
        if self.producer.role != "outer_coordinator":
            raise ValueError("coordinator review producer must be outer_coordinator")
        if self.producer != self.semantic_decision_owner:
            raise ValueError(
                "coordinator review producer must be the semantic decision owner"
            )
        if self.execution_result.artifact_kind != "execution_result":
            raise ValueError("coordinator review must bind an execution result")
        if self.parent_artifact != self.execution_result:
            raise ValueError("coordinator review parent must be its execution result")
        return self


def _canonical_receipt_status(
    result_status: ExecutionStatus,
    review_disposition: CoordinatorReviewDisposition,
) -> ReceiptStatus:
    statuses: dict[
        tuple[ExecutionStatus, CoordinatorReviewDisposition], ReceiptStatus
    ] = {
        ("succeeded", "accept"): "completed",
        ("succeeded", "reject"): "rejected",
        ("succeeded", "revise"): "revision_required",
        ("failed", "retry"): "retry_required",
        ("failed", "revise"): "revision_required",
        ("failed", "stop"): "stopped",
        ("interrupted", "retry"): "interrupted",
        ("interrupted", "revise"): "revision_required",
        ("interrupted", "stop"): "stopped",
        ("cancelled", "cancelled"): "cancelled",
        ("cancelled", "stop"): "stopped",
    }
    try:
        return statuses[(result_status, review_disposition)]
    except KeyError as exc:
        raise ValueError(
            "receipt execution status and review disposition are inconsistent"
        ) from exc


class EmbeddedDecisionReceipt(ContractArtifact):
    """Persistable provenance for accepted semantics, execution, and review."""

    artifact_kind: Literal["decision_receipt"] = "decision_receipt"
    schema_version: Literal[
        "content-agent-workflows.embedded-domain-decision-receipt.v1"
    ] = EMBEDDED_DECISION_RECEIPT_SCHEMA_VERSION
    receipt_status: ReceiptStatus
    semantic_decision_owner: ProducerIdentity
    evidence_providers: FrozenSequence[ProducerIdentity] = Field(min_length=1)
    proposal_providers: FrozenSequence[ProducerIdentity] = ()
    accepted_decision: ContractArtifactReference
    accepted_decision_digest: Sha256Digest
    human_decision: ContractArtifactReference | None = None
    executor: ProducerIdentity
    execution_result: ContractArtifactReference
    execution_authorization: ContractArtifactReference
    execution_effect: ExecutionEffect
    outer_stage_attempt_seal_key: Sha256Digest
    operation_id: Sha256Digest
    mutation_id: Sha256Digest | None = None
    attempt: int = Field(ge=1)
    prior_execution_authorizations: FrozenSequence[ContractArtifactReference]
    prior_execution_results: FrozenSequence[ContractArtifactReference]
    prior_coordinator_reviews: FrozenSequence[ContractArtifactReference]
    outputs: FrozenSequence[ExecutionArtifactBinding] = ()
    evidence: FrozenSequence[ProviderNeutralEvidenceRecord] = ()
    coordinator_review: ContractArtifactReference
    execution_status: ExecutionStatus
    review_disposition: CoordinatorReviewDisposition

    @model_validator(mode="after")
    def validate_receipt_provenance(self) -> Self:
        _validate_execution_effect(self.identity, self.execution_effect)
        if self.producer != self.semantic_decision_owner:
            raise ValueError("receipt producer must be the semantic_decision_owner")
        if self.semantic_decision_owner.role != "outer_coordinator":
            raise ValueError("semantic_decision_owner must be the outer_coordinator")
        if any(item.role != "evidence_provider" for item in self.evidence_providers):
            raise ValueError("evidence_providers contains a non-evidence provider")
        if any(item.role != "proposal_provider" for item in self.proposal_providers):
            raise ValueError("proposal_providers contains a non-proposal provider")
        if self.executor.role != "executor":
            raise ValueError("receipt executor must have executor role")
        if self.accepted_decision.artifact_kind != "coordinator_decision":
            raise ValueError("receipt accepted_decision must name coordinator decision")
        if self.execution_result.artifact_kind != "execution_result":
            raise ValueError("receipt execution_result must name execution result")
        if self.execution_authorization.artifact_kind != "execution_authorization":
            raise ValueError(
                "receipt execution_authorization must name an authorization"
            )
        if self.coordinator_review.artifact_kind != "coordinator_review":
            raise ValueError("receipt coordinator_review must name coordinator review")
        if self.parent_artifact != self.coordinator_review:
            raise ValueError("receipt parent must be the coordinator review")
        if self.outer_stage_attempt_seal_key != outer_stage_attempt_seal_key(
            self.identity
        ):
            raise ValueError(
                "receipt stage-attempt seal does not match outer ownership"
            )
        if self.execution_effect == "mutation":
            if self.mutation_id != self.operation_id:
                raise ValueError(
                    "mutating receipt requires mutation_id equal to operation_id"
                )
        elif self.mutation_id is not None:
            raise ValueError("non-mutating receipt cannot invent a mutation_id")
        expected_prior_count = self.attempt - 1
        prior_groups = (
            self.prior_execution_authorizations,
            self.prior_execution_results,
            self.prior_coordinator_reviews,
        )
        if any(len(group) != expected_prior_count for group in prior_groups):
            raise ValueError("receipt must retain the complete prior execution lineage")
        if any(
            ref.artifact_kind != "execution_authorization"
            for ref in self.prior_execution_authorizations
        ):
            raise ValueError(
                "prior_execution_authorizations contains a non-authorization"
            )
        if any(
            ref.artifact_kind != "execution_result"
            for ref in self.prior_execution_results
        ):
            raise ValueError("prior_execution_results contains a non-result")
        if any(
            ref.artifact_kind != "coordinator_review"
            for ref in self.prior_coordinator_reviews
        ):
            raise ValueError("prior_coordinator_reviews contains a non-review")
        if self.execution_status == "succeeded":
            if self.execution_effect == "mutation" and not self.outputs:
                raise ValueError(
                    "successful mutation receipt requires at least one exact output "
                    "binding"
                )
            if self.execution_effect == "non_mutating" and not (
                self.outputs
                or any(item.status == "available" for item in self.evidence)
            ):
                raise ValueError(
                    "successful non-mutating receipt requires an exact output "
                    "binding or available evidence"
                )
            unavailable = _unavailable_required_evidence(self.evidence)
            if unavailable:
                raise ValueError(
                    "successful receipt has unavailable required evidence: "
                    + ", ".join(unavailable)
                )
        expected_status = _canonical_receipt_status(
            self.execution_status,
            self.review_disposition,
        )
        if self.receipt_status != expected_status:
            raise ValueError(
                "receipt_status does not match execution status and review disposition"
            )
        participants = [
            self.semantic_decision_owner,
            *self.evidence_providers,
            *self.proposal_providers,
            self.executor,
        ]
        bound_implementations = set(self.identity.digests.implementations.values())
        if any(
            participant.implementation_digest not in bound_implementations
            for participant in participants
        ):
            raise ValueError("receipt participant implementation is not identity-bound")
        return self


class PersistedExecutionLineage(BaseModel):
    """Complete persisted artifacts for every execution attempt before the current."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorizations: FrozenSequence[BoundedExecutionAuthorization] = ()
    results: FrozenSequence[EmbeddedBoundedExecutionResult] = ()
    reviews: FrozenSequence[EmbeddedCoordinatorReview] = ()

    @model_validator(mode="after")
    def validate_complete_prior_attempts(self) -> Self:
        if not (len(self.authorizations) == len(self.results) == len(self.reviews)):
            raise ValueError(
                "persisted execution lineage requires one authorization, result, "
                "and review per prior attempt"
            )
        expected_attempts = list(range(1, len(self.authorizations) + 1))
        if [item.attempt for item in self.authorizations] != expected_attempts:
            raise ValueError(
                "persisted execution lineage must contain contiguous attempts"
            )
        return self


def _require_exact_identity(
    artifact: ContractArtifact,
    expected: EmbeddedDecisionIdentity,
    *,
    label: str,
) -> None:
    if artifact.identity != expected:
        raise EmbeddedDecisionContractError(f"{label} identity or digests are stale")


def _revalidate_contract_model(model: BaseModel, *, label: str) -> None:
    try:
        type(model).model_validate(model.model_dump(mode="python"))
    except (TypeError, ValueError) as exc:
        raise EmbeddedDecisionContractError(
            f"{label} is malformed or was modified after validation: {exc}"
        ) from exc


def _require_nondecreasing_timestamp(
    earlier: datetime,
    later: datetime,
    *,
    label: str,
) -> None:
    if later < earlier:
        raise EmbeddedDecisionContractError(f"{label} timestamp precedes its parent")


def _operation_id(
    identity: EmbeddedDecisionIdentity,
    *,
    accepted_decision_digest: str,
    execution_effect: ExecutionEffect,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.embedded-operation-id.v1",
            "identity": identity.model_dump(mode="json"),
            "accepted_decision_digest": accepted_decision_digest,
            "execution_effect": execution_effect,
        }
    )


def _validate_executor_identity(
    executor: ProducerIdentity,
    identity: EmbeddedDecisionIdentity,
) -> None:
    if executor.role != "executor":
        raise EmbeddedDecisionContractError("bounded execution requires an executor")
    if executor.implementation_digest not in set(
        identity.digests.implementations.values()
    ):
        raise EmbeddedDecisionContractError(
            "executor implementation digest is not bound by the decision identity"
        )


def _validate_decision_inputs(
    decision: EmbeddedCoordinatorDecision,
    *,
    evidence: Sequence[EmbeddedDomainEvidence],
    proposals: Sequence[EmbeddedDomainProposal],
    require_available_evidence: bool = True,
) -> None:
    _revalidate_contract_model(decision.identity, label="decision identity")
    _revalidate_contract_model(decision, label="coordinator decision")
    for evidence_artifact in evidence:
        _revalidate_contract_model(evidence_artifact, label="evidence")
    for proposal_artifact in proposals:
        _revalidate_contract_model(proposal_artifact, label="proposal")
    evidence_refs = tuple(artifact_reference(item) for item in evidence)
    proposal_refs = tuple(artifact_reference(item) for item in proposals)
    evidence_by_ref = {
        reference: evidence_item
        for reference, evidence_item in zip(evidence_refs, evidence, strict=True)
    }
    if decision.evidence_artifacts != evidence_refs:
        raise EmbeddedDecisionContractError(
            "coordinator decision does not bind the exact evidence artifact set"
        )
    if decision.proposal_artifacts != proposal_refs:
        raise EmbeddedDecisionContractError(
            "coordinator decision does not bind the exact proposal artifact set"
        )
    for evidence_artifact in evidence:
        _require_exact_identity(
            evidence_artifact,
            decision.identity,
            label="evidence",
        )
        _require_nondecreasing_timestamp(
            evidence_artifact.created_at,
            decision.created_at,
            label="coordinator decision",
        )
        unavailable = [
            record.evidence_id
            for record in evidence_artifact.records
            if record.required and record.status != "available"
        ]
        if require_available_evidence and unavailable:
            raise EmbeddedDecisionContractError(
                "required evidence is unavailable: " + ", ".join(unavailable)
            )
    for proposal_artifact in proposals:
        _require_exact_identity(
            proposal_artifact,
            decision.identity,
            label="proposal",
        )
        _require_nondecreasing_timestamp(
            proposal_artifact.created_at,
            decision.created_at,
            label="coordinator decision",
        )
        if any(
            ref not in evidence_refs for ref in proposal_artifact.evidence_artifacts
        ):
            raise EmbeddedDecisionContractError(
                "proposal references evidence outside the coordinator decision"
            )
        if proposal_artifact.proposal_digest != canonical_json_digest(
            proposal_artifact.proposal
        ):
            raise EmbeddedDecisionContractError(
                "proposal digest is stale relative to its semantic payload"
            )
        for evidence_ref in proposal_artifact.evidence_artifacts:
            _require_nondecreasing_timestamp(
                evidence_by_ref[evidence_ref].created_at,
                proposal_artifact.created_at,
                label="proposal",
            )


def validate_coordinator_decision_dependencies(
    decision: EmbeddedCoordinatorDecision,
    *,
    evidence: Sequence[EmbeddedDomainEvidence],
    proposals: Sequence[EmbeddedDomainProposal],
) -> EmbeddedCoordinatorDecision:
    """Validate one decision against its exact ordered evidence and proposals."""

    _validate_decision_inputs(
        decision,
        evidence=evidence,
        proposals=proposals,
        require_available_evidence=decision.disposition == "accept",
    )
    return decision


def validate_human_decision(
    human_decision: EmbeddedHumanDecision,
    decision: EmbeddedCoordinatorDecision,
) -> EmbeddedHumanDecision:
    """Validate a human disposition against the exact accepted decision."""

    _revalidate_contract_model(decision, label="coordinator decision")
    _revalidate_contract_model(human_decision, label="human decision")
    _require_exact_identity(
        human_decision,
        decision.identity,
        label="human decision",
    )
    if decision.disposition != "accept" or decision.accepted_decision_digest is None:
        raise EmbeddedDecisionContractError(
            "human decision requires accepted coordinator semantics"
        )
    if not decision.human_decision_required:
        raise EmbeddedDecisionContractError(
            "decision does not permit an invented human authority"
        )
    if human_decision.coordinator_decision != artifact_reference(decision):
        raise EmbeddedDecisionContractError(
            "human decision does not bind the exact coordinator decision"
        )
    if human_decision.reviewed_decision_digest != decision.accepted_decision_digest:
        raise EmbeddedDecisionContractError(
            "human decision reviewed a different accepted decision digest"
        )
    _require_nondecreasing_timestamp(
        decision.created_at,
        human_decision.created_at,
        label="human decision",
    )
    return human_decision


def _validated_human_decision(
    decision: EmbeddedCoordinatorDecision,
    human_decision: EmbeddedHumanDecision | None,
) -> ContractArtifactReference | None:
    if human_decision is None:
        if decision.human_decision_required:
            raise EmbeddedDecisionContractError(
                "accepted decision requires an exact human decision"
            )
        return None
    if not decision.human_decision_required:
        raise EmbeddedDecisionContractError(
            "decision does not permit an invented human authority"
        )
    validate_human_decision(human_decision, decision)
    if human_decision.disposition != "accept":
        raise EmbeddedDecisionContractError(
            f"human decision is {human_decision.disposition}, not accept"
        )
    return artifact_reference(human_decision)


def _validate_resume_artifacts(
    authorization: BoundedExecutionAuthorization,
    resumed_from_authorization: BoundedExecutionAuthorization,
    resumed_from_result: EmbeddedBoundedExecutionResult,
    resumed_from_review: EmbeddedCoordinatorReview,
    decision: EmbeddedCoordinatorDecision,
) -> None:
    _revalidate_contract_model(
        resumed_from_authorization,
        label="resumed-from authorization",
    )
    _require_exact_identity(
        resumed_from_authorization,
        authorization.identity,
        label="resumed-from authorization",
    )
    _revalidate_contract_model(resumed_from_result, label="resumed-from result")
    _require_exact_identity(
        resumed_from_result,
        authorization.identity,
        label="resumed-from result",
    )
    if (
        resumed_from_authorization.accepted_decision != authorization.accepted_decision
        or resumed_from_authorization.accepted_decision_digest
        != authorization.accepted_decision_digest
        or resumed_from_authorization.producer != decision.producer
        or resumed_from_authorization.human_decision != authorization.human_decision
        or resumed_from_authorization.execution_effect != authorization.execution_effect
        or resumed_from_authorization.operation_id != authorization.operation_id
        or resumed_from_authorization.mutation_id != authorization.mutation_id
        or resumed_from_authorization.attempt != authorization.attempt - 1
    ):
        raise EmbeddedDecisionContractError(
            "authorization does not resume the exact prior authorization"
        )
    validate_bounded_execution_result(
        resumed_from_result,
        resumed_from_authorization,
    )
    if (
        resumed_from_result.accepted_decision != authorization.accepted_decision
        or resumed_from_result.accepted_decision_digest
        != authorization.accepted_decision_digest
        or resumed_from_result.execution_effect != authorization.execution_effect
        or resumed_from_result.operation_id != authorization.operation_id
        or resumed_from_result.mutation_id != authorization.mutation_id
        or resumed_from_result.attempt != authorization.attempt - 1
        or authorization.resume_of != artifact_reference(resumed_from_result)
    ):
        raise EmbeddedDecisionContractError(
            "authorization does not resume the exact prior-attempt result"
        )
    validate_coordinator_review(resumed_from_review, resumed_from_result)
    if (
        resumed_from_review.semantic_decision_owner != decision.producer
        or resumed_from_review.disposition != "retry"
        or authorization.resume_review != artifact_reference(resumed_from_review)
    ):
        raise EmbeddedDecisionContractError(
            "authorization does not resume the exact owner-reviewed retry"
        )
    _require_nondecreasing_timestamp(
        resumed_from_review.created_at,
        authorization.created_at,
        label="resumed execution authorization",
    )


def _validate_authorization_history(
    authorizations: Sequence[BoundedExecutionAuthorization],
    results: Sequence[EmbeddedBoundedExecutionResult],
    reviews: Sequence[EmbeddedCoordinatorReview],
    *,
    decision: EmbeddedCoordinatorDecision,
    expected_identity: EmbeddedDecisionIdentity,
    human_ref: ContractArtifactReference | None,
    execution_effect: ExecutionEffect,
    operation_id: Sha256Digest,
    mutation_id: Sha256Digest | None,
    authority_time: datetime,
) -> None:
    expected_attempts = list(range(1, len(authorizations) + 1))
    if [item.attempt for item in authorizations] != expected_attempts:
        raise EmbeddedDecisionContractError(
            "persisted authorization history must contain contiguous unique attempts"
        )
    expected_resume_count = max(0, len(authorizations) - 1)
    if len(results) != expected_resume_count or len(reviews) != expected_resume_count:
        raise EmbeddedDecisionContractError(
            "persisted authorization history requires every exact prior result and "
            "review"
        )
    decision_ref = artifact_reference(decision)
    for existing in authorizations:
        _revalidate_contract_model(existing, label="existing authorization")
        _require_exact_identity(
            existing,
            expected_identity,
            label="existing authorization",
        )
        _validate_executor_identity(existing.executor, expected_identity)
        if (
            existing.accepted_decision != decision_ref
            or existing.accepted_decision_digest != decision.accepted_decision_digest
            or existing.producer != decision.producer
            or existing.human_decision != human_ref
            or existing.execution_effect != execution_effect
            or existing.operation_id != operation_id
            or existing.mutation_id != mutation_id
        ):
            raise EmbeddedDecisionContractError(
                "persisted authorization belongs to another decision or operation"
            )
        _require_nondecreasing_timestamp(
            authority_time,
            existing.created_at,
            label="existing execution authorization",
        )
    for index in range(1, len(authorizations)):
        _validate_resume_artifacts(
            authorizations[index],
            authorizations[index - 1],
            results[index - 1],
            reviews[index - 1],
            decision,
        )


def authorize_bounded_execution(
    decision: ContractArtifact,
    *,
    expected_identity: EmbeddedDecisionIdentity,
    evidence: Sequence[EmbeddedDomainEvidence],
    existing_authorizations: Sequence[BoundedExecutionAuthorization],
    proposals: Sequence[EmbeddedDomainProposal] = (),
    executor: ProducerIdentity,
    execution_effect: ExecutionEffect,
    human_decision: EmbeddedHumanDecision | None = None,
    historical_results: Sequence[EmbeddedBoundedExecutionResult] = (),
    historical_reviews: Sequence[EmbeddedCoordinatorReview] = (),
    prior_result: EmbeddedBoundedExecutionResult | None = None,
    prior_review: EmbeddedCoordinatorReview | None = None,
    created_at: datetime | None = None,
) -> BoundedExecutionAuthorization:
    """Fail closed unless exact outer-authored semantics may be executed once."""

    if not isinstance(decision, EmbeddedCoordinatorDecision):
        raise EmbeddedDecisionContractError(
            "execution requires a coordinator-authored accepted decision; "
            "a proposal is not execution authority"
        )
    _revalidate_contract_model(expected_identity, label="expected decision identity")
    _require_exact_identity(decision, expected_identity, label="coordinator decision")
    _validate_decision_inputs(decision, evidence=evidence, proposals=proposals)
    if decision.disposition != "accept" or decision.accepted_decision_digest is None:
        raise EmbeddedDecisionContractError(
            f"coordinator decision is {decision.disposition}, not accepted"
        )
    if decision.accepted_decision is None:  # pragma: no cover - model invariant
        raise EmbeddedDecisionContractError("accepted decision lacks semantic values")
    try:
        _validate_execution_effect(expected_identity, execution_effect)
    except ValueError as exc:
        raise EmbeddedDecisionContractError(str(exc)) from exc
    expected_decision_digest = accepted_semantic_decision_digest(
        decision.identity,
        decision.accepted_decision,
    )
    if decision.accepted_decision_digest != expected_decision_digest:
        raise EmbeddedDecisionContractError(
            "accepted decision digest is stale relative to identity or values"
        )
    _validate_executor_identity(executor, expected_identity)
    human_ref = _validated_human_decision(decision, human_decision)
    decision_ref = artifact_reference(decision)
    operation_id = _operation_id(
        expected_identity,
        accepted_decision_digest=decision.accepted_decision_digest,
        execution_effect=execution_effect,
    )
    mutation_id = operation_id if execution_effect == "mutation" else None
    authority_time = (
        human_decision.created_at if human_decision is not None else decision.created_at
    )
    ordered_authorizations = sorted(
        existing_authorizations,
        key=lambda item: item.attempt,
    )
    _validate_authorization_history(
        ordered_authorizations,
        historical_results,
        historical_reviews,
        decision=decision,
        expected_identity=expected_identity,
        human_ref=human_ref,
        execution_effect=execution_effect,
        operation_id=operation_id,
        mutation_id=mutation_id,
        authority_time=authority_time,
    )
    attempt = 1
    resume_of: ContractArtifactReference | None = None
    resume_review: ContractArtifactReference | None = None
    if prior_result is not None:
        if not ordered_authorizations:
            raise EmbeddedDecisionContractError(
                "resume requires the persisted prior authorization claim"
            )
        prior_authorization = ordered_authorizations[-1]
        if prior_result.attempt != prior_authorization.attempt:
            raise EmbeddedDecisionContractError(
                "latest persisted authorization has no result; reconcile its stable "
                "operation_id before resuming an earlier attempt"
            )
        validate_bounded_execution_result(prior_result, prior_authorization)
        if prior_review is None:
            raise EmbeddedDecisionContractError(
                "resume requires an exact outer-coordinator review"
            )
        validate_coordinator_review(prior_review, prior_result)
        if prior_review.semantic_decision_owner != decision.producer:
            raise EmbeddedDecisionContractError(
                "prior review was not authored by the semantic decision owner"
            )
        if prior_review.disposition != "retry":
            raise EmbeddedDecisionContractError(
                f"prior review is {prior_review.disposition}, not retry"
            )
        retryable_states = (
            {"not_started"}
            if execution_effect == "mutation"
            else {"not_started", "not_applicable"}
        )
        if prior_result.mutation_state not in retryable_states:
            raise EmbeddedDecisionContractError(
                "prior attempt may already have mutated output; review or verify its "
                "retained outputs instead of repeating mutation"
            )
        resume_of = artifact_reference(prior_result)
        resume_review = artifact_reference(prior_review)
        attempt = prior_result.attempt + 1
    elif prior_review is not None:
        raise EmbeddedDecisionContractError("prior_review requires prior_result")
    elif ordered_authorizations:
        raise EmbeddedDecisionContractError(
            "persisted authorization has no result; reconcile its stable operation_id "
            "instead of issuing another claim"
        )
    parent_time = (
        prior_review.created_at
        if prior_review is not None
        else (human_decision.created_at if human_decision else decision.created_at)
    )
    authorized_at = created_at or parent_time
    _require_nondecreasing_timestamp(
        parent_time,
        authorized_at,
        label="execution authorization",
    )
    parent_artifact = resume_review or human_ref or decision_ref
    return BoundedExecutionAuthorization(
        artifact_id=f"authorization-{operation_id}-{attempt:03d}",
        identity=expected_identity,
        producer=decision.producer,
        parent_artifact=parent_artifact,
        accepted_decision=decision_ref,
        accepted_decision_digest=decision.accepted_decision_digest,
        human_decision=human_ref,
        executor=executor,
        execution_effect=execution_effect,
        outer_stage_attempt_seal_key=outer_stage_attempt_seal_key(expected_identity),
        operation_id=operation_id,
        mutation_id=mutation_id,
        attempt=attempt,
        resume_of=resume_of,
        resume_review=resume_review,
        created_at=authorized_at,
    )


def validate_bounded_execution_result(
    result: EmbeddedBoundedExecutionResult,
    authorization: BoundedExecutionAuthorization,
) -> EmbeddedBoundedExecutionResult:
    """Validate an executor result against the exact one-shot authorization."""

    _revalidate_contract_model(authorization, label="execution authorization")
    _revalidate_contract_model(result, label="execution result")
    _require_exact_identity(result, authorization.identity, label="execution result")
    if result.parent_artifact != artifact_reference(authorization):
        raise EmbeddedDecisionContractError(
            "execution result parent differs from its persisted authorization claim"
        )
    if (
        result.accepted_decision != authorization.accepted_decision
        or result.accepted_decision_digest != authorization.accepted_decision_digest
        or result.producer != authorization.executor
        or result.execution_effect != authorization.execution_effect
        or result.operation_id != authorization.operation_id
        or result.mutation_id != authorization.mutation_id
        or result.attempt != authorization.attempt
        or result.resume_of != authorization.resume_of
    ):
        raise EmbeddedDecisionContractError(
            "execution result differs from its exact bounded authorization"
        )
    _require_nondecreasing_timestamp(
        authorization.created_at,
        result.created_at,
        label="execution result",
    )
    return result


def validate_coordinator_review(
    review: EmbeddedCoordinatorReview,
    result: EmbeddedBoundedExecutionResult,
) -> EmbeddedCoordinatorReview:
    """Validate an outer review of exact result bytes and terminal semantics."""

    _revalidate_contract_model(result, label="execution result")
    _revalidate_contract_model(review, label="coordinator review")
    _require_exact_identity(review, result.identity, label="coordinator review")
    if review.execution_result != artifact_reference(result):
        raise EmbeddedDecisionContractError(
            "coordinator review does not bind the exact execution result"
        )
    if (
        review.accepted_decision_digest != result.accepted_decision_digest
        or review.outputs != result.outputs
    ):
        raise EmbeddedDecisionContractError(
            "coordinator review decision digest or outputs differ from execution"
        )
    allowed: dict[ExecutionStatus, set[CoordinatorReviewDisposition]] = {
        "succeeded": {"accept", "reject", "revise"},
        "failed": {"retry", "revise", "stop"},
        "interrupted": {"retry", "revise", "stop"},
        "cancelled": {"cancelled", "stop"},
    }
    if review.disposition not in allowed[result.status]:
        raise EmbeddedDecisionContractError(
            f"review disposition {review.disposition} is invalid for {result.status}"
        )
    if review.disposition == "retry":
        retryable_states = (
            {"not_started"}
            if result.execution_effect == "mutation"
            else {"not_started", "not_applicable"}
        )
        if result.mutation_state not in retryable_states:
            raise EmbeddedDecisionContractError(
                "retry would risk duplicate mutation; review retained outputs instead"
            )
    _require_nondecreasing_timestamp(
        result.created_at,
        review.created_at,
        label="coordinator review",
    )
    return review


def _unique_producers(items: Sequence[ProducerIdentity]) -> list[ProducerIdentity]:
    unique: list[ProducerIdentity] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def _receipt_status(
    result: EmbeddedBoundedExecutionResult,
    review: EmbeddedCoordinatorReview,
) -> ReceiptStatus:
    return _canonical_receipt_status(result.status, review.disposition)


def build_decision_receipt(
    *,
    artifact_id: str,
    decision: EmbeddedCoordinatorDecision,
    evidence: Sequence[EmbeddedDomainEvidence],
    proposals: Sequence[EmbeddedDomainProposal],
    authorization: BoundedExecutionAuthorization,
    result: EmbeddedBoundedExecutionResult,
    review: EmbeddedCoordinatorReview,
    prior_lineage: PersistedExecutionLineage,
    human_decision: EmbeddedHumanDecision | None = None,
    created_at: datetime | None = None,
) -> EmbeddedDecisionReceipt:
    """Build a receipt only after revalidating the complete exact chain."""

    _validate_decision_inputs(decision, evidence=evidence, proposals=proposals)
    _revalidate_contract_model(prior_lineage, label="persisted execution lineage")
    _revalidate_contract_model(authorization, label="execution authorization")
    _require_exact_identity(decision, authorization.identity, label="decision")
    if (
        authorization.accepted_decision != artifact_reference(decision)
        or authorization.accepted_decision_digest != decision.accepted_decision_digest
        or authorization.producer != decision.producer
    ):
        raise EmbeddedDecisionContractError(
            "receipt authorization belongs to another coordinator or decision"
        )
    human_ref = _validated_human_decision(decision, human_decision)
    if human_ref != authorization.human_decision:
        raise EmbeddedDecisionContractError(
            "receipt human decision differs from execution authorization"
        )
    authority_time = (
        human_decision.created_at if human_decision is not None else decision.created_at
    )
    _require_nondecreasing_timestamp(
        authority_time,
        authorization.created_at,
        label="receipt execution authorization",
    )
    if len(prior_lineage.authorizations) != authorization.attempt - 1:
        raise EmbeddedDecisionContractError(
            "receipt requires the complete persisted lineage through attempt one"
        )
    _validate_authorization_history(
        [*prior_lineage.authorizations, authorization],
        prior_lineage.results,
        prior_lineage.reviews,
        decision=decision,
        expected_identity=authorization.identity,
        human_ref=human_ref,
        execution_effect=authorization.execution_effect,
        operation_id=authorization.operation_id,
        mutation_id=authorization.mutation_id,
        authority_time=authority_time,
    )
    validate_bounded_execution_result(result, authorization)
    validate_coordinator_review(review, result)
    if review.semantic_decision_owner != decision.producer:
        raise EmbeddedDecisionContractError(
            "coordinator review was not authored by the semantic decision owner"
        )
    receipt_time = created_at or review.created_at
    _require_nondecreasing_timestamp(
        review.created_at,
        receipt_time,
        label="decision receipt",
    )
    review_ref = artifact_reference(review)
    return EmbeddedDecisionReceipt(
        artifact_id=artifact_id,
        identity=decision.identity,
        producer=decision.producer,
        parent_artifact=review_ref,
        created_at=receipt_time,
        receipt_status=_receipt_status(result, review),
        semantic_decision_owner=decision.producer,
        evidence_providers=_unique_producers([item.producer for item in evidence]),
        proposal_providers=_unique_producers([item.producer for item in proposals]),
        accepted_decision=artifact_reference(decision),
        accepted_decision_digest=authorization.accepted_decision_digest,
        human_decision=human_ref,
        executor=result.producer,
        execution_authorization=artifact_reference(authorization),
        execution_result=artifact_reference(result),
        execution_effect=result.execution_effect,
        outer_stage_attempt_seal_key=authorization.outer_stage_attempt_seal_key,
        operation_id=result.operation_id,
        mutation_id=result.mutation_id,
        attempt=authorization.attempt,
        prior_execution_authorizations=[
            artifact_reference(item) for item in prior_lineage.authorizations
        ],
        prior_execution_results=[
            artifact_reference(item) for item in prior_lineage.results
        ],
        prior_coordinator_reviews=[
            artifact_reference(item) for item in prior_lineage.reviews
        ],
        outputs=result.outputs,
        evidence=result.evidence,
        coordinator_review=review_ref,
        execution_status=result.status,
        review_disposition=review.disposition,
    )


def validate_decision_receipt(
    receipt: EmbeddedDecisionReceipt,
    *,
    decision: EmbeddedCoordinatorDecision,
    evidence: Sequence[EmbeddedDomainEvidence],
    proposals: Sequence[EmbeddedDomainProposal],
    authorization: BoundedExecutionAuthorization,
    result: EmbeddedBoundedExecutionResult,
    review: EmbeddedCoordinatorReview,
    prior_lineage: PersistedExecutionLineage,
    human_decision: EmbeddedHumanDecision | None = None,
) -> EmbeddedDecisionReceipt:
    """Revalidate a deserialized receipt against every exact persisted artifact."""

    _revalidate_contract_model(receipt, label="decision receipt")
    expected = build_decision_receipt(
        artifact_id=receipt.artifact_id,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=prior_lineage,
        human_decision=human_decision,
        created_at=receipt.created_at,
    )
    if receipt != expected:
        raise EmbeddedDecisionContractError(
            "decision receipt differs from its complete persisted execution lineage"
        )
    return receipt


__all__ = [
    "EMBEDDED_COORDINATOR_DECISION_SCHEMA_VERSION",
    "EMBEDDED_COORDINATOR_REVIEW_SCHEMA_VERSION",
    "EMBEDDED_DECISION_EVIDENCE_SCHEMA_VERSION",
    "EMBEDDED_DECISION_PROPOSAL_SCHEMA_VERSION",
    "EMBEDDED_DECISION_RECEIPT_SCHEMA_VERSION",
    "EMBEDDED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION",
    "EMBEDDED_EXECUTION_RESULT_SCHEMA_VERSION",
    "EMBEDDED_HUMAN_DECISION_SCHEMA_VERSION",
    "AcceptedSemanticDecision",
    "ArtifactKind",
    "BoundedExecutionAuthorization",
    "ContractArtifact",
    "ContractArtifactReference",
    "CoordinatorDecisionDisposition",
    "CoordinatorReviewDisposition",
    "DomainProposalPayload",
    "EmbeddedBoundedExecutionResult",
    "EmbeddedCoordinatorDecision",
    "EmbeddedCoordinatorReview",
    "EmbeddedDecisionContractError",
    "EmbeddedDecisionIdentity",
    "EmbeddedDecisionReceipt",
    "EmbeddedDomainEvidence",
    "EmbeddedDomainProposal",
    "EmbeddedHumanDecision",
    "EvidenceStatus",
    "EvidenceType",
    "ExecutionEffect",
    "ExecutionStatus",
    "HumanDecisionDisposition",
    "MutationState",
    "NamedDecisionDigests",
    "PersistedExecutionLineage",
    "ProducerIdentity",
    "ProducerRole",
    "ProviderNeutralEvidenceRecord",
    "ReceiptStatus",
    "accepted_semantic_decision_digest",
    "artifact_reference",
    "authorize_bounded_execution",
    "build_decision_receipt",
    "canonical_json_digest",
    "outer_stage_attempt_seal_key",
    "validate_bounded_execution_result",
    "validate_coordinator_decision_dependencies",
    "validate_coordinator_review",
    "validate_decision_receipt",
    "validate_human_decision",
]
