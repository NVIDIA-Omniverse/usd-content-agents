# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract and negative tests for embedded semantic decision ownership."""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from datetime import UTC, datetime, timedelta
from operator import setitem
from typing import cast

import pytest
from pydantic import ValidationError

import content_agent_workflows.common as common_contracts
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
    DomainExecutionContext,
    DomainName,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
)
from content_agent_workflows.common.embedded_domain_decision import (
    AcceptedSemanticDecision,
    BoundedExecutionAuthorization,
    ContractArtifactReference,
    CoordinatorDecisionDisposition,
    CoordinatorReviewDisposition,
    DomainProposalPayload,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionContractError,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
    EvidenceStatus,
    ExecutionEffect,
    ExecutionStatus,
    HumanDecisionDisposition,
    MutationState,
    NamedDecisionDigests,
    PersistedExecutionLineage,
    ProducerIdentity,
    ProducerRole,
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

_START = datetime(2026, 8, 9, 20, 0, tzinfo=UTC)


def test_common_package_exports_shared_decision_surfaces() -> None:
    assert common_contracts.BoundedExecutionAuthorization is (
        BoundedExecutionAuthorization
    )
    assert common_contracts.ExecutionEffect is not None
    assert common_contracts.EMBEDDED_EXECUTION_AUTHORIZATION_SCHEMA_VERSION.endswith(
        ".v1"
    )
    assert common_contracts.PersistedExecutionLineage is PersistedExecutionLineage
    assert common_contracts.validate_decision_receipt is validate_decision_receipt
    assert common_contracts.EmbeddedDecisionArtifactStore is not None
    assert common_contracts.EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION.endswith(
        ".v1"
    )


def _binding(path: str, fill: str) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(path=path, sha256=fill * 64, size_bytes=1)


def _producer(role: ProducerRole, name: str, fill: str) -> ProducerIdentity:
    return ProducerIdentity(
        producer_id=name,
        role=role,
        implementation=f"test.{name}.v1",
        implementation_digest=None if role == "human_reviewer" else fill * 64,
    )


def _plan_parent() -> ContractArtifactReference:
    return ContractArtifactReference(
        artifact_kind="coordinator_plan",
        artifact_id="plan-001",
        schema_version="content-agent-workflows.asset-coordinator-plan.v1",
        sha256="2" * 64,
    )


def _identity(
    *,
    domain: DomainName = "articulation",
    source_fill: str = "3",
    configuration_fill: str = "4",
    prompt_fill: str = "5",
    reference_fill: str | None = None,
    capability_fill: str = "7",
    implementation_fill: str = "6",
) -> EmbeddedDecisionIdentity:
    source = _binding("/run/input.usdz", source_fill)
    context = DomainExecutionContext(
        schema_version=(
            DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2
            if domain == "validation"
            else DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION
        ),
        domain=domain,
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=_binding("/run/request.json", "1"),
            stage=domain,
            stage_attempt=1,
            coordinator_plan=_binding("/run/coordinator/plan-001.json", "2"),
            input_asset=source,
            domain_run_root=f"/run/stages/01-{domain}/domain-run",
        ),
    )
    return EmbeddedDecisionIdentity(
        execution_context=context,
        source=source,
        coordinator_plan=_plan_parent(),
        digests=NamedDecisionDigests(
            configuration={f"{domain}_config": configuration_fill * 64},
            prompt={"asset_prompt": prompt_fill * 64},
            references=(
                {} if reference_fill is None else {"reference_001": reference_fill * 64}
            ),
            capabilities={f"{domain}_inspection": capability_fill * 64},
            implementations={
                f"{domain}_adapter": implementation_fill * 64,
                "inspection_provider": "8" * 64,
                "proposal_provider": "a" * 64,
                "asset_coordinator": "b" * 64,
                "joint_authorer": "c" * 64,
            },
        ),
    )


def _evidence(
    identity: EmbeddedDecisionIdentity,
    *,
    status: EvidenceStatus = "available",
) -> EmbeddedDomainEvidence:
    return EmbeddedDomainEvidence(
        artifact_id="evidence-001",
        identity=identity,
        producer=_producer("evidence_provider", "inspection-provider", "8"),
        parent_artifact=_plan_parent(),
        created_at=_START,
        records=[
            ProviderNeutralEvidenceRecord(
                evidence_id="topology",
                evidence_type="inspection",
                status=status,
                required=True,
                summary=(
                    "Exact topology inventory."
                    if status == "available"
                    else "Inspection provider unavailable."
                ),
                artifacts=(
                    [_binding("/run/evidence/topology.json", "9")]
                    if status == "available"
                    else []
                ),
            )
        ],
    )


def _proposal(
    identity: EmbeddedDecisionIdentity,
    evidence: EmbeddedDomainEvidence,
) -> EmbeddedDomainProposal:
    payload = DomainProposalPayload(
        schema_version=(f"{identity.execution_context.domain}.semantic-proposal.v1"),
        values={"candidate_ids": ["hinge"]},
    )
    evidence_ref = artifact_reference(evidence)
    return EmbeddedDomainProposal(
        artifact_id="proposal-001",
        identity=identity,
        producer=_producer("proposal_provider", "joint-provider", "a"),
        parent_artifact=evidence_ref,
        created_at=_START + timedelta(seconds=1),
        evidence_artifacts=[evidence_ref],
        proposal=payload,
        proposal_digest=canonical_json_digest(payload),
    )


def _decision(
    identity: EmbeddedDecisionIdentity,
    evidence: EmbeddedDomainEvidence,
    proposal: EmbeddedDomainProposal,
    *,
    disposition: CoordinatorDecisionDisposition = "accept",
    human_required: bool = True,
) -> EmbeddedCoordinatorDecision:
    accepted = AcceptedSemanticDecision(
        schema_version=f"{identity.execution_context.domain}.semantic-decision.v1",
        values={"joints": [{"id": "hinge", "type": "revolute"}]},
    )
    return EmbeddedCoordinatorDecision(
        artifact_id="decision-001",
        identity=identity,
        producer=_producer("outer_coordinator", "asset-coordinator", "b"),
        parent_artifact=artifact_reference(proposal),
        created_at=_START + timedelta(seconds=2),
        disposition=disposition,
        evidence_artifacts=[artifact_reference(evidence)],
        proposal_artifacts=[artifact_reference(proposal)],
        accepted_decision=accepted if disposition == "accept" else None,
        accepted_decision_digest=(
            accepted_semantic_decision_digest(identity, accepted)
            if disposition == "accept"
            else None
        ),
        human_decision_required=human_required,
        rationale=f"Outer coordinator chose {disposition}.",
        revision_requests=(
            ["Provide complete member coverage."] if disposition == "revise" else []
        ),
    )


def _human(
    decision: EmbeddedCoordinatorDecision,
    *,
    disposition: HumanDecisionDisposition = "accept",
) -> EmbeddedHumanDecision:
    assert decision.accepted_decision_digest is not None
    decision_ref = artifact_reference(decision)
    return EmbeddedHumanDecision(
        artifact_id="human-001",
        identity=decision.identity,
        producer=_producer("human_reviewer", "reviewer", "0"),
        parent_artifact=decision_ref,
        created_at=_START + timedelta(seconds=3),
        coordinator_decision=decision_ref,
        reviewed_decision_digest=decision.accepted_decision_digest,
        disposition=disposition,
        rationale=f"Human chose {disposition}.",
        revision_requests=(
            ["Split the candidate graph."] if disposition == "revise" else []
        ),
    )


def _authorization_chain(
    *,
    domain: DomainName = "articulation",
) -> tuple[
    EmbeddedDecisionIdentity,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedCoordinatorDecision,
    EmbeddedHumanDecision,
    ProducerIdentity,
]:
    identity = _identity(domain=domain)
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)
    decision = _decision(identity, evidence, proposal)
    human = _human(decision)
    executor = _producer("executor", "joint-authorer", "c")
    return identity, evidence, proposal, decision, human, executor


def _result(
    *,
    authorization: BoundedExecutionAuthorization,
    decision: EmbeddedCoordinatorDecision,
    executor: ProducerIdentity,
    status: ExecutionStatus = "succeeded",
    mutation_state: MutationState = "verified",
    created_at: datetime | None = None,
) -> EmbeddedBoundedExecutionResult:
    outputs = (
        [_binding("/run/stages/01-articulation/rigged.usdz", "d")]
        if mutation_state != "not_started"
        else []
    )
    assert decision.accepted_decision_digest is not None
    return EmbeddedBoundedExecutionResult(
        artifact_id=f"result-{authorization.attempt:03d}",
        identity=decision.identity,
        producer=executor,
        parent_artifact=artifact_reference(authorization),
        created_at=created_at or _START + timedelta(seconds=5),
        accepted_decision=artifact_reference(decision),
        accepted_decision_digest=decision.accepted_decision_digest,
        execution_effect=authorization.execution_effect,
        operation_id=authorization.operation_id,
        mutation_id=authorization.mutation_id,
        attempt=authorization.attempt,
        resume_of=authorization.resume_of,
        status=status,
        mutation_state=mutation_state,
        outputs=outputs,
        error=None if status == "succeeded" else f"Execution {status}.",
    )


def _review(
    result: EmbeddedBoundedExecutionResult,
    decision: EmbeddedCoordinatorDecision,
    *,
    disposition: CoordinatorReviewDisposition,
    created_at: datetime | None = None,
) -> EmbeddedCoordinatorReview:
    result_ref = artifact_reference(result)
    return EmbeddedCoordinatorReview(
        artifact_id=f"review-{result.attempt:03d}",
        identity=result.identity,
        producer=decision.producer,
        parent_artifact=result_ref,
        created_at=created_at or result.created_at + timedelta(seconds=1),
        execution_result=result_ref,
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=result.accepted_decision_digest,
        disposition=disposition,
        outputs=result.outputs,
        findings=[f"Outer coordinator chose {disposition}."],
    )


def _available_execution_evidence() -> ProviderNeutralEvidenceRecord:
    return ProviderNeutralEvidenceRecord(
        evidence_id="mutation-readback",
        evidence_type="validation",
        status="available",
        required=True,
        summary="Mutation readback passed.",
        facts={"passed": True},
    )


def test_success_chain_records_exact_decision_execution_review_and_receipt() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    validate_bounded_execution_result(result, authorization)
    review = _review(result, decision, disposition="accept")
    validate_coordinator_review(review, result)

    receipt = build_decision_receipt(
        artifact_id="receipt-001",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
    )
    reconstructed_receipt = build_decision_receipt(
        artifact_id="receipt-001",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
    )

    assert receipt == reconstructed_receipt
    assert receipt.created_at == review.created_at
    assert receipt.receipt_status == "completed"
    assert receipt.semantic_decision_owner == decision.producer
    assert receipt.evidence_providers == (evidence.producer,)
    assert receipt.proposal_providers == (proposal.producer,)
    assert receipt.accepted_decision_digest == decision.accepted_decision_digest
    assert receipt.executor == executor
    assert receipt.execution_authorization == artifact_reference(authorization)
    assert receipt.execution_effect == "mutation"
    assert (
        receipt.outer_stage_attempt_seal_key
        == authorization.outer_stage_attempt_seal_key
    )
    assert receipt.operation_id == authorization.operation_id
    assert receipt.outputs == result.outputs
    assert receipt.review_disposition == "accept"
    stale_receipt_payload = receipt.model_dump(mode="python")
    stale_receipt_payload["outer_stage_attempt_seal_key"] = "f" * 64
    with pytest.raises(ValidationError, match="stage-attempt seal"):
        EmbeddedDecisionReceipt.model_validate(stale_receipt_payload)


@pytest.mark.parametrize("disposition", ["reject", "revise"])
def test_rejection_and_revision_cannot_reach_mutation(
    disposition: CoordinatorDecisionDisposition,
) -> None:
    identity = _identity()
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)
    decision = _decision(
        identity,
        evidence,
        proposal,
        disposition=disposition,
        human_required=False,
    )

    with pytest.raises(
        EmbeddedDecisionContractError,
        match=f"{disposition}, not accepted",
    ):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=_producer("executor", "joint-authorer", "c"),
            execution_effect="mutation",
        )


def test_required_human_decision_cannot_be_omitted() -> None:
    identity, evidence, proposal, decision, _human_decision, executor = (
        _authorization_chain()
    )

    with pytest.raises(EmbeddedDecisionContractError, match="requires an exact human"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
        )


@pytest.mark.parametrize("disposition", ["reject", "revise"])
def test_non_accepting_human_decision_cannot_authorize_execution(
    disposition: HumanDecisionDisposition,
) -> None:
    identity, evidence, proposal, decision, _human_decision, executor = (
        _authorization_chain()
    )

    with pytest.raises(
        EmbeddedDecisionContractError,
        match=f"human decision is {disposition}",
    ):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=_human(decision, disposition=disposition),
        )


def test_proposal_cannot_bypass_coordinator_decision() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)

    with pytest.raises(EmbeddedDecisionContractError, match="proposal is not"):
        authorize_bounded_execution(
            proposal,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            executor=_producer("executor", "joint-authorer", "c"),
            execution_effect="mutation",
        )


def test_unavailable_required_provider_evidence_fails_closed() -> None:
    identity = _identity()
    evidence = _evidence(identity, status="unavailable")
    proposal = _proposal(identity, evidence)
    decision = _decision(identity, evidence, proposal, human_required=False)

    with pytest.raises(EmbeddedDecisionContractError, match="unavailable: topology"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=_producer("executor", "joint-authorer", "c"),
            execution_effect="mutation",
        )


def test_malformed_proposal_is_rejected_before_decision() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    payload = DomainProposalPayload(
        schema_version="joint.candidate-graph-proposal.v1",
        values={"candidate_ids": ["hinge"]},
    )
    evidence_ref = artifact_reference(evidence)

    with pytest.raises(ValidationError, match="proposal_digest"):
        EmbeddedDomainProposal(
            artifact_id="proposal-malformed",
            identity=identity,
            producer=_producer("proposal_provider", "joint-provider", "a"),
            parent_artifact=evidence_ref,
            created_at=_START + timedelta(seconds=1),
            evidence_artifacts=[evidence_ref],
            proposal=payload,
            proposal_digest="f" * 64,
        )


def test_proposal_payload_drift_is_rejected_at_authorization() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)
    tampered_payload = proposal.proposal.model_copy(
        update={"values": {"candidate_ids": ["hinge"], "tampered": True}}
    )
    proposal = proposal.model_copy(update={"proposal": tampered_payload})
    decision = _decision(identity, evidence, proposal, human_required=False)

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=_producer("executor", "joint-authorer", "c"),
            execution_effect="mutation",
        )


def test_nested_contract_containers_are_structurally_immutable() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)

    with pytest.raises(TypeError):
        setitem(
            cast(MutableMapping[str, object], proposal.proposal.values),
            "tampered",
            True,
        )
    with pytest.raises(TypeError):
        setitem(
            cast(
                MutableSequence[object],
                proposal.proposal.values["candidate_ids"],
            ),
            0,
            "replacement",
        )
    with pytest.raises(AttributeError):
        proposal.evidence_artifacts.append(  # type: ignore[attr-defined]
            artifact_reference(evidence)
        )


def test_contract_revalidation_normalizes_serializer_type_errors() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    malformed_payload = proposal.proposal.model_copy(update={"values": object()})
    malformed_proposal = proposal.model_copy(update={"proposal": malformed_payload})

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[malformed_proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


@pytest.mark.parametrize(
    "tampered_artifact",
    ["decision_inputs", "evidence_plan", "proposal_evidence", "human_decision"],
)
def test_nested_upstream_chain_tampering_fails_closed(
    tampered_artifact: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    supplied_evidence = [evidence]
    supplied_proposals = [proposal]
    if tampered_artifact == "decision_inputs":
        decision = decision.model_copy(
            update={"evidence_artifacts": (), "proposal_artifacts": ()}
        )
        supplied_evidence = []
        supplied_proposals = []
    elif tampered_artifact == "evidence_plan":
        object.__setattr__(
            evidence,
            "parent_artifact",
            _plan_parent().model_copy(update={"artifact_id": "substituted-plan"}),
        )
    elif tampered_artifact == "proposal_evidence":
        proposal = proposal.model_copy(update={"evidence_artifacts": ()})
        supplied_proposals = [proposal]
    else:
        human = human.model_copy(
            update={"revision_requests": ("Late unreviewed revision.",)}
        )

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=supplied_evidence,
            existing_authorizations=[],
            proposals=supplied_proposals,
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


@pytest.mark.parametrize(
    ("substitution", "error"),
    [
        ("missing_evidence", "exact evidence artifact set"),
        ("missing_proposal", "exact proposal artifact set"),
        ("extra_evidence", "exact evidence artifact set"),
    ],
)
def test_decision_chain_rejects_missing_and_extra_artifact_substitutions(
    substitution: str,
    error: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    supplied_evidence = [evidence]
    supplied_proposals = [proposal]
    if substitution == "missing_evidence":
        supplied_evidence = []
    elif substitution == "missing_proposal":
        supplied_proposals = []
    else:
        supplied_evidence.append(
            evidence.model_copy(update={"artifact_id": "evidence-extra"})
        )

    with pytest.raises(EmbeddedDecisionContractError, match=error):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=supplied_evidence,
            existing_authorizations=[],
            proposals=supplied_proposals,
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


@pytest.mark.parametrize("field", ["artifact_id", "schema_version"])
def test_evidence_rejects_inexact_coordinator_plan_parent(field: str) -> None:
    identity = _identity()
    parent_payload = _plan_parent().model_dump(mode="python")
    parent_payload[field] = "wrong-plan" if field == "artifact_id" else "wrong.v999"

    with pytest.raises(ValidationError, match="exact active coordinator plan"):
        EmbeddedDomainEvidence(
            artifact_id="evidence-wrong-parent",
            identity=identity,
            producer=_producer("evidence_provider", "inspection-provider", "8"),
            parent_artifact=ContractArtifactReference.model_validate(parent_payload),
            created_at=_START,
            records=[
                ProviderNeutralEvidenceRecord(
                    evidence_id="topology",
                    evidence_type="inspection",
                    status="available",
                    summary="Exact topology inventory.",
                    artifacts=[_binding("/run/evidence/topology.json", "9")],
                )
            ],
        )


def test_semantic_payload_drift_is_rejected_at_authorization() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    assert decision.accepted_decision is not None
    tampered_decision = decision.accepted_decision.model_copy(
        update={
            "values": {
                "joints": [{"id": "hinge", "type": "revolute"}],
                "tampered": True,
            }
        }
    )
    decision = decision.model_copy(update={"accepted_decision": tampered_decision})

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


def test_executor_implementation_must_be_bound_by_identity() -> None:
    identity, evidence, proposal, decision, human, _executor = _authorization_chain()

    with pytest.raises(EmbeddedDecisionContractError, match="implementation digest"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=_producer("executor", "unbound-authorer", "f"),
            execution_effect="mutation",
            human_decision=human,
        )


@pytest.mark.parametrize(
    "stale_kind",
    [
        "source",
        "configuration",
        "prompt",
        "references",
        "capabilities",
        "implementations",
    ],
)
def test_stale_bound_identity_fails_closed(
    stale_kind: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    stale_identities = {
        "source": _identity(source_fill="e"),
        "configuration": _identity(configuration_fill="e"),
        "prompt": _identity(prompt_fill="e"),
        "references": _identity(reference_fill="e"),
        "capabilities": _identity(capability_fill="e"),
        "implementations": _identity(implementation_fill="e"),
    }
    stale_identity = stale_identities[stale_kind]

    with pytest.raises(EmbeddedDecisionContractError, match="stale"):
        authorize_bounded_execution(
            decision,
            expected_identity=stale_identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


def test_interruption_resumes_same_decision_without_duplicate_mutation() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    interrupted_review = _review(interrupted, decision, disposition="retry")
    validate_bounded_execution_result(interrupted, first)
    validate_coordinator_review(interrupted_review, interrupted)

    resumed = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=interrupted_review,
        created_at=_START + timedelta(seconds=7),
    )

    assert resumed.attempt == 2
    assert resumed.mutation_id == first.mutation_id
    assert resumed.resume_of == artifact_reference(interrupted)
    assert resumed.resume_review == artifact_reference(interrupted_review)


@pytest.mark.parametrize("attempt", [1, 2])
def test_authorization_schema_rejects_missing_or_invented_resume_links(
    attempt: int,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    payload = first.model_dump(mode="python")
    payload["attempt"] = attempt
    if attempt == 1:
        payload["resume_of"] = ContractArtifactReference(
            artifact_kind="execution_result",
            artifact_id="invented-result",
            schema_version="content-agent-workflows.embedded-execution-result.v1",
            sha256="e" * 64,
        )
        payload["resume_review"] = ContractArtifactReference(
            artifact_kind="coordinator_review",
            artifact_id="invented-review",
            schema_version="content-agent-workflows.embedded-coordinator-review.v1",
            sha256="f" * 64,
        )
        payload["parent_artifact"] = payload["resume_review"]

    expected = "first authorization attempt" if attempt == 1 else "after the first"
    with pytest.raises(ValidationError, match=expected):
        BoundedExecutionAuthorization.model_validate(payload)


def test_persisted_resume_link_must_match_exact_prior_result_and_review() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    review = _review(interrupted, decision, disposition="retry")
    second = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=review,
        created_at=_START + timedelta(seconds=7),
    )
    assert second.resume_of is not None
    wrong_result_ref = second.resume_of.model_copy(update={"sha256": "e" * 64})
    forged_second = second.model_copy(update={"resume_of": wrong_result_ref})

    with pytest.raises(EmbeddedDecisionContractError, match="exact prior-attempt"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[first, forged_second],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            historical_results=[interrupted],
            historical_reviews=[review],
        )


def test_persisted_resume_history_requires_exact_artifacts_and_chronology() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    review = _review(interrupted, decision, disposition="retry")
    second = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=review,
        created_at=_START + timedelta(seconds=7),
    )

    with pytest.raises(EmbeddedDecisionContractError, match="every exact prior"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[first, second],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )

    early_second = second.model_copy(
        update={"created_at": review.created_at - timedelta(seconds=1)}
    )
    with pytest.raises(EmbeddedDecisionContractError, match="timestamp precedes"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[first, early_second],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            historical_results=[interrupted],
            historical_reviews=[review],
        )


@pytest.mark.parametrize("drift", ["missing", "different"])
def test_persisted_human_gated_authorization_rejects_human_ref_drift(
    drift: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    if drift == "missing":
        forged = authorization.model_copy(
            update={
                "human_decision": None,
                "parent_artifact": artifact_reference(decision),
            }
        )
    else:
        other_human = artifact_reference(human).model_copy(
            update={"artifact_id": "human-other", "sha256": "e" * 64}
        )
        forged = authorization.model_copy(
            update={"human_decision": other_human, "parent_artifact": other_human}
        )

    with pytest.raises(EmbeddedDecisionContractError, match="another decision"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[forged],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


def test_non_human_gated_chain_rejects_invented_human_authority() -> None:
    identity = _identity()
    evidence = _evidence(identity)
    proposal = _proposal(identity, evidence)
    decision = _decision(
        identity,
        evidence,
        proposal,
        human_required=False,
    )
    human = _human(decision)
    executor = _producer("executor", "joint-authorer", "c")

    with pytest.raises(EmbeddedDecisionContractError, match="invented human"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
        )


def test_persisted_authorization_without_result_fails_closed_on_resume() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )

    assert authorization.artifact_kind == "execution_authorization"
    with pytest.raises(EmbeddedDecisionContractError, match="has no result"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[authorization],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            created_at=_START + timedelta(seconds=5),
        )


def test_forged_persisted_operation_identity_fails_schema_and_result_boundaries() -> (
    None
):
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    payload = authorization.model_dump(mode="python")
    payload["operation_id"] = "f" * 64
    payload["mutation_id"] = "f" * 64

    with pytest.raises(ValidationError, match="operation_id does not match"):
        BoundedExecutionAuthorization.model_validate(payload)

    stale_seal_payload = authorization.model_dump(mode="python")
    stale_seal_payload["outer_stage_attempt_seal_key"] = "f" * 64
    with pytest.raises(ValidationError, match="stage-attempt seal"):
        BoundedExecutionAuthorization.model_validate(stale_seal_payload)

    forged = authorization.model_copy(
        update={"operation_id": "f" * 64, "mutation_id": "f" * 64}
    )
    result = _result(
        authorization=forged,
        decision=decision,
        executor=executor,
    )
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_bounded_execution_result(result, forged)


def test_reviewed_pre_mutation_resume_keeps_operation_id_across_workers() -> None:
    identity, evidence, proposal, decision, human, first_executor = (
        _authorization_chain()
    )
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=first_executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=first_executor,
        status="interrupted",
        mutation_state="not_started",
    )
    review = _review(interrupted, decision, disposition="retry")
    next_executor = _producer("executor", "joint-authorer-worker-2", "c")

    resumed = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=next_executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=review,
        created_at=_START + timedelta(seconds=7),
    )

    assert resumed.operation_id == first.operation_id
    assert resumed.mutation_id == first.mutation_id
    assert resumed.executor == next_executor
    assert resumed.attempt == 2


def test_resume_rejects_review_from_a_different_semantic_owner() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    other_owner = _producer("outer_coordinator", "other-coordinator", "b")
    wrong_review = _review(
        interrupted,
        decision,
        disposition="retry",
    ).model_copy(
        update={
            "producer": other_owner,
            "semantic_decision_owner": other_owner,
        }
    )

    with pytest.raises(EmbeddedDecisionContractError, match="semantic decision owner"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[authorization],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            prior_result=interrupted,
            prior_review=wrong_review,
            created_at=_START + timedelta(seconds=7),
        )


def test_interruption_after_mutation_cannot_retry_and_duplicate_output() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="applied",
    )
    unsafe_review = _review(interrupted, decision, disposition="retry")

    with pytest.raises(EmbeddedDecisionContractError, match="duplicate mutation"):
        validate_coordinator_review(unsafe_review, interrupted)


@pytest.mark.parametrize("retained_kind", ["outputs", "evidence"])
def test_mutating_not_started_result_cannot_retain_artifacts(
    retained_kind: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    payload = result.model_dump(mode="python")
    if retained_kind == "outputs":
        output = _binding("/run/output.usdz", "e")
        payload["outputs"] = [output]
        result = result.model_copy(update={"outputs": (output,)})
    else:
        record = ProviderNeutralEvidenceRecord(
            evidence_id="partial",
            evidence_type="artifact",
            status="available",
            summary="Unexpected retained evidence.",
            facts={"retained": True},
        )
        payload["evidence"] = [record]
        result = result.model_copy(update={"evidence": (record,)})

    with pytest.raises(ValidationError, match="not_started result cannot retain"):
        EmbeddedBoundedExecutionResult.model_validate(payload)
    review = _review(result, decision, disposition="retry")
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_coordinator_review(review, result)


@pytest.mark.parametrize(
    ("domain", "execution_effect", "mutation_state"),
    [
        ("articulation", "mutation", "verified"),
        ("validation", "non_mutating", "not_applicable"),
    ],
)
def test_interrupted_execution_cannot_be_accepted_or_receipted_completed(
    domain: DomainName,
    execution_effect: ExecutionEffect,
    mutation_state: MutationState,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain=domain
    )
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect=execution_effect,
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state=mutation_state,
    )
    review = _review(interrupted, decision, disposition="accept")

    with pytest.raises(EmbeddedDecisionContractError, match="invalid for interrupted"):
        validate_coordinator_review(review, interrupted)
    with pytest.raises(EmbeddedDecisionContractError, match="invalid for interrupted"):
        build_decision_receipt(
            artifact_id="receipt-interrupted",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=authorization,
            result=interrupted,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
            created_at=_START + timedelta(seconds=7),
        )


def test_non_mutating_validation_result_completes_without_mutation_identity() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain="validation"
    )
    assert (
        identity.execution_context.schema_version
        == DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2
    )
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="non_mutating",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        mutation_state="not_applicable",
    )
    review = _review(result, decision, disposition="accept")
    receipt = build_decision_receipt(
        artifact_id="receipt-validation",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )

    assert receipt.receipt_status == "completed"
    assert receipt.execution_effect == "non_mutating"
    assert receipt.operation_id == authorization.operation_id
    assert receipt.mutation_id is None
    assert result.mutation_state == "not_applicable"

    for artifact_type, artifact in (
        (BoundedExecutionAuthorization, authorization),
        (EmbeddedBoundedExecutionResult, result),
        (EmbeddedDecisionReceipt, receipt),
    ):
        payload = artifact.model_dump(mode="python")
        payload["execution_effect"] = "mutation"
        payload["mutation_id"] = artifact.operation_id
        with pytest.raises(ValidationError, match="validation execution must be"):
            artifact_type.model_validate(payload)


def test_validation_mutation_is_rejected_before_authorization() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain="validation"
    )

    with pytest.raises(EmbeddedDecisionContractError, match="must be non_mutating"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            created_at=_START + timedelta(seconds=4),
        )


@pytest.mark.parametrize("retained_error_evidence", [False, True])
def test_completed_non_mutating_receipt_requires_output_or_available_evidence(
    retained_error_evidence: bool,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain="validation"
    )
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="non_mutating",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        mutation_state="not_applicable",
    )
    review = _review(result, decision, disposition="accept")
    receipt = build_decision_receipt(
        artifact_id="receipt-validation-proof",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )
    payload = receipt.model_dump(mode="python")
    payload["outputs"] = []
    payload["evidence"] = (
        [
            ProviderNeutralEvidenceRecord(
                evidence_id="optional-error",
                evidence_type="validation",
                status="error",
                required=False,
                summary="Optional proof failed.",
            )
        ]
        if retained_error_evidence
        else []
    )

    with pytest.raises(ValidationError, match="output binding or available evidence"):
        EmbeddedDecisionReceipt.model_validate(payload)


@pytest.mark.parametrize("failure", ["foreign_coordinator", "early_authorization"])
def test_receipt_revalidates_authorization_owner_and_human_chronology(
    failure: str,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    if failure == "foreign_coordinator":
        authorization = authorization.model_copy(
            update={
                "producer": _producer(
                    "outer_coordinator",
                    "foreign-coordinator",
                    "b",
                )
            }
        )
        expected = "another coordinator"
    else:
        authorization = authorization.model_copy(
            update={"created_at": human.created_at - timedelta(seconds=1)}
        )
        expected = "timestamp precedes"
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    review = _review(result, decision, disposition="accept")

    with pytest.raises(EmbeddedDecisionContractError, match=expected):
        build_decision_receipt(
            artifact_id=f"receipt-{failure}",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=authorization,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
            created_at=_START + timedelta(seconds=7),
        )


def test_resumed_receipt_requires_and_validates_exact_resume_artifacts() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
    )
    retry_review = _review(interrupted, decision, disposition="retry")
    second = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=retry_review,
        created_at=_START + timedelta(seconds=7),
    )
    result = _result(
        authorization=second,
        decision=decision,
        executor=executor,
        created_at=_START + timedelta(seconds=8),
    )
    review = _review(
        result,
        decision,
        disposition="accept",
        created_at=_START + timedelta(seconds=9),
    )

    with pytest.raises(EmbeddedDecisionContractError, match="complete persisted"):
        build_decision_receipt(
            artifact_id="receipt-resumed-missing-history",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=second,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
            created_at=_START + timedelta(seconds=10),
        )

    foreign_prior = first.model_copy(
        update={
            "producer": _producer(
                "outer_coordinator",
                "foreign-coordinator",
                "b",
            )
        }
    )
    with pytest.raises(EmbeddedDecisionContractError, match="persisted authorization"):
        build_decision_receipt(
            artifact_id="receipt-resumed-wrong-authorization",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=second,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(
                authorizations=[foreign_prior],
                results=[interrupted],
                reviews=[retry_review],
            ),
            human_decision=human,
            created_at=_START + timedelta(seconds=10),
        )

    receipt = build_decision_receipt(
        artifact_id="receipt-resumed",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=second,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(
            authorizations=[first],
            results=[interrupted],
            reviews=[retry_review],
        ),
        human_decision=human,
        created_at=_START + timedelta(seconds=10),
    )
    assert receipt.receipt_status == "completed"
    assert receipt.execution_status == "succeeded"


@pytest.mark.parametrize(
    ("execution_status", "review_disposition"),
    [
        ("interrupted", "retry"),
        ("failed", "revise"),
        ("failed", "stop"),
        ("cancelled", "cancelled"),
    ],
)
def test_receipt_model_rejects_fabricated_completed_status(
    execution_status: ExecutionStatus,
    review_disposition: CoordinatorReviewDisposition,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    review = _review(result, decision, disposition="accept")
    receipt = build_decision_receipt(
        artifact_id="receipt-valid-before-forgery",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )
    payload = receipt.model_dump(mode="python")
    payload["execution_status"] = execution_status
    payload["review_disposition"] = review_disposition

    with pytest.raises(ValidationError, match="receipt_status does not match"):
        EmbeddedDecisionReceipt.model_validate(payload)


@pytest.mark.parametrize("status", ["unavailable", "error", "unsupported"])
def test_non_mutating_success_rejects_unavailable_required_result_evidence(
    status: EvidenceStatus,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain="validation"
    )
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="non_mutating",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        mutation_state="not_applicable",
    )
    unavailable = ProviderNeutralEvidenceRecord(
        evidence_id="required-validation",
        evidence_type="validation",
        status=status,
        required=True,
        summary="Required validator did not produce available evidence.",
    )
    payload = result.model_dump(mode="python")
    payload["evidence"] = [unavailable]

    with pytest.raises(ValidationError, match="unavailable required evidence"):
        EmbeddedBoundedExecutionResult.model_validate(payload)

    result = result.model_copy(update={"evidence": (unavailable,)})
    review = _review(result, decision, disposition="accept")
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_coordinator_review(review, result)


@pytest.mark.parametrize("status", ["unavailable", "error", "unsupported"])
def test_mutating_success_rejects_unavailable_required_evidence_at_all_boundaries(
    status: EvidenceStatus,
) -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    unavailable = ProviderNeutralEvidenceRecord(
        evidence_id="required-mutation-readback",
        evidence_type="validation",
        status=status,
        required=True,
        summary="Required mutation readback is not available.",
    )
    result_payload = result.model_dump(mode="python")
    result_payload["evidence"] = [unavailable]

    with pytest.raises(ValidationError, match="unavailable required evidence"):
        EmbeddedBoundedExecutionResult.model_validate(result_payload)

    tampered_result = result.model_copy(update={"evidence": [unavailable]})
    tampered_review = _review(tampered_result, decision, disposition="accept")
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_coordinator_review(tampered_review, tampered_result)
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        build_decision_receipt(
            artifact_id=f"receipt-mutation-{status}-build",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=authorization,
            result=tampered_result,
            review=tampered_review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
            created_at=_START + timedelta(seconds=7),
        )

    review = _review(result, decision, disposition="accept")
    receipt = build_decision_receipt(
        artifact_id=f"receipt-mutation-{status}",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )
    receipt_payload = receipt.model_dump(mode="python")
    receipt_payload["evidence"] = [unavailable]

    with pytest.raises(ValidationError, match="unavailable required evidence"):
        EmbeddedDecisionReceipt.model_validate(receipt_payload)


def test_successful_mutation_requires_output_at_deserialization_boundaries() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    readback = _available_execution_evidence()
    outputless_result = result.model_dump(mode="python")
    outputless_result["outputs"] = []
    outputless_result["evidence"] = [readback]

    with pytest.raises(ValidationError, match="at least one exact output binding"):
        EmbeddedBoundedExecutionResult.model_validate(outputless_result)

    review = _review(result, decision, disposition="accept")
    receipt = build_decision_receipt(
        artifact_id="receipt-with-output-before-forgery",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )
    outputless_receipt = receipt.model_dump(mode="python")
    outputless_receipt["outputs"] = []
    outputless_receipt["evidence"] = [readback]

    with pytest.raises(ValidationError, match="at least one exact output binding"):
        EmbeddedDecisionReceipt.model_validate(outputless_receipt)

    forged_receipt = receipt.model_copy(update={"outputs": [], "evidence": [readback]})
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_decision_receipt(
            forged_receipt,
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=authorization,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
        )


def test_forged_outputless_mutation_cannot_be_accepted_or_receipted() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    result = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
    )
    forged = result.model_copy(
        update={"outputs": [], "evidence": [_available_execution_evidence()]}
    )
    review = _review(forged, decision, disposition="accept")

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        validate_coordinator_review(review, forged)
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        build_decision_receipt(
            artifact_id="receipt-outputless-mutation",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=authorization,
            result=forged,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
            created_at=_START + timedelta(seconds=7),
        )


def test_complete_lineage_rejects_outputless_successful_mutation_ancestor() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
        created_at=_START + timedelta(seconds=5),
    )
    retry_review = _review(
        interrupted,
        decision,
        disposition="retry",
        created_at=_START + timedelta(seconds=6),
    )
    second = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=retry_review,
        created_at=_START + timedelta(seconds=7),
    )
    result = _result(
        authorization=second,
        decision=decision,
        executor=executor,
        created_at=_START + timedelta(seconds=8),
    )
    review = _review(
        result,
        decision,
        disposition="accept",
        created_at=_START + timedelta(seconds=9),
    )
    forged_ancestor = interrupted.model_copy(
        update={
            "status": "succeeded",
            "mutation_state": "verified",
            "outputs": [],
            "evidence": [_available_execution_evidence()],
            "error": None,
        }
    )
    forged_retry_review = _review(
        forged_ancestor,
        decision,
        disposition="retry",
        created_at=_START + timedelta(seconds=6),
    )

    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        build_decision_receipt(
            artifact_id="receipt-outputless-ancestor",
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=second,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage.model_construct(
                authorizations=[first],
                results=[forged_ancestor],
                reviews=[forged_retry_review],
            ),
            human_decision=human,
            created_at=_START + timedelta(seconds=10),
        )


def test_attempt_three_receipt_requires_complete_exact_ancestor_lineage() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    first = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    first_result = _result(
        authorization=first,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
        created_at=_START + timedelta(seconds=5),
    )
    first_review = _review(
        first_result,
        decision,
        disposition="retry",
        created_at=_START + timedelta(seconds=6),
    )
    second = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=first_result,
        prior_review=first_review,
        created_at=_START + timedelta(seconds=7),
    )
    second_result = _result(
        authorization=second,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_started",
        created_at=_START + timedelta(seconds=8),
    )
    second_review = _review(
        second_result,
        decision,
        disposition="retry",
        created_at=_START + timedelta(seconds=9),
    )
    third = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[first, second],
        historical_results=[first_result],
        historical_reviews=[first_review],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=second_result,
        prior_review=second_review,
        created_at=_START + timedelta(seconds=10),
    )
    third_result = _result(
        authorization=third,
        decision=decision,
        executor=executor,
        created_at=_START + timedelta(seconds=11),
    )
    third_review = _review(
        third_result,
        decision,
        disposition="accept",
        created_at=_START + timedelta(seconds=12),
    )
    lineage = PersistedExecutionLineage(
        authorizations=[first, second],
        results=[first_result, second_result],
        reviews=[first_review, second_review],
    )

    def build_attempt_three_receipt(
        artifact_id: str,
        prior_lineage: PersistedExecutionLineage,
    ) -> EmbeddedDecisionReceipt:
        return build_decision_receipt(
            artifact_id=artifact_id,
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=third,
            result=third_result,
            review=third_review,
            prior_lineage=prior_lineage,
            human_decision=human,
            created_at=_START + timedelta(seconds=13),
        )

    with pytest.raises(EmbeddedDecisionContractError, match="complete persisted"):
        build_attempt_three_receipt(
            "receipt-attempt-three-missing-lineage",
            PersistedExecutionLineage(),
        )

    missing_first = PersistedExecutionLineage.model_construct(
        authorizations=[second],
        results=[second_result],
        reviews=[second_review],
    )
    with pytest.raises(EmbeddedDecisionContractError, match="modified after"):
        build_attempt_three_receipt(
            "receipt-attempt-three-missing-ancestor",
            missing_first,
        )

    fabricated_first = first.model_copy(update={"artifact_id": "authorization-fake"})
    fabricated_lineage = PersistedExecutionLineage(
        authorizations=[fabricated_first, second],
        results=[first_result, second_result],
        reviews=[first_review, second_review],
    )
    with pytest.raises(EmbeddedDecisionContractError, match="parent differs"):
        build_attempt_three_receipt(
            "receipt-attempt-three-fabricated-ancestor",
            fabricated_lineage,
        )

    assert second.resume_of is not None
    mismatched_second = second.model_copy(
        update={"resume_of": second.resume_of.model_copy(update={"sha256": "f" * 64})}
    )
    mismatched_lineage = PersistedExecutionLineage(
        authorizations=[first, mismatched_second],
        results=[first_result, second_result],
        reviews=[first_review, second_review],
    )
    with pytest.raises(EmbeddedDecisionContractError, match="prior-attempt result"):
        build_attempt_three_receipt(
            "receipt-attempt-three-mismatched-ancestor",
            mismatched_lineage,
        )

    receipt = build_attempt_three_receipt(
        "receipt-attempt-three",
        lineage,
    )
    assert receipt.attempt == 3
    assert len(receipt.prior_execution_authorizations) == 2

    incomplete_payload = receipt.model_dump(mode="python")
    incomplete_payload["prior_execution_authorizations"] = incomplete_payload[
        "prior_execution_authorizations"
    ][1:]
    with pytest.raises(ValidationError, match="complete prior execution lineage"):
        EmbeddedDecisionReceipt.model_validate(incomplete_payload)

    forged_refs = list(receipt.prior_execution_authorizations)
    forged_refs[0] = forged_refs[0].model_copy(update={"sha256": "e" * 64})
    deserialized_forgery = receipt.model_copy(
        update={"prior_execution_authorizations": forged_refs}
    )
    with pytest.raises(EmbeddedDecisionContractError, match="complete persisted"):
        validate_decision_receipt(
            deserialized_forgery,
            decision=decision,
            evidence=[evidence],
            proposals=[proposal],
            authorization=third,
            result=third_result,
            review=third_review,
            prior_lineage=lineage,
            human_decision=human,
        )


def test_interrupted_non_mutating_validation_can_retry_without_mutation_id() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain(
        domain="validation"
    )
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="non_mutating",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    interrupted = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="interrupted",
        mutation_state="not_applicable",
    )
    review = _review(interrupted, decision, disposition="retry")
    validate_coordinator_review(review, interrupted)

    resumed = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[authorization],
        proposals=[proposal],
        executor=executor,
        execution_effect="non_mutating",
        human_decision=human,
        prior_result=interrupted,
        prior_review=review,
        created_at=_START + timedelta(seconds=7),
    )

    assert resumed.attempt == 2
    assert resumed.operation_id == authorization.operation_id
    assert resumed.mutation_id is None


def test_cancellation_is_receipted_and_cannot_silently_resume() -> None:
    identity, evidence, proposal, decision, human, executor = _authorization_chain()
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=_START + timedelta(seconds=4),
    )
    cancelled = _result(
        authorization=authorization,
        decision=decision,
        executor=executor,
        status="cancelled",
        mutation_state="not_started",
    )
    review = _review(cancelled, decision, disposition="cancelled")
    receipt = build_decision_receipt(
        artifact_id="receipt-cancelled",
        decision=decision,
        evidence=[evidence],
        proposals=[proposal],
        authorization=authorization,
        result=cancelled,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        human_decision=human,
        created_at=_START + timedelta(seconds=7),
    )
    assert receipt.receipt_status == "cancelled"

    with pytest.raises(EmbeddedDecisionContractError, match="not retry"):
        authorize_bounded_execution(
            decision,
            expected_identity=identity,
            evidence=[evidence],
            existing_authorizations=[authorization],
            proposals=[proposal],
            executor=executor,
            execution_effect="mutation",
            human_decision=human,
            prior_result=cancelled,
            prior_review=review,
            created_at=_START + timedelta(seconds=8),
        )


def test_shared_decision_artifacts_reject_standalone_and_classic_contexts() -> None:
    for owner in ("domain_child_agent", "compatibility_pipeline"):
        standalone = DomainExecutionContext(
            schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
            domain="validation",
            mode="standalone",
            reasoning_loop_owner=owner,
        )
        with pytest.raises(ValidationError, match="embedded execution context"):
            EmbeddedDecisionIdentity(
                execution_context=standalone,
                source=_binding("/run/input.usdz", "3"),
                coordinator_plan=_plan_parent(),
                digests=NamedDecisionDigests(
                    configuration={"validation_config": "4" * 64},
                    prompt={"asset_prompt": "5" * 64},
                    capabilities={"validation": "6" * 64},
                    implementations={"validation_adapter": "7" * 64},
                ),
            )
