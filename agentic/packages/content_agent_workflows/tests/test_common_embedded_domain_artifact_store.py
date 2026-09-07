# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash, replay, and lineage tests for the embedded decision artifact store."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal

import pytest

import content_agent_workflows.common.embedded_domain_artifact_store as artifact_store_module
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
    DomainExecutionContext,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
)
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION,
    EmbeddedDecisionArtifactJournal,
    EmbeddedDecisionArtifactStore,
    EmbeddedDecisionArtifactStoreError,
    EmbeddedDecisionAuthorizationReplayError,
    EmbeddedDecisionMutationLeaseReconciliationError,
)
from content_agent_workflows.common.embedded_domain_decision import (
    AcceptedSemanticDecision,
    BoundedExecutionAuthorization,
    ContractArtifactReference,
    DomainProposalPayload,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
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
    outer_stage_attempt_seal_key,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _invoke_authorization_in_process(
    run_root: str,
    authorization_json: str,
    outcome_path: str,
) -> None:
    authorization = BoundedExecutionAuthorization.model_validate_json(
        authorization_json
    )
    store = EmbeddedDecisionArtifactStore(run_root)
    try:
        outcome = store.invoke_after_authorization_commit(
            authorization,
            lambda _claim: "executed",
        )
    except EmbeddedDecisionMutationLeaseReconciliationError as exc:
        outcome = exc.outcome.disposition
    Path(outcome_path).write_text(outcome, encoding="utf-8")


def _binding(path: str, fill: str) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(path=path, sha256=fill * 64, size_bytes=1)


def _producer(
    producer_id: str,
    role: ProducerRole,
    fill: str,
) -> ProducerIdentity:
    return ProducerIdentity(
        producer_id=producer_id,
        role=role,
        implementation=f"test.{producer_id}.v1",
        implementation_digest=None if role == "human_reviewer" else fill * 64,
    )


def _plan_reference() -> ContractArtifactReference:
    return ContractArtifactReference(
        artifact_kind="coordinator_plan",
        artifact_id="plan-001",
        schema_version="content-agent-workflows.asset-coordinator-plan.v1",
        sha256="2" * 64,
    )


def _identity() -> EmbeddedDecisionIdentity:
    source = _binding("/run/input.usdz", "3")
    context = DomainExecutionContext(
        schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="run-001",
            outer_request=_binding("/run/request.json", "1"),
            stage="articulation",
            stage_attempt=1,
            coordinator_plan=_binding("/run/coordinator/plan-001.json", "2"),
            input_asset=source,
            domain_run_root="/run/stages/03-articulation/domain-run",
        ),
    )
    return EmbeddedDecisionIdentity(
        execution_context=context,
        source=source,
        coordinator_plan=_plan_reference(),
        digests=NamedDecisionDigests(
            configuration={"joint_config": "4" * 64},
            prompt={"asset_prompt": "5" * 64},
            capabilities={"joint_inspection": "6" * 64},
            implementations={
                "evidence_provider": "7" * 64,
                "proposal_provider": "8" * 64,
                "asset_coordinator": "9" * 64,
                "joint_executor": "a" * 64,
            },
        ),
    )


def _artifact_chain() -> tuple[
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedCoordinatorDecision,
    EmbeddedHumanDecision,
    BoundedExecutionAuthorization,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionReceipt,
]:
    identity = _identity()
    evidence = EmbeddedDomainEvidence(
        artifact_id="evidence-001",
        identity=identity,
        producer=_producer("inspection-provider", "evidence_provider", "7"),
        parent_artifact=_plan_reference(),
        created_at=_START,
        records=[
            ProviderNeutralEvidenceRecord(
                evidence_id="topology",
                evidence_type="inspection",
                status="available",
                summary="Topology inspected.",
                facts={"components": 2},
            )
        ],
    )
    proposal_payload = DomainProposalPayload(
        schema_version="articulation.proposal.v1",
        values={"candidate": "hinge"},
    )
    proposal = EmbeddedDomainProposal(
        artifact_id="proposal-001",
        identity=identity,
        producer=_producer("joint-provider", "proposal_provider", "8"),
        parent_artifact=artifact_reference(evidence),
        created_at=_START + timedelta(seconds=1),
        evidence_artifacts=[artifact_reference(evidence)],
        proposal=proposal_payload,
        proposal_digest=canonical_json_digest(proposal_payload),
    )
    accepted = AcceptedSemanticDecision(
        schema_version="articulation.decision.v1",
        values={"joint": "hinge"},
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id="decision-001",
        identity=identity,
        producer=_producer("asset-coordinator", "outer_coordinator", "9"),
        parent_artifact=artifact_reference(proposal),
        created_at=_START + timedelta(seconds=2),
        disposition="accept",
        evidence_artifacts=[artifact_reference(evidence)],
        proposal_artifacts=[artifact_reference(proposal)],
        accepted_decision=accepted,
        accepted_decision_digest=accepted_semantic_decision_digest(
            identity,
            accepted,
        ),
        human_decision_required=True,
        rationale="Accepted exact joint semantics.",
    )
    decision_reference = artifact_reference(decision)
    assert decision.accepted_decision_digest is not None
    human = EmbeddedHumanDecision(
        artifact_id="human-001",
        identity=identity,
        producer=_producer("reviewer", "human_reviewer", "0"),
        parent_artifact=decision_reference,
        created_at=_START + timedelta(seconds=3),
        coordinator_decision=decision_reference,
        reviewed_decision_digest=decision.accepted_decision_digest,
        disposition="accept",
        rationale="Human accepted exact semantics.",
    )
    executor = _producer("joint-executor", "executor", "a")
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
    result = EmbeddedBoundedExecutionResult(
        artifact_id="result-001",
        identity=identity,
        producer=executor,
        parent_artifact=artifact_reference(authorization),
        created_at=_START + timedelta(seconds=5),
        accepted_decision=decision_reference,
        accepted_decision_digest=authorization.accepted_decision_digest,
        execution_effect="mutation",
        operation_id=authorization.operation_id,
        mutation_id=authorization.mutation_id,
        attempt=1,
        status="succeeded",
        mutation_state="verified",
        outputs=[_binding("/run/output.usdz", "b")],
    )
    review = EmbeddedCoordinatorReview(
        artifact_id="review-001",
        identity=identity,
        producer=decision.producer,
        parent_artifact=artifact_reference(result),
        created_at=_START + timedelta(seconds=6),
        execution_result=artifact_reference(result),
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=authorization.accepted_decision_digest,
        disposition="accept",
        outputs=result.outputs,
        findings=["Mutation output verified."],
    )
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
        created_at=_START + timedelta(seconds=7),
    )
    return evidence, proposal, decision, human, authorization, result, review, receipt


def _append_authority_chain(
    store: EmbeddedDecisionArtifactStore,
) -> tuple[
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedCoordinatorDecision,
    EmbeddedHumanDecision,
    BoundedExecutionAuthorization,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionReceipt,
]:
    artifacts = _artifact_chain()
    for initial_artifact in artifacts[:4]:
        store.append(initial_artifact)
    return artifacts


def _drifted_authority_chain(
    drift: Literal[
        "capability_digest",
        "source_path",
        "source_digest",
        "outer_request_path",
        "domain_run_root",
        "stage_attempt",
    ] = "capability_digest",
) -> tuple[
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedCoordinatorDecision,
    EmbeddedHumanDecision,
    BoundedExecutionAuthorization,
]:
    evidence, proposal, decision, human, authorization, *_rest = _artifact_chain()
    identity = decision.identity
    stage = identity.execution_context.embedded_stage
    assert stage is not None
    identity_updates: dict[str, object] = {}
    if drift == "capability_digest":
        digest_payload = identity.digests.model_dump(mode="json")
        digest_payload["capabilities"]["joint_inspection"] = "e" * 64
        identity_updates["digests"] = NamedDecisionDigests.model_validate(
            digest_payload
        )
    elif drift in {"source_path", "source_digest"}:
        source = identity.source.model_copy(
            update={
                "path": "/run/alias-input.usdz"
                if drift == "source_path"
                else identity.source.path,
                "sha256": "e" * 64
                if drift == "source_digest"
                else identity.source.sha256,
            }
        )
        drifted_stage = stage.model_copy(update={"input_asset": source})
        identity_updates.update(
            {
                "source": source,
                "execution_context": identity.execution_context.model_copy(
                    update={"embedded_stage": drifted_stage}
                ),
            }
        )
    else:
        if drift == "outer_request_path":
            drifted_stage = stage.model_copy(
                update={
                    "outer_request": stage.outer_request.model_copy(
                        update={"path": "/run/alias-request.json"}
                    )
                }
            )
        elif drift == "domain_run_root":
            drifted_stage = stage.model_copy(
                update={"domain_run_root": "/run/stages/alias-domain-run"}
            )
        else:
            drifted_stage = stage.model_copy(update={"stage_attempt": 2})
        identity_updates["execution_context"] = identity.execution_context.model_copy(
            update={"embedded_stage": drifted_stage}
        )
    drifted_identity = EmbeddedDecisionIdentity.model_validate(
        identity.model_copy(update=identity_updates).model_dump(mode="python")
    )
    drifted_evidence = evidence.model_copy(
        update={
            "artifact_id": f"evidence-drift-{drift}",
            "identity": drifted_identity,
        }
    )
    drifted_proposal = proposal.model_copy(
        update={
            "artifact_id": f"proposal-drift-{drift}",
            "identity": drifted_identity,
            "parent_artifact": artifact_reference(drifted_evidence),
            "evidence_artifacts": (artifact_reference(drifted_evidence),),
        }
    )
    assert decision.accepted_decision is not None
    drifted_decision = decision.model_copy(
        update={
            "artifact_id": f"decision-drift-{drift}",
            "identity": drifted_identity,
            "parent_artifact": artifact_reference(drifted_proposal),
            "evidence_artifacts": (artifact_reference(drifted_evidence),),
            "proposal_artifacts": (artifact_reference(drifted_proposal),),
            "accepted_decision_digest": accepted_semantic_decision_digest(
                drifted_identity,
                decision.accepted_decision,
            ),
        }
    )
    drifted_decision_reference = artifact_reference(drifted_decision)
    assert drifted_decision.accepted_decision_digest is not None
    drifted_human = human.model_copy(
        update={
            "artifact_id": f"human-drift-{drift}",
            "identity": drifted_identity,
            "parent_artifact": drifted_decision_reference,
            "coordinator_decision": drifted_decision_reference,
            "reviewed_decision_digest": drifted_decision.accepted_decision_digest,
        }
    )
    drifted_authorization = authorize_bounded_execution(
        drifted_decision,
        expected_identity=drifted_identity,
        evidence=(drifted_evidence,),
        existing_authorizations=(),
        proposals=(drifted_proposal,),
        executor=authorization.executor,
        execution_effect="mutation",
        human_decision=drifted_human,
    )
    return (
        drifted_evidence,
        drifted_proposal,
        drifted_decision,
        drifted_human,
        drifted_authorization,
    )


def test_store_round_trip_replay_and_complete_ordered_lineage(tmp_path: Path) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    artifacts = _artifact_chain()

    commits = [store.append(artifact) for artifact in artifacts]
    journal = store.journal()

    assert journal.schema_version == EMBEDDED_DECISION_ARTIFACT_JOURNAL_SCHEMA_VERSION
    assert [entry.sequence for entry in journal.entries] == list(range(1, 9))
    assert all(
        entry.previous_entry_sha256 == journal.entries[index - 1].entry_sha256
        for index, entry in enumerate(journal.entries[1:], start=1)
    )
    for artifact, commit in zip(artifacts, commits, strict=True):
        assert store.load_typed(commit.reference, type(artifact)) == artifact

    replay = store.append(artifacts[-1])
    assert replay.reference == commits[-1].reference
    assert replay.sequence == commits[-1].sequence
    assert replay.entry_sha256 == commits[-1].entry_sha256
    assert commits[-1].newly_committed is True
    assert replay.newly_committed is False
    assert len(store.journal().entries) == 8


def test_store_round_trips_complete_retry_lineage(tmp_path: Path) -> None:
    evidence, proposal, decision, human, first, *_rest = _artifact_chain()
    executor = first.executor
    first_result = EmbeddedBoundedExecutionResult(
        artifact_id="result-001-interrupted",
        identity=decision.identity,
        producer=executor,
        parent_artifact=artifact_reference(first),
        created_at=_START + timedelta(seconds=5),
        accepted_decision=artifact_reference(decision),
        accepted_decision_digest=first.accepted_decision_digest,
        execution_effect="mutation",
        operation_id=first.operation_id,
        mutation_id=first.mutation_id,
        attempt=1,
        status="interrupted",
        mutation_state="not_started",
        error="Interrupted before mutation.",
    )
    first_review = EmbeddedCoordinatorReview(
        artifact_id="review-001-retry",
        identity=decision.identity,
        producer=decision.producer,
        parent_artifact=artifact_reference(first_result),
        created_at=_START + timedelta(seconds=6),
        execution_result=artifact_reference(first_result),
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=first.accepted_decision_digest,
        disposition="retry",
        findings=("Safe to retry before mutation.",),
    )
    second = authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=(evidence,),
        existing_authorizations=(first,),
        proposals=(proposal,),
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=first_result,
        prior_review=first_review,
    )
    reconstructed_second = authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=(evidence,),
        existing_authorizations=(first,),
        proposals=(proposal,),
        executor=executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=first_result,
        prior_review=first_review,
    )

    assert second == reconstructed_second
    assert second.created_at == first_review.created_at
    second_result = EmbeddedBoundedExecutionResult(
        artifact_id="result-002",
        identity=decision.identity,
        producer=executor,
        parent_artifact=artifact_reference(second),
        created_at=_START + timedelta(seconds=8),
        accepted_decision=artifact_reference(decision),
        accepted_decision_digest=second.accepted_decision_digest,
        execution_effect="mutation",
        operation_id=second.operation_id,
        mutation_id=second.mutation_id,
        attempt=2,
        resume_of=artifact_reference(first_result),
        status="succeeded",
        mutation_state="verified",
        outputs=(_binding("/run/output.usdz", "b"),),
    )
    second_review = EmbeddedCoordinatorReview(
        artifact_id="review-002",
        identity=decision.identity,
        producer=decision.producer,
        parent_artifact=artifact_reference(second_result),
        created_at=_START + timedelta(seconds=9),
        execution_result=artifact_reference(second_result),
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=second.accepted_decision_digest,
        disposition="accept",
        outputs=second_result.outputs,
        findings=("Retry output verified.",),
    )
    receipt = build_decision_receipt(
        artifact_id="receipt-002",
        decision=decision,
        evidence=(evidence,),
        proposals=(proposal,),
        authorization=second,
        result=second_result,
        review=second_review,
        prior_lineage=PersistedExecutionLineage(
            authorizations=(first,),
            results=(first_result,),
            reviews=(first_review,),
        ),
        human_decision=human,
        created_at=_START + timedelta(seconds=10),
    )
    artifacts = (
        evidence,
        proposal,
        decision,
        human,
        first,
        first_result,
        first_review,
        second,
        second_result,
        second_review,
        receipt,
    )
    store = EmbeddedDecisionArtifactStore(tmp_path)

    for artifact in artifacts[:4]:
        store.append(artifact)
    assert (
        store.invoke_after_authorization_commit(
            first,
            lambda _claim: first_result,
        )
        == first_result
    )
    for retry_artifact in (first_result, first_review):
        store.append(retry_artifact)
    assert (
        store.invoke_after_authorization_commit(
            second,
            lambda _claim: second_result,
        )
        == second_result
    )
    for terminal_artifact in (second_result, second_review, receipt):
        store.append(terminal_artifact)

    assert len(store.journal().entries) == len(artifacts)
    assert store.load_typed(artifact_reference(receipt), EmbeddedDecisionReceipt) == (
        receipt
    )


def test_store_rejects_conflicts_traversal_and_missing_parent(tmp_path: Path) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    evidence, proposal, *_rest = _artifact_chain()
    store.append(evidence)
    changed_record = ProviderNeutralEvidenceRecord(
        evidence_id="topology",
        evidence_type="inspection",
        status="available",
        summary="Conflicting topology claim.",
        facts={"components": 3},
    )
    conflicting = evidence.model_copy(update={"records": (changed_record,)})

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="different canonical"):
        store.append(conflicting)
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="path traversal"):
        store.append(evidence.model_copy(update={"artifact_id": "../escape"}))
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="timestamp precedes"):
        store.append(
            proposal.model_copy(
                update={"created_at": evidence.created_at - timedelta(seconds=1)}
            )
        )

    empty_store = EmbeddedDecisionArtifactStore(tmp_path / "missing-parent")
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError, match="dependency is absent"
    ):
        empty_store.append(proposal)


def test_store_recovers_exact_orphan_after_interrupted_journal_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    evidence = _artifact_chain()[0]

    def interrupt_journal_write(_store_fd: int, _journal: object) -> None:
        raise OSError("simulated crash before journal commit")

    monkeypatch.setattr(store, "_write_journal_locked", interrupt_journal_write)
    with pytest.raises(OSError, match="simulated crash"):
        store.append(evidence)

    monkeypatch.undo()
    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    commit = recovered.append(evidence)

    assert commit.sequence == 1
    assert recovered.load_typed(commit.reference, EmbeddedDomainEvidence) == evidence
    assert len(list((recovered.store_root / "artifacts").rglob("*.json"))) == 1


def test_store_rejects_conflict_with_crash_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    evidence = _artifact_chain()[0]

    def interrupt_journal_write(_store_fd: int, _journal: object) -> None:
        raise OSError("simulated crash before journal commit")

    monkeypatch.setattr(store, "_write_journal_locked", interrupt_journal_write)
    with pytest.raises(OSError, match="simulated crash"):
        store.append(evidence)

    changed_record = ProviderNeutralEvidenceRecord(
        evidence_id="topology",
        evidence_type="inspection",
        status="available",
        summary="Conflicting crash-recovery claim.",
        facts={"components": 3},
    )
    conflicting = evidence.model_copy(update={"records": (changed_record,)})
    monkeypatch.undo()

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="different canonical"):
        EmbeddedDecisionArtifactStore(tmp_path).append(conflicting)


def test_store_fails_closed_on_tampered_bytes_and_symlink_roots(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path / "run")
    evidence = _artifact_chain()[0]
    commit = store.append(evidence)
    artifact_path = store.store_root / commit.relative_path
    artifact_path.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="malformed"):
        store.journal()

    journal_store = EmbeddedDecisionArtifactStore(tmp_path / "journal-run")
    journal_store.append(evidence)
    journal_path = journal_store.store_root / "journal.json"
    journal_payload = json.loads(journal_path.read_text(encoding="utf-8"))
    journal_payload["entries"][0]["reference"]["sha256"] = "f" * 64
    journal_path.write_text(
        json.dumps(journal_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="stale"):
        journal_store.journal()

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-root"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="symlinks"):
        EmbeddedDecisionArtifactStore(symlink_root)


def test_store_lock_rejects_hard_link_without_truncating_external_file(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path / "run")
    external = tmp_path / "external-state.txt"
    external.write_text("must remain intact", encoding="utf-8")
    os.link(external, store.store_root / ".journal.lock")

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="single-link"):
        store.append(_artifact_chain()[0])

    assert external.read_text(encoding="utf-8") == "must remain intact"


def test_store_rejects_root_rename_and_replacement_before_lock(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    displaced = tmp_path / "displaced-store"
    store.store_root.rename(displaced)
    store.store_root.mkdir()

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="renamed or replaced"):
        store.append(_artifact_chain()[0])

    assert not (store.store_root / "journal.json").exists()
    assert not (displaced / "journal.json").exists()


def test_store_rename_after_lock_publishes_only_on_pinned_inode_and_fails_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    evidence = _artifact_chain()[0]
    displaced = tmp_path / "displaced-store"
    original_write = store._write_journal_locked

    def rename_before_journal_commit(
        store_fd: int,
        journal: EmbeddedDecisionArtifactJournal,
    ) -> None:
        store.store_root.rename(displaced)
        store.store_root.mkdir()
        original_write(store_fd, journal)

    monkeypatch.setattr(store, "_write_journal_locked", rename_before_journal_commit)
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="renamed or replaced"):
        store.append(evidence)

    assert (displaced / "journal.json").is_file()
    assert len(list((displaced / "artifacts").rglob("*.json"))) == 1
    assert not (store.store_root / "journal.json").exists()
    assert not (store.store_root / "artifacts").exists()


def test_store_replacement_cannot_split_lock_or_duplicate_executor_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    displaced = tmp_path / "displaced-store"
    child_outcome = tmp_path / "child-outcome.txt"
    original_write = store._write_journal_locked
    child: BaseProcess | None = None

    def split_root_before_authorization_journal(
        store_fd: int,
        journal: EmbeddedDecisionArtifactJournal,
    ) -> None:
        nonlocal child
        store.store_root.rename(displaced)
        shutil.copytree(displaced, store.store_root)
        context = multiprocessing.get_context("fork")
        started_child = context.Process(
            target=_invoke_authorization_in_process,
            args=(
                str(run_root),
                authorization.model_dump_json(),
                str(child_outcome),
            ),
        )
        child = started_child
        started_child.start()
        original_write(store_fd, journal)

    monkeypatch.setattr(
        store,
        "_write_journal_locked",
        split_root_before_authorization_journal,
    )
    first_invocations: list[str] = []
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="renamed or replaced"):
        store.invoke_after_authorization_commit(
            authorization,
            lambda claim: first_invocations.append(claim.operation_id),
        )

    assert child is not None
    child.join(timeout=30)
    assert child.exitcode == 0
    assert child_outcome.read_text(encoding="utf-8") == "executed"
    assert first_invocations == []
    replacement = EmbeddedDecisionArtifactStore(run_root)
    assert (
        replacement.load_typed(
            artifact_reference(authorization),
            BoundedExecutionAuthorization,
        )
        == authorization
    )
    assert (
        sum(
            entry.reference == artifact_reference(authorization)
            for entry in replacement.journal().entries
        )
        == 1
    )


def test_copied_pre_authorization_store_cannot_duplicate_active_outer_lease(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    replacement = tmp_path / "replacement-store"
    shutil.copytree(store.store_root, replacement)
    displaced = tmp_path / "displaced-store"
    child_outcome = tmp_path / "child-outcome.txt"
    child: BaseProcess | None = None
    invocations: list[str] = []

    def original_executor(claim: BoundedExecutionAuthorization) -> str:
        nonlocal child
        store.store_root.rename(displaced)
        replacement.rename(store.store_root)
        context = multiprocessing.get_context("fork")
        started_child = context.Process(
            target=_invoke_authorization_in_process,
            args=(
                str(run_root),
                authorization.model_dump_json(),
                str(child_outcome),
            ),
        )
        child = started_child
        started_child.start()
        assert started_child.is_alive()
        invocations.append(claim.operation_id)
        return "executed"

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="renamed or replaced"):
        store.invoke_after_authorization_commit(
            authorization,
            original_executor,
        )

    assert child is not None
    child.join(timeout=30)
    assert child.exitcode == 0
    assert child_outcome.read_text(encoding="utf-8") == "stale_store"
    assert invocations == [authorization.operation_id]
    replacement_store = EmbeddedDecisionArtifactStore(run_root)
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="copied or replacement store",
    ):
        replacement_store.journal()


@pytest.mark.parametrize("anchor_kind", ["lock", "record"])
@pytest.mark.parametrize("substitution", ["symlink", "hard_link"])
def test_outer_mutation_lease_rejects_hostile_link_without_invocation(
    tmp_path: Path,
    anchor_kind: str,
    substitution: str,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    lock_name, record_name = store._mutation_lease_names(authorization)
    target = run_root / (lock_name if anchor_kind == "lock" else record_name)
    external = tmp_path / f"external-{anchor_kind}-{substitution}.txt"
    external.write_text("do-not-truncate", encoding="utf-8")
    if substitution == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)
    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> str:
        nonlocal invoked
        invoked = True
        return "unsafe"

    with pytest.raises(EmbeddedDecisionArtifactStoreError):
        store.invoke_after_authorization_commit(authorization, executor)

    assert invoked is False
    assert external.read_text(encoding="utf-8") == "do-not-truncate"


@pytest.mark.parametrize("substitution", ["lock", "record", "run_root"])
def test_outer_mutation_lease_detects_anchor_replacement_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    lock_name, record_name = store._mutation_lease_names(authorization)
    original_publish = store._publish_mutation_lease_locked
    displaced = tmp_path / f"displaced-{substitution}"

    def replace_anchor(
        run_root_fd: int,
        name: str,
        lease: artifact_store_module.DurableEmbeddedMutationLease,
    ) -> tuple[int, int]:
        identity = original_publish(run_root_fd, name, lease)
        if substitution == "lock":
            (run_root / lock_name).rename(displaced)
            (run_root / lock_name).write_text("replacement", encoding="utf-8")
        elif substitution == "record":
            (run_root / record_name).rename(displaced)
            shutil.copy2(displaced, run_root / record_name)
        else:
            run_root.rename(displaced)
            run_root.mkdir()
        return identity

    monkeypatch.setattr(store, "_publish_mutation_lease_locked", replace_anchor)
    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> str:
        nonlocal invoked
        invoked = True
        return "unsafe"

    with pytest.raises(EmbeddedDecisionArtifactStoreError):
        store.invoke_after_authorization_commit(authorization, executor)
    assert invoked is False


def test_crash_before_result_keeps_outer_lease_reconciliation_only(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    invocations: list[str] = []

    def crashing_executor(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(claim.operation_id)
        raise RuntimeError("simulated crash before result")

    with pytest.raises(RuntimeError, match="before result"):
        store.invoke_after_authorization_commit(
            authorization,
            crashing_executor,
        )

    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    with pytest.raises(EmbeddedDecisionAuthorizationReplayError) as replay_error:
        recovered.invoke_after_authorization_commit(
            authorization,
            crashing_executor,
        )
    assert replay_error.value.outcome.authorization == artifact_reference(authorization)
    assert replay_error.value.outcome.operation_id == authorization.operation_id
    assert replay_error.value.outcome.durable_commit.newly_committed is False
    assert invocations == [authorization.operation_id]


def test_crash_after_authorization_before_outer_lease_never_invokes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    invoked = False

    def interrupt_before_lease(*_args: object, **_kwargs: object) -> tuple[int, int]:
        raise OSError("simulated crash after authorization before outer lease")

    monkeypatch.setattr(
        store,
        "_publish_mutation_lease_locked",
        interrupt_before_lease,
    )
    with pytest.raises(OSError, match="before outer lease"):
        store.invoke_after_authorization_commit(
            authorization,
            lambda _claim: "unsafe",
        )
    monkeypatch.undo()

    recovered = EmbeddedDecisionArtifactStore(tmp_path)

    def executor(_claim: BoundedExecutionAuthorization) -> str:
        nonlocal invoked
        invoked = True
        return "unsafe"

    with pytest.raises(EmbeddedDecisionAuthorizationReplayError):
        recovered.invoke_after_authorization_commit(authorization, executor)
    assert invoked is False


def test_in_place_pre_authorization_journal_restore_is_reconciliation_only(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    journal_path = store.store_root / "journal.json"
    pre_authorization_journal = journal_path.read_bytes()
    invocations: list[str] = []

    def executor(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(claim.operation_id)
        return "executed"

    assert (
        store.invoke_after_authorization_commit(authorization, executor) == "executed"
    )
    assert invocations == [authorization.operation_id]

    journal_path.write_bytes(pre_authorization_journal)
    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    with pytest.raises(
        EmbeddedDecisionMutationLeaseReconciliationError
    ) as reconciliation_error:
        recovered.invoke_after_authorization_commit(authorization, executor)

    outcome = reconciliation_error.value.outcome
    assert outcome.disposition == "reconciliation_required"
    assert outcome.authorization == artifact_reference(authorization)
    assert outcome.operation_id == authorization.operation_id
    assert outcome.mutation_id == authorization.mutation_id
    assert invocations == [authorization.operation_id]
    assert len(recovered.journal().entries) == 4


def test_same_operation_attempt_with_different_authorization_fails_closed(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    evidence, proposal, decision, human, authorization, *_rest = (
        _append_authority_chain(store)
    )
    invocations: list[ContractArtifactReference] = []

    def execute_once(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(artifact_reference(claim))
        return "executed"

    assert (
        store.invoke_after_authorization_commit(authorization, execute_once)
        == "executed"
    )
    reconstructed = authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=[evidence],
        existing_authorizations=[],
        proposals=[proposal],
        executor=authorization.executor,
        execution_effect="mutation",
        human_decision=human,
        created_at=authorization.created_at + timedelta(seconds=1),
    )
    assert reconstructed.operation_id == authorization.operation_id
    assert reconstructed.mutation_id == authorization.mutation_id
    assert reconstructed.attempt == authorization.attempt
    assert artifact_reference(reconstructed) != artifact_reference(authorization)

    def execute_unsafe(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(artifact_reference(claim))
        return "unsafe"

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="Artifact identity already exists with different canonical bytes",
    ):
        store.append(reconstructed)
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="different authorization for this operation attempt",
    ):
        store.invoke_after_authorization_commit(
            reconstructed,
            execute_unsafe,
        )

    assert invocations == [artifact_reference(authorization)]


def test_reviewed_attempt_two_remains_invocable_under_the_operation_lease(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    evidence, proposal, decision, human, first, *_rest = _append_authority_chain(store)
    assert (
        store.invoke_after_authorization_commit(first, lambda _claim: "interrupted")
        == "interrupted"
    )
    interrupted = EmbeddedBoundedExecutionResult(
        artifact_id="result-interrupted-001",
        identity=first.identity,
        producer=first.executor,
        parent_artifact=artifact_reference(first),
        created_at=first.created_at + timedelta(seconds=1),
        accepted_decision=first.accepted_decision,
        accepted_decision_digest=first.accepted_decision_digest,
        execution_effect="mutation",
        operation_id=first.operation_id,
        mutation_id=first.mutation_id,
        attempt=1,
        status="interrupted",
        mutation_state="not_started",
        error="Executor was interrupted before mutation.",
    )
    retry_review = EmbeddedCoordinatorReview(
        artifact_id="review-retry-001",
        identity=first.identity,
        producer=decision.producer,
        parent_artifact=artifact_reference(interrupted),
        created_at=interrupted.created_at + timedelta(seconds=1),
        execution_result=artifact_reference(interrupted),
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=first.accepted_decision_digest,
        disposition="retry",
        findings=["Retry before mutation."],
    )
    store.append_result(interrupted)
    store.append_review(retry_review)
    resumed = authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=[evidence],
        existing_authorizations=[first],
        proposals=[proposal],
        executor=first.executor,
        execution_effect="mutation",
        human_decision=human,
        prior_result=interrupted,
        prior_review=retry_review,
        created_at=retry_review.created_at + timedelta(seconds=1),
    )

    invocations: list[int] = []

    def execute_retry(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(claim.attempt)
        return "resumed"

    assert resumed.attempt == 2
    assert resumed.operation_id == first.operation_id
    assert resumed.mutation_id == first.mutation_id
    assert (
        store.invoke_after_authorization_commit(
            resumed,
            execute_retry,
        )
        == "resumed"
    )
    assert invocations == [2]


def test_outer_lease_rejects_conflicting_identity_from_replacement_store(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    store = EmbeddedDecisionArtifactStore(run_root)
    *_authority, authorization, _result, _review, _receipt = _append_authority_chain(
        store
    )
    assert (
        store.invoke_after_authorization_commit(
            authorization,
            lambda _claim: "executed",
        )
        == "executed"
    )
    displaced = tmp_path / "displaced-store"
    store.store_root.rename(displaced)
    drifted_store = EmbeddedDecisionArtifactStore(run_root)
    drifted_chain = _drifted_authority_chain("source_path")
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="stale full decision identity",
    ):
        drifted_store.append(drifted_chain[0])
    drifted_authorization = drifted_chain[4]
    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> str:
        nonlocal invoked
        invoked = True
        return "unsafe"

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="conflicts with stale identity or operation",
    ):
        drifted_store.invoke_after_authorization_commit(
            drifted_authorization,
            executor,
        )
    assert invoked is False


@pytest.mark.parametrize(
    ("substitution", "error_pattern"),
    [("symlink", "descriptor-confined"), ("hard_link", "single-link")],
)
def test_store_rejects_substituted_crash_orphan(
    tmp_path: Path,
    substitution: str,
    error_pattern: str,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path / "run")
    evidence = _artifact_chain()[0]
    reference = artifact_reference(evidence)
    kind_directory = store.store_root / "artifacts" / reference.artifact_kind
    kind_directory.mkdir(parents=True)
    external = tmp_path / "external-artifact.json"
    external.write_text(
        artifact_store_module._canonical_artifact_text(evidence),
        encoding="utf-8",
    )
    target = kind_directory / f"{reference.sha256}.json"
    if substitution == "symlink":
        target.symlink_to(external)
    else:
        os.link(external, target)

    with pytest.raises(EmbeddedDecisionArtifactStoreError, match=error_pattern):
        store.append(evidence)


def test_store_serializes_identical_concurrent_replay(tmp_path: Path) -> None:
    evidence = _artifact_chain()[0]

    def append_once(_index: int) -> int:
        store = EmbeddedDecisionArtifactStore(tmp_path)
        return store.append(evidence).sequence

    with ThreadPoolExecutor(max_workers=8) as executor:
        sequences = list(executor.map(append_once, range(16)))

    assert sequences == [1] * 16
    assert len(EmbeddedDecisionArtifactStore(tmp_path).journal().entries) == 1


def test_authorization_is_durable_before_executor_and_terminal_appends(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    (
        _evidence,
        _proposal,
        _decision,
        _human,
        authorization,
        result,
        review,
        receipt,
    ) = _append_authority_chain(store)
    observed: list[str] = []

    def executor(
        claim: BoundedExecutionAuthorization,
    ) -> EmbeddedBoundedExecutionResult:
        loaded = store.load_typed(
            artifact_reference(claim),
            BoundedExecutionAuthorization,
        )
        assert loaded == claim
        observed.append("executor_after_durable_authorization")
        return result

    executed = store.invoke_after_authorization_commit(authorization, executor)
    store.append_result(executed)
    store.append_review(review)
    store.append_receipt(receipt)

    assert observed == ["executor_after_durable_authorization"]
    assert [entry.reference.artifact_kind for entry in store.journal().entries] == [
        "evidence",
        "proposal",
        "coordinator_decision",
        "human_decision",
        "execution_authorization",
        "execution_result",
        "coordinator_review",
        "decision_receipt",
    ]


def test_replayed_authorization_requires_reconciliation_without_invocation(
    tmp_path: Path,
) -> None:
    store = EmbeddedDecisionArtifactStore(tmp_path)
    (
        _evidence,
        _proposal,
        _decision,
        _human,
        authorization,
        _result,
        _review,
        _receipt,
    ) = _append_authority_chain(store)
    first_commit = store.commit_authorization(authorization)
    assert first_commit.newly_committed is True
    invoked = False

    def executor(
        _claim: BoundedExecutionAuthorization,
    ) -> EmbeddedBoundedExecutionResult:
        nonlocal invoked
        invoked = True
        raise AssertionError("replayed authorization must not invoke executor")

    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    with pytest.raises(EmbeddedDecisionAuthorizationReplayError) as replay_error:
        recovered.invoke_after_authorization_commit(authorization, executor)

    assert invoked is False
    assert replay_error.value.outcome.authorization == artifact_reference(authorization)
    assert replay_error.value.outcome.operation_id == authorization.operation_id
    assert replay_error.value.outcome.durable_commit.newly_committed is False


def test_store_rejects_second_mutating_operation_for_same_stage_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, proposal, first_decision, first_human, first_authorization, *_rest = (
        _artifact_chain()
    )
    store = EmbeddedDecisionArtifactStore(tmp_path / "append")
    for artifact in (
        evidence,
        proposal,
        first_decision,
        first_human,
        first_authorization,
    ):
        store.append(artifact)

    revised_semantics = AcceptedSemanticDecision(
        schema_version="articulation.decision.v1",
        values={"joint": "slider"},
    )
    second_decision = EmbeddedCoordinatorDecision(
        artifact_id="decision-002",
        identity=first_decision.identity,
        producer=first_decision.producer,
        parent_artifact=artifact_reference(proposal),
        created_at=_START + timedelta(seconds=8),
        disposition="accept",
        evidence_artifacts=(artifact_reference(evidence),),
        proposal_artifacts=(artifact_reference(proposal),),
        accepted_decision=revised_semantics,
        accepted_decision_digest=accepted_semantic_decision_digest(
            first_decision.identity,
            revised_semantics,
        ),
        human_decision_required=True,
        rationale="Accepted revised joint semantics.",
    )
    second_decision_reference = artifact_reference(second_decision)
    assert second_decision.accepted_decision_digest is not None
    second_human = EmbeddedHumanDecision(
        artifact_id="human-002",
        identity=first_decision.identity,
        producer=first_human.producer,
        parent_artifact=second_decision_reference,
        created_at=_START + timedelta(seconds=9),
        coordinator_decision=second_decision_reference,
        reviewed_decision_digest=second_decision.accepted_decision_digest,
        disposition="accept",
        rationale="Human accepted the revised semantics.",
    )
    second_authorization = authorize_bounded_execution(
        second_decision,
        expected_identity=first_decision.identity,
        evidence=(evidence,),
        existing_authorizations=(),
        proposals=(proposal,),
        executor=first_authorization.executor,
        execution_effect="mutation",
        human_decision=second_human,
    )
    for artifact in (second_decision, second_human):
        store.append(artifact)

    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="already has mutating authority",
    ):
        store.invoke_after_authorization_commit(second_authorization, executor)
    assert invoked is False

    persisted = EmbeddedDecisionArtifactStore(tmp_path / "load")
    monkeypatch.setattr(
        artifact_store_module,
        "_validate_cross_artifact_dependencies",
        lambda _artifact, _known: None,
    )
    for artifact in (
        evidence,
        proposal,
        first_decision,
        first_human,
        first_authorization,
        second_decision,
        second_human,
        second_authorization,
    ):
        persisted.append(artifact)
    monkeypatch.undo()

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="already has mutating authority",
    ):
        persisted.journal()


def test_crash_orphan_authorization_reconstructs_and_invokes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, proposal, decision, human, template, *_rest = _artifact_chain()
    store = EmbeddedDecisionArtifactStore(tmp_path)
    for artifact in (evidence, proposal, decision, human):
        store.append(artifact)

    def reconstruct() -> BoundedExecutionAuthorization:
        return authorize_bounded_execution(
            decision,
            expected_identity=decision.identity,
            evidence=(evidence,),
            existing_authorizations=(),
            proposals=(proposal,),
            executor=template.executor,
            execution_effect="mutation",
            human_decision=human,
        )

    authorization = reconstruct()
    assert reconstruct() == authorization
    assert authorization.created_at == human.created_at

    def interrupt_journal_write(_store_fd: int, _journal: object) -> None:
        raise OSError("simulated crash after authorization bytes")

    monkeypatch.setattr(store, "_write_journal_locked", interrupt_journal_write)
    with pytest.raises(OSError, match="after authorization bytes"):
        store.commit_authorization(authorization)
    monkeypatch.undo()

    conflicting = authorization.model_copy(
        update={"created_at": authorization.created_at + timedelta(seconds=1)}
    )
    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="different canonical"):
        recovered.commit_authorization(conflicting)

    reconstructed = reconstruct()
    invocations: list[str] = []

    def executor(claim: BoundedExecutionAuthorization) -> str:
        invocations.append(claim.operation_id)
        return "executed"

    assert recovered.invoke_after_authorization_commit(reconstructed, executor) == (
        "executed"
    )
    assert invocations == [authorization.operation_id]
    assert (
        recovered.load_typed(
            artifact_reference(authorization),
            BoundedExecutionAuthorization,
        )
        == authorization
    )

    with pytest.raises(EmbeddedDecisionAuthorizationReplayError):
        recovered.invoke_after_authorization_commit(reconstructed, executor)
    assert invocations == [authorization.operation_id]


@pytest.mark.parametrize(
    ("veto_disposition", "revision_requests"),
    [("reject", ()), ("revise", ("Revise the accepted semantics.",))],
)
def test_store_rejects_second_human_disposition_for_same_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    veto_disposition: str,
    revision_requests: tuple[str, ...],
) -> None:
    evidence, proposal, decision, accepted_human, authorization, *_rest = (
        _artifact_chain()
    )
    veto = accepted_human.model_copy(
        update={
            "artifact_id": f"human-{veto_disposition}",
            "disposition": veto_disposition,
            "rationale": "Human vetoed the exact coordinator decision.",
            "revision_requests": revision_requests,
        }
    )
    store = EmbeddedDecisionArtifactStore(tmp_path / "append")
    for artifact in (evidence, proposal, decision, veto):
        store.append(artifact)

    assert store.append(veto).newly_committed is False
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="already has a durable human decision",
    ):
        store.append(accepted_human)

    persisted = EmbeddedDecisionArtifactStore(tmp_path / "load")
    monkeypatch.setattr(
        artifact_store_module,
        "_validate_cross_artifact_dependencies",
        lambda _artifact, _known: None,
    )
    for artifact in (evidence, proposal, decision, veto, accepted_human):
        persisted.append(artifact)
    monkeypatch.undo()

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="already has a durable human decision",
    ):
        persisted.journal()

    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="already has a durable human decision",
    ):
        persisted.invoke_after_authorization_commit(authorization, executor)
    assert invoked is False


@pytest.mark.parametrize(
    "drift",
    [
        "capability_digest",
        "source_path",
        "source_digest",
        "outer_request_path",
        "domain_run_root",
    ],
)
def test_store_rejects_same_stage_attempt_identity_drift_at_every_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: Literal[
        "capability_digest",
        "source_path",
        "source_digest",
        "outer_request_path",
        "domain_run_root",
    ],
) -> None:
    first_chain = _artifact_chain()[:5]
    drifted_chain = _drifted_authority_chain(drift)
    first_authorization = first_chain[-1]
    drifted_authorization = drifted_chain[-1]
    assert isinstance(first_authorization, BoundedExecutionAuthorization)
    assert isinstance(drifted_authorization, BoundedExecutionAuthorization)
    assert first_authorization.identity != drifted_authorization.identity
    assert (
        first_authorization.outer_stage_attempt_seal_key
        == drifted_authorization.outer_stage_attempt_seal_key
    )

    store = EmbeddedDecisionArtifactStore(tmp_path / "append")
    for artifact in first_chain:
        store.append(artifact)
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="different full decision identity",
    ):
        store.append(drifted_chain[0])

    persisted = EmbeddedDecisionArtifactStore(tmp_path / "load-invoke")
    monkeypatch.setattr(
        artifact_store_module,
        "_validate_cross_artifact_dependencies",
        lambda _artifact, _known: None,
    )
    for artifact in (*first_chain, *drifted_chain[:-1]):
        persisted.append(artifact)
    monkeypatch.undo()

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="different full decision identity",
    ):
        persisted.journal()
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="different full decision identity",
    ):
        persisted.load_typed(
            artifact_reference(first_authorization),
            BoundedExecutionAuthorization,
        )

    invoked = False

    def executor(_claim: BoundedExecutionAuthorization) -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError,
        match="different full decision identity",
    ):
        persisted.invoke_after_authorization_commit(
            drifted_authorization,
            executor,
        )
    assert invoked is False


def test_store_allows_only_a_genuinely_advanced_outer_stage_attempt(
    tmp_path: Path,
) -> None:
    first_chain = _artifact_chain()[:5]
    advanced_chain = _drifted_authority_chain("stage_attempt")
    first_authorization = first_chain[-1]
    advanced_authorization = advanced_chain[-1]
    assert isinstance(first_authorization, BoundedExecutionAuthorization)
    assert isinstance(advanced_authorization, BoundedExecutionAuthorization)
    assert (
        first_authorization.outer_stage_attempt_seal_key
        != advanced_authorization.outer_stage_attempt_seal_key
    )
    assert advanced_authorization.identity.execution_context.embedded_stage is not None
    assert (
        advanced_authorization.identity.execution_context.embedded_stage.stage_attempt
        == 2
    )

    store = EmbeddedDecisionArtifactStore(tmp_path)
    for artifact in (*first_chain, *advanced_chain):
        store.append(artifact)

    assert (
        store.load_typed(
            artifact_reference(advanced_authorization),
            BoundedExecutionAuthorization,
        )
        == advanced_authorization
    )
    assert (
        sum(
            entry.reference.artifact_kind == "execution_authorization"
            for entry in store.journal().entries
        )
        == 2
    )


def test_crash_orphan_receipt_reconstructs_and_appends_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, proposal, decision, human, authorization, result, review, _receipt = (
        _artifact_chain()
    )
    store = EmbeddedDecisionArtifactStore(tmp_path)
    for artifact in (
        evidence,
        proposal,
        decision,
        human,
        authorization,
        result,
        review,
    ):
        store.append(artifact)

    def reconstruct() -> EmbeddedDecisionReceipt:
        return build_decision_receipt(
            artifact_id="receipt-reconstructed",
            decision=decision,
            evidence=(evidence,),
            proposals=(proposal,),
            authorization=authorization,
            result=result,
            review=review,
            prior_lineage=PersistedExecutionLineage(),
            human_decision=human,
        )

    receipt = reconstruct()
    assert reconstruct() == receipt
    assert receipt.created_at == review.created_at

    def interrupt_journal_write(_store_fd: int, _journal: object) -> None:
        raise OSError("simulated crash after receipt bytes")

    monkeypatch.setattr(store, "_write_journal_locked", interrupt_journal_write)
    with pytest.raises(OSError, match="after receipt bytes"):
        store.append_receipt(receipt)
    monkeypatch.undo()

    recovered = EmbeddedDecisionArtifactStore(tmp_path)
    conflicting = receipt.model_copy(
        update={"created_at": receipt.created_at + timedelta(seconds=1)}
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="different canonical"):
        recovered.append_receipt(conflicting)

    reconstructed = reconstruct()
    commit = recovered.append_receipt(reconstructed)
    assert commit.newly_committed is True
    assert (
        recovered.load_typed(
            artifact_reference(receipt),
            EmbeddedDecisionReceipt,
        )
        == receipt
    )
    assert recovered.append_receipt(reconstructed).newly_committed is False


def test_store_rejects_and_revalidates_locally_valid_forged_authority_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, proposal, accepted_decision, *_rest = _artifact_chain()
    rejected_decision = EmbeddedCoordinatorDecision(
        artifact_id="decision-rejected",
        identity=accepted_decision.identity,
        producer=accepted_decision.producer,
        parent_artifact=artifact_reference(proposal),
        created_at=accepted_decision.created_at,
        disposition="reject",
        evidence_artifacts=(artifact_reference(evidence),),
        proposal_artifacts=(artifact_reference(proposal),),
        rationale="Rejected semantics cannot authorize execution.",
    )
    rejected_reference = artifact_reference(rejected_decision)
    fake_digest = "d" * 64
    operation_id = canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.embedded-operation-id.v1",
            "identity": rejected_decision.identity.model_dump(mode="json"),
            "accepted_decision_digest": fake_digest,
            "execution_effect": "mutation",
        }
    )
    executor = _producer("joint-executor", "executor", "a")
    forged = BoundedExecutionAuthorization(
        artifact_id=f"authorization-{operation_id}-001",
        identity=rejected_decision.identity,
        producer=rejected_decision.producer,
        parent_artifact=rejected_reference,
        created_at=rejected_decision.created_at + timedelta(seconds=1),
        accepted_decision=rejected_reference,
        accepted_decision_digest=fake_digest,
        executor=executor,
        execution_effect="mutation",
        outer_stage_attempt_seal_key=outer_stage_attempt_seal_key(
            rejected_decision.identity
        ),
        operation_id=operation_id,
        mutation_id=operation_id,
        attempt=1,
    )

    store = EmbeddedDecisionArtifactStore(tmp_path / "append")
    for upstream_artifact in (evidence, proposal, rejected_decision):
        store.append(upstream_artifact)
    with pytest.raises(
        EmbeddedDecisionArtifactStoreError, match="reject, not accepted"
    ):
        store.append(forged)

    persisted = EmbeddedDecisionArtifactStore(tmp_path / "load")
    monkeypatch.setattr(
        artifact_store_module,
        "_validate_cross_artifact_dependencies",
        lambda _artifact, _known: None,
    )
    for forged_chain_artifact in (evidence, proposal, rejected_decision, forged):
        persisted.append(forged_chain_artifact)
    monkeypatch.undo()

    with pytest.raises(
        EmbeddedDecisionArtifactStoreError, match="reject, not accepted"
    ):
        persisted.journal()


def test_store_rejects_locally_valid_cross_chain_forgery_at_each_boundary(
    tmp_path: Path,
) -> None:
    evidence, proposal, decision, human, authorization, result, review, receipt = (
        _artifact_chain()
    )

    proposal_store = EmbeddedDecisionArtifactStore(tmp_path / "proposal")
    proposal_store.append(evidence)
    alternate_source = _binding("/run/substituted.usdz", "e")
    assert proposal.identity.execution_context.embedded_stage is not None
    alternate_stage = proposal.identity.execution_context.embedded_stage.model_copy(
        update={"input_asset": alternate_source}
    )
    alternate_context = proposal.identity.execution_context.model_copy(
        update={"embedded_stage": alternate_stage}
    )
    alternate_identity = proposal.identity.model_copy(
        update={"source": alternate_source, "execution_context": alternate_context}
    )
    forged_proposal = EmbeddedDomainProposal.model_validate(
        {
            **proposal.model_dump(mode="python"),
            "identity": alternate_identity,
        }
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="identity.*stale"):
        proposal_store.append(forged_proposal)

    human_store = EmbeddedDecisionArtifactStore(tmp_path / "human")
    for human_dependency in (evidence, proposal, decision):
        human_store.append(human_dependency)
    forged_human = EmbeddedHumanDecision.model_validate(
        {
            **human.model_dump(mode="python"),
            "reviewed_decision_digest": "f" * 64,
        }
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="different accepted"):
        human_store.append(forged_human)

    result_store = EmbeddedDecisionArtifactStore(tmp_path / "result")
    for result_dependency in (evidence, proposal, decision, human, authorization):
        result_store.append(result_dependency)
    forged_result = EmbeddedBoundedExecutionResult.model_validate(
        {
            **result.model_dump(mode="python"),
            "producer": _producer("foreign-executor", "executor", "a"),
        }
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="exact bounded"):
        result_store.append(forged_result)
    result_store.append(result)
    duplicate_result = result.model_copy(update={"artifact_id": "result-duplicate"})
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="already has"):
        result_store.append(duplicate_result)

    review_store = EmbeddedDecisionArtifactStore(tmp_path / "review")
    for review_dependency in (
        evidence,
        proposal,
        decision,
        human,
        authorization,
        result,
    ):
        review_store.append(review_dependency)
    foreign_owner = _producer("foreign-coordinator", "outer_coordinator", "9")
    forged_review = EmbeddedCoordinatorReview.model_validate(
        {
            **review.model_dump(mode="python"),
            "producer": foreign_owner,
            "semantic_decision_owner": foreign_owner,
        }
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="semantic decision"):
        review_store.append(forged_review)

    receipt_store = EmbeddedDecisionArtifactStore(tmp_path / "receipt")
    for receipt_dependency in (
        evidence,
        proposal,
        decision,
        human,
        authorization,
        result,
        review,
    ):
        receipt_store.append(receipt_dependency)
    forged_receipt = EmbeddedDecisionReceipt.model_validate(
        {
            **receipt.model_dump(mode="python"),
            "evidence_providers": (
                _producer("foreign-evidence", "evidence_provider", "7"),
            ),
        }
    )
    with pytest.raises(EmbeddedDecisionArtifactStoreError, match="complete persisted"):
        receipt_store.append(forged_receipt)
