# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-entrypoint regressions for outer-owned embedded articulation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import content_agent_workflows.asset_composition.state as asset_state_module
import pytest
from content_agent_workflows.articulation import (
    ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION,
    ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
    EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationFinalizationResult,
    ArticulationInferenceResult,
    ArticulationRunState,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationDecisionPatch,
    EmbeddedArticulationError,
    EmbeddedArticulationGraphChange,
    EmbeddedArticulationGraphRevision,
    EmbeddedArticulationGraphRevisionPatch,
    EmbeddedArticulationGroup,
    EmbeddedArticulationHumanAcceptance,
    EmbeddedArticulationHumanReviewPolicy,
    EmbeddedArticulationJoint,
    EmbeddedArticulationMembership,
    EmbeddedArticulationOuterReview,
    EmbeddedArticulationOutputEvidence,
    EmbeddedArticulationPostReviewPatch,
    EmbeddedArticulationPreparation,
    EmbeddedArticulationReadback,
    EmbeddedArticulationRuntime,
    EmbeddedArticulationTerminalReceipt,
    MembershipDispositionDocument,
    MockArticulationSceneEvidenceCollector,
    Stage2ArticulationCandidate,
    Stage2CandidateDocument,
    Stage2CandidateSummary,
    apply_embedded_articulation_decision_patch,
    apply_embedded_articulation_graph_revision,
    apply_embedded_articulation_post_review,
    bind_embedded_articulation_output_evidence,
    prepare_embedded_articulation_workflow,
    run_embedded_articulation_workflow,
)
from content_agent_workflows.articulation.embedded_decision import (
    _articulation_capabilities_from_evidence,
    _graph_candidate_document,
)
from content_agent_workflows.articulation.verified_operations import (
    VERIFIED_OPERATION_CONTRACT_CHECKPOINT,
    project_joint_graph_apply_result,
)
from content_agent_workflows.articulation.workflow import _source_identity
from content_agent_workflows.asset_composition import (
    AssetCompositionRun,
    AssetCompositionStateError,
    begin_stage,
    build_embedded_domain_execution_context,
    complete_stage,
    create_run,
    load_embedded_articulation_human_acceptance,
    load_verified_run,
    record_articulation_graph_revision,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    record_review_decisions,
    require_review,
    stage_directory,
)
from content_agent_workflows.common import (
    EmbeddedDecisionArtifactStore,
    EmbeddedDecisionAuthorizationReplayError,
)
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.common.embedded_domain_decision import (
    ContractArtifactReference,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EmbeddedHumanDecision,
    NamedDecisionDigests,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    artifact_reference,
    canonical_json_digest,
)
from content_agent_workflows.validation import (
    execution_artifact_binding,
    produce_canonical_visual_evidence,
)
from content_agent_workflows.validation.verified_operations import (
    VerifiedOperationError,
    ingest_verified_operation_result,
    verify_operation_envelope,
)
from pydantic import ValidationError

from content_workflow_cli import articulation_runner
from content_workflow_cli.cli import main as workflow_cli_main


def _run_direct_outer_transition(
    _state_path: Path,
    *,
    transition: Callable[[], ArticulationFinalizationResult],
) -> ArticulationFinalizationResult:
    return transition()


def _binding(path: Path) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _candidate(
    candidate_id: str,
    child: str,
) -> Stage2ArticulationCandidate:
    return Stage2ArticulationCandidate(
        candidate_id=candidate_id,
        motion_type="prismatic",
        moving_part_prims=(child,),
        fixed_parent_prim="/Assembly/RootUnit",
        parent_resolution_source="stage1_hint",
        joint_type_hint="prismatic",
        axis_hint="x",
        motion_axis_world=(1.0, 0.0, 0.0),
        confidence="high",
        role="moving_unit",
        field_sources={
            "motion_type": "predicted",
            "axis_hint": "predicted",
            "motion_axis_world": "predicted",
            "fixed_parent_prim": "stage1_hint",
        },
        axis_evidence=(
            {
                "source": "predicted",
                "description": "provider proposal axis",
                "value": "x",
                "prim_paths": (child,),
            },
        ),
        connectivity_evidence=(
            {
                "source": "stage1_hint",
                "description": "provider proposal edge",
                "value": "/Assembly/RootUnit",
                "prim_paths": ("/Assembly/RootUnit", child),
                "connectivity_role": "body0_body1_edge",
            },
        ),
        review_status="ready_for_rigger_input",
    )


def _provider_document() -> Stage2CandidateDocument:
    candidates = (
        _candidate("motion_a", "/Assembly/MovingUnitA"),
        _candidate("motion_b", "/Assembly/MovingUnitB"),
    )
    return Stage2CandidateDocument(
        summary=Stage2CandidateSummary(
            candidate_count=2,
            ready_candidate_count=2,
            review_required_candidate_count=0,
        ),
        candidates=candidates,
    )


def _membership() -> MembershipDispositionDocument:
    records = [
        {
            "disposition_id": "membership_a",
            "member_prim": "/Assembly/MovingUnitA",
            "motion_candidate_prim": "/Assembly/MovingUnitA",
            "physical_owner_prim": "/Assembly/MovingUnitA",
            "physical_owner_candidate_prim": "/Assembly/MovingUnitA",
            "disposition": "independent_motion",
            "source": "llm_adjudicated",
            "confidence": "high",
            "rationale": "provider proposal only",
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        },
        {
            "disposition_id": "membership_a_detail",
            "member_prim": "/Assembly/MovingUnitA/AttachedPiece",
            "motion_candidate_prim": "/Assembly/MovingUnitA",
            "physical_owner_prim": "/Assembly/MovingUnitA",
            "physical_owner_candidate_prim": "/Assembly/MovingUnitA",
            "disposition": "co_rigid",
            "source": "llm_adjudicated",
            "confidence": "high",
            "rationale": "provider proposal only",
            "downstream_boundary": "co_rigid_aggregation",
            "review_status": "resolved",
        },
        {
            "disposition_id": "membership_b",
            "member_prim": "/Assembly/MovingUnitB",
            "motion_candidate_prim": "/Assembly/MovingUnitB",
            "physical_owner_prim": "/Assembly/MovingUnitB",
            "physical_owner_candidate_prim": "/Assembly/MovingUnitB",
            "disposition": "independent_motion",
            "source": "llm_adjudicated",
            "confidence": "high",
            "rationale": "provider proposal only",
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        },
    ]
    return MembershipDispositionDocument.model_validate(
        {
            "schema_version": "joint-agent-membership-disposition-v1",
            "summary": {
                "disposition_count": 3,
                "disposition_counts": {
                    "independent_motion": 2,
                    "co_rigid": 1,
                },
                "review_required_count": 0,
                "pending_downstream_count": 0,
            },
            "dispositions": records,
        }
    )


class _GraphOnlyClient:
    def __init__(
        self,
        predictions_path: Path,
        *,
        exact_graph_match: bool = True,
        omit_co_rigid_member: bool = False,
    ) -> None:
        self.predictions_path = predictions_path
        self.exact_graph_match = exact_graph_match
        self.omit_co_rigid_member = omit_co_rigid_member
        self.author_requests: list[ArticulationAuthoringRequest] = []
        self.author_call_count = 0

    def configuration_sha256(self, _request: Any) -> str:
        return "4" * 64

    def infer(self, request: Any, *, resume: bool, cancel_checker: Any = None):
        del resume, cancel_checker
        return ArticulationInferenceResult(
            candidate_document=_provider_document(),
            membership_disposition_document=_membership(),
            backend_configuration_sha256=self.configuration_sha256(request),
            predictions_path=str(self.predictions_path.resolve()),
            predictions_sha256=file_sha256(self.predictions_path),
            metadata={"backend": "mock-provider", "provider_proposal_only": True},
        )

    def author(
        self, request: ArticulationAuthoringRequest, *, cancel_checker: Any = None
    ):
        del cancel_checker
        self.author_call_count += 1
        self.author_requests.append(request)
        document = Stage2CandidateDocument.model_validate_json(
            Path(request.candidate_document_path).read_bytes()
        )
        output = request.output_dir / "joint_rigger" / "rigged.usda"
        output.parent.mkdir(parents=True, exist_ok=True)
        attached_piece = (
            ""
            if self.omit_co_rigid_member
            else '        def Xform "AttachedPiece" {}\n'
        )
        output.write_text(
            """#usda 1.0
def Xform "Assembly" {
    def Xform "RootUnit" {}
    def Xform "MovingUnitA" {
"""
            + attached_piece
            + """    }
    def Xform "MovingUnitB" {}
}
""",
            encoding="utf-8",
        )
        return ArticulationAuthoringResult(
            status="succeeded",
            idempotency_key=request.idempotency_key,
            source_sha256=request.source_sha256,
            source_dependency_bundle_sha256=request.source_dependency_bundle_sha256,
            candidate_document_path=request.candidate_document_path,
            candidate_document_sha256=request.candidate_document_sha256,
            output_asset_path=str(output.resolve()),
            output_asset_sha256=file_sha256(output),
            authored_candidate_ids=document.candidate_ids,
            authored_joint_count=len(document.candidate_ids),
            metadata={"adapter": "graph-only-test"},
        )

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: Any = None,
    ) -> ArticulationValidationResult:
        del cancel_checker
        failures = () if self.exact_graph_match else ("readback mismatch",)
        return ArticulationValidationResult(
            status="pass" if self.exact_graph_match else "fail",
            output_asset_path=authoring.output_asset_path,
            expected_output_asset_sha256=authoring.output_asset_sha256,
            observed_output_asset_sha256=authoring.output_asset_sha256,
            expected_candidate_ids=expected_candidate_ids,
            validated_candidate_ids=(
                expected_candidate_ids if self.exact_graph_match else ()
            ),
            exact_graph_match=self.exact_graph_match,
            self_contained=True,
            failures=failures,
        )


class _ProviderNeutralAuthoringClient:
    def __init__(self) -> None:
        self.author_requests: list[ArticulationAuthoringRequest] = []

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: Any = None,
    ) -> ArticulationAuthoringResult:
        del cancel_checker
        self.author_requests.append(request)
        document = Stage2CandidateDocument.model_validate_json(
            Path(request.candidate_document_path).read_bytes()
        )
        output = request.output_dir / "joint_rigger" / "rigged.usda"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(Path(request.source_asset).read_bytes())
        return ArticulationAuthoringResult(
            status="succeeded",
            idempotency_key=request.idempotency_key,
            source_sha256=request.source_sha256,
            source_dependency_bundle_sha256=(request.source_dependency_bundle_sha256),
            candidate_document_path=request.candidate_document_path,
            candidate_document_sha256=request.candidate_document_sha256,
            output_asset_path=str(output.resolve()),
            output_asset_sha256=file_sha256(output),
            authored_candidate_ids=document.candidate_ids,
            authored_joint_count=len(document.candidate_ids),
            metadata={"adapter": "provider-neutral-test"},
        )

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: Any = None,
    ) -> ArticulationValidationResult:
        del cancel_checker
        return ArticulationValidationResult(
            status="pass",
            output_asset_path=authoring.output_asset_path,
            expected_output_asset_sha256=authoring.output_asset_sha256,
            observed_output_asset_sha256=authoring.output_asset_sha256,
            expected_candidate_ids=expected_candidate_ids,
            validated_candidate_ids=expected_candidate_ids,
            exact_graph_match=True,
            self_contained=True,
        )


def _graph() -> EmbeddedArticulationCanonicalGraph:
    evidence_ids = (
        "source-identity",
        "joint-topology-inspection",
        "joint-membership-inspection",
        "joint-visual-inspection",
        "joint-authoring-capabilities",
    )
    memberships = (
        EmbeddedArticulationMembership(
            member_prim="/Assembly/RootUnit",
            authoritative_owner_prim="/Assembly/RootUnit",
            group_id="root_group",
            disposition="co_rigid",
            state="source_backed",
            evidence_ids=evidence_ids,
        ),
        EmbeddedArticulationMembership(
            member_prim="/Assembly/MovingUnitA",
            authoritative_owner_prim="/Assembly/MovingUnitA",
            group_id="moving_a_group",
            disposition="independent_motion",
            state="source_backed",
            evidence_ids=evidence_ids,
        ),
        EmbeddedArticulationMembership(
            member_prim="/Assembly/MovingUnitA/AttachedPiece",
            authoritative_owner_prim="/Assembly/MovingUnitA",
            group_id="moving_a_group",
            disposition="co_rigid",
            state="source_backed",
            evidence_ids=evidence_ids,
        ),
        EmbeddedArticulationMembership(
            member_prim="/Assembly/MovingUnitB",
            authoritative_owner_prim="/Assembly/MovingUnitB",
            group_id="moving_b_group",
            disposition="independent_motion",
            state="source_backed",
            evidence_ids=evidence_ids,
        ),
    )
    groups = tuple(
        EmbeddedArticulationGroup(
            group_id=group_id,
            authoritative_owner_prim=owner,
            member_prims=members,
            role=role,
            role_state="known",
            evidence_ids=evidence_ids,
        )
        for group_id, owner, members, role in (
            ("root_group", "/Assembly/RootUnit", ("/Assembly/RootUnit",), "base"),
            (
                "moving_a_group",
                "/Assembly/MovingUnitA",
                (
                    "/Assembly/MovingUnitA",
                    "/Assembly/MovingUnitA/AttachedPiece",
                ),
                "primary_moving_member",
            ),
            (
                "moving_b_group",
                "/Assembly/MovingUnitB",
                ("/Assembly/MovingUnitB",),
                "secondary_moving_member",
            ),
        )
    )
    joints = tuple(
        EmbeddedArticulationJoint(
            joint_id=joint_id,
            body0_owner_prim="/Assembly/RootUnit",
            body1_owner_prim=child,
            body0_role="base",
            body1_role=role,
            role_state="known",
            joint_type="revolute",
            endpoint_state="source_backed",
            type_state="known",
            axis="z",
            axis_state="known",
            lower_limit=-30.0,
            upper_limit=45.0,
            limit_unit="degrees",
            limit_state="source_backed",
            frame_policy="body1_world_origin",
            frame_state="known",
            evidence_ids=evidence_ids,
        )
        for joint_id, child, role in (
            ("motion_a", "/Assembly/MovingUnitA", "primary_moving_member"),
            ("motion_b", "/Assembly/MovingUnitB", "secondary_moving_member"),
        )
    )
    return EmbeddedArticulationCanonicalGraph(
        graph_id="neutral-multi-owner-graph",
        source_sha256="0" * 64,  # replaced by the fixture
        source_dependency_bundle_sha256="0" * 64,
        source_member_prims=tuple(item.member_prim for item in memberships),
        authoritative_owner_prims=(
            "/Assembly/MovingUnitA",
            "/Assembly/MovingUnitB",
            "/Assembly/RootUnit",
        ),
        candidate_ids=("motion_a", "motion_b"),
        groups=groups,
        memberships=memberships,
        joints=joints,
    )


def test_neutral_multibody_chain_maps_only_canonical_owner_edges() -> None:
    """A second source-free shape covers serial multibody composition."""

    evidence_ids = (
        "source-identity",
        "joint-topology-inspection",
        "joint-membership-inspection",
        "joint-visual-inspection",
        "joint-authoring-capabilities",
    )
    owners = (
        "/Structure/Anchor",
        "/Structure/IntermediateBody",
        "/Structure/TerminalBody",
    )
    graph = EmbeddedArticulationCanonicalGraph(
        graph_id="neutral-serial-multibody-graph",
        source_sha256="1" * 64,
        source_dependency_bundle_sha256="2" * 64,
        source_member_prims=owners,
        authoritative_owner_prims=owners,
        candidate_ids=("intermediate_motion", "terminal_motion"),
        groups=tuple(
            EmbeddedArticulationGroup(
                group_id=f"group_{index}",
                authoritative_owner_prim=owner,
                member_prims=(owner,),
                role=role,
                role_state="source_backed",
                evidence_ids=evidence_ids,
            )
            for index, (owner, role) in enumerate(
                zip(owners, ("anchor", "link", "terminal"), strict=True)
            )
        ),
        memberships=tuple(
            EmbeddedArticulationMembership(
                member_prim=owner,
                authoritative_owner_prim=owner,
                group_id=f"group_{index}",
                disposition=("co_rigid" if index == 0 else "independent_motion"),
                state="source_backed",
                evidence_ids=evidence_ids,
            )
            for index, owner in enumerate(owners)
        ),
        joints=(
            EmbeddedArticulationJoint(
                joint_id="intermediate_motion",
                body0_owner_prim=owners[0],
                body1_owner_prim=owners[1],
                body0_role="anchor",
                body1_role="link",
                role_state="known",
                joint_type="revolute",
                endpoint_state="source_backed",
                type_state="known",
                axis="y",
                axis_state="known",
                lower_limit=-20.0,
                upper_limit=70.0,
                limit_unit="degrees",
                limit_state="source_backed",
                frame_policy="body1_world_origin",
                frame_state="known",
                evidence_ids=evidence_ids,
            ),
            EmbeddedArticulationJoint(
                joint_id="terminal_motion",
                body0_owner_prim=owners[1],
                body1_owner_prim=owners[2],
                body0_role="link",
                body1_role="terminal",
                role_state="known",
                joint_type="prismatic",
                endpoint_state="source_backed",
                type_state="known",
                axis="-z",
                axis_state="known",
                lower_limit=0.0,
                upper_limit=0.4,
                limit_unit="meters",
                limit_state="source_backed",
                frame_policy="body1_world_origin",
                frame_state="known",
                evidence_ids=evidence_ids,
            ),
        ),
    )

    document = _graph_candidate_document(graph)

    assert tuple(
        (candidate.fixed_parent_prim, candidate.moving_part_prims)
        for candidate in document.candidates
    ) == (
        (owners[0], (owners[1],)),
        (owners[1], (owners[2],)),
    )
    assert tuple(candidate.motion_type for candidate in document.candidates) == (
        "revolute",
        "prismatic",
    )
    assert tuple(candidate.axis_hint for candidate in document.candidates) == (
        "y",
        "-z",
    )
    assert all(
        candidate.parent_resolution_source == "accepted_manifest"
        and candidate.source_prediction_ids == ()
        for candidate in document.candidates
    )


def _runtime(context: DomainExecutionContext) -> EmbeddedArticulationRuntime:
    digests = {
        "embedded": "a" * 64,
        "joint": "b" * 64,
        "outer": "c" * 64,
    }
    identity = EmbeddedDecisionIdentity(
        execution_context=context,
        source=context.embedded_stage.input_asset,  # type: ignore[union-attr]
        coordinator_plan=ContractArtifactReference(
            artifact_kind="coordinator_plan",
            artifact_id="plan-1",
            schema_version="content-agent-workflows.asset-coordinator-plan.v1",
            sha256=context.embedded_stage.coordinator_plan.sha256,  # type: ignore[union-attr]
        ),
        digests=NamedDecisionDigests(
            configuration={"request": "d" * 64},
            prompt={"prompt": "e" * 64},
            capabilities={"articulation": "f" * 64},
            implementations=digests,
        ),
    )
    return EmbeddedArticulationRuntime(
        identity=identity,
        evidence_provider=ProducerIdentity(
            producer_id="joint-inspection",
            role="evidence_provider",
            implementation="test-joint",
            implementation_digest=digests["joint"],
        ),
        proposal_provider=ProducerIdentity(
            producer_id="joint-proposal",
            role="proposal_provider",
            implementation="test-joint",
            implementation_digest=digests["joint"],
        ),
        outer_coordinator=ProducerIdentity(
            producer_id="outer",
            role="outer_coordinator",
            implementation="test-outer",
            implementation_digest=digests["outer"],
        ),
        executor=ProducerIdentity(
            producer_id="graph-authorer",
            role="executor",
            implementation="test-graph-authorer",
            implementation_digest=digests["embedded"],
        ),
        capabilities=EmbeddedArticulationCapabilityLimits(),
    )


def _synthetic_source(
    tmp_path: Path,
    *,
    owner_count: int,
    member_count: int,
) -> tuple[Path, tuple[str, ...], tuple[str, ...]]:
    owners = tuple(f"/Synthetic/Owner{index:03d}" for index in range(owner_count))
    members = list(owners)
    child_index = 0
    while len(members) < member_count:
        owner_index = child_index % owner_count
        members.append(f"{owners[owner_index]}/Member{child_index:03d}")
        child_index += 1
    children_by_owner = {
        owner: [item for item in members if item.startswith(f"{owner}/")]
        for owner in owners
    }
    lines = ["#usda 1.0", 'def Xform "Synthetic" {']
    for owner in owners:
        lines.append(f'    def Xform "{owner.rsplit("/", 1)[-1]}" {{')
        lines.extend(
            f'        def Xform "{child.rsplit("/", 1)[-1]}" {{}}'
            for child in children_by_owner[owner]
        )
        lines.append("    }")
    lines.append("}")
    source = tmp_path / "synthetic_structure.usda"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source, tuple(members), owners


def _provider_neutral_context(
    tmp_path: Path,
    *,
    source: Path,
    run_dir: Path,
) -> DomainExecutionContext:
    plan = tmp_path / f"{run_dir.name}_plan.json"
    plan.write_text('{"plan":"outer-owned"}\n', encoding="utf-8")
    outer_request = tmp_path / f"{run_dir.name}_request.json"
    outer_request.write_text('{"request":"synthetic"}\n', encoding="utf-8")
    return DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id=f"outer-{run_dir.name}",
            outer_request=_binding(outer_request),
            stage="articulation",
            stage_attempt=1,
            coordinator_plan=_binding(plan),
            input_asset=_binding(source),
            domain_run_root=str(run_dir.resolve()),
        ),
    )


def _provider_neutral_preparation(
    tmp_path: Path,
    *,
    source: Path,
    members: tuple[str, ...],
    owners: tuple[str, ...],
    canonical_output_evidence_required: bool = False,
) -> EmbeddedArticulationPreparation:
    source_sha256, dependency_sha256 = _source_identity(str(source))
    evidence_path = tmp_path / "synthetic_contract_evidence.json"
    atomic_write_json(
        evidence_path,
        {
            "source_sha256": source_sha256,
            "source_member_count": len(members),
            "authoritative_owner_count": len(owners),
            "capture": "deterministic-synthetic-contract",
        },
    )
    evidence_binding = _binding(evidence_path)
    source_binding = _binding(source)
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=canonical_output_evidence_required
    )
    provider = ProducerIdentity(
        producer_id="deterministic-articulation-inspection",
        role="evidence_provider",
        implementation="synthetic-contract-inspector-v1",
        implementation_digest="1" * 64,
    )
    return EmbeddedArticulationPreparation(
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=dependency_sha256,
        configuration_sha256=canonical_json_digest(
            {"inspection_contract": "synthetic-v1"}
        ),
        evidence_provider=provider,
        source_hierarchy=ProviderNeutralEvidenceRecord(
            evidence_id="source-hierarchy-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete deterministic synthetic source hierarchy.",
            artifacts=(source_binding, evidence_binding),
            facts={"root_prim": "/Synthetic", "member_count": len(members)},
        ),
        source_members=ProviderNeutralEvidenceRecord(
            evidence_id="joint-source-member-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete deterministic synthetic source-member scope.",
            artifacts=(evidence_binding,),
            facts={"source_member_prims": list(members)},
        ),
        authoritative_owners=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoritative-owner-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete deterministic synthetic owner scope.",
            artifacts=(evidence_binding,),
            facts={"authoritative_owner_prims": list(owners)},
        ),
        capabilities=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoring-capabilities",
            evidence_type="capability",
            status="available",
            summary="Exact deterministic graph-authoring capability limits.",
            facts=capabilities.model_dump(mode="json"),
        ),
        renders=ProviderNeutralEvidenceRecord(
            evidence_id="joint-render-inspection",
            evidence_type="render",
            status="available",
            summary="Pre-bound synthetic render contract evidence.",
            artifacts=(evidence_binding,),
            facts={"capture": "synthetic", "live_render_invoked": False},
        ),
        scene=ProviderNeutralEvidenceRecord(
            evidence_id="joint-scene-inspection",
            evidence_type="inspection",
            status="available",
            summary="Pre-bound synthetic scene-tool contract evidence.",
            artifacts=(evidence_binding,),
            facts={"capture": "synthetic", "live_scene_tool_invoked": False},
        ),
    )


def _provider_neutral_runtime(
    context: DomainExecutionContext,
    preparation: EmbeddedArticulationPreparation,
    *,
    canonical_output_evidence_required: bool = False,
) -> EmbeddedArticulationRuntime:
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=canonical_output_evidence_required
    )
    identity = EmbeddedDecisionIdentity(
        execution_context=context,
        source=context.embedded_stage.input_asset,  # type: ignore[union-attr]
        coordinator_plan=ContractArtifactReference(
            artifact_kind="coordinator_plan",
            artifact_id="synthetic-plan",
            schema_version="content-agent-workflows.asset-coordinator-plan.v1",
            sha256=context.embedded_stage.coordinator_plan.sha256,  # type: ignore[union-attr]
        ),
        digests=NamedDecisionDigests(
            configuration={"preparation": preparation.configuration_sha256},
            prompt={"outer_instructions": "2" * 64},
            capabilities={"graph_authoring": canonical_json_digest(capabilities)},
            implementations={
                "deterministic_inspector": (
                    preparation.evidence_provider.implementation_digest
                ),
                "outer_coordinator": "3" * 64,
                "graph_authorer": "4" * 64,
            },
        ),
    )
    return EmbeddedArticulationRuntime(
        identity=identity,
        evidence_provider=preparation.evidence_provider,
        proposal_provider=None,
        outer_coordinator=ProducerIdentity(
            producer_id="outer-coding-agent",
            role="outer_coordinator",
            implementation="codex-or-claude",
            implementation_digest="3" * 64,
        ),
        executor=ProducerIdentity(
            producer_id="deterministic-graph-authorer",
            role="executor",
            implementation="owned-core-graph-authorer",
            implementation_digest="4" * 64,
        ),
        capabilities=capabilities,
    )


def test_capability_limits_preserve_legacy_false_serialization() -> None:
    legacy = EmbeddedArticulationCapabilityLimits()
    assert "canonical_output_evidence_required" not in legacy.model_dump(mode="json")

    current = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=True
    )
    assert current.model_dump(mode="json")["canonical_output_evidence_required"] is True


def _provider_neutral_request(
    source: Path,
    run_dir: Path,
    context: DomainExecutionContext,
) -> ArticulationWorkflowRequest:
    return ArticulationWorkflowRequest(
        source_asset=str(source.resolve()),
        output_dir=run_dir,
        intent="Author a complete synthetic articulation graph.",
        review_policy="all",
        max_candidate_count=256,
        metadata=metadata_with_domain_execution_context({}, context),
    )


def _synthetic_graph(
    preparation: EmbeddedArticulationPreparation,
    *,
    members: tuple[str, ...],
    owners: tuple[str, ...],
) -> EmbeddedArticulationCanonicalGraph:
    evidence_ids = (
        "source-identity",
        "source-hierarchy-inspection",
        "joint-source-member-inspection",
        "joint-authoritative-owner-inspection",
        "joint-authoring-capabilities",
        "joint-render-inspection",
        "joint-scene-inspection",
    )
    members_by_owner = {
        owner: tuple(
            member
            for member in members
            if member == owner or member.startswith(f"{owner}/")
        )
        for owner in owners
    }
    groups = tuple(
        EmbeddedArticulationGroup(
            group_id=f"group_{index:03d}",
            authoritative_owner_prim=owner,
            member_prims=members_by_owner[owner],
            role="base" if index == 0 else "moving_member",
            role_state="source_backed",
            evidence_ids=evidence_ids,
        )
        for index, owner in enumerate(owners)
    )
    owner_indexes = {owner: index for index, owner in enumerate(owners)}
    memberships = tuple(
        EmbeddedArticulationMembership(
            member_prim=member,
            authoritative_owner_prim=owner,
            group_id=f"group_{owner_indexes[owner]:03d}",
            disposition=(
                "independent_motion"
                if member == owner and owner_indexes[owner] > 0
                else "co_rigid"
            ),
            state="source_backed",
            evidence_ids=evidence_ids,
        )
        for owner in owners
        for member in members_by_owner[owner]
    )
    joints = tuple(
        EmbeddedArticulationJoint(
            joint_id=f"motion_{index:03d}",
            body0_owner_prim=owners[0],
            body1_owner_prim=owner,
            body0_role="base",
            body1_role="moving_member",
            role_state="known",
            joint_type="revolute",
            endpoint_state="source_backed",
            type_state="known",
            axis="z",
            axis_state="known",
            lower_limit=-45.0,
            upper_limit=45.0,
            limit_unit="degrees",
            limit_state="source_backed",
            frame_policy="body1_world_origin",
            frame_state="known",
            evidence_ids=evidence_ids,
        )
        for index, owner in enumerate(owners[1:], start=1)
    )
    return EmbeddedArticulationCanonicalGraph(
        graph_id="synthetic-provider-neutral-graph",
        source_sha256=preparation.source_sha256,
        source_dependency_bundle_sha256=(preparation.source_dependency_bundle_sha256),
        source_member_prims=members,
        authoritative_owner_prims=owners,
        candidate_ids=tuple(item.joint_id for item in joints),
        groups=groups,
        memberships=memberships,
        joints=joints,
    )


def test_provider_neutral_preparation_public_api_has_interactive_batch_parity(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    observed: dict[str, tuple[str, bool, bool]] = {}
    for mode in ("interactive", "batch"):
        run_dir = tmp_path / f"{mode}-run"
        context = _provider_neutral_context(
            tmp_path,
            source=source,
            run_dir=run_dir,
        )
        result = prepare_embedded_articulation_workflow(
            _provider_neutral_request(source, run_dir, context),
            mode=mode,
            runtime=_provider_neutral_runtime(context, preparation),
            preparation=preparation,
        )
        state = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        observed[mode] = (
            result.status,
            state.embedded_evidence is not None,
            state.embedded_proposal is None,
        )
        assert state.phase == "awaiting_decision"
        assert state.inference_result is None
        assert state.candidate_document is None
    assert observed == {
        "interactive": ("awaiting_decision", True, True),
        "batch": ("awaiting_decision", True, True),
    }


def test_completed_evidence_rejects_missing_capability_record(tmp_path: Path) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    run_dir = tmp_path / "missing-capability-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    prepared = prepare_embedded_articulation_workflow(
        _provider_neutral_request(source, run_dir, context),
        mode="batch",
        runtime=_provider_neutral_runtime(context, preparation),
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    incomplete = evidence.model_copy(
        update={
            "records": tuple(
                record
                for record in evidence.records
                if record.evidence_id != "joint-authoring-capabilities"
            )
        }
    )

    with pytest.raises(ValueError, match="lacks capability facts"):
        _articulation_capabilities_from_evidence(incomplete)


def test_selected_proposal_leaf_without_outcome_is_not_evaluated_not_passed(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    unselected = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    pending_payload = unselected.model_dump(mode="json")
    pending_payload["proposal_status"] = "not_evaluated"
    selected_pending = EmbeddedArticulationPreparation.model_validate(pending_payload)
    run_dir = tmp_path / "selected-pending-proposal-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    result = prepare_embedded_articulation_workflow(
        _provider_neutral_request(source, run_dir, context),
        mode="batch",
        runtime=_provider_neutral_runtime(context, selected_pending),
        preparation=selected_pending,
    )
    assert result.status == "awaiting_decision"
    assert result.embedded_proposal_path is None
    assert result.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(result.embedded_evidence_path).read_bytes()
    )
    preparation_record = next(
        record
        for record in evidence.records
        if record.evidence_id == "articulation-preparation"
    )
    assert preparation_record.facts["proposal_status"] == "not_evaluated"
    assert unselected.proposal_status == "not_requested"


def test_provider_neutral_preparation_rejects_unavailable_evidence(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    ).model_copy(
        update={
            "renders": ProviderNeutralEvidenceRecord(
                evidence_id="joint-render-inspection",
                evidence_type="render",
                status="unavailable",
                summary="Required deterministic render evidence is unavailable.",
            )
        }
    )
    run_dir = tmp_path / "unavailable-evidence-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    patch = EmbeddedArticulationDecisionPatch(
        identity_digest=canonical_json_digest(runtime.identity),
        evidence_digest=artifact_reference(evidence).sha256,
        proposal_digest=None,
        canonical_graph=_synthetic_graph(
            preparation,
            members=members,
            owners=owners,
        ),
        outer_review_disposition="accept",
        human_review=EmbeddedArticulationHumanReviewPolicy(status="not_requested"),
        rationale="Unavailable evidence must block the outer graph.",
    )

    with pytest.raises(
        EmbeddedArticulationError,
        match="Required embedded articulation evidence is unavailable",
    ):
        apply_embedded_articulation_decision_patch(
            run_dir,
            patch,
            runtime=runtime,
        )


def test_provider_neutral_preparation_rejects_drifted_evidence(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    run_dir = tmp_path / "drifted-evidence-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None

    atomic_write_json(
        tmp_path / "synthetic_contract_evidence.json",
        {"changed": True},
    )
    with pytest.raises(EmbeddedArticulationError, match="changed after.*capture"):
        prepare_embedded_articulation_workflow(
            request,
            mode="interactive",
            runtime=runtime,
            preparation=preparation,
        )


def test_provider_neutral_preparation_preserves_legacy_hardlinked_source(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    run_dir = tmp_path / "legacy-hardlinked-source-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    os.link(source, tmp_path / "legacy-source-alias.usda")

    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )

    assert prepared.status == "awaiting_decision"


def test_provider_neutral_preparation_preserves_legacy_noncanonical_evidence_path(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    evidence_binding = preparation.source_hierarchy.artifacts[1]
    alias_parent = tmp_path / "legacy-evidence-alias-parent"
    alias_parent.mkdir()
    noncanonical_binding = evidence_binding.model_copy(
        update={"path": str(alias_parent / ".." / Path(evidence_binding.path).name)}
    )

    def preserve_path_form(
        record: ProviderNeutralEvidenceRecord,
    ) -> ProviderNeutralEvidenceRecord:
        return record.model_copy(
            update={
                "artifacts": tuple(
                    noncanonical_binding if item == evidence_binding else item
                    for item in record.artifacts
                )
            }
        )

    preparation = preparation.model_copy(
        update={
            "source_hierarchy": preserve_path_form(preparation.source_hierarchy),
            "source_members": preserve_path_form(preparation.source_members),
            "authoritative_owners": preserve_path_form(
                preparation.authoritative_owners
            ),
            "renders": preserve_path_form(preparation.renders),
            "scene": preserve_path_form(preparation.scene),
        }
    )
    run_dir = tmp_path / "legacy-noncanonical-evidence-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)

    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )

    assert prepared.status == "awaiting_decision"


def test_provider_neutral_187_source_62_owner_reaches_exact_outer_review_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=62,
        member_count=187,
    )
    run_dir = tmp_path / "provider-neutral-scale-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    preparation_path = tmp_path / "articulation_preparation.json"
    atomic_write_json(preparation_path, preparation)
    outer_state = tmp_path / "outer_state.json"
    outer_state.write_text("{}\n", encoding="utf-8")
    runtime = _provider_neutral_runtime(context, preparation)
    local_client_calls: list[object] = []

    def reject_local_client(*args: object, **kwargs: object) -> None:
        local_client_calls.append((args, kwargs))
        raise AssertionError("provider-neutral path instantiated JointAgentLocalClient")

    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_embedded_articulation_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        reject_local_client,
    )
    monkeypatch.setattr(
        articulation_runner,
        "load_embedded_articulation_human_acceptance",
        lambda *_args, **_kwargs: None,
    )
    config = articulation_runner.ArticulationRunConfig(
        repo_root=Path.cwd(),
        source_asset=source,
        output_dir=run_dir,
        joint_config=None,
        embedded_preparation=preparation_path,
        intent="Author the complete provider-neutral synthetic graph.",
        review_policy="all",
        expected_candidate_count=61,
        max_candidate_count=64,
        embedded_run_state=outer_state,
        runner="codex",
    )

    prepared = articulation_runner.run_articulation_workflow(config)

    assert prepared.status == "awaiting_decision"
    assert prepared.embedded_proposal_path is None
    assert local_client_calls == []
    assert not (run_dir / "inference_result.json").exists()
    assert not (run_dir / "articulation_candidates.json").exists()
    observation = json.loads(
        (run_dir / "articulation_agent_observation.json").read_text(encoding="utf-8")
    )
    assert observation["proposal_digest"] is None
    assert len(observation["source_member_prims"]) == 187
    assert len(observation["authoritative_owner_prims"]) == 62
    launcher = json.loads(
        (run_dir / "articulation_agent_launcher.json").read_text(encoding="utf-8")
    )
    assert launcher["config"]["runner"] == "codex"
    assert launcher["config"]["joint_config"] is None

    graph = _synthetic_graph(
        preparation,
        members=members,
        owners=owners,
    )
    patch_path = run_dir / "provider_neutral_graph_patch.json"
    atomic_write_json(
        patch_path,
        EmbeddedArticulationDecisionPatch(
            identity_digest=observation["identity_digest"],
            evidence_digest=observation["evidence_digest"],
            proposal_digest=None,
            canonical_graph=graph,
            outer_review_disposition="revise",
            human_review=EmbeddedArticulationHumanReviewPolicy(status="not_requested"),
            rationale="Outer coordinator reviewed complete synthetic graph scope.",
            revision_requests=("Retain the scale review without authoring.",),
        ),
    )
    gated = articulation_runner.apply_articulation_agent_step(run_dir, patch_path)
    state = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )

    assert gated.status == "conditional"
    assert state.phase == "conditional"
    assert state.embedded_proposal is None
    assert state.embedded_canonical_graph is not None
    assert state.review_required_candidate_ids == ()
    assert state.unresolved_candidate_ids == graph.candidate_ids
    assert len(state.unresolved_candidate_ids) == 61
    assert state.embedded_outer_review is not None
    assert state.embedded_human_decision is None
    assert not (run_dir / "authoring_request.json").exists()
    assert local_client_calls == []


def test_provider_neutral_outer_only_review_completes_without_human_or_proposal(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "provider-neutral-outer-only-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    graph = _synthetic_graph(
        preparation,
        members=members,
        owners=owners,
    )
    decision_patch = EmbeddedArticulationDecisionPatch(
        identity_digest=canonical_json_digest(runtime.identity),
        evidence_digest=artifact_reference(evidence).sha256,
        proposal_digest=None,
        canonical_graph=graph,
        outer_review_disposition="accept",
        human_review=EmbeddedArticulationHumanReviewPolicy(status="not_requested"),
        rationale="Outer organizer accepted the exact synthetic graph.",
    )
    applied = apply_embedded_articulation_decision_patch(
        run_dir,
        decision_patch,
        runtime=runtime,
    )
    assert (
        apply_embedded_articulation_decision_patch(
            run_dir,
            decision_patch,
            runtime=runtime,
        )
        == applied
    )
    with pytest.raises(
        EmbeddedArticulationError,
        match="different outer graph review",
    ):
        apply_embedded_articulation_decision_patch(
            run_dir,
            decision_patch.model_copy(
                update={"rationale": "A different exact outer review."}
            ),
            runtime=runtime,
        )
    human_loader_calls: list[object] = []

    def reject_human_loader(*args: object, **kwargs: object) -> None:
        human_loader_calls.append((args, kwargs))
        raise AssertionError("not_requested path called a human-review loader")

    client = _ProviderNeutralAuthoringClient()
    authored = run_embedded_articulation_workflow(
        request,
        mode="interactive",
        client=client,
        runtime=runtime,
        human_acceptance_loader=reject_human_loader,
        preparation=preparation,
    )

    assert authored.status == "needs_review"
    assert authored.embedded_human_decision_path is None
    assert authored.embedded_proposal_path is None
    assert authored.embedded_outer_review_path is not None
    assert human_loader_calls == []
    assert len(client.author_requests) == 1
    assert client.author_requests[0].predictions_path is None
    outer_review = EmbeddedArticulationOuterReview.model_validate_json(
        Path(authored.embedded_outer_review_path).read_bytes()
    )
    assert outer_review.disposition == "accept"
    assert outer_review.reviewer == runtime.outer_coordinator
    assert outer_review.human_review.status == "not_requested"
    assert authored.embedded_execution_result_path is not None
    result_artifact = EmbeddedBoundedExecutionResult.model_validate_json(
        Path(authored.embedded_execution_result_path).read_bytes()
    )
    apply_embedded_articulation_post_review(
        run_dir,
        EmbeddedArticulationPostReviewPatch(
            execution_result_digest=artifact_reference(result_artifact).sha256,
            accepted_decision_digest=result_artifact.accepted_decision_digest,
            disposition="accept",
            findings=("Exact outer graph and saved-stage readback match.",),
        ),
        runtime=runtime,
    )
    completed = run_embedded_articulation_workflow(
        request,
        mode="interactive",
        client=client,
        runtime=runtime,
        human_acceptance_loader=reject_human_loader,
        preparation=preparation,
    )

    assert completed.status == "completed"
    assert completed.embedded_decision_receipt_path is not None
    assert len(client.author_requests) == 1
    receipt = EmbeddedDecisionReceipt.model_validate_json(
        Path(completed.embedded_decision_receipt_path).read_bytes()
    )
    assert receipt.human_decision is None
    assert receipt.proposal_providers == ()


def test_current_outer_native_completion_binds_visual_review_and_projects_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "output-bound-outer-native-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
        canonical_output_evidence_required=True,
    )
    runtime = _provider_neutral_runtime(
        context,
        preparation,
        canonical_output_evidence_required=True,
    )
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="batch",
        runtime=runtime,
        preparation=preparation,
    )
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()  # type: ignore[arg-type]
    )
    graph = _synthetic_graph(preparation, members=members, owners=owners)
    apply_embedded_articulation_decision_patch(
        run_dir,
        EmbeddedArticulationDecisionPatch(
            identity_digest=canonical_json_digest(runtime.identity),
            evidence_digest=artifact_reference(evidence).sha256,
            proposal_digest=None,
            canonical_graph=graph,
            outer_review_disposition="accept",
            human_review=EmbeddedArticulationHumanReviewPolicy(status="not_requested"),
            rationale="Outer organizer accepted the exact output-bound graph.",
        ),
        runtime=runtime,
    )
    client = _ProviderNeutralAuthoringClient()
    authored = run_embedded_articulation_workflow(
        request,
        mode="batch",
        client=client,
        runtime=runtime,
        human_acceptance_loader=lambda _state: None,
        preparation=preparation,
    )
    assert authored.status == "needs_review"
    state = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    assert state.authoring_result is not None
    authoring = ArticulationAuthoringResult.model_validate_json(
        Path(state.authoring_result.path).read_bytes()
    )
    output = Path(authoring.output_asset_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(visual, "_usd_dependency_bindings", lambda _path: ())

    def render_output(**kwargs: object) -> visual._UsdCliRenderOutcome:
        root = Path(str(kwargs["root"]))
        request = kwargs["request"]
        assert isinstance(request, visual.CanonicalVisualEvidenceRequest)
        image = root / "usd_cli" / "renders" / "canonical_output.png"
        response = image.with_name("canonical_output_response.json")
        camera = image.with_name("canonical_output_camera.json")
        receipt = root / "usd_cli" / "raw" / "usd_cli_command_receipts.jsonl"
        checkpoint = (
            root / "usd_cli" / "raw" / "usd_cli_command_receipts.checkpoint.json"
        )
        image.parent.mkdir(parents=True)
        receipt.parent.mkdir(parents=True)
        image.write_bytes(b"synthetic canonical render bytes")
        response.write_text('{"ok": true}', encoding="utf-8")
        camera.write_text('{"direction": "+x+y+z"}', encoding="utf-8")
        receipt.write_text('{"tool": "usd-cli"}\n', encoding="utf-8")
        checkpoint.write_text('{"receipt": "bound"}', encoding="utf-8")
        image_binding = execution_artifact_binding(image)
        response_binding = execution_artifact_binding(response)
        camera_binding = execution_artifact_binding(camera)
        receipt_binding = execution_artifact_binding(receipt)
        checkpoint_binding = execution_artifact_binding(checkpoint)
        metadata = {
            "backend": "ovrtx",
            "renderer": "ovrtx",
            "scene_tool": "usd-cli",
            "scene_tool_source_revision": "a" * 40,
            "session_id": "workflow-articulation-test",
            "workflow": "validation-canonical-visual-evidence",
            "views": list(request.views),
            "image_width": request.image_width,
            "image_height": request.image_height,
            "renderer_identities": [None],
        }
        report = {
            "schema_version": (
                "content-agent-workflows.canonical-visual-usd-cli-render-report.v1"
            ),
            "status": "completed",
            "backend": "ovrtx",
            "probe_schema_version": "usd-cli.render-probe.v1",
            "image_paths": [str(image)],
            "images": [image_binding.model_dump(mode="json")],
            "render_responses": [response_binding.model_dump(mode="json")],
            "camera_records": [camera_binding.model_dump(mode="json")],
            "usd_cli_command_receipt": receipt_binding.model_dump(mode="json"),
            "usd_cli_receipt_checkpoint": checkpoint_binding.model_dump(mode="json"),
            "metadata": metadata,
        }
        return visual._UsdCliRenderOutcome(
            report=report,
            images=(image_binding,),
            responses=(response_binding,),
            cameras=(camera_binding,),
            receipt=receipt_binding,
            checkpoint=checkpoint_binding,
            contract_path=Path(visual.__file__).resolve(),
        )

    monkeypatch.setattr(
        visual,
        "_render_with_package_owned_usd_cli",
        render_output,
    )
    authored_bytes = output.read_bytes()
    output.write_text('#usda 1.0\ndef Xform "Substituted" {}\n', encoding="utf-8")
    substituted_visual = produce_canonical_visual_evidence(
        source_usd=source,
        post_mutation_usd=output,
        output_dir=tmp_path / "substituted-visual-publication",
        backend="ovrtx",
    )
    with pytest.raises(
        EmbeddedArticulationError,
        match="exact authored readback bytes",
    ):
        bind_embedded_articulation_output_evidence(
            run_dir,
            substituted_visual.envelope.path,
        )
    output.write_bytes(authored_bytes)
    visual_publication = produce_canonical_visual_evidence(
        source_usd=source,
        post_mutation_usd=output,
        output_dir=tmp_path / "canonical-visual-publication",
        backend="ovrtx",
    )
    with pytest.raises(VerifiedOperationError, match="invalid.*envelope"):
        bind_embedded_articulation_output_evidence(
            run_dir,
            visual_publication.payload.path,
        )
    bound = bind_embedded_articulation_output_evidence(
        run_dir,
        visual_publication.envelope.path,
    )
    assert bound.embedded_output_evidence is not None
    output_evidence = EmbeddedArticulationOutputEvidence.model_validate_json(
        Path(bound.embedded_output_evidence.path).read_bytes()
    )
    result_artifact = EmbeddedBoundedExecutionResult.model_validate_json(
        Path(authored.embedded_execution_result_path).read_bytes()  # type: ignore[arg-type]
    )
    review_binding = bound.embedded_output_evidence.model_dump(mode="json")
    for field_name in (
        "embedded_outer_review",
        "embedded_output_evidence",
        "embedded_terminal_receipt",
    ):
        legacy_state_payload = bound.model_dump(mode="json")
        legacy_state_payload.update(
            {
                "schema_version": ("content-agent-workflows.articulation-run-state.v2"),
                "embedded_outer_review": None,
                "embedded_output_evidence": None,
                "embedded_terminal_receipt": None,
                field_name: review_binding,
            }
        )
        with pytest.raises(ValueError, match="require run-state v3"):
            ArticulationRunState.model_validate(legacy_state_payload)

    mismatched_runtime = runtime.model_copy(
        update={"capabilities": EmbeddedArticulationCapabilityLimits()}
    )
    with pytest.raises(
        EmbeddedArticulationError,
        match="runtime capabilities differ from persisted evidence",
    ):
        apply_embedded_articulation_post_review(
            run_dir,
            EmbeddedArticulationPostReviewPatch(
                schema_version=EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION,
                execution_result_digest=artifact_reference(result_artifact).sha256,
                accepted_decision_digest=result_artifact.accepted_decision_digest,
                output_evidence_digest=bound.embedded_output_evidence.sha256,
                inspected_render_image_sha256s=tuple(
                    item.sha256 for item in output_evidence.images
                ),
                disposition="accept",
                findings=("Runtime capability substitution must fail closed.",),
            ),
            runtime=mismatched_runtime,
        )
    image = Path(output_evidence.images[0].path)
    image.write_bytes(b"substituted visual bytes")
    with pytest.raises(VerifiedOperationError, match="binding is stale"):
        apply_embedded_articulation_post_review(
            run_dir,
            EmbeddedArticulationPostReviewPatch(
                schema_version=EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION,
                execution_result_digest=artifact_reference(result_artifact).sha256,
                accepted_decision_digest=result_artifact.accepted_decision_digest,
                output_evidence_digest=bound.embedded_output_evidence.sha256,
                inspected_render_image_sha256s=tuple(
                    item.sha256 for item in output_evidence.images
                ),
                disposition="accept",
                findings=("Substituted image must not pass review.",),
            ),
            runtime=runtime,
        )
    image.write_bytes(b"synthetic canonical render bytes")
    completed_state = apply_embedded_articulation_post_review(
        run_dir,
        EmbeddedArticulationPostReviewPatch(
            schema_version=EMBEDDED_ARTICULATION_POST_REVIEW_SCHEMA_VERSION,
            execution_result_digest=artifact_reference(result_artifact).sha256,
            accepted_decision_digest=result_artifact.accepted_decision_digest,
            output_evidence_digest=bound.embedded_output_evidence.sha256,
            inspected_render_image_sha256s=tuple(
                item.sha256 for item in output_evidence.images
            ),
            disposition="accept",
            findings=("Exact graph, output, and OVRTX image were reviewed.",),
        ),
        runtime=runtime,
    )
    assert completed_state.phase == "completed"
    assert completed_state.embedded_terminal_receipt is not None
    terminal = EmbeddedArticulationTerminalReceipt.model_validate_json(
        Path(completed_state.embedded_terminal_receipt.path).read_bytes()
    )
    assert terminal.output_evidence == bound.embedded_output_evidence
    assert terminal.images == output_evidence.images

    publication = project_joint_graph_apply_result(
        run_dir,
        output_dir=tmp_path / "graph-apply-projection",
    )
    assert (
        publication.shared_contract_checkpoint == VERIFIED_OPERATION_CONTRACT_CHECKPOINT
    )
    assert verify_operation_envelope(publication.result) == publication.result
    receipt = ingest_verified_operation_result(
        publication.envelope.path,
        output_dir=tmp_path / "shared-validation-ingress",
    )
    assert receipt.execution_mode == "provided"
    assert receipt.envelope.native_status == "pass"
    assert receipt.envelope.output.sha256 == file_sha256(output)


def test_exact_outer_review_retry_recovers_partial_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "partial-outer-review-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    patch = EmbeddedArticulationDecisionPatch(
        identity_digest=canonical_json_digest(runtime.identity),
        evidence_digest=artifact_reference(evidence).sha256,
        proposal_digest=None,
        canonical_graph=_synthetic_graph(
            preparation,
            members=members,
            owners=owners,
        ),
        outer_review_disposition="accept",
        human_review=EmbeddedArticulationHumanReviewPolicy(status="not_requested"),
        rationale="Exact retry must recover the persisted outer review.",
    )
    original_append = EmbeddedDecisionArtifactStore.append

    def interrupt_decision_append(
        store: EmbeddedDecisionArtifactStore,
        artifact: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(artifact, EmbeddedCoordinatorDecision):
            raise RuntimeError("simulated interruption before decision commit")
        return original_append(store, artifact, **kwargs)

    monkeypatch.setattr(
        EmbeddedDecisionArtifactStore,
        "append",
        interrupt_decision_append,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        apply_embedded_articulation_decision_patch(run_dir, patch, runtime=runtime)
    monkeypatch.setattr(EmbeddedDecisionArtifactStore, "append", original_append)

    interrupted = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    assert interrupted.embedded_coordinator_decision is None
    assert (run_dir / "canonical_articulation_graph.json").is_file()
    assert (run_dir / "embedded_articulation_outer_review.json").is_file()
    with pytest.raises(
        EmbeddedArticulationError,
        match="Outer graph review path contains different bytes",
    ):
        apply_embedded_articulation_decision_patch(
            run_dir,
            patch.model_copy(update={"rationale": "Conflicting retry."}),
            runtime=runtime,
        )

    recovered = apply_embedded_articulation_decision_patch(
        run_dir,
        patch,
        runtime=runtime,
    )
    assert recovered.embedded_coordinator_decision is not None
    assert recovered.embedded_outer_review is not None


def test_legacy_decision_patch_still_requires_provider_proposal_digest() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="v1.*require.*proposal"):
        EmbeddedArticulationDecisionPatch(
            schema_version=(
                "content-agent-workflows.embedded-articulation-decision-patch.v1"
            ),
            identity_digest="1" * 64,
            evidence_digest="2" * 64,
            proposal_digest=None,
            canonical_graph=graph,
            rationale="Invalid legacy patch without proposal provenance.",
        )


def test_v2_optional_proposal_patch_preserves_human_required_compatibility() -> None:
    patch = EmbeddedArticulationDecisionPatch(
        schema_version=(
            "content-agent-workflows.embedded-articulation-decision-patch.v2"
        ),
        identity_digest="1" * 64,
        evidence_digest="2" * 64,
        proposal_digest=None,
        canonical_graph=_graph(),
        rationale="Legacy optional-proposal artifact.",
    )

    assert patch.effective_outer_review_disposition == "accept"
    assert patch.effective_human_review.status == "human_required"
    assert patch.effective_human_review.reasons == ("legacy_compatibility",)

    invalid_v3 = patch.model_copy(
        update={
            "schema_version": (
                "content-agent-workflows.embedded-articulation-decision-patch.v3"
            )
        }
    )
    with pytest.raises(ValueError, match="v3.*require.*outer review"):
        _ = invalid_v3.effective_outer_review_disposition
    with pytest.raises(ValueError, match="v3.*require.*human review"):
        _ = invalid_v3.effective_human_review


def test_v2_checkpoint_resume_still_requires_exact_human_acceptance(
    tmp_path: Path,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "legacy-v2-human-gate-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    preparation = _provider_neutral_preparation(
        tmp_path,
        source=source,
        members=members,
        owners=owners,
    )
    runtime = _provider_neutral_runtime(context, preparation)
    request = _provider_neutral_request(source, run_dir, context)
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="interactive",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    graph = _synthetic_graph(
        preparation,
        members=members,
        owners=owners,
    )
    current = apply_embedded_articulation_decision_patch(
        run_dir,
        EmbeddedArticulationDecisionPatch(
            schema_version=(
                "content-agent-workflows.embedded-articulation-decision-patch.v2"
            ),
            identity_digest=canonical_json_digest(runtime.identity),
            evidence_digest=artifact_reference(evidence).sha256,
            proposal_digest=None,
            canonical_graph=graph,
            rationale="Legacy v2 checkpoint requires exact human acceptance.",
        ),
        runtime=runtime,
    )
    legacy = current.model_copy(
        update={
            "schema_version": "content-agent-workflows.articulation-run-state.v2",
            "embedded_outer_review": None,
        }
    )
    atomic_write_json(run_dir / "checkpoint.json", legacy)
    assert legacy.embedded_coordinator_decision is not None
    decision = EmbeddedCoordinatorDecision.model_validate_json(
        Path(legacy.embedded_coordinator_decision.path).read_bytes()
    )
    assert decision.human_decision_required

    client = _ProviderNeutralAuthoringClient()
    blocked = run_embedded_articulation_workflow(
        request,
        mode="interactive",
        client=client,
        runtime=runtime,
        human_acceptance_loader=lambda *_args, **_kwargs: None,
        preparation=preparation,
    )
    assert blocked.status == "needs_review"
    assert client.author_requests == []
    assert not (run_dir / "authoring_request.json").exists()

    decisions_path = run_dir / "legacy_human_decisions.json"
    decisions = {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    atomic_write_json(decisions_path, decisions)
    acceptance = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions_path),
        decisions_sha256=file_sha256(decisions_path),
        reviewer="legacy-human-reviewer",
        decisions=decisions,
    )
    authored = run_embedded_articulation_workflow(
        request,
        mode="interactive",
        client=client,
        runtime=runtime,
        human_acceptance_loader=lambda *_args, **_kwargs: acceptance,
        preparation=preparation,
    )
    assert authored.status == "needs_review"
    assert len(client.author_requests) == 1
    assert client.author_requests[0].predictions_path is None


def test_provider_backed_controller_requires_provider_client(
    tmp_path: Path,
) -> None:
    source, _members, _owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "provider-client-type-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    request = _provider_neutral_request(source, run_dir, context)
    outer_state = tmp_path / "outer_state.json"
    outer_state.write_text("{}\n", encoding="utf-8")
    config = articulation_runner.ArticulationRunConfig(
        repo_root=Path.cwd(),
        source_asset=source,
        output_dir=run_dir,
        joint_config=tmp_path / "joint.yaml",
        intent="Reject an authoring-only provider-backed client.",
        embedded_run_state=outer_state,
    )

    with pytest.raises(ValueError, match="requires a JointAgentLocalClient"):
        articulation_runner._embedded_articulation_controller(
            request,
            config=config,
            mode="batch",
            client=_ProviderNeutralAuthoringClient(),
        )


def test_provider_backed_controller_rejects_separate_provider_client(
    tmp_path: Path,
) -> None:
    source, _members, _owners = _synthetic_source(
        tmp_path,
        owner_count=2,
        member_count=5,
    )
    run_dir = tmp_path / "split-provider-client-run"
    context = _provider_neutral_context(
        tmp_path,
        source=source,
        run_dir=run_dir,
    )
    request = _provider_neutral_request(source, run_dir, context)
    outer_state = tmp_path / "outer_state.json"
    outer_state.write_text("{}\n", encoding="utf-8")
    config = articulation_runner.ArticulationRunConfig(
        repo_root=Path.cwd(),
        source_asset=source,
        output_dir=run_dir,
        joint_config=tmp_path / "joint.yaml",
        intent="Reject separate provider-backed inference and authoring clients.",
        embedded_run_state=outer_state,
    )
    authoring_only = _ProviderNeutralAuthoringClient()

    with pytest.raises(
        ValueError,
        match="must use one client for inference and authoring",
    ):
        articulation_runner._embedded_articulation_controller(
            request,
            config=config,
            mode="batch",
            client=authoring_only,
            provider_client=_ProviderNeutralAuthoringClient(),  # type: ignore[arg-type]
        )


def _configured_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exact_graph_match: bool = True,
    omit_co_rigid_member: bool = False,
) -> tuple[
    articulation_runner.ArticulationRunConfig,
    _GraphOnlyClient,
    dict[str, EmbeddedArticulationHumanAcceptance | None],
]:
    source = tmp_path / "neutral_structure.usda"
    source.write_text(
        """#usda 1.0
def Xform "Assembly" {
    def Xform "RootUnit" {}
    def Xform "MovingUnitA" {
        def Xform "AttachedPiece" {}
    }
    def Xform "MovingUnitB" {}
}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "joint.yaml"
    config_path.write_text("project:\n  name: neutral\n", encoding="utf-8")
    predictions = tmp_path / "provider_predictions.json"
    atomic_write_json(
        predictions,
        {
            "malicious_membership": "/Provider/Override",
            "joint_type": "prismatic",
            "axis": "x",
        },
    )
    outer = tmp_path / "asset_run.json"
    outer.write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    outer_request = tmp_path / "asset_request.json"
    outer_request.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "domain-run"
    context = DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="outer-neutral-run",
            outer_request=_binding(outer_request),
            stage="articulation",
            stage_attempt=1,
            coordinator_plan=_binding(plan),
            input_asset=_binding(source),
            domain_run_root=str(run_dir.resolve()),
        ),
    )
    runtime = _runtime(context)
    client = _GraphOnlyClient(
        predictions,
        exact_graph_match=exact_graph_match,
        omit_co_rigid_member=omit_co_rigid_member,
    )
    human: dict[str, EmbeddedArticulationHumanAcceptance | None] = {"value": None}
    monkeypatch.setattr(
        articulation_runner,
        "build_embedded_domain_execution_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_embedded_articulation_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    monkeypatch.setattr(
        articulation_runner,
        "JointAgentLocalClient",
        lambda *_args, **_kwargs: client,
    )
    collector = MockArticulationSceneEvidenceCollector()
    monkeypatch.setattr(
        articulation_runner,
        "_build_live_evidence_collector",
        lambda *_args, **_kwargs: collector,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_scene_evidence_is_pending",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        articulation_runner,
        "load_embedded_articulation_human_acceptance",
        lambda *_args, **_kwargs: human["value"],
    )

    def load_outer_review_snapshot(_path: Path) -> SimpleNamespace:
        checkpoint = ArticulationRunState.model_validate_json(
            (run_dir / "checkpoint.json").read_bytes()
        )
        acceptance = human["value"]
        assert checkpoint.embedded_canonical_graph is not None
        assert acceptance is not None
        candidate_path = Path(checkpoint.embedded_canonical_graph.path)
        return SimpleNamespace(
            stages={
                "articulation": SimpleNamespace(
                    review_candidates=_binding(candidate_path),
                    review_decisions=_binding(Path(acceptance.decisions_path)),
                )
            }
        )

    monkeypatch.setattr(
        articulation_runner,
        "load_verified_run",
        load_outer_review_snapshot,
    )
    return (
        articulation_runner.ArticulationRunConfig(
            repo_root=Path.cwd(),
            source_asset=source,
            output_dir=run_dir,
            joint_config=config_path,
            intent="Build a complete reviewed articulation graph.",
            review_policy="all",
            embedded_run_state=outer,
        ),
        client,
        human,
    )


def _apply_graph(
    config: articulation_runner.ArticulationRunConfig,
) -> EmbeddedArticulationCanonicalGraph:
    observation = json.loads(
        (config.output_dir / "articulation_agent_observation.json").read_text()
    )
    checkpoint = json.loads((config.output_dir / "checkpoint.json").read_text())
    graph = _graph().model_copy(
        update={
            "source_sha256": checkpoint["source_sha256"],
            "source_dependency_bundle_sha256": checkpoint[
                "source_dependency_bundle_sha256"
            ],
        }
    )
    patch_path = config.output_dir / "canonical_graph_patch.json"
    atomic_write_json(
        patch_path,
        EmbeddedArticulationDecisionPatch(
            identity_digest=observation["identity_digest"],
            evidence_digest=observation["evidence_digest"],
            proposal_digest=observation["proposal_digest"],
            canonical_graph=graph,
            outer_review_disposition="accept",
            human_review=EmbeddedArticulationHumanReviewPolicy(
                status="human_required",
                reasons=("task_policy",),
            ),
            rationale="Outer coordinator resolved complete neutral topology.",
        ),
    )
    articulation_runner.apply_articulation_agent_step(config.output_dir, patch_path)
    return graph


def test_real_embedded_entrypoint_blocks_proposal_and_prediction_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(tmp_path, monkeypatch)

    prepared = articulation_runner.run_articulation_workflow(config)
    assert prepared.status == "needs_review"
    graph = _apply_graph(config)
    assert client.author_call_count == 0

    decisions = config.output_dir / "human_decisions.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    authored = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )

    assert authored.status == "needs_review"
    assert (
        authored.schema_version
        == ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION
    )
    assert (
        json.loads((config.output_dir / "checkpoint.json").read_text())[
            "schema_version"
        ]
        == ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION
    )
    assert authored.embedded_decision_receipt_path is None
    assert client.author_call_count == 1
    request = client.author_requests[0]
    assert request.predictions_path is None
    assert request.predictions_sha256 is None
    accepted = Stage2CandidateDocument.model_validate_json(
        Path(request.candidate_document_path).read_bytes()
    )
    assert all(item.motion_type == "revolute" for item in accepted.candidates)
    assert all(item.axis_hint == "z" for item in accepted.candidates)
    assert all(item.source_prediction_ids == () for item in accepted.candidates)
    assert "/Provider/Override" not in Path(request.candidate_document_path).read_text()
    assert authored.embedded_readback_path is not None
    readback = EmbeddedArticulationReadback.model_validate_json(
        Path(authored.embedded_readback_path).read_bytes()
    )
    assert readback.memberships == tuple(
        item.model_dump(mode="json") for item in graph.memberships
    )
    assert readback.groups == tuple(
        item.model_dump(mode="json") for item in graph.groups
    )
    assert readback.exact_membership_match
    assert readback.exact_co_rigid_disposition_match

    result_artifact = EmbeddedBoundedExecutionResult.model_validate_json(
        Path(authored.embedded_execution_result_path).read_bytes()
    )
    post_patch = config.output_dir / "post_review_patch.json"
    atomic_write_json(
        post_patch,
        EmbeddedArticulationPostReviewPatch(
            execution_result_digest=artifact_reference(result_artifact).sha256,
            accepted_decision_digest=result_artifact.accepted_decision_digest,
            disposition="accept",
            findings=("Exact graph and saved membership readback match.",),
        ),
    )
    completed = articulation_runner.finalize_articulation_agent_step(
        config.output_dir, post_patch
    )
    assert completed.status == "completed"
    assert (
        completed.schema_version
        == ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION
    )
    assert completed.embedded_decision_receipt_path is not None


def test_incomplete_coverage_blocks_before_human_or_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, _human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    observation = json.loads(
        (config.output_dir / "articulation_agent_observation.json").read_text()
    )
    checkpoint = json.loads((config.output_dir / "checkpoint.json").read_text())
    graph = _graph().model_copy(
        update={
            "source_sha256": checkpoint["source_sha256"],
            "source_dependency_bundle_sha256": checkpoint[
                "source_dependency_bundle_sha256"
            ],
            "source_member_prims": ("/Assembly/RootUnit",),
        }
    )
    patch_path = config.output_dir / "incomplete_patch.json"
    atomic_write_json(
        patch_path,
        EmbeddedArticulationDecisionPatch(
            identity_digest=observation["identity_digest"],
            evidence_digest=observation["evidence_digest"],
            proposal_digest=observation["proposal_digest"],
            canonical_graph=graph,
            outer_review_disposition="accept",
            human_review=EmbeddedArticulationHumanReviewPolicy(
                status="human_required",
                reasons=("task_policy",),
            ),
            rationale="Incomplete on purpose.",
        ),
    )
    with pytest.raises(EmbeddedArticulationError, match="source-member coverage"):
        articulation_runner.apply_articulation_agent_step(config.output_dir, patch_path)
    assert client.author_call_count == 0


def test_duplicate_moving_owner_blocks_before_human_or_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, _human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    observation = json.loads(
        (config.output_dir / "articulation_agent_observation.json").read_text()
    )
    checkpoint = json.loads((config.output_dir / "checkpoint.json").read_text())
    graph = _graph().model_copy(
        update={
            "source_sha256": checkpoint["source_sha256"],
            "source_dependency_bundle_sha256": checkpoint[
                "source_dependency_bundle_sha256"
            ],
            "joints": (
                _graph().joints[0],
                _graph()
                .joints[1]
                .model_copy(update={"body1_owner_prim": "/Assembly/MovingUnitA"}),
            ),
        }
    )
    patch_path = config.output_dir / "duplicate_moving_owner_patch.json"
    atomic_write_json(
        patch_path,
        EmbeddedArticulationDecisionPatch(
            identity_digest=observation["identity_digest"],
            evidence_digest=observation["evidence_digest"],
            proposal_digest=observation["proposal_digest"],
            canonical_graph=graph,
            outer_review_disposition="accept",
            human_review=EmbeddedArticulationHumanReviewPolicy(
                status="human_required",
                reasons=("task_policy",),
            ),
            rationale="Duplicate moving owner on purpose.",
        ),
    )
    with pytest.raises(EmbeddedArticulationError, match="one moving owner"):
        articulation_runner.apply_articulation_agent_step(config.output_dir, patch_path)
    assert client.author_call_count == 0


@pytest.mark.parametrize("disposition", ["reject", "revise"])
def test_human_reject_or_revise_terminally_blocks_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    config, client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / f"human_{disposition}.json"
    atomic_write_json(
        decisions, {candidate_id: disposition for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: disposition for candidate_id in graph.candidate_ids},
    )
    result = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    assert result.status == "conditional"
    assert client.author_call_count == 0
    observation = json.loads(
        (config.output_dir / "articulation_agent_observation.json").read_text()
    )
    assert (
        observation["schema_version"]
        == "content-workflow-cli.embedded-articulation-observation.v4"
    )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.embedded_human_decision is not None
    if disposition == "revise":
        assert observation["graph_revision_inputs"]["human_decisions"] == (
            _binding(decisions).model_dump(mode="json")
        )
        assert observation["graph_revision_inputs"]["human_decision"] == (
            checkpoint.embedded_human_decision.model_dump(mode="json")
        )
    else:
        assert observation["graph_revision_inputs"] is None


def _prepare_graph_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    articulation_runner.ArticulationRunConfig,
    _GraphOnlyClient,
    dict[str, EmbeddedArticulationHumanAcceptance | None],
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationGraphRevisionPatch,
    EmbeddedArticulationRuntime,
]:
    config, client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    decisions[graph.candidate_ids[0]] = "revise"
    review_source = config.output_dir / "human_revise_one_limit.json"
    atomic_write_json(review_source, decisions)
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    decisions_path = review_dir / f"articulation-{file_sha256(review_source)}.json"
    atomic_write_json(decisions_path, decisions)
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions_path),
        decisions_sha256=file_sha256(decisions_path),
        reviewer="human-reviewer",
        decisions=decisions,
    )
    conditional = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions_path,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    assert conditional.status == "conditional"
    state = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert state.embedded_evidence is not None
    assert state.embedded_canonical_graph is not None
    assert state.embedded_human_decision is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(state.embedded_evidence.path).read_bytes()
    )
    proposal = (
        EmbeddedDomainProposal.model_validate_json(
            Path(state.embedded_proposal.path).read_bytes()
        )
        if state.embedded_proposal is not None
        else None
    )
    human_decision = EmbeddedHumanDecision.model_validate_json(
        Path(state.embedded_human_decision.path).read_bytes()
    )
    revised_joint = graph.joints[0].model_copy(update={"upper_limit": 46.0})
    revised_graph = graph.model_copy(
        update={"joints": (revised_joint, *graph.joints[1:])}
    )
    request = articulation_runner._load_request(config.output_dir)
    runtime = articulation_runner._embedded_articulation_runtime(
        request,
        config,
        client,
    )
    patch = EmbeddedArticulationGraphRevisionPatch(
        expected_state_revision=state.revision,
        identity_digest=canonical_json_digest(runtime.identity),
        evidence_digest=artifact_reference(evidence).sha256,
        proposal_digest=(artifact_reference(proposal).sha256 if proposal else None),
        parent_canonical_graph_sha256=state.embedded_canonical_graph.sha256,
        parent_canonical_graph_digest=canonical_json_digest(graph),
        human_decision_sha256=state.embedded_human_decision.sha256,
        human_decisions=_binding(decisions_path),
        reviewer=human_decision.producer.producer_id,
        revision_reason="Human review raised the upper limit for motion_a.",
        requested_at=datetime.now(UTC),
        changes=(
            EmbeddedArticulationGraphChange(
                candidate_id=graph.candidate_ids[0],
                field="upper_limit",
                previous_value=graph.joints[0].upper_limit,
                revised_value=revised_joint.upper_limit,
            ),
        ),
        revised_graph=revised_graph,
        revised_graph_digest=canonical_json_digest(revised_graph),
    )
    return config, client, human, graph, patch, runtime


def test_graph_revision_preserves_parent_and_requires_complete_new_digest_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human, graph, patch, runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    before = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert before.embedded_canonical_graph is not None
    assert before.embedded_outer_review is not None
    assert before.embedded_coordinator_decision is not None
    assert before.embedded_human_decision is not None
    parent_bindings = (
        before.embedded_canonical_graph,
        before.embedded_outer_review,
        before.embedded_coordinator_decision,
        before.embedded_human_decision,
    )
    parent_bytes = tuple(Path(item.path).read_bytes() for item in parent_bindings)

    revised = apply_embedded_articulation_graph_revision(
        config.output_dir,
        patch,
        runtime=runtime,
    )

    assert revised.phase == "needs_review"
    assert revised.revision == patch.expected_state_revision + 1
    assert revised.embedded_human_decision is None
    assert revised.embedded_graph_revision is not None
    assert revised.embedded_canonical_graph is not None
    assert revised.embedded_canonical_graph.sha256 != parent_bindings[0].sha256
    legacy_state_payload = revised.model_dump(mode="json")
    legacy_state_payload["schema_version"] = (
        "content-agent-workflows.articulation-run-state.v3"
    )
    with pytest.raises(ValidationError, match="graph revision requires run-state v4"):
        ArticulationRunState.model_validate(legacy_state_payload)
    serialized_v3 = revised.model_copy(
        update={
            "schema_version": "content-agent-workflows.articulation-run-state.v3",
            "embedded_graph_revision": None,
        }
    ).model_dump(mode="json")
    assert "embedded_graph_revision" not in serialized_v3
    ArticulationRunState.model_validate(serialized_v3)
    assert (
        tuple(Path(item.path).read_bytes() for item in parent_bindings) == parent_bytes
    )
    record = EmbeddedArticulationGraphRevision.model_validate_json(
        Path(revised.embedded_graph_revision.path).read_bytes()
    )
    assert record.parent_canonical_graph == parent_bindings[0]
    assert record.revised_canonical_graph == revised.embedded_canonical_graph
    assert record.parent_state_revision == patch.expected_state_revision
    assert record.revision_patch_digest == canonical_json_digest(patch)
    assert record.changes == patch.changes
    assert record.requested_at == patch.requested_at
    assert record.created_at >= record.requested_at
    revised_decision = EmbeddedCoordinatorDecision.model_validate_json(
        Path(record.revised_coordinator_decision.path).read_bytes()
    )
    assert revised_decision.created_at == record.created_at

    asset_state_path = tmp_path / "replacement_asset_run.json"
    create_run(
        asset_state_path,
        run_id="graph-revision-outer-run",
        request_path=tmp_path / "asset_request.json",
        source_asset=config.source_asset,
        coordinator_mode="legacy",
    )
    begin_stage(asset_state_path, "articulation", actor="asset-coordinator")
    require_review(
        asset_state_path,
        candidates_path=parent_bindings[0].path,
        actor="asset-coordinator",
    )
    bound = record_review_decisions(
        asset_state_path,
        decisions_path=patch.human_decisions.path,
        reviewer="human-reviewer",
    )
    initial_attempt = bound.stages["articulation"].attempt_count
    asset_revised = record_articulation_graph_revision(
        asset_state_path,
        revision_receipt_path=revised.embedded_graph_revision.path,
        candidates_path=revised.embedded_canonical_graph.path,
        actor="asset-coordinator",
    )
    asset_stage = asset_revised.stages["articulation"]
    assert asset_stage.status == "needs_review"
    assert asset_stage.attempt_count == initial_attempt == 1
    assert asset_stage.review_candidates is not None
    assert (
        asset_stage.review_candidates.sha256 == revised.embedded_canonical_graph.sha256
    )
    assert asset_stage.review_decisions is None
    assert len(asset_stage.superseded_reviews) == 1
    assert load_verified_run(asset_state_path) == asset_revised
    legacy_asset_payload = asset_revised.model_dump(mode="json")
    legacy_asset_payload["schema_version"] = (
        "content-agent-workflows.asset-composition-run.v2"
    )
    with pytest.raises(ValidationError, match="superseded Articulation reviews"):
        AssetCompositionRun.model_validate(legacy_asset_payload)
    serialized_v2 = asset_revised.model_copy(
        update={"schema_version": "content-agent-workflows.asset-composition-run.v2"}
    ).model_dump(mode="json")
    assert "superseded_reviews" not in serialized_v2["stages"]["articulation"]
    AssetCompositionRun.model_validate(serialized_v2)

    with pytest.raises(EmbeddedArticulationError, match="graph digest is stale"):
        articulation_runner.resume_articulation_workflow(
            config.output_dir,
            repo_root=config.repo_root,
        )
    assert client.author_call_count == 0

    revised_graph = patch.revised_graph
    accepted_path = config.output_dir / "human_accept_revised_graph.json"
    accepted = {candidate_id: "accept" for candidate_id in revised_graph.candidate_ids}
    atomic_write_json(accepted_path, accepted)
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(revised_graph),
        decisions_path=str(accepted_path),
        decisions_sha256=file_sha256(accepted_path),
        reviewer="human-reviewer",
        decisions=accepted,
    )
    record_review_decisions(
        asset_state_path,
        decisions_path=accepted_path,
        reviewer="human-reviewer",
    )
    with pytest.raises(AssetCompositionStateError, match="not bound"):
        load_embedded_articulation_human_acceptance(
            asset_state_path,
            canonical_graph=parent_bindings[0].path,
        )
    asset_acceptance = load_embedded_articulation_human_acceptance(
        asset_state_path,
        canonical_graph=revised.embedded_canonical_graph.path,
    )
    assert asset_acceptance is not None
    assert asset_acceptance.canonical_graph_sha256 == canonical_json_digest(
        revised_graph
    )
    begin_stage(asset_state_path, "articulation", actor="asset-coordinator")
    refinement_review = (
        stage_directory(asset_state_path, "articulation") / "refinement-review.json"
    )
    atomic_write_json(refinement_review, {"decision": "refine"})
    refinement_binding = asset_state_module._binding(
        refinement_review,
        label="Articulation refinement review",
    )
    refinement_state = load_verified_run(asset_state_path)
    asset_state_module._prepare_refinement_attempt(
        refinement_state,
        stage="articulation",
        reason_review=refinement_binding,
        output_asset=None,
        output_dependencies=[],
        evidence=[refinement_binding],
        actor="asset-coordinator",
    )
    refined = asset_state_module._write_run(asset_state_path, refinement_state)
    active_stage = refined.stages["articulation"]
    assert active_stage.review_candidates is None
    assert active_stage.review_decisions is None
    assert active_stage.superseded_reviews == []
    assert len(active_stage.superseded_attempts[-1].superseded_reviews) == 1
    serialized_refined_v2 = refined.model_copy(
        update={"schema_version": "content-agent-workflows.asset-composition-run.v2"}
    ).model_dump(mode="json")
    serialized_stage = serialized_refined_v2["stages"]["articulation"]
    assert "superseded_reviews" not in serialized_stage
    assert "superseded_reviews" not in serialized_stage["superseded_attempts"][-1]
    AssetCompositionRun.model_validate(serialized_refined_v2)
    begin_stage(asset_state_path, "articulation", actor="asset-coordinator")
    replacement_candidates = (
        stage_directory(asset_state_path, "articulation")
        / "replacement-candidates.json"
    )
    atomic_write_json(replacement_candidates, {"graph": "fresh-attempt"})
    fresh_review = require_review(
        asset_state_path,
        candidates_path=replacement_candidates,
        actor="asset-coordinator",
    )
    assert fresh_review.stages["articulation"].status == "needs_review"
    assert load_verified_run(asset_state_path) == fresh_review
    authored = articulation_runner.resume_articulation_workflow(
        config.output_dir,
        repo_root=config.repo_root,
    )
    assert authored.status == "needs_review"
    assert client.author_call_count == 1
    assert client.author_requests[0].accepted_candidate_ids == graph.candidate_ids
    current = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert current.embedded_human_decision is not None
    current_human = EmbeddedHumanDecision.model_validate_json(
        Path(current.embedded_human_decision.path).read_bytes()
    )
    assert current_human.artifact_id == "articulation-human-graph-decision-r002"
    Path(parent_bindings[0].path).write_bytes(parent_bytes[0] + b"\n")
    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_run(asset_state_path)


def test_revised_graph_completes_public_outer_asset_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, members, owners = _synthetic_source(
        tmp_path,
        owner_count=3,
        member_count=6,
    )
    outer_root = tmp_path / "outer-asset-run"
    outer_root.mkdir()
    raw_root = outer_root / "raw"
    raw_root.mkdir()
    joint_config = outer_root / "joint.yaml"
    joint_config.write_text("review_policy: all\n", encoding="utf-8")
    materials_yaml = outer_root / "materials.yaml"
    materials_yaml.write_text("materials: []\n", encoding="utf-8")
    materials_usd = outer_root / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    outer_request = outer_root / "request.json"
    atomic_write_json(
        outer_request,
        {
            "schema_version": "content-agents.asset-composition-request.v1",
            "created_at": "2026-08-15T00:00:00+00:00",
            "workflow": "asset.run",
            "coordinator_mode": "single_reasoning_loop",
            "run_id": "outer-revised-graph-run",
            "run_dir": str(outer_root.resolve()),
            "run_state": str((outer_root / "asset_run.json").resolve()),
            "repository_root": str(tmp_path.resolve()),
            "source_asset": str(source.resolve()),
            "prompt": "author a reviewed articulation",
            "physics_validation_mode": "runtime_required",
            "joint_config": str(joint_config.resolve()),
            "joint_config_binding": _binding(joint_config).model_dump(mode="json"),
            "materials_yaml": str(materials_yaml.resolve()),
            "materials_yaml_binding": _binding(materials_yaml).model_dump(mode="json"),
            "materials_usd": str(materials_usd.resolve()),
            "materials_usd_binding": _binding(materials_usd).model_dump(mode="json"),
            "materials_usd_dependencies": [],
            "reference_images": [],
            "reference_files": [],
            "reference_bindings": [],
            "runtime": {
                "runner": "codex",
                "model": None,
                "model_reasoning_effort": None,
                "scene_tool_timeout_seconds": 60.0,
                "child_timeout_seconds": 3600.0,
                "codex_base_url": None,
                "codex_sandbox_mode": "workspace-write",
                "codex_config": {},
                "claude_config": {},
                "claude_permission_mode": "default",
                "claude_max_turns": None,
                "claude_execution_mode": "sdk",
            },
        },
    )
    outer_state = outer_root / "asset_run.json"
    create_run(
        outer_state,
        run_id="outer-revised-graph-run",
        request_path=outer_request,
        source_asset=source,
        coordinator_mode="single_reasoning_loop",
    )

    def record_plan(*, evidence_path: Path, reason: str) -> None:
        plan_path = (
            raw_root
            / f"plan-{len(load_verified_run(outer_state).coordinator.plan_revisions) + 1:03d}.json"
        )
        atomic_write_json(
            plan_path,
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": "articulation",
                "objective": "Publish the exact reviewed canonical graph.",
                "steps": [
                    {
                        "stage": "articulation",
                        "objective": "Preserve and validate immutable graph revisions.",
                        "acceptance_evidence": ["complete revised graph chain"],
                        "may_revisit": True,
                    }
                ],
                "evidence_paths": [str(evidence_path.resolve())],
                "revision_reason": reason,
            },
        )
        record_coordinator_plan(
            outer_state,
            plan_path=plan_path,
            actor="outer-coordinator",
        )

    def record_review(
        *,
        decision: str,
        evidence_paths: list[Path],
        output: Path | None = None,
    ) -> None:
        review_path = (
            raw_root
            / f"review-{len(load_verified_run(outer_state).coordinator.evidence_reviews) + 1:03d}.json"
        )
        atomic_write_json(
            review_path,
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "articulation",
                "output_asset_path": (
                    str(output.resolve()) if output is not None else None
                ),
                "evidence_paths": [str(item.resolve()) for item in evidence_paths],
                "findings": ["Exact graph revision provenance is complete."],
                "decision": decision,
                "target_stage": None,
                "decision_summary": (
                    "Pause for exact graph review."
                    if decision == "await_review"
                    else "Accept the exact revised graph and saved-stage evidence."
                ),
                "repair_scope": [],
            },
        )
        record_coordinator_evidence_review(
            outer_state,
            review_path=review_path,
            actor="outer-coordinator",
        )

    record_plan(
        evidence_path=outer_request,
        reason="Start the exact provider-neutral Articulation attempt.",
    )
    begin_stage(outer_state, "articulation", actor="outer-coordinator")
    domain_run = stage_directory(outer_state, "articulation") / "domain-run"
    domain_run.mkdir()
    context = build_embedded_domain_execution_context(
        outer_state,
        domain="articulation",
        input_asset=source,
        output_dir=domain_run,
    )
    preparation = _provider_neutral_preparation(
        domain_run,
        source=source,
        members=members,
        owners=owners,
    )
    preparation_path = domain_run / "provider_neutral_preparation.json"
    atomic_write_json(preparation_path, preparation)
    runtime = _provider_neutral_runtime(context, preparation)
    request, launcher_config = articulation_runner._build_request(
        articulation_runner.ArticulationRunConfig(
            repo_root=Path.cwd(),
            source_asset=source,
            output_dir=domain_run,
            joint_config=None,
            intent="Author a complete synthetic articulation graph.",
            review_policy="all",
            max_candidate_count=256,
            embedded_run_state=outer_state,
            embedded_preparation=preparation_path,
        )
    )
    assert request.execution_context == context
    articulation_runner._write_articulation_agent_launcher(
        domain_run,
        launcher_config,
        request=request,
    )
    prepared = prepare_embedded_articulation_workflow(
        request,
        mode="batch",
        runtime=runtime,
        preparation=preparation,
    )
    assert prepared.embedded_evidence_path is not None
    evidence = EmbeddedDomainEvidence.model_validate_json(
        Path(prepared.embedded_evidence_path).read_bytes()
    )
    graph = _synthetic_graph(preparation, members=members, owners=owners)
    apply_embedded_articulation_decision_patch(
        domain_run,
        EmbeddedArticulationDecisionPatch(
            identity_digest=canonical_json_digest(runtime.identity),
            evidence_digest=artifact_reference(evidence).sha256,
            proposal_digest=None,
            canonical_graph=graph,
            outer_review_disposition="accept",
            human_review=EmbeddedArticulationHumanReviewPolicy(
                status="human_required",
                reasons=("task_policy",),
            ),
            rationale="Require exact human review of the outer-authored graph.",
        ),
        runtime=runtime,
    )
    initial_graph_path = domain_run / "canonical_articulation_graph.json"
    record_review(
        decision="await_review",
        evidence_paths=[initial_graph_path],
    )
    require_review(
        outer_state,
        candidates_path=initial_graph_path,
        actor="outer-coordinator",
    )
    revise_decisions = {
        candidate_id: ("revise" if index == 0 else "accept")
        for index, candidate_id in enumerate(graph.candidate_ids)
    }
    revise_path = raw_root / "human-revise.json"
    atomic_write_json(revise_path, revise_decisions)
    revised_outer = record_review_decisions(
        outer_state,
        decisions_path=revise_path,
        reviewer="human-reviewer",
    )

    def outer_human_acceptance(
        _run_dir: Path,
        _canonical_graph: Any,
    ) -> EmbeddedArticulationHumanAcceptance | None:
        current = load_verified_run(outer_state).stages["articulation"]
        assert current.review_candidates is not None
        return load_embedded_articulation_human_acceptance(
            outer_state,
            canonical_graph=current.review_candidates.path,
        )

    conditional = run_embedded_articulation_workflow(
        request,
        mode="batch",
        client=_ProviderNeutralAuthoringClient(),
        runtime=runtime,
        human_acceptance_loader=outer_human_acceptance,
        preparation=preparation,
    )
    assert conditional.status == "conditional"
    checkpoint = ArticulationRunState.model_validate_json(
        (domain_run / "checkpoint.json").read_bytes()
    )
    assert checkpoint.embedded_canonical_graph is not None
    assert checkpoint.embedded_outer_review is not None
    assert checkpoint.embedded_coordinator_decision is not None
    assert checkpoint.embedded_human_decision is not None
    outer_review_decisions = revised_outer.stages["articulation"].review_decisions
    assert outer_review_decisions is not None
    human_decision = EmbeddedHumanDecision.model_validate_json(
        Path(checkpoint.embedded_human_decision.path).read_bytes()
    )
    revised_joint = graph.joints[0].model_copy(update={"upper_limit": 46.0})
    revised_graph = graph.model_copy(
        update={"joints": (revised_joint, *graph.joints[1:])}
    )
    revision_patch = EmbeddedArticulationGraphRevisionPatch(
        expected_state_revision=checkpoint.revision,
        identity_digest=canonical_json_digest(runtime.identity),
        evidence_digest=artifact_reference(evidence).sha256,
        proposal_digest=None,
        parent_canonical_graph_sha256=checkpoint.embedded_canonical_graph.sha256,
        parent_canonical_graph_digest=canonical_json_digest(graph),
        human_decision_sha256=checkpoint.embedded_human_decision.sha256,
        human_decisions=_binding(Path(outer_review_decisions.path)),
        reviewer=human_decision.producer.producer_id,
        revision_reason="Human review revised one upper limit.",
        requested_at=datetime.now(UTC),
        changes=(
            EmbeddedArticulationGraphChange(
                candidate_id=graph.candidate_ids[0],
                field="upper_limit",
                previous_value=graph.joints[0].upper_limit,
                revised_value=revised_joint.upper_limit,
            ),
        ),
        revised_graph=revised_graph,
        revised_graph_digest=canonical_json_digest(revised_graph),
    )
    revision_patch_path = domain_run / "graph_revision_patch.json"
    atomic_write_json(revision_patch_path, revision_patch)
    monkeypatch.setattr(
        articulation_runner,
        "_embedded_articulation_runtime",
        lambda *_args, **_kwargs: runtime,
    )
    assert (
        workflow_cli_main(
            [
                "articulation",
                "revise-graph",
                "--run-dir",
                str(domain_run),
                "--revision-patch",
                str(revision_patch_path),
                "--repo-root",
                str(Path.cwd()),
                "--json",
            ]
        )
        == 0
    )
    revised_checkpoint = ArticulationRunState.model_validate_json(
        (domain_run / "checkpoint.json").read_bytes()
    )
    assert revised_checkpoint.embedded_graph_revision is not None
    assert revised_checkpoint.embedded_canonical_graph is not None
    revised_outer_state = load_verified_run(outer_state).stages["articulation"]
    assert revised_outer_state.status == "needs_review"
    assert len(revised_outer_state.superseded_reviews) == 1
    accept_path = raw_root / "human-accept-revised.json"
    atomic_write_json(
        accept_path,
        {candidate_id: "accept" for candidate_id in revised_graph.candidate_ids},
    )
    record_review_decisions(
        outer_state,
        decisions_path=accept_path,
        reviewer="human-reviewer",
    )
    client = _ProviderNeutralAuthoringClient()
    authored = run_embedded_articulation_workflow(
        request,
        mode="batch",
        client=client,
        runtime=runtime,
        human_acceptance_loader=outer_human_acceptance,
        preparation=preparation,
    )
    assert authored.status == "needs_review"
    authored_state = ArticulationRunState.model_validate_json(
        (domain_run / "checkpoint.json").read_bytes()
    )
    assert authored_state.embedded_execution_result is not None
    result_artifact = EmbeddedBoundedExecutionResult.model_validate_json(
        Path(authored_state.embedded_execution_result.path).read_bytes()
    )
    apply_embedded_articulation_post_review(
        domain_run,
        EmbeddedArticulationPostReviewPatch(
            execution_result_digest=artifact_reference(result_artifact).sha256,
            accepted_decision_digest=result_artifact.accepted_decision_digest,
            disposition="accept",
            findings=("Exact revised graph and saved-stage readback match.",),
        ),
        runtime=runtime,
    )
    completed = run_embedded_articulation_workflow(
        request,
        mode="batch",
        client=client,
        runtime=runtime,
        human_acceptance_loader=outer_human_acceptance,
        preparation=preparation,
    )
    assert completed.status == "completed"
    assert completed.output_asset_path is not None
    output = Path(completed.output_asset_path)

    accepted_review = (
        load_verified_run(outer_state).stages["articulation"].review_decisions
    )
    assert accepted_review is not None
    record_plan(
        evidence_path=Path(accepted_review.path),
        reason="Resume the same attempt after complete revised-digest review.",
    )
    resumed = begin_stage(outer_state, "articulation", actor="outer-coordinator")
    assert resumed.stages["articulation"].attempt_count == 1
    evidence_paths = sorted(
        (
            path
            for path in domain_run.rglob("*.json")
            if path.name not in {"articulation_agent_observation.json"}
        ),
        key=str,
    )
    record_review(
        decision="accept",
        evidence_paths=evidence_paths,
        output=output,
    )
    accepted_outer = complete_stage(
        outer_state,
        "articulation",
        output_asset=output,
        evidence_paths=evidence_paths,
        summary="Accepted the exact immutable revised graph chain.",
        actor="outer-coordinator",
    )

    outer_stage = accepted_outer.stages["articulation"]
    assert outer_stage.status == "completed"
    assert outer_stage.attempt_count == 1
    assert len(outer_stage.superseded_reviews) == 1
    assert outer_stage.review_candidates is not None
    assert (
        outer_stage.review_candidates.sha256
        == revised_checkpoint.embedded_canonical_graph.sha256
    )
    assert load_verified_run(outer_state) == accepted_outer


def test_graph_revision_retry_recovers_after_receipt_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, _graph, patch, runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    from content_agent_workflows.articulation import embedded_decision

    real_persist = embedded_decision._persist_state
    failed = False

    def interrupt_final_checkpoint(
        root: Path,
        state: ArticulationRunState,
    ) -> ArticulationRunState:
        nonlocal failed
        if not failed and state.embedded_graph_revision is not None:
            failed = True
            raise RuntimeError("simulated interruption before revision checkpoint")
        return real_persist(root, state)

    monkeypatch.setattr(
        embedded_decision,
        "_persist_state",
        interrupt_final_checkpoint,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        apply_embedded_articulation_graph_revision(
            config.output_dir,
            patch,
            runtime=runtime,
        )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.embedded_graph_revision is None
    interrupted_receipt = EmbeddedArticulationGraphRevision.model_validate_json(
        (
            config.output_dir
            / "graph_revisions"
            / "revision-002"
            / "graph_revision.json"
        ).read_bytes()
    )
    assert interrupted_receipt.requested_at == patch.requested_at
    assert interrupted_receipt.created_at >= patch.requested_at
    monkeypatch.setattr(embedded_decision, "_persist_state", real_persist)

    recovered = apply_embedded_articulation_graph_revision(
        config.output_dir,
        patch,
        runtime=runtime,
    )
    assert recovered.phase == "needs_review"
    assert recovered.embedded_graph_revision is not None
    recovered_receipt = EmbeddedArticulationGraphRevision.model_validate_json(
        Path(recovered.embedded_graph_revision.path).read_bytes()
    )
    assert recovered_receipt == interrupted_receipt
    assert (
        apply_embedded_articulation_graph_revision(
            config.output_dir,
            patch,
            runtime=runtime,
        )
        == recovered
    )


def test_graph_revision_rejects_future_caller_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, _graph, patch, runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    future_patch = patch.model_copy(
        update={"requested_at": datetime.now(UTC) + timedelta(days=1)}
    )

    with pytest.raises(EmbeddedArticulationError, match="timestamp is in the future"):
        apply_embedded_articulation_graph_revision(
            config.output_dir,
            future_patch,
            runtime=runtime,
        )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.embedded_graph_revision is None
    assert not (config.output_dir / "graph_revisions").exists()


def test_graph_revision_rejects_endpoint_and_topology_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, graph, patch, runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    drifted_joint = patch.revised_graph.joints[0].model_copy(
        update={"body1_owner_prim": graph.joints[1].body1_owner_prim}
    )
    drifted_graph = patch.revised_graph.model_copy(
        update={"joints": (drifted_joint, *patch.revised_graph.joints[1:])}
    )
    drifted_patch = patch.model_copy(
        update={
            "revised_graph": drifted_graph,
            "revised_graph_digest": canonical_json_digest(drifted_graph),
        }
    )

    with pytest.raises(EmbeddedArticulationError, match="unsupported joint drift"):
        apply_embedded_articulation_graph_revision(
            config.output_dir,
            drifted_patch,
            runtime=runtime,
        )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.revision == patch.expected_state_revision
    assert checkpoint.embedded_graph_revision is None


def test_public_graph_revision_operation_updates_domain_and_outer_bookkeeping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, _graph, patch, _runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    patch_path = config.output_dir / "graph_revision_patch.json"
    atomic_write_json(patch_path, patch)
    outer_calls: list[dict[str, object]] = []
    outer_prevalidations: list[dict[str, object]] = []
    writes_under_lock: list[str] = []
    lock_held = False
    real_lock = articulation_runner._skill_routed_run_lock
    real_summary = articulation_runner.write_articulation_workflow_summary
    real_observation = articulation_runner._write_articulation_observation_if_needed

    @contextmanager
    def tracked_lock(
        run_dir: Path,
        *,
        domain: Literal["articulation", "texture"],
    ) -> Iterator[None]:
        nonlocal lock_held
        with real_lock(run_dir, domain=domain):
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

    def tracked_summary(*args: object, **kwargs: object) -> Any:
        assert lock_held
        writes_under_lock.append("summary")
        return real_summary(*args, **kwargs)

    def tracked_observation(*args: object, **kwargs: object) -> Any:
        assert lock_held
        writes_under_lock.append("observation")
        return real_observation(*args, **kwargs)

    def capture_outer_revision(
        state_path: Path,
        **kwargs: object,
    ) -> object:
        outer_calls.append({"state_path": state_path, **kwargs})
        return object()

    def capture_outer_prevalidation(
        state_path: Path,
        **kwargs: object,
    ) -> object:
        outer_prevalidations.append({"state_path": state_path, **kwargs})
        return object()

    monkeypatch.setattr(
        articulation_runner,
        "run_asset_coordinator_transition",
        _run_direct_outer_transition,
    )
    monkeypatch.setattr(
        articulation_runner,
        "record_articulation_graph_revision",
        capture_outer_revision,
    )
    monkeypatch.setattr(
        articulation_runner,
        "validate_articulation_graph_revision_request",
        capture_outer_prevalidation,
    )
    monkeypatch.setattr(articulation_runner, "_skill_routed_run_lock", tracked_lock)
    monkeypatch.setattr(
        articulation_runner,
        "write_articulation_workflow_summary",
        tracked_summary,
    )
    monkeypatch.setattr(
        articulation_runner,
        "_write_articulation_observation_if_needed",
        tracked_observation,
    )

    result = articulation_runner.revise_articulation_graph_workflow(
        config.output_dir,
        patch_path,
        repo_root=config.repo_root,
    )

    assert result.status == "needs_review"
    assert result.embedded_graph_revision_path is not None
    assert len(outer_calls) == 1
    assert len(outer_prevalidations) == 1
    assert outer_prevalidations[0]["state_path"] == config.embedded_run_state
    assert outer_prevalidations[0]["human_decisions"] == patch.human_decisions
    assert outer_calls[0]["state_path"] == config.embedded_run_state
    assert outer_calls[0]["revision_receipt_path"] == (
        result.embedded_graph_revision_path
    )
    assert outer_calls[0]["candidates_path"] == result.embedded_canonical_graph_path
    observation = json.loads(
        (config.output_dir / "articulation_agent_observation.json").read_text(
            encoding="utf-8"
        )
    )
    assert observation["checkpoint_revision"] == patch.expected_state_revision + 1
    assert observation["canonical_graph_path"] == result.embedded_canonical_graph_path
    assert observation["canonical_graph_digest"] == patch.revised_graph_digest
    assert observation["graph_revision_path"] == result.embedded_graph_revision_path
    legacy_result_payload = result.model_dump(mode="json")
    legacy_result_payload["schema_version"] = (
        "content-agent-workflows.articulation-finalization-result.v3"
    )
    with pytest.raises(
        ValidationError, match="graph revision requires finalization v4"
    ):
        ArticulationFinalizationResult.model_validate(legacy_result_payload)
    serialized_v3 = result.model_copy(
        update={
            "schema_version": (
                "content-agent-workflows.articulation-finalization-result.v3"
            ),
            "embedded_graph_revision_path": None,
        }
    ).model_dump(mode="json")
    assert "embedded_graph_revision_path" not in serialized_v3
    ArticulationFinalizationResult.model_validate(serialized_v3)
    assert writes_under_lock == ["summary", "observation"]


def test_public_graph_revision_rechecks_links_before_reading_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, _graph, patch, _runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    patch_path = config.output_dir / "graph_revision_patch.json"
    atomic_write_json(patch_path, patch)
    outside_patch = tmp_path / "outside_graph_revision_patch.json"
    outside_patch.write_text("not valid JSON\n", encoding="utf-8")

    def replace_patch_before_transition(
        _state_path: Path,
        *,
        transition: Any,
    ) -> object:
        patch_path.unlink()
        patch_path.symlink_to(outside_patch)
        return transition()

    monkeypatch.setattr(
        articulation_runner,
        "run_asset_coordinator_transition",
        replace_patch_before_transition,
    )

    with pytest.raises(RuntimeError, match="symlinks are not allowed"):
        articulation_runner.revise_articulation_graph_workflow(
            config.output_dir,
            patch_path,
            repo_root=config.repo_root,
        )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.revision == patch.expected_state_revision
    assert checkpoint.embedded_graph_revision is None


def test_public_graph_revision_rejects_outer_binding_before_domain_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, _human, _graph, patch, _runtime = _prepare_graph_revision(
        tmp_path,
        monkeypatch,
    )
    outer_state = config.embedded_run_state
    assert outer_state is not None
    outer_state.unlink()
    create_run(
        outer_state,
        run_id="outer-prevalidation-run",
        request_path=tmp_path / "asset_request.json",
        source_asset=config.source_asset,
        coordinator_mode="legacy",
    )
    checkpoint = ArticulationRunState.model_validate_json(
        (config.output_dir / "checkpoint.json").read_bytes()
    )
    assert checkpoint.embedded_canonical_graph is not None
    begin_stage(outer_state, "articulation", actor="asset-coordinator")
    require_review(
        outer_state,
        candidates_path=checkpoint.embedded_canonical_graph.path,
        actor="asset-coordinator",
    )
    record_review_decisions(
        outer_state,
        decisions_path=patch.human_decisions.path,
        reviewer="human-reviewer",
    )

    equivalent_decisions = tmp_path / "content-equivalent-decisions.json"
    equivalent_decisions.write_bytes(Path(patch.human_decisions.path).read_bytes())
    wrong_patch = patch.model_copy(
        update={"human_decisions": _binding(equivalent_decisions)}
    )
    patch_path = config.output_dir / "graph_revision_patch.json"
    atomic_write_json(patch_path, wrong_patch)
    original_checkpoint = (config.output_dir / "checkpoint.json").read_bytes()

    monkeypatch.setattr(
        articulation_runner,
        "run_asset_coordinator_transition",
        _run_direct_outer_transition,
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="differs from the exact active outer review",
    ):
        articulation_runner.revise_articulation_graph_workflow(
            config.output_dir,
            patch_path,
            repo_root=config.repo_root,
        )
    assert (config.output_dir / "checkpoint.json").read_bytes() == original_checkpoint
    assert not (config.output_dir / "graph_revisions").exists()

    atomic_write_json(patch_path, patch)
    revised = articulation_runner.revise_articulation_graph_workflow(
        config.output_dir,
        patch_path,
        repo_root=config.repo_root,
    )
    assert revised.embedded_graph_revision_path is not None
    outer_revised = load_verified_run(outer_state)
    assert outer_revised.stages["articulation"].status == "needs_review"
    assert outer_revised.stages["articulation"].review_decisions is None


def test_human_graph_digest_drift_blocks_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_drift.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256="9" * 64,
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    with pytest.raises(EmbeddedArticulationError, match="digest is stale"):
        articulation_runner.review_articulation_workflow(
            config.output_dir,
            decisions,
            reviewer="human-reviewer",
            repo_root=config.repo_root,
        )
    assert client.author_call_count == 0


def test_predictions_injection_cannot_reach_graph_authorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    injected = ArticulationAuthoringRequest(
        source_asset=str(config.source_asset),
        source_sha256="1" * 64,
        source_dependency_bundle_sha256="2" * 64,
        candidate_document_path=str(config.output_dir / "provider.json"),
        candidate_document_sha256="3" * 64,
        accepted_candidate_ids=graph.candidate_ids,
        idempotency_key="4" * 64,
        predictions_path=str(client.predictions_path),
        predictions_sha256=file_sha256(client.predictions_path),
        output_dir=config.output_dir,
    )
    atomic_write_json(config.output_dir / "authoring_request.json", injected)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    assert client.author_call_count == 1
    assert client.author_requests[0].predictions_path is None
    persisted = ArticulationAuthoringRequest.model_validate_json(
        (config.output_dir / "authoring_request.json").read_bytes()
    )
    assert persisted.predictions_path is None
    assert persisted.candidate_document_path.endswith(
        "approved_articulation_candidates.json"
    )


def test_readback_mismatch_commits_no_result_or_receipt_and_no_duplicate_authoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(
        tmp_path, monkeypatch, exact_graph_match=False
    )
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    with pytest.raises(EmbeddedArticulationError, match="topology readback failed"):
        articulation_runner.review_articulation_workflow(
            config.output_dir,
            decisions,
            reviewer="human-reviewer",
            repo_root=config.repo_root,
        )
    assert client.author_call_count == 1
    with pytest.raises(
        EmbeddedDecisionAuthorizationReplayError,
        match="reconcil|replay|authorization",
    ):
        articulation_runner.review_articulation_workflow(
            config.output_dir,
            decisions,
            reviewer="human-reviewer",
            repo_root=config.repo_root,
        )
    assert client.author_call_count == 1


def test_saved_stage_missing_co_rigid_member_blocks_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(
        tmp_path,
        monkeypatch,
        omit_co_rigid_member=True,
    )
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    with pytest.raises(EmbeddedArticulationError, match="lost canonical members"):
        articulation_runner.review_articulation_workflow(
            config.output_dir,
            decisions,
            reviewer="human-reviewer",
            repo_root=config.repo_root,
        )
    assert client.author_call_count == 1


def test_safe_resume_after_readback_does_not_duplicate_authoring_and_requires_post_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    first = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    resumed = articulation_runner.resume_articulation_workflow(
        config.output_dir,
        repo_root=config.repo_root,
    )
    assert first.status == resumed.status == "needs_review"
    assert resumed.embedded_decision_receipt_path is None
    assert client.author_call_count == 1


def test_post_review_digest_mismatch_blocks_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    authored = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    patch = config.output_dir / "bad_post_review.json"
    atomic_write_json(
        patch,
        EmbeddedArticulationPostReviewPatch(
            execution_result_digest="8" * 64,
            accepted_decision_digest="7" * 64,
            disposition="accept",
            findings=("stale",),
        ),
    )
    with pytest.raises(EmbeddedArticulationError, match="digest is stale"):
        articulation_runner.finalize_articulation_agent_step(config.output_dir, patch)
    assert authored.embedded_decision_receipt_path is None


def test_failing_validation_replacement_blocks_post_review_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _client, human = _configured_run(tmp_path, monkeypatch)
    articulation_runner.run_articulation_workflow(config)
    graph = _apply_graph(config)
    decisions = config.output_dir / "human_accept.json"
    atomic_write_json(
        decisions, {candidate_id: "accept" for candidate_id in graph.candidate_ids}
    )
    human["value"] = EmbeddedArticulationHumanAcceptance(
        canonical_graph_sha256=canonical_json_digest(graph),
        decisions_path=str(decisions),
        decisions_sha256=file_sha256(decisions),
        reviewer="human-reviewer",
        decisions={candidate_id: "accept" for candidate_id in graph.candidate_ids},
    )
    authored = articulation_runner.review_articulation_workflow(
        config.output_dir,
        decisions,
        reviewer="human-reviewer",
        repo_root=config.repo_root,
    )
    validation_path = config.output_dir / "validation_evidence.json"
    validation = ArticulationValidationResult.model_validate_json(
        validation_path.read_bytes()
    ).model_copy(
        update={
            "status": "fail",
            "validated_candidate_ids": (),
            "exact_graph_match": False,
            "failures": ("replacement validation failed",),
        }
    )
    atomic_write_json(validation_path, validation)
    checkpoint_path = config.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["validation_result"]["sha256"] = file_sha256(validation_path)
    atomic_write_json(checkpoint_path, checkpoint)
    assert authored.embedded_execution_result_path is not None
    result_artifact = EmbeddedBoundedExecutionResult.model_validate_json(
        Path(authored.embedded_execution_result_path).read_bytes()
    )
    patch_path = config.output_dir / "post_review_patch.json"
    atomic_write_json(
        patch_path,
        EmbeddedArticulationPostReviewPatch(
            execution_result_digest=artifact_reference(result_artifact).sha256,
            accepted_decision_digest=result_artifact.accepted_decision_digest,
            disposition="accept",
            findings=("Attempt to accept replaced validation.",),
        ),
    )
    with pytest.raises(EmbeddedArticulationError, match="exact persisted"):
        articulation_runner.finalize_articulation_agent_step(
            config.output_dir, patch_path
        )
    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "awaiting_post_review"
    assert persisted["embedded_decision_receipt"] is None
