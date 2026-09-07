# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Articulation custody and bounded-refinement regressions."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

import content_agent_workflows.articulation.asset_leaf_adapter as asset_leaf_adapter
import content_agent_workflows.articulation.client as articulation_client
import content_agent_workflows.articulation.standalone_decision as standalone_decision
import content_agent_workflows.articulation.workflow as articulation_workflow
from content_agent_workflows.articulation import (
    ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
    ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION,
    STANDALONE_ARTICULATION_ASSET_IDENTITY_SCHEMA_VERSION,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationAuthorLeafProgress,
    ArticulationFocusedLeafInvocation,
    ArticulationFocusedLeafResult,
    ArticulationPreparationPublication,
    ArticulationRunState,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    ArtifactJsonArticulationProposalProvider,
    EmbeddedArticulationCanonicalGraph,
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationError,
    EmbeddedArticulationGroup,
    EmbeddedArticulationJoint,
    EmbeddedArticulationMembership,
    EmbeddedArticulationOutputEvidence,
    EmbeddedArticulationRigidLinkOperation,
    JointAgentGraphAuthoringClient,
    StandaloneArticulationAuthoringReceipt,
    StandaloneArticulationCandidateDecision,
    StandaloneArticulationCleanupReceipt,
    StandaloneArticulationDecisionPatch,
    StandaloneArticulationError,
    StandaloneArticulationIdentity,
    StandaloneArticulationObservation,
    StandaloneArticulationPostReviewPatch,
    StandaloneArticulationPreparation,
    StandaloneArticulationTerminalReceipt,
    apply_standalone_articulation_decision_patch,
    apply_standalone_articulation_post_review,
    articulation_asset_leaf_runtime_bindings,
    bind_standalone_articulation_output_evidence,
    build_standalone_articulation_observation,
    prepare_standalone_articulation_workflow,
    request_embedded_articulation_provider_proposal,
    run_articulation_author_asset_leaf,
    run_articulation_evidence_asset_leaf,
    run_articulation_publish_asset_leaf,
    run_articulation_review_asset_leaf,
    write_articulation_workflow_summary,
)
from content_agent_workflows.articulation.workflow import _source_identity
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
    DomainExecutionContext,
    ExecutionArtifactBinding,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.common.embedded_domain_decision import (
    DomainProposalPayload,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    canonical_json_digest,
)
from content_agent_workflows.validation import (
    execution_artifact_binding,
    produce_canonical_visual_evidence,
)


def _execution_binding(path: Path) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def test_authoring_receipt_requires_complete_membership_operation_custody() -> None:
    artifact = ArtifactBinding(path="/retained/artifact.json", sha256="1" * 64)
    output = ExecutionArtifactBinding(
        path="/retained/output.usda",
        sha256="2" * 64,
        size_bytes=1,
    )

    with pytest.raises(ValueError, match="must be selected together"):
        StandaloneArticulationAuthoringReceipt(
            identity_digest="3" * 64,
            decision_patch=artifact,
            decision_ledger=artifact,
            canonical_graph=artifact,
            approved_candidates=artifact,
            authoring_request=artifact,
            authoring_result=artifact,
            validation=artifact,
            readback=artifact,
            output_asset=output,
            accepted_candidate_ids=("joint-1",),
            accepted_authoring_plan=artifact,
        )


def test_incomplete_membership_derivative_prefix_is_retryable(tmp_path: Path) -> None:
    prepared = tmp_path / "accepted_membership_source.usda"
    receipt = tmp_path / "accepted_membership_operation_receipt.json"
    prepared.write_text("#usda 1.0\n", encoding="utf-8")

    assert articulation_client._reset_incomplete_accepted_membership_prefix(
        prepared,
        receipt,
    )
    assert not prepared.exists()
    assert not receipt.exists()

    prepared.write_text("#usda 1.0\n", encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    assert not articulation_client._reset_incomplete_accepted_membership_prefix(
        prepared,
        receipt,
    )
    assert prepared.is_file()
    assert receipt.is_file()


def test_accepted_execution_request_retains_validated_body_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "accepted_membership_source.usda"
    prepared.write_text("#usda 1.0\n", encoding="utf-8")
    identity = SimpleNamespace(root_sha256="1" * 64)

    class FakePlan:
        def __init__(self, rigid_bodies: tuple[str, ...]) -> None:
            self.rigid_bodies = rigid_bodies

        def model_copy(self, *, update: dict[str, object]) -> FakePlan:
            return FakePlan(tuple(update.get("rigid_bodies", self.rigid_bodies)))

    class FakeAcceptedPlan:
        def __init__(self) -> None:
            self.source_asset: object = SimpleNamespace(root_sha256="0" * 64)
            self.plan = FakePlan(("/Assembly/PromotedSupport",))

        def model_copy(self, *, update: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(
                source_asset=update.get("source_asset", self.source_asset),
                plan=update.get("plan", self.plan),
            )

    monkeypatch.setattr(
        articulation_client,
        "_validate_accepted_membership_derivative",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "world_understanding.functions.physics.joint_rigger.identify_usd_artifact",
        lambda *_args, **_kwargs: identity,
    )

    projected = articulation_client._accepted_execution_request(
        SimpleNamespace(),
        FakeAcceptedPlan(),
    )

    assert projected.source_asset is identity
    assert projected.plan.rigid_bodies == ("/Assembly/PromotedSupport",)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source.usda"
    source.write_text(
        """#usda 1.0
def Xform "Assembly" {
    def Xform "Base" {}
    def Xform "Drawer" {}
}
""",
        encoding="utf-8",
    )
    return source


def _single_owner_source(tmp_path: Path) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics

    source = tmp_path / "single-owner-source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    assert stage is not None
    assembly = UsdGeom.Xform.Define(stage, "/Assembly")
    stage.SetDefaultPrim(assembly.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/Assembly/Base")
    UsdGeom.Xform.Define(stage, "/Assembly/Base/Drawer")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    stage.GetRootLayer().Save()
    return source


def _preparation(tmp_path: Path, source: Path) -> StandaloneArticulationPreparation:
    source_sha256, dependency_sha256 = _source_identity(str(source))
    evidence_path = tmp_path / "evidence.json"
    atomic_write_json(evidence_path, {"capture": "deterministic"})
    evidence = _execution_binding(evidence_path)
    source_binding = _execution_binding(source)
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=True
    )
    provider = ProducerIdentity(
        producer_id="standalone-test-inspector",
        role="evidence_provider",
        implementation="deterministic-test-inspection",
        implementation_digest="1" * 64,
    )
    return StandaloneArticulationPreparation(
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=dependency_sha256,
        configuration_sha256=canonical_json_digest({"inspection": "test-v1"}),
        evidence_provider=provider,
        source_hierarchy=ProviderNeutralEvidenceRecord(
            evidence_id="source-hierarchy-inspection",
            evidence_type="inspection",
            status="available",
            summary="Exact source hierarchy.",
            artifacts=(source_binding, evidence),
            facts={"root_prim": "/Assembly"},
        ),
        source_members=ProviderNeutralEvidenceRecord(
            evidence_id="joint-source-member-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete source membership.",
            artifacts=(evidence,),
            facts={"source_member_prims": ["/Assembly/Base", "/Assembly/Drawer"]},
        ),
        authoritative_owners=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoritative-owner-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete owner membership.",
            artifacts=(evidence,),
            facts={
                "authoritative_owner_prims": [
                    "/Assembly/Base",
                    "/Assembly/Drawer",
                ]
            },
        ),
        capabilities=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoring-capabilities",
            evidence_type="capability",
            status="available",
            summary="Exact graph-only capability limits.",
            facts=capabilities.model_dump(mode="json"),
        ),
        renders=ProviderNeutralEvidenceRecord(
            evidence_id="joint-render-inspection",
            evidence_type="render",
            status="available",
            summary="Source-bound diagnostic render evidence.",
            artifacts=(evidence,),
            facts={"diagnostic_only": True},
        ),
        scene=ProviderNeutralEvidenceRecord(
            evidence_id="joint-scene-inspection",
            evidence_type="inspection",
            status="available",
            summary="Source-bound scene inspection.",
            artifacts=(evidence,),
            facts={"scene_tool": "usd-cli"},
        ),
    )


def _single_owner_preparation(
    tmp_path: Path,
    source: Path,
) -> StandaloneArticulationPreparation:
    preparation = _preparation(tmp_path, source)
    drawer = "/Assembly/Base/Drawer"
    capabilities = EmbeddedArticulationCapabilityLimits(
        canonical_output_evidence_required=True,
        rigid_link_body_membership_authoring_supported=True,
    )
    return preparation.model_copy(
        update={
            "source_members": preparation.source_members.model_copy(
                update={
                    "facts": {
                        "source_member_prims": ["/Assembly/Base", drawer],
                    }
                }
            ),
            "authoritative_owners": preparation.authoritative_owners.model_copy(
                update={
                    "facts": {
                        "authoritative_owner_prims": ["/Assembly/Base"],
                        "membership_policy": "retained-explicit-membership-v1",
                        "membership_rows": [
                            {
                                "member_prim": "/Assembly/Base",
                                "authoritative_owner_prim": "/Assembly/Base",
                                "disposition": "co_rigid",
                            },
                            {
                                "member_prim": drawer,
                                "authoritative_owner_prim": "/Assembly/Base",
                                "disposition": "co_rigid",
                            },
                        ],
                    }
                }
            ),
            "capabilities": preparation.capabilities.model_copy(
                update={"facts": capabilities.model_dump(mode="json")}
            ),
        }
    )


def _request(
    source: Path,
    run_dir: Path,
) -> ArticulationWorkflowRequest:
    context = DomainExecutionContext(
        domain="articulation",
        mode="standalone",
        reasoning_loop_owner="domain_child_agent",
    )
    return ArticulationWorkflowRequest(
        source_asset=str(source.resolve()),
        output_dir=run_dir,
        review_policy="all",
        metadata=metadata_with_domain_execution_context({}, context),
    )


def test_standalone_preparation_binds_exact_asset_leaf_ownership(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "asset-author-attempt"
    context = DomainExecutionContext(
        schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V3,
        domain="articulation",
        mode="standalone",
        reasoning_loop_owner="asset_coordinator",
    )
    request = ArticulationWorkflowRequest(
        source_asset=str(source.resolve()),
        output_dir=run_dir,
        review_policy="none",
        metadata=metadata_with_domain_execution_context({}, context),
    )

    prepare_standalone_articulation_workflow(
        request,
        mode="batch",
        preparation=_preparation(tmp_path, source),
    )
    identity = StandaloneArticulationIdentity.model_validate_json(
        (run_dir / "standalone_articulation_identity.json").read_bytes()
    )

    assert (
        identity.schema_version == STANDALONE_ARTICULATION_ASSET_IDENTITY_SCHEMA_VERSION
    )
    assert identity.execution_context == context


def _graph(
    preparation: StandaloneArticulationPreparation,
) -> EmbeddedArticulationCanonicalGraph:
    evidence_ids = (
        "source-hierarchy-inspection",
        "joint-source-member-inspection",
        "joint-authoritative-owner-inspection",
        "joint-authoring-capabilities",
        "joint-render-inspection",
        "joint-scene-inspection",
    )
    return EmbeddedArticulationCanonicalGraph(
        graph_id="standalone-drawer-graph",
        source_sha256=preparation.source_sha256,
        source_dependency_bundle_sha256=(preparation.source_dependency_bundle_sha256),
        source_member_prims=("/Assembly/Base", "/Assembly/Drawer"),
        authoritative_owner_prims=("/Assembly/Base", "/Assembly/Drawer"),
        candidate_ids=("drawer_slide",),
        groups=(
            EmbeddedArticulationGroup(
                group_id="base_group",
                authoritative_owner_prim="/Assembly/Base",
                member_prims=("/Assembly/Base",),
                role="base",
                role_state="source_backed",
                evidence_ids=evidence_ids,
            ),
            EmbeddedArticulationGroup(
                group_id="drawer_group",
                authoritative_owner_prim="/Assembly/Drawer",
                member_prims=("/Assembly/Drawer",),
                role="drawer",
                role_state="source_backed",
                evidence_ids=evidence_ids,
            ),
        ),
        memberships=(
            EmbeddedArticulationMembership(
                member_prim="/Assembly/Base",
                authoritative_owner_prim="/Assembly/Base",
                group_id="base_group",
                disposition="co_rigid",
                state="source_backed",
                evidence_ids=evidence_ids,
            ),
            EmbeddedArticulationMembership(
                member_prim="/Assembly/Drawer",
                authoritative_owner_prim="/Assembly/Drawer",
                group_id="drawer_group",
                disposition="independent_motion",
                state="source_backed",
                evidence_ids=evidence_ids,
            ),
        ),
        joints=(
            EmbeddedArticulationJoint(
                joint_id="drawer_slide",
                body0_owner_prim="/Assembly/Base",
                body1_owner_prim="/Assembly/Drawer",
                body0_role="base",
                body1_role="drawer",
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
                evidence_ids=evidence_ids,
            ),
        ),
    )


def _single_owner_graph(
    preparation: StandaloneArticulationPreparation,
) -> EmbeddedArticulationCanonicalGraph:
    evidence_ids = (
        "source-hierarchy-inspection",
        "joint-source-member-inspection",
        "joint-authoritative-owner-inspection",
        "joint-authoring-capabilities",
        "joint-render-inspection",
        "joint-scene-inspection",
    )
    drawer = "/Assembly/Base/Drawer"
    return EmbeddedArticulationCanonicalGraph(
        graph_id="standalone-single-owner-drawer-graph",
        source_sha256=preparation.source_sha256,
        source_dependency_bundle_sha256=(preparation.source_dependency_bundle_sha256),
        source_member_prims=("/Assembly/Base", drawer),
        authoritative_owner_prims=("/Assembly/Base", drawer),
        candidate_ids=("drawer_slide",),
        groups=(
            EmbeddedArticulationGroup(
                group_id="base_group",
                authoritative_owner_prim="/Assembly/Base",
                member_prims=("/Assembly/Base",),
                role="base",
                role_state="source_backed",
                evidence_ids=evidence_ids,
            ),
            EmbeddedArticulationGroup(
                group_id="drawer_group",
                authoritative_owner_prim=drawer,
                member_prims=(drawer,),
                role="drawer",
                role_state="source_backed",
                evidence_ids=evidence_ids,
            ),
        ),
        memberships=(
            EmbeddedArticulationMembership(
                member_prim="/Assembly/Base",
                authoritative_owner_prim="/Assembly/Base",
                group_id="base_group",
                disposition="co_rigid",
                state="source_backed",
                evidence_ids=evidence_ids,
            ),
            EmbeddedArticulationMembership(
                member_prim=drawer,
                authoritative_owner_prim=drawer,
                group_id="drawer_group",
                disposition="independent_motion",
                state="source_backed",
                evidence_ids=evidence_ids,
            ),
        ),
        joints=(
            EmbeddedArticulationJoint(
                joint_id="drawer_slide",
                body0_owner_prim="/Assembly/Base",
                body1_owner_prim=drawer,
                body0_role="base",
                body1_role="drawer",
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
                evidence_ids=evidence_ids,
            ),
        ),
        rigid_link_operations=(
            EmbeddedArticulationRigidLinkOperation(
                operation_id="promote-drawer-rigid-body",
                body_prim_path=drawer,
                previous_authoritative_owner_prim="/Assembly/Base",
                state="source_backed",
                evidence_ids=evidence_ids,
            ),
        ),
    )


def _hospital_graph_from_preparation(
    preparation: StandaloneArticulationPreparation,
) -> EmbeddedArticulationCanonicalGraph:
    """Build the exercised graph only from child-visible source evidence."""

    evidence_ids = (
        preparation.source_hierarchy.evidence_id,
        preparation.source_members.evidence_id,
        preparation.authoritative_owners.evidence_id,
        preparation.capabilities.evidence_id,
        preparation.renders.evidence_id,
        preparation.scene.evidence_id,
        *(record.evidence_id for record in preparation.additional_evidence),
    )
    members = tuple(preparation.source_members.facts["source_member_prims"])
    axis_rows = preparation.additional_evidence[0].facts["axis_evidence"]
    roles = {
        members[0]: "hospital_bed_base",
        members[1]: "caster_fork",
        members[2]: "caster_wheel",
    }
    group_ids = {
        member: f"hospital_bed_source_group_{index}"
        for index, member in enumerate(members)
    }
    return EmbeddedArticulationCanonicalGraph(
        graph_id="hospital-bed-source-evidence-graph",
        source_sha256=preparation.source_sha256,
        source_dependency_bundle_sha256=(preparation.source_dependency_bundle_sha256),
        source_member_prims=members,
        authoritative_owner_prims=members,
        candidate_ids=tuple(row["candidate_id"] for row in axis_rows),
        groups=tuple(
            EmbeddedArticulationGroup(
                group_id=group_ids[member],
                authoritative_owner_prim=member,
                member_prims=(member,),
                role=roles[member],
                role_state="source_backed",
                evidence_ids=evidence_ids,
            )
            for member in members
        ),
        memberships=tuple(
            EmbeddedArticulationMembership(
                member_prim=member,
                authoritative_owner_prim=member,
                group_id=group_ids[member],
                disposition=("co_rigid" if index == 0 else "independent_motion"),
                state="source_backed",
                evidence_ids=evidence_ids,
            )
            for index, member in enumerate(members)
        ),
        joints=tuple(
            EmbeddedArticulationJoint(
                joint_id=row["candidate_id"],
                body0_owner_prim=row["body0"],
                body1_owner_prim=row["body1"],
                body0_role=roles[row["body0"]],
                body1_role=roles[row["body1"]],
                role_state="source_backed",
                joint_type="revolute",
                endpoint_state="source_backed",
                type_state="source_backed",
                axis=row["axis"],
                axis_state="source_backed",
                lower_limit=None,
                upper_limit=None,
                limit_unit="degrees",
                limit_state="source_backed",
                frame_policy="body1_world_origin",
                frame_state="source_backed",
                evidence_ids=evidence_ids,
            )
            for row in axis_rows
        ),
        rigid_link_operations=tuple(
            EmbeddedArticulationRigidLinkOperation(
                operation_id=f"promote-hospital-member-{index}",
                body_prim_path=member,
                previous_authoritative_owner_prim=members[0],
                state="source_backed",
                evidence_ids=evidence_ids,
            )
            for index, member in enumerate(members[1:], start=1)
        ),
    )


class _GraphOnlyClient:
    def __init__(self, *, exact_graph_match: bool = True) -> None:
        self.author_calls = 0
        self.exact_graph_match = exact_graph_match

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: Any = None,
    ) -> ArticulationAuthoringResult:
        del cancel_checker
        self.author_calls += 1
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
            authored_candidate_ids=request.accepted_candidate_ids,
            authored_joint_count=len(request.accepted_candidate_ids),
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
            failures=() if self.exact_graph_match else ("readback mismatch",),
        )


class _RaisingExecutionClient(_GraphOnlyClient):
    def __init__(self, failure_site: str) -> None:
        super().__init__()
        self.failure_site = failure_site

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: Any = None,
    ) -> ArticulationAuthoringResult:
        if self.failure_site == "author":
            raise ValueError("axis cannot produce an orthonormal local joint frame")
        return super().author(request, cancel_checker=cancel_checker)

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: Any = None,
    ) -> ArticulationValidationResult:
        if self.failure_site == "validate":
            raise RuntimeError("saved-stage readback adapter failed")
        return super().validate(
            authoring,
            expected_candidate_ids=expected_candidate_ids,
            cancel_checker=cancel_checker,
        )


def _accepted_patch(
    run_dir: Path,
    preparation: StandaloneArticulationPreparation,
) -> StandaloneArticulationDecisionPatch:
    observation = build_standalone_articulation_observation(run_dir)
    evidence_ids = tuple(
        record["evidence_id"] for record in observation.evidence_records
    )
    return StandaloneArticulationDecisionPatch(
        expected_state_revision=observation.state_revision,
        identity_digest=observation.identity_digest,
        request_sha256=observation.request_sha256,
        preparation_sha256=observation.preparation_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        configuration_sha256=observation.configuration_sha256,
        disposition="accept",
        candidate_decisions=(
            StandaloneArticulationCandidateDecision(
                candidate_id="drawer_slide",
                disposition="accept",
                evidence_ids=evidence_ids,
                rationale="Source-backed drawer slide semantics are complete.",
            ),
        ),
        canonical_graph=_graph(preparation),
        evidence_requirements=evidence_ids,
        rationale="Accept the complete source-backed drawer graph.",
    )


def test_asset_author_leaf_accepts_proposal_bound_preparation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    preparation = _preparation(tmp_path, source).model_copy(
        update={"proposal_status": "not_evaluated"}
    )
    preparation_path = tmp_path / "embedded_articulation_preparation.json"
    atomic_write_json(preparation_path, preparation)
    evidence_binding = preparation.source_hierarchy.artifacts[-1]
    publication = ArticulationPreparationPublication(
        retained_root=str(tmp_path.resolve()),
        readback=evidence_binding,
        source=_execution_binding(source),
        dependencies=(),
        configuration=evidence_binding,
        inspector_implementation=evidence_binding,
        saved_stage=_execution_binding(source),
        renders=(evidence_binding,),
        scene_artifacts=(evidence_binding,),
        retained_closure_digest=canonical_json_digest(
            {"fixture": "asset-author-proposal-bound-preparation"}
        ),
        preparation=_execution_binding(preparation_path),
        preparation_digest=canonical_json_digest(preparation),
        evidence_provider=preparation.evidence_provider,
        publisher_implementation_sha256="2" * 64,
    )
    publication_path = tmp_path / "articulation_preparation_publication.json"
    atomic_write_json(publication_path, publication)
    provider_payload_path = tmp_path / "provider_payload.json"
    atomic_write_json(
        provider_payload_path,
        DomainProposalPayload(
            schema_version="standalone-author-test-provider.v1",
            values={"candidate_hints": [{"id": "drawer_slide"}]},
        ),
    )
    proposal_attempt = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal-attempt",
        intent="Suggest the exact source-backed drawer slide.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="standalone-author-test-provider",
            capability_id="articulation-proposal-v1",
            payload_path=provider_payload_path,
        ),
    )
    attempt = tmp_path / "author-attempt"
    attempt.mkdir()
    patch_path = attempt / "articulation_decision_patch.json"
    invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.author.v1",
        attempt_root=str(attempt.resolve()),
        source=_execution_binding(source),
        preparation_publication=_execution_binding(publication_path),
        proposal_result=proposal_attempt.terminal_receipt,
        decision_patch_path=str(patch_path.resolve()),
        intent="Author only the exact source-backed drawer slide.",
        provider_status="provided",
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    progress = run_articulation_author_asset_leaf(invocation_path)

    assert isinstance(progress, ArticulationAuthorLeafProgress)
    assert progress.native_status == "awaiting_decision"
    assert progress.provider_status == "provided"
    observation = StandaloneArticulationObservation.model_validate_json(
        Path(progress.observation.path).read_bytes()
    )
    assert observation.optional_proposal is not None
    frozen = StandaloneArticulationPreparation.model_validate_json(
        (attempt / "native/standalone_articulation_preparation.json").read_bytes()
    )
    assert frozen.proposal_status == "available"
    assert frozen.proposal is not None
    assert frozen.proposal.preparation_digest == publication.preparation_digest
    assert canonical_json_digest(frozen) != publication.preparation_digest


@pytest.mark.parametrize("refine_once", (False, True), ids=("accept", "refine"))
@pytest.mark.parametrize(
    "precreate_native",
    (
        pytest.param("absent", id="native-absent"),
        pytest.param("empty", id="native-empty"),
        pytest.param(
            "junction",
            id="native-junction",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="directory junctions are Windows-specific"
            ),
        ),
        pytest.param(
            "state-identity-hardlink",
            id="native-state-identity-hardlink",
        ),
        pytest.param(
            "state-checkpoint-hardlink",
            id="native-state-checkpoint-hardlink",
        ),
        pytest.param(
            "state-identity-symlink",
            id="native-state-identity-symlink",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="POSIX state symlink regression"
            ),
        ),
        pytest.param(
            "state-checkpoint-symlink",
            id="native-state-checkpoint-symlink",
            marks=pytest.mark.skipif(
                os.name == "nt", reason="POSIX state symlink regression"
            ),
        ),
        pytest.param(
            "state-identity-junction",
            id="native-state-identity-junction",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="state junctions are Windows-specific"
            ),
        ),
        pytest.param(
            "state-checkpoint-junction",
            id="native-state-checkpoint-junction",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="state junctions are Windows-specific"
            ),
        ),
    ),
)
def test_asset_author_leaf_pauses_for_outer_patch_then_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refine_once: bool,
    precreate_native: Literal[
        "absent",
        "empty",
        "junction",
        "state-identity-hardlink",
        "state-checkpoint-hardlink",
        "state-identity-symlink",
        "state-checkpoint-symlink",
        "state-identity-junction",
        "state-checkpoint-junction",
    ],
) -> None:
    if refine_once and precreate_native.startswith("state-"):
        pytest.skip("unsafe reused state is independent of refinement policy")
    source = _source(tmp_path)
    preparation = _preparation(tmp_path, source)
    preparation_path = tmp_path / "embedded_articulation_preparation.json"
    atomic_write_json(preparation_path, preparation)
    evidence_binding = preparation.source_hierarchy.artifacts[-1]
    publication = ArticulationPreparationPublication(
        retained_root=str(tmp_path.resolve()),
        readback=evidence_binding,
        source=_execution_binding(source),
        dependencies=(),
        configuration=evidence_binding,
        inspector_implementation=evidence_binding,
        saved_stage=_execution_binding(source),
        renders=(evidence_binding,),
        scene_artifacts=(evidence_binding,),
        retained_closure_digest=canonical_json_digest(
            {"fixture": "asset-author-preparation"}
        ),
        preparation=_execution_binding(preparation_path),
        preparation_digest=canonical_json_digest(preparation),
        evidence_provider=preparation.evidence_provider,
        publisher_implementation_sha256="2" * 64,
    )
    publication_path = tmp_path / "articulation_preparation_publication.json"
    atomic_write_json(publication_path, publication)
    attempt = tmp_path / "author-attempt"
    attempt.mkdir()
    patch_path = attempt / "articulation_decision_patch.json"
    invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.author.v1",
        attempt_root=str(attempt.resolve()),
        source=_execution_binding(source),
        preparation_publication=_execution_binding(publication_path),
        decision_patch_path=str(patch_path.resolve()),
        intent="Author only the exact source-backed drawer slide.",
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    if precreate_native == "empty":
        (attempt / "native").mkdir()
    elif precreate_native == "junction":
        outside_native = tmp_path / "outside-native"
        outside_native.mkdir()
        subprocess.run(
            (
                "cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/J",
                str(attempt / "native"),
                str(outside_native),
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        rejected = run_articulation_author_asset_leaf(invocation_path)
        assert isinstance(rejected, ArticulationFocusedLeafResult)
        assert rejected.native_disposition == "failed"
        assert rejected.native_status == "failed"
        assert rejected.error == (
            "ValueError: Articulation native workspace is not a real directory"
        )
        assert not any(outside_native.iterdir())
        return
    elif precreate_native.startswith("state-"):
        native = attempt / "native"
        native.mkdir()
        outside_native = tmp_path / "outside-native"
        outside_native.mkdir()
        identity_path = native / "standalone_articulation_identity.json"
        checkpoint_path = native / "checkpoint.json"
        _, state_leaf, link_kind = precreate_native.split("-", maxsplit=2)
        unsafe_path = identity_path if state_leaf == "identity" else checkpoint_path
        direct_path = checkpoint_path if state_leaf == "identity" else identity_path
        direct_path.write_text("{}\n", encoding="utf-8")
        if link_kind == "junction":
            outside_target = outside_native / state_leaf
            outside_target.mkdir()
            subprocess.run(
                (
                    "cmd.exe",
                    "/d",
                    "/c",
                    "mklink",
                    "/J",
                    str(unsafe_path),
                    str(outside_target),
                ),
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            outside_target = outside_native / f"{state_leaf}.json"
            outside_target.write_text("{}\n", encoding="utf-8")
            if link_kind == "symlink":
                unsafe_path.symlink_to(outside_target)
            else:
                os.link(outside_target, unsafe_path)

        rejected = run_articulation_author_asset_leaf(invocation_path)
        assert isinstance(rejected, ArticulationFocusedLeafResult)
        assert rejected.native_disposition == "failed"
        assert rejected.native_status == "failed"
        assert rejected.error == (
            f"ValueError: Articulation native {state_leaf} must be a direct "
            "single-link regular file"
        )
        if link_kind == "junction":
            assert outside_target.is_dir()
            assert not any(outside_target.iterdir())
        else:
            assert outside_target.read_text(encoding="utf-8") == "{}\n"
        return

    progress = run_articulation_author_asset_leaf(invocation_path)

    assert isinstance(progress, ArticulationAuthorLeafProgress)
    assert progress.native_status == "awaiting_decision"
    assert progress.decision_patch_path == str(patch_path.resolve())
    assert not patch_path.exists()

    accepted = _accepted_patch(attempt / "native", preparation)
    expected_native_patch_path = attempt / "native/articulation_decision_patch.json"
    if refine_once:
        evidence_ids = accepted.evidence_requirements
        initial = StandaloneArticulationDecisionPatch(
            expected_state_revision=accepted.expected_state_revision,
            identity_digest=accepted.identity_digest,
            request_sha256=accepted.request_sha256,
            preparation_sha256=accepted.preparation_sha256,
            source_sha256=accepted.source_sha256,
            source_dependency_bundle_sha256=(accepted.source_dependency_bundle_sha256),
            configuration_sha256=accepted.configuration_sha256,
            disposition="revise",
            candidate_decisions=(
                StandaloneArticulationCandidateDecision(
                    candidate_id="drawer_slide",
                    disposition="refine",
                    evidence_ids=evidence_ids,
                    issue_codes=("membership_unresolved",),
                    rationale="Confirm the one source-owner membership boundary.",
                ),
            ),
            evidence_requirements=evidence_ids,
            rationale="Request exactly one candidate-bound refinement.",
        )
        atomic_write_json(patch_path, initial)
        refinement = run_articulation_author_asset_leaf(invocation_path)
        assert isinstance(refinement, ArticulationAuthorLeafProgress)
        refined_observation = StandaloneArticulationObservation.model_validate_json(
            Path(refinement.observation.path).read_bytes()
        )
        state = ArticulationRunState.model_validate_json(
            (attempt / "native/checkpoint.json").read_bytes()
        )
        assert len(state.standalone_refinement_history) == 1
        assert state.standalone_decision_patch is not None
        replacement = accepted.model_copy(
            update={
                "expected_state_revision": refined_observation.state_revision,
                "refinement_attempt": 1,
                "parent_patch_sha256": state.standalone_decision_patch.sha256,
                "issue_packet_sha256": (state.standalone_refinement_history[-1].sha256),
            }
        )
        atomic_write_json(Path(refinement.decision_patch_path), replacement)
        expected_native_patch_path = Path(refinement.decision_patch_path)
    else:
        atomic_write_json(
            patch_path,
            accepted.model_dump(mode="json", exclude_defaults=True),
        )
    result = run_articulation_author_asset_leaf(
        invocation_path,
        client=_GraphOnlyClient(),
    )

    assert isinstance(result, ArticulationFocusedLeafResult)
    assert result.native_disposition == "passed"
    assert result.native_status == "awaiting_post_review"
    assert result.output is not None
    phase_receipt_path = attempt / "articulation_author_terminal_receipt.json"
    phase_receipt_bytes = phase_receipt_path.read_bytes()
    (attempt / "articulation_leaf_result.json").unlink()
    repeated_result = run_articulation_author_asset_leaf(
        invocation_path,
        client=_GraphOnlyClient(),
    )
    assert repeated_result == result
    assert phase_receipt_path.read_bytes() == phase_receipt_bytes
    native_patch_path = expected_native_patch_path
    assert all(item.path != str(patch_path) for item in result.evidence)
    assert any(item.path == str(native_patch_path) for item in result.evidence)
    assert file_sha256(patch_path) != file_sha256(native_patch_path)
    author_binding = next(
        item
        for item in articulation_asset_leaf_runtime_bindings()
        if item.descriptor.leaf_id == "articulation.author.v1"
    )
    projection = author_binding.project(
        invocation,
        result,
        invocation_artifact=_execution_binding(invocation_path),
        result_artifact=_execution_binding(attempt / "articulation_leaf_result.json"),
    )
    projected = (
        projection.payload.native_terminal_receipt,
        *projection.payload.evidence,
        *projection.payload.saved_stage_readbacks,
    )
    assert all(Path(item.path).is_relative_to(attempt) for item in projected)

    author_result_binding = _execution_binding(
        attempt / "articulation_leaf_result.json"
    )
    author_artifacts = {
        Path(item.path).name: item
        for item in (*result.evidence, *result.saved_stage_readbacks)
    }
    identity = StandaloneArticulationIdentity.model_validate_json(
        Path(
            author_artifacts["standalone_articulation_identity.json"].path
        ).read_bytes()
    )
    validation_attempt = tmp_path / "validation-attempt"
    validation_attempt.mkdir()
    visual_paths = {
        name: validation_attempt / name
        for name in (
            "canonical_visual_envelope.json",
            "canonical_visual_projection.json",
            "canonical_visual_request.json",
            "canonical_visual_payload.json",
            "render_report.json",
            "render.png",
        )
    }
    for name, path in visual_paths.items():
        path.write_bytes(name.encode("utf-8"))
    visual_bindings = {
        name: _execution_binding(path) for name, path in visual_paths.items()
    }
    output_evidence = EmbeddedArticulationOutputEvidence(
        canonical_visual_envelope=visual_bindings["canonical_visual_envelope.json"],
        canonical_visual_projection=visual_bindings["canonical_visual_projection.json"],
        canonical_visual_request=visual_bindings["canonical_visual_request.json"],
        canonical_visual_payload=visual_bindings["canonical_visual_payload.json"],
        canonical_visual_contract_sha256="3" * 64,
        source=identity.source,
        post_mutation_output=result.output,
        dependencies=(),
        dependency_closure_sha256=canonical_json_digest({"dependencies": []}),
        render_report=visual_bindings["render_report.json"],
        images=(visual_bindings["render.png"],),
        backend_alias="ovrtx",
        render_metadata={"renderer": "ovrtx"},
    )

    def build_output_evidence(
        path: str,
        *,
        expected_source: ExecutionArtifactBinding,
        expected_output: ExecutionArtifactBinding,
    ) -> EmbeddedArticulationOutputEvidence:
        assert path == str(visual_paths["canonical_visual_envelope.json"])
        assert expected_source == identity.source
        assert expected_output == result.output
        return output_evidence

    monkeypatch.setattr(
        asset_leaf_adapter,
        "build_embedded_articulation_output_evidence",
        build_output_evidence,
    )
    monkeypatch.setattr(
        asset_leaf_adapter,
        "validate_embedded_articulation_output_evidence",
        lambda _evidence: None,
    )
    evidence_attempt = tmp_path / "evidence-attempt"
    evidence_attempt.mkdir()
    evidence_invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.evidence.v1",
        attempt_root=str(evidence_attempt.resolve()),
        author_result=author_result_binding,
        canonical_visual_envelope=visual_bindings["canonical_visual_envelope.json"],
    )
    evidence_invocation_path = evidence_attempt / "invocation.json"
    atomic_write_json(evidence_invocation_path, evidence_invocation)
    evidence_result = run_articulation_evidence_asset_leaf(evidence_invocation_path)
    assert evidence_result.native_disposition == "passed"
    evidence_result_binding = _execution_binding(
        evidence_attempt / "articulation_leaf_result.json"
    )
    evidence_artifact = next(
        item
        for item in evidence_result.evidence
        if Path(item.path).name == "standalone_articulation_output_evidence.json"
    )

    review_attempt = tmp_path / "review-attempt"
    review_attempt.mkdir()
    post_review = StandaloneArticulationPostReviewPatch(
        identity_digest=canonical_json_digest(identity),
        authoring_receipt_sha256=author_artifacts[
            "standalone_articulation_authoring_receipt.json"
        ].sha256,
        canonical_graph_sha256=author_artifacts[
            "canonical_articulation_graph.json"
        ].sha256,
        readback_sha256=author_artifacts[
            "standalone_articulation_readback.json"
        ].sha256,
        output_evidence_sha256=evidence_artifact.sha256,
        inspected_render_image_sha256s=(visual_bindings["render.png"].sha256,),
        disposition="accept",
        reviewer=ProducerIdentity(
            producer_id="asset-outer-coordinator",
            role="outer_coordinator",
            implementation="asset-articulation-review-v1",
            implementation_digest="4" * 64,
        ),
        findings=("Exact graph, readback, and OVRTX evidence agree.",),
    )
    post_review_path = review_attempt / "post_review_patch.json"
    atomic_write_json(post_review_path, post_review)
    review_invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.review.v1",
        attempt_root=str(review_attempt.resolve()),
        author_result=author_result_binding,
        evidence_result=evidence_result_binding,
        post_review_patch=_execution_binding(post_review_path),
    )
    review_invocation_path = review_attempt / "invocation.json"
    atomic_write_json(review_invocation_path, review_invocation)
    review_result = run_articulation_review_asset_leaf(review_invocation_path)
    assert review_result.native_disposition == "passed"

    publish_attempt = tmp_path / "publish-attempt"
    publish_attempt.mkdir()
    publish_invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.publish.v1",
        attempt_root=str(publish_attempt.resolve()),
        author_result=author_result_binding,
        evidence_result=evidence_result_binding,
        review_result=_execution_binding(
            review_attempt / "articulation_leaf_result.json"
        ),
    )
    publish_invocation_path = publish_attempt / "invocation.json"
    atomic_write_json(publish_invocation_path, publish_invocation)
    publish_result = run_articulation_publish_asset_leaf(publish_invocation_path)
    assert publish_result.native_disposition == "passed"
    terminal = StandaloneArticulationTerminalReceipt.model_validate_json(
        Path(publish_result.native_terminal_receipt.path).read_bytes()
    )
    assert terminal.status == "completed"
    assert terminal.output_asset == result.output
    assert terminal.decision_patch.path == str(native_patch_path)
    assert len(terminal.refinement_history) == int(refine_once)
    if refine_once:
        assert Path(terminal.refinement_history[0].path).name == "issue_packet.json"
        cleanup = StandaloneArticulationCleanupReceipt.model_validate_json(
            (
                publish_attempt / "native/standalone_articulation_cleanup_receipt.json"
            ).read_bytes()
        )
        assert terminal.refinement_history[0] in cleanup.retained_artifacts
    ledger = (
        standalone_decision.StandaloneArticulationDecisionLedger.model_validate_json(
            native_patch_path.with_name(
                "articulation_decision_ledger.json"
            ).read_bytes()
        )
    )
    assert terminal.decision_patch == ledger.decision_patch

    focused = {
        item.descriptor.leaf_id: item
        for item in articulation_asset_leaf_runtime_bindings()
    }
    for leaf_id, leaf_invocation, leaf_result, leaf_invocation_path, result_path in (
        (
            "articulation.evidence.v1",
            evidence_invocation,
            evidence_result,
            evidence_invocation_path,
            evidence_attempt / "articulation_leaf_result.json",
        ),
        (
            "articulation.review.v1",
            review_invocation,
            review_result,
            review_invocation_path,
            review_attempt / "articulation_leaf_result.json",
        ),
        (
            "articulation.publish.v1",
            publish_invocation,
            publish_result,
            publish_invocation_path,
            publish_attempt / "articulation_leaf_result.json",
        ),
    ):
        projection = focused[leaf_id].project(
            leaf_invocation,
            leaf_result,
            invocation_artifact=_execution_binding(leaf_invocation_path),
            result_artifact=_execution_binding(result_path),
        )
        leaf_root = leaf_invocation_path.parent
        assert all(
            Path(item.path).is_relative_to(leaf_root)
            for item in (
                projection.payload.native_terminal_receipt,
                *projection.payload.evidence,
                *projection.payload.saved_stage_readbacks,
            )
        )

    drift_attempt = tmp_path / "author-drift-attempt"
    drift_attempt.mkdir()
    drift_patch_path = drift_attempt / "articulation_decision_patch.json"
    drift_invocation = invocation.model_copy(
        update={
            "attempt_root": str(drift_attempt.resolve()),
            "decision_patch_path": str(drift_patch_path.resolve()),
        }
    )
    drift_invocation_path = drift_attempt / "invocation.json"
    atomic_write_json(drift_invocation_path, drift_invocation)
    drift_progress = run_articulation_author_asset_leaf(drift_invocation_path)
    assert isinstance(drift_progress, ArticulationAuthorLeafProgress)
    atomic_write_json(
        drift_patch_path,
        _accepted_patch(drift_attempt / "native", preparation).model_dump(
            mode="json",
            exclude_defaults=True,
        ),
    )
    real_apply = asset_leaf_adapter.apply_standalone_articulation_decision_patch

    def apply_with_semantic_drift(*args: Any, **kwargs: Any) -> ArticulationRunState:
        state = real_apply(*args, **kwargs)
        assert state.standalone_decision_patch is not None
        normalized_path = Path(state.standalone_decision_patch.path)
        normalized = StandaloneArticulationDecisionPatch.model_validate_json(
            normalized_path.read_bytes()
        )
        normalized_path.write_text(
            normalized.model_copy(
                update={"rationale": "Semantically drifted after validation."}
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        return state

    monkeypatch.setattr(
        asset_leaf_adapter,
        "apply_standalone_articulation_decision_patch",
        apply_with_semantic_drift,
    )
    drift_result = run_articulation_author_asset_leaf(
        drift_invocation_path,
        client=_GraphOnlyClient(),
    )
    assert isinstance(drift_result, ArticulationFocusedLeafResult)
    assert drift_result.native_disposition == "failed"
    assert drift_result.error is not None
    assert "normalized decision patch changed outer semantics" in drift_result.error

    (attempt / "articulation_leaf_result.json").unlink()
    phase_receipt_path.write_bytes(phase_receipt_bytes + b" ")
    tampered_receipt_result = run_articulation_author_asset_leaf(
        invocation_path,
        client=_GraphOnlyClient(),
    )
    assert isinstance(tampered_receipt_result, ArticulationFocusedLeafResult)
    assert tampered_receipt_result.native_disposition == "failed"
    assert tampered_receipt_result.error is not None
    assert "differs from the exact expected receipt" in tampered_receipt_result.error


def test_asset_author_leaf_returns_bounded_refinement_pause(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    source_bytes = source.read_bytes()
    preparation = _preparation(tmp_path, source)
    preparation_path = tmp_path / "embedded_articulation_preparation.json"
    atomic_write_json(preparation_path, preparation)
    evidence_binding = preparation.source_hierarchy.artifacts[-1]
    publication = ArticulationPreparationPublication(
        retained_root=str(tmp_path.resolve()),
        readback=evidence_binding,
        source=_execution_binding(source),
        dependencies=(),
        configuration=evidence_binding,
        inspector_implementation=evidence_binding,
        saved_stage=_execution_binding(source),
        renders=(evidence_binding,),
        scene_artifacts=(evidence_binding,),
        retained_closure_digest=canonical_json_digest(
            {"fixture": "asset-author-refinement"}
        ),
        preparation=_execution_binding(preparation_path),
        preparation_digest=canonical_json_digest(preparation),
        evidence_provider=preparation.evidence_provider,
        publisher_implementation_sha256="2" * 64,
    )
    publication_path = tmp_path / "articulation_preparation_publication.json"
    atomic_write_json(publication_path, publication)
    attempt = tmp_path / "author-refinement-attempt"
    attempt.mkdir()
    initial_patch_path = attempt / "articulation_decision_patch.json"
    invocation = ArticulationFocusedLeafInvocation(
        leaf_id="articulation.author.v1",
        attempt_root=str(attempt.resolve()),
        source=_execution_binding(source),
        preparation_publication=_execution_binding(publication_path),
        decision_patch_path=str(initial_patch_path.resolve()),
        intent="Refine once, then author the exact source-backed drawer slide.",
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    first = run_articulation_author_asset_leaf(invocation_path)
    assert isinstance(first, ArticulationAuthorLeafProgress)
    accepted = _accepted_patch(attempt / "native", preparation)
    evidence_ids = accepted.evidence_requirements
    initial = StandaloneArticulationDecisionPatch(
        expected_state_revision=accepted.expected_state_revision,
        identity_digest=accepted.identity_digest,
        request_sha256=accepted.request_sha256,
        preparation_sha256=accepted.preparation_sha256,
        source_sha256=accepted.source_sha256,
        source_dependency_bundle_sha256=(accepted.source_dependency_bundle_sha256),
        configuration_sha256=accepted.configuration_sha256,
        disposition="revise",
        candidate_decisions=(
            StandaloneArticulationCandidateDecision(
                candidate_id="drawer_slide",
                disposition="refine",
                evidence_ids=evidence_ids,
                issue_codes=("membership_unresolved",),
                rationale="Confirm the one source-owner membership boundary.",
            ),
        ),
        evidence_requirements=evidence_ids,
        rationale="Request exactly one candidate-bound refinement.",
    )
    atomic_write_json(initial_patch_path, initial)
    initial_patch_bytes = initial_patch_path.read_bytes()

    second = run_articulation_author_asset_leaf(invocation_path)

    assert isinstance(second, ArticulationAuthorLeafProgress)
    assert second.native_status == "awaiting_decision"
    assert second.decision_patch_path.endswith(
        "refinement/attempt-001/articulation_decision_patch.json"
    )
    assert Path(second.observation.path).name == (
        "articulation_author_refinement_observation-01.json"
    )
    assert not (attempt / "articulation_leaf_result.json").exists()
    assert source.read_bytes() == source_bytes
    state = ArticulationRunState.model_validate_json(
        (attempt / "native/checkpoint.json").read_bytes()
    )
    assert len(state.standalone_refinement_history) == 1
    assert state.standalone_decision_patch is not None
    refined_observation = StandaloneArticulationObservation.model_validate_json(
        Path(second.observation.path).read_bytes()
    )
    replacement = accepted.model_copy(
        update={
            "expected_state_revision": refined_observation.state_revision,
            "refinement_attempt": 1,
            "parent_patch_sha256": state.standalone_decision_patch.sha256,
            "issue_packet_sha256": state.standalone_refinement_history[-1].sha256,
        }
    )
    atomic_write_json(Path(second.decision_patch_path), replacement)

    result = run_articulation_author_asset_leaf(
        invocation_path,
        client=_GraphOnlyClient(),
    )

    assert isinstance(result, ArticulationFocusedLeafResult)
    assert result.native_disposition == "passed"
    assert result.native_status == "awaiting_post_review"
    assert initial_patch_path.read_bytes() == initial_patch_bytes
    assert source.read_bytes() == source_bytes
    evidence_names = {Path(item.path).name for item in result.evidence}
    assert "articulation_author_observation.json" in evidence_names
    assert "articulation_author_refinement_observation-01.json" in evidence_names
    assert "issue_packet.json" in evidence_names


def _accepted_single_owner_patch(
    run_dir: Path,
    preparation: StandaloneArticulationPreparation,
) -> StandaloneArticulationDecisionPatch:
    return _accepted_patch(run_dir, preparation).model_copy(
        update={
            "canonical_graph": _single_owner_graph(preparation),
            "rationale": (
                "Accept the exact source-backed graph and explicit rigid-body "
                "membership promotion."
            ),
        }
    )


@pytest.mark.parametrize(
    ("violation", "request_updates", "error_match"),
    (
        (
            "motion",
            {"allowed_motion_types": ("prismatic",)},
            "motion types outside the request scope",
        ),
        (
            "expected_count",
            {"expected_candidate_count": 2},
            "expected_candidate_count",
        ),
        (
            "max_count",
            {"max_candidate_count": 1},
            "max_candidate_count",
        ),
    ),
)
def test_standalone_request_scope_is_enforced_before_graph_freeze(
    tmp_path: Path,
    violation: str,
    request_updates: dict[str, object],
    error_match: str,
) -> None:
    case_dir = tmp_path / violation
    case_dir.mkdir()
    source = _source(case_dir)
    run_dir = case_dir / "run"
    preparation = _preparation(case_dir, source)
    request = _request(source, run_dir).model_copy(update=request_updates)
    prepare_standalone_articulation_workflow(
        request,
        mode="batch",
        preparation=preparation,
    )
    patch = _accepted_patch(run_dir, preparation)
    graph = patch.canonical_graph
    assert graph is not None
    if violation == "motion":
        graph = graph.model_copy(
            update={
                "joints": (
                    graph.joints[0].model_copy(update={"joint_type": "revolute"}),
                )
            }
        )
        patch = patch.model_copy(update={"canonical_graph": graph})
    elif violation == "max_count":
        second_id = "drawer_slide_2"
        graph = graph.model_copy(
            update={
                "candidate_ids": (graph.candidate_ids[0], second_id),
                "joints": (
                    graph.joints[0],
                    graph.joints[0].model_copy(update={"joint_id": second_id}),
                ),
            }
        )
        patch = patch.model_copy(
            update={
                "candidate_decisions": (
                    *patch.candidate_decisions,
                    patch.candidate_decisions[0].model_copy(
                        update={"candidate_id": second_id}
                    ),
                ),
                "canonical_graph": graph,
            }
        )
    client = _GraphOnlyClient()

    with pytest.raises(StandaloneArticulationError, match=error_match):
        apply_standalone_articulation_decision_patch(
            run_dir,
            patch,
            client=client,
        )

    assert client.author_calls == 0
    assert not (run_dir / "articulation_decision_patch.json").exists()
    assert not (run_dir / "articulation_decision_ledger.json").exists()
    assert not (run_dir / "canonical_articulation_graph.json").exists()


@pytest.mark.parametrize(
    ("failure_site", "issue_code"),
    (("author", "frame_unresolved"), ("validate", "readback_mismatch")),
)
def test_execution_adapter_exception_refines_then_publishes_terminal_receipt(
    tmp_path: Path,
    failure_site: str,
    issue_code: str,
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    patch = _accepted_patch(run_dir, preparation)
    client = _RaisingExecutionClient(failure_site)

    first = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=client,
    )

    assert first.phase == "awaiting_decision"
    assert len(first.standalone_refinement_history) == 1
    issue_packet = json.loads(
        Path(first.standalone_refinement_history[-1].path).read_text(encoding="utf-8")
    )
    assert issue_packet["candidate_issue_codes"] == {"drawer_slide": [issue_code]}

    observation = build_standalone_articulation_observation(run_dir)
    replacement = patch.model_copy(
        update={
            "expected_state_revision": observation.state_revision,
            "refinement_attempt": 1,
            "parent_patch_sha256": first.standalone_decision_patch.sha256,
            "issue_packet_sha256": first.standalone_refinement_history[-1].sha256,
        }
    )
    terminal = apply_standalone_articulation_decision_patch(
        run_dir,
        replacement,
        client=client,
    )

    assert terminal.phase == "conditional"
    assert terminal.standalone_terminal_receipt is not None
    receipt = json.loads(
        Path(terminal.standalone_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "non_success"
    assert receipt["terminal_disposition"] == "refinement_cap"
    assert receipt["issue_codes"] == [issue_code]
    assert receipt["source_mutated"] is False


def test_rigid_link_plan_projection_exception_enters_bounded_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _single_owner_source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _single_owner_preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise ValueError("accepted membership projection failed")

    monkeypatch.setattr(
        standalone_decision,
        "_accepted_rigid_link_authoring_plan",
        fail_projection,
    )

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        _accepted_single_owner_patch(run_dir, preparation),
        client=_GraphOnlyClient(),
    )

    assert state.phase == "awaiting_decision"
    issue_packet = json.loads(
        Path(state.standalone_refinement_history[-1].path).read_text(encoding="utf-8")
    )
    assert issue_packet["candidate_issue_codes"] == {
        "drawer_slide": ["membership_unresolved"]
    }


def test_bound_rigid_link_plan_compares_joint_ids_in_canonical_order(
    tmp_path: Path,
) -> None:
    fixture = (
        Path(__file__).resolve().parents[4]
        / "agentic/benchmark/content_benchmark/workflows/joint/fixtures"
        / "hospital_bed_single_owner_caster_v1"
    )
    source = fixture / "source.usda"
    run_dir = tmp_path / "run"
    preparation = StandaloneArticulationPreparation.model_validate_json(
        (fixture / "preparation.json").read_bytes()
    )
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    state = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    graph = _hospital_graph_from_preparation(preparation)
    graph = graph.model_copy(
        update={
            "candidate_ids": tuple(reversed(graph.candidate_ids)),
            "joints": tuple(reversed(graph.joints)),
        }
    )
    graph_path = run_dir / "nonlexicographic_graph.json"
    atomic_write_json(graph_path, graph)
    graph_binding = ArtifactBinding(
        path=str(graph_path.resolve()),
        sha256=file_sha256(graph_path),
    )
    plan_binding = standalone_decision._accepted_rigid_link_authoring_plan(
        run_dir,
        state,
        graph,
        graph_binding,
    )
    assert plan_binding is not None

    plan = articulation_client._load_bound_accepted_authoring_plan(
        SimpleNamespace(
            accepted_authoring_plan_path=plan_binding.path,
            accepted_authoring_plan_sha256=plan_binding.sha256,
            source_sha256=state.source_sha256,
            source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
            accepted_candidate_ids=graph.candidate_ids,
        )
    )

    assert tuple(item.topology.joint_id for item in plan.plan.joints) == (
        "caster_front_left_swivel",
        "caster_front_left_wheel_roll",
    )


def test_promoted_body_must_own_itself_without_exact_membership_rows(
    tmp_path: Path,
) -> None:
    source = _single_owner_source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _single_owner_preparation(tmp_path, source)
    preparation = preparation.model_copy(
        update={
            "authoritative_owners": preparation.authoritative_owners.model_copy(
                update={
                    "facts": {
                        "authoritative_owner_prims": ["/Assembly/Base"],
                    }
                }
            )
        }
    )
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    graph = _single_owner_graph(preparation)
    drawer = "/Assembly/Base/Drawer"
    graph = graph.model_copy(
        update={
            "groups": (
                graph.groups[0].model_copy(
                    update={"member_prims": ("/Assembly/Base", drawer)}
                ),
                graph.groups[1],
            ),
            "memberships": (
                graph.memberships[0],
                graph.memberships[1].model_copy(
                    update={
                        "authoritative_owner_prim": "/Assembly/Base",
                        "group_id": graph.groups[0].group_id,
                    }
                ),
            ),
        }
    )
    patch = _accepted_single_owner_patch(run_dir, preparation).model_copy(
        update={"canonical_graph": graph}
    )

    with pytest.raises(
        EmbeddedArticulationError,
        match="promoted body must own itself",
    ):
        apply_standalone_articulation_decision_patch(
            run_dir,
            patch,
            client=_GraphOnlyClient(),
        )


def test_standalone_handoff_recomputes_source_dependency_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _single_owner_source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _single_owner_preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    monkeypatch.setattr(
        articulation_workflow,
        "_source_identity",
        lambda _source: (preparation.source_sha256, "f" * 64),
    )

    with pytest.raises(
        StandaloneArticulationError,
        match="source dependency identity changed",
    ):
        build_standalone_articulation_observation(run_dir)


def test_standalone_child_patch_freezes_then_authors_exact_graph(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepared = prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    assert prepared.status == "awaiting_decision"
    client = _GraphOnlyClient()

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        _accepted_patch(run_dir, preparation),
        client=client,
    )

    assert state.schema_version == ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
    assert state.phase == "awaiting_post_review"
    assert client.author_calls == 1
    assert state.accepted_candidate_ids == ("drawer_slide",)
    assert state.standalone_decision_patch is not None
    assert state.standalone_decision_ledger is not None
    assert state.standalone_canonical_graph is not None
    assert state.standalone_authoring_receipt is not None
    assert state.standalone_readback is not None
    assert (run_dir / "articulation_decision_patch.json").is_file()
    assert (run_dir / "articulation_decision_ledger.json").is_file()
    assert (run_dir / "canonical_articulation_graph.json").is_file()
    assert state.inference_result is None
    assert state.candidate_document is None


def test_standalone_authors_joints_while_preserving_explicit_fixed_membership(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    preparation = preparation.model_copy(
        update={
            "authoritative_owners": preparation.authoritative_owners.model_copy(
                update={
                    "facts": {
                        "authoritative_owner_prims": [
                            "/Assembly/Base",
                            "/Assembly/Drawer",
                        ],
                        "membership_policy": "retained-explicit-membership-v1",
                        "membership_rows": [
                            {
                                "member_prim": "/Assembly/Base",
                                "authoritative_owner_prim": "/Assembly/Base",
                                "disposition": "explicit_fixed",
                            },
                            {
                                "member_prim": "/Assembly/Drawer",
                                "authoritative_owner_prim": "/Assembly/Drawer",
                                "disposition": "independent_motion",
                            },
                        ],
                    }
                }
            )
        }
    )
    graph = _graph(preparation)
    graph = graph.model_copy(
        update={
            "memberships": (
                graph.memberships[0].model_copy(
                    update={"disposition": "explicit_fixed"}
                ),
                graph.memberships[1],
            )
        }
    )
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    patch = _accepted_patch(run_dir, preparation).model_copy(
        update={"canonical_graph": graph}
    )
    client = _GraphOnlyClient()

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=client,
    )

    assert state.phase == "awaiting_post_review"
    assert client.author_calls == 1
    assert state.standalone_readback is not None
    readback = json.loads(
        Path(state.standalone_readback.path).read_text(encoding="utf-8")
    )
    assert readback["memberships"][0]["disposition"] == "explicit_fixed"
    assert readback["exact_membership_match"] is True


def test_single_owner_patch_authors_accepted_rigid_body_and_exact_joint(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    source = _single_owner_source(tmp_path)
    source_bytes = source.read_bytes()
    source_sha256 = file_sha256(source)
    run_dir = tmp_path / "run"
    preparation = _single_owner_preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        _accepted_single_owner_patch(run_dir, preparation),
        client=JointAgentGraphAuthoringClient(),
    )

    assert state.phase == "awaiting_post_review"
    assert source.read_bytes() == source_bytes
    assert file_sha256(source) == source_sha256
    assert state.standalone_authoring_receipt is not None
    receipt = json.loads(
        Path(state.standalone_authoring_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["rigid_link_operation_ids"] == ["promote-drawer-rigid-body"]
    assert receipt["accepted_authoring_plan"]["path"] == str(
        (run_dir / "accepted_joint_rigger_input.json").resolve()
    )
    assert receipt["membership_operation_receipt"]["path"] == str(
        (
            run_dir / "joint_rigger" / "accepted_membership_operation_receipt.json"
        ).resolve()
    )
    assert receipt["masses_authored"] is False
    assert receipt["colliders_authored"] is False
    assert receipt["drives_authored"] is False

    output = Path(receipt["output_asset"]["path"])
    source_stage = Usd.Stage.Open(str(source))
    output_stage = Usd.Stage.Open(str(output))
    assert source_stage is not None
    assert output_stage is not None
    drawer = "/Assembly/Base/Drawer"
    assert not source_stage.GetPrimAtPath(drawer).HasAPI(UsdPhysics.RigidBodyAPI)
    assert output_stage.GetPrimAtPath(drawer).HasAPI(UsdPhysics.RigidBodyAPI)
    authored_joints = [
        prim for prim in output_stage.Traverse() if prim.IsA(UsdPhysics.PrismaticJoint)
    ]
    assert len(authored_joints) == 1
    joint = UsdPhysics.PrismaticJoint(authored_joints[0])
    assert tuple(str(item) for item in joint.GetBody0Rel().GetTargets()) == (
        "/Assembly/Base",
    )
    assert tuple(str(item) for item in joint.GetBody1Rel().GetTargets()) == (drawer,)
    assert state.standalone_readback is not None
    readback = json.loads(
        Path(state.standalone_readback.path).read_text(encoding="utf-8")
    )
    [operation] = readback["rigid_link_operations"]
    assert operation["body_prim_path"] == drawer
    assert operation["operation"] == "apply_rigid_body_membership"
    assert operation["operation_id"] == "promote-drawer-rigid-body"
    assert operation["previous_authoritative_owner_prim"] == "/Assembly/Base"
    assert operation["disposition"] == "independent_motion"
    assert operation["state"] == "source_backed"
    assert operation["evidence_ids"]


def test_accepted_membership_derivative_rejects_self_attested_tampering(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom
    from world_understanding.functions.physics.joint_rigger import JointRiggerInputV2

    source = _single_owner_source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _single_owner_preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    apply_standalone_articulation_decision_patch(
        run_dir,
        _accepted_single_owner_patch(run_dir, preparation),
        client=JointAgentGraphAuthoringClient(),
    )
    request = ArticulationAuthoringRequest.model_validate_json(
        (run_dir / "authoring_request.json").read_bytes()
    )
    accepted_plan = JointRiggerInputV2.model_validate_json(
        (run_dir / "accepted_joint_rigger_input.json").read_bytes()
    )
    prepared = run_dir / "joint_rigger" / "accepted_membership_source.usda"
    receipt_path = (
        run_dir / "joint_rigger" / "accepted_membership_operation_receipt.json"
    )
    stage = Usd.Stage.Open(str(prepared))
    assert stage is not None
    UsdGeom.Xform.Define(stage, "/Injected")
    assert stage.GetRootLayer().Save()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["sha256"] = file_sha256(prepared)
    atomic_write_json(receipt_path, receipt)

    with pytest.raises(
        ValueError,
        match="differs from the exact trusted source-plus-operation projection",
    ):
        articulation_client._validate_accepted_membership_derivative(
            request,
            accepted_plan,
        )


def test_standalone_post_authoring_entrypoints_reject_embedded_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded_state = SimpleNamespace(
        schema_version=ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
        phase="awaiting_post_review",
    )
    monkeypatch.setattr(standalone_decision, "_state", lambda _root: embedded_state)

    with pytest.raises(
        StandaloneArticulationError,
        match="Standalone output evidence binding requires awaiting_post_review",
    ):
        bind_standalone_articulation_output_evidence(
            tmp_path,
            tmp_path / "canonical_visual_envelope.json",
        )

    with pytest.raises(
        StandaloneArticulationError,
        match="Standalone post review requires awaiting_post_review",
    ):
        apply_standalone_articulation_post_review(tmp_path, None)  # type: ignore[arg-type]


def test_sanitized_hospital_bed_authors_source_evidence_graph_without_source_mutation(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    fixture = (
        Path(__file__).resolve().parents[4]
        / "agentic/benchmark/content_benchmark/workflows/joint/fixtures"
        / "hospital_bed_single_owner_caster_v1"
    )
    source = fixture / "source.usda"
    source_bytes = source.read_bytes()
    preparation = StandaloneArticulationPreparation.model_validate_json(
        (fixture / "preparation.json").read_bytes()
    )
    graph = _hospital_graph_from_preparation(preparation)
    run_dir = tmp_path / "hospital-bed-run"
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    observation = build_standalone_articulation_observation(run_dir)
    evidence_ids = tuple(
        record["evidence_id"] for record in observation.evidence_records
    )
    initial_patch = StandaloneArticulationDecisionPatch(
        expected_state_revision=observation.state_revision,
        identity_digest=observation.identity_digest,
        request_sha256=observation.request_sha256,
        preparation_sha256=observation.preparation_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        configuration_sha256=observation.configuration_sha256,
        disposition="revise",
        candidate_decisions=(
            StandaloneArticulationCandidateDecision(
                candidate_id=graph.candidate_ids[0],
                disposition="refine",
                evidence_ids=(
                    "joint-authoritative-owner-inspection",
                    "joint-authoring-capabilities",
                ),
                issue_codes=("membership_unresolved",),
                rationale=(
                    "Confirm the single-owner membership boundary before authoring."
                ),
            ),
            StandaloneArticulationCandidateDecision(
                candidate_id=graph.candidate_ids[1],
                disposition="accept",
                evidence_ids=evidence_ids,
                rationale="The frozen source-backed oracle supplies every fact.",
            ),
        ),
        evidence_requirements=evidence_ids,
        rationale="Resolve the one focused body-membership question first.",
    )
    client = JointAgentGraphAuthoringClient()
    first = apply_standalone_articulation_decision_patch(
        run_dir,
        initial_patch,
        client=client,
    )
    assert first.phase == "awaiting_decision"
    assert first.standalone_decision_patch is not None
    assert len(first.standalone_refinement_history) == 1
    assert source.read_bytes() == source_bytes
    original_patch_bytes = (run_dir / "articulation_decision_patch.json").read_bytes()
    refined = build_standalone_articulation_observation(run_dir)
    assert refined.refinement_attempt == 1
    assert refined.issue_packet is not None
    assert refined.issue_packet["candidate_issue_codes"] == {
        graph.candidate_ids[0]: ["membership_unresolved"]
    }
    patch = StandaloneArticulationDecisionPatch(
        expected_state_revision=refined.state_revision,
        refinement_attempt=1,
        parent_patch_sha256=first.standalone_decision_patch.sha256,
        issue_packet_sha256=first.standalone_refinement_history[-1].sha256,
        identity_digest=refined.identity_digest,
        request_sha256=refined.request_sha256,
        preparation_sha256=refined.preparation_sha256,
        source_sha256=refined.source_sha256,
        source_dependency_bundle_sha256=refined.source_dependency_bundle_sha256,
        configuration_sha256=refined.configuration_sha256,
        disposition="accept",
        candidate_decisions=tuple(
            StandaloneArticulationCandidateDecision(
                candidate_id=candidate_id,
                disposition="accept",
                evidence_ids=evidence_ids,
                rationale="Focused evidence confirms every frozen graph fact.",
            )
            for candidate_id in graph.candidate_ids
        ),
        canonical_graph=graph,
        evidence_requirements=evidence_ids,
        rationale="Accept the refined independently frozen hospital-bed graph.",
    )

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=client,
    )

    assert state.phase == "awaiting_post_review"
    assert (run_dir / "articulation_decision_patch.json").read_bytes() == (
        original_patch_bytes
    )
    assert (
        run_dir / "refinement/attempt-001/articulation_decision_patch.json"
    ).is_file()
    assert source.read_bytes() == source_bytes
    assert state.standalone_authoring_receipt is not None
    receipt = json.loads(
        Path(state.standalone_authoring_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["accepted_candidate_ids"] == list(graph.candidate_ids)
    assert receipt["rigid_link_operation_ids"] == [
        "promote-hospital-member-1",
        "promote-hospital-member-2",
    ]
    membership_receipt = json.loads(
        Path(receipt["membership_operation_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert membership_receipt["source_mutated"] is False
    assert membership_receipt["operation_paths"] == [
        operation.body_prim_path for operation in graph.rigid_link_operations
    ]
    output_stage = Usd.Stage.Open(receipt["output_asset"]["path"])
    assert output_stage is not None
    expected_bodies = {
        "/HospitalBed",
        "/HospitalBed/CasterFrontLeft/Fork",
        "/HospitalBed/CasterFrontLeft/Fork/Wheel",
    }
    assert {
        str(prim.GetPath())
        for prim in output_stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    } == expected_bodies
    authored_joints = [
        prim for prim in output_stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    assert len(authored_joints) == 2
    assert state.standalone_readback is not None
    readback = json.loads(
        Path(state.standalone_readback.path).read_text(encoding="utf-8")
    )
    assert readback["joint_ids"] == list(graph.candidate_ids)
    assert readback["rigid_link_operations"] == [
        operation.model_dump(mode="json") for operation in graph.rigid_link_operations
    ]


def test_standalone_refinement_is_candidate_bound_and_capped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    observation = build_standalone_articulation_observation(run_dir)
    evidence_ids = tuple(
        record["evidence_id"] for record in observation.evidence_records
    )
    initial = StandaloneArticulationDecisionPatch(
        expected_state_revision=observation.state_revision,
        identity_digest=observation.identity_digest,
        request_sha256=observation.request_sha256,
        preparation_sha256=observation.preparation_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        configuration_sha256=observation.configuration_sha256,
        disposition="revise",
        candidate_decisions=(
            StandaloneArticulationCandidateDecision(
                candidate_id="drawer_slide",
                disposition="refine",
                evidence_ids=("joint-authoritative-owner-inspection",),
                issue_codes=("parent_unresolved",),
                rationale="The authoritative parent needs focused evidence.",
            ),
        ),
        evidence_requirements=evidence_ids,
        rationale="Request only focused parent evidence.",
    )
    client = _GraphOnlyClient()
    first = apply_standalone_articulation_decision_patch(
        run_dir, initial, client=client
    )
    assert first.phase == "awaiting_decision"
    assert len(first.standalone_refinement_history) == 1
    assert client.author_calls == 0
    original_patch_bytes = (run_dir / "articulation_decision_patch.json").read_bytes()

    refined_observation = build_standalone_articulation_observation(run_dir)
    assert refined_observation.refinement_attempt == 1
    assert refined_observation.issue_packet is not None
    replacement = initial.model_copy(
        update={
            "expected_state_revision": refined_observation.state_revision,
            "refinement_attempt": 1,
            "parent_patch_sha256": first.standalone_decision_patch.sha256,
            "issue_packet_sha256": first.standalone_refinement_history[-1].sha256,
        }
    )
    wrong_scope = replacement.model_copy(
        update={
            "candidate_decisions": (
                replacement.candidate_decisions[0].model_copy(
                    update={"candidate_id": "unbounded_candidate"}
                ),
            )
        }
    )
    with pytest.raises(
        StandaloneArticulationError,
        match="changes the bounded candidate scope",
    ):
        apply_standalone_articulation_decision_patch(
            run_dir,
            wrong_scope,
            client=client,
        )

    alternate_evidence = next(
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id != "joint-authoritative-owner-inspection"
    )
    missing_required_evidence = replacement.model_copy(
        update={"evidence_requirements": (alternate_evidence,)}
    )
    with pytest.raises(
        StandaloneArticulationError,
        match="omits required issue-packet evidence",
    ):
        apply_standalone_articulation_decision_patch(
            run_dir,
            missing_required_evidence,
            client=client,
        )

    missing_required_citation = replacement.model_copy(
        update={
            "candidate_decisions": (
                replacement.candidate_decisions[0].model_copy(
                    update={"evidence_ids": (alternate_evidence,)}
                ),
            )
        }
    )
    with pytest.raises(
        StandaloneArticulationError,
        match="omits required issue-packet evidence",
    ):
        apply_standalone_articulation_decision_patch(
            run_dir,
            missing_required_citation,
            client=client,
        )
    terminal = apply_standalone_articulation_decision_patch(
        run_dir,
        replacement,
        client=client,
    )

    assert terminal.phase == "conditional"
    assert "cap exhausted" in (terminal.error or "")
    assert client.author_calls == 0
    assert (run_dir / "articulation_decision_patch.json").read_bytes() == (
        original_patch_bytes
    )
    assert (
        run_dir / "refinement" / "attempt-001" / "articulation_decision_patch.json"
    ).is_file()
    reparsed = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    assert reparsed.phase == "conditional"
    assert reparsed.standalone_terminal_receipt is not None
    receipt = json.loads(
        Path(reparsed.standalone_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "non_success"
    assert receipt["terminal_disposition"] == "refinement_cap"
    assert receipt["source_mutated"] is False
    assert receipt["issue_codes"] == ["parent_unresolved"]
    assert len(receipt["refinement_history"]) == 1
    for key in (
        "authoring_receipt",
        "output_asset",
        "readback",
        "output_evidence",
        "post_review",
        "cleanup",
    ):
        assert receipt[key] is None


def test_standalone_rejection_publishes_non_success_without_authoring(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    source_bytes = source.read_bytes()
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    observation = build_standalone_articulation_observation(run_dir)
    evidence_ids = tuple(
        record["evidence_id"] for record in observation.evidence_records
    )
    patch = StandaloneArticulationDecisionPatch(
        expected_state_revision=observation.state_revision,
        identity_digest=observation.identity_digest,
        request_sha256=observation.request_sha256,
        preparation_sha256=observation.preparation_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        configuration_sha256=observation.configuration_sha256,
        disposition="reject",
        candidate_decisions=(
            StandaloneArticulationCandidateDecision(
                candidate_id="drawer_slide",
                disposition="reject",
                evidence_ids=evidence_ids,
                rationale="The evidence does not support an authorable joint.",
            ),
        ),
        evidence_requirements=evidence_ids,
        rationale="Reject all authoring authority without mutating the source.",
    )
    client = _GraphOnlyClient()

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=client,
    )

    assert state.phase == "conditional"
    assert client.author_calls == 0
    assert source.read_bytes() == source_bytes
    assert state.standalone_terminal_receipt is not None
    receipt = json.loads(
        Path(state.standalone_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "non_success"
    assert receipt["terminal_disposition"] == "reject"
    assert receipt["source_mutated"] is False
    assert receipt["decision_patch"]["sha256"] == state.standalone_decision_patch.sha256
    assert receipt["decision_ledger"]["sha256"] == (
        state.standalone_decision_ledger.sha256
    )
    assert receipt["canonical_graph"] is None
    assert receipt["authoring_receipt"] is None
    assert receipt["output_asset"] is None


def test_saved_stage_mismatch_becomes_focused_immutable_issue_packet(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    patch = _accepted_patch(run_dir, preparation)

    state = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=_GraphOnlyClient(exact_graph_match=False),
    )

    assert state.phase == "awaiting_decision"
    assert state.accepted_candidate_ids == ()
    assert state.unresolved_candidate_ids == ("drawer_slide",)
    assert len(state.standalone_refinement_history) == 1
    observation = build_standalone_articulation_observation(run_dir)
    assert observation.refinement_attempt == 1
    assert observation.issue_packet is not None
    assert observation.issue_packet["candidate_issue_codes"] == {
        "drawer_slide": ["readback_mismatch"]
    }
    assert observation.issue_packet["failed_artifacts"]


@pytest.mark.parametrize("review_disposition", ("accept", "reject"))
def test_terminal_receipt_binds_output_review_and_complete_custody(
    tmp_path: Path,
    monkeypatch: Any,
    review_disposition: Literal["accept", "reject"],
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "run"
    preparation = _preparation(tmp_path, source)
    prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    patch = _accepted_patch(run_dir, preparation)
    state = apply_standalone_articulation_decision_patch(
        run_dir,
        patch,
        client=_GraphOnlyClient(),
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
        render_root = root / "usd_cli" / "renders"
        receipt = root / "usd_cli" / "raw" / "usd_cli_command_receipts.jsonl"
        checkpoint = (
            root / "usd_cli" / "raw" / "usd_cli_command_receipts.checkpoint.json"
        )
        render_root.mkdir(parents=True)
        receipt.parent.mkdir(parents=True)
        image_bindings: list[ExecutionArtifactBinding] = []
        response_bindings: list[ExecutionArtifactBinding] = []
        camera_bindings: list[ExecutionArtifactBinding] = []
        for index, direction in enumerate(request.views):
            image = render_root / f"canonical_output_{index}.png"
            response = render_root / f"canonical_output_{index}_response.json"
            camera = render_root / f"canonical_output_{index}_camera.json"
            # A rejected review must be able to bind the exact observed tuple
            # even when multiple requested views produced identical bytes.
            image.write_bytes(b"standalone canonical render")
            response.write_text('{"ok": true}', encoding="utf-8")
            camera.write_text(
                json.dumps({"direction": direction}),
                encoding="utf-8",
            )
            image_bindings.append(execution_artifact_binding(image))
            response_bindings.append(execution_artifact_binding(response))
            camera_bindings.append(execution_artifact_binding(camera))
        receipt.write_text('{"tool": "usd-cli"}\n', encoding="utf-8")
        checkpoint.write_text('{"receipt": "bound"}', encoding="utf-8")
        receipt_binding = execution_artifact_binding(receipt)
        checkpoint_binding = execution_artifact_binding(checkpoint)
        metadata = {
            "backend": "ovrtx",
            "renderer": "ovrtx",
            "scene_tool": "usd-cli",
            "scene_tool_source_revision": "a" * 40,
            "session_id": "standalone-articulation-test",
            "workflow": "validation-canonical-visual-evidence",
            "views": list(request.views),
            "image_width": request.image_width,
            "image_height": request.image_height,
            "renderer_identities": [None for _view in request.views],
        }
        report = {
            "schema_version": (
                "content-agent-workflows.canonical-visual-usd-cli-render-report.v1"
            ),
            "status": "completed",
            "backend": "ovrtx",
            "probe_schema_version": "usd-cli.render-probe.v1",
            "image_paths": [binding.path for binding in image_bindings],
            "images": [binding.model_dump(mode="json") for binding in image_bindings],
            "render_responses": [
                binding.model_dump(mode="json") for binding in response_bindings
            ],
            "camera_records": [
                binding.model_dump(mode="json") for binding in camera_bindings
            ],
            "usd_cli_command_receipt": receipt_binding.model_dump(mode="json"),
            "usd_cli_receipt_checkpoint": checkpoint_binding.model_dump(mode="json"),
            "metadata": metadata,
        }
        return visual._UsdCliRenderOutcome(
            report=report,
            images=tuple(image_bindings),
            responses=tuple(response_bindings),
            cameras=tuple(camera_bindings),
            receipt=receipt_binding,
            checkpoint=checkpoint_binding,
            contract_path=Path(visual.__file__).resolve(),
        )

    monkeypatch.setattr(visual, "_render_with_package_owned_usd_cli", render_output)
    visual_publication = produce_canonical_visual_evidence(
        source_usd=source,
        post_mutation_usd=output,
        output_dir=tmp_path / "canonical-visual-publication",
        backend="ovrtx",
        views=(("+x+y+z", "+x-y+z") if review_disposition == "reject" else ("+x+y+z",)),
    )
    state = bind_standalone_articulation_output_evidence(
        run_dir,
        visual_publication.envelope.path,
    )
    assert state.standalone_output_evidence is not None
    from content_agent_workflows.articulation import (
        EmbeddedArticulationOutputEvidence,
    )

    output_evidence = EmbeddedArticulationOutputEvidence.model_validate_json(
        Path(state.standalone_output_evidence.path).read_bytes()
    )
    review = StandaloneArticulationPostReviewPatch(
        identity_digest=patch.identity_digest,
        authoring_receipt_sha256=state.standalone_authoring_receipt.sha256,
        canonical_graph_sha256=state.standalone_canonical_graph.sha256,
        readback_sha256=state.standalone_readback.sha256,
        output_evidence_sha256=state.standalone_output_evidence.sha256,
        inspected_render_image_sha256s=tuple(
            image.sha256 for image in output_evidence.images
        ),
        disposition=review_disposition,
        reviewer=ProducerIdentity(
            producer_id="standalone-output-review",
            role="outer_coordinator",
            implementation="separate-articulation-output-review-v1",
            implementation_digest="2" * 64,
        ),
        findings=("Exact output render matches the accepted drawer graph.",),
    )
    state = apply_standalone_articulation_post_review(run_dir, review)
    reparsed = ArticulationRunState.model_validate_json(
        (run_dir / "checkpoint.json").read_bytes()
    )
    if review_disposition == "reject":
        assert len(set(review.inspected_render_image_sha256s)) == 1
        assert state.phase == "conditional"
        assert state.accepted_candidate_ids == ()
        assert state.rejected_candidate_ids == ("drawer_slide",)
        assert state.standalone_terminal_receipt is not None
        terminal = StandaloneArticulationTerminalReceipt.model_validate_json(
            Path(state.standalone_terminal_receipt.path).read_bytes()
        )
        assert terminal.status == "non_success"
        assert terminal.terminal_disposition == "reject"
        assert terminal.output_asset is None
        assert terminal.post_review is None
        assert {item.sha256 for item in terminal.failed_artifacts} == {
            state.standalone_authoring_receipt.sha256,
            state.standalone_readback.sha256,
            state.standalone_output_evidence.sha256,
            state.standalone_post_review.sha256,
        }
        with pytest.raises(ValueError, match="accepted render image digests"):
            StandaloneArticulationPostReviewPatch.model_validate(
                {
                    **review.model_dump(mode="json"),
                    "disposition": "accept",
                }
            )
        assert reparsed == state
        return
    assert state.phase == "completed"
    assert state.standalone_cleanup is not None
    assert state.standalone_terminal_receipt is not None

    result = write_articulation_workflow_summary(
        state,
        output_dir=run_dir,
        authoring=authoring,
    )
    assert result.status == "completed"
    assert result.success
    assert result.standalone_terminal_receipt_path is not None

    resumed = prepare_standalone_articulation_workflow(
        _request(source, run_dir),
        mode="batch",
        preparation=preparation,
    )
    assert resumed.status == "completed"
    assert resumed.success
    assert resumed.standalone_terminal_receipt_path == (
        result.standalone_terminal_receipt_path
    )
