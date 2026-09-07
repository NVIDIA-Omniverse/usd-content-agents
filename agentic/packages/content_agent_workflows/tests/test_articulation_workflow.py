# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from filelock import FileLock

import content_agent_workflows.articulation.decision as articulation_decision
from content_agent_workflows.articulation import (
    SKILL_ROUTED_DECISION_METADATA_KEY,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationCandidateDecision,
    ArticulationDecisionPatch,
    ArticulationFinalizationResult,
    ArticulationInferenceResult,
    ArticulationRunState,
    ArticulationValidationResult,
    ArticulationWorkflowError,
    ArticulationWorkflowInterrupted,
    ArticulationWorkflowRequest,
    CancelChecker,
    JointAgentInferenceTerminalError,
    JointAgentLocalClient,
    MembershipDispositionDocument,
    MockArticulationCall,
    MockArticulationSceneEvidenceCollector,
    Stage2CandidateDocument,
    apply_articulation_decision_patch,
    build_articulation_review_receipt,
    build_articulation_step_observation,
    run_batch_articulation_workflow,
    run_interactive_articulation_workflow,
    validate_completed_articulation_checkpoint,
)
from content_agent_workflows.articulation import (
    MockArticulationWorkflowClient as _RawMockArticulationWorkflowClient,
)
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_METADATA_KEY,
    DomainExecutionContext,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
    metadata_with_domain_execution_context,
)


def _candidate(
    candidate_id: str,
    index: int,
    *,
    confidence: str = "high",
    motion_type: str = "prismatic",
    review_status: str = "ready_for_rigger_input",
    unresolved_reason_codes: list[str] | None = None,
    moving_part_prim: str | None = None,
    fixed_parent_prim: str = "/World/Cabinet",
    lower_limit: float | None = None,
    upper_limit: float | None = None,
) -> dict[str, Any]:
    moving_part = moving_part_prim or f"/World/Cabinet/Drawer_{index:02d}"
    has_limits = lower_limit is not None or upper_limit is not None
    return {
        "schema_version": "joint-agent-stage2-v0",
        "candidate_id": candidate_id,
        "motion_type": motion_type,
        "joint_type_hint": motion_type,
        "moving_part_prims": [moving_part],
        "fixed_parent_prim": fixed_parent_prim,
        "parent_resolution_source": "stage1_hint",
        "axis_hint": "x",
        "motion_axis_world": [1.0, 0.0, 0.0],
        "confidence": confidence,
        "role": "drawer",
        "field_sources": {
            "motion_type": "predicted",
            "axis_hint": "predicted",
            "motion_axis_world": "predicted",
            "fixed_parent_prim": "stage1_hint",
        },
        "axis_evidence": [
            {
                "source": "predicted",
                "description": "source-bound signed axis",
                "value": "x",
                "prim_paths": [moving_part],
            }
        ],
        "connectivity_evidence": [
            {
                "source": "stage1_hint",
                "description": "source-bound parent-child edge",
                "value": fixed_parent_prim,
                "prim_paths": [fixed_parent_prim, moving_part],
                "connectivity_role": "body0_body1_edge",
            }
        ],
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "limit_unit": ("degrees" if motion_type == "revolute" else "meters")
        if has_limits
        else "unknown",
        "limit_source": "authored_metadata" if has_limits else "unknown",
        "limit_readiness": "source_backed" if has_limits else "not_provided",
        "limit_evidence": [
            {
                "source": "authored_metadata",
                "description": "source-authored joint range",
                "prim_paths": [moving_part],
            }
        ]
        if has_limits
        else [],
        "review_status": review_status,
        "unresolved_reason_codes": unresolved_reason_codes or [],
        "evidence": f"source-bound drawer candidate {index}",
    }


def _document(*candidates: dict[str, Any]) -> Stage2CandidateDocument:
    ready = sum(
        candidate["review_status"] == "ready_for_rigger_input"
        for candidate in candidates
    )
    return Stage2CandidateDocument.model_validate(
        {
            "schema_version": "joint-agent-stage2-v0",
            "summary": {
                "candidate_count": len(candidates),
                "ready_candidate_count": ready,
                "review_required_candidate_count": len(candidates) - ready,
                "joint_type_counts": {
                    "prismatic": sum(
                        candidate["motion_type"] == "prismatic"
                        for candidate in candidates
                    )
                },
            },
            "candidates": list(candidates),
        }
    )


def _accepted_independent_membership(
    candidate_document: Stage2CandidateDocument,
) -> MembershipDispositionDocument:
    records = tuple(
        {
            "disposition_id": f"membership_{candidate.candidate_id}",
            "member_prim": candidate.moving_part_prims[0],
            "motion_candidate_prim": candidate.moving_part_prims[0],
            "physical_owner_prim": candidate.moving_part_prims[0],
            "physical_owner_candidate_prim": candidate.moving_part_prims[0],
            "disposition": "independent_motion",
            "source": "accepted_manifest",
            "confidence": candidate.confidence,
            "rationale": "Accepted independent-motion test fixture.",
            "source_prediction_ids": candidate.source_prediction_ids,
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        }
        for candidate in sorted(
            (
                candidate
                for candidate in candidate_document.candidates
                if candidate.moving_part_prims
            ),
            key=lambda candidate: candidate.moving_part_prims[0],
        )
    )
    return MembershipDispositionDocument.model_validate(
        {
            "schema_version": "joint-agent-membership-disposition-v1",
            "summary": {
                "disposition_count": len(records),
                "disposition_counts": (
                    {"independent_motion": len(records)} if records else {}
                ),
                "review_required_count": 0,
                "pending_downstream_count": 0,
            },
            "dispositions": records,
        }
    )


class MockArticulationWorkflowClient(_RawMockArticulationWorkflowClient):
    """Test adapter that binds explicit accepted membership fixture evidence."""

    def __init__(
        self,
        candidate_document: Stage2CandidateDocument,
        *,
        membership_disposition_document: MembershipDispositionDocument | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            candidate_document,
            membership_disposition_document=(
                membership_disposition_document
                if membership_disposition_document is not None
                else _accepted_independent_membership(candidate_document)
            ),
            **kwargs,
        )


class _NonArticulatedWorkflowClient(MockArticulationWorkflowClient):
    """Return a provider-bound, successful empty articulation inference."""

    def __init__(
        self,
        candidate_document: Stage2CandidateDocument,
        *,
        diagnostics_path: Path,
        membership_disposition_document: MembershipDispositionDocument | None = None,
    ) -> None:
        super().__init__(
            candidate_document,
            membership_disposition_document=membership_disposition_document,
        )
        self._diagnostics_path = diagnostics_path

    def infer(
        self,
        request: ArticulationWorkflowRequest,
        *,
        resume: bool,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationInferenceResult:
        result = super().infer(
            request,
            resume=resume,
            cancel_checker=cancel_checker,
        )
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "structure_analysis_outcome": "not_articulated",
                    "structure_analysis_reasoning": (
                        "Provider-backed structure analysis found zero degrees of "
                        "freedom and no articulated segments."
                    ),
                    "structure_analysis_evidence": {
                        "accepted": True,
                        "robot_type": "hospital bed",
                        "dof": 0,
                        "segment_names": [],
                        "source_prim_inventory": ["/World/HospitalBed/Frame"],
                        "provider_response_diagnostics_path": str(
                            self._diagnostics_path.resolve()
                        ),
                        "provider_response_diagnostics_sha256": file_sha256(
                            self._diagnostics_path
                        ),
                    },
                }
            }
        )


def _request(
    tmp_path: Path,
    *,
    review_policy: str = "uncertain",
    allowed_motion_types: tuple[str, ...] = ("revolute", "prismatic"),
    expected_candidate_count: int | None = None,
) -> ArticulationWorkflowRequest:
    source = tmp_path / "sm_filecabinet_d01_01.usda"
    source.write_text(
        '#usda 1.0\n\ndef Xform "FileCabinet" {\n}\n',
        encoding="utf-8",
    )
    return ArticulationWorkflowRequest(
        source_asset=str(source),
        output_dir=tmp_path / "run",
        intent=(
            "Analyze this file cabinet, review uncertain candidates, and write "
            "only approved prismatic drawer joints."
        ),
        review_policy=review_policy,
        allowed_motion_types=allowed_motion_types,
        expected_candidate_count=expected_candidate_count,
    )


def _source_identity(path: str | Path) -> tuple[str, str]:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    resolved = Path(path).resolve()
    identity = identify_usd_artifact(resolved, uri=resolved.as_uri())
    assert identity.dependency_bundle_sha256 is not None
    return identity.root_sha256, identity.dependency_bundle_sha256


def _operation_count(
    client: MockArticulationWorkflowClient,
    operation: str,
) -> int:
    return sum(call.operation == operation for call in client.calls)


def _write_joint_rigger_contract_artifacts(
    *,
    output_path: Path,
    diagnostics_path: Path,
    result_path: Path,
    joint_paths: tuple[str, ...],
    root_sha256: str | None = None,
    dependency_bundle_sha256: str | None = None,
    input_sha256: str = "1" * 64,
    plan_sha256: str = "2" * 64,
    backend_name: str = "stage2_candidate_edges",
) -> None:
    from world_understanding.functions.physics.joint_rigger import (
        ArtifactIdentityV1,
        JointDiagnosticV1,
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    if root_sha256 is None or dependency_bundle_sha256 is None:
        observed_root_sha256, observed_dependency_bundle_sha256 = _source_identity(
            output_path
        )
        root_sha256 = root_sha256 or observed_root_sha256
        dependency_bundle_sha256 = (
            dependency_bundle_sha256 or observed_dependency_bundle_sha256
        )
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version="world-understanding-joint-rigger-diagnostics-v1",
        backend_name=backend_name,
        joint_diagnostics=tuple(
            JointDiagnosticV1(joint_id=joint_path) for joint_path in joint_paths
        ),
    )
    result = JointRiggerResultV1(
        schema_version="world-understanding-joint-rigger-result-v1",
        status="succeeded",
        input_sha256=input_sha256,
        plan_sha256=plan_sha256,
        output_artifact=ArtifactIdentityV1(
            uri=output_path.resolve().as_uri(),
            root_sha256=root_sha256,
            dependency_bundle_sha256=dependency_bundle_sha256,
        ),
        diagnostics=diagnostics,
    )
    diagnostics_path.write_text(
        diagnostics.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    result_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _write_rigged_usdz(
    tmp_path: Path,
    candidate_document: Stage2CandidateDocument,
) -> tuple[Path, tuple[str, ...]]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    from content_agent_workflows.articulation.client import (
        _expected_stage2_candidate_custom_data,
    )

    source_path = tmp_path / "rigged.usda"
    output_path = tmp_path / "rigged.usdz"
    stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.Xform.Define(stage, "/World/Cabinet")
    UsdGeom.Scope.Define(stage, "/World/Joints")
    joint_paths: list[str] = []
    for candidate in candidate_document.candidates:
        assert candidate.fixed_parent_prim is not None
        moving_path = candidate.moving_part_prims[0]
        UsdGeom.Xform.Define(stage, moving_path)
        joint_path = f"/World/Joints/{candidate.candidate_id}"
        joint_paths.append(joint_path)
        schema = (
            UsdPhysics.RevoluteJoint
            if candidate.motion_type == "revolute"
            else UsdPhysics.PrismaticJoint
        )
        joint = schema.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(candidate.fixed_parent_prim)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(moving_path)])
        joint.CreateAxisAttr(candidate.axis_hint[-1].upper())
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(identity)
        joint.CreateLocalRot1Attr(identity)
        for attribute, value in (
            (joint.GetLowerLimitAttr(), candidate.lower_limit),
            (joint.GetUpperLimitAttr(), candidate.upper_limit),
        ):
            if value is not None:
                authored_value = float(value)
                if candidate.motion_type == "prismatic":
                    authored_value /= meters_per_unit
                attribute.Set(authored_value)
        prim = joint.GetPrim()
        prim.SetCustomData(_expected_stage2_candidate_custom_data(candidate))
    assert stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    return output_path, tuple(joint_paths)


def test_stage2_custom_data_missing_field_source_fails_closed(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.articulation.client import (
        _validate_saved_owned_core_graph,
    )

    complete = _candidate("candidate_0001", 1)
    complete_document = _document(complete)
    output_path, joint_paths = _write_rigged_usdz(tmp_path, complete_document)
    missing_source = copy.deepcopy(complete)
    missing_source["field_sources"].pop("motion_axis_world")
    incomplete_document = _document(missing_source)

    _, failures, _ = _validate_saved_owned_core_graph(
        output_path,
        incomplete_document,
        diagnostic_joint_paths=joint_paths,
        require_stage2_candidate_custom_data=True,
    )

    assert any("transitional customData" in failure for failure in failures)


class _ExactReadbackFixtureClient(MockArticulationWorkflowClient):
    """Use deterministic inference/authoring with the real PXR readback gate."""

    def author(
        self,
        request: ArticulationAuthoringRequest,
        *,
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationAuthoringResult:
        authored = super().author(request, cancel_checker=cancel_checker)
        if authored.metadata.get("fixture_exact_readback"):
            return authored

        candidate_document = Stage2CandidateDocument.model_validate_json(
            Path(request.candidate_document_path).read_text(encoding="utf-8")
        )
        output_path = Path(authored.output_asset_path)
        output_path.unlink()
        output_path, joint_paths = _write_rigged_usdz(
            output_path.parent,
            candidate_document,
        )
        diagnostics_path = Path(authored.diagnostics_path or "")
        result_path = Path(authored.joint_rigger_result_path or "")
        _write_joint_rigger_contract_artifacts(
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            result_path=result_path,
            joint_paths=joint_paths,
        )
        exact = authored.model_copy(
            update={
                "output_asset_sha256": hashlib.sha256(
                    output_path.read_bytes()
                ).hexdigest(),
                "diagnostics_sha256": hashlib.sha256(
                    diagnostics_path.read_bytes()
                ).hexdigest(),
                "joint_rigger_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "metadata": {
                    **authored.metadata,
                    "fixture_exact_readback": True,
                },
            }
        )
        recovery_path = output_path.parent / "workflow_authoring_result.json"
        recovery_path.write_text(
            exact.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return exact

    def validate(
        self,
        authoring: ArticulationAuthoringResult,
        *,
        expected_candidate_ids: tuple[str, ...],
        cancel_checker: CancelChecker | None = None,
    ) -> ArticulationValidationResult:
        self.calls.append(
            MockArticulationCall(
                operation="validate",
                candidate_ids=expected_candidate_ids,
            )
        )
        return JointAgentLocalClient(
            {"project": {}, "input": {}, "steps": {}}
        ).validate(
            authoring,
            expected_candidate_ids=expected_candidate_ids,
            cancel_checker=cancel_checker,
        )


def _saved_graph_authoring(
    tmp_path: Path,
    candidate_document: Stage2CandidateDocument,
    output_path: Path,
    joint_paths: tuple[str, ...],
    *,
    root_sha256: str | None = None,
    dependency_bundle_sha256: str | None = None,
) -> ArticulationAuthoringResult:
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics_path = tmp_path / "joint-rigger-diagnostics.json"
    result_path = tmp_path / "joint-rigger-result.json"
    _write_joint_rigger_contract_artifacts(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        result_path=result_path,
        joint_paths=joint_paths,
        root_sha256=root_sha256,
        dependency_bundle_sha256=dependency_bundle_sha256,
    )
    return ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(output_path),
        output_asset_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        authored_candidate_ids=candidate_document.candidate_ids,
        authored_joint_count=len(candidate_document.candidates),
        diagnostics_path=str(diagnostics_path),
        diagnostics_sha256=hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        joint_rigger_result_path=str(result_path),
        joint_rigger_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )


def _owned_core_saved_graph_authoring(
    tmp_path: Path,
    candidate_document: Stage2CandidateDocument,
    *,
    tamper_joint_id: bool = False,
    relocate_body_endpoints: bool = False,
    tamper_body0_link_id: bool = False,
    tamper_body1_link_id: bool = False,
) -> ArticulationAuthoringResult:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from world_understanding.functions.physics.joint_rigger import (
        ArtifactIdentityV1,
        FieldDecisionV1,
        FieldProvenanceV1,
        JointDiagnosticV1,
        JointRiggerDiagnosticsV1,
        JointRiggerResultV1,
    )

    _, physical_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    plan_sha256 = "2" * 64
    authoring_version = "world-understanding-joint-topology-author-v1"
    contract_artifact = ArtifactIdentityV1(
        uri="memory://joint-agent/articulation-contract/test",
        root_sha256="a" * 64,
    )
    diagnostics: list[JointDiagnosticV1] = []
    for index, (candidate, physical_path) in enumerate(
        zip(candidate_document.candidates, physical_paths, strict=True)
    ):
        assert candidate.fixed_parent_prim is not None
        logical_joint_id = f"/World/LogicalJoints/joint_{index:04d}"
        authored_body0 = candidate.fixed_parent_prim
        authored_body1 = candidate.moving_part_prims[0]
        if relocate_body_endpoints:
            authored_body0 = (
                f"{candidate.fixed_parent_prim}/__JointAgent_"
                + hashlib.sha256(
                    candidate.fixed_parent_prim.encode("utf-8")
                ).hexdigest()[:16]
            )
            UsdGeom.Xform.Define(stage, authored_body0)
            joint = UsdPhysics.Joint(stage.GetPrimAtPath(physical_path))
            joint.GetBody0Rel().SetTargets([Sdf.Path(authored_body0)])

        def provenance(
            prim_path: str,
            property_name: str,
            *,
            link_id: str | None = None,
        ) -> FieldProvenanceV1:
            properties = [f"joint:{logical_joint_id}.{property_name}"]
            if link_id is not None:
                properties.append(f"link:{link_id}.body_prim_path")
            return FieldProvenanceV1(
                source="accepted_manifest",
                artifact=contract_artifact,
                prim_path=prim_path,
                properties=tuple(properties),
                derivation="articulation_contract_v1_to_joint_rigger_input_v1",
                evidence=f"Bound {property_name} test evidence.",
            )

        decisions = (
            FieldDecisionV1(
                field="topology.body0",
                disposition="accepted",
                provenance=provenance(
                    authored_body0,
                    "body0_link",
                    link_id=(
                        "/World/tampered-body0"
                        if relocate_body_endpoints and tamper_body0_link_id
                        else candidate.fixed_parent_prim
                        if relocate_body_endpoints
                        else None
                    ),
                ),
            ),
            FieldDecisionV1(
                field="topology.body1",
                disposition="accepted",
                provenance=provenance(
                    authored_body1,
                    "body1_link",
                    link_id=(
                        "/World/tampered-body1"
                        if relocate_body_endpoints and tamper_body1_link_id
                        else candidate.moving_part_prims[0]
                        if relocate_body_endpoints
                        else None
                    ),
                ),
            ),
            FieldDecisionV1(
                field="topology.joint_type",
                disposition="accepted",
                provenance=provenance(
                    authored_body1,
                    "motion_type",
                ),
            ),
            FieldDecisionV1(
                field="topology.axis_stage",
                disposition="accepted",
                provenance=provenance(
                    authored_body1,
                    "axis_stage",
                ),
            ),
            FieldDecisionV1(
                field="usd.joint_prim_path",
                disposition="defaulted",
                reason_code="deterministic_joint_path",
                detail=physical_path,
            ),
        )
        diagnostic = JointDiagnosticV1(
            joint_id=logical_joint_id,
            field_decisions=decisions,
        )
        diagnostics.append(diagnostic)
        prim = stage.GetPrimAtPath(physical_path)
        assert prim and prim.IsValid()
        prim.ClearCustomDataByKey("jointAgent:candidateId")
        prim.ClearCustomDataByKey("jointAgent:sourceSchemaVersion")
        prim.SetCustomDataByKey(
            "jointRigger:jointId",
            "tampered-logical-id" if tamper_joint_id else logical_joint_id,
        )
        prim.SetCustomDataByKey("jointRigger:planSha256", plan_sha256)
        prim.SetCustomDataByKey(
            "jointRigger:authoringVersion",
            authoring_version,
        )
        prim.SetCustomDataByKey(
            "jointRigger:fieldDecisions",
            json.dumps(
                [
                    decision.model_dump(mode="json", exclude_none=True)
                    for decision in diagnostic.field_decisions
                ],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )

    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "owned-core-rigged.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    output_root_sha256, output_dependency_sha256 = _source_identity(output_path)
    diagnostic_document = JointRiggerDiagnosticsV1(
        schema_version="world-understanding-joint-rigger-diagnostics-v1",
        backend_name="owned_topology",
        backend_version=authoring_version,
        joint_diagnostics=tuple(diagnostics),
    )
    result = JointRiggerResultV1(
        schema_version="world-understanding-joint-rigger-result-v1",
        status="succeeded",
        input_sha256="1" * 64,
        plan_sha256=plan_sha256,
        output_artifact=ArtifactIdentityV1(
            uri=output_path.resolve().as_uri(),
            root_sha256=output_root_sha256,
            dependency_bundle_sha256=output_dependency_sha256,
        ),
        diagnostics=diagnostic_document,
    )
    candidate_path = tmp_path / "owned-core-approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics_path = tmp_path / "owned-core-diagnostics.json"
    diagnostics_path.write_text(
        diagnostic_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "owned-core-result.json"
    result_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(output_path),
        output_asset_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        authored_candidate_ids=candidate_document.candidate_ids,
        authored_joint_count=len(candidate_document.candidates),
        diagnostics_path=str(diagnostics_path),
        diagnostics_sha256=hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        joint_rigger_result_path=str(result_path),
        joint_rigger_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )


def test_authoring_idempotency_key_binds_prediction_digest(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.articulation.models import ArtifactBinding
    from content_agent_workflows.articulation.workflow import (
        _authoring_idempotency_key,
    )

    candidate_binding = ArtifactBinding(
        path=str(tmp_path / "approved.json"),
        sha256="3" * 64,
    )
    common = {
        "source_asset": str(tmp_path / "source.usda"),
        "source_sha256": "1" * 64,
        "source_dependency_bundle_sha256": "2" * 64,
        "candidate_document": candidate_binding,
        "accepted_candidate_ids": ("candidate_0001",),
        "predictions_path": str(tmp_path / "predictions.json"),
        "output_dir": tmp_path / "run",
    }

    first = _authoring_idempotency_key(
        **common,
        predictions_sha256="4" * 64,
    )
    second = _authoring_idempotency_key(
        **common,
        predictions_sha256="5" * 64,
    )

    assert first != second


def test_articulation_request_exposes_validated_embedded_execution_context(
    tmp_path: Path,
) -> None:
    binding = ExecutionArtifactBinding(
        path=str(tmp_path / "bound.json"),
        sha256="1" * 64,
        size_bytes=17,
    )
    context = DomainExecutionContext(
        domain="articulation",
        mode="embedded",
        reasoning_loop_owner="asset_coordinator",
        embedded_stage=EmbeddedStageBinding(
            outer_run_id="asset-run",
            outer_request=binding,
            stage="articulation",
            stage_attempt=2,
            coordinator_plan=binding,
            input_asset=binding,
            domain_run_root=str(tmp_path / "articulation"),
        ),
    )
    request = ArticulationWorkflowRequest(
        source_asset=str(tmp_path / "cabinet.usdz"),
        output_dir=tmp_path / "articulation",
        metadata=metadata_with_domain_execution_context({}, context),
    )

    assert request.execution_context == context
    assert "execution_context" not in request.model_dump()


def test_articulation_request_rejects_wrong_domain_execution_context(
    tmp_path: Path,
) -> None:
    metadata = {
        DOMAIN_EXECUTION_CONTEXT_METADATA_KEY: {
            "schema_version": "content-agent-workflows.domain-execution-context.v1",
            "domain": "texture",
            "mode": "standalone",
            "reasoning_loop_owner": "compatibility_pipeline",
            "embedded_stage": None,
        }
    }

    with pytest.raises(ValueError, match="does not match the workflow request"):
        ArticulationWorkflowRequest(
            source_asset=str(tmp_path / "cabinet.usdz"),
            output_dir=tmp_path / "articulation",
            metadata=metadata,
        )


def test_legacy_articulation_request_has_no_execution_context(tmp_path: Path) -> None:
    request = ArticulationWorkflowRequest(
        source_asset=str(tmp_path / "cabinet.usdz"),
        output_dir=tmp_path / "articulation",
    )

    assert request.metadata == {}
    assert request.execution_context is None
    assert DOMAIN_EXECUTION_CONTEXT_METADATA_KEY not in request.model_dump()["metadata"]


def test_file_cabinet_six_drawers_complete_with_exact_readback(
    tmp_path: Path,
) -> None:
    document = _document(
        *[_candidate(f"candidate_{index:04d}", index) for index in range(1, 7)]
    )
    client = _ExactReadbackFixtureClient(document)
    request = _request(
        tmp_path,
        review_policy="all",
        allowed_motion_types=("prismatic",),
        expected_candidate_count=6,
    )

    expected_ids = tuple(f"candidate_{index:04d}" for index in range(1, 7))
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    assert paused.review_required_candidate_ids == expected_ids
    build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept" for candidate_id in expected_ids},
        reviewer="file-cabinet-owner",
    )
    result = run_interactive_articulation_workflow(request, client=client)

    assert result.status == "completed"
    assert result.success is True
    assert result.accepted_candidate_ids == expected_ids
    assert result.unresolved_candidate_ids == ()
    assert result.output_asset_path is not None
    assert Path(result.output_asset_path).suffix == ".usdz"
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1
    assert client.calls[1].candidate_ids == expected_ids
    validation = json.loads(
        (request.output_dir / "validation_evidence.json").read_text(encoding="utf-8")
    )
    assert validation["metadata"]["backend"] == "joint-agent-owned-core"
    assert validation["validated_candidate_ids"] == list(expected_ids)
    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    receipt_payload = json.loads(
        (request.output_dir / "review_receipt.json").read_text(encoding="utf-8")
    )
    authoring_request = json.loads(
        (request.output_dir / "authoring_request.json").read_text(encoding="utf-8")
    )
    authoring_result = json.loads(
        (request.output_dir / "authoring_result.json").read_text(encoding="utf-8")
    )
    dependency_digests = {
        payload["source_dependency_bundle_sha256"]
        for payload in (
            checkpoint,
            receipt_payload,
            authoring_request,
            authoring_result,
        )
    }
    assert len(dependency_digests) == 1

    for artifact_name in (
        "request.json",
        "articulation_candidates.json",
        "review_receipt.json",
        "approved_articulation_candidates.json",
        "authoring_result.json",
        "validation_evidence.json",
        "checkpoint.json",
        "workflow_progress.json",
        "final_summary.json",
    ):
        assert (request.output_dir / artifact_name).is_file()
    from world_understanding.functions.physics.joint_rigger import JointRiggerResultV1

    JointRiggerResultV1.model_validate_json(
        (request.output_dir / "joint_rigger" / "result.json").read_text(
            encoding="utf-8"
        )
    )

    repeated = run_interactive_articulation_workflow(request, client=client)
    assert repeated == result
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1


def test_non_articulated_asset_succeeds_and_skips_downstream_leaves(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "structure-provider-responses.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attempts": [],
                "whole_asset_structure": {
                    "accepted": True,
                    "reason_codes": [],
                    "robot_type": "hospital bed",
                    "dof": 0,
                    "segment_names": [],
                    "source_prim_inventory": ["/World/HospitalBed/Frame"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = _NonArticulatedWorkflowClient(
        _document(),
        diagnostics_path=diagnostics_path,
    )
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path)

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )

    assert result.success is True
    assert result.status == "not_articulated"
    assert result.candidate_ids == ()
    assert result.inference_result_path is not None
    assert result.candidate_document_path is not None
    assert result.scene_evidence_path is None
    assert result.review_receipt_path is None
    assert result.authoring_result_path is None
    assert result.validation_result_path is None
    assert "not articulated" in result.message
    assert collector.call_count == 0
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0
    assert _operation_count(client, "validate") == 0

    repeated = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert repeated == result
    assert collector.call_count == 0
    assert _operation_count(client, "infer") == 1


def test_non_articulated_asset_rejects_mismatched_provider_journal(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "structure-provider-responses.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attempts": [],
                "whole_asset_structure": {
                    "accepted": True,
                    "reason_codes": [],
                    "robot_type": "forklift",
                    "dof": 0,
                    "segment_names": [],
                    "source_prim_inventory": ["/World/HospitalBed/Frame"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = _NonArticulatedWorkflowClient(
        _document(), diagnostics_path=diagnostics_path
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="provider-response diagnostics do not match",
    ):
        run_interactive_articulation_workflow(_request(tmp_path), client=client)


def test_non_articulated_asset_with_pending_membership_requires_review(
    tmp_path: Path,
) -> None:
    diagnostics_path = tmp_path / "structure-provider-responses.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "attempts": [],
                "whole_asset_structure": {
                    "accepted": True,
                    "reason_codes": [],
                    "robot_type": "hospital bed",
                    "dof": 0,
                    "segment_names": [],
                    "source_prim_inventory": ["/World/HospitalBed/Frame"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    membership_id = "membership_frame"
    membership_document = _membership_document(
        {
            "disposition_id": membership_id,
            "member_prim": "/World/HospitalBed/Frame",
            "motion_candidate_prim": None,
            "disposition": "unresolved",
            "source": "predicted",
            "confidence": "medium",
            "rationale": "Rigid-frame membership still requires review.",
            "source_prediction_ids": ["/World/HospitalBed/Frame"],
            "downstream_boundary": "human_review",
            "review_status": "review_required",
            "reason_codes": ["membership_evidence_conflict"],
            "unresolved_questions": ["Confirm the rigid-frame membership."],
        }
    )
    client = _NonArticulatedWorkflowClient(
        _document(),
        diagnostics_path=diagnostics_path,
        membership_disposition_document=membership_document,
    )

    result = run_interactive_articulation_workflow(_request(tmp_path), client=client)

    assert result.success is False
    assert result.status == "needs_review"
    assert result.candidate_ids == ()
    assert result.review_required_membership_disposition_ids == (membership_id,)
    assert result.unresolved_membership_disposition_ids == (membership_id,)
    assert result.authoring_result_path is None
    assert _operation_count(client, "author") == 0


def test_unproven_empty_candidate_set_remains_terminal(tmp_path: Path) -> None:
    client = MockArticulationWorkflowClient(_document())
    request = _request(tmp_path)

    with pytest.raises(
        ArticulationWorkflowError,
        match="unproven empty candidate set",
    ):
        run_interactive_articulation_workflow(request, client=client)

    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0
    assert _operation_count(client, "validate") == 0


def test_recovered_unproven_empty_candidate_set_remains_terminal(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document())
    request = _request(tmp_path)
    request.output_dir.mkdir(parents=True)
    inference = client.infer(request, resume=False)
    (request.output_dir / "inference_result.json").write_text(
        inference.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="unproven empty candidate set",
    ):
        run_interactive_articulation_workflow(request, client=client)

    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0
    assert _operation_count(client, "validate") == 0


def test_uncertain_candidate_pauses_then_resumes_without_reinference(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="medium"))
    )
    request = _request(tmp_path)

    paused = run_interactive_articulation_workflow(request, client=client)

    assert paused.status == "needs_review"
    assert paused.review_required_candidate_ids == (candidate_id,)
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0
    assert _operation_count(client, "validate") == 0

    receipt = build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept"},
        reviewer="fixture-owner",
    )
    assert json.loads(
        (request.output_dir / "review_receipt.json").read_text(encoding="utf-8")
    ) == receipt.model_dump(mode="json")
    completed = run_interactive_articulation_workflow(
        request,
        client=client,
    )

    assert completed.status == "completed"
    assert completed.accepted_candidate_ids == (candidate_id,)
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1


def test_skill_routed_articulation_patch_binds_review_and_candidate_edit(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="high"))
    )
    base_request = _request(tmp_path, review_policy="all")
    request = base_request.model_copy(
        update={
            "metadata": {
                SKILL_ROUTED_DECISION_METADATA_KEY: True,
            }
        }
    )

    paused = run_batch_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0

    with pytest.raises(
        ArticulationWorkflowError,
        match="requires a validated agent decision ledger",
    ):
        build_articulation_review_receipt(
            request.output_dir,
            {candidate_id: "accept"},
            reviewer="premature-reviewer",
        )

    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    original = client.candidate_document.candidates[0]
    edited = original.model_copy(update={"confidence": "medium"})
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                replacement_candidate=edited,
                rationale="The evidence supports the same joint with medium confidence.",
                confidence=0.8,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Reviewed hierarchy, endpoints, axis, and source evidence.",
    )
    workflow_lock = FileLock(str(request.output_dir / ".articulation-workflow.lock"))
    with workflow_lock:
        apply_articulation_decision_patch(
            request.output_dir,
            patch,
            workflow_lock=workflow_lock,
        )
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept"},
        reviewer="fixture-owner",
    )
    reviewed_path = request.output_dir / "agent_reviewed_articulation_candidates.json"
    assert receipt.candidate_document_sha256 == file_sha256(reviewed_path)
    assert receipt.candidate_document_sha256 != state.candidate_document.sha256

    result = run_batch_articulation_workflow(request, client=client)

    assert result.status == "completed"
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1
    approved = Stage2CandidateDocument.model_validate_json(
        (request.output_dir / "approved_articulation_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert approved.candidates[0].confidence == "medium"


def test_skill_routed_articulation_rejects_noncanonical_decision_lock(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="workflow_lock must protect the canonical output lock",
    ):
        apply_articulation_decision_patch(
            tmp_path / "run",
            ArticulationDecisionPatch.model_construct(),
            workflow_lock=FileLock(str(tmp_path / "other.lock")),
        )


def test_skill_routed_articulation_recomputes_reviewed_candidates_from_patch(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="high"))
    )
    request = _request(tmp_path, review_policy="all").model_copy(
        update={"metadata": {SKILL_ROUTED_DECISION_METADATA_KEY: True}}
    )
    run_batch_articulation_workflow(request, client=client)
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                rationale="The bound evidence supports the unchanged candidate.",
                confidence=0.9,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Reviewed the exact bound candidate evidence.",
    )
    apply_articulation_decision_patch(request.output_dir, patch)
    reviewed_path = request.output_dir / "agent_reviewed_articulation_candidates.json"
    reviewed_payload = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed_payload["candidates"][0]["confidence"] = "low"
    reviewed_path.write_text(
        json.dumps(reviewed_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger_path = request.output_dir / "articulation_decision_ledger.json"
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_payload["reviewed_candidate_document"]["sha256"] = file_sha256(reviewed_path)
    ledger_path.write_text(
        json.dumps(ledger_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match the validated patch"):
        articulation_decision.load_articulation_decision_ledger(
            request.output_dir,
            state=state,
        )


def test_skill_routed_articulation_rejects_authoring_evidence_edits(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="high"))
    )
    request = _request(tmp_path, review_policy="all").model_copy(
        update={"metadata": {SKILL_ROUTED_DECISION_METADATA_KEY: True}}
    )
    run_batch_articulation_workflow(request, client=client)
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    original = client.candidate_document.candidates[0]
    edited = original.model_copy(
        update={
            "axis_hint": "-x",
            "motion_axis_world": (-1.0, 0.0, 0.0),
        }
    )
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                replacement_candidate=edited,
                rationale="Attempted to rewrite Joint-owned readiness evidence.",
                confidence=0.9,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Attempted an unsafe authoring-evidence edit.",
    )

    with pytest.raises(ValueError, match="immutable authoring or evidence fields"):
        apply_articulation_decision_patch(request.output_dir, patch)


def test_skill_routed_articulation_rejects_extension_field_edits(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    candidate_payload = _candidate(candidate_id, 1, confidence="high")
    candidate_payload["stage1_provenance"] = "trusted-source"
    client = MockArticulationWorkflowClient(_document(candidate_payload))
    request = _request(tmp_path, review_policy="all").model_copy(
        update={"metadata": {SKILL_ROUTED_DECISION_METADATA_KEY: True}}
    )
    run_batch_articulation_workflow(request, client=client)
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    original = client.candidate_document.candidates[0]
    edited = type(original).model_validate(
        {
            **original.model_dump(mode="json"),
            "stage1_provenance": "rewritten-by-child",
            "injected_field": "untrusted",
        }
    )
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                replacement_candidate=edited,
                rationale="Attempted to rewrite pass-through Joint provenance.",
                confidence=0.9,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Attempted an unsafe pass-through field edit.",
    )

    with pytest.raises(
        ValueError,
        match="immutable authoring or evidence fields: injected_field, stage1_provenance",
    ):
        apply_articulation_decision_patch(request.output_dir, patch)


def test_skill_routed_articulation_rejects_extension_field_type_changes(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    candidate_payload = _candidate(candidate_id, 1, confidence="high")
    candidate_payload["stage1_provenance"] = True
    client = MockArticulationWorkflowClient(_document(candidate_payload))
    request = _request(tmp_path, review_policy="all").model_copy(
        update={"metadata": {SKILL_ROUTED_DECISION_METADATA_KEY: True}}
    )
    run_batch_articulation_workflow(request, client=client)
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    original = client.candidate_document.candidates[0]
    edited_payload = original.model_dump(mode="json")
    edited_payload["stage1_provenance"] = 1
    edited = type(original).model_validate(edited_payload)
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                replacement_candidate=edited,
                rationale="Attempted to coerce a pass-through provenance type.",
                confidence=0.9,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Attempted an unsafe pass-through field type change.",
    )

    with pytest.raises(
        ValueError,
        match="immutable authoring or evidence fields: stage1_provenance",
    ):
        apply_articulation_decision_patch(request.output_dir, patch)


@pytest.mark.parametrize(
    "interrupted_after",
    [
        "articulation_decision_patch.json",
        "agent_reviewed_articulation_candidates.json",
        "articulation_decision_ledger.json",
    ],
)
def test_skill_routed_articulation_patch_recovers_after_each_atomic_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_after: str,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="high"))
    )
    base_request = _request(tmp_path, review_policy="all")
    request = base_request.model_copy(
        update={"metadata": {SKILL_ROUTED_DECISION_METADATA_KEY: True}}
    )
    paused = run_batch_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                rationale="The exact evidence supports this joint.",
                confidence=0.9,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Reviewed the exact candidate and usd-cli scene evidence.",
    )
    real_atomic_write_json = articulation_decision.atomic_write_json
    interrupted = False

    def interrupt_after_write(path: Path, payload: object) -> Path:
        nonlocal interrupted
        result = real_atomic_write_json(path, payload)
        if not interrupted and path.name == interrupted_after:
            interrupted = True
            raise RuntimeError(f"interrupted after {interrupted_after}")
        return result

    monkeypatch.setattr(
        articulation_decision,
        "atomic_write_json",
        interrupt_after_write,
    )
    with pytest.raises(RuntimeError, match="interrupted after"):
        apply_articulation_decision_patch(request.output_dir, patch)

    monkeypatch.setattr(
        articulation_decision,
        "atomic_write_json",
        real_atomic_write_json,
    )
    recovered = apply_articulation_decision_patch(request.output_dir, patch)
    repeated = apply_articulation_decision_patch(request.output_dir, patch)

    assert recovered == repeated
    assert (request.output_dir / "articulation_decision_patch.json").is_file()
    assert (
        request.output_dir / "agent_reviewed_articulation_candidates.json"
    ).is_file()
    assert (request.output_dir / "articulation_decision_ledger.json").is_file()


def test_skill_routed_articulation_rejects_stale_patch_before_authoring(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="medium"))
    )
    base_request = _request(tmp_path, review_policy="all")
    request = base_request.model_copy(
        update={
            "metadata": {
                SKILL_ROUTED_DECISION_METADATA_KEY: True,
            }
        }
    )
    run_batch_articulation_workflow(request, client=client)
    state = ArticulationRunState.model_validate_json(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    observation = build_articulation_step_observation(
        state,
        output_dir=request.output_dir,
    )
    patch = ArticulationDecisionPatch(
        request_sha256=observation.request_sha256,
        source_sha256=observation.source_sha256,
        source_dependency_bundle_sha256=(observation.source_dependency_bundle_sha256),
        candidate_document_sha256=observation.candidate_document_sha256,
        scene_evidence_sha256=observation.scene_evidence_sha256,
        checkpoint_revision=observation.checkpoint_revision + 1,
        decisions=(
            ArticulationCandidateDecision(
                candidate_id=candidate_id,
                decision="accept",
                rationale="Reviewed the exact candidate evidence.",
                confidence=0.8,
                evidence_paths=tuple(observation.evidence_sha256_by_path),
            ),
        ),
        evidence_summary="Reviewed candidate evidence.",
    )

    with pytest.raises(ValueError, match="stale or cross-run"):
        apply_articulation_decision_patch(request.output_dir, patch)

    assert _operation_count(client, "author") == 0


def test_review_authors_only_accepted_candidates_in_source_order(
    tmp_path: Path,
) -> None:
    first = "candidate_0001"
    second = "candidate_0002"
    document_payload = _document(
        _candidate(first, 1, confidence="medium"),
        _candidate(
            second,
            2,
            confidence="medium",
            lower_limit=0.0,
            upper_limit=0.5,
        ),
    ).model_dump(mode="json")
    document_payload["summary"].update(
        {
            "unresolved_axis_count": 1,
            "unresolved_parent_count": 1,
            "review_status_counts": {"ready_for_rigger_input": 2},
            "limit_readiness_counts": {
                "not_provided": 1,
                "source_backed": 1,
            },
            "reason_code_counts": {"axis_missing": 1},
        }
    )
    client = MockArticulationWorkflowClient(
        Stage2CandidateDocument.model_validate(document_payload)
    )
    request = _request(tmp_path)
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"

    receipt = build_articulation_review_receipt(
        request.output_dir,
        {second: "reject", first: "accept"},
        reviewer="fixture-owner",
    )
    result = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )

    assert result.status == "completed"
    assert result.accepted_candidate_ids == (first,)
    assert result.rejected_candidate_ids == (second,)
    assert client.calls[1].operation == "author"
    assert client.calls[1].candidate_ids == (first,)
    approved = json.loads(
        (request.output_dir / "approved_articulation_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["candidate_id"] for item in approved["candidates"]] == [first]
    assert approved["summary"]["candidate_count"] == 1
    assert approved["summary"]["review_required_candidate_count"] == 0
    assert approved["summary"]["unresolved_axis_count"] == 0
    assert approved["summary"]["unresolved_parent_count"] == 0
    assert approved["summary"]["review_status_counts"] == {"ready_for_rigger_input": 1}
    assert approved["summary"]["limit_readiness_counts"] == {"not_provided": 1}
    assert approved["summary"]["reason_code_counts"] == {}


def test_batch_without_review_authors_only_native_ready_candidates(
    tmp_path: Path,
) -> None:
    ready_id = "candidate_0001"
    unsupported_id = "candidate_0002"
    client = MockArticulationWorkflowClient(
        _document(
            _candidate(ready_id, 1),
            _candidate(
                unsupported_id,
                2,
                motion_type="spherical",
                review_status="review_required",
                unresolved_reason_codes=["candidate_flag_conflict"],
            ),
        )
    )
    request = _request(
        tmp_path,
        review_policy="none",
        allowed_motion_types=("prismatic",),
    )

    result = run_batch_articulation_workflow(request, client=client)

    assert result.status == "conditional"
    assert result.accepted_candidate_ids == (ready_id,)
    assert result.unresolved_candidate_ids == (unsupported_id,)
    assert result.review_receipt_path is None
    assert result.output_asset_path is not None
    assert _operation_count(client, "author") == 1
    assert client.calls[1].candidate_ids == (ready_id,)


def _membership_document(
    *records: dict[str, Any],
) -> MembershipDispositionDocument:
    ordered = tuple(sorted(records, key=lambda record: str(record["member_prim"])))
    counts: dict[str, int] = {}
    for record in ordered:
        disposition = str(record["disposition"])
        counts[disposition] = counts.get(disposition, 0) + 1
    return MembershipDispositionDocument.model_validate(
        {
            "schema_version": "joint-agent-membership-disposition-v1",
            "summary": {
                "disposition_count": len(ordered),
                "disposition_counts": dict(sorted(counts.items())),
                "review_required_count": sum(
                    record["review_status"] == "review_required" for record in ordered
                ),
                "pending_downstream_count": sum(
                    record["disposition"] in {"explicit_fixed", "unresolved"}
                    for record in ordered
                ),
            },
            "dispositions": list(ordered),
        }
    )


def test_typed_fixed_membership_authors_only_independent_motion(
    tmp_path: Path,
) -> None:
    moving_id = "candidate_0001"
    fixed_id = "candidate_0002"
    carrier = "/World/Cabinet/Carrier"
    accessory = "/World/Cabinet/Accessory"
    candidate_document = _document(
        _candidate(moving_id, 1, moving_part_prim=carrier),
        _candidate(
            fixed_id,
            2,
            moving_part_prim=accessory,
            fixed_parent_prim=carrier,
            review_status="review_required",
            unresolved_reason_codes=["candidate_flag_conflict"],
        ),
    )
    membership_document = _membership_document(
        {
            "disposition_id": "membership_accessory",
            "member_prim": accessory,
            "motion_candidate_prim": accessory,
            "physical_owner_prim": accessory,
            "physical_owner_candidate_prim": accessory,
            "disposition": "explicit_fixed",
            "attachment_parent_prim": carrier,
            "attachment_child_prim": accessory,
            "source": "human_reviewed",
            "confidence": "high",
            "rationale": "Reviewed as a distinct attached link.",
            "source_prediction_ids": [accessory],
            "downstream_boundary": "fixed_authoring_v2",
            "review_status": "resolved",
            "reason_codes": ["fixed_authoring_evidence_required"],
            "unresolved_questions": [
                "Provide accepted attachment-frame evidence through the separate "
                "fixed-authoring capability."
            ],
        },
        {
            "disposition_id": "membership_carrier",
            "member_prim": carrier,
            "motion_candidate_prim": carrier,
            "physical_owner_prim": carrier,
            "physical_owner_candidate_prim": carrier,
            "disposition": "independent_motion",
            "source": "predicted",
            "confidence": "high",
            "source_prediction_ids": [carrier],
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        },
    )
    client = MockArticulationWorkflowClient(
        candidate_document,
        membership_disposition_document=membership_document,
    )

    result = run_interactive_articulation_workflow(
        _request(tmp_path),
        client=client,
    )

    assert result.status == "conditional"
    assert result.accepted_candidate_ids == (moving_id,)
    assert result.rejected_candidate_ids == (fixed_id,)
    assert result.unresolved_candidate_ids == (fixed_id,)
    assert result.unresolved_membership_disposition_ids == ("membership_accessory",)
    assert result.membership_dispositions == membership_document.dispositions
    assert result.output_asset_path is not None
    assert _operation_count(client, "author") == 1
    assert client.calls[1].candidate_ids == (moving_id,)


def test_unresolved_membership_review_propagates_without_authoring(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    accessory = "/World/Cabinet/Accessory"
    membership_id = "membership_accessory"
    membership_document = _membership_document(
        {
            "disposition_id": membership_id,
            "member_prim": accessory,
            "motion_candidate_prim": accessory,
            "disposition": "unresolved",
            "source": "predicted",
            "confidence": "medium",
            "rationale": "Motion and attachment evidence conflict.",
            "source_prediction_ids": [accessory],
            "downstream_boundary": "human_review",
            "review_status": "review_required",
            "reason_codes": ["membership_evidence_conflict"],
            "unresolved_questions": [
                "Resolve the conflicting physical membership evidence."
            ],
        }
    )
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, moving_part_prim=accessory)),
        membership_disposition_document=membership_document,
    )
    request = _request(tmp_path)

    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    assert paused.review_required_candidate_ids == (candidate_id,)
    assert paused.review_required_membership_disposition_ids == (membership_id,)
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "reject"},
        reviewer="fixture-owner",
    )
    result = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )

    assert result.status == "needs_review"
    assert result.review_required_membership_disposition_ids == (membership_id,)
    assert result.unresolved_candidate_ids == (candidate_id,)
    assert result.unresolved_membership_disposition_ids == (membership_id,)
    assert result.membership_dispositions == membership_document.dispositions
    assert result.output_asset_path is None
    assert _operation_count(client, "author") == 0


def test_membership_enabled_inference_rejects_missing_or_partial_coverage() -> None:
    first_prim = "/World/Cabinet/First"
    second_prim = "/World/Cabinet/Second"
    candidate_document = _document(
        _candidate("candidate_0001", 1, moving_part_prim=first_prim),
        _candidate("candidate_0002", 2, moving_part_prim=second_prim),
    )
    partial_document = _membership_document(
        {
            "disposition_id": "membership_first",
            "member_prim": first_prim,
            "motion_candidate_prim": first_prim,
            "physical_owner_prim": first_prim,
            "physical_owner_candidate_prim": first_prim,
            "disposition": "independent_motion",
            "source": "predicted",
            "confidence": "high",
            "source_prediction_ids": [first_prim],
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        }
    )
    complete_document = _accepted_independent_membership(candidate_document)

    with pytest.raises(ValueError, match="do not cover authorable Stage 2"):
        ArticulationInferenceResult(
            candidate_document=candidate_document,
            membership_disposition_document=partial_document,
            backend_configuration_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="requires the typed membership"):
        ArticulationInferenceResult(
            candidate_document=candidate_document,
            backend_configuration_sha256="a" * 64,
            metadata={"backend": "joint-agent-local"},
        )
    with pytest.raises(ValueError, match="fixed-pipeline membership fallback"):
        ArticulationInferenceResult(
            candidate_document=candidate_document,
            membership_disposition_document=complete_document,
            backend_configuration_sha256="a" * 64,
            metadata={
                "membership_disposition_required": True,
                "membership_disposition_schema_version": (
                    "joint-agent-membership-disposition-v1"
                ),
                "membership_disposition_classic_fallback_used": True,
            },
        )
    with pytest.raises(ValueError, match="requires its typed document"):
        ArticulationInferenceResult(
            candidate_document=candidate_document,
            backend_configuration_sha256="a" * 64,
            metadata={
                "membership_disposition_schema_version": (
                    "joint-agent-membership-disposition-v1"
                )
            },
        )


def test_workflow_rejects_authorable_candidates_without_membership(
    tmp_path: Path,
) -> None:
    client = _RawMockArticulationWorkflowClient(
        _document(_candidate("candidate_0001", 1))
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="Typed membership dispositions are required",
    ):
        run_batch_articulation_workflow(_request(tmp_path), client=client)

    assert _operation_count(client, "author") == 0


def test_joint_type_hint_fallback_does_not_bypass_authorability_gate() -> None:
    candidate_payload = _candidate(
        "candidate_0001",
        1,
        motion_type="unknown",
    )
    candidate_payload["joint_type_hint"] = "prismatic"
    candidate = _document(candidate_payload).candidates[0]

    assert candidate.articulation_v1_type == "prismatic"
    assert not candidate.is_articulation_v1_authorable


def test_native_unready_or_unsupported_candidate_cannot_be_accepted(
    tmp_path: Path,
) -> None:
    ready_id = "candidate_0001"
    spherical_id = "candidate_0002"
    client = MockArticulationWorkflowClient(
        _document(
            _candidate(ready_id, 1),
            _candidate(
                spherical_id,
                2,
                motion_type="spherical",
                review_status="review_required",
                unresolved_reason_codes=["joint_type_conflict"],
            ),
        )
    )
    request = _request(tmp_path)
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.review_required_candidate_ids == (spherical_id,)

    with pytest.raises(
        ArticulationWorkflowError,
        match="cannot be accepted",
    ):
        build_articulation_review_receipt(
            request.output_dir,
            {spherical_id: "accept"},
            reviewer="fixture-owner",
        )
    assert _operation_count(client, "author") == 0

    safe_receipt = build_articulation_review_receipt(
        request.output_dir,
        {spherical_id: "reject"},
        reviewer="fixture-owner",
    )
    result = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=safe_receipt,
    )
    assert result.status == "completed"
    assert result.accepted_candidate_ids == (ready_id,)
    assert result.rejected_candidate_ids == (spherical_id,)
    assert client.calls[1].candidate_ids == (ready_id,)


def test_request_motion_scope_requires_rejecting_non_prismatic_candidate(
    tmp_path: Path,
) -> None:
    prismatic_id = "candidate_0001"
    revolute_id = "candidate_0002"
    client = MockArticulationWorkflowClient(
        _document(
            _candidate(prismatic_id, 1),
            _candidate(revolute_id, 2, motion_type="revolute"),
        )
    )
    request = _request(
        tmp_path,
        review_policy="all",
        allowed_motion_types=("prismatic",),
        expected_candidate_count=2,
    )
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.review_required_candidate_ids == (prismatic_id, revolute_id)

    with pytest.raises(ArticulationWorkflowError, match="cannot be accepted"):
        build_articulation_review_receipt(
            request.output_dir,
            {prismatic_id: "accept", revolute_id: "accept"},
            reviewer="fixture-owner",
        )

    receipt = build_articulation_review_receipt(
        request.output_dir,
        {prismatic_id: "accept", revolute_id: "reject"},
        reviewer="fixture-owner",
    )
    result = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )
    assert result.status == "completed"
    assert result.accepted_candidate_ids == (prismatic_id,)
    assert result.rejected_candidate_ids == (revolute_id,)
    assert client.calls[1].candidate_ids == (prismatic_id,)


def test_expected_candidate_count_fails_closed_before_authoring(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(
        _document(
            *[_candidate(f"candidate_{index:04d}", index) for index in range(1, 6)]
        )
    )
    request = _request(
        tmp_path,
        allowed_motion_types=("prismatic",),
        expected_candidate_count=6,
    )

    with pytest.raises(ArticulationWorkflowError, match="expected 6, got 5"):
        run_interactive_articulation_workflow(request, client=client)
    assert _operation_count(client, "author") == 0


def test_completed_checkpoint_replay_reapplies_candidate_count_limits(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(
        _document(
            _candidate("candidate_0001", 1),
            _candidate("candidate_0002", 2),
        )
    )
    request = _request(tmp_path, allowed_motion_types=("prismatic",))
    result = run_interactive_articulation_workflow(request, client=client)
    assert result.status == "completed"

    request_path = Path(result.request_path or "")
    bound_request = ArticulationWorkflowRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    checkpoint = ArticulationRunState.model_validate_json(
        Path(result.checkpoint_path).read_text(encoding="utf-8")
    )
    for request_updates, error_match in (
        (
            {"max_candidate_count": 1},
            "candidate count exceeds the request safety bound",
        ),
        (
            {"expected_candidate_count": 1},
            "candidate count does not match the request",
        ),
    ):
        forged_request = bound_request.model_copy(update=request_updates)
        request_path.write_text(
            forged_request.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        forged_checkpoint = checkpoint.model_copy(
            update={
                "request": checkpoint.request.model_copy(
                    update={"sha256": file_sha256(request_path)}
                )
            }
        )

        with pytest.raises(ArticulationWorkflowError, match=error_match):
            validate_completed_articulation_checkpoint(
                forged_checkpoint,
                request=forged_request,
            )


def test_completed_checkpoint_replay_rejects_spliced_authoring_predictions(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path, allowed_motion_types=("prismatic",))
    result = run_interactive_articulation_workflow(request, client=client)
    assert result.status == "completed"

    request_path = Path(result.request_path or "")
    bound_request = ArticulationWorkflowRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    checkpoint = ArticulationRunState.model_validate_json(
        Path(result.checkpoint_path).read_text(encoding="utf-8")
    )
    assert checkpoint.authoring_request is not None
    authoring_request_path = Path(checkpoint.authoring_request.path)
    authoring_request = ArticulationAuthoringRequest.model_validate_json(
        authoring_request_path.read_text(encoding="utf-8")
    )
    spliced_predictions = request.output_dir / "spliced-predictions.json"
    spliced_predictions.write_text('{"candidate":"different"}\n', encoding="utf-8")
    spliced_authoring_request = authoring_request.model_copy(
        update={
            "predictions_path": str(spliced_predictions.resolve()),
            "predictions_sha256": file_sha256(spliced_predictions),
        }
    )
    authoring_request_path.write_text(
        spliced_authoring_request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    forged_checkpoint = checkpoint.model_copy(
        update={
            "authoring_request": checkpoint.authoring_request.model_copy(
                update={"sha256": file_sha256(authoring_request_path)}
            )
        }
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="authoring request prediction binding differs from inference",
    ):
        validate_completed_articulation_checkpoint(
            forged_checkpoint,
            request=bound_request,
        )


def test_rejected_parent_candidate_blocks_child_subgraph_before_authoring(
    tmp_path: Path,
) -> None:
    parent_id = "candidate_parent"
    child_id = "candidate_child"
    parent_body = "/World/Cabinet/Drawer_01"
    client = MockArticulationWorkflowClient(
        _document(
            _candidate(
                parent_id,
                1,
                confidence="medium",
                moving_part_prim=parent_body,
            ),
            _candidate(
                child_id,
                2,
                confidence="medium",
                moving_part_prim=f"{parent_body}/Handle",
                fixed_parent_prim=parent_body,
            ),
        )
    )
    request = _request(tmp_path, review_policy="all")
    run_interactive_articulation_workflow(request, client=client)
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {parent_id: "reject", child_id: "accept"},
        reviewer="fixture-owner",
    )

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )

    assert result.status == "conditional"
    assert result.unresolved_candidate_ids == (parent_id, child_id)
    assert result.output_asset_path is None
    assert result.authoring_result_path is None
    assert "do not form a closed articulation graph" in result.message
    assert "useful articulation output exists" not in result.message
    assert _operation_count(client, "author") == 0


def test_graph_closure_rejects_accepted_candidate_without_moving_prim() -> None:
    from content_agent_workflows.articulation.workflow import (
        _graph_closure_blockers,
    )

    payload = _candidate("candidate_0001", 1)
    payload["moving_part_prims"] = []
    document = _document(payload)

    with pytest.raises(
        ArticulationWorkflowError,
        match="must bind exactly one moving-part prim",
    ):
        _graph_closure_blockers(document, document.candidate_ids)


def test_stage2_axis_contract_rejects_upstream_invalid_candidate() -> None:
    payload = _candidate("candidate_0001", 1)
    payload["axis_hint"] = "unknown"
    with pytest.raises(ValueError, match="axis-aligned axis_hint"):
        _document(payload)


def test_stage2_contract_rejects_coerced_axis_and_unknown_provenance() -> None:
    boolean_axis = _candidate("candidate_0001", 1)
    boolean_axis["motion_axis_world"] = [True, False, False]
    with pytest.raises(ValueError):
        _document(boolean_axis)

    invented_source = _candidate("candidate_0001", 1)
    invented_source["field_sources"]["motion_type"] = "invented"
    with pytest.raises(ValueError):
        _document(invented_source)


def test_stage2_mismatched_evidence_is_not_authorable(
    tmp_path: Path,
) -> None:
    bad_axis = _candidate("candidate_0001", 1)
    bad_axis["axis_evidence"][0]["value"] = "y"
    bad_connectivity = _candidate("candidate_0001", 1)
    bad_connectivity["connectivity_evidence"][0]["value"] = "/World/Wrong"
    bad_limit_source = _candidate(
        "candidate_0001",
        1,
        lower_limit=0.0,
        upper_limit=0.5,
    )
    bad_limit_source["limit_evidence"][0]["source"] = "source_metadata"

    for payload in (bad_axis, bad_connectivity, bad_limit_source):
        candidate = _document(payload).candidates[0]
        assert candidate.is_native_ready is True
        assert candidate.is_articulation_v1_authorable is False

    client = MockArticulationWorkflowClient(_document(bad_axis))
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    with pytest.raises(ArticulationWorkflowError, match="cannot be accepted"):
        build_articulation_review_receipt(
            request.output_dir,
            {"candidate_0001": "accept"},
            reviewer="fixture-owner",
        )
    assert _operation_count(client, "author") == 0


@pytest.mark.parametrize(
    ("endpoint", "invalid_path"),
    [
        ("parent", "World/Cabinet"),
        ("parent", "/"),
        ("parent", "/World/Cabinet.attr"),
        ("parent", "/World{variant=selected}/Cabinet"),
        ("parent", "/World//Cabinet"),
        ("child", "World/Drawer"),
        ("child", "/"),
        ("child", "/World/Drawer.attr"),
        ("child", "/World{variant=selected}/Drawer"),
        ("child", "/World//Drawer"),
    ],
)
def test_stage2_invalid_endpoint_path_never_reaches_authoring(
    tmp_path: Path,
    endpoint: str,
    invalid_path: str,
) -> None:
    candidate_payload = _candidate(
        "candidate_0001",
        1,
        fixed_parent_prim=(invalid_path if endpoint == "parent" else "/World/Cabinet"),
        moving_part_prim=(
            invalid_path if endpoint == "child" else "/World/Cabinet/Drawer_01"
        ),
    )
    candidate_document = _document(candidate_payload)
    candidate = candidate_document.candidates[0]
    assert candidate.is_native_ready is True
    assert candidate.is_articulation_v1_authorable is False

    class AuthorTrapClient(_RawMockArticulationWorkflowClient):
        author_attempts = 0

        def author(
            self,
            request: ArticulationAuthoringRequest,
            *,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationAuthoringResult:
            del request, cancel_checker
            self.author_attempts += 1
            raise AssertionError("invalid endpoint reached authoring")

    client = AuthorTrapClient(candidate_document)
    request = _request(tmp_path)
    paused = run_interactive_articulation_workflow(request, client=client)

    assert paused.status == "needs_review"
    with pytest.raises(ArticulationWorkflowError, match="cannot be accepted"):
        build_articulation_review_receipt(
            request.output_dir,
            {"candidate_0001": "accept"},
            reviewer="fixture-owner",
        )
    assert client.author_attempts == 0


def test_resume_after_inference_artifact_does_not_repeat_inference(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def interrupt(boundary: str) -> None:
        if boundary == "inference_artifacts_written":
            raise ArticulationWorkflowInterrupted("simulated process exit")

    with pytest.raises(ArticulationWorkflowInterrupted):
        run_interactive_articulation_workflow(
            request,
            client=client,
            phase_boundary_hook=interrupt,
        )
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0

    result = run_interactive_articulation_workflow(request, client=client)
    assert result.status == "completed"
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 1


def test_resume_after_authoring_artifact_does_not_duplicate_joints(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def interrupt(boundary: str) -> None:
        if boundary == "authoring_artifact_written":
            raise ArticulationWorkflowInterrupted("simulated process exit")

    with pytest.raises(ArticulationWorkflowInterrupted):
        run_interactive_articulation_workflow(
            request,
            client=client,
            phase_boundary_hook=interrupt,
        )
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 0

    result = run_interactive_articulation_workflow(request, client=client)
    assert result.status == "completed"
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1


def test_resume_wraps_missing_bound_inference_artifact(
    tmp_path: Path,
) -> None:
    class BoundInferenceClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            result = super().infer(
                request,
                resume=resume,
                cancel_checker=cancel_checker,
            )
            predictions_path = tmp_path / "predictions.json"
            predictions_path.write_text('{"prediction": "drawer"}\n', encoding="utf-8")
            return result.model_copy(
                update={
                    "predictions_path": str(predictions_path),
                    "predictions_sha256": hashlib.sha256(
                        predictions_path.read_bytes()
                    ).hexdigest(),
                }
            )

    client = BoundInferenceClient(
        _document(_candidate("candidate_0001", 1, confidence="medium"))
    )
    request = _request(tmp_path)
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    (tmp_path / "predictions.json").unlink()

    with pytest.raises(
        ArticulationWorkflowError,
        match="Joint Agent predictions is missing",
    ):
        run_interactive_articulation_workflow(request, client=client)


@pytest.mark.parametrize(
    ("artifact_name", "path_field", "digest_field", "label"),
    [
        (
            "predictions.json",
            "predictions_path",
            "predictions_sha256",
            "Joint Agent predictions",
        ),
        (
            "candidate-report.html",
            "report_path",
            "report_sha256",
            "Joint Agent candidate report",
        ),
    ],
)
def test_fresh_inference_rejects_side_artifact_drift_before_checkpoint(
    tmp_path: Path,
    artifact_name: str,
    path_field: str,
    digest_field: str,
    label: str,
) -> None:
    class DriftOnceInferenceClient(MockArticulationWorkflowClient):
        inference_attempts = 0

        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            result = super().infer(
                request,
                resume=resume,
                cancel_checker=cancel_checker,
            )
            self.inference_attempts += 1
            artifact_path = tmp_path / artifact_name
            artifact_path.write_text("bound evidence\n", encoding="utf-8")
            digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if self.inference_attempts == 1:
                artifact_path.write_text("drifted evidence\n", encoding="utf-8")
            return result.model_copy(
                update={
                    path_field: str(artifact_path),
                    digest_field: digest,
                }
            )

    client = DriftOnceInferenceClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    with pytest.raises(ArticulationWorkflowError, match=rf"{label} digest mismatch"):
        run_interactive_articulation_workflow(request, client=client)

    assert not (request.output_dir / "inference_result.json").exists()
    assert not (request.output_dir / "articulation_candidates.json").exists()
    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "inferring"
    assert _operation_count(client, "author") == 0

    completed = run_interactive_articulation_workflow(request, client=client)
    assert completed.status == "completed"
    assert _operation_count(client, "infer") == 2
    assert _operation_count(client, "author") == 1


def test_authoring_rejects_prediction_drift_after_inference_checkpoint(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    bound_payload = '{"prediction": "drawer"}\n'

    class BoundPredictionClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            result = super().infer(
                request,
                resume=resume,
                cancel_checker=cancel_checker,
            )
            predictions_path.write_text(bound_payload, encoding="utf-8")
            return result.model_copy(
                update={
                    "predictions_path": str(predictions_path),
                    "predictions_sha256": hashlib.sha256(
                        predictions_path.read_bytes()
                    ).hexdigest(),
                }
            )

    client = BoundPredictionClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def drift_after_checkpoint(boundary: str) -> None:
        if boundary == "inference_artifacts_written":
            predictions_path.write_text(
                '{"prediction": "changed"}\n',
                encoding="utf-8",
            )

    with pytest.raises(
        ArticulationWorkflowError,
        match="authoring evidence binding failed.*predictions",
    ):
        run_interactive_articulation_workflow(
            request,
            client=client,
            phase_boundary_hook=drift_after_checkpoint,
        )

    authoring_request = json.loads(
        (request.output_dir / "authoring_request.json").read_text(encoding="utf-8")
    )
    assert authoring_request["predictions_path"] == str(predictions_path)
    assert (
        authoring_request["predictions_sha256"]
        == hashlib.sha256(bound_payload.encode("utf-8")).hexdigest()
    )
    assert not (request.output_dir / "joint_rigger").exists()
    assert _operation_count(client, "author") == 0

    predictions_path.write_text(bound_payload, encoding="utf-8")
    completed = run_interactive_articulation_workflow(request, client=client)
    assert completed.status == "completed"
    assert _operation_count(client, "author") == 1


@pytest.mark.parametrize(
    ("relative_path", "label"),
    [
        ("joint_rigger/diagnostics.json", "Joint Rigger diagnostics"),
        ("joint_rigger/result.json", "Joint Rigger result"),
        ("joint_rigger/rigged.usdz", "Published articulation output"),
    ],
)
def test_terminal_resume_wraps_missing_bound_artifact(
    tmp_path: Path,
    relative_path: str,
    label: str,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    result = run_interactive_articulation_workflow(request, client=client)
    assert result.status == "completed"
    (request.output_dir / relative_path).unlink()

    with pytest.raises(ArticulationWorkflowError, match=rf"{label} is missing"):
        run_interactive_articulation_workflow(request, client=client)


def test_resume_after_backend_publish_recovers_without_duplicate_side_effect(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def interrupt(boundary: str) -> None:
        if boundary == "authoring_backend_returned":
            raise ArticulationWorkflowInterrupted("simulated post-publish exit")

    with pytest.raises(ArticulationWorkflowInterrupted):
        run_interactive_articulation_workflow(
            request,
            client=client,
            phase_boundary_hook=interrupt,
        )
    assert _operation_count(client, "author") == 1
    assert not (request.output_dir / "authoring_result.json").exists()
    assert (request.output_dir / "joint_rigger" / "authoring_attempt.json").is_file()

    result = run_interactive_articulation_workflow(request, client=client)

    assert result.status == "completed"
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "author_recover") == 1
    assert _operation_count(client, "validate") == 1


def test_exact_readback_mismatch_is_conditional(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1)),
        validated_candidate_ids=(),
        exact_graph_match=False,
    )
    request = _request(tmp_path)

    result = run_interactive_articulation_workflow(request, client=client)

    assert result.status == "conditional"
    assert result.success is False
    assert result.unresolved_candidate_ids == (candidate_id,)
    validation = json.loads(
        (request.output_dir / "validation_evidence.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "fail"
    assert validation["exact_graph_match"] is False


def test_exact_readback_targets_only_candidate_ids_missing_from_validation(
    tmp_path: Path,
) -> None:
    first_id = "candidate_0001"
    second_id = "candidate_0002"
    client = MockArticulationWorkflowClient(
        _document(
            _candidate(first_id, 1),
            _candidate(second_id, 2),
        ),
        validated_candidate_ids=(first_id,),
        exact_graph_match=False,
    )
    request = _request(tmp_path)

    result = run_interactive_articulation_workflow(request, client=client)

    assert result.status == "conditional"
    assert result.unresolved_candidate_ids == (second_id,)
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1


def test_output_digest_drift_is_persisted_as_validation_failure(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def interrupt(boundary: str) -> None:
        if boundary == "authoring_artifact_written":
            raise ArticulationWorkflowInterrupted("pause before validation")

    with pytest.raises(ArticulationWorkflowInterrupted):
        run_interactive_articulation_workflow(
            request,
            client=client,
            phase_boundary_hook=interrupt,
        )
    output_path = request.output_dir / "joint_rigger" / "rigged.usdz"
    with zipfile.ZipFile(output_path, mode="w") as archive:
        archive.writestr("drifted.usda", '#usda 1.0\n\ndef Xform "Drifted" {}\n')

    result = run_interactive_articulation_workflow(request, client=client)

    assert result.status == "conditional"
    validation = json.loads(
        (request.output_dir / "validation_evidence.json").read_text(encoding="utf-8")
    )
    assert (
        validation["expected_output_asset_sha256"]
        != validation["observed_output_asset_sha256"]
    )
    assert any("digest" in failure for failure in validation["failures"])


def test_late_cancellation_wins_over_authored_output(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def cancel_checker() -> bool:
        return (request.output_dir / "joint_rigger" / "rigged.usdz").is_file()

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        cancel_checker=cancel_checker,
    )

    assert result.status == "cancelled"
    assert result.success is False
    assert result.output_asset_path is not None
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 0

    repeated = run_interactive_articulation_workflow(request, client=client)
    assert repeated.status == "cancelled"
    assert _operation_count(client, "author") == 1


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    [
        ("missing", "inference result is missing"),
        ("corrupt", "Invalid articulation inference result"),
        ("digest_drift", "inference result digest mismatch"),
        ("predictions_missing", "Joint Agent predictions is missing"),
        ("predictions_modified", "Joint Agent predictions digest mismatch"),
        ("report_missing", "Joint Agent candidate report is missing"),
        ("report_modified", "Joint Agent candidate report digest mismatch"),
    ],
)
def test_cancelled_replay_preserves_integrity_error_and_sanitizes_summary(
    tmp_path: Path,
    damage: str,
    expected_error: str,
) -> None:
    class BoundSideArtifactClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            result = super().infer(
                request,
                resume=resume,
                cancel_checker=cancel_checker,
            )
            predictions_path = request.output_dir / "joint_predictions.json"
            report_path = request.output_dir / "joint_report.json"
            predictions_path.write_text('{"joint": "drawer"}\n', encoding="utf-8")
            report_path.write_text('{"status": "ready"}\n', encoding="utf-8")
            return result.model_copy(
                update={
                    "predictions_path": str(predictions_path),
                    "predictions_sha256": hashlib.sha256(
                        predictions_path.read_bytes()
                    ).hexdigest(),
                    "report_path": str(report_path),
                    "report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                }
            )

    client = BoundSideArtifactClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    inference_path = request.output_dir / "inference_result.json"

    cancelled = run_interactive_articulation_workflow(
        request,
        client=client,
        cancel_checker=inference_path.is_file,
    )
    assert cancelled.status == "cancelled"
    assert cancelled.predictions_path is not None
    assert cancelled.report_path is not None

    if damage == "missing":
        inference_path.unlink()
        checkpoint_path = request.output_dir / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["review_required_membership_disposition_ids"] = [
            "membership_candidate_0001"
        ]
        checkpoint["unresolved_membership_disposition_ids"] = [
            "membership_candidate_0001"
        ]
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif damage == "corrupt":
        inference_path.write_bytes(b"{")
        checkpoint_path = request.output_dir / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["inference_result"]["sha256"] = hashlib.sha256(b"{").hexdigest()
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif damage == "digest_drift":
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        inference["predictions_path"] = "/untrusted/predictions.json"
        inference["predictions_sha256"] = "0" * 64
        inference["report_path"] = "/untrusted/report.json"
        inference["report_sha256"] = "0" * 64
        inference_path.write_text(
            json.dumps(inference, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        artifact_name, artifact_damage = damage.rsplit("_", maxsplit=1)
        artifact_path = request.output_dir / f"joint_{artifact_name}.json"
        if artifact_damage == "missing":
            artifact_path.unlink()
        else:
            artifact_path.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(ArticulationWorkflowError, match=expected_error):
        run_interactive_articulation_workflow(request, client=client)

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "cancelled"
    expected_inference_result_path = (
        None if damage in {"missing", "digest_drift"} else str(inference_path.resolve())
    )
    assert summary["inference_result_path"] == expected_inference_result_path
    expected_predictions_path = (
        None
        if damage in {"missing", "corrupt", "digest_drift"}
        or damage.startswith("predictions_")
        else str(request.output_dir / "joint_predictions.json")
    )
    expected_report_path = (
        None
        if damage in {"missing", "corrupt", "digest_drift"}
        or damage.startswith("report_")
        else str(request.output_dir / "joint_report.json")
    )
    assert summary["predictions_path"] == expected_predictions_path
    assert summary["report_path"] == expected_report_path
    if damage == "missing":
        assert summary["review_required_membership_disposition_ids"] == []
        assert summary["unresolved_membership_disposition_ids"] == []


@pytest.mark.parametrize("terminal_status", ("completed", "conditional"))
def test_terminal_replay_integrity_failure_invalidates_artifact_summary(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    if terminal_status == "completed":
        client = MockArticulationWorkflowClient(
            _document(_candidate("candidate_0001", 1))
        )
        request = _request(tmp_path)
        initial = run_interactive_articulation_workflow(request, client=client)
    else:
        client = MockArticulationWorkflowClient(
            _document(
                _candidate("candidate_0001", 1),
                _candidate(
                    "candidate_0002",
                    2,
                    motion_type="spherical",
                    review_status="review_required",
                    unresolved_reason_codes=["candidate_flag_conflict"],
                ),
            )
        )
        request = _request(
            tmp_path,
            review_policy="none",
            allowed_motion_types=("prismatic",),
        )
        initial = run_batch_articulation_workflow(request, client=client)

    def replay() -> Any:
        if terminal_status == "completed":
            return run_interactive_articulation_workflow(request, client=client)
        return run_batch_articulation_workflow(request, client=client)

    assert initial.status == terminal_status
    assert initial.output_asset_path is not None
    validation_path = request.output_dir / "validation_evidence.json"
    original_validation = validation_path.read_bytes()
    validation_path.write_text('{"tampered": true}\n', encoding="utf-8")

    with pytest.raises(
        ArticulationWorkflowError,
        match="validation result digest mismatch",
    ):
        replay()

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert "integrity" in invalidated["message"].lower()
    for field_name in (
        "inference_result_path",
        "candidate_document_path",
        "scene_evidence_path",
        "review_receipt_path",
        "approved_candidate_document_path",
        "authoring_request_path",
        "authoring_result_path",
        "output_asset_path",
        "diagnostics_path",
        "joint_rigger_result_path",
        "validation_result_path",
    ):
        assert invalidated[field_name] is None

    validation_path.write_bytes(original_validation)
    restored = replay()
    assert restored.status == terminal_status
    assert restored.output_asset_path == initial.output_asset_path


def test_terminal_integrity_fallback_never_leaves_stale_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import finalizer

    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"
    validation_path = request.output_dir / "validation_evidence.json"
    validation_path.write_text('{"tampered": true}\n', encoding="utf-8")
    real_atomic_write_json = finalizer.atomic_write_json

    def fail_failed_summary(path: Path, payload: Any) -> Path:
        if (
            Path(path).name == "final_summary.json"
            and getattr(payload, "status", None) == "failed"
        ):
            raise OSError("simulated failed-summary replacement failure")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(finalizer, "atomic_write_json", fail_failed_summary)
    with pytest.raises(
        OSError,
        match="failed-summary replacement failure",
    ):
        run_interactive_articulation_workflow(request, client=client)

    assert not (request.output_dir / "final_summary.json").exists()
    progress = json.loads(
        (request.output_dir / "workflow_progress.json").read_text(encoding="utf-8")
    )
    assert progress["phase"] == "failed"


def test_completed_replay_requires_validation_binding(tmp_path: Path) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["validation_result"] = None
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="missing required evidence: validation result",
    ):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["validation_result_path"] is None
    assert invalidated["output_asset_path"] is None


def test_completed_replay_rejects_bound_failed_validation(tmp_path: Path) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"
    validation_path = request.output_dir / "validation_evidence.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation.update(
        {
            "status": "fail",
            "exact_graph_match": False,
            "validated_candidate_ids": [],
            "failures": ["simulated replay validation failure"],
        }
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["validation_result"]["sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="requires passing readback validation",
    ):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["validation_result_path"] is None
    assert invalidated["output_asset_path"] is None


def test_completed_replay_rederives_required_review_scope(tmp_path: Path) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {"candidate_0001": "accept"},
        reviewer="workflow-test",
    )
    completed = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )
    assert completed.status == "completed"

    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["review_required_candidate_ids"] = []
    checkpoint["review_receipt"] = None
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="review-required scope differs",
    ):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["review_receipt_path"] is None
    assert invalidated["output_asset_path"] is None


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    (
        ("invalid_bound_result", "Invalid bound authoring result"),
        ("supplied_result_mismatch", "authoring differs from the bound"),
    ),
)
def test_completed_finalizer_loads_the_bound_authoring_result(
    tmp_path: Path,
    damage: str,
    expected_error: str,
) -> None:
    from content_agent_workflows.articulation.finalizer import (
        write_articulation_workflow_summary,
    )
    from content_agent_workflows.articulation.models import (
        ArticulationRunState,
        ArtifactBinding,
    )

    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    completed = run_interactive_articulation_workflow(request, client=client)
    assert completed.status == "completed"
    state = ArticulationRunState.model_validate(
        json.loads((request.output_dir / "checkpoint.json").read_text(encoding="utf-8"))
    )
    assert state.authoring_result is not None
    authoring_path = Path(state.authoring_result.path)
    authoring = ArticulationAuthoringResult.model_validate(
        json.loads(authoring_path.read_text(encoding="utf-8"))
    )
    supplied_authoring = authoring
    if damage == "invalid_bound_result":
        authoring_path.write_text('{"bogus": true}\n', encoding="utf-8")
        state = state.model_copy(
            update={
                "authoring_result": ArtifactBinding(
                    path=str(authoring_path),
                    sha256=hashlib.sha256(authoring_path.read_bytes()).hexdigest(),
                )
            }
        )
    else:
        supplied_authoring = authoring.model_copy(
            update={"backend_call_id": "forged-backend-call"}
        )

    with pytest.raises(ValueError, match=expected_error):
        write_articulation_workflow_summary(
            state,
            output_dir=request.output_dir,
            authoring=supplied_authoring,
        )


@pytest.mark.parametrize(
    ("damage", "expected_error"),
    (
        ("candidate", "candidate document digest mismatch"),
        ("request", "Existing articulation request conflicts"),
        ("source", "Source asset digest differs"),
    ),
)
def test_terminal_preflight_failure_invalidates_success_summary(
    tmp_path: Path,
    damage: str,
    expected_error: str,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"

    damaged_path = {
        "candidate": request.output_dir / "articulation_candidates.json",
        "request": request.output_dir / "request.json",
        "source": Path(request.source_asset),
    }[damage]
    original_bytes = damaged_path.read_bytes()
    if damage == "request":
        payload = json.loads(original_bytes)
        payload["intent"] = "tampered request"
        damaged_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        damaged_path.write_bytes(original_bytes + b"\n")

    with pytest.raises(ArticulationWorkflowError, match=expected_error):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert "integrity" in invalidated["message"].lower()
    for field_name in (
        "inference_result_path",
        "candidate_document_path",
        "scene_evidence_path",
        "review_receipt_path",
        "approved_candidate_document_path",
        "authoring_request_path",
        "authoring_result_path",
        "output_asset_path",
        "diagnostics_path",
        "joint_rigger_result_path",
        "validation_result_path",
    ):
        assert invalidated[field_name] is None

    damaged_path.write_bytes(original_bytes)
    restored = run_interactive_articulation_workflow(request, client=client)
    assert restored.status == "completed"
    assert restored.output_asset_path == initial.output_asset_path


def test_unreadable_terminal_checkpoint_invalidates_success_summary(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"

    checkpoint_path = request.output_dir / "checkpoint.json"
    original_checkpoint = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(b"{")

    with pytest.raises(
        ArticulationWorkflowError,
        match="Invalid articulation checkpoint",
    ):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert invalidated["request_path"] is None
    assert invalidated["output_asset_path"] is None
    assert invalidated["validation_result_path"] is None

    checkpoint_path.write_bytes(original_checkpoint)
    restored = run_interactive_articulation_workflow(request, client=client)
    assert restored.status == "completed"
    assert restored.output_asset_path == initial.output_asset_path


def test_terminal_configuration_error_invalidates_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"
    original_configuration_sha256 = client.configuration_sha256

    def fail_configuration(_request: ArticulationWorkflowRequest) -> str:
        raise ValueError("simulated configuration callback failure")

    monkeypatch.setattr(client, "configuration_sha256", fail_configuration)
    with pytest.raises(ValueError, match="configuration callback failure"):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert invalidated["output_asset_path"] is None
    assert invalidated["validation_result_path"] is None

    monkeypatch.setattr(
        client,
        "configuration_sha256",
        original_configuration_sha256,
    )
    restored = run_interactive_articulation_workflow(request, client=client)
    assert restored.status == "completed"
    assert restored.output_asset_path == initial.output_asset_path


def test_terminal_summary_rehash_error_invalidates_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    initial = run_interactive_articulation_workflow(request, client=client)
    assert initial.status == "completed"
    write_summary = workflow.write_articulation_workflow_summary
    call_count = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated summary rehash failure")
        return write_summary(*args, **kwargs)

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        fail_once,
    )
    with pytest.raises(OSError, match="summary rehash failure"):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert invalidated["output_asset_path"] is None
    assert invalidated["validation_result_path"] is None

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        write_summary,
    )
    restored = run_interactive_articulation_workflow(request, client=client)
    assert restored.status == "completed"
    assert restored.output_asset_path == initial.output_asset_path


def test_initial_terminal_summary_error_invalidates_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    write_summary = workflow.write_articulation_workflow_summary
    call_count = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated initial summary publication failure")
        return write_summary(*args, **kwargs)

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        fail_once,
    )
    with pytest.raises(OSError, match="initial summary publication failure"):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["success"] is False
    assert invalidated["output_asset_path"] is None
    assert invalidated["validation_result_path"] is None


def test_cancellation_summary_error_publishes_sanitized_terminal_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)
    write_summary = workflow.write_articulation_workflow_summary
    call_count = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated cancellation summary failure")
        return write_summary(*args, **kwargs)

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        fail_once,
    )
    with pytest.raises(OSError, match="cancellation summary failure"):
        run_interactive_articulation_workflow(
            request,
            client=client,
            cancel_checker=lambda: True,
        )

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "cancelled"
    assert summary["success"] is False
    assert summary["output_asset_path"] is None


def test_conditional_summary_error_invalidates_artifact_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    client = MockArticulationWorkflowClient(
        _document(
            _candidate(
                "candidate_0001",
                1,
                motion_type="spherical",
                review_status="review_required",
                unresolved_reason_codes=["candidate_flag_conflict"],
            )
        )
    )
    request = _request(
        tmp_path,
        review_policy="none",
        allowed_motion_types=("prismatic",),
    )
    write_summary = workflow.write_articulation_workflow_summary
    call_count = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated conditional summary failure")
        return write_summary(*args, **kwargs)

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        fail_once,
    )
    with pytest.raises(OSError, match="conditional summary failure"):
        run_batch_articulation_workflow(request, client=client)

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["success"] is False
    assert summary["candidate_document_path"] is None


def test_summary_recording_failure_does_not_mask_backend_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    class FailingInferenceClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            del request, resume, cancel_checker
            raise RuntimeError("original inference failure")

    client = FailingInferenceClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def fail_summary(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("simulated summary recording failure")

    monkeypatch.setattr(
        workflow,
        "write_articulation_workflow_summary",
        fail_summary,
    )
    with pytest.raises(
        ArticulationWorkflowError,
        match="Joint Agent inference failed: original inference failure",
    ) as exc_info:
        run_interactive_articulation_workflow(request, client=client)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "original inference failure"
    assert any(
        "Recording the durable backend failure also failed" in note
        for note in exc_info.value.__cause__.__notes__
    )


def test_required_reconciliation_is_typed_agentic_terminal_result(
    tmp_path: Path,
) -> None:
    recovery_action = {
        "kind": "resume",
        "operation": "joint_agent.api.pipeline",
        "resume": True,
        "clean": False,
        "failed_step": "infer_articulation_candidates",
        "checkpoint_path": str(tmp_path / "joint" / ".pipeline_state.json"),
    }
    terminal_status = {
        "requested": True,
        "attempted": True,
        "accepted": False,
        "outcome": "failed",
        "failure_stage": "validation",
        "error_type": "ValueError",
        "success": False,
        "terminal": True,
        "required": True,
        "unresolved_decisions": {
            "candidate_artifact_path": str(tmp_path / "joint" / "candidates.json"),
            "review_required_candidate_count": 2,
        },
        "recovery_action": recovery_action,
    }

    class TerminalInferenceClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            del request, resume, cancel_checker
            raise JointAgentInferenceTerminalError(terminal_status)

    client = TerminalInferenceClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    result = run_interactive_articulation_workflow(request, client=client)

    assert result.success is False
    assert result.status == "failed"
    assert result.mode == "interactive"
    assert result.terminal_status == terminal_status
    assert result.recovery_action == recovery_action
    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["backend_terminal_status"] == terminal_status
    final_summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert final_summary["success"] is False
    assert final_summary["terminal_status"] == terminal_status
    assert final_summary["recovery_action"] == recovery_action


@pytest.mark.parametrize("damage", ["missing", "corrupt", "digest_drift"])
def test_failed_summary_does_not_mask_backend_error_or_advertise_untrusted_paths(
    tmp_path: Path,
    damage: str,
) -> None:
    class FailingSideArtifactClient(MockArticulationWorkflowClient):
        def infer(
            self,
            request: ArticulationWorkflowRequest,
            *,
            resume: bool,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationInferenceResult:
            result = super().infer(
                request,
                resume=resume,
                cancel_checker=cancel_checker,
            )
            predictions_path = request.output_dir / "joint_predictions.json"
            report_path = request.output_dir / "joint_report.json"
            predictions_path.write_text('{"joint": "drawer"}\n', encoding="utf-8")
            report_path.write_text('{"status": "ready"}\n', encoding="utf-8")
            return result.model_copy(
                update={
                    "predictions_path": str(predictions_path),
                    "predictions_sha256": hashlib.sha256(
                        predictions_path.read_bytes()
                    ).hexdigest(),
                    "report_path": str(report_path),
                    "report_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                }
            )

        def author(
            self,
            request: ArticulationAuthoringRequest,
            *,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationAuthoringResult:
            del cancel_checker
            inference_path = request.output_dir / "inference_result.json"
            if damage == "missing":
                inference_path.unlink()
            elif damage == "corrupt":
                inference_path.write_bytes(b"{")
            else:
                inference = json.loads(inference_path.read_text(encoding="utf-8"))
                inference["predictions_path"] = "/untrusted/predictions.json"
                inference["predictions_sha256"] = "0" * 64
                inference["report_path"] = "/untrusted/report.json"
                inference["report_sha256"] = "0" * 64
                inference_path.write_text(
                    json.dumps(inference, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            raise RuntimeError("original backend failure")

    client = FailingSideArtifactClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    with pytest.raises(
        ArticulationWorkflowError,
        match="Joint Agent authoring failed: original backend failure",
    ):
        run_interactive_articulation_workflow(request, client=client)

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["predictions_path"] is None
    assert summary["report_path"] is None


@pytest.mark.parametrize("drift", ["missing", "modified"])
def test_terminal_cancelled_resume_rejects_authored_output_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    client = MockArticulationWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    def cancel_checker() -> bool:
        return (request.output_dir / "joint_rigger" / "rigged.usdz").is_file()

    cancelled = run_interactive_articulation_workflow(
        request,
        client=client,
        cancel_checker=cancel_checker,
    )
    assert cancelled.status == "cancelled"
    output_path = request.output_dir / "joint_rigger" / "rigged.usdz"
    original_output = output_path.read_bytes()
    if drift == "missing":
        output_path.unlink()
        expected_message = "Published articulation output is missing"
    else:
        output_path.write_bytes(output_path.read_bytes() + b"drift")
        expected_message = "Published articulation output digest mismatch"

    with pytest.raises(ArticulationWorkflowError, match=expected_message):
        run_interactive_articulation_workflow(request, client=client)

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "cancelled"
    assert invalidated["output_asset_path"] is None

    output_path.write_bytes(original_output)
    restored = run_interactive_articulation_workflow(request, client=client)
    assert restored.status == "cancelled"
    assert restored.output_asset_path == str(output_path)
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 0


def test_evidence_binding_failure_remains_resumable(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.articulation.client import (
        _ArticulationEvidenceBindingError,
    )

    class BindingFailureOnceClient(MockArticulationWorkflowClient):
        attempted_authoring = False

        def author(
            self,
            request: ArticulationAuthoringRequest,
            *,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationAuthoringResult:
            if not self.attempted_authoring:
                self.attempted_authoring = True
                raise _ArticulationEvidenceBindingError(
                    "authoring candidate document digest changed before use"
                )
            return super().author(request, cancel_checker=cancel_checker)

    client = BindingFailureOnceClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    with pytest.raises(
        ArticulationWorkflowError,
        match="authoring evidence binding failed",
    ):
        run_interactive_articulation_workflow(request, client=client)

    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "authoring"
    assert checkpoint["error"] is None

    completed = run_interactive_articulation_workflow(request, client=client)
    assert completed.status == "completed"
    assert _operation_count(client, "author") == 1


def test_resume_rejects_source_and_candidate_evidence_drift(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="medium"))
    )
    request = _request(tmp_path)
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"

    source = Path(request.source_asset)
    source.write_text(
        source.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8"
    )
    with pytest.raises(ArticulationWorkflowError, match="Source asset digest"):
        run_interactive_articulation_workflow(request, client=client)

    source.write_text(
        '#usda 1.0\n\ndef Xform "FileCabinet" {\n}\n',
        encoding="utf-8",
    )
    candidate_path = request.output_dir / "articulation_candidates.json"
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_payload["tampered"] = True
    candidate_path.write_text(
        json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArticulationWorkflowError, match="candidate document digest"):
        run_interactive_articulation_workflow(request, client=client)


def test_resume_rejects_composed_source_dependency_drift(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    client = MockArticulationWorkflowClient(
        _document(_candidate(candidate_id, 1, confidence="medium"))
    )
    request = _request(tmp_path)
    source = Path(request.source_asset)
    dependency = source.with_name("cabinet_geometry.usda")
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "CabinetGeometry" {}\n',
        encoding="utf-8",
    )
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@cabinet_geometry.usda@]\n)\n"
        '\ndef Xform "FileCabinet" {}\n',
        encoding="utf-8",
    )
    root_sha256_before, dependency_sha256_before = _source_identity(source)

    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept"},
        reviewer="fixture-owner",
    )
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "ChangedCabinetGeometry" {}\n',
        encoding="utf-8",
    )
    root_sha256_after, dependency_sha256_after = _source_identity(source)

    assert root_sha256_after == root_sha256_before
    assert dependency_sha256_after != dependency_sha256_before
    with pytest.raises(
        ArticulationWorkflowError,
        match="Source dependency bundle digest",
    ):
        run_interactive_articulation_workflow(request, client=client)
    assert _operation_count(client, "infer") == 1
    assert _operation_count(client, "author") == 0


def test_review_receipt_must_cover_exact_checkpointed_scope(
    tmp_path: Path,
) -> None:
    client = MockArticulationWorkflowClient(
        _document(
            _candidate("candidate_0001", 1, confidence="medium"),
            _candidate("candidate_0002", 2, confidence="medium"),
        )
    )
    request = _request(tmp_path)
    run_interactive_articulation_workflow(request, client=client)

    with pytest.raises(ArticulationWorkflowError, match="cover exactly"):
        build_articulation_review_receipt(
            request.output_dir,
            {"candidate_0001": "accept"},
            reviewer="fixture-owner",
        )


@pytest.mark.parametrize("tamper_provider_evidence", [False, True])
def test_local_joint_adapter_wraps_inference_without_apply_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_provider_evidence: bool,
) -> None:
    request = _request(tmp_path, allowed_motion_types=("prismatic",))
    candidate_document = _document(_candidate("candidate_0001", 1))
    candidates_path = tmp_path / "joint-candidates.json"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "candidates.html"
    structure_evidence_path = tmp_path / "structure-provider-responses.json"
    stage1_evidence_path = tmp_path / "stage1-provider-responses.json"
    candidates_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text('{"id": "/World/Cabinet"}\n', encoding="utf-8")
    report_path.write_text("<html>candidate report</html>\n", encoding="utf-8")
    structure_evidence_path.write_text('{"attempts": []}\n', encoding="utf-8")
    stage1_evidence_path.write_text('{"attempts": []}\n', encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_pipeline(config: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
        captured["config"] = config
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            success=True,
            error=None,
            step_results={
                "analyze_structure": {
                    "structure_provider_response_diagnostics_path": str(
                        structure_evidence_path
                    ),
                    "structure_provider_response_diagnostics_sha256": (
                        "0" * 64
                        if tamper_provider_evidence
                        else file_sha256(structure_evidence_path)
                    ),
                    "structure_metadata": {
                        "structure_outcome": "articulated",
                        "reasoning": "Provider-backed structure was articulated.",
                        "evidence": {"accepted": True, "dof": 1},
                    },
                },
                "predict": {
                    "predictions_path": str(predictions_path),
                    "provider_response_diagnostics_path": str(stage1_evidence_path),
                    "provider_response_diagnostics_sha256": file_sha256(
                        stage1_evidence_path
                    ),
                },
                "infer_articulation_candidates": {
                    "articulation_candidates_path": str(candidates_path),
                    "articulation_report_path": str(report_path),
                },
            },
            session_id="joint-session",
            completed_steps=["predict", "infer_articulation_candidates"],
            working_dir=tmp_path / "joint-work",
        )

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    api_module = types.ModuleType("joint_agent.api")
    api_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    functions_module = types.ModuleType("joint_agent.functions")
    functions_module.__path__ = []  # type: ignore[attr-defined]
    membership_module = types.ModuleType("joint_agent.functions.membership_disposition")
    candidate_prim = candidate_document.candidates[0].moving_part_prims[0]
    membership_document = _membership_document(
        {
            "disposition_id": "membership_candidate_0001",
            "member_prim": candidate_prim,
            "motion_candidate_prim": candidate_prim,
            "physical_owner_prim": candidate_prim,
            "physical_owner_candidate_prim": candidate_prim,
            "disposition": "independent_motion",
            "source": "predicted",
            "confidence": "high",
            "source_prediction_ids": ["/World/Cabinet"],
            "downstream_boundary": "moving_candidate_generation",
            "review_status": "resolved",
        }
    )

    def fake_membership_disposition(
        predictions: Any,
        candidate_document: Stage2CandidateDocument,
        *,
        output_key: str = "classification",
        candidate_joint_types: tuple[str, ...] = ("revolute", "prismatic"),
    ) -> MembershipDispositionDocument:
        captured["membership_predictions"] = list(predictions)
        captured["membership_candidates"] = candidate_document
        captured["membership_output_key"] = output_key
        captured["membership_joint_types"] = candidate_joint_types
        return membership_document

    membership_module.infer_membership_dispositions = (  # type: ignore[attr-defined]
        fake_membership_disposition
    )
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.api", api_module)
    monkeypatch.setitem(sys.modules, "joint_agent.functions", functions_module)
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.membership_disposition",
        membership_module,
    )

    client = JointAgentLocalClient(
        {
            "project": {"name": "test"},
            "input": {"usd_path": "stale.usda"},
            "steps": {},
        }
    )
    if tamper_provider_evidence:
        with pytest.raises(
            ValueError,
            match="analyze_structure provider-response diagnostics digest changed",
        ):
            client.infer(request, resume=True)
        return

    result = client.infer(request, resume=True)

    assert result.candidate_document == candidate_document
    assert result.membership_disposition_document == membership_document
    assert result.predictions_sha256 == file_sha256(predictions_path)
    assert result.metadata["membership_disposition_required"] is True
    assert result.metadata["membership_disposition_classic_fallback_used"] is False
    assert result.metadata["structure_analysis_outcome"] == "articulated"
    assert result.metadata["structure_analysis_reasoning"] == (
        "Provider-backed structure was articulated."
    )
    assert result.metadata["structure_analysis_evidence"] == {
        "accepted": True,
        "dof": 1,
    }
    assert result.metadata["structure_provider_response_diagnostics_path"] == str(
        structure_evidence_path.resolve()
    )
    assert result.metadata["structure_provider_response_diagnostics_sha256"] == (
        file_sha256(structure_evidence_path)
    )
    assert result.metadata["stage1_provider_response_diagnostics_path"] == str(
        stage1_evidence_path.resolve()
    )
    assert result.metadata["stage1_provider_response_diagnostics_sha256"] == (
        file_sha256(stage1_evidence_path)
    )
    assert result.backend_run_id == "joint-session"
    assert captured["config"]["input"]["usd_path"] == str(
        Path(request.source_asset).resolve()
    )
    assert captured["config"]["steps"]["apply_joint_rigger"]["enabled"] is False
    assert captured["config"]["steps"]["author_physics_schemas"]["enabled"] is False
    assert captured["config"]["steps"]["infer_articulation_candidates"][
        "candidate_joint_types"
    ] == ["prismatic"]
    assert (
        captured["config"]["steps"]["build_dataset_prepare_dataset"]["prompts"]["user"]
        == request.intent
    )
    assert captured["kwargs"]["resume"] is True
    assert captured["membership_predictions"] == [{"id": "/World/Cabinet"}]
    assert captured["membership_candidates"] == candidate_document
    assert captured["membership_output_key"] == "classification"
    assert captured["membership_joint_types"] == ("prismatic",)


def test_local_joint_adapter_fails_closed_without_membership_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, allowed_motion_types=("prismatic",))
    candidate_document = _document(_candidate("candidate_0001", 1))
    candidates_path = tmp_path / "joint-candidates.json"
    predictions_path = tmp_path / "predictions.jsonl"
    candidates_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text('{"id": "/World/Cabinet"}\n', encoding="utf-8")

    def fake_pipeline(config: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            error=None,
            step_results={
                "predict": {"predictions_path": str(predictions_path)},
                "infer_articulation_candidates": {
                    "articulation_candidates_path": str(candidates_path),
                },
            },
            session_id="joint-session",
            completed_steps=["predict", "infer_articulation_candidates"],
            working_dir=tmp_path / "joint-work",
        )

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    api_module = types.ModuleType("joint_agent.api")
    api_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    functions_module = types.ModuleType("joint_agent.functions")
    functions_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.api", api_module)
    monkeypatch.setitem(sys.modules, "joint_agent.functions", functions_module)

    client = JointAgentLocalClient(
        {
            "project": {"name": "test"},
            "input": {"usd_path": "stale.usda"},
            "steps": {},
        }
    )
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.membership_disposition",
        None,
    )
    with pytest.raises(RuntimeError, match="membership disposition failed closed"):
        client.infer(request, resume=True)


def test_local_joint_adapter_suppresses_membership_for_verified_non_articulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path,
        allowed_motion_types=("prismatic",),
        expected_candidate_count=0,
    )
    candidate_document = _document()
    candidates_path = tmp_path / "joint-candidates.json"
    predictions_path = tmp_path / "predictions.jsonl"
    structure_evidence_path = tmp_path / "structure-provider-responses.json"
    stage1_evidence_path = tmp_path / "stage1-provider-responses.json"
    candidates_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text('{"id": "/World/Ladder"}\n', encoding="utf-8")
    structure_evidence_path.write_text(
        json.dumps(
            {
                "attempts": [],
                "whole_asset_structure": {
                    "accepted": True,
                    "dof": 0,
                    "segment_names": [],
                    "source_prim_inventory": ["/World/Ladder"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stage1_evidence_path.write_text('{"attempts": []}\n', encoding="utf-8")

    def fake_pipeline(config: dict[str, Any], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            error=None,
            step_results={
                "analyze_structure": {
                    "structure_provider_response_diagnostics_path": str(
                        structure_evidence_path
                    ),
                    "structure_provider_response_diagnostics_sha256": file_sha256(
                        structure_evidence_path
                    ),
                    "structure_metadata": {
                        "structure_outcome": "not_articulated",
                        "reasoning": (
                            "Provider-backed analysis found zero degrees of freedom."
                        ),
                        "evidence": {
                            "accepted": True,
                            "dof": 0,
                            "segment_names": [],
                            "source_prim_inventory": ["/World/Ladder"],
                        },
                    },
                },
                "predict": {
                    "predictions_path": str(predictions_path),
                    "provider_response_diagnostics_path": str(stage1_evidence_path),
                    "provider_response_diagnostics_sha256": file_sha256(
                        stage1_evidence_path
                    ),
                },
                "infer_articulation_candidates": {
                    "articulation_candidates_path": str(candidates_path),
                },
            },
            session_id="joint-session",
            completed_steps=["predict", "infer_articulation_candidates"],
            working_dir=tmp_path / "joint-work",
        )

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    api_module = types.ModuleType("joint_agent.api")
    api_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    functions_module = types.ModuleType("joint_agent.functions")
    functions_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.api", api_module)
    monkeypatch.setitem(sys.modules, "joint_agent.functions", functions_module)
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.membership_disposition",
        None,
    )

    result = JointAgentLocalClient(
        {"project": {"name": "test"}, "input": {}, "steps": {}}
    ).infer(request, resume=False)

    assert result.candidate_document == candidate_document
    assert result.membership_disposition_document.dispositions == ()
    assert result.metadata["membership_disposition_required"] is False
    assert result.metadata["membership_disposition_suppressed_by_structure"] is True
    assert result.metadata["structure_analysis_outcome"] == "not_articulated"


@pytest.mark.parametrize(
    ("failure_stage", "error_type", "reason"),
    [
        ("validation", "ValueError", None),
        (
            "whole_asset_structure",
            "ProviderResponseConformanceTerminalError",
            "incompatible_whole_asset_taxonomy",
        ),
    ],
    ids=["topology-reconciliation", "provider-response-conformance"],
)
def test_local_joint_adapter_preserves_typed_reconciliation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    error_type: str,
    reason: str | None,
) -> None:
    request = _request(tmp_path)
    terminal_status = {
        "requested": True,
        "attempted": True,
        "accepted": False,
        "outcome": "failed",
        "failure_stage": failure_stage,
        "error_type": error_type,
        "reason": reason,
        "diagnostics_artifact_path": str(tmp_path / "provider-attempts.json"),
        "diagnostics_artifact_sha256": "a" * 64,
        "diagnostics_artifact_status": "persisted",
        "diagnostics_persistence_error_type": None,
        "success": False,
        "terminal": True,
        "required": True,
        "recovery_action": {
            "kind": "resume",
            "operation": "joint_agent.api.pipeline",
            "resume": True,
            "clean": False,
            "failed_step": "infer_articulation_candidates",
            "checkpoint_path": str(tmp_path / ".pipeline_state.json"),
        },
    }

    def fake_pipeline(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            error="Pipeline execution failed",
            terminal_status=terminal_status,
        )

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    api_module = types.ModuleType("joint_agent.api")
    api_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.api", api_module)

    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})
    with pytest.raises(JointAgentInferenceTerminalError) as exc_info:
        client.infer(request, resume=False)

    assert exc_info.value.status == terminal_status


@pytest.mark.parametrize("terminal_shape", ["none", "missing"])
def test_local_joint_adapter_preserves_legacy_failure_without_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_shape: str,
) -> None:
    request = _request(tmp_path)

    def fake_pipeline(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        fields: dict[str, Any] = {
            "success": False,
            "error": "legacy pipeline failure",
        }
        if terminal_shape == "none":
            fields["terminal_status"] = None
        return SimpleNamespace(**fields)

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    api_module = types.ModuleType("joint_agent.api")
    api_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.api", api_module)

    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})
    with pytest.raises(RuntimeError, match="legacy pipeline failure"):
        client.infer(request, resume=False)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("success", True),
        ("terminal", False),
        ("required", False),
        ("outcome", "accepted"),
    ],
)
def test_required_reconciliation_terminal_status_rejects_nonfailure_payloads(
    tmp_path: Path,
    field_name: str,
    invalid_value: Any,
) -> None:
    recovery_action = {
        "kind": "resume",
        "operation": "joint_agent.api.pipeline",
        "resume": True,
        "clean": False,
        "failed_step": "infer_articulation_candidates",
        "checkpoint_path": str(tmp_path / ".pipeline_state.json"),
    }
    terminal_status = {
        "requested": True,
        "attempted": True,
        "accepted": False,
        "outcome": "failed",
        "failure_stage": "validation",
        "error_type": "ValueError",
        "success": False,
        "terminal": True,
        "required": True,
        "recovery_action": recovery_action,
    }
    terminal_status[field_name] = invalid_value

    with pytest.raises(ValueError):
        JointAgentInferenceTerminalError(terminal_status)


def _terminal_status_with_diagnostics_evidence(
    tmp_path: Path,
    *,
    recovery_kind: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "requested": True,
        "attempted": True,
        "accepted": False,
        "outcome": "failed",
        "failure_stage": "stage1_provider_evidence",
        "error_type": "ProviderResponseConformanceTerminalError",
        "success": False,
        "terminal": True,
        "required": True,
        "recovery_action": {
            "kind": recovery_kind,
            "operation": "joint_agent.api.pipeline",
            "resume": recovery_kind == "resume",
            "clean": recovery_kind == "restart",
            "failed_step": "predict",
            "checkpoint_path": str(tmp_path / ".pipeline_state.json"),
        },
        **evidence,
    }


@pytest.mark.parametrize(
    ("recovery_kind", "evidence"),
    [
        ("resume", {}),
        (
            "resume",
            {
                "diagnostics_artifact_status": "persisted",
                "diagnostics_artifact_path": "/tmp/provider-attempts.json",
                "diagnostics_artifact_sha256": "a" * 64,
            },
        ),
        (
            "restart",
            {"diagnostics_artifact_status": "unavailable"},
        ),
    ],
    ids=["legacy", "persisted-resume", "unavailable-restart"],
)
def test_terminal_diagnostics_evidence_accepts_consistent_recovery(
    tmp_path: Path,
    recovery_kind: str,
    evidence: dict[str, Any],
) -> None:
    status = _terminal_status_with_diagnostics_evidence(
        tmp_path,
        recovery_kind=recovery_kind,
        evidence=evidence,
    )

    assert JointAgentInferenceTerminalError(status).status == status


@pytest.mark.parametrize(
    ("recovery_kind", "evidence", "message"),
    [
        (
            "resume",
            {
                "diagnostics_artifact_status": "persisted",
                "diagnostics_artifact_path": "/tmp/provider-attempts.json",
            },
            "persisted diagnostics evidence requires path and digest",
        ),
        (
            "resume",
            {
                "diagnostics_artifact_status": "persisted",
                "diagnostics_artifact_sha256": "a" * 64,
            },
            "persisted diagnostics evidence requires path and digest",
        ),
        (
            "restart",
            {
                "diagnostics_artifact_status": "persisted",
                "diagnostics_artifact_path": "/tmp/provider-attempts.json",
                "diagnostics_artifact_sha256": "a" * 64,
            },
            "persisted diagnostics evidence requires resume recovery",
        ),
        (
            "restart",
            {
                "diagnostics_artifact_status": "unavailable",
                "diagnostics_artifact_path": "/tmp/provider-attempts.json",
            },
            "unavailable diagnostics evidence cannot carry path or digest",
        ),
        (
            "restart",
            {
                "diagnostics_artifact_status": "unavailable",
                "diagnostics_artifact_sha256": "a" * 64,
            },
            "unavailable diagnostics evidence cannot carry path or digest",
        ),
        (
            "resume",
            {"diagnostics_artifact_status": "unavailable"},
            "unavailable diagnostics evidence requires restart recovery",
        ),
        (
            "resume",
            {"diagnostics_artifact_sha256": "a" * 64},
            "legacy terminal status cannot carry new diagnostics evidence",
        ),
        (
            "resume",
            {"diagnostics_persistence_error_type": "OSError"},
            "legacy terminal status cannot carry new diagnostics evidence",
        ),
        (
            "restart",
            {},
            "restart recovery requires unavailable diagnostics evidence",
        ),
    ],
    ids=[
        "persisted-missing-digest",
        "persisted-missing-path",
        "persisted-restart",
        "unavailable-with-path",
        "unavailable-with-digest",
        "unavailable-resume",
        "legacy-with-digest",
        "legacy-with-persistence-error",
        "restart-without-status",
    ],
)
def test_terminal_diagnostics_evidence_rejects_contradictory_recovery(
    tmp_path: Path,
    recovery_kind: str,
    evidence: dict[str, Any],
    message: str,
) -> None:
    status = _terminal_status_with_diagnostics_evidence(
        tmp_path,
        recovery_kind=recovery_kind,
        evidence=evidence,
    )

    with pytest.raises(ValueError, match=message):
        JointAgentInferenceTerminalError(status)


def test_final_result_requires_exact_terminal_recovery_action(tmp_path: Path) -> None:
    recovery_action = {
        "kind": "resume",
        "operation": "joint_agent.api.pipeline",
        "resume": True,
        "clean": False,
        "failed_step": "infer_articulation_candidates",
        "checkpoint_path": str(tmp_path / ".pipeline_state.json"),
    }
    terminal_status = {
        "requested": True,
        "attempted": True,
        "accepted": False,
        "outcome": "failed",
        "failure_stage": "validation",
        "error_type": "ValueError",
        "success": False,
        "terminal": True,
        "required": True,
        "recovery_action": recovery_action,
    }
    common = {
        "success": False,
        "status": "failed",
        "mode": "interactive",
        "output_dir": str(tmp_path),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "workflow_progress_path": str(tmp_path / "workflow_progress.json"),
        "final_summary_path": str(tmp_path / "final_summary.json"),
        "message": "Required topology reconciliation failed.",
    }

    with pytest.raises(ValueError, match="recovery action must match"):
        ArticulationFinalizationResult(
            **common,
            terminal_status=terminal_status,
            recovery_action={
                **recovery_action,
                "checkpoint_path": str(tmp_path / "other-checkpoint.json"),
            },
        )

    legacy_payload = ArticulationFinalizationResult(**common).model_dump(mode="json")
    assert "terminal_status" not in legacy_payload
    assert "recovery_action" not in legacy_payload


@pytest.mark.parametrize(
    "recovery_action",
    [
        {
            "kind": "restart",
            "operation": "joint_agent.api.pipeline",
            "resume": False,
            "clean": True,
            "failed_step": "predict",
            "checkpoint_path": "/tmp/joint/.pipeline_state.json",
        },
        {
            "kind": "resume",
            "operation": "joint_agent.api.pipeline",
            "resume": True,
            "clean": False,
            "failed_step": "predict",
            "checkpoint_path": "/tmp/joint/.pipeline_state.json",
        },
    ],
)
def test_terminal_recovery_action_preserves_exact_supported_mode(
    recovery_action: dict[str, Any],
) -> None:
    from content_agent_workflows.articulation.models import ArticulationRecoveryAction

    assert ArticulationRecoveryAction.model_validate(recovery_action).model_dump() == (
        recovery_action
    )


@pytest.mark.parametrize(
    ("kind", "resume", "clean"),
    [("resume", False, True), ("restart", True, False)],
)
def test_terminal_recovery_action_rejects_inconsistent_flags(
    kind: str,
    resume: bool,
    clean: bool,
) -> None:
    from content_agent_workflows.articulation.models import ArticulationRecoveryAction

    with pytest.raises(ValueError, match="flags must match"):
        ArticulationRecoveryAction(
            kind=kind,
            operation="joint_agent.api.pipeline",
            resume=resume,
            clean=clean,
            failed_step="predict",
            checkpoint_path="/tmp/joint/.pipeline_state.json",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "embedded_outer_review_path",
        "embedded_output_evidence_path",
        "embedded_terminal_receipt_path",
    ),
)
def test_v2_final_result_rejects_v3_output_review_paths(
    tmp_path: Path,
    field_name: str,
) -> None:
    payload = {
        "schema_version": (
            "content-agent-workflows.articulation-finalization-result.v2"
        ),
        "success": False,
        "status": "failed",
        "mode": "interactive",
        "output_dir": str(tmp_path),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "workflow_progress_path": str(tmp_path / "workflow_progress.json"),
        "final_summary_path": str(tmp_path / "final_summary.json"),
        "message": "Synthetic failed finalization.",
        field_name: str(tmp_path / f"{field_name}.json"),
    }

    with pytest.raises(ValueError, match="require finalization v3"):
        ArticulationFinalizationResult.model_validate(payload)


def test_local_joint_adapter_configuration_binds_relative_path_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    config = {
        "project": {},
        "input": {"reference_images": ["refs/view.png"]},
        "steps": {},
    }
    first_cwd = tmp_path / "first-cwd"
    second_cwd = tmp_path / "second-cwd"
    first_cwd.mkdir()
    second_cwd.mkdir()
    inline = JointAgentLocalClient(config)
    monkeypatch.chdir(first_cwd)
    first_inline_digest = inline.configuration_sha256(request)
    monkeypatch.chdir(second_cwd)
    second_inline_digest = inline.configuration_sha256(request)
    assert first_inline_digest != second_inline_digest

    first = JointAgentLocalClient(
        config,
        source_config_path=tmp_path / "first" / "config.yaml",
    )
    second = JointAgentLocalClient(
        config,
        source_config_path=tmp_path / "second" / "config.yaml",
    )

    assert first.configuration_sha256(request) != second.configuration_sha256(request)
    monkeypatch.chdir(first_cwd)
    first_anchored_digest = first.configuration_sha256(request)
    monkeypatch.chdir(second_cwd)
    assert first.configuration_sha256(request) == first_anchored_digest
    with_session_a = JointAgentLocalClient(config, session_id="session-a")
    with_session_b = JointAgentLocalClient(config, session_id="session-b")
    assert with_session_a.configuration_sha256(
        request
    ) != with_session_b.configuration_sha256(request)


def test_resume_rejects_backend_configuration_drift_before_authoring(
    tmp_path: Path,
) -> None:
    class MutableConfigurationClient(MockArticulationWorkflowClient):
        configuration_digest = "1" * 64

        def configuration_sha256(
            self,
            request: ArticulationWorkflowRequest,
        ) -> str:
            del request
            return self.configuration_digest

    client = MutableConfigurationClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    durable_paths = tuple(
        request.output_dir / name
        for name in (
            "checkpoint.json",
            "workflow_progress.json",
            "final_summary.json",
        )
    )
    durable_bytes = {path: path.read_bytes() for path in durable_paths}

    client.configuration_digest = "2" * 64
    with pytest.raises(
        ArticulationWorkflowError,
        match="configuration differs",
    ):
        run_interactive_articulation_workflow(request, client=client)
    assert _operation_count(client, "author") == 0
    assert {path: path.read_bytes() for path in durable_paths} == durable_bytes


def test_configuration_drift_does_not_preserve_corrupt_nonterminal_summary(
    tmp_path: Path,
) -> None:
    class MutableConfigurationClient(MockArticulationWorkflowClient):
        configuration_digest = "1" * 64

        def configuration_sha256(
            self,
            request: ArticulationWorkflowRequest,
        ) -> str:
            del request
            return self.configuration_digest

    client = MutableConfigurationClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    candidate_path = request.output_dir / "articulation_candidates.json"
    candidate_path.write_bytes(candidate_path.read_bytes() + b"\n")
    client.configuration_digest = "2" * 64

    with pytest.raises(
        ArticulationWorkflowError,
        match="configuration differs",
    ):
        run_interactive_articulation_workflow(request, client=client)

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["candidate_document_path"] is None
    assert summary["inference_result_path"] is None


@pytest.mark.parametrize(
    ("index_name", "tampered_field", "tampered_value"),
    (
        ("final_summary.json", "message", "tampered durable summary"),
        ("workflow_progress.json", "phase", "completed"),
    ),
)
def test_configuration_drift_preserves_only_valid_durable_indexes(
    tmp_path: Path,
    index_name: str,
    tampered_field: str,
    tampered_value: str,
) -> None:
    class MutableConfigurationClient(MockArticulationWorkflowClient):
        configuration_digest = "1" * 64

        def configuration_sha256(
            self,
            request: ArticulationWorkflowRequest,
        ) -> str:
            del request
            return self.configuration_digest

    client = MutableConfigurationClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(request, client=client)
    assert paused.status == "needs_review"
    index_path = request.output_dir / index_name
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    index_payload[tampered_field] = tampered_value
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    client.configuration_digest = "2" * 64

    with pytest.raises(
        ArticulationWorkflowError,
        match="configuration differs",
    ):
        run_interactive_articulation_workflow(request, client=client)

    summary = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["output_asset_path"] is None
    progress = json.loads(
        (request.output_dir / "workflow_progress.json").read_text(encoding="utf-8")
    )
    assert progress["phase"] == "failed"


def test_local_joint_adapter_matches_current_repository_api_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repository_root / "apps" / "joint_agent"))
    for module_name in tuple(sys.modules):
        if module_name == "joint_agent" or module_name.startswith("joint_agent."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()

    from joint_agent.api import pipeline
    from joint_agent.functions.joint_rigger_adapter import apply_joint_rigger

    assert {
        "config",
        "skip_steps",
        "session_id",
        "resume",
        "clean",
        "verbose",
        "cancel_checker",
        "source_config_path",
    }.issubset(inspect.signature(pipeline).parameters)
    assert {
        "input_usd_path",
        "predictions_path",
        "output_usd_path",
        "diagnostics_path",
        "validation_path",
        "articulation_candidates_path",
        "adapter",
        "on_missing_dependency",
        "on_unready_candidates",
        "apply_masses",
        "apply_collision",
    }.issubset(inspect.signature(apply_joint_rigger).parameters)


def test_local_joint_adapter_rechecks_bindings_before_authoring_side_effect(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    candidate_document = _document(_candidate("candidate_0002", 2))
    candidate_path = tmp_path / "swapped-approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(
        request.source_asset
    )
    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})
    swapped = ArticulationAuthoringRequest(
        source_asset=request.source_asset,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=("candidate_0001",),
        idempotency_key="4" * 64,
        output_dir=request.output_dir,
    )
    with pytest.raises(ValueError, match="accepted_candidate_ids"):
        client.author(swapped)

    drifted = swapped.model_copy(
        update={
            "accepted_candidate_ids": ("candidate_0002",),
            "source_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="source digest"):
        client.author(drifted)


def test_local_joint_adapter_rejects_dependency_drift_before_authoring(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    source = Path(request.source_asset)
    dependency = source.with_name("cabinet_geometry.usda")
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "CabinetGeometry" {}\n',
        encoding="utf-8",
    )
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@cabinet_geometry.usda@]\n)\n"
        '\ndef Xform "FileCabinet" {}\n',
        encoding="utf-8",
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(source)
    candidate_document = _document(_candidate("candidate_0001", 1))
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    authoring_request = ArticulationAuthoringRequest(
        source_asset=str(source),
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=("candidate_0001",),
        idempotency_key="4" * 64,
        output_dir=request.output_dir,
    )
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "ChangedCabinetGeometry" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source dependency bundle changed"):
        JointAgentLocalClient({"project": {}, "input": {}, "steps": {}}).author(
            authoring_request
        )
    assert not (request.output_dir / "joint_rigger").exists()


def test_local_joint_adapter_authoring_is_topology_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import content_agent_workflows.articulation.client as articulation_client
    from content_agent_workflows.articulation.client import (
        _ArticulationEvidenceBindingError,
    )

    request = _request(tmp_path)
    candidate_document = _document(_candidate("candidate_0001", 1))
    candidates_path = tmp_path / "approved.json"
    candidates_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    from world_understanding.functions.physics.joint_rigger import (
        ArtifactIdentityV1,
        FieldProvenanceV1,
        JointPlanV1,
        JointRiggerInputV1,
        JointRiggerPlanV1,
        JointTopologyV1,
        canonical_sha256,
    )

    artifact = ArtifactIdentityV1(
        uri="memory://workflow-test/source",
        root_sha256="a" * 64,
    )

    def provenance(field: str, prim_path: str) -> FieldProvenanceV1:
        return FieldProvenanceV1(
            source="accepted_manifest",
            artifact=artifact,
            prim_path=prim_path,
            properties=(field,),
            evidence=f"Bound {field} fixture evidence.",
        )

    fake_bound_request = JointRiggerInputV1(
        schema_version="world-understanding-joint-rigger-input-v1",
        source_asset=artifact,
        plan=JointRiggerPlanV1(
            schema_version="world-understanding-joint-rigger-plan-v1",
            joints=(
                JointPlanV1(
                    topology=JointTopologyV1(
                        joint_id="/World/Joints/candidate_0001",
                        joint_type="prismatic",
                        body0="/World/Cabinet",
                        body1="/World/Cabinet/Drawer_01",
                        axis_stage=(1.0, 0.0, 0.0),
                        field_provenance={
                            "joint_type": provenance(
                                "joint_type",
                                "/World/Cabinet/Drawer_01",
                            ),
                            "body0": provenance("body0", "/World/Cabinet"),
                            "body1": provenance(
                                "body1",
                                "/World/Cabinet/Drawer_01",
                            ),
                            "axis_stage": provenance(
                                "axis_stage",
                                "/World/Cabinet/Drawer_01",
                            ),
                        },
                    ),
                ),
            ),
        ),
    )
    captured: dict[str, Any] = {}

    def fake_apply_joint_rigger(**kwargs: Any) -> dict[str, Any]:
        call_count = int(captured.get("call_count", 0)) + 1
        captured.update(kwargs)
        captured["call_count"] = call_count
        output_path = Path(kwargs["output_usd_path"])
        diagnostics_path = Path(kwargs["diagnostics_path"])
        result_path = Path(kwargs["validation_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mock-usdz")
        _write_joint_rigger_contract_artifacts(
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            result_path=result_path,
            joint_paths=("/World/Joints/candidate_0001",),
            root_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
            dependency_bundle_sha256="3" * 64,
            input_sha256=canonical_sha256(fake_bound_request),
            plan_sha256=canonical_sha256(fake_bound_request.plan),
            backend_name="owned_topology",
        )
        return {
            "joint_rigger_status": "authored",
            "authored_joint_count": 1,
        }

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    functions_module = types.ModuleType("joint_agent.functions")
    functions_module.__path__ = []  # type: ignore[attr-defined]
    adapter_module = types.ModuleType("joint_agent.functions.joint_rigger_adapter")
    adapter_module.apply_joint_rigger = fake_apply_joint_rigger  # type: ignore[attr-defined]
    core_bridge_module = types.ModuleType(
        "joint_agent.functions.joint_rigger_core_bridge"
    )
    core_bridge_module.build_stage2_articulation_contract_input = (  # type: ignore[attr-defined]
        lambda **_kwargs: fake_bound_request
    )
    core_bridge_module.build_stage2_candidate_edges_input = (  # type: ignore[attr-defined]
        lambda **_kwargs: fake_bound_request
    )
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.functions", functions_module)
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.joint_rigger_adapter",
        adapter_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.joint_rigger_core_bridge",
        core_bridge_module,
    )
    monkeypatch.setattr(
        articulation_client,
        "_validate_saved_owned_core_contract",
        lambda *_args, **_kwargs: (),
    )

    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})
    source_sha256, source_dependency_bundle_sha256 = _source_identity(
        request.source_asset
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_payload = '{"prediction": "drawer"}\n'
    predictions_path.write_text(predictions_payload, encoding="utf-8")
    predictions_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    authoring_request = ArticulationAuthoringRequest(
        source_asset=request.source_asset,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidates_path),
        candidate_document_sha256=hashlib.sha256(
            candidates_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=("candidate_0001",),
        idempotency_key="4" * 64,
        predictions_path=str(predictions_path),
        predictions_sha256=predictions_sha256,
        output_dir=request.output_dir,
    )
    result = client.author(authoring_request)
    predictions_path.write_text(
        '{"prediction": "changed"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="authoring predictions digest changed before use",
    ):
        client.author(authoring_request)
    assert captured["call_count"] == 1

    predictions_path.write_text(predictions_payload, encoding="utf-8")
    recovered = client.author(authoring_request)

    assert result.authored_candidate_ids == ("candidate_0001",)
    assert recovered.authored_candidate_ids == ("candidate_0001",)
    assert recovered.metadata["recovered_from_published_artifacts"] is True
    assert captured["call_count"] == 1
    assert captured["adapter"] == "owned_core"
    assert captured["on_unready_candidates"] == "block"
    assert captured["apply_masses"] is False
    assert captured["apply_collision"] is False
    assert captured["predictions_path"] == str(predictions_path)
    attempt = json.loads(
        (request.output_dir / "joint_rigger" / "authoring_attempt.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["predictions_path"] == str(predictions_path)
    assert attempt["predictions_sha256"] == predictions_sha256

    publication_dir = request.output_dir / "joint_rigger"
    diagnostics_path = publication_dir / "diagnostics.json"
    result_path = publication_dir / "result.json"
    original_diagnostics = diagnostics_path.read_bytes()
    original_result = result_path.read_bytes()
    diagnostics_payload = json.loads(original_diagnostics)
    result_payload = json.loads(original_result)
    diagnostics_payload["backend_name"] = "stage2_candidate_edges"
    result_payload["diagnostics"]["backend_name"] = "stage2_candidate_edges"
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload),
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="stage2_candidate_edges results cannot carry a predictions binding",
    ):
        client.author(authoring_request)
    diagnostics_path.write_bytes(original_diagnostics)
    result_path.write_bytes(original_result)

    result_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="published authoring evidence is invalid",
    ):
        client.author(authoring_request)


@pytest.mark.parametrize(
    "published_prefix",
    [
        ("diagnostics.json",),
        ("diagnostics.json", "result.json"),
    ],
    ids=["diagnostics-only", "diagnostics-and-result"],
)
def test_local_joint_adapter_retries_root_last_report_prefix_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_prefix: tuple[str, ...],
) -> None:
    from world_understanding.functions.physics.joint_rigger import (
        ArtifactIdentityV1,
        JointRiggerInputV1,
        JointRiggerPlanV1,
        canonical_sha256,
    )

    import content_agent_workflows.articulation.client as articulation_client

    fake_bound_request = JointRiggerInputV1(
        schema_version="world-understanding-joint-rigger-input-v1",
        source_asset=ArtifactIdentityV1(
            uri="memory://workflow-test/source",
            root_sha256="a" * 64,
        ),
        plan=JointRiggerPlanV1(
            schema_version="world-understanding-joint-rigger-plan-v1",
            joints=(),
        ),
    )
    backend_calls = 0

    def fake_apply_joint_rigger(**kwargs: Any) -> dict[str, Any]:
        nonlocal backend_calls
        backend_calls += 1
        output_path = Path(kwargs["output_usd_path"])
        diagnostics_path = Path(kwargs["diagnostics_path"])
        result_path = Path(kwargs["validation_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if backend_calls == 1:
            _write_joint_rigger_contract_artifacts(
                output_path=output_path,
                diagnostics_path=diagnostics_path,
                result_path=result_path,
                joint_paths=("/World/Joints/candidate_0001",),
                root_sha256="7" * 64,
                dependency_bundle_sha256="8" * 64,
                input_sha256=canonical_sha256(fake_bound_request),
                plan_sha256=canonical_sha256(fake_bound_request.plan),
            )
            for report_path in (diagnostics_path, result_path):
                if report_path.name not in published_prefix:
                    report_path.unlink()
            raise ArticulationWorkflowInterrupted(
                "simulated process death before root promotion"
            )

        output_path.write_bytes(b"republished-usdz")
        _write_joint_rigger_contract_artifacts(
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            result_path=result_path,
            joint_paths=("/World/Joints/candidate_0001",),
            root_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
            dependency_bundle_sha256="9" * 64,
            input_sha256=canonical_sha256(fake_bound_request),
            plan_sha256=canonical_sha256(fake_bound_request.plan),
        )
        return {
            "joint_rigger_status": "authored",
            "authored_joint_count": 1,
        }

    joint_module = types.ModuleType("joint_agent")
    joint_module.__path__ = []  # type: ignore[attr-defined]
    functions_module = types.ModuleType("joint_agent.functions")
    functions_module.__path__ = []  # type: ignore[attr-defined]
    adapter_module = types.ModuleType("joint_agent.functions.joint_rigger_adapter")
    adapter_module.apply_joint_rigger = fake_apply_joint_rigger  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent", joint_module)
    monkeypatch.setitem(sys.modules, "joint_agent.functions", functions_module)
    monkeypatch.setitem(
        sys.modules,
        "joint_agent.functions.joint_rigger_adapter",
        adapter_module,
    )
    monkeypatch.setattr(
        articulation_client,
        "_build_bound_owned_core_request",
        lambda _request: fake_bound_request,
    )
    monkeypatch.setattr(
        articulation_client,
        "_validate_saved_owned_core_contract",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        articulation_client,
        "_validate_saved_owned_core_graph",
        lambda *_args, **_kwargs: (
            ("candidate_0001",),
            (),
            ("/World/Joints/candidate_0001",),
        ),
    )

    class LocalAuthoringWorkflowClient(MockArticulationWorkflowClient):
        def __init__(self, candidate_document: Stage2CandidateDocument) -> None:
            super().__init__(candidate_document)
            self.local_client = JointAgentLocalClient(
                {"project": {}, "input": {}, "steps": {}}
            )

        def author(
            self,
            request: ArticulationAuthoringRequest,
            *,
            cancel_checker: CancelChecker | None = None,
        ) -> ArticulationAuthoringResult:
            return self.local_client.author(
                request,
                cancel_checker=cancel_checker,
            )

    client = LocalAuthoringWorkflowClient(_document(_candidate("candidate_0001", 1)))
    request = _request(tmp_path)

    with pytest.raises(
        ArticulationWorkflowInterrupted,
        match="before root promotion",
    ):
        run_interactive_articulation_workflow(request, client=client)

    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "authoring"
    assert checkpoint["error"] is None
    publication_dir = request.output_dir / "joint_rigger"
    assert (publication_dir / "authoring_attempt.json").is_file()
    assert not (publication_dir / "rigged.usdz").exists()
    assert not (request.output_dir / "authoring_result.json").exists()
    assert {
        path.name
        for path in (
            publication_dir / "diagnostics.json",
            publication_dir / "result.json",
        )
        if path.is_file()
    } == set(published_prefix)

    completed = run_interactive_articulation_workflow(request, client=client)

    assert completed.status == "completed"
    assert backend_calls == 2
    assert (publication_dir / "rigged.usdz").read_bytes() == b"republished-usdz"
    assert (request.output_dir / "authoring_result.json").is_file()


def test_local_joint_adapter_rejects_stale_complete_artifacts_without_attempt(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    candidate_document = _document(_candidate("candidate_0001", 1))
    candidates_path = tmp_path / "approved.json"
    candidates_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    output_dir = request.output_dir / "joint_rigger"
    output_dir.mkdir(parents=True)
    output_path = output_dir / "rigged.usdz"
    diagnostics_path = output_dir / "diagnostics.json"
    result_path = output_dir / "result.json"
    output_path.write_bytes(b"stale-usdz")
    _write_joint_rigger_contract_artifacts(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        result_path=result_path,
        joint_paths=("/World/Joints/candidate_0001",),
        root_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        dependency_bundle_sha256="3" * 64,
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(
        request.source_asset
    )
    authoring_request = ArticulationAuthoringRequest(
        source_asset=request.source_asset,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidates_path),
        candidate_document_sha256=hashlib.sha256(
            candidates_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=("candidate_0001",),
        idempotency_key="4" * 64,
        output_dir=request.output_dir,
    )

    with pytest.raises(RuntimeError, match="without a matching prior"):
        JointAgentLocalClient({"project": {}, "input": {}, "steps": {}}).author(
            authoring_request
        )
    assert not (output_dir / "authoring_attempt.json").exists()


def test_local_joint_adapter_validates_joint_rigger_identity(
    tmp_path: Path,
) -> None:
    candidate_document = _document(
        _candidate("candidate_0001", 1, lower_limit=0.0, upper_limit=0.5),
        _candidate("candidate_0002", 2),
    )
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    output_path, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    from pxr import Usd

    stage = Usd.Stage.Open(str(output_path))
    assert stage is not None
    saved_joint = stage.GetPrimAtPath(joint_paths[0])
    # Some OpenUSD builds expose this schema fallback and some do not. When
    # present, it must remain composed rather than authored by the workflow.
    if "userDocBrief" in saved_joint.GetCustomData():
        assert not saved_joint.HasAuthoredCustomDataKey("userDocBrief")
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    diagnostics_path = tmp_path / "joint-rigger-diagnostics.json"
    result_path = tmp_path / "joint-rigger-result.json"
    _write_joint_rigger_contract_artifacts(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        result_path=result_path,
        joint_paths=joint_paths,
    )
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(output_path),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=("candidate_0001", "candidate_0002"),
        authored_joint_count=2,
        diagnostics_path=str(diagnostics_path),
        diagnostics_sha256=hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        joint_rigger_result_path=str(result_path),
        joint_rigger_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=("candidate_0001", "candidate_0002"),
    )

    assert validation.status == "pass"
    assert validation.exact_graph_match is True
    assert not any("transitional customData" in item for item in validation.failures)
    assert validation.self_contained is True
    assert validation.validated_candidate_ids == (
        "candidate_0001",
        "candidate_0002",
    )
    assert validation.metadata["request_identity_source_bound"] is False


def test_local_joint_adapter_validates_owned_core_relocated_joint_identity(
    tmp_path: Path,
) -> None:
    candidate_document = _document(
        _candidate("candidate_0001", 1),
        _candidate("candidate_0002", 2),
    )
    authoring = _owned_core_saved_graph_authoring(
        tmp_path,
        candidate_document,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "pass"
    assert validation.exact_graph_match is True
    assert validation.validated_candidate_ids == candidate_document.candidate_ids
    assert (
        validation.metadata["joint_diagnostic_ids"]
        != validation.metadata["joint_diagnostic_paths"]
    )


def test_local_joint_adapter_rejects_unbound_owned_core_relocated_body_endpoints(
    tmp_path: Path,
) -> None:
    candidate_document = _document(
        _candidate(
            "candidate_0001",
            1,
            fixed_parent_prim="/World/Cabinet",
            moving_part_prim="/World/Cabinet/Drawer_01",
        )
    )
    authoring = _owned_core_saved_graph_authoring(
        tmp_path,
        candidate_document,
        relocate_body_endpoints=True,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any(
        "unsupported authored body0 endpoint" in failure
        for failure in validation.failures
    )


def test_local_joint_adapter_validates_real_fixed_aggregate_contract_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    repository_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repository_root / "apps" / "joint_agent"))
    for module_name in tuple(sys.modules):
        if module_name == "joint_agent" or module_name.startswith("joint_agent."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()

    source_path = tmp_path / "fixed-aggregate.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for prim_path in ("/World/base", "/World/base_trim", "/World/door"):
        UsdGeom.Xform.Define(stage, prim_path)
        UsdGeom.Cube.Define(stage, f"{prim_path}/visual")
    UsdGeom.Xform(stage.GetPrimAtPath("/World/door")).AddTranslateOp().Set(
        Gf.Vec3d(1.0, 0.0, 0.0)
    )
    assert stage.GetRootLayer().Save()
    del stage

    candidate_document = _document(
        _candidate(
            "candidate_0001",
            1,
            fixed_parent_prim="/World/base",
            moving_part_prim="/World/door",
        )
    )
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    prediction_rows = tuple(
        {
            "id": prim_path,
            "classification": {
                "role": "body",
                "is_articulation_candidate": False,
                "joint_type_hint": "none",
                "instance_id": "cabinet_fixed_body",
                "provenance": {
                    "field_sources": {"instance_id": "predicted"},
                },
            },
        }
        for prim_path in ("/World/base", "/World/base_trim")
    )
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(source_path)
    authoring_request = ArticulationAuthoringRequest(
        source_asset=str(source_path),
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=candidate_document.candidate_ids,
        idempotency_key="4" * 64,
        predictions_path=str(predictions_path),
        predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        output_dir=tmp_path / "run",
    )
    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})

    authoring = client.author(authoring_request)
    validation = client.validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "pass", validation.failures
    assert validation.exact_graph_match is True
    assert validation.metadata["request_identity_source_bound"] is True
    output_stage = Usd.Stage.Open(authoring.output_asset_path)
    assert output_stage is not None
    joints = tuple(
        UsdPhysics.Joint(prim)
        for prim in output_stage.TraverseAll()
        if prim.IsA(UsdPhysics.Joint)
    )
    assert len(joints) == 1
    assert str(joints[0].GetBody0Rel().GetTargets()[0]).startswith(
        "/World/__JointAgent_"
    )
    assert tuple(str(path) for path in joints[0].GetBody1Rel().GetTargets()) == (
        "/World/door",
    )

    del joints, output_stage
    output_path = Path(authoring.output_asset_path)
    result_path = Path(authoring.joint_rigger_result_path or "")
    diagnostics_path = Path(authoring.diagnostics_path or "")
    original_output_bytes = output_path.read_bytes()
    original_result_bytes = result_path.read_bytes()
    original_diagnostics_bytes = diagnostics_path.read_bytes()

    from content_agent_workflows.articulation.client import (
        _ArticulationEvidenceBindingError,
    )

    for field, expected_failure in (
        (
            "input_sha256",
            "result input identity differs from the exact bound request",
        ),
        (
            "plan_sha256",
            "result plan identity differs from the exact bound request",
        ),
    ):
        result_payload = json.loads(original_result_bytes)
        result_payload[field] = "f" * 64
        result_path.chmod(0o600)
        result_path.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        drifted_authoring = authoring.model_copy(
            update={
                "joint_rigger_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            }
        )

        rejected = client.validate(
            drifted_authoring,
            expected_candidate_ids=candidate_document.candidate_ids,
        )

        assert rejected.status == "fail"
        assert rejected.exact_graph_match is False
        assert any(expected_failure in failure for failure in rejected.failures), (
            rejected.failures
        )
        with pytest.raises(
            _ArticulationEvidenceBindingError,
            match="published authoring evidence is invalid",
        ):
            client.author(authoring_request)

    result_payload = json.loads(original_result_bytes)
    diagnostics_payload = result_payload["diagnostics"]
    diagnostics_payload["field_decisions"] = [
        decision
        for decision in diagnostics_payload["field_decisions"]
        if not decision["field"].startswith("articulation_roots[")
    ]
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_path.chmod(0o600)
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stripped_diagnostics_authoring = authoring.model_copy(
        update={
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "diagnostics_sha256": hashlib.sha256(
                diagnostics_path.read_bytes()
            ).hexdigest(),
        }
    )

    rejected = client.validate(
        stripped_diagnostics_authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert rejected.status == "fail"
    assert rejected.exact_graph_match is False
    assert any(
        "diagnostics top-level are missing planned field decision(s): "
        "articulation_roots[" in failure
        for failure in rejected.failures
    ), rejected.failures
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="published authoring evidence is invalid",
    ):
        client.author(authoring_request)

    result_payload = json.loads(original_result_bytes)
    diagnostics_payload = result_payload["diagnostics"]
    diagnostics_payload["joint_diagnostics"][0]["field_decisions"].append(
        {
            "detail": "forged unplanned decision",
            "disposition": "ignored",
            "field": "forged_joint_fact",
            "reason_code": "not_provided",
        }
    )
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_path.chmod(0o600)
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forged_diagnostics_authoring = authoring.model_copy(
        update={
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "diagnostics_sha256": hashlib.sha256(
                diagnostics_path.read_bytes()
            ).hexdigest(),
        }
    )

    rejected = client.validate(
        forged_diagnostics_authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert rejected.status == "fail"
    assert rejected.exact_graph_match is False
    assert any(
        "unexpected field decision(s): forged_joint_fact" in failure
        for failure in rejected.failures
    ), rejected.failures
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="published authoring evidence is invalid",
    ):
        client.author(authoring_request)

    for damage, expected_failure in (
        ("joint_marker", "wrong Joint Rigger plan identity"),
        ("drive", "Saved Joint Rigger contract validation failed"),
        ("aggregate_metadata", "aggregate_member_changed"),
        ("aggregate_member", "authored_aggregate_mismatch"),
        ("articulation_root", "authored_articulation_roots_mismatch"),
        ("rigid_body_api", "Saved RigidBodyAPI inventory differs"),
        ("mass_api", "Saved MassAPI inventory differs"),
        ("collision_api", "Saved CollisionAPI inventory differs"),
    ):
        if output_path.exists():
            output_path.unlink()
        output_path.write_bytes(original_output_bytes)
        result_path.chmod(0o600)
        result_path.write_bytes(original_result_bytes)
        diagnostics_path.chmod(0o600)
        diagnostics_path.write_bytes(original_diagnostics_bytes)

        extracted = tmp_path / f"tampered-package-{damage}"
        with zipfile.ZipFile(output_path) as archive:
            member_names = tuple(
                item.filename for item in archive.infolist() if not item.is_dir()
            )
            archive.extractall(extracted)
        root_member = extracted / member_names[0]
        tampered_stage = Usd.Stage.Open(str(root_member))
        assert tampered_stage is not None
        tampered_joint_prim = next(
            prim for prim in tampered_stage.TraverseAll() if prim.IsA(UsdPhysics.Joint)
        )
        tampered_joint = UsdPhysics.Joint(tampered_joint_prim)
        if damage == "joint_marker":
            tampered_joint_prim.SetCustomDataByKey(
                "jointRigger:planSha256",
                "f" * 64,
            )
        elif damage == "drive":
            drive = UsdPhysics.DriveAPI.Apply(tampered_joint_prim, "linear")
            assert drive
            drive.CreateStiffnessAttr(100.0)
        else:
            body0_path = tampered_joint.GetBody0Rel().GetTargets()[0]
            aggregate_body = tampered_stage.GetPrimAtPath(body0_path)
            assert aggregate_body and aggregate_body.IsValid()
            if damage == "aggregate_metadata":
                aggregate_body.SetCustomDataByKey(
                    "jointRigger:rigidLinkMemberSnapshotSha256",
                    "f" * 64,
                )
            elif damage == "aggregate_member":
                aggregate_children = tuple(aggregate_body.GetAllChildren())
                assert aggregate_children
                tampered_stage.RemovePrim(aggregate_children[0].GetPath())
            elif damage == "rigid_body_api":
                assert UsdPhysics.RigidBodyAPI.Apply(aggregate_body)
            elif damage == "mass_api":
                assert UsdPhysics.MassAPI.Apply(aggregate_body)
            elif damage == "collision_api":
                assert UsdPhysics.CollisionAPI.Apply(aggregate_body)
            else:
                root = next(
                    prim
                    for prim in tampered_stage.TraverseAll()
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                )
                assert root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        assert tampered_stage.GetRootLayer().Save()
        del tampered_joint, tampered_joint_prim, tampered_stage

        tampered_output = tmp_path / f"tampered-{damage}.usdz"
        writer = Usd.ZipFileWriter.CreateNew(str(tampered_output))
        assert writer
        for member_name in member_names:
            assert writer.AddFile(str(extracted / member_name), member_name)
        assert writer.Save()
        output_path.unlink()
        tampered_output.replace(output_path)

        result_payload = json.loads(original_result_bytes)
        root_sha256, dependency_sha256 = _source_identity(output_path)
        result_payload["output_artifact"]["root_sha256"] = root_sha256
        result_payload["output_artifact"]["dependency_bundle_sha256"] = (
            dependency_sha256
        )
        result_path.chmod(0o600)
        result_path.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tampered_authoring = authoring.model_copy(
            update={
                "output_asset_sha256": hashlib.sha256(
                    output_path.read_bytes()
                ).hexdigest(),
                "joint_rigger_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
            }
        )

        rejected = client.validate(
            tampered_authoring,
            expected_candidate_ids=candidate_document.candidate_ids,
        )

        assert rejected.status == "fail"
        assert rejected.exact_graph_match is False
        assert any(expected_failure in failure for failure in rejected.failures), (
            rejected.failures
        )
        with pytest.raises(
            _ArticulationEvidenceBindingError,
            match="published authoring evidence is invalid",
        ):
            client.author(authoring_request)


def test_local_joint_adapter_rejects_real_v1_drive_mass_and_collision_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from content_agent_workflows.articulation.client import (
        _ArticulationEvidenceBindingError,
    )

    repository_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repository_root / "apps" / "joint_agent"))
    for module_name in tuple(sys.modules):
        if module_name == "joint_agent" or module_name.startswith("joint_agent."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()

    source_path = tmp_path / "v1-cabinet.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Cabinet")
    UsdGeom.Xform.Define(stage, "/World/Cabinet/Drawer_01")
    UsdGeom.Xform(stage.GetPrimAtPath("/World/Cabinet/Drawer_01")).AddTranslateOp().Set(
        Gf.Vec3d(1.0, 0.0, 0.0)
    )
    assert stage.GetRootLayer().Save()
    del stage

    candidate_document = _document(_candidate("candidate_0001", 1))
    candidate_path = tmp_path / "approved-v1.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "v1-predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "/World/Cabinet",
                "classification": {
                    "role": "body",
                    "is_articulation_candidate": False,
                    "joint_type_hint": "none",
                    "instance_id": "cabinet_fixed_body",
                    "provenance": {
                        "field_sources": {"instance_id": "predicted"},
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(source_path)
    authoring_request = ArticulationAuthoringRequest(
        source_asset=str(source_path),
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=candidate_document.candidate_ids,
        idempotency_key="4" * 64,
        predictions_path=str(predictions_path),
        predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        output_dir=tmp_path / "run-v1",
    )
    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})

    authoring = client.author(authoring_request)
    assert (
        authoring.metadata["joint_rigger_input"]["schema_version"]
        == "world-understanding-joint-rigger-input-v1"
    )
    validation = client.validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )
    assert validation.status == "pass", validation.failures

    output_path = Path(authoring.output_asset_path)
    result_path = Path(authoring.joint_rigger_result_path or "")
    diagnostics_path = Path(authoring.diagnostics_path or "")
    original_result_bytes = result_path.read_bytes()
    original_diagnostics_bytes = diagnostics_path.read_bytes()
    result_payload = json.loads(original_result_bytes)
    diagnostics_payload = result_payload["diagnostics"]
    diagnostics_payload["field_decisions"].append(
        {
            "detail": "forged unplanned decision",
            "disposition": "ignored",
            "field": "forged_top_level_fact",
            "reason_code": "not_provided",
        }
    )
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_path.chmod(0o600)
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forged_diagnostics_authoring = authoring.model_copy(
        update={
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "diagnostics_sha256": hashlib.sha256(
                diagnostics_path.read_bytes()
            ).hexdigest(),
        }
    )

    rejected = client.validate(
        forged_diagnostics_authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert rejected.status == "fail"
    assert rejected.exact_graph_match is False
    assert any(
        "unexpected field decision(s): forged_top_level_fact" in failure
        for failure in rejected.failures
    ), rejected.failures
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="published authoring evidence is invalid",
    ):
        client.author(authoring_request)
    result_path.chmod(0o600)
    result_path.write_bytes(original_result_bytes)
    diagnostics_path.chmod(0o600)
    diagnostics_path.write_bytes(original_diagnostics_bytes)

    extracted = tmp_path / "tampered-v1-package"
    with zipfile.ZipFile(output_path) as archive:
        member_names = tuple(
            item.filename for item in archive.infolist() if not item.is_dir()
        )
        archive.extractall(extracted)
    root_member = extracted / member_names[0]
    tampered_stage = Usd.Stage.Open(str(root_member))
    assert tampered_stage is not None
    joint_prim = next(
        prim for prim in tampered_stage.TraverseAll() if prim.IsA(UsdPhysics.Joint)
    )
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
    assert drive
    drive.CreateStiffnessAttr(100.0)
    fixed_body = tampered_stage.GetPrimAtPath("/World/Cabinet")
    assert UsdPhysics.RigidBodyAPI.Apply(fixed_body)
    assert UsdPhysics.ArticulationRootAPI.Apply(fixed_body)
    assert UsdPhysics.MassAPI.Apply(fixed_body)
    assert UsdPhysics.CollisionAPI.Apply(fixed_body)
    assert tampered_stage.GetRootLayer().Save()
    del tampered_stage

    tampered_output = tmp_path / "tampered-v1.usdz"
    writer = Usd.ZipFileWriter.CreateNew(str(tampered_output))
    assert writer
    for member_name in member_names:
        assert writer.AddFile(str(extracted / member_name), member_name)
    assert writer.Save()
    output_path.unlink()
    tampered_output.replace(output_path)

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    root_sha256, dependency_sha256 = _source_identity(output_path)
    result_payload["output_artifact"]["root_sha256"] = root_sha256
    result_payload["output_artifact"]["dependency_bundle_sha256"] = dependency_sha256
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_authoring = authoring.model_copy(
        update={
            "output_asset_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        }
    )

    rejected = client.validate(
        tampered_authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert rejected.status == "fail"
    assert rejected.exact_graph_match is False
    for expected_failure in (
        "Saved Joint Rigger contract validation failed",
        "Saved RigidBodyAPI inventory differs",
        "Saved ArticulationRootAPI inventory differs",
        "Saved MassAPI inventory differs",
        "Saved CollisionAPI inventory differs",
    ):
        assert any(expected_failure in failure for failure in rejected.failures), (
            rejected.failures
        )
    with pytest.raises(
        _ArticulationEvidenceBindingError,
        match="published authoring evidence is invalid",
    ):
        client.author(authoring_request)


@pytest.mark.parametrize(
    ("damage", "expected_failure"),
    [
        (
            "body1_link",
            "does not resolve to exactly one approved candidate by source/link "
            "provenance",
        ),
        (
            "body0_link",
            "body0 link provenance incompatible with the approved candidate",
        ),
    ],
)
def test_local_joint_adapter_rejects_owned_core_relocated_endpoint_drift(
    tmp_path: Path,
    damage: str,
    expected_failure: str,
) -> None:
    candidate_document = _document(_candidate("candidate_0001", 1))
    authoring = _owned_core_saved_graph_authoring(
        tmp_path,
        candidate_document,
        relocate_body_endpoints=True,
        tamper_body0_link_id=damage == "body0_link",
        tamper_body1_link_id=damage == "body1_link",
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any(expected_failure in failure for failure in validation.failures)


def test_local_joint_adapter_rejects_owned_core_joint_identity_drift(
    tmp_path: Path,
) -> None:
    candidate_document = _document(_candidate("candidate_0001", 1))
    authoring = _owned_core_saved_graph_authoring(
        tmp_path,
        candidate_document,
        tamper_joint_id=True,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("wrong Joint Rigger ID" in failure for failure in validation.failures)


def test_local_joint_adapter_rejects_owned_core_without_authored_paths(
    tmp_path: Path,
) -> None:
    candidate_document = _document(_candidate("candidate_0001", 1))
    output_path, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )
    diagnostics_path = Path(authoring.diagnostics_path or "")
    result_path = Path(authoring.joint_rigger_result_path or "")
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics_payload.update(
        {
            "backend_name": "owned_topology",
            "backend_version": "world-understanding-joint-topology-author-v1",
        }
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload["diagnostics"] = diagnostics_payload
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authoring = authoring.model_copy(
        update={
            "diagnostics_sha256": hashlib.sha256(
                diagnostics_path.read_bytes()
            ).hexdigest(),
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        }
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any(
        "omit every authored joint prim path" in failure
        for failure in validation.failures
    )


@pytest.mark.parametrize(
    "missing_endpoint_field",
    ("topology.body0", "topology.body1"),
)
def test_owned_core_binding_rejects_endpoint_provenance_without_prim_path(
    missing_endpoint_field: str,
) -> None:
    from world_understanding.functions.physics.joint_rigger import (
        ArtifactIdentityV1,
        FieldDecisionV1,
        FieldProvenanceV1,
        JointDiagnosticV1,
    )

    from content_agent_workflows.articulation.client import (
        _resolve_owned_core_diagnostic_bindings,
    )

    candidate_document = _document(_candidate("candidate_0001", 1))
    candidate = candidate_document.candidates[0]
    assert candidate.fixed_parent_prim is not None
    joint_id = "/World/LogicalJoints/joint_0000"
    artifact = ArtifactIdentityV1(
        uri="memory://joint-agent/articulation-contract/test",
        root_sha256="a" * 64,
    )

    def endpoint_provenance(
        field: str,
        prim_path: str,
        property_name: str,
    ) -> FieldProvenanceV1:
        if field == missing_endpoint_field:
            return FieldProvenanceV1(
                source="template_default",
                evidence="Endpoint provenance without a prim path.",
            )
        return FieldProvenanceV1(
            source="accepted_manifest",
            artifact=artifact,
            prim_path=prim_path,
            properties=(f"joint:{joint_id}.{property_name}",),
            evidence="Exact endpoint provenance.",
        )

    diagnostic = JointDiagnosticV1(
        joint_id=joint_id,
        field_decisions=(
            FieldDecisionV1(
                field="topology.body0",
                disposition="accepted",
                provenance=endpoint_provenance(
                    "topology.body0",
                    candidate.fixed_parent_prim,
                    "body0_link",
                ),
            ),
            FieldDecisionV1(
                field="topology.body1",
                disposition="accepted",
                provenance=endpoint_provenance(
                    "topology.body1",
                    candidate.moving_part_prims[0],
                    "body1_link",
                ),
            ),
            FieldDecisionV1(
                field="usd.joint_prim_path",
                disposition="defaulted",
                reason_code="deterministic_joint_path",
                detail="/World/Joints/joint_0000",
            ),
        ),
    )

    bindings, _authored_paths, failures = _resolve_owned_core_diagnostic_bindings(
        candidate_document,
        joint_diagnostics=(diagnostic,),
        plan_sha256="2" * 64,
        backend_name="owned_topology",
        backend_version="world-understanding-joint-topology-author-v1",
    )

    assert bindings == ()
    assert (
        f"Joint Rigger diagnostic {joint_id} lacks accepted body endpoint provenance."
        in failures
    )


def test_local_joint_adapter_rejects_external_usdz_dependency(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    candidate_document = _document(_candidate("candidate_0001", 1))
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    external_path = tmp_path / "external.usda"
    external_path.write_text(
        '#usda 1.0\n\ndef Xform "External" {}\n',
        encoding="utf-8",
    )
    external_asset_path = tmp_path / "external.bin"
    external_asset_path.write_bytes(b"external")
    internal_path = tmp_path / "internal.usda"
    internal_path.write_text(
        "#usda 1.0\n"
        "(\n"
        f"    subLayers = [@{external_path.resolve()}@]\n"
        ")\n"
        '\ndef Scope "Nested" {\n'
        f"    custom asset externalAsset = @{external_asset_path.resolve()}@\n"
        "}\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    stage.GetRootLayer().subLayerPaths.append("internal.usda")
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "external-reference.usdz"
    writer = Usd.ZipFileWriter.CreateNew(str(output_path))
    assert writer
    assert writer.AddFile(str(source_path), "rigged.usda")
    assert writer.AddFile(str(internal_path), "internal.usda")
    assert writer.Save()
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
        root_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        dependency_bundle_sha256="3" * 64,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.self_contained is False
    joined_failures = "\n".join(validation.failures)
    assert "outside the sealed archive" in joined_failures
    assert str(external_path.resolve()) in joined_failures
    assert str(external_asset_path.resolve()) in joined_failures


def test_raw_usd_identity_rejects_resolver_backed_remote_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.physics.joint_rigger as joint_rigger
    import world_understanding.functions.physics.joint_rigger.reference as reference

    from content_agent_workflows.articulation.client import (
        _validate_self_contained_raw_usd_identity,
    )

    output_path = tmp_path / "rigged.usda"
    output_path.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    root_sha256 = file_sha256(output_path)
    dependency_bundle_sha256 = "3" * 64
    monkeypatch.setattr(
        reference,
        "usd_dependency_inventory",
        lambda _path: (
            SimpleNamespace(
                package_outer_identifier=str(output_path.resolve()),
                asset_identifier="omniverse://library.example/remote.usda",
                identifier="omniverse://library.example/remote.usda",
            ),
        ),
    )
    monkeypatch.setattr(
        joint_rigger,
        "local_usd_dependency_paths",
        lambda _path: (output_path.resolve(),),
    )
    monkeypatch.setattr(
        joint_rigger,
        "identify_usd_artifact",
        lambda _path, *, uri: SimpleNamespace(
            root_sha256=root_sha256,
            dependency_bundle_sha256=dependency_bundle_sha256,
            uri=uri,
        ),
    )

    valid, observed_digest, failures = _validate_self_contained_raw_usd_identity(
        output_path,
        expected_root_sha256=root_sha256,
        expected_dependency_bundle_sha256=dependency_bundle_sha256,
    )

    assert valid is False
    assert observed_digest == dependency_bundle_sha256
    assert failures == (
        "Published raw USD resolves external URI dependencies: "
        "omniverse://library.example/remote.usda",
    )


def test_local_joint_adapter_rejects_dependency_bundle_identity_mismatch(
    tmp_path: Path,
) -> None:
    candidate_document = _document(_candidate("candidate_0001", 1))
    output_path, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
        dependency_bundle_sha256="0" * 64,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.self_contained is False
    assert any(
        "dependency bundle identity" in failure for failure in validation.failures
    )


@pytest.mark.parametrize(
    ("path_field", "label"),
    [
        ("joint_rigger_result_path", "Joint Rigger result"),
        ("candidate_document_path", "approved candidate document"),
    ],
)
def test_local_joint_adapter_rejects_bound_json_drift_before_parsing(
    tmp_path: Path,
    path_field: str,
    label: str,
) -> None:
    candidate_document = _document(_candidate("candidate_0001", 1))
    output_path, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )
    path_value = getattr(authoring, path_field)
    assert isinstance(path_value, str)
    Path(path_value).write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{label} digest changed before use"):
        JointAgentLocalClient({"project": {}, "input": {}, "steps": {}}).validate(
            authoring,
            expected_candidate_ids=candidate_document.candidate_ids,
        )


def test_local_joint_adapter_rejects_sidecar_only_candidate_match(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdUtils

    candidate_id = "candidate_0001"
    candidate_document = _document(_candidate(candidate_id, 1))
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint_prim = stage.GetPrimAtPath(joint_paths[0])
    joint_prim.ClearCustomDataByKey("jointAgent:candidateId")
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "fabricated.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    diagnostics_path = tmp_path / "joint-rigger-diagnostics.json"
    result_path = tmp_path / "joint-rigger-result.json"
    _write_joint_rigger_contract_artifacts(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        result_path=result_path,
        joint_paths=joint_paths,
    )
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(output_path),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=(candidate_id,),
        authored_joint_count=1,
        diagnostics_path=str(diagnostics_path),
        diagnostics_sha256=hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        joint_rigger_result_path=str(result_path),
        joint_rigger_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=(candidate_id,),
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert validation.validated_candidate_ids == ()


def test_local_joint_adapter_rejects_unmarked_instanced_physics_joint(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    candidate_document = _document(_candidate("candidate_0001", 1))
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)

    hidden_path = tmp_path / "hidden-rig.usda"
    hidden_stage = Usd.Stage.CreateNew(str(hidden_path))
    hidden_root = UsdGeom.Xform.Define(hidden_stage, "/HiddenRig")
    hidden_stage.SetDefaultPrim(hidden_root.GetPrim())
    UsdPhysics.FixedJoint.Define(hidden_stage, "/HiddenRig/HiddenJoint")
    assert hidden_stage.GetRootLayer().Save()

    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    instance_root = UsdGeom.Xform.Define(stage, "/World/HiddenRig").GetPrim()
    assert instance_root.GetReferences().AddReference(str(hidden_path))
    assert instance_root.SetInstanceable(True)
    assert stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(source_path))
    assert reopened is not None
    hidden_joint_path = "/World/HiddenRig/HiddenJoint"
    hidden_joint = reopened.GetPrimAtPath(hidden_joint_path)
    assert hidden_joint.IsInstanceProxy()
    assert hidden_joint_path not in {
        str(prim.GetPath()) for prim in reopened.Traverse()
    }

    output_path = tmp_path / "unmarked-instanced-joint.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any(
        "unapproved physics joints" in failure and hidden_joint_path in failure
        for failure in validation.failures
    )


def test_local_joint_adapter_rejects_saved_endpoint_drift(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdPhysics, UsdUtils

    candidate_id = "candidate_0001"
    candidate_document = _document(_candidate(candidate_id, 1))
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    joint.GetBody1Rel().SetTargets([Sdf.Path("/World/Cabinet")])
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "endpoint-drift.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    diagnostics_path = tmp_path / "joint-rigger-diagnostics.json"
    result_path = tmp_path / "joint-rigger-result.json"
    _write_joint_rigger_contract_artifacts(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        result_path=result_path,
        joint_paths=joint_paths,
    )
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(output_path),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=(candidate_id,),
        authored_joint_count=1,
        diagnostics_path=str(diagnostics_path),
        diagnostics_sha256=hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        joint_rigger_result_path=str(result_path),
        joint_rigger_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=(candidate_id,),
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("body1" in failure for failure in validation.failures)


def test_local_joint_adapter_rejects_saved_axis_drift(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdPhysics, UsdUtils

    candidate_document = _document(_candidate("candidate_0001", 1))
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    joint.GetAxisAttr().Set("Y")
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "axis-drift.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("axis" in failure for failure in validation.failures)


def test_local_joint_adapter_rejects_saved_shared_anchor_drift(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdUtils

    candidate_document = _document(_candidate("candidate_0001", 1))
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    drifted_anchor = Gf.Vec3f(1.0, 0.0, 0.0)
    joint.GetLocalPos0Attr().Set(drifted_anchor)
    joint.GetLocalPos1Attr().Set(drifted_anchor)
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "anchor-drift.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("anchor" in failure for failure in validation.failures)


def test_local_joint_adapter_accepts_float32_shared_anchor_roundoff(
    tmp_path: Path,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdPhysics, UsdUtils

    candidate_document = _document(_candidate("candidate_0001", 1))
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(3e-6, 0.0, 0.0))
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "anchor-roundoff.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "pass"
    assert validation.exact_graph_match is True


def test_local_joint_adapter_rejects_saved_prismatic_limit_drift(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdPhysics, UsdUtils

    candidate_document = _document(
        _candidate(
            "candidate_0001",
            1,
            lower_limit=0.0,
            upper_limit=0.5,
        )
    )
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    joint.GetUpperLimitAttr().Set(123.0)
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "limit-drift.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("upper limit" in failure for failure in validation.failures)


def test_local_joint_adapter_rejects_saved_limit_time_samples(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdPhysics, UsdUtils

    candidate_document = _document(
        _candidate(
            "candidate_0001",
            1,
            lower_limit=0.0,
            upper_limit=0.5,
        )
    )
    _, joint_paths = _write_rigged_usdz(tmp_path, candidate_document)
    source_path = tmp_path / "rigged.usda"
    stage = Usd.Stage.Open(str(source_path))
    assert stage is not None
    joint = UsdPhysics.PrismaticJoint.Get(stage, joint_paths[0])
    authored_default = joint.GetUpperLimitAttr().Get()
    joint.GetUpperLimitAttr().Set(authored_default, Usd.TimeCode(1.0))
    assert stage.GetRootLayer().Save()
    output_path = tmp_path / "limit-time-sample.usdz"
    assert UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(source_path)),
        str(output_path),
    )
    authoring = _saved_graph_authoring(
        tmp_path,
        candidate_document,
        output_path,
        joint_paths,
    )

    validation = JointAgentLocalClient(
        {"project": {}, "input": {}, "steps": {}}
    ).validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert validation.status == "fail"
    assert validation.exact_graph_match is False
    assert any("time samples" in failure for failure in validation.failures)


def test_scene_evidence_is_review_bound_and_indexed(
    tmp_path: Path,
) -> None:
    document = _document(
        *[_candidate(f"candidate_{index:04d}", index) for index in range(1, 7)]
    )
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(
        tmp_path,
        review_policy="all",
        allowed_motion_types=("prismatic",),
        expected_candidate_count=6,
    )

    paused = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )

    assert paused.status == "needs_review"
    assert collector.call_count == 1
    assert paused.inference_result_path
    assert paused.scene_evidence_path
    manifest_path = Path(paused.scene_evidence_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept" for candidate_id in document.candidate_ids},
        reviewer="workflow-test",
    )
    assert receipt.scene_evidence_sha256 == manifest_sha256

    completed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        review_receipt=receipt,
    )

    assert completed.status == "completed"
    assert collector.call_count == 1
    assert completed.approved_candidate_document_path
    assert completed.authoring_request_path
    assert completed.diagnostics_path
    assert completed.joint_rigger_result_path


def test_completed_replay_cannot_delete_persisted_scene_evidence_requirement(
    tmp_path: Path,
) -> None:
    client = _ExactReadbackFixtureClient(_document(_candidate("candidate_0001", 1)))
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path)
    completed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert completed.status == "completed"
    assert completed.scene_evidence_path is not None
    request_path = request.output_dir / "request.json"
    persisted_request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert (
        persisted_request_payload["metadata"][
            "content_agent_workflows.scene_evidence_required"
        ]
        is True
    )

    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["scene_evidence_configuration_sha256"] = None
    checkpoint["scene_evidence"] = None
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted_request = ArticulationWorkflowRequest.model_validate(
        persisted_request_payload
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="requires a Scene evidence collector",
    ):
        run_interactive_articulation_workflow(
            persisted_request,
            client=client,
        )

    invalidated = json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert invalidated["status"] == "failed"
    assert invalidated["scene_evidence_path"] is None
    assert invalidated["output_asset_path"] is None


def test_cancellation_after_scene_collection_stops_before_review(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.articulation import scene_evidence

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    cancelled = False
    owned_collection_dir: Path | None = None

    class CancellingSceneCollector:
        def __init__(self) -> None:
            self.uncommitted_result: Any | None = None
            self.collection_id: str | None = None
            self.marker_sha256: str | None = None

        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            nonlocal cancelled, owned_collection_dir
            result = delegate.collect(workflow_request, **kwargs)
            collection_id, owned_collection_dir, collection_artifact = (
                scene_evidence._create_collection_directory(Path(kwargs["output_dir"]))
            )
            cancelled = True
            self.collection_id = collection_id
            self.marker_sha256 = collection_artifact.sha256
            self.uncommitted_result = result.model_copy(
                update={
                    "collection_id": collection_id,
                    "collection_artifact": collection_artifact,
                }
            )
            return self.uncommitted_result

        def discard_uncommitted_collection(
            self,
            result: Any,
            *,
            evidence_root: Path,
        ) -> bool:
            if result is not self.uncommitted_result:
                raise ValueError("result lacks the invocation ownership capability")
            assert self.collection_id is not None
            assert self.marker_sha256 is not None
            removed = scene_evidence._discard_owned_collection(
                evidence_root,
                collection_id=self.collection_id,
                expected_marker_sha256=self.marker_sha256,
            )
            self.uncommitted_result = None
            return removed

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=CancellingSceneCollector(),
        cancel_checker=lambda: cancelled,
    )

    assert result.status == "cancelled"
    assert result.success is False
    assert result.scene_evidence_path is None
    assert result.review_receipt_path is None
    assert owned_collection_dir is not None
    assert not owned_collection_dir.exists()
    state = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert state["phase"] == "cancelled"
    assert state["scene_evidence"] is None

    fresh_collector = MockArticulationSceneEvidenceCollector()
    replayed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=fresh_collector,
    )

    assert replayed == result
    assert replayed.scene_evidence_path is None
    assert fresh_collector.call_count == 0
    assert _operation_count(client, "infer") == 1


def test_cancellation_during_scene_evidence_verification_stops_before_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    cancelled = False
    real_verify = workflow.verify_articulation_scene_evidence

    def verify_then_cancel(*args: Any, **kwargs: Any) -> Any:
        nonlocal cancelled
        result = real_verify(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(
        workflow,
        "verify_articulation_scene_evidence",
        verify_then_cancel,
    )

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        cancel_checker=lambda: cancelled,
    )

    assert result.status == "cancelled"
    assert result.success is False
    assert result.scene_evidence_path is None
    assert not (request.output_dir / "scene_evidence" / "manifest.json").exists()
    state = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert state["phase"] == "cancelled"
    assert state["scene_evidence"] is None


@pytest.mark.parametrize("failure_after_manifest_commit", (False, True))
def test_scene_evidence_manifest_failure_discards_only_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_after_manifest_commit: bool,
) -> None:
    from content_agent_workflows.articulation import scene_evidence, workflow

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")

    class OwnedSceneCollector:
        def __init__(self) -> None:
            self.collection_dir: Path | None = None
            self.collection_id: str | None = None
            self.marker_sha256: str | None = None
            self.uncommitted_result: Any | None = None
            self.released = False

        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            result = delegate.collect(workflow_request, **kwargs)
            collection_id, collection_dir, collection_artifact = (
                scene_evidence._create_collection_directory(Path(kwargs["output_dir"]))
            )
            self.collection_dir = collection_dir
            self.collection_id = collection_id
            self.marker_sha256 = collection_artifact.sha256
            self.uncommitted_result = result.model_copy(
                update={
                    "collection_id": collection_id,
                    "collection_artifact": collection_artifact,
                }
            )
            return self.uncommitted_result

        def discard_uncommitted_collection(
            self,
            result: Any,
            *,
            evidence_root: Path,
        ) -> bool:
            if result is not self.uncommitted_result:
                raise ValueError("result lacks the invocation ownership capability")
            assert self.collection_id is not None
            assert self.marker_sha256 is not None
            removed = scene_evidence._discard_owned_collection(
                evidence_root,
                collection_id=self.collection_id,
                expected_marker_sha256=self.marker_sha256,
            )
            self.uncommitted_result = None
            return removed

        def release_committed_collection(
            self,
            result: Any,
            *,
            manifest_path: Path,
        ) -> None:
            if result is not self.uncommitted_result:
                raise ValueError("result lacks the invocation ownership capability")
            assert json.loads(manifest_path.read_text(encoding="utf-8")) == (
                result.model_dump(mode="json")
            )
            self.uncommitted_result = None
            self.released = True

    collector = OwnedSceneCollector()
    real_write_once_json = workflow._write_once_json

    def fail_manifest_publication(
        path: Path,
        value: Any,
        *,
        label: str,
    ) -> Any:
        if label != "Scene evidence manifest":
            return real_write_once_json(path, value, label=label)
        if failure_after_manifest_commit:
            real_write_once_json(path, value, label=label)
        raise RuntimeError("simulated scene evidence manifest publication failure")

    monkeypatch.setattr(
        workflow,
        "verify_articulation_scene_evidence",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(workflow, "_write_once_json", fail_manifest_publication)

    if failure_after_manifest_commit:
        result = run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )
        assert result.status == "needs_review"
        assert result.scene_evidence_path is not None
    else:
        with pytest.raises(
            ArticulationWorkflowError,
            match="simulated scene evidence manifest publication failure",
        ):
            run_interactive_articulation_workflow(
                request,
                client=client,
                scene_evidence_collector=collector,
            )

    assert collector.collection_dir is not None
    assert collector.collection_dir.exists() is failure_after_manifest_commit
    assert collector.released is failure_after_manifest_commit
    manifest_path = request.output_dir / "scene_evidence" / "manifest.json"
    assert manifest_path.exists() is failure_after_manifest_commit


@pytest.mark.parametrize("cancel_point", ("manifest_write", "collection_release"))
def test_cancellation_after_manifest_commit_preserves_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_point: str,
) -> None:
    from content_agent_workflows.articulation import workflow

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")

    class ReleaseCancellingCollector:
        def __init__(self) -> None:
            self.release_count = 0
            self.discard_count = 0

        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            return delegate.collect(workflow_request, **kwargs)

        def release_committed_collection(
            self,
            result: Any,
            *,
            manifest_path: Path,
        ) -> None:
            assert json.loads(manifest_path.read_text(encoding="utf-8")) == (
                result.model_dump(mode="json")
            )
            self.release_count += 1
            if cancel_point == "collection_release":
                raise asyncio.CancelledError

        def discard_uncommitted_collection(
            self,
            result: Any,
            *,
            evidence_root: Path,
        ) -> bool:
            self.discard_count += 1
            return False

    collector = ReleaseCancellingCollector()
    real_write_once_json = workflow._write_once_json

    def cancel_after_manifest_write(
        path: Path,
        value: Any,
        *,
        label: str,
    ) -> Any:
        binding = real_write_once_json(path, value, label=label)
        if label == "Scene evidence manifest" and cancel_point == "manifest_write":
            raise asyncio.CancelledError
        return binding

    monkeypatch.setattr(workflow, "_write_once_json", cancel_after_manifest_write)
    result = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )

    manifest_path = (request.output_dir / "scene_evidence" / "manifest.json").resolve()
    assert result.status == "cancelled"
    assert result.scene_evidence_path == str(manifest_path)
    assert manifest_path.is_file()
    assert collector.release_count == (1 if cancel_point == "collection_release" else 0)
    assert collector.discard_count == 0
    assert delegate.call_count == 1
    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "cancelled"
    assert checkpoint["scene_evidence"]["path"] == str(manifest_path)
    assert checkpoint["scene_evidence"]["sha256"] == file_sha256(manifest_path)

    fresh_collector = MockArticulationSceneEvidenceCollector()
    replayed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=fresh_collector,
    )
    assert replayed == result
    assert fresh_collector.call_count == 0
    assert _operation_count(client, "infer") == 1


def test_scene_evidence_directory_symlink_is_rejected_before_collection(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    request.output_dir.mkdir(parents=True)
    (request.output_dir / "scene_evidence").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="evidence directory must not be a symbolic link",
    ):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )

    assert _operation_count(client, "infer") == 0
    assert collector.call_count == 0
    assert list(outside.iterdir()) == []


def test_scene_evidence_manifest_symlink_is_rejected_on_review(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert paused.status == "needs_review"
    assert paused.scene_evidence_path is not None

    manifest_path = Path(paused.scene_evidence_path)
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(outside_manifest)

    with pytest.raises(
        ArticulationWorkflowError,
        match="evidence manifest must not be a symbolic link",
    ):
        build_articulation_review_receipt(
            request.output_dir,
            {"candidate_0001": "accept"},
            reviewer="workflow-test",
        )

    assert not (request.output_dir / "review_receipt.json").exists()


def test_interruption_after_scene_evidence_checkpoint_resumes_without_recollection(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")

    def interrupt_after_evidence(boundary: str) -> None:
        if boundary == "scene_evidence_checkpointed":
            raise ArticulationWorkflowInterrupted("simulated process loss")

    with pytest.raises(ArticulationWorkflowInterrupted, match="process loss"):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
            phase_boundary_hook=interrupt_after_evidence,
        )

    assert _operation_count(client, "infer") == 1
    assert collector.call_count == 1
    resumed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        phase_boundary_hook=interrupt_after_evidence,
    )
    assert resumed.status == "needs_review"
    assert _operation_count(client, "infer") == 1
    assert collector.call_count == 1


def test_scene_evidence_collection_failure_checkpoints_inference_for_retry(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")

    class FailOnceSceneCollector:
        def __init__(self) -> None:
            self.call_count = 0

        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("simulated usd-cli scene render failure")
            return delegate.collect(workflow_request, **kwargs)

    collector = FailOnceSceneCollector()
    with pytest.raises(
        ArticulationWorkflowError,
        match="usd-cli evidence collection failed",
    ):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )

    checkpoint_path = request.output_dir / "checkpoint.json"
    failed_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert failed_checkpoint["phase"] == "collecting_evidence"
    assert failed_checkpoint["inference_result"] is not None
    assert failed_checkpoint["candidate_document"] is not None
    assert failed_checkpoint["candidate_ids"] == ["candidate_0001"]
    assert "simulated usd-cli scene render failure" in failed_checkpoint["error"]
    assert _operation_count(client, "infer") == 1

    resumed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )

    assert resumed.status == "needs_review"
    assert collector.call_count == 2
    assert delegate.call_count == 1
    assert _operation_count(client, "infer") == 1
    recovered_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert recovered_checkpoint["phase"] == "needs_review"
    assert recovered_checkpoint["scene_evidence"] is not None
    assert recovered_checkpoint["error"] is None


@pytest.mark.parametrize("cancel_mode", ("never", "sticky", "one_shot"))
def test_collecting_evidence_resume_reclaims_orphan_before_retry(
    tmp_path: Path,
    cancel_mode: str,
) -> None:
    from content_agent_workflows.articulation import scene_evidence

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    events: list[str] = []
    orphan_dir: Path | None = None
    cancel_checks = 0

    def resume_cancel_checker() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_mode == "sticky" or (
            cancel_mode == "one_shot" and cancel_checks == 1
        )

    class CrashThenRecoverCollector:
        def __init__(self) -> None:
            self.collect_count = 0

        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def reclaim_orphaned_collections(
            self,
            *,
            evidence_root: Path,
        ) -> tuple[str, ...]:
            events.append("reclaim")
            assert not (evidence_root / "manifest.json").exists()
            return scene_evidence._reclaim_orphaned_collections(evidence_root)

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            nonlocal orphan_dir
            events.append("collect")
            self.collect_count += 1
            if self.collect_count == 1:
                Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
                _, orphan_dir, _ = scene_evidence._create_collection_directory(
                    Path(kwargs["output_dir"])
                )
                candidate_dir = orphan_dir / "candidate-0000"
                candidate_dir.mkdir()
                staging_dir = candidate_dir / ".focus-000-view-00-download-crash123"
                staging_dir.mkdir()
                (staging_dir / "image.png").write_bytes(b"partial image")
                raise SystemExit("simulated process kill")
            return delegate.collect(workflow_request, **kwargs)

    collector = CrashThenRecoverCollector()
    with pytest.raises(SystemExit, match="simulated process kill"):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )

    assert events == ["collect"]
    assert orphan_dir is not None and orphan_dir.is_dir()
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "collecting_evidence"
    assert checkpoint["scene_evidence"] is None
    assert not (request.output_dir / "scene_evidence" / "manifest.json").exists()
    assert _operation_count(client, "infer") == 1

    cancel_on_resume = cancel_mode != "never"
    resumed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        cancel_checker=resume_cancel_checker if cancel_on_resume else None,
    )

    assert resumed.status == ("cancelled" if cancel_on_resume else "needs_review")
    assert events == (
        ["collect", "reclaim"]
        if cancel_on_resume
        else ["collect", "reclaim", "collect"]
    )
    assert orphan_dir is not None and not orphan_dir.exists()
    assert collector.collect_count == (1 if cancel_on_resume else 2)
    assert delegate.call_count == (0 if cancel_on_resume else 1)
    assert _operation_count(client, "infer") == 1
    assert cancel_checks == (1 if cancel_on_resume else 0)
    recovered_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert recovered_checkpoint["phase"] == (
        "cancelled" if cancel_on_resume else "needs_review"
    )
    assert (recovered_checkpoint["scene_evidence"] is not None) is (
        not cancel_on_resume
    )
    if cancel_on_resume:
        replayed = run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )
        assert replayed == resumed
        assert events == ["collect", "reclaim"]


def test_collecting_evidence_resume_binds_committed_manifest_before_cancel(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    delegate = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    events: list[str] = []

    class CrashAfterManifestCollector:
        def configuration_sha256(
            self,
            workflow_request: ArticulationWorkflowRequest,
        ) -> str:
            return delegate.configuration_sha256(workflow_request)

        def reclaim_orphaned_collections(
            self,
            *,
            evidence_root: Path,
        ) -> tuple[str, ...]:
            events.append("reclaim")
            return ()

        def collect(
            self,
            workflow_request: ArticulationWorkflowRequest,
            **kwargs: Any,
        ) -> Any:
            events.append("collect")
            return delegate.collect(workflow_request, **kwargs)

        def release_committed_collection(
            self,
            result: Any,
            *,
            manifest_path: Path,
        ) -> None:
            events.append("release")
            assert json.loads(manifest_path.read_text(encoding="utf-8")) == (
                result.model_dump(mode="json")
            )
            raise SystemExit("simulated process loss after manifest commit")

    collector = CrashAfterManifestCollector()
    with pytest.raises(SystemExit, match="process loss after manifest commit"):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )

    manifest_path = (request.output_dir / "scene_evidence" / "manifest.json").resolve()
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["phase"] == "collecting_evidence"
    assert checkpoint["scene_evidence"] is None
    assert manifest_path.is_file()
    assert events == ["collect", "release"]
    assert delegate.call_count == 1

    cancelled = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        cancel_checker=lambda: True,
    )

    assert cancelled.status == "cancelled"
    assert cancelled.scene_evidence_path == str(manifest_path)
    assert events == ["collect", "release"]
    assert delegate.call_count == 1
    recovered_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert recovered_checkpoint["phase"] == "cancelled"
    assert recovered_checkpoint["scene_evidence"]["path"] == str(manifest_path)
    replayed = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert replayed == cancelled
    assert events == ["collect", "release"]


def test_run_without_scene_collector_ignores_stale_manifest(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    document = _document(_candidate(candidate_id, 1))
    client = _ExactReadbackFixtureClient(document)
    request = _request(tmp_path, review_policy="all")
    stale_manifest_path = request.output_dir / "scene_evidence" / "manifest.json"
    stale_manifest_path.parent.mkdir(parents=True)
    stale_manifest_path.write_text('{"stale": true}\n', encoding="utf-8")

    paused = run_interactive_articulation_workflow(
        request,
        client=client,
    )

    assert paused.status == "needs_review"
    assert paused.scene_evidence_path is None
    receipt = build_articulation_review_receipt(
        request.output_dir,
        {candidate_id: "accept"},
        reviewer="workflow-test",
    )
    assert receipt.scene_evidence_sha256 is None

    completed = run_interactive_articulation_workflow(
        request,
        client=client,
        review_receipt=receipt,
    )
    assert completed.status == "completed"
    assert _operation_count(client, "infer") == 1


def test_resume_rejects_tampered_scene_render(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert paused.scene_evidence_path
    manifest = json.loads(Path(paused.scene_evidence_path).read_text(encoding="utf-8"))
    image_path = Path(manifest["candidates"][0]["renders"][0]["image_artifact"]["path"])
    image_path.write_bytes(b"tampered")

    with pytest.raises(
        ArticulationWorkflowError,
        match="Scene articulation evidence.*digest mismatch",
    ):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
        )

    assert _operation_count(client, "infer") == 1
    assert collector.call_count == 1


def test_fresh_completion_reverifies_scene_evidence_after_validation(
    tmp_path: Path,
) -> None:
    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path)

    def tamper_after_validation(boundary: str) -> None:
        if boundary != "validation_artifact_written":
            return
        manifest_path = request.output_dir / "scene_evidence" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_path = Path(
            manifest["candidates"][0]["renders"][0]["image_artifact"]["path"]
        )
        image_path.write_bytes(b"tampered after validation")

    with pytest.raises(
        ArticulationWorkflowError,
        match="Scene articulation evidence.*digest mismatch",
    ):
        run_interactive_articulation_workflow(
            request,
            client=client,
            scene_evidence_collector=collector,
            phase_boundary_hook=tamper_after_validation,
        )

    checkpoint = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["phase"] == "validating"
    assert not (request.output_dir / "final_summary.json").exists()


def test_cancellation_during_final_scene_reverification_prevents_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import workflow

    document = _document(_candidate("candidate_0001", 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path)
    cancelled = False
    real_load = workflow._load_scene_evidence

    def load_then_cancel(*args: Any, **kwargs: Any) -> Any:
        nonlocal cancelled
        result = real_load(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(workflow, "_load_scene_evidence", load_then_cancel)

    result = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
        cancel_checker=lambda: cancelled,
    )

    assert result.status == "cancelled"
    assert result.success is False
    assert result.scene_evidence_path is not None
    assert result.output_asset_path is not None
    assert _operation_count(client, "author") == 1
    assert _operation_count(client, "validate") == 1
    state = json.loads(
        (request.output_dir / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert state["phase"] == "cancelled"
    assert not json.loads(
        (request.output_dir / "final_summary.json").read_text(encoding="utf-8")
    )["success"]


def test_review_receipt_rejects_tampered_scene_render_without_mutation(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    document = _document(_candidate(candidate_id, 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert paused.status == "needs_review"
    assert paused.scene_evidence_path
    manifest = json.loads(Path(paused.scene_evidence_path).read_text(encoding="utf-8"))
    image_path = Path(manifest["candidates"][0]["renders"][0]["image_artifact"]["path"])
    image_path.write_bytes(b"tampered")
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint_before_review = checkpoint_path.read_bytes()

    with pytest.raises(
        ArticulationWorkflowError,
        match="Scene articulation evidence.*digest mismatch",
    ):
        build_articulation_review_receipt(
            request.output_dir,
            {candidate_id: "accept"},
            reviewer="workflow-test",
        )

    assert not (request.output_dir / "review_receipt.json").exists()
    assert checkpoint_path.read_bytes() == checkpoint_before_review


def test_review_receipt_rejects_missing_request_required_scene_evidence(
    tmp_path: Path,
) -> None:
    candidate_id = "candidate_0001"
    document = _document(_candidate(candidate_id, 1))
    client = _ExactReadbackFixtureClient(document)
    collector = MockArticulationSceneEvidenceCollector()
    request = _request(tmp_path, review_policy="all")
    paused = run_interactive_articulation_workflow(
        request,
        client=client,
        scene_evidence_collector=collector,
    )
    assert paused.status == "needs_review"
    checkpoint_path = request.output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["scene_evidence_configuration_sha256"] = None
    checkpoint["scene_evidence"] = None
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationWorkflowError,
        match="missing request-required Scene evidence",
    ):
        build_articulation_review_receipt(
            request.output_dir,
            {candidate_id: "accept"},
            reviewer="workflow-test",
        )

    assert not (request.output_dir / "review_receipt.json").exists()


@pytest.mark.parametrize(
    "drift",
    ("mass_value", "mass_time_sample", "mass_removed", "added_physics_attribute"),
)
def test_local_joint_adapter_rejects_saved_physics_value_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """Reject a saved package whose pre-existing physics values were changed."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    repository_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repository_root / "apps" / "joint_agent"))
    for module_name in tuple(sys.modules):
        if module_name == "joint_agent" or module_name.startswith("joint_agent."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()

    source_path = tmp_path / "massed-aggregate.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    for prim_path in ("/World/base", "/World/base_trim", "/World/door"):
        UsdGeom.Xform.Define(stage, prim_path)
        UsdGeom.Cube.Define(stage, f"{prim_path}/visual")
    UsdGeom.Xform(stage.GetPrimAtPath("/World/door")).AddTranslateOp().Set(
        Gf.Vec3d(1.0, 0.0, 0.0)
    )
    door_prim = stage.GetPrimAtPath("/World/door")
    assert UsdPhysics.RigidBodyAPI.Apply(door_prim)
    assert UsdPhysics.MassAPI.Apply(door_prim).CreateMassAttr(12.0)
    assert stage.GetRootLayer().Save()
    del stage, door_prim

    candidate_document = _document(
        _candidate(
            "candidate_0001",
            1,
            fixed_parent_prim="/World/base",
            moving_part_prim="/World/door",
        )
    )
    candidate_path = tmp_path / "approved.json"
    candidate_path.write_text(
        candidate_document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    prediction_rows = tuple(
        {
            "id": prim_path,
            "classification": {
                "role": "body",
                "is_articulation_candidate": False,
                "joint_type_hint": "none",
                "instance_id": "cabinet_fixed_body",
                "provenance": {
                    "field_sources": {"instance_id": "predicted"},
                },
            },
        }
        for prim_path in ("/World/base", "/World/base_trim")
    )
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(source_path)
    authoring_request = ArticulationAuthoringRequest(
        source_asset=str(source_path),
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        accepted_candidate_ids=candidate_document.candidate_ids,
        idempotency_key="4" * 64,
        predictions_path=str(predictions_path),
        predictions_sha256=hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        output_dir=tmp_path / "run",
    )
    client = JointAgentLocalClient({"project": {}, "input": {}, "steps": {}})

    authoring = client.author(authoring_request)
    accepted = client.validate(
        authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert accepted.status == "pass", accepted.failures

    output_path = Path(authoring.output_asset_path)
    result_path = Path(authoring.joint_rigger_result_path or "")
    original_result_bytes = result_path.read_bytes()

    extracted = tmp_path / "mass-drift-package"
    with zipfile.ZipFile(output_path) as archive:
        member_names = tuple(
            item.filename for item in archive.infolist() if not item.is_dir()
        )
        archive.extractall(extracted)
    root_member = extracted / member_names[0]
    drifted_stage = Usd.Stage.Open(str(root_member))
    assert drifted_stage is not None
    drifted_door = drifted_stage.GetPrimAtPath("/World/door")
    assert drifted_door and drifted_door.HasAPI(UsdPhysics.MassAPI)
    mass_attribute = UsdPhysics.MassAPI(drifted_door).GetMassAttr()
    assert mass_attribute.Get() == 12.0
    if drift == "mass_value":
        assert mass_attribute.Set(99.0)
    elif drift == "mass_time_sample":
        assert mass_attribute.Set(99.0, 0.0)
    elif drift == "mass_removed":
        assert drifted_door.RemoveProperty(mass_attribute.GetName())
    else:
        assert drifted_door.CreateAttribute(
            "physics:rigidBodyEnabled",
            Sdf.ValueTypeNames.Bool,
        ).Set(False)
    assert drifted_stage.GetRootLayer().Save()
    del mass_attribute, drifted_door, drifted_stage

    drifted_output = tmp_path / "mass-drift.usdz"
    writer = Usd.ZipFileWriter.CreateNew(str(drifted_output))
    assert writer
    for member_name in member_names:
        assert writer.AddFile(str(extracted / member_name), member_name)
    assert writer.Save()
    output_path.unlink()
    drifted_output.replace(output_path)

    result_payload = json.loads(original_result_bytes)
    root_sha256, dependency_sha256 = _source_identity(output_path)
    result_payload["output_artifact"]["root_sha256"] = root_sha256
    result_payload["output_artifact"]["dependency_bundle_sha256"] = dependency_sha256
    result_path.chmod(0o600)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    drifted_authoring = authoring.model_copy(
        update={
            "output_asset_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "joint_rigger_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
        }
    )

    rejected = client.validate(
        drifted_authoring,
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert rejected.status == "fail", rejected.failures
    assert rejected.exact_graph_match is False
    assert any(
        "physics property state differs from the source-bound expectation" in failure
        for failure in rejected.failures
    ), rejected.failures

    # Rewriting the recorded expectation to match the drift must not launder it:
    # the bound inventory digest covers the authored physics property state.
    laundered_metadata = copy.deepcopy(dict(drifted_authoring.metadata))
    laundered_inventory = laundered_metadata["physics_api_inventory"]
    laundered_inventory["physics_property_state"] = {
        prim_path: entries
        for prim_path, entries in laundered_inventory["physics_property_state"].items()
        if prim_path != "/World/door"
    }
    laundered = client.validate(
        drifted_authoring.model_copy(update={"metadata": laundered_metadata}),
        expected_candidate_ids=candidate_document.candidate_ids,
    )

    assert laundered.status == "fail", laundered.failures
    assert any(
        "Bound physics API inventory digest differs from its payload" in failure
        for failure in laundered.failures
    ), laundered.failures
