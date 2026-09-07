# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from content_agent_workflows.articulation.finalizer import (
    verify_articulation_workflow_summary,
    write_articulation_workflow_summary,
)
from content_agent_workflows.articulation.models import (
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationInferenceResult,
    ArticulationReviewEntry,
    ArticulationReviewReceipt,
    ArticulationRunState,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    Stage2CandidateDocument,
)
from content_agent_workflows.articulation.scene_evidence import (
    ArticulationSceneEvidenceResult,
    MockArticulationSceneEvidenceCollector,
)

_STATE_PATH_FIELDS = {
    "request": "request_path",
    "inference_result": "inference_result_path",
    "candidate_document": "candidate_document_path",
    "scene_evidence": "scene_evidence_path",
    "review_receipt": "review_receipt_path",
    "approved_candidate_document": "approved_candidate_document_path",
    "authoring_request": "authoring_request_path",
    "authoring_result": "authoring_result_path",
    "validation_result": "validation_result_path",
}

_AUTHORING_PATH_DIGEST_FIELDS = {
    "output_asset_path": "output_asset_sha256",
    "diagnostics_path": "diagnostics_sha256",
    "joint_rigger_result_path": "joint_rigger_result_sha256",
}


def _write_binding(path: Path) -> ArtifactBinding:
    path.write_text('{"status": "bound"}\n', encoding="utf-8")
    return ArtifactBinding(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _write_model_binding(path: Path, model: BaseModel) -> ArtifactBinding:
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return ArtifactBinding(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _source_identity(path: Path) -> tuple[str, str]:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    identity = identify_usd_artifact(path, uri=path.resolve().as_uri())
    assert identity.dependency_bundle_sha256 is not None
    return identity.root_sha256, identity.dependency_bundle_sha256


def _completed_state_with_required_evidence(
    tmp_path: Path,
    *,
    request_requires_scene_evidence: bool = True,
) -> tuple[ArticulationRunState, ArticulationAuthoringResult]:
    artifact_dir = tmp_path / "completed-bindings"
    artifact_dir.mkdir()
    candidate_id = "candidate_0001"
    source_path = tmp_path / "source.usda"
    source_path.write_text(
        """#usda 1.0

def Xform "World" {
    def Xform "Cabinet" {
        def Cube "Drawer_01" {}
    }
}
""",
        encoding="utf-8",
    )
    request = ArticulationWorkflowRequest(
        source_asset=str(source_path),
        output_dir=tmp_path / "summary",
        review_policy="all",
        metadata=(
            {"content_agent_workflows.scene_evidence_required": True}
            if request_requires_scene_evidence
            else {}
        ),
    )
    request_binding = _write_model_binding(artifact_dir / "request.json", request)
    candidate_document = Stage2CandidateDocument.model_validate(
        {
            "schema_version": "joint-agent-stage2-v0",
            "summary": {
                "candidate_count": 1,
                "ready_candidate_count": 1,
                "review_required_candidate_count": 0,
            },
            "candidates": [
                {
                    "schema_version": "joint-agent-stage2-v0",
                    "candidate_id": candidate_id,
                    "motion_type": "prismatic",
                    "joint_type_hint": "prismatic",
                    "moving_part_prims": ["/World/Cabinet/Drawer_01"],
                    "fixed_parent_prim": "/World/Cabinet",
                    "parent_resolution_source": "stage1_hint",
                    "axis_hint": "x",
                    "motion_axis_world": [1.0, 0.0, 0.0],
                    "confidence": "high",
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
                            "prim_paths": ["/World/Cabinet/Drawer_01"],
                        }
                    ],
                    "connectivity_evidence": [
                        {
                            "source": "stage1_hint",
                            "description": "source-bound parent-child edge",
                            "value": "/World/Cabinet",
                            "prim_paths": [
                                "/World/Cabinet",
                                "/World/Cabinet/Drawer_01",
                            ],
                            "connectivity_role": "body0_body1_edge",
                        }
                    ],
                    "review_status": "ready_for_rigger_input",
                }
            ],
        }
    )
    inference = ArticulationInferenceResult(
        candidate_document=candidate_document,
        backend_configuration_sha256="4" * 64,
    )
    inference_binding = _write_model_binding(
        artifact_dir / "inference.json",
        inference,
    )
    candidate_binding = _write_model_binding(
        artifact_dir / "candidates.json",
        candidate_document,
    )
    source_sha256, source_dependency_bundle_sha256 = _source_identity(source_path)
    scene_collector = MockArticulationSceneEvidenceCollector()
    scene_evidence_root = request.output_dir / "scene_evidence"
    scene_evidence = scene_collector.collect(
        request,
        request_sha256=request_binding.sha256,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document=candidate_document,
        candidate_document_sha256=candidate_binding.sha256,
        output_dir=scene_evidence_root,
    )
    scene_binding = _write_model_binding(
        scene_evidence_root / "manifest.json",
        scene_evidence,
    )
    scene_configuration_sha256 = scene_collector.configuration_sha256(request)
    receipt = ArticulationReviewReceipt(
        request_sha256=request_binding.sha256,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_sha256=candidate_binding.sha256,
        scene_evidence_sha256=scene_binding.sha256,
        reviewer="finalizer-test",
        decisions=(
            ArticulationReviewEntry(
                candidate_id=candidate_id,
                decision="accept",
            ),
        ),
    )
    receipt_binding = _write_model_binding(
        artifact_dir / "review-receipt.json",
        receipt,
    )
    approved_binding = _write_model_binding(
        artifact_dir / "approved-candidates.json",
        candidate_document,
    )
    authoring_request = ArticulationAuthoringRequest(
        source_asset=request.source_asset,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        candidate_document_path=approved_binding.path,
        candidate_document_sha256=approved_binding.sha256,
        accepted_candidate_ids=(candidate_id,),
        idempotency_key="6" * 64,
        output_dir=request.output_dir,
    )
    authoring_request_binding = _write_model_binding(
        artifact_dir / "authoring-request.json",
        authoring_request,
    )
    output_path = artifact_dir / "rigged.usdz"
    output_path.write_bytes(b"bound rigged output")
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key=authoring_request.idempotency_key,
        source_sha256=authoring_request.source_sha256,
        source_dependency_bundle_sha256=(
            authoring_request.source_dependency_bundle_sha256
        ),
        candidate_document_path=authoring_request.candidate_document_path,
        candidate_document_sha256=authoring_request.candidate_document_sha256,
        output_asset_path=str(output_path),
        output_asset_sha256=output_sha256,
        authored_candidate_ids=(candidate_id,),
        authored_joint_count=1,
    )
    authoring_binding = _write_model_binding(
        artifact_dir / "authoring-result.json",
        authoring,
    )
    validation = ArticulationValidationResult(
        status="pass",
        output_asset_path=str(output_path),
        expected_output_asset_sha256=output_sha256,
        observed_output_asset_sha256=output_sha256,
        expected_candidate_ids=(candidate_id,),
        validated_candidate_ids=(candidate_id,),
        exact_graph_match=True,
        self_contained=True,
    )
    validation_binding = _write_model_binding(
        artifact_dir / "validation.json",
        validation,
    )
    state = ArticulationRunState(
        mode="interactive",
        phase="completed",
        request=request_binding,
        source_asset=request.source_asset,
        source_sha256=source_sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        backend_configuration_sha256="4" * 64,
        scene_evidence_configuration_sha256=scene_configuration_sha256,
        inference_result=inference_binding,
        candidate_document=candidate_binding,
        scene_evidence=scene_binding,
        review_receipt=receipt_binding,
        approved_candidate_document=approved_binding,
        authoring_request=authoring_request_binding,
        authoring_result=authoring_binding,
        validation_result=validation_binding,
        candidate_ids=(candidate_id,),
        review_required_candidate_ids=(candidate_id,),
        accepted_candidate_ids=(candidate_id,),
    )
    return state, authoring


@pytest.mark.parametrize("entrypoint", ("write", "verify"))
@pytest.mark.parametrize("nested_artifact", ("scene_snapshot", "render_image"))
@pytest.mark.parametrize(
    "request_requires_scene_evidence",
    (True, False),
    ids=("request-required", "legacy-bound"),
)
def test_completed_summary_rejects_tampered_nested_scene_artifact(
    tmp_path: Path,
    entrypoint: str,
    nested_artifact: str,
    request_requires_scene_evidence: bool,
) -> None:
    state, authoring = _completed_state_with_required_evidence(
        tmp_path,
        request_requires_scene_evidence=request_requires_scene_evidence,
    )
    assert state.scene_evidence is not None
    manifest = ArticulationSceneEvidenceResult.model_validate_json(
        Path(state.scene_evidence.path).read_bytes()
    )
    if entrypoint == "verify":
        baseline = write_articulation_workflow_summary(
            state,
            output_dir=tmp_path / "summary",
            authoring=authoring,
        )
        assert baseline.success is True
    if nested_artifact == "scene_snapshot":
        artifact_path = Path(manifest.scene_snapshot_artifact.path)
    else:
        artifact_path = Path(manifest.candidates[0].renders[0].image_artifact.path)
    artifact_path.write_bytes(b"tampered nested scene artifact")

    with pytest.raises(
        ValueError,
        match="Invalid bound Scene articulation evidence.*digest mismatch",
    ):
        if entrypoint == "write":
            write_articulation_workflow_summary(
                state,
                output_dir=tmp_path / "summary",
                authoring=authoring,
            )
        else:
            verify_articulation_workflow_summary(
                state,
                output_dir=tmp_path / "summary",
                authoring=authoring,
            )
    if entrypoint == "write":
        assert not (tmp_path / "summary" / "final_summary.json").exists()


@pytest.mark.parametrize("omitted_field", ("configuration", "manifest"))
def test_completed_summary_rejects_incomplete_legacy_scene_binding(
    tmp_path: Path,
    omitted_field: str,
) -> None:
    state, authoring = _completed_state_with_required_evidence(
        tmp_path,
        request_requires_scene_evidence=False,
    )
    if omitted_field == "configuration":
        state = state.model_copy(update={"scene_evidence_configuration_sha256": None})
    else:
        state = state.model_copy(update={"scene_evidence": None})

    with pytest.raises(ValueError, match="Scene evidence"):
        write_articulation_workflow_summary(
            state,
            output_dir=tmp_path / "summary",
            authoring=authoring,
        )
    assert not (tmp_path / "summary" / "final_summary.json").exists()


def _cancelled_state(tmp_path: Path) -> tuple[ArticulationRunState, dict[str, Path]]:
    artifact_dir = tmp_path / "bindings"
    artifact_dir.mkdir()
    paths = {name: artifact_dir / f"{name}.json" for name in _STATE_PATH_FIELDS}
    bindings = {name: _write_binding(path) for name, path in paths.items()}
    return (
        ArticulationRunState(
            mode="interactive",
            phase="cancelled",
            request=bindings["request"],
            source_asset=str(tmp_path / "source.usda"),
            source_sha256="1" * 64,
            source_dependency_bundle_sha256="2" * 64,
            backend_configuration_sha256="3" * 64,
            inference_result=bindings["inference_result"],
            candidate_document=bindings["candidate_document"],
            scene_evidence=bindings["scene_evidence"],
            review_receipt=bindings["review_receipt"],
            approved_candidate_document=bindings["approved_candidate_document"],
            authoring_request=bindings["authoring_request"],
            authoring_result=bindings["authoring_result"],
            validation_result=bindings["validation_result"],
        ),
        paths,
    )


def _semantic_motion_needs_review_state(
    tmp_path: Path,
) -> tuple[ArticulationRunState, Path]:
    artifact_dir = tmp_path / "semantic-motion-bindings"
    artifact_dir.mkdir()
    request_binding = _write_binding(artifact_dir / "request.json")
    candidate_document = Stage2CandidateDocument.model_validate(
        {
            "summary": {
                "candidate_count": 2,
                "ready_candidate_count": 0,
                "review_required_candidate_count": 2,
            },
            "candidates": [
                {
                    "candidate_id": "candidate_0001",
                    "motion_type": "unknown",
                    "joint_type_hint": "unknown",
                    "moving_part_prims": ["/World/tool/linkage"],
                    "semantic_role": "folding_linkage",
                    "motion_capability": {
                        "kind": "unsupported",
                        "source": "predicted",
                        "evidence": "Several coupled pivots are visible.",
                        "missing_contract": "compound_multi_axis_motion",
                    },
                    "review_status": "review_required",
                    "unresolved_reason_codes": ["role_deferred_0_5"],
                    "unresolved_questions": [
                        "Provide the missing compound motion contract."
                    ],
                },
                {
                    "candidate_id": "candidate_0002",
                    "motion_type": "unknown",
                    "joint_type_hint": "unknown",
                    "moving_part_prims": ["/World/appliance/mechanism"],
                    "semantic_role": "rotary_distributor",
                    "motion_capability": {
                        "kind": "unresolved",
                        "source": "predicted",
                        "evidence": "Rotation is visually plausible but not proven.",
                        "missing_evidence": ["passivity"],
                    },
                    "review_status": "review_required",
                    "unresolved_reason_codes": ["role_deferred_0_5"],
                    "unresolved_questions": [
                        "Provide explicit evidence that the mechanism is passive."
                    ],
                },
            ],
        }
    )
    candidate_binding = _write_model_binding(
        artifact_dir / "candidates.json",
        candidate_document,
    )
    inference = ArticulationInferenceResult(
        candidate_document=candidate_document,
        backend_configuration_sha256="3" * 64,
    )
    inference_path = artifact_dir / "inference.json"
    inference_binding = _write_model_binding(inference_path, inference)
    candidate_ids = candidate_document.candidate_ids
    return (
        ArticulationRunState(
            mode="interactive",
            phase="needs_review",
            request=request_binding,
            source_asset=str(tmp_path / "source.usda"),
            source_sha256="1" * 64,
            source_dependency_bundle_sha256="2" * 64,
            backend_configuration_sha256=inference.backend_configuration_sha256,
            inference_result=inference_binding,
            candidate_document=candidate_binding,
            candidate_ids=candidate_ids,
            review_required_candidate_ids=candidate_ids,
        ),
        inference_path,
    )


def test_needs_review_summary_surfaces_typed_motion_capability_decisions(
    tmp_path: Path,
) -> None:
    state, _ = _semantic_motion_needs_review_state(tmp_path)
    output_dir = tmp_path / "summary"

    result = write_articulation_workflow_summary(state, output_dir=output_dir)

    assert tuple(
        decision.candidate_id for decision in result.motion_capability_decisions
    ) == ("candidate_0001", "candidate_0002")
    assert tuple(
        decision.motion_capability.kind
        for decision in result.motion_capability_decisions
    ) == ("unsupported", "unresolved")
    persisted = json.loads((output_dir / "final_summary.json").read_text())
    assert persisted["motion_capability_decisions"][0]["unresolved_reason_codes"] == [
        "motion_capability_unsupported",
        "role_deferred_0_5",
    ]
    assert (
        persisted["motion_capability_decisions"][0]["motion_capability"][
            "missing_contract"
        ]
        == "compound_multi_axis_motion"
    )
    assert persisted["motion_capability_decisions"][1]["motion_capability"][
        "missing_evidence"
    ] == ["passivity"]
    assert persisted["motion_capability_decisions"][1]["unresolved_reason_codes"] == [
        "motion_capability_unresolved",
        "role_deferred_0_5",
    ]


def test_needs_review_summary_rejects_missing_semantic_motion_evidence(
    tmp_path: Path,
) -> None:
    state, inference_path = _semantic_motion_needs_review_state(tmp_path)
    inference_path.write_text('{"status": "tampered"}\n', encoding="utf-8")
    output_dir = tmp_path / "summary"

    with pytest.raises(ValueError, match="valid bound inference result"):
        write_articulation_workflow_summary(state, output_dir=output_dir)

    assert not (output_dir / "final_summary.json").exists()


def test_provider_neutral_summary_allows_optional_proposal_without_inference(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "embedded-bindings"
    artifact_dir.mkdir()
    request_binding = _write_binding(artifact_dir / "request.json")
    evidence_binding = _write_binding(artifact_dir / "evidence.json")
    proposal_binding = _write_binding(artifact_dir / "proposal.json")
    state = ArticulationRunState(
        schema_version="content-agent-workflows.articulation-run-state.v3",
        mode="batch",
        phase="needs_review",
        request=request_binding,
        source_asset=str(tmp_path / "source.usda"),
        source_sha256="1" * 64,
        source_dependency_bundle_sha256="2" * 64,
        backend_configuration_sha256="3" * 64,
        embedded_evidence=evidence_binding,
        embedded_proposal=proposal_binding,
        candidate_ids=("motion_001",),
        review_required_candidate_ids=("motion_001",),
    )

    result = write_articulation_workflow_summary(
        state,
        output_dir=tmp_path / "summary",
    )

    assert result.status == "needs_review"
    assert result.inference_result_path is None
    assert result.embedded_evidence_path == evidence_binding.path
    assert result.embedded_proposal_path == proposal_binding.path


@pytest.mark.parametrize("binding_name", tuple(_STATE_PATH_FIELDS))
@pytest.mark.parametrize("damage", ("missing", "tampered"))
def test_summary_only_indexes_digest_verified_checkpoint_bindings(
    tmp_path: Path,
    binding_name: str,
    damage: str,
) -> None:
    state, paths = _cancelled_state(tmp_path)
    damaged_path = paths[binding_name]
    if damage == "missing":
        damaged_path.unlink()
    else:
        damaged_path.write_text('{"status": "tampered"}\n', encoding="utf-8")

    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    result = write_articulation_workflow_summary(
        state,
        output_dir=output_dir,
    )

    for state_field, result_field in _STATE_PATH_FIELDS.items():
        expected_path = None if state_field == binding_name else str(paths[state_field])
        assert getattr(result, result_field) == expected_path


@pytest.mark.parametrize("path_field", tuple(_AUTHORING_PATH_DIGEST_FIELDS))
@pytest.mark.parametrize("damage", ("missing", "tampered"))
def test_summary_only_indexes_digest_verified_authoring_artifacts(
    tmp_path: Path,
    path_field: str,
    damage: str,
) -> None:
    state, state_paths = _cancelled_state(tmp_path)
    artifact_paths = {
        "output_asset_path": tmp_path / "rigged.usdz",
        "diagnostics_path": tmp_path / "joint-rigger-diagnostics.json",
        "joint_rigger_result_path": tmp_path / "joint-rigger-result.json",
    }
    for path in artifact_paths.values():
        path.write_bytes(f"bound:{path.name}".encode())
    candidate_path = tmp_path / "approved-candidates.json"
    candidate_path.write_text('{"candidates": ["candidate_0001"]}\n', encoding="utf-8")
    authoring = ArticulationAuthoringResult(
        status="succeeded",
        idempotency_key="4" * 64,
        source_sha256="5" * 64,
        source_dependency_bundle_sha256="6" * 64,
        candidate_document_path=str(candidate_path),
        candidate_document_sha256=hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest(),
        output_asset_path=str(artifact_paths["output_asset_path"]),
        output_asset_sha256=hashlib.sha256(
            artifact_paths["output_asset_path"].read_bytes()
        ).hexdigest(),
        authored_candidate_ids=("candidate_0001",),
        authored_joint_count=1,
        diagnostics_path=str(artifact_paths["diagnostics_path"]),
        diagnostics_sha256=hashlib.sha256(
            artifact_paths["diagnostics_path"].read_bytes()
        ).hexdigest(),
        joint_rigger_result_path=str(artifact_paths["joint_rigger_result_path"]),
        joint_rigger_result_sha256=hashlib.sha256(
            artifact_paths["joint_rigger_result_path"].read_bytes()
        ).hexdigest(),
    )
    damaged_path = artifact_paths[path_field]
    if damage == "missing":
        damaged_path.unlink()
    else:
        damaged_path.write_bytes(b"tampered")

    output_dir = tmp_path / "summary"
    output_dir.mkdir()
    result = write_articulation_workflow_summary(
        state,
        output_dir=output_dir,
        authoring=authoring,
    )

    for sibling_field in _AUTHORING_PATH_DIGEST_FIELDS:
        expected_path = (
            None if sibling_field == path_field else str(artifact_paths[sibling_field])
        )
        assert getattr(result, sibling_field) == expected_path
    assert result.request_path == str(state_paths["request"])


def test_completed_summary_rejects_tampered_checkpoint_binding(
    tmp_path: Path,
) -> None:
    state, paths = _cancelled_state(tmp_path)
    completed = state.model_copy(update={"phase": "completed"})
    paths["candidate_document"].write_text(
        '{"status": "tampered"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "summary"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="candidate document digest mismatch"):
        write_articulation_workflow_summary(
            completed,
            output_dir=output_dir,
        )


def test_completed_summary_rejects_tampered_authoring_artifact(
    tmp_path: Path,
) -> None:
    state, paths = _cancelled_state(tmp_path)
    completed = state.model_copy(
        update={
            "phase": "completed",
            "candidate_ids": ("candidate_0001",),
            "accepted_candidate_ids": ("candidate_0001",),
        }
    )
    candidate_path = paths["approved_candidate_document"]
    output_path = tmp_path / "rigged.usdz"
    output_path.write_bytes(b"bound rigged output")
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
        output_asset_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        authored_candidate_ids=("candidate_0001",),
        authored_joint_count=1,
    )
    authoring_request = ArticulationAuthoringRequest(
        source_asset=str(tmp_path / "source.usda"),
        source_sha256=authoring.source_sha256,
        source_dependency_bundle_sha256=(authoring.source_dependency_bundle_sha256),
        candidate_document_path=authoring.candidate_document_path,
        candidate_document_sha256=authoring.candidate_document_sha256,
        accepted_candidate_ids=authoring.authored_candidate_ids,
        idempotency_key=authoring.idempotency_key,
        output_dir=tmp_path,
    )
    paths["authoring_request"].write_text(
        authoring_request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    paths["authoring_result"].write_text(
        authoring.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    completed = completed.model_copy(
        update={
            "authoring_request": ArtifactBinding(
                path=str(paths["authoring_request"]),
                sha256=hashlib.sha256(
                    paths["authoring_request"].read_bytes()
                ).hexdigest(),
            ),
            "authoring_result": ArtifactBinding(
                path=str(paths["authoring_result"]),
                sha256=hashlib.sha256(
                    paths["authoring_result"].read_bytes()
                ).hexdigest(),
            ),
        }
    )
    output_path.write_bytes(b"tampered rigged output")
    output_dir = tmp_path / "summary"
    output_dir.mkdir()

    with pytest.raises(
        ValueError, match="Published articulation output digest mismatch"
    ):
        write_articulation_workflow_summary(
            completed,
            output_dir=output_dir,
            authoring=authoring,
        )


@pytest.mark.parametrize("entrypoint", ("write", "verify"))
@pytest.mark.parametrize(
    ("omission", "expected_error"),
    (
        ("scene", "missing request-required Scene evidence"),
        ("review", "review-required scope differs"),
    ),
)
def test_completed_summary_rederives_requirements_from_bound_request(
    tmp_path: Path,
    entrypoint: str,
    omission: str,
    expected_error: str,
) -> None:
    state, authoring = _completed_state_with_required_evidence(tmp_path)
    output_dir = tmp_path / "summary"
    baseline = write_articulation_workflow_summary(
        state,
        output_dir=output_dir,
        authoring=authoring,
    )
    assert baseline.status == "completed"
    if omission == "scene":
        mutated = state.model_copy(
            update={
                "scene_evidence_configuration_sha256": None,
                "scene_evidence": None,
            }
        )
    else:
        mutated = state.model_copy(
            update={
                "review_required_candidate_ids": (),
                "review_receipt": None,
            }
        )

    with pytest.raises(ValueError, match=expected_error):
        if entrypoint == "write":
            write_articulation_workflow_summary(
                mutated,
                output_dir=output_dir,
                authoring=authoring,
            )
        else:
            verify_articulation_workflow_summary(
                mutated,
                output_dir=output_dir,
                authoring=authoring,
            )


def test_completed_summary_hashes_and_parses_the_same_request_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import finalizer

    state, authoring = _completed_state_with_required_evidence(tmp_path)
    assert state.request is not None
    request_path = Path(state.request.path)
    original_request = ArticulationWorkflowRequest.model_validate_json(
        request_path.read_text(encoding="utf-8")
    )
    downgraded_request = original_request.model_copy(
        update={
            "review_policy": "none",
            "metadata": {},
        }
    )
    downgraded_path = request_path.with_name("downgraded-request.json")
    downgraded_path.write_text(
        downgraded_request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    def swap_after_hash(path: str | Path) -> str:
        artifact_path = Path(path)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if artifact_path == request_path:
            downgraded_path.replace(request_path)
        return digest

    # This hook reproduces the former hash-then-reopen implementation. The
    # current implementation never calls it because it hashes and parses one
    # pinned byte read.
    monkeypatch.setattr(
        finalizer,
        "file_sha256",
        swap_after_hash,
        raising=False,
    )
    weakened_state = state.model_copy(
        update={
            "scene_evidence_configuration_sha256": None,
            "scene_evidence": None,
            "review_required_candidate_ids": (),
            "review_receipt": None,
        }
    )

    with pytest.raises(
        ValueError,
        match="missing request-required Scene evidence",
    ):
        write_articulation_workflow_summary(
            weakened_state,
            output_dir=tmp_path / "summary",
            authoring=authoring,
        )
    assert (
        request_path.read_bytes()
        == (original_request.model_dump_json(indent=2) + "\n").encode()
    )
