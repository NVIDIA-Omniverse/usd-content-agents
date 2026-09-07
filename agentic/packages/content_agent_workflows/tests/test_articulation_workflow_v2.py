# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from world_understanding.functions.physics.joint_rigger import (
    identify_usd_artifact,
)

from content_agent_workflows.articulation.models import ArticulationWorkflowRequest
from content_agent_workflows.articulation.models_v2 import (
    ArticulationV2ArtifactIdentity,
    ArticulationV2AttachmentFrame,
    ArticulationV2DistanceContract,
    ArticulationV2DistanceReadback,
    ArticulationV2DistanceSelector,
    ArticulationV2FixedContract,
    ArticulationV2FixedReadback,
    ArticulationV2FixedSelector,
    ArticulationV2OutputTarget,
    ArticulationV2PublicationResult,
    ArticulationV2ReadbackResult,
    ArticulationV2ReleaseSelection,
    ArticulationV2ReviewContext,
    ArticulationV2ReviewReceipt,
    ArticulationV2StaticAttestation,
    ArticulationV2UsdArtifactIdentity,
    ArticulationV2WorkflowRequest,
    articulation_v2_canonical_sha256,
)
from content_agent_workflows.articulation.workflow_v2 import (
    ArticulationV2ReleaseSelectionError,
    ArticulationV2WorkflowError,
    build_articulation_v2_review_receipt,
    record_articulation_v2_publication,
    record_articulation_v2_readback,
    run_articulation_v2_workflow,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _opaque_identity(path: Path, *, uri: str) -> ArticulationV2ArtifactIdentity:
    payload = path.read_bytes()
    return ArticulationV2ArtifactIdentity(
        uri=uri,
        path=path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
    )


def _usd_identity(path: Path, *, uri: str) -> ArticulationV2UsdArtifactIdentity:
    payload = path.read_bytes()
    identity = identify_usd_artifact(path, uri=uri)
    assert identity.dependency_bundle_sha256 is not None
    return ArticulationV2UsdArtifactIdentity(
        uri=uri,
        path=path,
        sha256=identity.root_sha256,
        size_bytes=len(payload),
        dependency_bundle_sha256=identity.dependency_bundle_sha256,
    )


def _attachment(
    *,
    position: tuple[float, float, float],
) -> ArticulationV2AttachmentFrame:
    return ArticulationV2AttachmentFrame(
        position_meters=position,
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


class _ExactContractTestReleaseRouter:
    """Unit-only #870 boundary double; it is not qualification evidence."""

    def __init__(self, selection: ArticulationV2ReleaseSelection) -> None:
        self.selection = selection
        self.calls = 0

    def resolve(
        self,
        request: ArticulationV2WorkflowRequest,
    ) -> ArticulationV2ReleaseSelection:
        del request
        self.calls += 1
        return self.selection


def _release_selection(
    reference: ArticulationV2ArtifactIdentity,
    *,
    kind: str,
) -> ArticulationV2ReleaseSelection:
    capability_id = (
        "fixed.explicit_two_body_constraint"
        if kind == "fixed"
        else "distance.bounded_two_body_constraint"
    )
    property_profile = (
        "explicit_two_body_constraint" if kind == "fixed" else "source_backed_bounds"
    )
    row_admission_sha256 = "1" * 64
    return ArticulationV2ReleaseSelection(
        manifest_version="0.6.0",
        manifest_sha256="2" * 64,
        capability_id=capability_id,
        selector_kind=kind,
        property_profile=property_profile,
        row_admission_sha256=row_admission_sha256,
        static_attestation=ArticulationV2StaticAttestation(
            scorecard_sha256="3" * 64,
            run_plan_sha256="4" * 64,
            capability_manifest_sha256="5" * 64,
            run_id="contract-only-test-run",
            row_admission_sha256=row_admission_sha256,
        ),
        reference_asset_id="joint_ref_selector_scoped_test_only",
        reference_artifact_key=f"{kind}_reference_asset",
        reference=reference,
    )


def _request(
    tmp_path: Path,
    *,
    kind: str = "fixed",
    output_dir: Path | None = None,
) -> ArticulationV2WorkflowRequest:
    source_path = tmp_path / "source.usda"
    reference_path = tmp_path / "reference.usda"
    source_dependency = tmp_path / "source_dependency.usda"
    reference_dependency = tmp_path / "reference_dependency.usda"
    source_dependency.write_bytes(b'#usda 1.0\ndef Xform "SourceDependency" {}\n')
    reference_dependency.write_bytes(b'#usda 1.0\ndef Xform "ReferenceDependency" {}\n')
    source_path.write_bytes(b"#usda 1.0\n( subLayers = [@source_dependency.usda@] )\n")
    reference_path.write_bytes(
        b"#usda 1.0\n( subLayers = [@reference_dependency.usda@] )\n"
    )
    source = _usd_identity(source_path, uri="asset://workflow/source")
    reference = _usd_identity(
        reference_path,
        uri=f"asset://workflow/{kind}-reference",
    )
    body0_attachment = _attachment(position=(0.0, 0.0, 0.05))
    body1_attachment = _attachment(position=(0.0, 0.0, -0.2))
    if kind == "fixed":
        selector = ArticulationV2FixedSelector(
            contract=ArticulationV2FixedContract(
                source_joint_prim="/World/SourceJoints/fixed",
                body0="/World/Fixed/Base",
                body1="/World/Fixed/Tool",
                body0_attachment=body0_attachment,
                body1_attachment=body1_attachment,
            )
        )
    else:
        selector = ArticulationV2DistanceSelector(
            contract=ArticulationV2DistanceContract(
                source_joint_prim="/World/SourceJoints/distance",
                body0="/World/Distance/Anchor",
                body1="/World/Distance/Payload",
                body0_attachment=body0_attachment,
                body1_attachment=body1_attachment,
                minimum_distance_meters=0.05,
                maximum_distance_meters=0.75,
            )
        )
    intent = f"Review and publish the selected {kind} constraint."
    return ArticulationV2WorkflowRequest(
        mode="interactive",
        intent=intent,
        intent_sha256=_sha256_bytes(intent.encode("utf-8")),
        selector=selector,
        release_selection=_release_selection(reference, kind=kind),
        source=source,
        reference=reference,
        output=ArticulationV2OutputTarget(
            uri=f"asset://workflow/{kind}-output",
            path=tmp_path / f"{kind}-output.usda",
            format="raw_usd",
        ),
        output_dir=output_dir or tmp_path / "workflow-output",
    )


def _review_context(
    request: ArticulationV2WorkflowRequest,
) -> ArticulationV2ReviewContext:
    evidence_path = request.output_dir.parent / "scene-review.json"
    evidence_path.write_bytes(b'{"views":["front","side"]}\n')
    return ArticulationV2ReviewContext(
        request_sha256=_file_sha256(request.output_dir / "request.v2.json"),
        release_selection_sha256=articulation_v2_canonical_sha256(
            request.release_selection
        ),
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source_sha256=request.source.sha256,
        reference_sha256=request.reference.sha256,
        scene_evidence=_opaque_identity(
            evidence_path,
            uri="asset://workflow/scene-review",
        ),
    )


def _advance_to_ready(
    request: ArticulationV2WorkflowRequest,
    router: _ExactContractTestReleaseRouter,
) -> Any:
    run_articulation_v2_workflow(request, release_router=router)
    run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=_review_context(request),
    )
    receipt = build_articulation_v2_review_receipt(
        request.output_dir,
        release_router=router,
        decision="accept",
        reviewer="contract-test-reviewer",
    )
    return run_articulation_v2_workflow(
        request,
        release_router=router,
        review_receipt=receipt,
    )


def _publication(
    request: ArticulationV2WorkflowRequest,
    *,
    idempotency_key: str,
) -> ArticulationV2PublicationResult:
    dependency_path = request.output.path.with_name(
        f"{request.output.path.stem}_dependency.usda"
    )
    dependency_path.write_bytes(b'#usda 1.0\ndef Xform "PublishedDependency" {}\n')
    request.output.path.write_text(
        f"#usda 1.0\n( subLayers = [@{dependency_path.name}@] )\n",
        encoding="utf-8",
    )
    return ArticulationV2PublicationResult(
        request_sha256=_file_sha256(request.output_dir / "request.v2.json"),
        release_selection_sha256=articulation_v2_canonical_sha256(
            request.release_selection
        ),
        review_receipt_sha256=_file_sha256(
            request.output_dir / "review_receipt.v2.json"
        ),
        authoring_intent_sha256=_file_sha256(
            request.output_dir / "authoring_intent.v2.json"
        ),
        idempotency_key=idempotency_key,
        facade_call_id="external-facade-call-contract-test",
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source=request.source,
        reference=request.reference,
        output_target=request.output,
        output_artifact=_usd_identity(request.output.path, uri=request.output.uri),
        authoring_contract_sha256=articulation_v2_canonical_sha256(
            request.selector.contract
        ),
    )


def _matching_constraint(
    request: ArticulationV2WorkflowRequest,
) -> ArticulationV2FixedReadback | ArticulationV2DistanceReadback:
    contract = request.selector.contract
    common = {
        "joint_prim": f"/World/Joints/{contract.kind}",
        "body0": contract.body0,
        "body1": contract.body1,
        "body0_attachment": contract.body0_attachment,
        "body1_attachment": contract.body1_attachment,
    }
    if contract.kind == "fixed":
        return ArticulationV2FixedReadback(**common)
    return ArticulationV2DistanceReadback(
        **common,
        minimum_distance_meters=contract.minimum_distance_meters,
        maximum_distance_meters=contract.maximum_distance_meters,
    )


def _readback(
    request: ArticulationV2WorkflowRequest,
    publication: ArticulationV2PublicationResult,
    *,
    status: Literal["pass", "fail"] = "pass",
    constraint: ArticulationV2FixedReadback
    | ArticulationV2DistanceReadback
    | None = None,
) -> ArticulationV2ReadbackResult:
    passed = status == "pass"
    evidence_path = request.output_dir.parent / f"saved-stage-{status}.json"
    evidence_path.write_bytes(
        b'{"exact_saved_stage_match":true}\n'
        if passed
        else b'{"exact_saved_stage_match":false}\n'
    )
    return ArticulationV2ReadbackResult(
        status=status,
        publication_result_sha256=_file_sha256(
            request.output_dir / "publication_result.v2.json"
        ),
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source=request.source,
        reference=request.reference,
        output_artifact=publication.output_artifact,
        evidence_artifact=_opaque_identity(
            evidence_path,
            uri=f"asset://workflow/saved-stage-{status}",
        ),
        constraint=constraint
        if constraint is not None
        else (_matching_constraint(request) if passed else None),
        exact_saved_stage_match=passed,
        failures=() if passed else ("saved_stage_constraint_mismatch",),
    )


def test_v2_models_are_discriminated_and_do_not_widen_v1() -> None:
    v1_schema = ArticulationWorkflowRequest.model_json_schema()
    assert v1_schema["properties"]["schema_version"]["const"] == (
        "content-agent-workflows.articulation-request.v1"
    )
    assert "release_selection" not in v1_schema["properties"]

    schema = ArticulationV2WorkflowRequest.model_json_schema()
    selector = schema["properties"]["selector"]
    assert selector["discriminator"]["propertyName"] == "kind"
    assert set(selector["discriminator"]["mapping"]) == {"distance", "fixed"}
    selection_schema = ArticulationV2ReleaseSelection.model_json_schema()
    assert selection_schema["properties"]["manifest_schema_version"]["const"] == (
        "joint-agent-capability-manifest-v3"
    )
    assert (
        articulation_v2_canonical_sha256({"b": 2, "a": 1})
        == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    )

    with pytest.raises(ValidationError, match="ordered interval"):
        ArticulationV2DistanceContract(
            source_joint_prim="/World/Joints/distance",
            body0="/World/A",
            body1="/World/B",
            body0_attachment=_attachment(position=(0.0, 0.0, 0.0)),
            body1_attachment=_attachment(position=(0.0, 0.0, 0.0)),
            minimum_distance_meters=1.0,
            maximum_distance_meters=0.5,
        )

    canonicalized = ArticulationV2AttachmentFrame(
        position_meters=(0.0, 0.0, 0.0),
        orientation_wxyz=(-1.0, 0.0, 0.0, 0.0),
    )
    assert canonicalized.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert "-0.0" not in canonicalized.model_dump_json()


def test_quaternion_canonical_digest_survives_json_round_trip() -> None:
    component = 2.0**-0.5
    frame = ArticulationV2AttachmentFrame(
        position_meters=(-0.0, 0.0, 0.0),
        orientation_wxyz=(-component, 0.0, component, 0.0),
    )

    reloaded = ArticulationV2AttachmentFrame.model_validate_json(
        frame.model_dump_json()
    )

    assert frame.orientation_wxyz == (
        component,
        0.0,
        -component,
        0.0,
    )
    assert "-0.0" not in frame.model_dump_json()
    assert articulation_v2_canonical_sha256(frame) == (
        articulation_v2_canonical_sha256(reloaded)
    )

    distance = ArticulationV2DistanceContract(
        source_joint_prim="/World/Joints/distance",
        body0="/World/A",
        body1="/World/B",
        body0_attachment=frame,
        body1_attachment=frame,
        minimum_distance_meters=-0.0,
        maximum_distance_meters=0.0,
    )
    distance_readback = ArticulationV2DistanceReadback(
        joint_prim="/World/Authored/distance",
        body0=distance.body0,
        body1=distance.body1,
        body0_attachment=frame,
        body1_attachment=frame,
        minimum_distance_meters=-0.0,
        maximum_distance_meters=0.0,
    )
    assert "-0.0" not in distance.model_dump_json()
    assert "-0.0" not in distance_readback.model_dump_json()


def test_v2_request_rejects_cross_substitution_and_qualification_roots(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    document = request.model_dump(mode="json")
    document["selector"]["capability_id"] = "distance.bounded_two_body_constraint"
    with pytest.raises(ValidationError):
        ArticulationV2WorkflowRequest.model_validate(document)

    document = request.model_dump(mode="json")
    document["qualification_evidence_root"] = "/tmp/issue-976"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArticulationV2WorkflowRequest.model_validate(document)

    document = request.model_dump(mode="json")
    document["release_selection"]["manifest_schema_version"] = (
        "joint-agent-capability-manifest-v2"
    )
    with pytest.raises(ValidationError, match="joint-agent-capability-manifest-v3"):
        ArticulationV2WorkflowRequest.model_validate(document)

    document = request.model_dump(mode="json")
    document["intent"] = "Changed prompt identity."
    with pytest.raises(ValidationError, match="intent_sha256"):
        ArticulationV2WorkflowRequest.model_validate(document)


def test_default_router_rejects_before_workflow_startup(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert not request.output_dir.exists()

    with pytest.raises(
        ArticulationV2ReleaseSelectionError,
        match="non-authoring|not release-selected|before Scene startup",
    ):
        run_articulation_v2_workflow(request)

    assert not request.output_dir.exists()
    assert not request.output.path.exists()


def test_release_router_must_echo_the_digest_bound_selection(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mismatched = request.release_selection.model_copy(
        update={"manifest_sha256": "9" * 64}
    )
    router = _ExactContractTestReleaseRouter(mismatched)

    with pytest.raises(
        ArticulationV2ReleaseSelectionError,
        match="differs from the digest-bound workflow request",
    ):
        run_articulation_v2_workflow(request, release_router=router)

    assert not request.output_dir.exists()


def test_release_router_rejects_type_coercive_selection_before_startup(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    invalid_reference = request.release_selection.reference.model_copy(
        update={"size_bytes": float(request.release_selection.reference.size_bytes)}
    )
    invalid_selection = request.release_selection.model_copy(
        update={"reference": invalid_reference}
    )
    router = _ExactContractTestReleaseRouter(invalid_selection)

    with pytest.raises(
        ArticulationV2ReleaseSelectionError,
        match="invalid release selection model",
    ):
        run_articulation_v2_workflow(request, release_router=router)

    assert not request.output_dir.exists()
    assert not request.output.path.exists()


def test_write_once_request_rejects_json_type_coercion(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    request.output_dir.mkdir()
    document = request.model_dump(mode="json")
    document["require_explicit_review"] = 1
    (request.output_dir / "request.v2.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    with pytest.raises(
        ArticulationV2WorkflowError,
        match="Existing articulation-v2 request conflicts",
    ):
        run_articulation_v2_workflow(request, release_router=router)

    assert not (request.output_dir / "checkpoint.v2.json").exists()


def test_write_once_revalidates_caller_constructed_models(tmp_path: Path) -> None:
    request = _request(tmp_path, kind="distance")
    contract = request.selector.contract
    assert contract.kind == "distance"
    invalid_contract = contract.model_copy(update={"minimum_distance_meters": -1.0})
    invalid_selector = request.selector.model_copy(
        update={"contract": invalid_contract}
    )
    invalid_request = request.model_copy(update={"selector": invalid_selector})
    router = _ExactContractTestReleaseRouter(invalid_request.release_selection)

    with pytest.raises(
        ArticulationV2WorkflowError,
        match="Invalid articulation-v2 request model",
    ):
        run_articulation_v2_workflow(invalid_request, release_router=router)

    assert not (request.output_dir / "request.v2.json").exists()
    assert not (request.output_dir / "checkpoint.v2.json").exists()


def test_strict_revalidation_rejects_nested_hidden_qualification_root(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, kind="distance")
    contract = request.selector.contract.model_copy(
        update={"qualification_evidence_root": "/tmp/issue-976"}
    )
    selector = request.selector.model_copy(update={"contract": contract})
    invalid_request = request.model_copy(update={"selector": selector})
    router = _ExactContractTestReleaseRouter(request.release_selection)

    with pytest.raises(
        ArticulationV2WorkflowError,
        match="Invalid articulation-v2 request model",
    ):
        run_articulation_v2_workflow(invalid_request, release_router=router)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        articulation_v2_canonical_sha256(contract)

    assert not request.output_dir.exists()
    assert not request.output.path.exists()


def test_strict_revalidation_rejects_extended_request_schema(tmp_path: Path) -> None:
    class ExtendedWorkflowRequest(ArticulationV2WorkflowRequest):
        qualification_evidence_root: str

    request = _request(tmp_path)
    extended_request = ExtendedWorkflowRequest.model_validate(
        {
            **request.model_dump(mode="python"),
            "qualification_evidence_root": "/tmp/issue-976",
        }
    )
    router = _ExactContractTestReleaseRouter(request.release_selection)

    with pytest.raises(
        ArticulationV2WorkflowError,
        match="Invalid articulation-v2 request model",
    ):
        run_articulation_v2_workflow(extended_request, release_router=router)
    with pytest.raises(TypeError, match="exact models_v2 model class"):
        articulation_v2_canonical_sha256(extended_request)

    assert not request.output_dir.exists()
    assert not request.output.path.exists()


def test_recorder_does_not_create_request_without_checkpoint(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    request.output_dir.mkdir()
    request.output.path.write_bytes(b'#usda 1.0\ndef Xform "Unpublished" {}\n')
    publication = ArticulationV2PublicationResult(
        request_sha256="a" * 64,
        release_selection_sha256="b" * 64,
        review_receipt_sha256="c" * 64,
        authoring_intent_sha256="d" * 64,
        idempotency_key="e" * 64,
        facade_call_id="must-not-be-recorded",
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source=request.source,
        reference=request.reference,
        output_target=request.output,
        output_artifact=_usd_identity(request.output.path, uri=request.output.uri),
        authoring_contract_sha256=articulation_v2_canonical_sha256(
            request.selector.contract
        ),
    )

    with pytest.raises(
        ArticulationV2WorkflowError,
        match="articulation-v2 checkpoint is missing",
    ):
        record_articulation_v2_publication(
            request,
            publication,
            release_router=router,
        )

    assert not (request.output_dir / "request.v2.json").exists()


def test_rejected_review_is_durable_and_never_prepares_authoring(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    run_articulation_v2_workflow(request, release_router=router)
    run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=_review_context(request),
    )
    receipt = build_articulation_v2_review_receipt(
        request.output_dir,
        release_router=router,
        decision="reject",
        reviewer="contract-test-reviewer",
    )

    rejected = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_receipt=receipt,
    )

    assert rejected.status == "cancelled"
    assert rejected.authoring_intent_path is None
    assert rejected.idempotency_key is None
    assert rejected.facade_call_count == 0
    assert run_articulation_v2_workflow(request, release_router=router) == rejected
    with pytest.raises(
        ArticulationV2WorkflowError,
        match="checkpointed handoff decision",
    ):
        run_articulation_v2_workflow(
            request,
            release_router=router,
            review_receipt=receipt.model_copy(update={"decision": "accept"}),
        )


def test_review_receipt_re_admits_the_checkpointed_selection(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    run_articulation_v2_workflow(request, release_router=router)
    run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=_review_context(request),
    )
    mismatched = request.release_selection.model_copy(
        update={"manifest_sha256": "9" * 64}
    )

    with pytest.raises(
        ArticulationV2ReleaseSelectionError,
        match="differs from the digest-bound workflow request",
    ):
        build_articulation_v2_review_receipt(
            request.output_dir,
            release_router=_ExactContractTestReleaseRouter(mismatched),
            decision="accept",
            reviewer="contract-test-reviewer",
        )

    assert not (request.output_dir / "review_receipt.v2.json").exists()


def test_v2_review_resume_idempotency_and_external_result_recording(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)

    initialized = run_articulation_v2_workflow(request, release_router=router)
    assert initialized.status == "awaiting_review_context"
    assert initialized.facade_call_count == 0
    assert initialized.authoring_intent_path is None

    context = _review_context(request)
    needs_review = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=context,
    )
    assert needs_review.status == "needs_review"
    assert needs_review.facade_call_count == 0

    receipt = build_articulation_v2_review_receipt(
        request.output_dir,
        release_router=router,
        decision="accept",
        reviewer="contract-test-reviewer",
    )
    ready = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_receipt=receipt,
    )
    assert ready.status == "ready_for_authoring"
    assert ready.idempotency_key is not None
    assert ready.facade_call_count == 0
    assert "has not invoked an authorer" in ready.message

    checkpoint_bytes = ready.checkpoint_path.read_bytes()
    repeated = run_articulation_v2_workflow(request, release_router=router)
    assert repeated == ready
    assert repeated.checkpoint_path.read_bytes() == checkpoint_bytes

    request.output.path.write_bytes(b'#usda 1.0\ndef Xform "Published" {}\n')
    output_identity = _usd_identity(
        request.output.path,
        uri=request.output.uri,
    )
    publication = ArticulationV2PublicationResult(
        request_sha256=_file_sha256(request.output_dir / "request.v2.json"),
        release_selection_sha256=articulation_v2_canonical_sha256(
            request.release_selection
        ),
        review_receipt_sha256=_file_sha256(
            request.output_dir / "review_receipt.v2.json"
        ),
        authoring_intent_sha256=_file_sha256(
            request.output_dir / "authoring_intent.v2.json"
        ),
        idempotency_key=ready.idempotency_key,
        facade_call_id="external-facade-call-contract-test",
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source=request.source,
        reference=request.reference,
        output_target=request.output,
        output_artifact=output_identity,
        authoring_contract_sha256=articulation_v2_canonical_sha256(
            request.selector.contract
        ),
    )
    published = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    assert published.status == "published"
    assert published.facade_call_count == 1

    resumed_after_publication = run_articulation_v2_workflow(
        request,
        release_router=router,
    )
    assert resumed_after_publication == published
    assert resumed_after_publication.facade_call_count == 1

    evidence_path = request.output_dir.parent / "saved-stage-readback.json"
    evidence_path.write_bytes(b'{"exact_saved_stage_match":true}\n')
    contract = request.selector.contract
    assert contract.kind == "fixed"
    readback = ArticulationV2ReadbackResult(
        status="pass",
        publication_result_sha256=_file_sha256(
            request.output_dir / "publication_result.v2.json"
        ),
        selector_kind=request.selector.kind,
        capability_id=request.selector.capability_id,
        source=request.source,
        reference=request.reference,
        output_artifact=output_identity,
        evidence_artifact=_opaque_identity(
            evidence_path,
            uri="asset://workflow/saved-stage-readback",
        ),
        constraint=ArticulationV2FixedReadback(
            joint_prim="/World/Joints/fixed",
            body0=contract.body0,
            body1=contract.body1,
            body0_attachment=contract.body0_attachment,
            body1_attachment=contract.body1_attachment,
        ),
        exact_saved_stage_match=True,
    )
    completed = record_articulation_v2_readback(
        request,
        readback,
        release_router=router,
    )
    assert completed.success is True
    assert completed.status == "completed"
    assert completed.facade_call_count == 1
    assert completed.output_asset_path == request.output.path

    final_checkpoint = completed.checkpoint_path.read_bytes()
    final_resume = run_articulation_v2_workflow(request, release_router=router)
    assert final_resume == completed
    assert final_resume.checkpoint_path.read_bytes() == final_checkpoint

    evidence_bytes = evidence_path.read_bytes()
    evidence_path.write_bytes(b'{"exact_saved_stage_match":false}\n')
    with pytest.raises(
        RuntimeError,
        match="saved-stage readback evidence (digest|size) mismatch",
    ):
        run_articulation_v2_workflow(request, release_router=router)
    evidence_path.write_bytes(evidence_bytes)

    output_bytes = request.output.path.read_bytes()
    request.output.path.write_bytes(output_bytes + b"tampered")
    with pytest.raises(
        RuntimeError, match="published v2 output (digest|size) mismatch"
    ):
        run_articulation_v2_workflow(request, release_router=router)


def test_external_publication_accepts_persisted_quaternion_contract_digest(
    tmp_path: Path,
) -> None:
    document = _request(tmp_path).model_dump(mode="json")
    component = 2.0**-0.5
    document["selector"]["contract"]["body0_attachment"]["orientation_wxyz"] = [
        -component,
        0.0,
        component,
        0.0,
    ]
    request = ArticulationV2WorkflowRequest.model_validate(document)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    durable_request = ArticulationV2WorkflowRequest.model_validate_json(
        (request.output_dir / "request.v2.json").read_bytes()
    )
    publication = _publication(
        durable_request,
        idempotency_key=ready.idempotency_key,
    )

    published = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )

    assert published.status == "published"
    assert publication.authoring_contract_sha256 == articulation_v2_canonical_sha256(
        request.selector.contract
    )


def test_external_publication_accepts_canonicalized_negative_zero_contract(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, kind="distance")
    contract = request.selector.contract.model_copy(
        update={"minimum_distance_meters": -0.0}
    )
    request = request.model_copy(
        update={"selector": request.selector.model_copy(update={"contract": contract})}
    )
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    durable_request = ArticulationV2WorkflowRequest.model_validate_json(
        (request.output_dir / "request.v2.json").read_bytes()
    )
    publication = _publication(
        durable_request,
        idempotency_key=ready.idempotency_key,
    )

    published = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )

    assert published.status == "published"
    assert durable_request.selector.contract.minimum_distance_meters == 0.0
    assert articulation_v2_canonical_sha256(
        request.selector.contract
    ) == articulation_v2_canonical_sha256(durable_request.selector.contract)


@pytest.mark.parametrize("kind", ["fixed", "distance"])
def test_v2_cancellation_is_durable_and_never_authors(
    tmp_path: Path,
    kind: str,
) -> None:
    request = _request(tmp_path, kind=kind)
    router = _ExactContractTestReleaseRouter(request.release_selection)

    cancelled = run_articulation_v2_workflow(
        request,
        release_router=router,
        cancel_checker=lambda: True,
    )
    assert cancelled.status == "cancelled"
    assert cancelled.success is False
    assert cancelled.facade_call_count == 0
    assert cancelled.authoring_intent_path is None
    assert not request.output.path.exists()

    resumed = run_articulation_v2_workflow(request, release_router=router)
    assert resumed == cancelled
    assert resumed.facade_call_count == 0


@pytest.mark.parametrize(
    ("dependency_name", "label"),
    [
        ("source_dependency.usda", "articulation-v2 source"),
        ("reference_dependency.usda", "articulation-v2 reference"),
    ],
)
def test_v2_resume_rejects_composed_source_or_reference_drift(
    tmp_path: Path,
    dependency_name: str,
    label: str,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    initialized = run_articulation_v2_workflow(request, release_router=router)
    dependency_path = tmp_path / dependency_name
    root_path = request.source.path if "source" in label else request.reference.path
    root_sha256 = _file_sha256(root_path)

    dependency_path.write_bytes(
        dependency_path.read_bytes() + b'\ndef Xform "Tampered" {}\n'
    )

    assert _file_sha256(root_path) == root_sha256
    with pytest.raises(
        RuntimeError,
        match=rf"{label} dependency bundle digest mismatch",
    ):
        run_articulation_v2_workflow(request, release_router=router)
    assert initialized.checkpoint_path.exists()


def test_cancelled_terminal_state_remains_readable_with_stale_output(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    cancelled = run_articulation_v2_workflow(
        request,
        release_router=router,
        cancel_checker=lambda: True,
    )
    request.output.path.write_bytes(b'#usda 1.0\ndef Xform "Stale" {}\n')

    assert run_articulation_v2_workflow(request, release_router=router) == cancelled


def test_cancellation_retry_ignores_unpersisted_review_input(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    run_articulation_v2_workflow(request, release_router=router)
    context = _review_context(request)
    cancelled = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=context,
        cancel_checker=lambda: True,
    )

    retried = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=context,
        cancel_checker=lambda: True,
    )

    assert cancelled.status == "cancelled"
    assert cancelled.review_context_path is None
    assert retried == cancelled


def test_authoring_intent_is_cancellation_point_of_no_return(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    ready_checkpoint = ready.checkpoint_path.read_bytes()

    def _unexpected_cancel_poll() -> bool:
        raise AssertionError("cancellation must not be polled after durable handoff")

    repeated = run_articulation_v2_workflow(
        request,
        release_router=router,
        cancel_checker=_unexpected_cancel_poll,
    )
    assert repeated == ready
    assert repeated.checkpoint_path.read_bytes() == ready_checkpoint

    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    published = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    published_checkpoint = published.checkpoint_path.read_bytes()
    resumed = run_articulation_v2_workflow(
        request,
        release_router=router,
        cancel_checker=lambda: True,
    )
    assert resumed == published
    assert resumed.checkpoint_path.read_bytes() == published_checkpoint

    completed = record_articulation_v2_readback(
        request,
        _readback(request, publication),
        release_router=router,
    )
    assert completed.status == "completed"


def test_status_poll_during_external_authoring_does_not_wedge(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    request.output.path.write_bytes(b"#usda 1.0\n")

    def _unexpected_cancel_poll() -> bool:
        raise AssertionError("cancellation must not be polled after durable handoff")

    polled = run_articulation_v2_workflow(
        request,
        release_router=router,
        cancel_checker=_unexpected_cancel_poll,
    )

    assert polled == ready
    assert polled.status == "ready_for_authoring"

    persisted_receipt = ArticulationV2ReviewReceipt.model_validate_json(
        (request.output_dir / "review_receipt.v2.json").read_bytes()
    )
    with pytest.raises(
        ArticulationV2WorkflowError,
        match="checkpointed handoff decision",
    ):
        run_articulation_v2_workflow(
            request,
            release_router=router,
            review_receipt=persisted_receipt.model_copy(
                update={"note": "conflicting post-handoff decision"}
            ),
        )


def test_terminal_retry_rejects_invalid_equal_review_model(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    run_articulation_v2_workflow(request, release_router=router)
    context = _review_context(request)
    run_articulation_v2_workflow(
        request,
        release_router=router,
        review_context=context,
    )
    receipt = build_articulation_v2_review_receipt(
        request.output_dir,
        release_router=router,
        decision="reject",
        reviewer="reviewer@example.com",
    )
    terminal = run_articulation_v2_workflow(
        request,
        release_router=router,
        review_receipt=receipt,
    )
    invalid_evidence = context.scene_evidence.model_copy(
        update={"size_bytes": float(context.scene_evidence.size_bytes)}
    )
    invalid_context = context.model_copy(update={"scene_evidence": invalid_evidence})

    assert terminal.status == "cancelled"
    assert invalid_context == context
    with pytest.raises(
        ArticulationV2WorkflowError,
        match="Invalid articulation-v2 review context model",
    ):
        run_articulation_v2_workflow(
            request,
            release_router=router,
            review_context=invalid_context,
        )


def test_symlinked_output_directory_uses_one_canonical_binding(
    tmp_path: Path,
) -> None:
    canonical_parent = tmp_path / "canonical"
    canonical_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(canonical_parent, target_is_directory=True)
    request = _request(
        tmp_path,
        output_dir=alias_parent / "nested" / ".." / "workflow-output",
    )
    router = _ExactContractTestReleaseRouter(request.release_selection)

    ready = _advance_to_ready(request, router)

    assert request.output_dir == canonical_parent / "workflow-output"
    assert ready.status == "ready_for_authoring"
    assert ready.review_receipt_path == (
        canonical_parent / "workflow-output" / "review_receipt.v2.json"
    )


@pytest.mark.parametrize("kind", ["fixed", "distance"])
def test_failed_readback_is_durable_and_resumable(
    tmp_path: Path,
    kind: str,
) -> None:
    request = _request(tmp_path, kind=kind)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    readback = _readback(request, publication, status="fail")

    failed = record_articulation_v2_readback(
        request,
        readback,
        release_router=router,
    )

    assert failed.status == "failed"
    assert failed.success is False
    assert failed.facade_call_count == 1
    assert failed.readback_result_path is not None
    checkpoint_bytes = failed.checkpoint_path.read_bytes()
    readback_bytes = failed.readback_result_path.read_bytes()
    resumed = run_articulation_v2_workflow(request, release_router=router)
    assert resumed == failed
    assert resumed.checkpoint_path.read_bytes() == checkpoint_bytes
    assert failed.readback_result_path.read_bytes() == readback_bytes

    replayed_publication = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    assert replayed_publication == failed

    readback.evidence_artifact.path.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(RuntimeError, match="readback evidence (digest|size) mismatch"):
        run_articulation_v2_workflow(request, release_router=router)


def test_passing_readback_requires_complete_attachment_frames(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    published = record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    contract = request.selector.contract
    assert contract.kind == "fixed"
    mismatched = ArticulationV2FixedReadback(
        joint_prim="/World/Joints/fixed",
        body0=contract.body0,
        body1=contract.body1,
        body0_attachment=_attachment(position=(0.0, 0.0, 0.15)),
        body1_attachment=contract.body1_attachment,
    )

    with pytest.raises(RuntimeError, match="complete selected constraint"):
        record_articulation_v2_readback(
            request,
            _readback(request, publication, constraint=mismatched),
            release_router=router,
        )

    assert run_articulation_v2_workflow(request, release_router=router) == published


def test_distance_readback_binds_frames_and_ordered_interval(tmp_path: Path) -> None:
    request = _request(tmp_path, kind="distance")
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )

    completed = record_articulation_v2_readback(
        request,
        _readback(request, publication),
        release_router=router,
    )

    assert completed.status == "completed"
    assert completed.success is True


@pytest.mark.parametrize("kind", ["fixed", "distance"])
def test_saved_stage_float32_readback_uses_release_tolerance(
    tmp_path: Path,
    kind: str,
) -> None:
    request = _request(tmp_path, kind=kind)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    constraint = _matching_constraint(request)
    document = constraint.model_dump(mode="json")
    for attachment_name in ("body0_attachment", "body1_attachment"):
        document[attachment_name]["position_meters"] = [
            _float32(component)
            for component in document[attachment_name]["position_meters"]
        ]
        document[attachment_name]["orientation_wxyz"] = [
            _float32(component)
            for component in document[attachment_name]["orientation_wxyz"]
        ]
    if kind == "distance":
        document["minimum_distance_meters"] = _float32(
            document["minimum_distance_meters"]
        )
        document["maximum_distance_meters"] = _float32(
            document["maximum_distance_meters"]
        )
        observed = ArticulationV2DistanceReadback.model_validate(document)
    else:
        observed = ArticulationV2FixedReadback.model_validate(document)
    readback = _readback(request, publication, constraint=observed)

    completed = record_articulation_v2_readback(
        request,
        readback,
        release_router=router,
    )

    assert readback.readback_tolerance == 1e-6
    assert completed.status == "completed"


def test_saved_stage_quaternion_readback_compares_modulo_sign(
    tmp_path: Path,
) -> None:
    document = _request(tmp_path).model_dump(mode="json")
    component = 2.0**-0.5
    document["selector"]["contract"]["body0_attachment"]["orientation_wxyz"] = [
        0.0,
        component,
        component,
        0.0,
    ]
    request = ArticulationV2WorkflowRequest.model_validate(document)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    observed_document = _matching_constraint(request).model_dump(mode="json")
    observed_document["body0_attachment"]["orientation_wxyz"] = [
        -1e-9,
        component,
        component,
        0.0,
    ]
    observed = ArticulationV2FixedReadback.model_validate(observed_document)

    completed = record_articulation_v2_readback(
        request,
        _readback(request, publication, constraint=observed),
        release_router=router,
    )

    assert completed.status == "completed"


def test_published_output_dependency_drift_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    router = _ExactContractTestReleaseRouter(request.release_selection)
    ready = _advance_to_ready(request, router)
    assert ready.idempotency_key is not None
    publication = _publication(request, idempotency_key=ready.idempotency_key)
    record_articulation_v2_publication(
        request,
        publication,
        release_router=router,
    )
    dependency_path = request.output.path.with_name(
        f"{request.output.path.stem}_dependency.usda"
    )
    dependency_path.write_bytes(
        dependency_path.read_bytes() + b'\ndef Xform "Tampered" {}\n'
    )

    with pytest.raises(
        RuntimeError,
        match="published v2 output dependency bundle digest mismatch",
    ):
        run_articulation_v2_workflow(request, release_router=router)
