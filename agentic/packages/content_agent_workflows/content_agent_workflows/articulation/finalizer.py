# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical progress and summary artifacts for articulation-v1."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256

from .decision import (
    SKILL_ROUTED_DECISION_METADATA_KEY,
    load_articulation_decision_ledger,
)
from .models import (
    ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION,
    ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION,
    ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION,
    ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION,
    ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationFinalizationResult,
    ArticulationFinalStatus,
    ArticulationInferenceResult,
    ArticulationMotionCapabilityDecision,
    ArticulationReviewReceipt,
    ArticulationRunState,
    ArticulationValidationResult,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    MotionCapabilityDecisionReasonCode,
    Stage2CandidateDocument,
)
from .scene_evidence import (
    ArticulationSceneEvidenceResult,
    verify_articulation_scene_evidence,
)

_SCENE_EVIDENCE_REQUIRED_METADATA_KEY = (
    "content_agent_workflows.scene_evidence_required"
)
_DIGEST_CHUNK_SIZE = 1024 * 1024


def _status_for_state(state: ArticulationRunState) -> ArticulationFinalStatus:
    if state.phase == "awaiting_decision":
        return "awaiting_decision"
    if state.phase == "completed":
        return "completed"
    if state.phase == "not_articulated":
        return "not_articulated"
    if state.phase == "conditional":
        return "conditional"
    if state.phase == "cancelled":
        return "cancelled"
    if state.phase == "failed":
        return "failed"
    if state.phase == "needs_review":
        return "needs_review"
    if state.phase == "awaiting_post_review":
        return "needs_review"
    raise ValueError(f"Cannot finalize non-reportable phase {state.phase!r}")


def _motion_capability_decisions(
    inference: ArticulationInferenceResult | None,
) -> tuple[ArticulationMotionCapabilityDecision, ...]:
    """Project typed semantic decisions into the native terminal report."""

    if inference is None:
        return ()
    decisions: list[ArticulationMotionCapabilityDecision] = []
    for candidate in inference.candidate_document.candidates:
        capability = candidate.motion_capability
        if capability is None:
            continue
        semantic_reason_code: MotionCapabilityDecisionReasonCode | None
        if capability.kind == "unsupported":
            semantic_reason_code = "motion_capability_unsupported"
        elif capability.kind == "unresolved":
            semantic_reason_code = "motion_capability_unresolved"
        elif "joint_type_conflict" in candidate.unresolved_reason_codes:
            semantic_reason_code = "motion_capability_conflict"
        else:
            semantic_reason_code = None
        reason_codes: tuple[MotionCapabilityDecisionReasonCode, ...] = tuple(
            candidate.unresolved_reason_codes
        )
        if semantic_reason_code is not None:
            reason_codes = (semantic_reason_code, *reason_codes)
        decisions.append(
            ArticulationMotionCapabilityDecision(
                candidate_id=candidate.candidate_id,
                semantic_role=candidate.semantic_role,
                motion_capability=capability,
                review_status=candidate.review_status,
                unresolved_reason_codes=reason_codes,
                unresolved_questions=candidate.unresolved_questions,
                evidence=candidate.evidence,
            )
        )
    return tuple(decisions)


def _message_for_state(
    state: ArticulationRunState,
    *,
    authoring: ArticulationAuthoringResult | None,
) -> str:
    if state.phase == "awaiting_decision":
        if state.standalone_preparation is not None:
            return (
                "Deterministic provider-neutral preparation is frozen; the "
                "Articulation reasoning child must write only the complete typed "
                "decision patch."
            )
        return (
            "Deterministic provider-neutral evidence is persisted; the outer "
            "coordinator must author the complete canonical graph."
        )
    if state.phase == "completed":
        return (
            "Approved articulation-v1 candidates were authored and matched exact "
            "saved USDZ readback."
        )
    if state.phase == "not_articulated":
        return (
            "Structure analysis completed successfully with bound evidence: the "
            "asset is not articulated and has no articulation candidates."
        )
    if state.phase == "conditional":
        if not state.accepted_candidate_ids:
            return "No articulation-v1 candidates were approved for authoring."
        if authoring is None:
            return state.error or (
                "Approved candidates were not authored; unresolved candidate or "
                "physical-membership evidence requires follow-up."
            )
        if state.unresolved_membership_disposition_ids:
            return (
                "The independent articulation output is valid, but typed physical "
                "membership still requires review or its separate downstream boundary."
            )
        return (
            "A useful articulation output exists, but exact readback or unresolved "
            "candidate evidence requires follow-up."
        )
    if state.phase == "cancelled":
        return "The articulation workflow stopped at a durable cancellation boundary."
    if state.phase == "failed":
        return state.error or "The articulation workflow failed."
    if state.phase == "awaiting_post_review":
        if state.standalone_readback is not None:
            return (
                "Exact standalone Articulation readback is persisted; independent "
                "output review is required before publication."
            )
        return (
            "Exact embedded articulation readback is persisted; the outer "
            "coordinator must review it before a receipt or stage completion."
        )
    if state.review_required_membership_disposition_ids:
        return (
            "Physical-membership inference is checkpointed; rerun through the "
            "typed Joint method with reviewed membership evidence. Candidate "
            "review receipts do not resolve the listed membership IDs."
        )
    return (
        "Candidate inference is checkpointed; explicit decisions are required for "
        "the listed candidate IDs."
    )


def _load_bound_inference_for_summary(
    binding: ArtifactBinding | None,
) -> ArticulationInferenceResult | None:
    """Best-effort load only when the exact checkpointed bytes remain intact."""

    if binding is None:
        return None
    try:
        verified = _read_verified_artifact_bytes(
            binding.path,
            binding.sha256,
            label="inference result",
        )
        if verified is None:
            return None
        _, payload = verified
        return ArticulationInferenceResult.model_validate(json.loads(payload))
    except (OSError, RuntimeError, ValueError):
        return None


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns if os.name != "nt" else 0,
    )


def _verify_artifact_content(
    path_value: str | None,
    expected_sha256: str | None,
    *,
    label: str,
    fail_closed: bool = False,
    capture_bytes: bool,
) -> tuple[str, bytes | None] | None:
    """Hash one pinned artifact and optionally retain its exact bytes."""

    if path_value is None or expected_sha256 is None:
        return None
    path = Path(path_value).expanduser()
    file_descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        file_descriptor = os.open(path, flags)
        opened_metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            while chunk := stream.read(_DIGEST_CHUNK_SIZE):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        final_descriptor_metadata = os.fstat(file_descriptor)
        current_path_metadata = os.stat(path, follow_symlinks=False)
        if _stat_signature(final_descriptor_metadata) != _stat_signature(
            opened_metadata
        ) or _stat_signature(current_path_metadata) != _stat_signature(opened_metadata):
            raise ValueError(f"{label} changed while being read: {path}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{label} digest mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    except (OSError, RuntimeError, ValueError) as exc:
        if fail_closed:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Cannot read {label}: {exc}") from exc
        return None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
    payload = b"".join(chunks) if chunks is not None else None
    return path_value, payload


def _read_verified_artifact_bytes(
    path_value: str | None,
    expected_sha256: str | None,
    *,
    label: str,
    fail_closed: bool = False,
) -> tuple[str, bytes] | None:
    """Read, hash, and pin one JSON artifact for same-byte parsing."""

    verified = _verify_artifact_content(
        path_value,
        expected_sha256,
        label=label,
        fail_closed=fail_closed,
        capture_bytes=True,
    )
    if verified is None:
        return None
    path, payload = verified
    assert payload is not None
    return path, payload


def _verified_optional_artifact_path(
    path_value: str | None,
    expected_sha256: str | None,
    *,
    label: str,
    fail_closed: bool = False,
) -> str | None:
    """Return one optional artifact path only while its bytes match."""

    verified = _verify_artifact_content(
        path_value,
        expected_sha256,
        label=label,
        fail_closed=fail_closed,
        capture_bytes=False,
    )
    return verified[0] if verified is not None else None


def _verified_binding_bytes(
    binding: ArtifactBinding,
    *,
    label: str,
) -> bytes:
    verified = _read_verified_artifact_bytes(
        binding.path,
        binding.sha256,
        label=label,
        fail_closed=True,
    )
    assert verified is not None
    return verified[1]


def _verified_binding_path(
    binding: ArtifactBinding | None,
    *,
    label: str,
    fail_closed: bool = False,
) -> str | None:
    """Return one checkpoint-bound path only while its bytes match."""

    if binding is None:
        return None
    return _verified_optional_artifact_path(
        binding.path,
        binding.sha256,
        label=label,
        fail_closed=fail_closed,
    )


def _require_completed_request_and_review_evidence(
    state: ArticulationRunState,
) -> None:
    """Re-derive completion requirements from digest-bound request evidence."""

    try:
        request = ArticulationWorkflowRequest.model_validate(
            json.loads(
                _verified_binding_bytes(
                    state.request,
                    label="articulation request",
                )
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid bound articulation request: {exc}") from exc

    candidate_binding = state.candidate_document
    assert candidate_binding is not None
    try:
        candidate_document = Stage2CandidateDocument.model_validate(
            json.loads(
                _verified_binding_bytes(
                    candidate_binding,
                    label="candidate document",
                )
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid bound candidate document: {exc}") from exc

    scene_required = request.metadata.get(
        _SCENE_EVIDENCE_REQUIRED_METADATA_KEY,
        False,
    )
    if not isinstance(scene_required, bool):
        raise ValueError("Scene evidence requirement metadata must be a boolean")
    scene_binding = state.scene_evidence
    collector_configuration_sha256 = state.scene_evidence_configuration_sha256
    if scene_required and (
        collector_configuration_sha256 is None or scene_binding is None
    ):
        raise ValueError(
            "Completed articulation summary is missing request-required Scene evidence"
        )
    if (scene_binding is None) != (collector_configuration_sha256 is None):
        raise ValueError(
            "Completed articulation summary has an incomplete Scene evidence binding"
        )
    if scene_binding is not None:
        assert collector_configuration_sha256 is not None
        evidence_root = request.output_dir.expanduser().resolve() / "scene_evidence"
        expected_manifest_path = evidence_root / "manifest.json"
        manifest_path = Path(scene_binding.path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = manifest_path.absolute()
        if manifest_path != expected_manifest_path:
            raise ValueError(
                "Completed articulation summary Scene evidence manifest path "
                "does not match the bound workflow output"
            )
        try:
            scene_evidence = ArticulationSceneEvidenceResult.model_validate_json(
                _verified_binding_bytes(
                    scene_binding,
                    label="Scene evidence manifest",
                )
            )
            verify_articulation_scene_evidence(
                scene_evidence,
                request=request,
                request_sha256=state.request.sha256,
                source_sha256=state.source_sha256,
                source_dependency_bundle_sha256=(state.source_dependency_bundle_sha256),
                candidate_document=candidate_document,
                candidate_document_sha256=candidate_binding.sha256,
                collector_configuration_sha256=collector_configuration_sha256,
                evidence_root=evidence_root,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid bound Scene articulation evidence: {exc}"
            ) from exc

    review_required: list[str] = []
    auto_accepted: list[str] = []
    unresolved_without_review: list[str] = []
    for candidate in candidate_document.candidates:
        if (
            not candidate.is_articulation_v1_authorable
            or candidate.articulation_v1_type not in request.allowed_motion_types
        ):
            if request.review_policy == "none":
                unresolved_without_review.append(candidate.candidate_id)
            else:
                review_required.append(candidate.candidate_id)
        elif request.review_policy == "all" or (
            request.review_policy == "uncertain" and candidate.confidence != "high"
        ):
            review_required.append(candidate.candidate_id)
        else:
            auto_accepted.append(candidate.candidate_id)

    expected_review_required = tuple(review_required)
    if state.candidate_ids != candidate_document.candidate_ids:
        raise ValueError(
            "Completed articulation summary candidate IDs differ from the bound "
            "candidate document"
        )
    if state.review_required_candidate_ids != expected_review_required:
        raise ValueError(
            "Completed articulation summary review-required scope differs from "
            "the bound request and candidates"
        )

    effective_candidate_document = candidate_document
    effective_candidate_document_sha256 = candidate_binding.sha256
    if request.metadata.get(SKILL_ROUTED_DECISION_METADATA_KEY):
        try:
            decision_context = load_articulation_decision_ledger(
                request.output_dir,
                state=state,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid bound Articulation decision ledger: {exc}"
            ) from exc
        if decision_context is None:
            raise ValueError(
                "Completed skill-routed articulation summary is missing its "
                "decision ledger"
            )
        _decision_patch, effective_candidate_document = decision_context
        effective_candidate_document_sha256 = file_sha256(
            request.output_dir / "agent_reviewed_articulation_candidates.json"
        )

    receipt: ArticulationReviewReceipt | None = None
    if expected_review_required:
        receipt_binding = state.review_receipt
        if receipt_binding is None:
            raise ValueError(
                "Completed articulation summary is missing its required review receipt"
            )
        try:
            receipt = ArticulationReviewReceipt.model_validate(
                json.loads(
                    _verified_binding_bytes(
                        receipt_binding,
                        label="review receipt",
                    )
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid bound review receipt: {exc}") from exc
        if (
            receipt.request_sha256 != state.request.sha256
            or receipt.source_sha256 != state.source_sha256
            or receipt.source_dependency_bundle_sha256
            != state.source_dependency_bundle_sha256
            or receipt.candidate_document_sha256 != effective_candidate_document_sha256
            or receipt.scene_evidence_sha256
            != (
                state.scene_evidence.sha256
                if state.scene_evidence is not None
                else None
            )
            or tuple(decision.candidate_id for decision in receipt.decisions)
            != expected_review_required
        ):
            raise ValueError(
                "Completed articulation summary review receipt differs from its "
                "bound request and candidates"
            )
    elif state.review_receipt is not None:
        raise ValueError(
            "Completed articulation summary has an unexpected review receipt"
        )

    accepted = set(auto_accepted)
    rejected: set[str] = set()
    candidate_by_id = effective_candidate_document.candidate_by_id()
    if receipt is not None:
        for decision in receipt.decisions:
            candidate = candidate_by_id[decision.candidate_id]
            if decision.decision == "accept":
                if (
                    not candidate.is_articulation_v1_authorable
                    or candidate.articulation_v1_type
                    not in request.allowed_motion_types
                ):
                    raise ValueError(
                        "Completed articulation summary receipt accepts a "
                        "non-authorable candidate"
                    )
                accepted.add(decision.candidate_id)
            else:
                rejected.add(decision.candidate_id)
    candidate_order = effective_candidate_document.candidate_ids
    expected_accepted = tuple(
        candidate_id for candidate_id in candidate_order if candidate_id in accepted
    )
    expected_rejected = tuple(
        candidate_id for candidate_id in candidate_order if candidate_id in rejected
    )
    expected_unresolved = tuple(
        candidate_id
        for candidate_id in candidate_order
        if candidate_id in set(unresolved_without_review)
    )
    if (
        state.accepted_candidate_ids != expected_accepted
        or state.rejected_candidate_ids != expected_rejected
        or state.unresolved_candidate_ids != expected_unresolved
    ):
        raise ValueError(
            "Completed articulation summary review outcome differs from its "
            "bound request, candidates, and receipt"
        )


def _require_completed_summary_evidence(
    state: ArticulationRunState,
    *,
    authoring: ArticulationAuthoringResult | None,
) -> None:
    """Reject a success index unless its durable evidence chain is complete."""

    if state.standalone_terminal_receipt is not None:
        from .standalone_decision import (
            validate_completed_standalone_articulation_checkpoint,
        )

        validate_completed_standalone_articulation_checkpoint(
            state,
            authoring=authoring,
        )
        return

    if state.embedded_decision_receipt is not None:
        from .embedded_decision import (
            validate_completed_embedded_articulation_checkpoint,
        )

        validate_completed_embedded_articulation_checkpoint(
            state,
            authoring=authoring,
        )
        return

    required_bindings = {
        "inference result": state.inference_result,
        "candidate document": state.candidate_document,
        "approved candidate document": state.approved_candidate_document,
        "authoring request": state.authoring_request,
        "authoring result": state.authoring_result,
        "validation result": state.validation_result,
    }
    if state.scene_evidence_configuration_sha256 is not None:
        required_bindings["Scene evidence manifest"] = state.scene_evidence
    if state.review_required_candidate_ids:
        required_bindings["review receipt"] = state.review_receipt
    missing = tuple(
        label for label, binding in required_bindings.items() if binding is None
    )
    if missing:
        raise ValueError(
            "Completed articulation summary is missing required evidence: "
            f"{', '.join(missing)}"
        )
    for label, binding in required_bindings.items():
        assert binding is not None
        _verified_binding_path(
            binding,
            label=label,
            fail_closed=True,
        )

    authoring_binding = state.authoring_result
    assert authoring_binding is not None
    try:
        bound_authoring = ArticulationAuthoringResult.model_validate(
            json.loads(
                _verified_binding_bytes(
                    authoring_binding,
                    label="authoring result",
                )
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid bound authoring result: {exc}") from exc
    if authoring is None:
        raise ValueError(
            "Completed articulation summary requires the bound authoring result"
        )
    if authoring != bound_authoring:
        raise ValueError(
            "Completed articulation summary authoring differs from the bound "
            "authoring result"
        )

    authoring_request_binding = state.authoring_request
    assert authoring_request_binding is not None
    try:
        bound_authoring_request = ArticulationAuthoringRequest.model_validate(
            json.loads(
                _verified_binding_bytes(
                    authoring_request_binding,
                    label="authoring request",
                )
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid bound authoring request: {exc}") from exc
    if bound_authoring_request.accepted_candidate_ids != state.accepted_candidate_ids:
        raise ValueError(
            "Completed articulation summary authoring request scope differs from "
            "accepted candidates"
        )
    approved_binding = state.approved_candidate_document
    assert approved_binding is not None
    if (
        bound_authoring_request.candidate_document_path != approved_binding.path
        or bound_authoring_request.candidate_document_sha256 != approved_binding.sha256
    ):
        raise ValueError(
            "Completed articulation summary authoring request differs from the "
            "approved candidate document"
        )
    if (
        bound_authoring.idempotency_key != bound_authoring_request.idempotency_key
        or bound_authoring.source_sha256 != bound_authoring_request.source_sha256
        or bound_authoring.source_dependency_bundle_sha256
        != bound_authoring_request.source_dependency_bundle_sha256
        or bound_authoring.candidate_document_path
        != bound_authoring_request.candidate_document_path
        or bound_authoring.candidate_document_sha256
        != bound_authoring_request.candidate_document_sha256
        or bound_authoring.authored_candidate_ids
        != bound_authoring_request.accepted_candidate_ids
    ):
        raise ValueError(
            "Completed articulation summary authoring result differs from its "
            "bound request"
        )
    if authoring.authored_candidate_ids != state.accepted_candidate_ids:
        raise ValueError(
            "Completed articulation summary authoring scope differs from accepted "
            "candidates"
        )
    _verified_optional_artifact_path(
        authoring.output_asset_path,
        authoring.output_asset_sha256,
        label="Published articulation output",
        fail_closed=True,
    )
    _require_completed_request_and_review_evidence(state)

    validation_binding = state.validation_result
    assert validation_binding is not None
    validation = ArticulationValidationResult.model_validate(
        json.loads(
            _verified_binding_bytes(
                validation_binding,
                label="validation result",
            )
        )
    )
    if validation.status != "pass":
        raise ValueError(
            "Completed articulation summary requires passing validation evidence"
        )
    if validation.expected_candidate_ids != state.accepted_candidate_ids:
        raise ValueError(
            "Completed articulation summary validation scope differs from accepted "
            "candidates"
        )
    if (
        validation.output_asset_path != authoring.output_asset_path
        or validation.expected_output_asset_sha256 != authoring.output_asset_sha256
    ):
        raise ValueError(
            "Completed articulation summary validation differs from the authored output"
        )


def articulation_workflow_progress_payload(
    state: ArticulationRunState,
) -> dict[str, Any]:
    """Build the canonical progress document for one checkpoint."""

    return {
        "schema_version": "content-agent-workflows.articulation-progress-log.v1",
        "revision": state.revision,
        "phase": state.phase,
        "candidate_ids": list(state.candidate_ids),
        "review_required_candidate_ids": list(state.review_required_candidate_ids),
        "accepted_candidate_ids": list(state.accepted_candidate_ids),
        "rejected_candidate_ids": list(state.rejected_candidate_ids),
        "unresolved_candidate_ids": list(state.unresolved_candidate_ids),
        "review_required_membership_disposition_ids": list(
            state.review_required_membership_disposition_ids
        ),
        "unresolved_membership_disposition_ids": list(
            state.unresolved_membership_disposition_ids
        ),
        "transitions": [
            transition.model_dump(mode="json") for transition in state.transitions
        ],
    }


def _build_articulation_workflow_summary(
    state: ArticulationRunState,
    *,
    output_dir: Path,
    authoring: ArticulationAuthoringResult | None = None,
) -> ArticulationFinalizationResult:
    """Build the current stable artifact index without mutating durable state."""

    resolved_output_dir = output_dir.expanduser().resolve()
    checkpoint_path = resolved_output_dir / "checkpoint.json"
    progress_path = resolved_output_dir / "workflow_progress.json"
    summary_path = resolved_output_dir / "final_summary.json"
    status = _status_for_state(state)
    if status == "completed":
        _require_completed_summary_evidence(state, authoring=authoring)

    fail_closed = status in {"completed", "not_articulated"}
    inference = _load_bound_inference_for_summary(state.inference_result)
    if (
        status in {"completed", "needs_review", "not_articulated"}
        and inference is None
        and state.embedded_evidence is None
        and state.standalone_preparation is None
    ):
        raise ValueError(
            f"{status} articulation summary requires a valid bound inference result"
        )
    predictions_path = (
        _verified_optional_artifact_path(
            inference.predictions_path,
            inference.predictions_sha256,
            label="Joint Agent predictions",
            fail_closed=fail_closed,
        )
        if inference is not None
        else None
    )
    report_path = (
        _verified_optional_artifact_path(
            inference.report_path,
            inference.report_sha256,
            label="Joint Agent candidate report",
            fail_closed=fail_closed,
        )
        if inference is not None
        else None
    )
    result = ArticulationFinalizationResult(
        schema_version=(
            ARTICULATION_STANDALONE_FINALIZATION_RESULT_SCHEMA_VERSION
            if state.schema_version == ARTICULATION_STANDALONE_RUN_STATE_SCHEMA_VERSION
            else ARTICULATION_EMBEDDED_FINALIZATION_RESULT_SCHEMA_VERSION
            if state.schema_version == ARTICULATION_EMBEDDED_RUN_STATE_SCHEMA_VERSION
            else "content-agent-workflows.articulation-finalization-result.v3"
            if state.schema_version
            == "content-agent-workflows.articulation-run-state.v3"
            else "content-agent-workflows.articulation-finalization-result.v2"
            if state.schema_version
            == "content-agent-workflows.articulation-run-state.v2"
            else ARTICULATION_FINALIZATION_RESULT_SCHEMA_VERSION
        ),
        success=status in {"completed", "not_articulated"},
        status=status,
        mode=state.mode,
        output_dir=str(resolved_output_dir),
        output_asset_path=(
            _verified_optional_artifact_path(
                authoring.output_asset_path,
                authoring.output_asset_sha256,
                label="Published articulation output",
                fail_closed=fail_closed,
            )
            if authoring is not None
            else None
        ),
        candidate_ids=state.candidate_ids,
        review_required_candidate_ids=state.review_required_candidate_ids,
        accepted_candidate_ids=state.accepted_candidate_ids,
        rejected_candidate_ids=state.rejected_candidate_ids,
        unresolved_candidate_ids=state.unresolved_candidate_ids,
        motion_capability_decisions=_motion_capability_decisions(inference),
        membership_dispositions=(
            inference.membership_disposition_document.dispositions
            if inference is not None
            and inference.membership_disposition_document is not None
            else ()
        ),
        review_required_membership_disposition_ids=(
            state.review_required_membership_disposition_ids
        ),
        unresolved_membership_disposition_ids=(
            state.unresolved_membership_disposition_ids
        ),
        request_path=_verified_binding_path(
            state.request,
            label="articulation request",
            fail_closed=fail_closed,
        ),
        checkpoint_path=str(checkpoint_path),
        inference_result_path=_verified_binding_path(
            state.inference_result,
            label="inference result",
            fail_closed=fail_closed,
        ),
        predictions_path=predictions_path,
        report_path=report_path,
        candidate_document_path=_verified_binding_path(
            state.candidate_document,
            label="candidate document",
            fail_closed=fail_closed,
        ),
        scene_evidence_path=_verified_binding_path(
            state.scene_evidence,
            label="Scene evidence manifest",
            fail_closed=fail_closed,
        ),
        review_receipt_path=_verified_binding_path(
            state.review_receipt,
            label="review receipt",
            fail_closed=fail_closed,
        ),
        approved_candidate_document_path=_verified_binding_path(
            state.approved_candidate_document,
            label="approved candidate document",
            fail_closed=fail_closed,
        ),
        authoring_request_path=_verified_binding_path(
            state.authoring_request,
            label="authoring request",
            fail_closed=fail_closed,
        ),
        authoring_result_path=_verified_binding_path(
            state.authoring_result,
            label="authoring result",
            fail_closed=fail_closed,
        ),
        diagnostics_path=(
            _verified_optional_artifact_path(
                authoring.diagnostics_path,
                authoring.diagnostics_sha256,
                label="Joint Rigger diagnostics",
                fail_closed=fail_closed,
            )
            if authoring is not None
            else None
        ),
        joint_rigger_result_path=(
            _verified_optional_artifact_path(
                authoring.joint_rigger_result_path,
                authoring.joint_rigger_result_sha256,
                label="Joint Rigger result",
                fail_closed=fail_closed,
            )
            if authoring is not None
            else None
        ),
        validation_result_path=_verified_binding_path(
            state.validation_result,
            label="validation result",
            fail_closed=fail_closed,
        ),
        embedded_evidence_path=_verified_binding_path(
            state.embedded_evidence,
            label="embedded articulation evidence",
            fail_closed=fail_closed,
        ),
        embedded_proposal_path=_verified_binding_path(
            state.embedded_proposal,
            label="embedded articulation proposal",
            fail_closed=fail_closed,
        ),
        embedded_canonical_graph_path=_verified_binding_path(
            state.embedded_canonical_graph,
            label="embedded canonical articulation graph",
            fail_closed=fail_closed,
        ),
        embedded_graph_revision_path=_verified_binding_path(
            state.embedded_graph_revision,
            label="embedded articulation graph revision",
            fail_closed=fail_closed,
        ),
        embedded_outer_review_path=_verified_binding_path(
            state.embedded_outer_review,
            label="embedded articulation outer graph review",
            fail_closed=fail_closed,
        ),
        embedded_coordinator_decision_path=_verified_binding_path(
            state.embedded_coordinator_decision,
            label="embedded articulation coordinator decision",
            fail_closed=fail_closed,
        ),
        embedded_human_decision_path=_verified_binding_path(
            state.embedded_human_decision,
            label="embedded articulation human decision",
            fail_closed=fail_closed,
        ),
        embedded_execution_authorization_path=_verified_binding_path(
            state.embedded_execution_authorization,
            label="embedded articulation execution authorization",
            fail_closed=fail_closed,
        ),
        embedded_readback_path=_verified_binding_path(
            state.embedded_readback,
            label="embedded articulation readback",
            fail_closed=fail_closed,
        ),
        embedded_execution_result_path=_verified_binding_path(
            state.embedded_execution_result,
            label="embedded articulation execution result",
            fail_closed=fail_closed,
        ),
        embedded_output_evidence_path=_verified_binding_path(
            state.embedded_output_evidence,
            label="embedded articulation output evidence",
            fail_closed=fail_closed,
        ),
        embedded_coordinator_review_path=_verified_binding_path(
            state.embedded_coordinator_review,
            label="embedded articulation coordinator review",
            fail_closed=fail_closed,
        ),
        embedded_decision_receipt_path=_verified_binding_path(
            state.embedded_decision_receipt,
            label="embedded articulation decision receipt",
            fail_closed=fail_closed,
        ),
        embedded_terminal_receipt_path=_verified_binding_path(
            state.embedded_terminal_receipt,
            label="embedded articulation terminal receipt",
            fail_closed=fail_closed,
        ),
        standalone_identity_path=_verified_binding_path(
            state.standalone_identity,
            label="standalone articulation identity",
            fail_closed=fail_closed,
        ),
        standalone_preparation_path=_verified_binding_path(
            state.standalone_preparation,
            label="standalone articulation preparation",
            fail_closed=fail_closed,
        ),
        standalone_proposal_path=_verified_binding_path(
            state.standalone_proposal,
            label="standalone articulation proposal",
            fail_closed=fail_closed,
        ),
        standalone_decision_patch_path=_verified_binding_path(
            state.standalone_decision_patch,
            label="standalone articulation decision patch",
            fail_closed=fail_closed,
        ),
        standalone_decision_ledger_path=_verified_binding_path(
            state.standalone_decision_ledger,
            label="standalone articulation decision ledger",
            fail_closed=fail_closed,
        ),
        standalone_canonical_graph_path=_verified_binding_path(
            state.standalone_canonical_graph,
            label="standalone canonical articulation graph",
            fail_closed=fail_closed,
        ),
        standalone_authoring_receipt_path=_verified_binding_path(
            state.standalone_authoring_receipt,
            label="standalone articulation authoring receipt",
            fail_closed=fail_closed,
        ),
        standalone_readback_path=_verified_binding_path(
            state.standalone_readback,
            label="standalone articulation readback",
            fail_closed=fail_closed,
        ),
        standalone_output_evidence_path=_verified_binding_path(
            state.standalone_output_evidence,
            label="standalone articulation output evidence",
            fail_closed=fail_closed,
        ),
        standalone_post_review_path=_verified_binding_path(
            state.standalone_post_review,
            label="standalone articulation post review",
            fail_closed=fail_closed,
        ),
        standalone_cleanup_path=_verified_binding_path(
            state.standalone_cleanup,
            label="standalone articulation cleanup receipt",
            fail_closed=fail_closed,
        ),
        standalone_terminal_receipt_path=_verified_binding_path(
            state.standalone_terminal_receipt,
            label="standalone articulation terminal receipt",
            fail_closed=fail_closed,
        ),
        standalone_refinement_history_paths=tuple(
            path
            for index, binding in enumerate(state.standalone_refinement_history)
            if (
                path := _verified_binding_path(
                    binding,
                    label=f"standalone articulation refinement artifact {index}",
                    fail_closed=fail_closed,
                )
            )
            is not None
        ),
        workflow_progress_path=str(progress_path),
        final_summary_path=str(summary_path),
        message=_message_for_state(state, authoring=authoring),
        terminal_status=state.backend_terminal_status,
        recovery_action=(
            state.backend_terminal_status.get("recovery_action")
            if state.backend_terminal_status is not None
            else None
        ),
    )
    return result


def verify_articulation_workflow_summary(
    state: ArticulationRunState,
    *,
    output_dir: Path,
    authoring: ArticulationAuthoringResult | None = None,
) -> None:
    """Require persisted progress and summary indexes to match one checkpoint."""

    resolved_output_dir = output_dir.expanduser().resolve()
    progress_path = resolved_output_dir / "workflow_progress.json"
    summary_path = resolved_output_dir / "final_summary.json"
    expected_progress = articulation_workflow_progress_payload(state)
    expected_summary = _build_articulation_workflow_summary(
        state,
        output_dir=resolved_output_dir,
        authoring=authoring,
    )
    try:
        actual_progress = json.loads(progress_path.read_bytes())
        actual_summary_payload = json.loads(summary_path.read_bytes())
        actual_summary = ArticulationFinalizationResult.model_validate(
            actual_summary_payload
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid persisted articulation summary index: {exc}"
        ) from exc
    if actual_progress != expected_progress:
        raise ValueError(
            "Persisted articulation workflow progress differs from the checkpoint"
        )
    if actual_summary != expected_summary:
        raise ValueError(
            "Persisted articulation final summary differs from the checkpoint"
        )


def write_articulation_workflow_summary(
    state: ArticulationRunState,
    *,
    output_dir: Path,
    authoring: ArticulationAuthoringResult | None = None,
) -> ArticulationFinalizationResult:
    """Write current progress and a stable artifact index."""

    resolved_output_dir = output_dir.expanduser().resolve()
    progress_path = resolved_output_dir / "workflow_progress.json"
    summary_path = resolved_output_dir / "final_summary.json"
    result = _build_articulation_workflow_summary(
        state,
        output_dir=resolved_output_dir,
        authoring=authoring,
    )
    atomic_write_json(progress_path, articulation_workflow_progress_payload(state))
    atomic_write_json(summary_path, result)
    return result


__all__ = [
    "articulation_workflow_progress_payload",
    "verify_articulation_workflow_summary",
    "write_articulation_workflow_summary",
]
