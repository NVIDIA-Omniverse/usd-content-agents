# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone outer-assessment boundary for focused Validation operations."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import (
    ExecutionArtifactBinding,
    domain_execution_context_from_metadata,
)
from content_agent_workflows.common.embedded_domain_decision import (
    ProviderNeutralEvidenceRecord,
    canonical_json_digest,
)

from .coordinator import (
    VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME,
    VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
    VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
    VALIDATION_COORDINATOR_PREPARATION_NAME,
    ValidationCoordinatorAcceptedPlan,
    ValidationCoordinatorExecutionReceipt,
    ValidationCoordinatorPlanPatch,
    ValidationCoordinatorPreparation,
)
from .embedded_assessment import (
    CANONICAL_VALIDATION_ASSESSMENT_NAME,
    VALIDATION_TERMINAL_RECEIPT_NAME,
    EmbeddedValidationAssessmentError,
    ValidationCoordinatorAssessment,
    ValidationCoordinatorReviewDraft,
    ValidationTerminalReceipt,
    _binding,
    _gate_dispositions,
    _load_model,
    _package_evidence_record,
    _template_evidence_records,
    _visual_evidence_record,
    load_embedded_validation_run,
    validate_coordinator_assessment,
)
from .models import ValidationWorkflowRun
from .operations import (
    VALIDATION_OPERATION_INDEX_NAME,
    load_validated_validation_operation_index,
)
from .workflow import ValidationWorkflowError

STANDALONE_VALIDATION_EVIDENCE_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.standalone-validation-evidence-index.v1"
)
STANDALONE_VALIDATION_EXECUTION_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.standalone-validation-execution-index.v1"
)
STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME: Final = "standalone_validation_evidence.json"
STANDALONE_VALIDATION_EXECUTION_INDEX_NAME: Final = (
    "standalone_validation_execution.json"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StandaloneValidationEvidenceIndex(_FrozenModel):
    """Mode-local wrapper around the same provider-neutral evidence records."""

    schema_version: Literal[
        "content-agent-workflows.standalone-validation-evidence-index.v1"
    ] = STANDALONE_VALIDATION_EVIDENCE_INDEX_SCHEMA_VERSION
    assessment_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation_index: ExecutionArtifactBinding
    records: tuple[ProviderNeutralEvidenceRecord, ...] = Field(min_length=1)
    nested_agent_launched: Literal[False] = False


class StandaloneValidationExecutionIndex(_FrozenModel):
    """Digest-bound canonical assessment awaiting or completing outer review."""

    schema_version: Literal[
        "content-agent-workflows.standalone-validation-execution-index.v1"
    ] = STANDALONE_VALIDATION_EXECUTION_INDEX_SCHEMA_VERSION
    assessment_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_id: str = Field(min_length=1)
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_index: ExecutionArtifactBinding
    canonical_assessment: ExecutionArtifactBinding | None = None
    terminal_receipt: ExecutionArtifactBinding | None = None
    authorized: bool
    nested_agent_launched: Literal[False] = False


def _write_once(path: Path, value: BaseModel) -> None:
    if path.is_symlink():
        raise EmbeddedValidationAssessmentError(
            f"standalone Validation artifact must not be a symlink: {path}"
        )
    if path.exists():
        try:
            existing = value.__class__.model_validate(load_json(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise EmbeddedValidationAssessmentError(
                f"Invalid existing standalone Validation artifact {path}: {exc}"
            ) from exc
        if existing != value:
            raise EmbeddedValidationAssessmentError(
                f"standalone Validation artifact already differs: {path}"
            )
        return
    atomic_write_json(path, value)


def _load_standalone_run(output_dir: str | Path) -> ValidationWorkflowRun:
    run = load_embedded_validation_run(output_dir)
    context = domain_execution_context_from_metadata(
        run.request.metadata,
        expected_domain="validation",
    )
    if context is not None and context.mode != "standalone":
        raise EmbeddedValidationAssessmentError(
            "standalone Validation assessment cannot consume an embedded run"
        )
    return run


def _standalone_identity(
    *,
    run: ValidationWorkflowRun,
    operation_index: ExecutionArtifactBinding,
    coordinator_bindings: dict[str, ExecutionArtifactBinding],
) -> str:
    return canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.validation-assessment-identity.v1",
            "mode": "standalone",
            "workflow_identity": run.checkpoint.workflow_identity.identity_digest,
            "operation_index": operation_index.model_dump(mode="json"),
            "request": file_sha256(Path(run.request_path)),
            "plan": file_sha256(Path(run.plan_path)),
            "result": file_sha256(Path(run.result_path)),
            "evidence": file_sha256(Path(run.evidence_path)),
            "summary": file_sha256(Path(run.final_summary_path)),
            "coordinator": {
                key: binding.model_dump(mode="json")
                for key, binding in sorted(coordinator_bindings.items())
            },
        }
    )


def _coordinator_bindings(root: Path) -> dict[str, ExecutionArtifactBinding]:
    paths = {
        "preparation": root / VALIDATION_COORDINATOR_PREPARATION_NAME,
        "plan_patch": root / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        "accepted_plan": root / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME,
        "execution": root / VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
    }
    present = {
        name for name, path in paths.items() if path.exists() or path.is_symlink()
    }
    if not present:
        return {}
    if present != set(paths):
        raise EmbeddedValidationAssessmentError(
            "standalone Validation has a partial coordinator decision chain"
        )
    preparation = _load_model(paths["preparation"], ValidationCoordinatorPreparation)
    patch = _load_model(paths["plan_patch"], ValidationCoordinatorPlanPatch)
    accepted = _load_model(paths["accepted_plan"], ValidationCoordinatorAcceptedPlan)
    execution = _load_model(paths["execution"], ValidationCoordinatorExecutionReceipt)
    bindings = {name: _binding(path) for name, path in paths.items()}
    accepted_nested_bindings = (
        accepted.preparation,
        accepted.plan_patch,
        accepted.child_output,
        accepted.child_launch_descriptor,
        accepted.operation_preparation,
        accepted.compatibility_request,
        accepted.compatibility_plan,
    )
    execution_nested_bindings = (
        *execution.operation_results,
        execution.operation_index,
        execution.validation_result,
        execution.validation_evidence,
        execution.final_summary,
    )
    if (
        patch.preparation_digest != preparation.preparation_digest
        or accepted.preparation_digest != preparation.preparation_digest
        or accepted.accepted_plan_digest != execution.accepted_plan_digest
        or accepted.preparation != bindings["preparation"]
        or accepted.plan_patch != bindings["plan_patch"]
        or execution.accepted_plan != bindings["accepted_plan"]
        or execution.operation_index != _binding(root / VALIDATION_OPERATION_INDEX_NAME)
        or any(
            _binding(binding.path) != binding for binding in accepted_nested_bindings
        )
        or any(
            _binding(binding.path) != binding for binding in execution_nested_bindings
        )
    ):
        raise EmbeddedValidationAssessmentError(
            "standalone Validation coordinator decision chain is stale"
        )
    return bindings


def _build_standalone_validation_evidence(
    output_dir: str | Path,
) -> StandaloneValidationEvidenceIndex:
    run = _load_standalone_run(output_dir)
    root = Path(run.output_dir)
    operation_index_path = root / VALIDATION_OPERATION_INDEX_NAME
    try:
        operation_index = load_validated_validation_operation_index(root)
    except ValidationWorkflowError as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid standalone Validation operation evidence: {exc}"
        ) from exc
    if not operation_index.required_operations_complete:
        raise EmbeddedValidationAssessmentError(
            "standalone assessment requires every mandatory selected operation"
        )
    operation_index_binding = _binding(operation_index_path)
    coordinator_bindings = _coordinator_bindings(root)
    mandatory_by_template = {
        item.template_name: item.mandatory for item in operation_index.operations
    }
    records = tuple(
        record
        for result in run.result.template_results
        for record in _template_evidence_records(
            run,
            result,
            required=mandatory_by_template[result.template_name],
        )
    ) + (
        _visual_evidence_record(
            run,
            required=mandatory_by_template.get("render_valid", False),
        ),
        _package_evidence_record(run),
        ProviderNeutralEvidenceRecord(
            evidence_id="validation-cross-stage-integrity",
            evidence_type="validation",
            status="unsupported",
            required=False,
            summary="Cross-stage evidence was not requested in standalone mode.",
        ),
    )
    return StandaloneValidationEvidenceIndex(
        assessment_identity_sha256=_standalone_identity(
            run=run,
            operation_index=operation_index_binding,
            coordinator_bindings=coordinator_bindings,
        ),
        operation_index=operation_index_binding,
        records=records,
    )


def prepare_standalone_validation_evidence(
    output_dir: str | Path,
) -> StandaloneValidationEvidenceIndex:
    """Expose focused native facts and images without starting another reasoner."""

    index = _build_standalone_validation_evidence(output_dir)
    root = Path(output_dir).expanduser().resolve()
    _write_once(root / STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME, index)
    return index


def _load_evidence_index(root: Path) -> StandaloneValidationEvidenceIndex:
    try:
        index = StandaloneValidationEvidenceIndex.model_validate(
            load_json(root / STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME)
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid standalone Validation evidence index: {exc}"
        ) from exc
    refreshed = _build_standalone_validation_evidence(root)
    if refreshed != index:
        raise EmbeddedValidationAssessmentError(
            "standalone Validation evidence identity is stale"
        )
    return index


def assess_standalone_validation(
    output_dir: str | Path,
    *,
    assessment_path: str | Path,
) -> StandaloneValidationExecutionIndex:
    """Validate and publish the outer-authored standalone assessment."""

    root = Path(output_dir).expanduser().resolve()
    index = _load_evidence_index(root)
    assessment = _load_model(
        Path(assessment_path).expanduser().resolve(),
        ValidationCoordinatorAssessment,
    )
    validate_coordinator_assessment(assessment, evidence=index.records)
    authorized = assessment.terminal_disposition == "pass"
    canonical_binding = None
    if authorized:
        canonical_path = root / CANONICAL_VALIDATION_ASSESSMENT_NAME
        _write_once(canonical_path, assessment)
        canonical_binding = _binding(canonical_path)
    execution = StandaloneValidationExecutionIndex(
        assessment_identity_sha256=index.assessment_identity_sha256,
        assessment_id=assessment.assessment_id,
        assessment_sha256=canonical_json_digest(assessment),
        evidence_index=_binding(root / STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME),
        canonical_assessment=canonical_binding,
        authorized=authorized,
    )
    _write_once(root / STANDALONE_VALIDATION_EXECUTION_INDEX_NAME, execution)
    return execution


def review_standalone_validation_assessment(
    output_dir: str | Path,
    *,
    review_path: str | Path,
) -> ValidationTerminalReceipt:
    """Seal the same terminal receipt shape used by composed Validation."""

    root = Path(output_dir).expanduser().resolve()
    index = _load_evidence_index(root)
    execution = _load_model(
        root / STANDALONE_VALIDATION_EXECUTION_INDEX_NAME,
        StandaloneValidationExecutionIndex,
    )
    if not execution.authorized or execution.canonical_assessment is None:
        raise EmbeddedValidationAssessmentError(
            "rejected or revision-required standalone assessment has no output to review"
        )
    if (
        execution.assessment_identity_sha256 != index.assessment_identity_sha256
        or _binding(execution.evidence_index.path) != execution.evidence_index
        or _binding(execution.canonical_assessment.path)
        != execution.canonical_assessment
    ):
        raise EmbeddedValidationAssessmentError(
            "standalone Validation execution identity is stale"
        )
    assessment = _load_model(
        Path(execution.canonical_assessment.path),
        ValidationCoordinatorAssessment,
    )
    review = _load_model(
        Path(review_path).expanduser().resolve(),
        ValidationCoordinatorReviewDraft,
    )
    if review.disposition == "accept" and assessment.terminal_disposition != "pass":
        raise EmbeddedValidationAssessmentError(
            "outer review cannot accept a non-passing standalone assessment"
        )
    coordinator_bindings = _coordinator_bindings(root)
    required_checks_successful = True
    coordinator_execution = coordinator_bindings.get("execution")
    if coordinator_execution is not None:
        required_checks_successful = _load_model(
            Path(coordinator_execution.path),
            ValidationCoordinatorExecutionReceipt,
        ).required_checks_successful
    receipt = ValidationTerminalReceipt(
        mode="standalone",
        assessment_identity_sha256=index.assessment_identity_sha256,
        evidence_index=execution.evidence_index,
        canonical_assessment=execution.canonical_assessment,
        coordinator_review=_binding(Path(review_path).expanduser().resolve()),
        gate_dispositions=_gate_dispositions(assessment),
        terminal_disposition=assessment.terminal_disposition,
        review_disposition=review.disposition,
        receipt_status=(
            "completed"
            if review.disposition == "accept"
            and assessment.terminal_disposition == "pass"
            and required_checks_successful
            else "rejected"
        ),
        coordinator_preparation=coordinator_bindings.get("preparation"),
        coordinator_plan_patch=coordinator_bindings.get("plan_patch"),
        accepted_coordinator_plan=coordinator_bindings.get("accepted_plan"),
        coordinator_execution=coordinator_bindings.get("execution"),
        operation_index=(index.operation_index if coordinator_bindings else None),
        coordinator_planning_agent_launched=bool(coordinator_bindings),
    )
    receipt_path = root / VALIDATION_TERMINAL_RECEIPT_NAME
    _write_once(receipt_path, receipt)
    completed_execution = execution.model_copy(
        update={"terminal_receipt": _binding(receipt_path)}
    )
    execution_path = root / STANDALONE_VALIDATION_EXECUTION_INDEX_NAME
    atomic_write_json(execution_path, completed_execution)
    return receipt


__all__ = [
    "STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME",
    "STANDALONE_VALIDATION_EXECUTION_INDEX_NAME",
    "StandaloneValidationEvidenceIndex",
    "StandaloneValidationExecutionIndex",
    "assess_standalone_validation",
    "prepare_standalone_validation_evidence",
    "review_standalone_validation_assessment",
]
