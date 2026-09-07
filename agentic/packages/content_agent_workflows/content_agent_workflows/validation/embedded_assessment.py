# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outer-coordinator assessment boundary for embedded Validation execution.

The native Validation workflow remains the factual evidence provider.  In an
embedded asset run its aggregate verdict and any ``look_right`` judgment are
not the composed-workflow verdict.  This module adapts the terminal native
bundle into provider-neutral evidence and an optional critique proposal, then
requires the outer coordinator to author a typed assessment.  A deterministic
non-mutating finalizer may publish that assessment only after the shared
authorization is durably committed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)
from world_understanding.validation import (
    ValidationPlan,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateResult,
    aggregate_validation_verdict,
)
from world_understanding.validation.cli import (
    ValidationCliError,
    finalize_validation_result,
)

from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import (
    ExecutionArtifactBinding,
    domain_execution_context_from_metadata,
)
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EmbeddedDecisionArtifactStore,
    EmbeddedDecisionAuthorizationReplayError,
)
from content_agent_workflows.common.embedded_domain_decision import (
    AcceptedSemanticDecision,
    BoundedExecutionAuthorization,
    ContractArtifactReference,
    CoordinatorDecisionDisposition,
    DomainProposalPayload,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    EvidenceStatus,
    PersistedExecutionLineage,
    ProducerIdentity,
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

from .models import (
    VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
    ValidationArtifactIdentity,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowRun,
    ValidationWorkflowStatus,
)

VALIDATION_COORDINATOR_ASSESSMENT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-assessment.v1"
)
VALIDATION_COORDINATOR_REVIEW_DRAFT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-review-draft.v1"
)
EMBEDDED_VALIDATION_EVIDENCE_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-validation-evidence-index.v1"
)
EMBEDDED_VALIDATION_EXECUTION_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-validation-execution-index.v1"
)
EMBEDDED_VALIDATION_RECEIPT_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-validation-receipt-index.v2"
)
_LEGACY_EMBEDDED_VALIDATION_RECEIPT_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.embedded-validation-receipt-index.v1"
)
VALIDATION_TERMINAL_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-terminal-receipt.v1"
)

EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME: Final = "embedded_validation_evidence.json"
EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME: Final = "embedded_validation_execution.json"
EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME: Final = "embedded_validation_receipt.json"
CANONICAL_VALIDATION_ASSESSMENT_NAME: Final = "canonical_validation_assessment.json"
VALIDATION_TERMINAL_RECEIPT_NAME: Final = "validation_terminal_receipt.json"

_TEMPLATE_GATE: Final[dict[str, ValidationGateName]] = {
    "render_valid": "static_validation",
    "physics_sane": "static_validation",
    "physical_behavior": "runtime_validation",
    "look_right": "visual_quality",
}


class EmbeddedValidationAssessmentError(RuntimeError):
    """Raised when embedded Validation cannot advance without weakening a gate."""


ValidationGateName = Literal[
    "static_validation",
    "runtime_validation",
    "visual_quality",
    "package_integrity",
    "cross_stage_integrity",
]
ValidationEvidenceVerdict = Literal[
    "pass",
    "warn",
    "fail",
    "needs_remediation",
    "error",
    "unsupported",
]
ValidationFindingSeverity = Literal["info", "warning", "error", "critical"]
ValidationFindingDisposition = Literal[
    "accepted",
    "remediate",
    "waived",
    "deferred",
]
ValidationGateDisposition = Literal["pass", "fail", "waive", "defer"]
ValidationTerminalDisposition = Literal[
    "pass",
    "fail",
    "needs_remediation",
    "deferred",
    "blocked",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationAssessmentFinding(_FrozenModel):
    """One outer-accepted factual finding and its explicit disposition."""

    finding_id: str = Field(min_length=1)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1)
    source_issue_codes: tuple[str, ...] = ()
    severity: ValidationFindingSeverity
    summary: str = Field(min_length=1)
    affected_artifacts: tuple[str, ...] = ()
    affected_prims: tuple[str, ...] = ()
    remediation_requirements: tuple[str, ...] = ()
    disposition: ValidationFindingDisposition
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.disposition == "remediate" and not self.remediation_requirements:
            raise ValueError(
                "remediation findings require explicit remediation_requirements"
            )
        if self.disposition in {"waived", "deferred"} and not self.rationale.strip():
            raise ValueError("waived or deferred findings require a rationale")
        return self


class ValidationGateAssessment(_FrozenModel):
    """Outer disposition of one independent Validation evidence class."""

    gate: ValidationGateName
    required: bool = True
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    disposition: ValidationGateDisposition
    rationale: str = Field(min_length=1)


class ValidationCoordinatorAssessment(_FrozenModel):
    """Canonical semantic Validation assessment authored by the outer loop."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-assessment.v1"
    ] = VALIDATION_COORDINATOR_ASSESSMENT_SCHEMA_VERSION
    assessment_id: str = Field(min_length=1)
    created_at: datetime
    gates: tuple[ValidationGateAssessment, ...] = Field(min_length=1)
    findings: tuple[ValidationAssessmentFinding, ...] = ()
    terminal_disposition: ValidationTerminalDisposition
    summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_assessment_shape(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("assessment created_at must include a timezone")
        gate_names = [gate.gate for gate in self.gates]
        if len(gate_names) != len(set(gate_names)):
            raise ValueError("assessment gate names must be unique")
        evidence_ids = [
            evidence_id for gate in self.gates for evidence_id in gate.evidence_ids
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("assessment evidence IDs must belong to one gate")
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("assessment finding IDs must be unique")
        if self.terminal_disposition == "pass":
            invalid_gates = [
                gate.gate
                for gate in self.gates
                if gate.required and gate.disposition not in {"pass", "waive"}
            ]
            if invalid_gates:
                raise ValueError(
                    "passing assessment has unresolved required gates: "
                    + ", ".join(invalid_gates)
                )
            invalid_findings = [
                finding.finding_id
                for finding in self.findings
                if finding.disposition in {"remediate", "deferred"}
            ]
            if invalid_findings:
                raise ValueError(
                    "passing assessment has unresolved findings: "
                    + ", ".join(invalid_findings)
                )
        return self


class ValidationCoordinatorReviewDraft(_FrozenModel):
    """Outer-authored review of the exact non-mutating assessment output."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-review-draft.v1"
    ] = VALIDATION_COORDINATOR_REVIEW_DRAFT_SCHEMA_VERSION
    created_at: datetime
    disposition: Literal["accept", "reject", "revise", "retry", "stop", "cancelled"]
    findings: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_created_at(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("review created_at must include a timezone")
        return self


class EmbeddedValidationEvidenceIndex(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-validation-evidence-index.v1"
    ] = EMBEDDED_VALIDATION_EVIDENCE_INDEX_SCHEMA_VERSION
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[ContractArtifactReference, ...] = Field(min_length=1)
    proposals: tuple[ContractArtifactReference, ...] = ()


class EmbeddedValidationExecutionIndex(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-validation-execution-index.v1"
    ] = EMBEDDED_VALIDATION_EXECUTION_INDEX_SCHEMA_VERSION
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_id: str = Field(min_length=1)
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coordinator_decision: ContractArtifactReference
    execution_authorization: ContractArtifactReference | None = None
    execution_result: ContractArtifactReference | None = None
    canonical_assessment: ExecutionArtifactBinding | None = None
    authorized: bool


class EmbeddedValidationReceiptIndex(_FrozenModel):
    schema_version: Literal[
        "content-agent-workflows.embedded-validation-receipt-index.v2"
    ] = EMBEDDED_VALIDATION_RECEIPT_INDEX_SCHEMA_VERSION
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_assessment: ExecutionArtifactBinding
    coordinator_review: ContractArtifactReference
    coordinator_review_draft: ExecutionArtifactBinding
    decision_receipt: ContractArtifactReference
    terminal_receipt: ExecutionArtifactBinding
    receipt_status: str = Field(min_length=1)
    gate_dispositions: dict[
        ValidationGateName,
        Literal["pass", "fail", "waive", "defer", "not_evaluated"],
    ] = Field(default_factory=dict)


class ValidationTerminalReceipt(_FrozenModel):
    """Mode-neutral final Validation assessment/readback receipt."""

    schema_version: Literal[
        "content-agent-workflows.validation-terminal-receipt.v1"
    ] = VALIDATION_TERMINAL_RECEIPT_SCHEMA_VERSION
    mode: Literal["standalone", "embedded"]
    assessment_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_index: ExecutionArtifactBinding
    canonical_assessment: ExecutionArtifactBinding
    coordinator_review: ExecutionArtifactBinding
    gate_dispositions: dict[
        ValidationGateName,
        Literal["pass", "fail", "waive", "defer", "not_evaluated"],
    ]
    terminal_disposition: ValidationTerminalDisposition
    review_disposition: Literal[
        "accept", "reject", "revise", "retry", "stop", "cancelled"
    ]
    receipt_status: Literal["completed", "rejected"]
    publication_kind: Literal["validation_assessment"] = "validation_assessment"
    source_mutated: Literal[False] = False
    cleanup_disposition: Literal["not_required"] = "not_required"
    coordinator_preparation: ExecutionArtifactBinding | None = None
    coordinator_plan_patch: ExecutionArtifactBinding | None = None
    accepted_coordinator_plan: ExecutionArtifactBinding | None = None
    coordinator_execution: ExecutionArtifactBinding | None = None
    operation_index: ExecutionArtifactBinding | None = None
    coordinator_planning_agent_launched: bool = False
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_coordinator_chain(self) -> Self:
        coordinator_bindings = (
            self.coordinator_preparation,
            self.coordinator_plan_patch,
            self.accepted_coordinator_plan,
            self.coordinator_execution,
            self.operation_index,
        )
        if self.coordinator_planning_agent_launched != all(
            binding is not None for binding in coordinator_bindings
        ):
            raise ValueError(
                "planning-agent readback requires the complete coordinator chain"
            )
        if any(binding is not None for binding in coordinator_bindings) and not all(
            binding is not None for binding in coordinator_bindings
        ):
            raise ValueError("terminal receipt has a partial coordinator chain")
        return self


_INDEPENDENT_GATES: Final[tuple[ValidationGateName, ...]] = cast(
    tuple[ValidationGateName, ...],
    get_args(ValidationGateName),
)


def _gate_dispositions(
    assessment: ValidationCoordinatorAssessment,
) -> dict[
    ValidationGateName,
    Literal["pass", "fail", "waive", "defer", "not_evaluated"],
]:
    return {
        gate_name: next(
            (gate.disposition for gate in assessment.gates if gate.gate == gate_name),
            "not_evaluated",
        )
        for gate_name in _INDEPENDENT_GATES
    }


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise EmbeddedValidationAssessmentError(
            f"Validation evidence must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise EmbeddedValidationAssessmentError(
                f"Validation evidence must be a regular file: {resolved}"
            )
        return ExecutionArtifactBinding(
            path=str(resolved),
            sha256=file_sha256(resolved),
            size_bytes=resolved.stat().st_size,
        )
    except OSError as exc:
        raise EmbeddedValidationAssessmentError(
            f"Validation evidence is missing or unreadable: {candidate}: {exc}"
        ) from exc


def _load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid embedded Validation artifact {path}: {exc}"
        ) from exc


def _load_embedded_validation_receipt_index(
    path: Path,
) -> EmbeddedValidationReceiptIndex:
    if path.is_symlink():
        raise EmbeddedValidationAssessmentError(
            f"embedded Validation receipt index must not be a symlink: {path}"
        )
    try:
        payload = load_json(path)
    except (OSError, ValueError) as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid embedded Validation receipt index {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EmbeddedValidationAssessmentError(
            f"Invalid embedded Validation receipt index {path}: expected an object"
        )
    if payload.get("schema_version") == (
        _LEGACY_EMBEDDED_VALIDATION_RECEIPT_INDEX_SCHEMA_VERSION
    ):
        raise EmbeddedValidationAssessmentError(
            "Legacy embedded Validation receipt index v1 is intentionally "
            "invalidated because it lacks terminal-receipt and independent-gate "
            "bindings; create a fresh direct evidence chain and v2 receipt"
        )
    try:
        return EmbeddedValidationReceiptIndex.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid embedded Validation receipt index {path}: {exc}"
        ) from exc


def load_embedded_validation_run(output_dir: str | Path) -> ValidationWorkflowRun:
    """Load one terminal native Validation bundle without executing templates."""

    root = Path(output_dir).expanduser().resolve()
    request = _load_model(root / "validation_request.json", ValidationRequest)
    plan = _load_model(root / "validation_plan.json", ValidationPlan)
    result = _load_model(root / "validation_result.json", ValidationResult)
    checkpoint = _load_model(
        root / "validation_checkpoint.json", ValidationWorkflowCheckpoint
    )
    summary = load_json(root / "final_summary.json")
    if not isinstance(summary, dict) or summary.get("status") != "completed":
        raise EmbeddedValidationAssessmentError(
            "embedded assessment requires a completed native Validation bundle"
        )
    if checkpoint.cancellation_requested or any(
        record.accepted_result is None or record.state.value != "completed"
        for record in checkpoint.records
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded assessment requires complete accepted template records"
        )
    accepted_results = tuple(
        record.accepted_result.result
        for record in checkpoint.records
        if record.accepted_result is not None
    )
    native_result_is_consistent = (
        result.template_results == accepted_results
        and result.verdict == aggregate_validation_verdict(accepted_results)
    )
    if (
        checkpoint.workflow_identity.request_digest != canonical_json_digest(request)
        or checkpoint.plan_digest != canonical_json_digest(plan)
        or result.request != request
        or result.plan != plan
        or not (
            native_result_is_consistent
            or _matches_policy_finalized_result(
                result,
                accepted_results=accepted_results,
            )
        )
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded assessment requires one internally consistent native "
            "Validation bundle"
        )
    return ValidationWorkflowRun(
        status=ValidationWorkflowStatus.COMPLETED,
        output_dir=str(root),
        request=request,
        plan=plan,
        result=result,
        checkpoint=checkpoint,
        request_path=str(root / "validation_request.json"),
        plan_path=str(root / "validation_plan.json"),
        result_path=str(root / "validation_result.json"),
        checkpoint_path=str(root / "validation_checkpoint.json"),
        evidence_path=str(root / "validation_evidence.json"),
        final_summary_path=str(root / "final_summary.json"),
    )


def _matches_policy_finalized_result(
    result: ValidationResult,
    *,
    accepted_results: tuple[ValidationTemplateResult, ...],
) -> bool:
    """Re-derive an explicitly documented policy transformation exactly.

    Checkpoint template results remain the raw factual evidence. The published
    result may differ only when the same trusted finalizer deterministically
    applies a matched expected-negative policy or a dependency-unavailable
    blocking policy from the digest-bound request.
    """

    expected_result = result.metadata.get("expected_result")
    gate_evaluation = result.metadata.get("gate_policy_evaluation")
    expected_negative_matched = (
        isinstance(expected_result, dict) and expected_result.get("matched") is True
    )
    dependency_unavailable_blocked = (
        isinstance(gate_evaluation, dict)
        and gate_evaluation.get("blocked") is True
        and gate_evaluation.get("reason") == "dependency_unavailable"
    )
    if not (expected_negative_matched or dependency_unavailable_blocked):
        return False

    metadata = dict(result.metadata)
    metadata.pop("expected_result", None)
    metadata.pop("gate_policy_evaluation", None)
    raw_result = result.model_copy(
        update={
            "verdict": aggregate_validation_verdict(accepted_results),
            "template_results": accepted_results,
            "issues": tuple(
                issue
                for template_result in accepted_results
                for issue in template_result.issues
            ),
            "metrics": {
                template_result.template_name: template_result.metrics
                for template_result in accepted_results
            },
            "evidence": {
                template_result.template_name: template_result.evidence
                for template_result in accepted_results
            },
            "metadata": metadata,
        }
    )
    try:
        rederived = finalize_validation_result(
            raw_result,
            request=result.request,
            artifact_paths=result.artifact_paths,
            runner="content-agent-workflows.validation",
        )
    except ValidationCliError:
        return False
    return rederived == result


def _verified_execution_binding(
    value: BaseModel, *, label: str
) -> ExecutionArtifactBinding:
    try:
        expected = ExecutionArtifactBinding.model_validate(
            value.model_dump(mode="json")
        )
    except ValidationError as exc:  # pragma: no cover - compatibility guard
        raise EmbeddedValidationAssessmentError(
            f"Invalid {label} binding: {exc}"
        ) from exc
    actual = _binding(expected.path)
    if actual != expected:
        raise EmbeddedValidationAssessmentError(f"{label} binding is stale")
    return actual


_EmbeddedCrossStageContext = tuple[
    ExecutionArtifactBinding,
    tuple[ExecutionArtifactBinding, ...],
    tuple[ExecutionArtifactBinding, ...],
    dict[str, str],
]


def _embedded_cross_stage_context(
    run: ValidationWorkflowRun,
    *,
    run_state_path: str | Path,
) -> _EmbeddedCrossStageContext:
    """Verify and bind upstream handoffs before semantic assessment."""

    # This dependency is intentionally embedded-only. Keeping it lazy lets the
    # public standalone Validation operations import without loading the asset
    # composition workflow or its coordinator surface.
    from content_agent_workflows.asset_composition import (
        ArtifactBinding,
        AssetCrossStageValidation,
        load_verified_run,
    )

    context = domain_execution_context_from_metadata(
        run.request.metadata,
        expected_domain="validation",
    )
    if context is None or context.embedded_stage is None:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation cross-stage evidence requires an outer stage binding"
        )
    outer = load_verified_run(Path(run_state_path).expanduser().resolve())
    if outer.run_id != context.embedded_stage.outer_run_id:
        raise EmbeddedValidationAssessmentError(
            "Validation execution context belongs to another outer run"
        )
    stage = outer.stages["validation"]
    if stage.input_asset is None:
        raise EmbeddedValidationAssessmentError(
            "outer Validation stage has no frozen input asset"
        )
    expected_input = _verified_execution_binding(
        stage.input_asset,
        label="Validation stage input",
    )
    if expected_input != context.embedded_stage.input_asset:
        raise EmbeddedValidationAssessmentError(
            "Validation stage input differs from its embedded execution context"
        )
    cross_stage_path = Path(run.output_dir).parent / "cross_stage_validation.json"
    cross_stage = _load_model(cross_stage_path, AssetCrossStageValidation)
    if cross_stage.run_id != outer.run_id:
        raise EmbeddedValidationAssessmentError(
            "cross-stage evidence belongs to another outer run"
        )
    if cross_stage.validation_input != stage.input_asset:
        raise EmbeddedValidationAssessmentError(
            "cross-stage evidence names a different Validation input"
        )
    handoffs: list[ExecutionArtifactBinding] = []
    expected_handoffs: dict[str, ArtifactBinding] = {}
    for name in ("articulation", "material", "texture", "physics"):
        handoff = outer.stages[name].handoff
        if handoff is None:
            raise EmbeddedValidationAssessmentError(
                f"outer {name} stage has no accepted handoff"
            )
        expected_handoffs[name] = handoff
        handoffs.append(_verified_execution_binding(handoff, label=f"{name} handoff"))
    if cross_stage.accepted_handoffs != expected_handoffs:
        raise EmbeddedValidationAssessmentError(
            "cross-stage evidence does not bind the accepted upstream handoffs"
        )
    claim_evidence: list[ExecutionArtifactBinding] = []
    seen: set[tuple[str, str]] = set()
    claim_statuses: dict[str, str] = {}
    for claim in cross_stage.claims:
        claim_statuses[claim.name] = claim.status
        for evidence in claim.evidence:
            verified = _verified_execution_binding(
                evidence,
                label=f"cross-stage {claim.name} evidence",
            )
            key = (verified.path, verified.sha256)
            if key not in seen:
                seen.add(key)
                claim_evidence.append(verified)
    return (
        _binding(cross_stage_path),
        tuple(handoffs),
        tuple(claim_evidence),
        claim_statuses,
    )


def _implementation_digests() -> dict[str, str]:
    import world_understanding.agentic.validation_scaffold as validation_scaffold
    import world_understanding.functions.cv.look_right as look_right
    import world_understanding.validation.scaffold_runner as scaffold_runner

    package_root = Path(__file__).resolve().parent
    workflows_root = package_root.parent
    components = {
        "validation_assessment_adapter": file_sha256(Path(__file__).resolve()),
        "validation_workflow": file_sha256(package_root / "workflow.py"),
        "validation_scaffold_runner": file_sha256(Path(scaffold_runner.__file__)),
        "validation_scaffold": file_sha256(Path(validation_scaffold.__file__)),
        "validation_look_right": file_sha256(Path(look_right.__file__)),
        "asset_coordinator": file_sha256(
            workflows_root / "asset_composition" / "coordinator.py"
        ),
    }
    components["validation_execution_stack"] = canonical_json_digest(
        cast(dict[str, JsonValue], components)
    )
    components["validation_look_right_stack"] = canonical_json_digest(
        {
            name: components[name]
            for name in (
                "validation_workflow",
                "validation_scaffold_runner",
                "validation_scaffold",
                "validation_look_right",
            )
        }
    )
    return components


def build_validation_decision_identity(
    run: ValidationWorkflowRun,
    *,
    run_state_path: str | Path,
    _cross_stage_context: _EmbeddedCrossStageContext | None = None,
) -> EmbeddedDecisionIdentity:
    """Rebuild the exact shared identity for one completed native bundle."""

    context = domain_execution_context_from_metadata(
        run.request.metadata,
        expected_domain="validation",
    )
    if context is None or context.mode != "embedded":
        raise EmbeddedValidationAssessmentError(
            "native Validation request is not explicitly embedded"
        )
    from content_agent_workflows.asset_composition import (
        build_embedded_domain_decision_identity,
    )

    if _cross_stage_context is None:
        _cross_stage_context = _embedded_cross_stage_context(
            run,
            run_state_path=run_state_path,
        )
    cross_stage, handoffs, _, _ = _cross_stage_context

    template_capability = canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.validation-capability.v1",
            "templates": dict(run.checkpoint.workflow_identity.template_versions),
            "ordered_templates": [
                record.template_name for record in run.checkpoint.records
            ],
            "assessment_schema": VALIDATION_COORDINATOR_ASSESSMENT_SCHEMA_VERSION,
        }
    )
    configuration_digests = {
        "validation_workflow_identity": (
            run.checkpoint.workflow_identity.identity_digest
        ),
        "validation_plan": run.checkpoint.plan_digest,
        "validation_result": file_sha256(Path(run.result_path)),
        "validation_evidence": file_sha256(Path(run.evidence_path)),
        "validation_cross_stage": cross_stage.sha256,
        **{
            f"upstream_handoff_{name}": binding.sha256
            for name, binding in zip(
                ("articulation", "material", "texture", "physics"),
                handoffs,
                strict=True,
            )
        },
    }
    from .operations import (
        VALIDATION_OPERATION_INDEX_NAME,
        load_validated_validation_operation_index,
    )

    operation_index_path = Path(run.output_dir) / VALIDATION_OPERATION_INDEX_NAME
    focused_operations = (
        run.result.metadata.get("execution_mode") == "outer_selected_operations"
    )
    if operation_index_path.is_symlink():
        raise EmbeddedValidationAssessmentError(
            "Invalid embedded Validation operation evidence: operation index "
            f"must not be a symlink: {operation_index_path}"
        )
    if focused_operations or operation_index_path.exists():
        from .workflow import ValidationWorkflowError

        try:
            load_validated_validation_operation_index(run.output_dir)
        except ValidationWorkflowError as exc:
            raise EmbeddedValidationAssessmentError(
                f"Invalid embedded Validation operation evidence: {exc}"
            ) from exc
        configuration_digests["validation_operation_index"] = _binding(
            operation_index_path
        ).sha256
    identity = build_embedded_domain_decision_identity(
        run_state_path,
        domain="validation",
        input_asset=run.request.inputs[0],
        output_dir=run.output_dir,
        capability_digests={"validation": template_capability},
        implementation_digests=_implementation_digests(),
        configuration_digests=configuration_digests,
    )
    if identity.execution_context != context:
        raise EmbeddedValidationAssessmentError(
            "persisted Validation execution context is stale relative to asset state"
        )
    return identity


def _template_verdict(result: ValidationTemplateResult) -> ValidationEvidenceVerdict:
    verdicts: dict[str, ValidationEvidenceVerdict] = {
        "passed": "pass",
        "warn": "warn",
        "failed": "fail",
        "needs_refinement": "needs_remediation",
        "error": "error",
        "skipped": "unsupported",
    }
    return verdicts[result.status]


def _record_artifacts(
    run: ValidationWorkflowRun,
    *,
    template_name: str,
) -> tuple[ExecutionArtifactBinding, ...]:
    paths = {
        run.result_path,
        run.evidence_path,
        run.final_summary_path,
        run.checkpoint_path,
    }
    record = next(
        item for item in run.checkpoint.records if item.template_name == template_name
    )
    accepted = record.accepted_result
    if accepted is None:  # pragma: no cover - terminal bundle guard
        raise EmbeddedValidationAssessmentError(
            f"Validation template {template_name} has no accepted result"
        )
    paths.add(accepted.result_path)
    paths.update(
        artifact.path
        for artifact in accepted.evidence_artifacts
        if artifact.kind == "file"
    )
    return tuple(_binding(path) for path in sorted(paths))


def _focused_operation_requirements(
    run: ValidationWorkflowRun,
) -> dict[str, bool] | None:
    from .operations import (
        VALIDATION_OPERATION_INDEX_NAME,
        load_validated_validation_operation_index,
    )
    from .workflow import ValidationWorkflowError

    operation_index_path = Path(run.output_dir) / VALIDATION_OPERATION_INDEX_NAME
    if not (operation_index_path.exists() or operation_index_path.is_symlink()):
        return None
    try:
        operation_index = load_validated_validation_operation_index(run.output_dir)
    except ValidationWorkflowError as exc:
        raise EmbeddedValidationAssessmentError(
            f"Invalid focused Validation operation evidence: {exc}"
        ) from exc
    return {item.template_name: item.mandatory for item in operation_index.operations}


def _template_evidence_record(
    run: ValidationWorkflowRun,
    result: ValidationTemplateResult,
    *,
    required: bool,
) -> ProviderNeutralEvidenceRecord:
    gate = _TEMPLATE_GATE.get(result.template_name)
    if gate is None:
        raise EmbeddedValidationAssessmentError(
            f"embedded Validation does not recognize template {result.template_name!r}"
        )
    verdict = _template_verdict(result)
    status: EvidenceStatus = (
        "error"
        if verdict == "error"
        else "unsupported"
        if verdict == "unsupported"
        else "available"
    )
    facts: dict[str, Any] = {}
    if status == "available":
        facts = {
            "gate": gate,
            "template_name": result.template_name,
            "tool_identity": {
                "name": result.template_name,
                "version": run.checkpoint.workflow_identity.template_versions[
                    result.template_name
                ],
            },
            "profile_identity": {
                "request_digest": run.checkpoint.workflow_identity.request_digest,
                "policy_digest": run.checkpoint.workflow_identity.policy_digest,
                "backend_digest": run.checkpoint.workflow_identity.backend_digest,
            },
        }
        if result.template_name == "look_right":
            facts.update(
                {
                    "evidence_role": "critique_proposal_available",
                    "execution_status": result.status,
                    "issue_codes": [issue.code for issue in result.issues],
                }
            )
        else:
            facts.update(
                {
                    "verdict": verdict,
                    "issue_codes": [issue.code for issue in result.issues],
                    "raw_template_result": result.model_dump(mode="json"),
                }
            )
    return ProviderNeutralEvidenceRecord(
        evidence_id=f"validation-template-{result.template_name}",
        evidence_type=("critique" if gate == "visual_quality" else "validation"),
        status=status,
        required=required,
        summary=(
            f"{result.template_name} produced separate {gate} evidence with "
            f"native status {result.status}."
        ),
        artifacts=_record_artifacts(run, template_name=result.template_name),
        facts=facts,
    )


def _template_evidence_records(
    run: ValidationWorkflowRun,
    result: ValidationTemplateResult,
    *,
    required: bool | None = None,
) -> tuple[ProviderNeutralEvidenceRecord, ...]:
    """Return the required outcome plus available deterministic report provenance."""

    outcome = _template_evidence_record(
        run,
        result,
        required=(
            result.template_name != "look_right" if required is None else required
        ),
    )
    if outcome.status == "available":
        return (outcome,)
    gate = _TEMPLATE_GATE[result.template_name]
    report = ProviderNeutralEvidenceRecord(
        evidence_id=f"{outcome.evidence_id}-report",
        evidence_type="artifact",
        status="available",
        required=False,
        summary=(
            f"Raw {result.template_name} report and exact tool/profile identity; "
            f"the required outcome remains {outcome.status}."
        ),
        artifacts=outcome.artifacts,
        facts={
            "gate": gate,
            "template_name": result.template_name,
            "tool_identity": {
                "name": result.template_name,
                "version": run.checkpoint.workflow_identity.template_versions[
                    result.template_name
                ],
            },
            "profile_identity": {
                "request_digest": run.checkpoint.workflow_identity.request_digest,
                "policy_digest": run.checkpoint.workflow_identity.policy_digest,
                "backend_digest": run.checkpoint.workflow_identity.backend_digest,
            },
            "issue_codes": [issue.code for issue in result.issues],
            "raw_template_result": result.model_dump(mode="json"),
        },
    )
    return outcome, report


VALIDATION_REFERENCE_IMAGE_SUFFIXES: Final = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def _verified_validation_artifact(
    artifact: ValidationArtifactIdentity,
    *,
    label: str,
) -> None:
    candidate = Path(artifact.path)
    if candidate.is_symlink() or artifact.kind == "missing":
        raise EmbeddedValidationAssessmentError(f"{label} is missing or unsafe")
    if artifact.kind == "external":
        # Model validation re-derives the canonical declaration digest. The
        # containing source bytes bind where the runtime module was authored;
        # renderer metadata binds the runtime that resolves it.
        return
    if artifact.kind == "file":
        if not candidate.is_file() or file_sha256(candidate) != artifact.sha256:
            raise EmbeddedValidationAssessmentError(f"{label} binding is stale")
        return
    if not candidate.is_dir() or artifact_set_digest((candidate,)) != artifact.sha256:
        raise EmbeddedValidationAssessmentError(f"{label} directory binding is stale")


def _package_evidence_record(
    run: ValidationWorkflowRun,
) -> ProviderNeutralEvidenceRecord:
    try:
        payload = load_json(Path(run.evidence_path))
    except (OSError, ValueError) as exc:
        raise EmbeddedValidationAssessmentError(
            f"Validation package evidence is missing or malformed: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EmbeddedValidationAssessmentError(
            "Validation package evidence must be a JSON object"
        )
    expected_sources = [
        artifact.model_dump(mode="json")
        for artifact in run.checkpoint.workflow_identity.source_artifacts
    ]
    expected_templates = {
        record.template_name: record.accepted_result
        for record in run.checkpoint.records
        if record.accepted_result is not None
    }
    if (
        payload.get("schema_version") != VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION
        or payload.get("workflow_identity_digest")
        != run.checkpoint.workflow_identity.identity_digest
        or payload.get("plan_digest") != run.checkpoint.plan_digest
        or payload.get("source_before") != expected_sources
        or not isinstance(payload.get("source_after"), list)
        or not isinstance(payload.get("templates"), dict)
        or set(payload["templates"]) != set(expected_templates)
    ):
        raise EmbeddedValidationAssessmentError(
            "Validation package evidence differs from the accepted native bundle"
        )
    source_after = tuple(
        ValidationArtifactIdentity.model_validate(item)
        for item in payload["source_after"]
    )
    for artifact in source_after:
        _verified_validation_artifact(artifact, label="Validation source readback")
    for template_name, accepted in expected_templates.items():
        template_payload = payload["templates"][template_name]
        expected_payload = {
            "status": accepted.result.status,
            "result_path": accepted.result_path,
            "result_sha256": accepted.result_sha256,
            "evidence_artifacts": [
                artifact.model_dump(mode="json")
                for artifact in accepted.evidence_artifacts
            ],
        }
        if template_payload != expected_payload:
            raise EmbeddedValidationAssessmentError(
                f"Validation package evidence for {template_name} is stale"
            )
        if _binding(accepted.result_path).sha256 != accepted.result_sha256:
            raise EmbeddedValidationAssessmentError(
                f"Validation accepted result for {template_name} is stale"
            )
        for artifact in accepted.evidence_artifacts:
            _verified_validation_artifact(
                artifact,
                label=f"Validation {template_name} evidence",
            )
    source_unchanged = (
        tuple(run.checkpoint.workflow_identity.source_artifacts) == source_after
        and payload.get("source_unchanged") is True
    )
    artifacts = tuple(
        _binding(path)
        for path in (
            run.request_path,
            run.plan_path,
            run.result_path,
            run.checkpoint_path,
            run.evidence_path,
            run.final_summary_path,
        )
    )
    return ProviderNeutralEvidenceRecord(
        evidence_id="validation-package-integrity",
        evidence_type="artifact",
        status="available",
        required=True,
        summary="Saved native Validation bundle and source readback integrity.",
        artifacts=artifacts,
        facts={
            "gate": "package_integrity",
            "verdict": "pass" if source_unchanged else "fail",
            "source_unchanged": source_unchanged,
            "saved_artifact_readback": True,
        },
    )


def _image_bindings_from_identity(
    artifact: ValidationArtifactIdentity,
) -> tuple[ExecutionArtifactBinding, ...]:
    _verified_validation_artifact(artifact, label="Validation visual evidence")
    path = Path(artifact.path)
    if artifact.kind == "file":
        return (
            (_binding(path),)
            if path.suffix.lower() in VALIDATION_REFERENCE_IMAGE_SUFFIXES
            else ()
        )
    return tuple(
        _binding(candidate)
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and candidate.suffix.lower() in VALIDATION_REFERENCE_IMAGE_SUFFIXES
    )


def _visual_evidence_record(
    run: ValidationWorkflowRun,
    *,
    required: bool | None = None,
) -> ProviderNeutralEvidenceRecord:
    render_record = next(
        (
            record
            for record in run.checkpoint.records
            if record.template_name == "render_valid"
        ),
        None,
    )
    render_images: list[ExecutionArtifactBinding] = []
    if render_record is not None and render_record.accepted_result is not None:
        for artifact in render_record.accepted_result.evidence_artifacts:
            render_images.extend(_image_bindings_from_identity(artifact))
    reference_images: list[ExecutionArtifactBinding] = []
    for artifact in run.checkpoint.workflow_identity.reference_artifacts:
        reference_images.extend(_image_bindings_from_identity(artifact))
    unique_artifacts: dict[tuple[str, str], ExecutionArtifactBinding] = {}
    for binding in (*render_images, *reference_images):
        unique_artifacts[(binding.path, binding.sha256)] = binding
    if required is None:
        required = render_record is not None
    if not render_images:
        return ProviderNeutralEvidenceRecord(
            evidence_id="validation-outer-visual-evidence",
            evidence_type="render",
            status="unavailable" if required else "unsupported",
            required=required,
            summary=(
                "Outer visual assessment requires current render images."
                if required
                else "Visual evidence was not requested by the frozen plan."
            ),
            artifacts=tuple(unique_artifacts.values()),
        )
    return ProviderNeutralEvidenceRecord(
        evidence_id="validation-outer-visual-evidence",
        evidence_type="render",
        status="available",
        required=required,
        summary=(
            "Digest-bound render and reference images for direct outer-model "
            "inspection; no semantic verdict is asserted by deterministic code."
        ),
        artifacts=tuple(unique_artifacts.values()),
        facts={
            "gate": "visual_quality",
            "semantic_authority": "outer_coordinator",
            "render_images": [item.model_dump(mode="json") for item in render_images],
            "reference_images": [
                item.model_dump(mode="json") for item in reference_images
            ],
        },
    )


def _cross_stage_evidence_record(
    context: _EmbeddedCrossStageContext,
) -> ProviderNeutralEvidenceRecord:
    cross_stage, handoffs, claim_evidence, claim_statuses = context
    return ProviderNeutralEvidenceRecord(
        evidence_id="validation-cross-stage-integrity",
        evidence_type="validation",
        status="available",
        required=True,
        summary="Verified upstream handoffs and coordinator-authored cross-stage claims.",
        artifacts=(cross_stage, *handoffs, *claim_evidence),
        facts={
            "gate": "cross_stage_integrity",
            "verdict": ("warn" if "warn" in set(claim_statuses.values()) else "pass"),
            "claim_statuses": cast(dict[str, JsonValue], claim_statuses),
            "upstream_handoffs": [item.model_dump(mode="json") for item in handoffs],
            "cross_stage_receipt": cross_stage.model_dump(mode="json"),
        },
    )


def _write_index(path: Path, value: BaseModel) -> None:
    if path.is_symlink():
        raise EmbeddedValidationAssessmentError(
            f"embedded Validation index must not be a symlink: {path}"
        )
    if path.exists():
        try:
            existing = value.__class__.model_validate(load_json(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise EmbeddedValidationAssessmentError(
                f"Invalid existing embedded Validation index {path}: {exc}"
            ) from exc
        if existing != value:
            raise EmbeddedValidationAssessmentError(
                f"embedded Validation index already differs: {path}"
            )
        return
    atomic_write_json(path, value)


def prepare_embedded_validation_evidence(
    run: ValidationWorkflowRun,
    *,
    run_state_path: str | Path,
) -> EmbeddedValidationEvidenceIndex:
    """Persist native factual evidence and optional judge critique proposal."""

    cross_stage_context = _embedded_cross_stage_context(
        run,
        run_state_path=run_state_path,
    )
    identity = build_validation_decision_identity(
        run,
        run_state_path=run_state_path,
        _cross_stage_context=cross_stage_context,
    )
    implementation_digests = _implementation_digests()
    provider = ProducerIdentity(
        producer_id="validation-template-workflow",
        role="evidence_provider",
        implementation="content_agent_workflows.validation.execution_stack",
        implementation_digest=implementation_digests["validation_execution_stack"],
    )
    focused_requirements = _focused_operation_requirements(run)
    if focused_requirements is not None:
        missing_dispositions = tuple(
            result.template_name
            for result in run.result.template_results
            if result.template_name not in focused_requirements
        )
        if missing_dispositions:
            raise EmbeddedValidationAssessmentError(
                "focused Validation result has no operation-index disposition: "
                + ", ".join(missing_dispositions)
            )
    template_record_groups = tuple(
        _template_evidence_records(
            run,
            result,
            required=(
                focused_requirements[result.template_name]
                if focused_requirements is not None
                else result.template_name != "look_right"
            ),
        )
        for result in run.result.template_results
    )
    if not template_record_groups:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation requires at least one factual template result"
        )
    visual_required = (
        focused_requirements.get("render_valid", False)
        if focused_requirements is not None
        else any(
            result.template_name == "render_valid"
            for result in run.result.template_results
        )
    )
    record_groups = (
        *template_record_groups,
        (_visual_evidence_record(run, required=visual_required),),
        (_package_evidence_record(run),),
        (_cross_stage_evidence_record(cross_stage_context),),
    )
    store = EmbeddedDecisionArtifactStore(run.output_dir)
    evidence = tuple(
        EmbeddedDomainEvidence(
            artifact_id=(
                "validation-"
                f"{records[0].facts.get('gate', 'unavailable')}-evidence-"
                f"{records[0].evidence_id}-"
                + run.checkpoint.workflow_identity.identity_digest[:16]
            ),
            identity=identity,
            producer=provider,
            parent_artifact=identity.coordinator_plan,
            created_at=run.checkpoint.updated_at,
            records=records,
        )
        for records in record_groups
    )
    for artifact in evidence:
        store.append(artifact)
    proposals: list[EmbeddedDomainProposal] = []
    look_right = next(
        (
            result
            for result in run.result.template_results
            if result.template_name == "look_right"
        ),
        None,
    )
    if look_right is not None and look_right.metadata.get("selection_state") != (
        "not_evaluated"
    ):
        proposal_provider = ProducerIdentity(
            producer_id="validation-look-right-judge",
            role="proposal_provider",
            implementation="world_understanding.validation.look_right_stack",
            implementation_digest=implementation_digests["validation_look_right_stack"],
        )
        payload = DomainProposalPayload(
            schema_version="content-agent-workflows.validation-critique-proposal.v1",
            values={
                "gate": "visual_quality",
                "template_name": "look_right",
                "proposed_verdict": _template_verdict(look_right),
                "critique": look_right.model_dump(mode="json"),
            },
        )
        look_right_evidence = next(
            artifact
            for artifact in evidence
            if artifact.records[0].evidence_id == "validation-template-look_right"
        )
        proposal = EmbeddedDomainProposal(
            artifact_id=(
                "validation-look-right-proposal-"
                + run.checkpoint.workflow_identity.identity_digest[:24]
            ),
            identity=identity,
            producer=proposal_provider,
            parent_artifact=artifact_reference(look_right_evidence),
            created_at=run.checkpoint.updated_at,
            evidence_artifacts=(artifact_reference(look_right_evidence),),
            proposal=payload,
            proposal_digest=canonical_json_digest(payload),
        )
        store.append(proposal)
        proposals.append(proposal)
    index = EmbeddedValidationEvidenceIndex(
        identity_sha256=canonical_json_digest(identity),
        evidence=tuple(artifact_reference(item) for item in evidence),
        proposals=tuple(artifact_reference(item) for item in proposals),
    )
    _write_index(
        Path(run.output_dir) / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        index,
    )
    return index


def _load_evidence_chain(
    run: ValidationWorkflowRun,
    identity: EmbeddedDecisionIdentity,
) -> tuple[
    EmbeddedDecisionArtifactStore,
    EmbeddedValidationEvidenceIndex,
    tuple[EmbeddedDomainEvidence, ...],
    tuple[EmbeddedDomainProposal, ...],
]:
    root = Path(run.output_dir)
    index = _load_model(
        root / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        EmbeddedValidationEvidenceIndex,
    )
    if index.identity_sha256 != canonical_json_digest(identity):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation evidence identity is stale"
        )
    store = EmbeddedDecisionArtifactStore(root)
    evidence = tuple(
        store.load_typed(reference, EmbeddedDomainEvidence)
        for reference in index.evidence
    )
    proposals = tuple(
        store.load_typed(reference, EmbeddedDomainProposal)
        for reference in index.proposals
    )
    return store, index, evidence, proposals


def _assessment_evidence_records(
    evidence: tuple[EmbeddedDomainEvidence, ...]
    | tuple[ProviderNeutralEvidenceRecord, ...],
) -> dict[str, ProviderNeutralEvidenceRecord]:
    records: dict[str, ProviderNeutralEvidenceRecord] = {}
    for artifact in evidence:
        artifact_records = (
            (artifact,)
            if isinstance(artifact, ProviderNeutralEvidenceRecord)
            else artifact.records
        )
        for record in artifact_records:
            if record.evidence_id in records:
                raise EmbeddedValidationAssessmentError(
                    f"duplicate Validation evidence ID: {record.evidence_id}"
                )
            records[record.evidence_id] = record
    return records


def validate_coordinator_assessment(
    assessment: ValidationCoordinatorAssessment,
    *,
    evidence: tuple[EmbeddedDomainEvidence, ...]
    | tuple[ProviderNeutralEvidenceRecord, ...],
    proposals: tuple[EmbeddedDomainProposal, ...] = (),
) -> ValidationCoordinatorAssessment:
    """Fail closed when required factual evidence is unresolved or contradicted."""

    records = _assessment_evidence_records(evidence)
    assessed_ids = {
        evidence_id for gate in assessment.gates for evidence_id in gate.evidence_ids
    }
    required_ids = {
        evidence_id for evidence_id, record in records.items() if record.required
    }
    missing = sorted(required_ids - assessed_ids)
    unknown = sorted(assessed_ids - set(records))
    if missing or unknown:
        raise EmbeddedValidationAssessmentError(
            "assessment evidence coverage is incomplete or unknown; "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    # Advisory proposal issue codes are deliberately not required findings.
    # The outer reasoner may cite and disposition them, but optional critique
    # cannot become mandatory canonical authority merely because it exists.
    _ = proposals
    issue_codes: set[str] = set()
    passing = assessment.terminal_disposition == "pass"
    for gate in assessment.gates:
        gate_records = [records[evidence_id] for evidence_id in gate.evidence_ids]
        if not gate.required and any(record.required for record in gate_records):
            raise EmbeddedValidationAssessmentError(
                f"required evidence cannot be downgraded for gate {gate.gate}"
            )
        for record in gate_records:
            facts_gate = record.facts.get("gate")
            if record.status == "available" and facts_gate != gate.gate:
                raise EmbeddedValidationAssessmentError(
                    f"evidence {record.evidence_id} belongs to {facts_gate!r}, "
                    f"not {gate.gate!r}"
                )
            raw_codes = record.facts.get("issue_codes", ())
            if isinstance(raw_codes, tuple | list):
                issue_codes.update(
                    code for code in raw_codes if isinstance(code, str) and code
                )
        if not passing or not gate.required:
            continue
        required_gate_records = [record for record in gate_records if record.required]
        if gate.gate == "visual_quality" and not any(
            record.evidence_type == "render"
            and record.status == "available"
            and bool(record.facts.get("render_images"))
            for record in required_gate_records
        ):
            raise EmbeddedValidationAssessmentError(
                "required visual_quality gate cannot pass without available "
                "required current-render evidence"
            )
        if not required_gate_records:
            raise EmbeddedValidationAssessmentError(
                f"required gate {gate.gate} must cite required evidence"
            )
        unavailable = [
            record.evidence_id
            for record in required_gate_records
            if record.status != "available"
        ]
        verdicts = {
            record.facts.get("verdict")
            for record in required_gate_records
            if record.status == "available"
        }
        blocking = verdicts & {
            "fail",
            "needs_remediation",
            "error",
            "unsupported",
        }
        contradictory = "pass" in verdicts and bool(
            verdicts & {"fail", "needs_remediation"}
        )
        if unavailable or blocking or contradictory:
            raise EmbeddedValidationAssessmentError(
                f"required gate {gate.gate} cannot pass; unavailable={unavailable!r}, "
                f"verdicts={sorted(str(item) for item in verdicts)!r}"
            )
        if "warn" in verdicts and gate.disposition != "waive":
            raise EmbeddedValidationAssessmentError(
                f"required gate {gate.gate} has warnings without an explicit waiver"
            )
        if (
            gate.gate != "visual_quality"
            and gate.disposition == "pass"
            and verdicts != {"pass"}
        ):
            raise EmbeddedValidationAssessmentError(
                f"required gate {gate.gate} is not an explicit factual pass"
            )
    covered_issue_codes = {
        code for finding in assessment.findings for code in finding.source_issue_codes
    }
    if issue_codes - covered_issue_codes:
        raise EmbeddedValidationAssessmentError(
            "assessment leaves native issue codes unresolved: "
            + ", ".join(sorted(issue_codes - covered_issue_codes))
        )
    for finding in assessment.findings:
        unknown_sources = set(finding.source_evidence_ids) - set(records)
        if unknown_sources:
            raise EmbeddedValidationAssessmentError(
                f"finding {finding.finding_id} cites unknown evidence: "
                + ", ".join(sorted(unknown_sources))
            )
        source_codes: set[str] = set()
        for evidence_id in finding.source_evidence_ids:
            raw_codes = records[evidence_id].facts.get("issue_codes", ())
            if isinstance(raw_codes, tuple | list):
                source_codes.update(
                    code for code in raw_codes if isinstance(code, str) and code
                )
        unknown_codes = set(finding.source_issue_codes) - source_codes
        if unknown_codes:
            raise EmbeddedValidationAssessmentError(
                f"finding {finding.finding_id} cites issue codes absent from its "
                "source evidence: " + ", ".join(sorted(unknown_codes))
            )
    return assessment


def _coordinator(identity: EmbeddedDecisionIdentity) -> ProducerIdentity:
    return ProducerIdentity(
        producer_id="asset-coordinator",
        role="outer_coordinator",
        implementation="content_agent_workflows.asset_composition.coordinator",
        implementation_digest=identity.digests.implementations["asset_coordinator"],
    )


def _executor(identity: EmbeddedDecisionIdentity) -> ProducerIdentity:
    return ProducerIdentity(
        producer_id="validation-assessment-finalizer",
        role="executor",
        implementation="content_agent_workflows.validation.embedded_assessment",
        implementation_digest=identity.digests.implementations[
            "validation_assessment_adapter"
        ],
    )


def _assessment_from_decision(
    decision: EmbeddedCoordinatorDecision,
) -> ValidationCoordinatorAssessment:
    if decision.accepted_decision is None:
        raise EmbeddedValidationAssessmentError(
            "coordinator decision does not contain an accepted assessment"
        )
    try:
        return ValidationCoordinatorAssessment.model_validate(
            dict(decision.accepted_decision.values)
        )
    except ValidationError as exc:
        raise EmbeddedValidationAssessmentError(
            f"accepted Validation assessment is malformed: {exc}"
        ) from exc


def _publish_canonical_assessment(
    authorization: BoundedExecutionAuthorization,
    *,
    output_dir: Path,
    decision: EmbeddedCoordinatorDecision,
    evidence: tuple[EmbeddedDomainEvidence, ...],
    proposals: tuple[EmbeddedDomainProposal, ...],
) -> ExecutionArtifactBinding:
    if authorization.execution_effect != "non_mutating" or (
        authorization.mutation_id is not None
    ):
        raise EmbeddedValidationAssessmentError(
            "Validation assessment finalization must be non-mutating"
        )
    assessment = validate_coordinator_assessment(
        _assessment_from_decision(decision),
        evidence=evidence,
        proposals=proposals,
    )
    path = output_dir / CANONICAL_VALIDATION_ASSESSMENT_NAME
    if path.exists():
        existing = _load_model(path, ValidationCoordinatorAssessment)
        if existing != assessment:
            raise EmbeddedValidationAssessmentError(
                "canonical Validation assessment already differs"
            )
    else:
        atomic_write_json(path, assessment)
    return _binding(path)


def assess_embedded_validation(
    output_dir: str | Path,
    *,
    run_state_path: str | Path,
    assessment_path: str | Path,
) -> EmbeddedValidationExecutionIndex:
    """Persist an outer assessment and run its authorized non-mutating finalizer."""

    run = load_embedded_validation_run(output_dir)
    identity = build_validation_decision_identity(run, run_state_path=run_state_path)
    store, index, evidence, proposals = _load_evidence_chain(run, identity)
    assessment = _load_model(
        Path(assessment_path).expanduser().resolve(),
        ValidationCoordinatorAssessment,
    )
    validate_coordinator_assessment(
        assessment,
        evidence=evidence,
        proposals=proposals,
    )
    assessment_sha256 = canonical_json_digest(assessment)
    root = Path(run.output_dir)
    execution_index_path = root / EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME
    if execution_index_path.exists():
        existing = _load_model(
            execution_index_path,
            EmbeddedValidationExecutionIndex,
        )
        if (
            existing.identity_sha256 != index.identity_sha256
            or existing.assessment_id != assessment.assessment_id
            or existing.assessment_sha256 != assessment_sha256
        ):
            raise EmbeddedValidationAssessmentError(
                "existing embedded Validation execution belongs to another assessment"
            )
        return existing
    owner = _coordinator(identity)
    accepted = AcceptedSemanticDecision(
        schema_version=VALIDATION_COORDINATOR_ASSESSMENT_SCHEMA_VERSION,
        values=assessment.model_dump(mode="json"),
    )
    disposition: CoordinatorDecisionDisposition = (
        "accept"
        if assessment.terminal_disposition == "pass"
        else "reject"
        if assessment.terminal_disposition == "fail"
        else "revise"
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id=f"validation-assessment-{assessment.assessment_id}",
        identity=identity,
        producer=owner,
        parent_artifact=(
            index.proposals[-1] if index.proposals else index.evidence[-1]
        ),
        created_at=assessment.created_at,
        disposition=disposition,
        evidence_artifacts=index.evidence,
        proposal_artifacts=index.proposals,
        accepted_decision=accepted if disposition == "accept" else None,
        accepted_decision_digest=(
            accepted_semantic_decision_digest(identity, accepted)
            if disposition == "accept"
            else None
        ),
        human_decision_required=False,
        rationale=assessment.summary,
        revision_requests=(
            (
                "Resolve required Validation evidence and author a new assessment "
                "in a new outer stage attempt."
            ),
        )
        if disposition == "revise"
        else (),
    )
    store.append(decision)
    if disposition != "accept":
        execution_index = EmbeddedValidationExecutionIndex(
            identity_sha256=index.identity_sha256,
            assessment_id=assessment.assessment_id,
            assessment_sha256=assessment_sha256,
            coordinator_decision=artifact_reference(decision),
            authorized=False,
        )
        _write_index(execution_index_path, execution_index)
        return execution_index
    executor = _executor(identity)
    authorization = authorize_bounded_execution(
        decision,
        expected_identity=identity,
        evidence=evidence,
        existing_authorizations=(),
        proposals=proposals,
        executor=executor,
        execution_effect="non_mutating",
        created_at=assessment.created_at,
    )
    try:
        output = store.invoke_after_authorization_commit(
            authorization,
            lambda committed: _publish_canonical_assessment(
                committed,
                output_dir=root,
                decision=decision,
                evidence=evidence,
                proposals=proposals,
            ),
        )
        result = EmbeddedBoundedExecutionResult(
            artifact_id=f"validation-assessment-result-{authorization.operation_id}",
            identity=identity,
            producer=executor,
            parent_artifact=artifact_reference(authorization),
            created_at=authorization.created_at,
            accepted_decision=artifact_reference(decision),
            accepted_decision_digest=authorization.accepted_decision_digest,
            execution_effect="non_mutating",
            operation_id=authorization.operation_id,
            mutation_id=None,
            attempt=authorization.attempt,
            status="succeeded",
            mutation_state="not_applicable",
            outputs=(output,),
            evidence=(
                ProviderNeutralEvidenceRecord(
                    evidence_id="canonical-validation-assessment",
                    evidence_type="validation",
                    status="available",
                    required=True,
                    summary="Canonical outer-coordinator Validation assessment.",
                    artifacts=(output,),
                    facts={
                        "assessment_id": assessment.assessment_id,
                        "terminal_disposition": assessment.terminal_disposition,
                    },
                ),
            ),
        )
    except EmbeddedDecisionAuthorizationReplayError as exc:
        raise EmbeddedValidationAssessmentError(
            "Validation finalizer authorization was already committed without a "
            "completed execution index; reconciliation is required"
        ) from exc
    except Exception as exc:
        result = EmbeddedBoundedExecutionResult(
            artifact_id=f"validation-assessment-result-{authorization.operation_id}",
            identity=identity,
            producer=executor,
            parent_artifact=artifact_reference(authorization),
            created_at=authorization.created_at,
            accepted_decision=artifact_reference(decision),
            accepted_decision_digest=authorization.accepted_decision_digest,
            execution_effect="non_mutating",
            operation_id=authorization.operation_id,
            mutation_id=None,
            attempt=authorization.attempt,
            status="failed",
            mutation_state="not_started",
            error=str(exc),
        )
        store.append_result(result)
        raise EmbeddedValidationAssessmentError(
            f"Validation assessment finalizer failed after authorization commit: {exc}"
        ) from exc
    validate_bounded_execution_result(result, authorization)
    store.append_result(result)
    execution_index = EmbeddedValidationExecutionIndex(
        identity_sha256=index.identity_sha256,
        assessment_id=assessment.assessment_id,
        assessment_sha256=assessment_sha256,
        coordinator_decision=artifact_reference(decision),
        execution_authorization=artifact_reference(authorization),
        execution_result=artifact_reference(result),
        canonical_assessment=output,
        authorized=True,
    )
    _write_index(execution_index_path, execution_index)
    return execution_index


def review_embedded_validation_assessment(
    output_dir: str | Path,
    *,
    run_state_path: str | Path,
    review_path: str | Path,
) -> EmbeddedValidationReceiptIndex:
    """Persist an outer review and seal the exact shared decision receipt."""

    run = load_embedded_validation_run(output_dir)
    identity = build_validation_decision_identity(run, run_state_path=run_state_path)
    store, evidence_index, evidence, proposals = _load_evidence_chain(run, identity)
    root = Path(run.output_dir)
    review_draft = _load_model(
        Path(review_path).expanduser().resolve(),
        ValidationCoordinatorReviewDraft,
    )
    review_sha256 = canonical_json_digest(review_draft)
    receipt_index_path = root / EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME
    if receipt_index_path.exists():
        existing = _load_embedded_validation_receipt_index(receipt_index_path)
        if (
            existing.identity_sha256 != evidence_index.identity_sha256
            or existing.review_sha256 != review_sha256
        ):
            raise EmbeddedValidationAssessmentError(
                "existing embedded Validation receipt belongs to another review"
            )
        return existing
    execution_index = _load_model(
        root / EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        EmbeddedValidationExecutionIndex,
    )
    if not execution_index.authorized:
        raise EmbeddedValidationAssessmentError(
            "rejected or revision-required assessment has no execution to review"
        )
    if (
        execution_index.execution_authorization is None
        or execution_index.execution_result is None
        or execution_index.canonical_assessment is None
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation execution index is incomplete"
        )
    decision = store.load_typed(
        execution_index.coordinator_decision,
        EmbeddedCoordinatorDecision,
    )
    authorization = store.load_typed(
        execution_index.execution_authorization,
        BoundedExecutionAuthorization,
    )
    result = store.load_typed(
        execution_index.execution_result,
        EmbeddedBoundedExecutionResult,
    )
    if review_draft.disposition == "accept":
        assessment = _load_model(
            Path(execution_index.canonical_assessment.path),
            ValidationCoordinatorAssessment,
        )
        if assessment.terminal_disposition != "pass":
            raise EmbeddedValidationAssessmentError(
                "outer review cannot accept a non-passing canonical assessment"
            )
        if _binding(execution_index.canonical_assessment.path) != (
            execution_index.canonical_assessment
        ):
            raise EmbeddedValidationAssessmentError(
                "canonical Validation assessment bytes are stale"
            )
    else:
        assessment = _load_model(
            Path(execution_index.canonical_assessment.path),
            ValidationCoordinatorAssessment,
        )
    review = EmbeddedCoordinatorReview(
        artifact_id=f"validation-assessment-review-{authorization.operation_id}",
        identity=identity,
        producer=decision.producer,
        parent_artifact=artifact_reference(result),
        created_at=review_draft.created_at,
        execution_result=artifact_reference(result),
        semantic_decision_owner=decision.producer,
        accepted_decision_digest=authorization.accepted_decision_digest,
        disposition=review_draft.disposition,
        outputs=result.outputs,
        findings=review_draft.findings,
    )
    validate_coordinator_review(review, result)
    store.append_review(review)
    receipt = build_decision_receipt(
        artifact_id=f"validation-assessment-receipt-{authorization.operation_id}",
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
        created_at=review_draft.created_at,
    )
    validate_decision_receipt(
        receipt,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
    )
    store.append_receipt(receipt)
    gate_dispositions = _gate_dispositions(assessment)
    review_draft_binding = _binding(Path(review_path).expanduser().resolve())
    terminal_receipt = ValidationTerminalReceipt(
        mode="embedded",
        assessment_identity_sha256=evidence_index.identity_sha256,
        evidence_index=_binding(root / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME),
        canonical_assessment=execution_index.canonical_assessment,
        coordinator_review=review_draft_binding,
        gate_dispositions=gate_dispositions,
        terminal_disposition=assessment.terminal_disposition,
        review_disposition=review_draft.disposition,
        receipt_status=(
            "completed"
            if receipt.receipt_status == "completed"
            and review_draft.disposition == "accept"
            else "rejected"
        ),
    )
    terminal_receipt_path = root / VALIDATION_TERMINAL_RECEIPT_NAME
    _write_index(terminal_receipt_path, terminal_receipt)
    receipt_index = EmbeddedValidationReceiptIndex(
        identity_sha256=evidence_index.identity_sha256,
        review_sha256=review_sha256,
        canonical_assessment=execution_index.canonical_assessment,
        coordinator_review=artifact_reference(review),
        coordinator_review_draft=review_draft_binding,
        decision_receipt=artifact_reference(receipt),
        terminal_receipt=_binding(terminal_receipt_path),
        receipt_status=receipt.receipt_status,
        gate_dispositions=gate_dispositions,
    )
    _write_index(receipt_index_path, receipt_index)
    return receipt_index


def validate_completed_embedded_validation_receipt(
    output_dir: str | Path,
    *,
    expected_identity: EmbeddedDecisionIdentity | None = None,
) -> tuple[ValidationCoordinatorAssessment, EmbeddedDecisionReceipt]:
    """Revalidate the persisted completed chain for composed-stage acceptance."""

    root = Path(output_dir).expanduser().resolve()
    evidence_index = _load_model(
        root / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        EmbeddedValidationEvidenceIndex,
    )
    execution_index = _load_model(
        root / EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        EmbeddedValidationExecutionIndex,
    )
    receipt_index = _load_embedded_validation_receipt_index(
        root / EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME
    )
    if receipt_index.receipt_status != "completed":
        raise EmbeddedValidationAssessmentError(
            "embedded Validation receipt is not completed"
        )
    if _binding(receipt_index.canonical_assessment.path) != (
        receipt_index.canonical_assessment
    ):
        raise EmbeddedValidationAssessmentError(
            "canonical Validation assessment binding is stale"
        )
    assessment = _load_model(
        Path(receipt_index.canonical_assessment.path),
        ValidationCoordinatorAssessment,
    )
    expected_gate_dispositions = _gate_dispositions(assessment)
    if receipt_index.gate_dispositions != expected_gate_dispositions:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation receipt gate dispositions are incomplete or stale"
        )
    if _binding(receipt_index.terminal_receipt.path) != receipt_index.terminal_receipt:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation terminal receipt binding is stale"
        )
    terminal_receipt = _load_model(
        Path(receipt_index.terminal_receipt.path),
        ValidationTerminalReceipt,
    )
    if _binding(receipt_index.coordinator_review_draft.path) != (
        receipt_index.coordinator_review_draft
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation terminal review binding is stale"
        )
    review_draft = _load_model(
        Path(receipt_index.coordinator_review_draft.path),
        ValidationCoordinatorReviewDraft,
    )
    store = EmbeddedDecisionArtifactStore(root)
    review = store.load_typed(
        receipt_index.coordinator_review,
        EmbeddedCoordinatorReview,
    )
    expected_review_draft = ValidationCoordinatorReviewDraft(
        created_at=review.created_at,
        disposition=review.disposition,
        findings=tuple(review.findings),
    )
    if (
        review_draft != expected_review_draft
        or canonical_json_digest(review_draft) != receipt_index.review_sha256
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation terminal review differs from the accepted "
            "coordinator review"
        )
    expected_terminal = ValidationTerminalReceipt(
        mode="embedded",
        assessment_identity_sha256=evidence_index.identity_sha256,
        evidence_index=_binding(root / EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME),
        canonical_assessment=receipt_index.canonical_assessment,
        coordinator_review=receipt_index.coordinator_review_draft,
        gate_dispositions=expected_gate_dispositions,
        terminal_disposition=assessment.terminal_disposition,
        review_disposition=review.disposition,
        receipt_status="completed",
    )
    if terminal_receipt != expected_terminal:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation terminal receipt is incomplete or stale"
        )
    evidence = tuple(
        store.load_typed(reference, EmbeddedDomainEvidence)
        for reference in evidence_index.evidence
    )
    proposals = tuple(
        store.load_typed(reference, EmbeddedDomainProposal)
        for reference in evidence_index.proposals
    )
    decision = store.load_typed(
        execution_index.coordinator_decision,
        EmbeddedCoordinatorDecision,
    )
    if (
        execution_index.execution_authorization is None
        or execution_index.execution_result is None
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation execution is incomplete"
        )
    authorization = store.load_typed(
        execution_index.execution_authorization,
        BoundedExecutionAuthorization,
    )
    result = store.load_typed(
        execution_index.execution_result,
        EmbeddedBoundedExecutionResult,
    )
    receipt = store.load_typed(
        receipt_index.decision_receipt,
        EmbeddedDecisionReceipt,
    )
    if (
        not execution_index.authorized
        or execution_index.assessment_id != assessment.assessment_id
        or execution_index.assessment_sha256 != canonical_json_digest(assessment)
        or execution_index.canonical_assessment != receipt_index.canonical_assessment
        or receipt_index.receipt_status != receipt.receipt_status
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation indexes disagree about the completed assessment"
        )
    selected_identity = expected_identity or receipt.identity
    expected_sha = canonical_json_digest(selected_identity)
    if {
        evidence_index.identity_sha256,
        execution_index.identity_sha256,
        receipt_index.identity_sha256,
    } != {expected_sha}:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation indexes do not bind the active identity"
        )
    if receipt.identity != selected_identity:
        raise EmbeddedValidationAssessmentError(
            "embedded Validation receipt belongs to another decision identity"
        )
    validate_decision_receipt(
        receipt,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
    )
    accepted_assessment = _assessment_from_decision(decision)
    if accepted_assessment != assessment:
        raise EmbeddedValidationAssessmentError(
            "canonical assessment differs from the accepted coordinator decision"
        )
    validate_coordinator_assessment(
        assessment,
        evidence=evidence,
        proposals=proposals,
    )
    if (
        assessment.terminal_disposition != "pass"
        or receipt.execution_effect != "non_mutating"
        or receipt.mutation_id is not None
        or receipt.review_disposition != "accept"
    ):
        raise EmbeddedValidationAssessmentError(
            "embedded Validation chain does not prove accepted non-mutating success"
        )
    return assessment, receipt


__all__ = [
    "CANONICAL_VALIDATION_ASSESSMENT_NAME",
    "EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME",
    "EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME",
    "EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME",
    "EMBEDDED_VALIDATION_EVIDENCE_INDEX_SCHEMA_VERSION",
    "EMBEDDED_VALIDATION_EXECUTION_INDEX_SCHEMA_VERSION",
    "EMBEDDED_VALIDATION_RECEIPT_INDEX_SCHEMA_VERSION",
    "VALIDATION_COORDINATOR_ASSESSMENT_SCHEMA_VERSION",
    "VALIDATION_COORDINATOR_REVIEW_DRAFT_SCHEMA_VERSION",
    "VALIDATION_TERMINAL_RECEIPT_NAME",
    "VALIDATION_TERMINAL_RECEIPT_SCHEMA_VERSION",
    "VALIDATION_REFERENCE_IMAGE_SUFFIXES",
    "EmbeddedValidationAssessmentError",
    "EmbeddedValidationEvidenceIndex",
    "EmbeddedValidationExecutionIndex",
    "EmbeddedValidationReceiptIndex",
    "ValidationAssessmentFinding",
    "ValidationCoordinatorAssessment",
    "ValidationCoordinatorReviewDraft",
    "ValidationGateAssessment",
    "ValidationTerminalReceipt",
    "assess_embedded_validation",
    "build_validation_decision_identity",
    "load_embedded_validation_run",
    "prepare_embedded_validation_evidence",
    "review_embedded_validation_assessment",
    "validate_completed_embedded_validation_receipt",
    "validate_coordinator_assessment",
]
