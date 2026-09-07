# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic catalog adapters over existing public Validation boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, RootModel
from world_understanding.validation import ValidationTemplateResult

from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.validation import (
    VALIDATION_OPERATION_PREPARATION_NAME,
    VERIFIED_OPERATION_INGEST_INDEX_NAME,
    CanonicalVisualEvidencePublication,
    CanonicalVisualEvidenceRequest,
    ValidationOperationPreparation,
    ValidationOperationResult,
    ValidationWorkflowError,
    VerifiedOperationError,
    VerifiedOperationIngestReceipt,
    VerifiedValidationOperationEnvelope,
    execution_artifact_binding,
    load_validation_operation_preparation,
    load_verified_operation_ingest_index,
    verify_execution_artifact_binding,
    verify_operation_envelope,
)

from .catalog import AssetLeafRuntimeBinding, AssetLeafRuntimeBundle
from .models import (
    ArtifactBinding,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
)

CANONICAL_OVRTX_EVIDENCE_LEAF_ID = "validation.canonical-ovrtx-evidence.v1"
FOCUSED_VALIDATION_OPERATION_LEAF_ID = "validation.focused-operation.v1"
PROVIDED_VALIDATION_INGRESS_LEAF_ID = "validation.provided-ingress.v1"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalOvrtxEvidenceLeafInvocation(_FrozenModel):
    """Exact arguments of the existing canonical OVRTX evidence entrypoint."""

    output_dir: str = Field(min_length=1)
    post_mutation_usd: str = Field(min_length=1)
    source_usd: str = Field(min_length=1)
    backend: Literal["remote", "ovrtx"]
    views: tuple[str, ...] = Field(min_length=1)
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    operation_id: str = "visual.canonical-ovrtx"
    gate_id: str = "visual.canonical-evidence"


class CanonicalOvrtxEvidenceLeafTerminalResult(_FrozenModel):
    """Typed native failure/cancellation retained by the shared adapter."""

    schema_version: Literal[
        "content-agent-workflows.canonical-ovrtx-leaf-terminal.v1"
    ] = "content-agent-workflows.canonical-ovrtx-leaf-terminal.v1"
    output_dir: str = Field(min_length=1)
    operation_id: str = "visual.canonical-ovrtx"
    gate_id: str = "visual.canonical-evidence"
    native_disposition: Literal["failed", "cancelled"]
    native_status: str = Field(min_length=1, max_length=240)
    native_request: ArtifactBinding
    native_terminal_receipt: ArtifactBinding
    evidence: tuple[ArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ArtifactBinding, ...] = Field(min_length=1)
    resource_claims: tuple[str, ...] = ()
    resource_release_receipts: tuple[ArtifactBinding, ...] = ()
    summary: str = Field(min_length=1)
    error: str = Field(min_length=1)


class CanonicalOvrtxEvidenceLeafResult(
    RootModel[
        CanonicalVisualEvidencePublication | CanonicalOvrtxEvidenceLeafTerminalResult
    ]
):
    """Closed success/failure result union for the canonical OVRTX leaf."""


class SharedValidationLeafTerminalResult(_FrozenModel):
    """Typed native failure/cancellation shared by Validation adapters."""

    schema_version: Literal[
        "content-agent-workflows.shared-validation-leaf-terminal.v1"
    ] = "content-agent-workflows.shared-validation-leaf-terminal.v1"
    output_dir: str = Field(min_length=1)
    invocation_artifact: ArtifactBinding
    native_disposition: Literal["failed", "cancelled"]
    native_status: str = Field(min_length=1, max_length=240)
    native_terminal_receipt: ArtifactBinding
    evidence: tuple[ArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ArtifactBinding, ...] = Field(min_length=1)
    resource_claims: tuple[str, ...] = ()
    resource_release_receipts: tuple[ArtifactBinding, ...] = ()
    summary: str = Field(min_length=1)
    error: str = Field(min_length=1)


class FocusedValidationOperationLeafInvocation(_FrozenModel):
    """Exact arguments of one existing focused Validation check."""

    output_dir: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    prior_result_paths: tuple[str, ...] = ()


class ProvidedValidationIngressLeafInvocation(_FrozenModel):
    """Exact arguments of the existing provided-operation ingress."""

    envelope_path: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)


class FocusedValidationOperationLeafResult(
    RootModel[ValidationOperationResult | SharedValidationLeafTerminalResult]
):
    """Closed native result union for one focused Validation operation."""


class ProvidedValidationIngressLeafResult(
    RootModel[VerifiedOperationIngestReceipt | SharedValidationLeafTerminalResult]
):
    """Closed native result union for provided Validation ingress."""


def _artifact(binding: BaseModel) -> ArtifactBinding:
    return ArtifactBinding.model_validate(binding.model_dump(mode="json"))


def _bound_file(path: str, sha256: str) -> ArtifactBinding:
    candidate = Path(path)
    try:
        observed = execution_artifact_binding(candidate)
    except (OSError, VerifiedOperationError) as exc:
        raise ValueError(
            f"projected artifact is missing or unsafe: {candidate}"
        ) from exc
    if observed.sha256 != sha256:
        raise ValueError(f"projected artifact digest is stale: {candidate}")
    return _artifact(observed)


def _unique_bindings(
    *groups: tuple[ArtifactBinding, ...],
) -> tuple[ArtifactBinding, ...]:
    unique: dict[tuple[str, str, int], ArtifactBinding] = {}
    for binding in (binding for group in groups for binding in group):
        unique[(binding.path, binding.sha256, binding.size_bytes)] = binding
    return tuple(unique.values())


def _canonical_ovrtx_projector(
    invocation: BaseModel,
    result: BaseModel,
    _context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(CanonicalOvrtxEvidenceLeafInvocation, invocation)
    resolved_result = cast(CanonicalOvrtxEvidenceLeafResult, result).root
    expected_output_root = _resolved_path(request.output_dir)
    if isinstance(resolved_result, CanonicalOvrtxEvidenceLeafTerminalResult):
        terminal = resolved_result
        if (
            _resolved_path(terminal.output_dir) != expected_output_root
            or terminal.operation_id != request.operation_id
            or terminal.gate_id != request.gate_id
        ):
            raise ValueError(
                "canonical OVRTX terminal result belongs to another invocation"
            )
        native_request_binding = _verified_declared_binding(
            terminal.native_request,
            label="canonical OVRTX native request",
        )
        serialized_native_request = verify_execution_artifact_binding(
            terminal.native_request,
            label="canonical OVRTX native request",
        )
        native_request = CanonicalVisualEvidenceRequest.model_validate_json(
            serialized_native_request
        )
        _validate_canonical_native_request(
            request,
            native_request,
            request_path=native_request_binding.path,
            expected_output_root=expected_output_root,
        )
        terminal_receipt = _verified_declared_binding(
            terminal.native_terminal_receipt,
            label="canonical OVRTX native terminal receipt",
        )
        evidence = tuple(
            _verified_declared_binding(binding, label="canonical OVRTX evidence")
            for binding in terminal.evidence
        )
        readbacks = tuple(
            _verified_declared_binding(
                binding,
                label="canonical OVRTX saved-stage readback",
            )
            for binding in terminal.saved_stage_readbacks
        )
        releases = tuple(
            _verified_declared_binding(
                binding,
                label="canonical OVRTX resource release",
            )
            for binding in terminal.resource_release_receipts
        )
        _require_distinct_artifacts(
            native_request_binding,
            terminal_receipt,
            *evidence,
            *readbacks,
            *releases,
            label="canonical OVRTX terminal artifacts",
        )
        for binding in (
            native_request_binding,
            terminal_receipt,
            *evidence,
            *readbacks,
            *releases,
        ):
            try:
                Path(binding.path).resolve(strict=True).relative_to(
                    expected_output_root
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "canonical OVRTX terminal artifact escaped output_dir"
                ) from exc
        return AssetLeafProjectionPayload(
            native_disposition=terminal.native_disposition,
            native_status=terminal.native_status,
            native_terminal_receipt=terminal_receipt,
            evidence=evidence,
            saved_stage_readbacks=readbacks,
            resource_claims=terminal.resource_claims,
            resource_release_receipts=releases,
            summary=terminal.summary,
            error=terminal.error,
        )

    publication = resolved_result
    envelope = publication.result
    if (
        envelope.operation_id != request.operation_id
        or envelope.gate_id != request.gate_id
    ):
        raise ValueError("canonical OVRTX result belongs to another operation or gate")
    if envelope.native_status != "pass":
        raise ValueError(
            "canonical OVRTX result did not retain its native pass disposition"
        )
    if (
        not envelope.required
        or envelope.authority != "outer_review_input"
        or envelope.evidence_type != "visual.canonical-ovrtx"
        or envelope.native_report_type != "usd-cli.ovrtx-render-report"
        or envelope.native_payload_type != "visual.canonical-usd-cli-ovrtx-payload"
    ):
        raise ValueError("canonical OVRTX result contract identity changed")

    verify_operation_envelope(envelope)
    serialized_envelope = verify_execution_artifact_binding(
        publication.envelope,
        label="canonical OVRTX envelope",
    )
    if (
        VerifiedValidationOperationEnvelope.model_validate_json(serialized_envelope)
        != envelope
    ):
        raise ValueError("canonical OVRTX publication substituted its envelope")

    serialized_request = verify_execution_artifact_binding(
        publication.request,
        label="canonical OVRTX request",
    )
    native_request = CanonicalVisualEvidenceRequest.model_validate_json(
        serialized_request
    )
    _validate_canonical_native_request(
        request,
        native_request,
        request_path=publication.request.path,
        expected_output_root=expected_output_root,
    )
    if (
        publication.render_report != envelope.native_report
        or publication.payload != envelope.native_payload
        or publication.projection != envelope.projection
    ):
        raise ValueError("canonical OVRTX publication artifact chain changed")

    artifacts = tuple(_artifact(binding) for binding in envelope.artifacts)
    native_results = tuple(
        _artifact(binding)
        for binding in (envelope.native_report, envelope.native_payload)
        if binding is not None
    )
    evidence = _unique_bindings(
        (
            _artifact(publication.render_report),
            _artifact(publication.payload),
            _artifact(publication.projection),
            _artifact(publication.envelope),
        ),
        artifacts,
        native_results,
    )
    return AssetLeafProjectionPayload(
        native_disposition="passed",
        native_status=envelope.native_status,
        native_terminal_receipt=_artifact(publication.envelope),
        evidence=evidence,
        saved_stage_readbacks=(
            _artifact(publication.payload),
            _artifact(publication.projection),
            _artifact(publication.envelope),
        ),
        summary="Canonical usd-cli/OVRTX evidence publication completed.",
    )


def _resolved_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("selected leaf invocation paths must be absolute")
    return candidate.resolve()


def _validate_canonical_native_request(
    request: CanonicalOvrtxEvidenceLeafInvocation,
    native_request: CanonicalVisualEvidenceRequest,
    *,
    request_path: str,
    expected_output_root: Path,
) -> None:
    _verify_execution_binding_identity(
        native_request.source,
        label="canonical OVRTX native source",
    )
    _verify_execution_binding_identity(
        native_request.post_mutation_output,
        label="canonical OVRTX native output",
    )
    for index, dependency in enumerate(native_request.dependencies):
        _verify_execution_binding_identity(
            dependency,
            label=f"canonical OVRTX native dependency {index}",
        )
    if (
        Path(request_path).resolve(strict=True).parent != expected_output_root
        or _resolved_path(native_request.source.path)
        != _resolved_path(request.source_usd)
        or _resolved_path(native_request.post_mutation_output.path)
        != _resolved_path(request.post_mutation_usd)
        or native_request.backend != request.backend
        or native_request.views != request.views
        or native_request.image_width != request.image_width
        or native_request.image_height != request.image_height
    ):
        raise ValueError("canonical OVRTX result differs from its invocation")


def _verify_execution_binding_identity(binding: BaseModel, *, label: str) -> None:
    try:
        verify_execution_artifact_binding(binding, label=label)
        observed = _artifact(execution_artifact_binding(binding.path))
    except (OSError, VerifiedOperationError) as exc:
        raise ValueError(f"{label} is missing, unsafe, or stale") from exc
    if observed != _artifact(binding):
        raise ValueError(f"{label} identity drifted")


def _verified_declared_binding(
    binding: ArtifactBinding,
    *,
    label: str,
) -> ArtifactBinding:
    try:
        verify_execution_artifact_binding(binding, label=label)
        observed = _artifact(execution_artifact_binding(binding.path))
    except (OSError, VerifiedOperationError) as exc:
        raise ValueError(f"{label} is missing, unsafe, or stale") from exc
    if observed != binding:
        raise ValueError(f"{label} identity drifted")
    return observed


def _require_distinct_artifacts(
    *bindings: ArtifactBinding,
    label: str,
) -> None:
    try:
        identities = {
            (
                str(Path(binding.path).resolve(strict=True)),
                binding.sha256,
                binding.size_bytes,
            )
            for binding in bindings
        }
    except OSError as exc:
        raise ValueError(f"{label} contains a missing or unsafe path") from exc
    if len(identities) != len(bindings):
        raise ValueError(f"{label} must be pairwise distinct")


def _project_shared_validation_terminal(
    *,
    output_dir: str,
    terminal: SharedValidationLeafTerminalResult,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    output_root = _resolved_path(output_dir)
    if _resolved_path(terminal.output_dir) != output_root:
        raise ValueError("shared Validation terminal result belongs to another output")
    invocation = _verified_declared_binding(
        terminal.invocation_artifact,
        label="shared Validation invocation artifact",
    )
    if invocation != context.invocation_artifact:
        raise ValueError("shared Validation terminal result substituted its invocation")
    terminal_receipt = _verified_declared_binding(
        terminal.native_terminal_receipt,
        label="shared Validation native terminal receipt",
    )
    evidence = tuple(
        _verified_declared_binding(binding, label="shared Validation evidence")
        for binding in terminal.evidence
    )
    readbacks = tuple(
        _verified_declared_binding(
            binding,
            label="shared Validation saved-stage readback",
        )
        for binding in terminal.saved_stage_readbacks
    )
    releases = tuple(
        _verified_declared_binding(
            binding,
            label="shared Validation resource release",
        )
        for binding in terminal.resource_release_receipts
    )
    _require_distinct_artifacts(
        invocation,
        terminal_receipt,
        *evidence,
        *readbacks,
        *releases,
        label="shared Validation terminal artifacts",
    )
    for binding in (terminal_receipt, *evidence, *readbacks, *releases):
        try:
            Path(binding.path).resolve(strict=True).relative_to(output_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "shared Validation terminal artifact escaped output_dir"
            ) from exc
    return AssetLeafProjectionPayload(
        native_disposition=terminal.native_disposition,
        native_status=terminal.native_status,
        native_terminal_receipt=terminal_receipt,
        evidence=evidence,
        saved_stage_readbacks=readbacks,
        resource_claims=terminal.resource_claims,
        resource_release_receipts=releases,
        summary=terminal.summary,
        error=terminal.error,
    )


def _bound_payload(
    path: str | Path,
    *,
    label: str,
) -> tuple[ArtifactBinding, bytes]:
    try:
        binding = execution_artifact_binding(path)
        payload = verify_execution_artifact_binding(binding, label=label)
    except (OSError, VerifiedOperationError) as exc:
        raise ValueError(f"{label} is missing, unsafe, or stale: {path}") from exc
    return _artifact(binding), payload


def _validated_focused_operation_result(
    *,
    output_root: Path,
    template_name: str,
    preparation_digest: str,
    expected_result: ValidationOperationResult | None = None,
) -> tuple[ValidationOperationResult, ArtifactBinding, ArtifactBinding]:
    operation_path = (
        output_root / "operations" / template_name / "operation_result.json"
    )
    if operation_path.is_symlink() or not operation_path.is_file():
        raise ValueError(
            "focused Validation canonical operation result is missing or unsafe: "
            f"{operation_path}"
        )
    operation_receipt, operation_payload = _bound_payload(
        operation_path,
        label="focused Validation canonical operation result",
    )
    try:
        operation = ValidationOperationResult.model_validate_json(operation_payload)
    except ValueError as exc:
        raise ValueError(
            "focused Validation canonical operation result is invalid: "
            f"{operation_path}"
        ) from exc
    if operation.template_name != template_name:
        raise ValueError(
            "focused Validation canonical operation result belongs to another template"
        )
    if operation.preparation_digest != preparation_digest:
        raise ValueError(
            "focused Validation canonical operation result belongs to another "
            "preparation"
        )
    if expected_result is not None and operation != expected_result:
        raise ValueError(
            "focused Validation result differs from its canonical operation result"
        )

    expected_template_path = operation_path.parent / "template_result.json"
    template_path = _resolved_path(operation.template_result_path)
    if template_path != expected_template_path.resolve():
        raise ValueError(
            "focused Validation result substituted its canonical template result"
        )
    terminal, template_payload = _bound_payload(
        operation.template_result_path,
        label="focused Validation canonical template result",
    )
    if terminal.sha256 != operation.template_result_sha256:
        raise ValueError(
            "focused Validation canonical template result digest is stale: "
            f"{template_path}"
        )
    try:
        template_result = ValidationTemplateResult.model_validate_json(template_payload)
    except ValueError as exc:
        raise ValueError(
            f"focused Validation canonical template result is invalid: {template_path}"
        ) from exc
    if template_result != operation.template_result:
        raise ValueError(
            "focused Validation template result differs from its canonical bytes"
        )
    return operation, operation_receipt, terminal


def _focused_owned_evidence(
    operation: ValidationOperationResult,
    *,
    output_root: Path,
) -> tuple[ArtifactBinding, ...]:
    if any(binding.kind != "file" for binding in operation.evidence_artifacts):
        raise ValueError(
            "focused Validation projection requires file-bound evidence artifacts"
        )
    source_identities = {
        (binding.path, binding.sha256, binding.kind)
        for binding in (*operation.source_before, *operation.source_after)
    }
    # Focused native results may name their digest-bound source as evidence. Keep
    # that input identity in the immutable result, while promoting only files
    # owned by this operation into the composed receipt.
    owned: list[ArtifactBinding] = []
    for binding in operation.evidence_artifacts:
        projected = _bound_file(binding.path, cast(str, binding.sha256))
        if _resolved_path(projected.path).is_relative_to(output_root):
            owned.append(projected)
        elif (binding.path, binding.sha256, binding.kind) not in source_identities:
            raise ValueError(
                "focused Validation external evidence is not a bound source identity"
            )
    return tuple(owned)


def _focused_validation_projector(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(FocusedValidationOperationLeafInvocation, invocation)
    resolved_result = cast(FocusedValidationOperationLeafResult, result).root
    if isinstance(resolved_result, SharedValidationLeafTerminalResult):
        return _project_shared_validation_terminal(
            output_dir=request.output_dir,
            terminal=resolved_result,
            context=context,
        )
    operation = resolved_result
    if operation.template_name != request.template_name:
        raise ValueError("focused Validation result belongs to another template")

    output_root = _resolved_path(request.output_dir)
    try:
        preparation = load_validation_operation_preparation(output_root)
    except (OSError, ValueError, ValidationWorkflowError) as exc:
        raise ValueError(
            "focused Validation preparation is missing or invalid"
        ) from exc
    preparation_binding, preparation_payload = _bound_payload(
        output_root / VALIDATION_OPERATION_PREPARATION_NAME,
        label="focused Validation preparation",
    )
    try:
        bound_preparation = ValidationOperationPreparation.model_validate_json(
            preparation_payload
        )
    except ValueError as exc:
        raise ValueError("focused Validation preparation bytes are invalid") from exc
    if bound_preparation != preparation:
        raise ValueError("focused Validation preparation identity changed during read")
    preparation_digest = canonical_json_digest(preparation)
    if operation.preparation_digest != preparation_digest:
        raise ValueError("focused Validation result belongs to another preparation")
    if operation.source_before != preparation.workflow_identity.source_artifacts:
        raise ValueError(
            "focused Validation result changed its prepared source identity"
        )

    planned_steps = tuple(
        step
        for step in preparation.plan.steps
        if step.template_name == request.template_name
    )
    if len(planned_steps) != 1:
        raise ValueError(
            "focused Validation invocation is absent from its preparation plan"
        )
    work_item = planned_steps[0].metadata.get("agentic_work_item")
    if not isinstance(work_item, dict) or not isinstance(
        work_item.get("depends_on"), list
    ):
        raise ValueError("focused Validation dependency plan is missing or invalid")
    dependencies = work_item["depends_on"]
    if any(
        not isinstance(dependency, str)
        or not dependency.startswith("validation:")
        or dependency == "validation:"
        for dependency in dependencies
    ):
        raise ValueError("focused Validation dependency plan is missing or invalid")
    expected_prior_templates = tuple(
        dependency.removeprefix("validation:") for dependency in dependencies
    )
    expected_prior_paths = tuple(
        (output_root / "operations" / name / "operation_result.json").resolve()
        for name in expected_prior_templates
    )
    actual_prior_paths = tuple(
        _resolved_path(path) for path in request.prior_result_paths
    )
    if actual_prior_paths != expected_prior_paths:
        raise ValueError(
            "focused Validation invocation differs from its prepared prior-result chain"
        )
    prior_receipts: list[ArtifactBinding] = []
    prior_evidence: list[ArtifactBinding] = []
    for prior_template in expected_prior_templates:
        prior, prior_receipt, prior_terminal = _validated_focused_operation_result(
            output_root=output_root,
            template_name=prior_template,
            preparation_digest=preparation_digest,
        )
        if prior.source_before != preparation.workflow_identity.source_artifacts:
            raise ValueError(
                "focused Validation prior result changed its prepared source identity"
            )
        prior_receipts.extend((prior_receipt, prior_terminal))
        prior_evidence.extend(_focused_owned_evidence(prior, output_root=output_root))
    _, operation_receipt, terminal = _validated_focused_operation_result(
        output_root=output_root,
        template_name=request.template_name,
        preparation_digest=preparation_digest,
        expected_result=operation,
    )

    if operation.template_result.passed:
        disposition: Literal["passed", "failed", "not_evaluated"] = "passed"
        error = None
    elif operation.template_result.status == "skipped" and not operation.mandatory:
        disposition = "not_evaluated"
        error = None
    else:
        disposition = "failed"
        error = (
            f"focused Validation {operation.template_name} ended with native "
            f"status {operation.template_result.status}"
        )
    owned_evidence = _focused_owned_evidence(operation, output_root=output_root)
    evidence = _unique_bindings(
        (preparation_binding,),
        tuple(prior_receipts),
        tuple(prior_evidence),
        (operation_receipt, terminal),
        owned_evidence,
    )
    return AssetLeafProjectionPayload(
        native_disposition=disposition,
        native_status=str(operation.template_result.status),
        native_terminal_receipt=operation_receipt,
        evidence=evidence,
        saved_stage_readbacks=_unique_bindings(
            (preparation_binding,),
            tuple(prior_receipts),
            (operation_receipt, terminal),
        ),
        summary=f"Focused Validation {operation.template_name} completed.",
        error=error,
    )


def _provided_validation_projector(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = cast(ProvidedValidationIngressLeafInvocation, invocation)
    resolved_result = cast(ProvidedValidationIngressLeafResult, result).root
    if isinstance(resolved_result, SharedValidationLeafTerminalResult):
        return _project_shared_validation_terminal(
            output_dir=request.output_dir,
            terminal=resolved_result,
            context=context,
        )
    receipt = resolved_result
    if _resolved_path(receipt.envelope_binding.path) != _resolved_path(
        request.envelope_path
    ):
        raise ValueError("provided Validation result belongs to another envelope")
    envelope = receipt.envelope
    if envelope.native_status == "pass":
        disposition: Literal["passed", "failed", "not_evaluated"] = "passed"
        error = None
    elif envelope.native_status == "not_requested":
        raise ValueError(
            "provided Validation not_requested is a graph omission, not a leaf result"
        )
    elif envelope.native_status in {"not_evaluated", "warn"}:
        disposition = "not_evaluated"
        error = None
    else:
        disposition = "failed"
        error = (
            f"provided Validation {envelope.operation_id} retained native "
            f"status {envelope.native_status}"
        )
    output_root = _resolved_path(request.output_dir)
    try:
        index, receipts = load_verified_operation_ingest_index(output_root)
        ingest_index = _artifact(
            execution_artifact_binding(
                output_root / VERIFIED_OPERATION_INGEST_INDEX_NAME
            )
        )
    except (OSError, VerifiedOperationError) as exc:
        raise ValueError(
            "provided Validation local ingest artifacts are missing or invalid"
        ) from exc
    if len(index.operations) != 1 or receipts != (receipt,):
        raise ValueError(
            "provided Validation result differs from its local ingest index"
        )
    ingest_receipt = _artifact(index.operations[0].ingest_receipt)
    for binding in (ingest_receipt, ingest_index):
        try:
            Path(binding.path).resolve(strict=True).relative_to(output_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "provided Validation local ingest artifacts escaped output_dir"
            ) from exc
    return AssetLeafProjectionPayload(
        native_disposition=disposition,
        native_status=envelope.native_status,
        native_terminal_receipt=ingest_receipt,
        operation_indexes=(ingest_index,),
        evidence=(ingest_receipt,),
        saved_stage_readbacks=(ingest_receipt, ingest_index),
        summary=f"Provided Validation {envelope.operation_id} ingress completed.",
        error=error,
    )


def canonical_ovrtx_asset_leaf_runtime_binding(
    *,
    leaf_id: str = CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    projector_id: str = "asset.projector.canonical-ovrtx-evidence.v1",
) -> AssetLeafRuntimeBinding:
    """Register one independently schedulable canonical OVRTX evidence identity."""

    return AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint=(
            "content_agent_workflows.validation.produce_canonical_visual_evidence"
        ),
        invocation_model=CanonicalOvrtxEvidenceLeafInvocation,
        result_model=CanonicalOvrtxEvidenceLeafResult,
        projector_id=projector_id,
        projector=_canonical_ovrtx_projector,
        required_artifact_categories=("evidence", "saved_stage_readback"),
    )


def shared_asset_leaf_runtime_bundle() -> AssetLeafRuntimeBundle:
    """Register shared public operations without importing optional domains."""

    return AssetLeafRuntimeBundle.create(
        bundle_id="shared",
        bindings=(
            canonical_ovrtx_asset_leaf_runtime_binding(),
            AssetLeafRuntimeBinding.create(
                leaf_id=FOCUSED_VALIDATION_OPERATION_LEAF_ID,
                entrypoint="content_agent_workflows.validation.run_validation_operation",
                invocation_model=FocusedValidationOperationLeafInvocation,
                result_model=FocusedValidationOperationLeafResult,
                projector_id="asset.projector.focused-validation-operation.v1",
                projector=_focused_validation_projector,
                required_artifact_categories=("evidence", "saved_stage_readback"),
                incompatible_leaf_ids=(PROVIDED_VALIDATION_INGRESS_LEAF_ID,),
            ),
            AssetLeafRuntimeBinding.create(
                leaf_id=PROVIDED_VALIDATION_INGRESS_LEAF_ID,
                entrypoint=(
                    "content_agent_workflows.validation.ingest_verified_operation_result"
                ),
                invocation_model=ProvidedValidationIngressLeafInvocation,
                result_model=ProvidedValidationIngressLeafResult,
                projector_id="asset.projector.provided-validation-ingress.v1",
                projector=_provided_validation_projector,
                required_artifact_categories=("evidence", "saved_stage_readback"),
                incompatible_leaf_ids=(FOCUSED_VALIDATION_OPERATION_LEAF_ID,),
            ),
        ),
    )


__all__ = [
    "CANONICAL_OVRTX_EVIDENCE_LEAF_ID",
    "FOCUSED_VALIDATION_OPERATION_LEAF_ID",
    "PROVIDED_VALIDATION_INGRESS_LEAF_ID",
    "CanonicalOvrtxEvidenceLeafInvocation",
    "CanonicalOvrtxEvidenceLeafResult",
    "CanonicalOvrtxEvidenceLeafTerminalResult",
    "FocusedValidationOperationLeafInvocation",
    "FocusedValidationOperationLeafResult",
    "ProvidedValidationIngressLeafInvocation",
    "ProvidedValidationIngressLeafResult",
    "SharedValidationLeafTerminalResult",
    "canonical_ovrtx_asset_leaf_runtime_binding",
    "shared_asset_leaf_runtime_bundle",
]
