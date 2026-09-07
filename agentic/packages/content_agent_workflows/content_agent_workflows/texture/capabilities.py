# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Texture capabilities selected directly by an outer reasoner.

This module deliberately does not contain an orchestrator.  Each public function is
one bounded operation.  Its typed inputs carry the immutable identities needed by
the next operation, while the caller decides which optional leaves to invoke.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
)

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    bind_usd_dependency_closure,
    verify_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding

from .embedded_decision import canonical_json_digest
from .models import (
    TextureAcceptanceCriteria,
    TextureExecutionResult,
    TextureGeneratorInputs,
    TextureInspectionResult,
    TexturePlanDocument,
    TexturePlanTarget,
    TexturePreservationConstraints,
    TextureProvidedImageArtifact,
    TextureReferenceArtifact,
    TextureUnitArtifact,
    TextureWorkflowRequest,
)
from .runtime import texture_plan_digest
from .scope_validation import validate_texture_scope_invariants

TextureOperationName = Literal[
    "inspect",
    "propose",
    "generate",
    "evidence",
    "critique",
    "review",
    "publish",
]
TextureOperationSelectionState = Literal["requested", "not_requested"]
TextureOperationOutcomeState = Literal[
    "completed",
    "not_requested",
    "not_evaluated",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextureOperationSelection(_StrictFrozenModel):
    """Frozen task-scoped operation selection authored by the outer reasoner."""

    inspect: Literal["requested"] = "requested"
    propose: TextureOperationSelectionState = "not_requested"
    generate: TextureOperationSelectionState = "not_requested"
    evidence: TextureOperationSelectionState = "not_requested"
    critique: TextureOperationSelectionState = "not_requested"
    review: TextureOperationSelectionState = "not_requested"
    publish: TextureOperationSelectionState = "not_requested"

    @model_validator(mode="after")
    def _require_mutation_gates(self) -> Self:
        if self.generate == "requested" and (
            self.evidence != "requested"
            or self.review != "requested"
            or self.publish != "requested"
        ):
            raise ValueError(
                "Texture generation requires evidence, outer review, and publication "
                "gates in the frozen operation selection"
            )
        if self.publish == "requested" and (
            self.evidence != "requested" or self.review != "requested"
        ):
            raise ValueError(
                "Texture publication requires deterministic evidence and outer review"
            )
        if self.critique == "requested" and self.evidence != "requested":
            raise ValueError("Texture critique requires candidate evidence")
        return self

    def state_for(self, operation: TextureOperationName) -> str:
        """Return the frozen selection for one named capability."""

        return str(getattr(self, operation))


class TextureOperationOutcome(_StrictFrozenModel):
    """Non-ambiguous status for one selected or unselected capability."""

    operation: TextureOperationName
    state: TextureOperationOutcomeState
    artifact: ExecutionArtifactBinding | None = None
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bind_only_completed_artifacts(self) -> Self:
        if self.state != "completed" and self.artifact is not None:
            raise ValueError("only completed Texture operations may bind an artifact")
        return self


class TextureOperationStatus(_StrictFrozenModel):
    """Status table carried by every focused-operation packet."""

    operations: tuple[TextureOperationOutcome, ...] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def _cover_each_operation_once(self) -> Self:
        names = tuple(item.operation for item in self.operations)
        expected: tuple[TextureOperationName, ...] = (
            "inspect",
            "propose",
            "generate",
            "evidence",
            "critique",
            "review",
            "publish",
        )
        if names != expected:
            raise ValueError(
                "Texture operation status must use canonical operation order"
            )
        return self

    def outcome(self, operation: TextureOperationName) -> TextureOperationOutcome:
        return next(item for item in self.operations if item.operation == operation)


class TextureCapabilityRequest(_StrictFrozenModel):
    """Frozen input and task selection for independent Texture capabilities."""

    schema_version: Literal["content-agent-workflows.texture-capability-request.v1"] = (
        "content-agent-workflows.texture-capability-request.v1"
    )
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    output_dir: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    material_prim_paths: tuple[str, ...] = ()
    prim_paths: tuple[str, ...] = ()
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    operations: TextureOperationSelection
    texture_size: int = Field(default=1024, ge=64, le=16384)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scope_and_references(self) -> Self:
        if bool(self.material_prim_paths) == bool(self.prim_paths):
            raise ValueError(
                "Texture capability request requires exactly one material or prim scope"
            )
        paths = tuple(item.artifact.path for item in self.reference_artifacts)
        roles = tuple(item.role for item in self.reference_artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("Texture reference artifact paths must be unique")
        if len(roles) != len(set(roles)):
            raise ValueError("Texture reference artifact roles must be unique")
        return self


class TexturePreparationPacket(_StrictFrozenModel):
    """Provider-free source, scope, UV, reference, and initial-render evidence."""

    schema_version: Literal["content-agent-workflows.texture-preparation.v1"] = (
        "content-agent-workflows.texture-preparation.v1"
    )
    request: TextureCapabilityRequest
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope_plan: TexturePlanDocument
    scope_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    inspection: TextureInspectionResult
    operation_status: TextureOperationStatus


class TextureProviderProposalPacket(_StrictFrozenModel):
    """Optional advisory service proposal; never the canonical Texture plan."""

    schema_version: Literal["content-agent-workflows.texture-provider-proposal.v1"] = (
        "content-agent-workflows.texture-provider-proposal.v1"
    )
    preparation: ExecutionArtifactBinding
    provider: str = Field(min_length=1)
    proposal: TexturePlanDocument
    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_resume_state: dict[str, Any] = Field(default_factory=dict)
    operation_status: TextureOperationStatus


class TextureOuterPlan(_StrictFrozenModel):
    """Canonical target and generation decision authored by the outer reasoner."""

    schema_version: Literal[
        "content-agent-workflows.texture-outer-capability-plan.v1"
    ] = "content-agent-workflows.texture-outer-capability-plan.v1"
    preparation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    scope_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    advisory_proposal: ExecutionArtifactBinding | None = None
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    operations: TextureOperationSelection
    targets: tuple[TexturePlanTarget, ...] = Field(min_length=1)
    preservation: TexturePreservationConstraints
    acceptance: TextureAcceptanceCriteria
    capability_constraints: tuple[str, ...] = Field(min_length=1)
    stop_policy: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_targets(self) -> Self:
        unit_ids = tuple(target.unit_id for target in self.targets)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("outer Texture targets must be unique")
        if self.operations.generate != "requested":
            raise ValueError("outer Texture mutation plan must request generation")
        execution_modes = {
            target.generator_inputs.execution_mode for target in self.targets
        }
        if len(execution_modes) != 1:
            raise ValueError(
                "outer Texture plan must select one execution mode for all targets"
            )
        return self

    @property
    def target_unit_ids(self) -> tuple[str, ...]:
        return tuple(target.unit_id for target in self.targets)

    @property
    def execution_mode(self) -> Literal["provider_generate", "apply_provided"]:
        return self.targets[0].generator_inputs.execution_mode

    @property
    def provided_images(self) -> tuple[TextureProvidedImageArtifact, ...]:
        return tuple(
            image
            for target in self.targets
            for image in target.generator_inputs.provided_images
        )


class TextureGeneratorLeafRequest(_StrictFrozenModel):
    """Exact explicit input to one novel-texture generator leaf."""

    outer_plan: ExecutionArtifactBinding
    preparation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...] = ()
    intent: str = Field(min_length=1)
    scope_plan: TexturePlanDocument
    target_unit_ids: tuple[str, ...] = Field(min_length=1)
    generator_inputs: tuple[TextureGeneratorInputs, ...] = Field(min_length=1)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    output_dir: str = Field(min_length=1)


class TextureGenerationPacket(_StrictFrozenModel):
    """Provenance-bound candidate returned by an explicit generator leaf."""

    schema_version: Literal["content-agent-workflows.texture-generation.v1"] = (
        "content-agent-workflows.texture-generation.v1"
    )
    preparation: ExecutionArtifactBinding
    outer_plan: ExecutionArtifactBinding
    provider_proposal: ExecutionArtifactBinding | None = None
    generator_provider: str = Field(min_length=1)
    generator_capability: str = Field(min_length=1)
    execution_mode: Literal["provider_generate", "apply_provided"] = "provider_generate"
    generator_inputs: tuple[TextureGeneratorInputs, ...] = Field(min_length=1)
    provided_images: tuple[TextureProvidedImageArtifact, ...] = ()
    execution: TextureExecutionResult
    candidate: ExecutionArtifactBinding
    unit_artifacts: tuple[TextureUnitArtifact, ...] = Field(min_length=1)
    generator_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    operation_status: TextureOperationStatus


class TextureUnitRenderEvidence(_StrictFrozenModel):
    """Matched source/candidate OVRTX images for one exact target."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    source_images: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    candidate_images: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_matched_views(self) -> Self:
        if len(self.source_images) != len(self.candidate_images):
            raise ValueError("Texture evidence requires matched source/candidate views")
        return self


class TextureCandidateEvidencePacket(_StrictFrozenModel):
    """Deterministic candidate mutation, scope, closure, and OVRTX evidence."""

    schema_version: Literal["content-agent-workflows.texture-candidate-evidence.v1"] = (
        "content-agent-workflows.texture-candidate-evidence.v1"
    )
    preparation: ExecutionArtifactBinding
    outer_plan: ExecutionArtifactBinding
    generation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    candidate: ExecutionArtifactBinding
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    provided_images: tuple[TextureProvidedImageArtifact, ...] = ()
    unit_evidence: tuple[TextureUnitRenderEvidence, ...] = Field(min_length=1)
    static_evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    renderer_metadata: dict[str, Any] = Field(min_length=1)
    operation_status: TextureOperationStatus


class TextureCritiqueFinding(_StrictFrozenModel):
    """Advisory domain finding over exact evidence; never acceptance authority."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    status: Literal["pass", "fail", "not_evaluated"]
    summary: str = Field(min_length=1)


class TextureCritiquePacket(_StrictFrozenModel):
    """Optional provider critique with provenance distinct from outer review."""

    schema_version: Literal["content-agent-workflows.texture-critique.v1"] = (
        "content-agent-workflows.texture-critique.v1"
    )
    candidate_evidence: ExecutionArtifactBinding
    provider: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    findings: tuple[TextureCritiqueFinding, ...] = Field(min_length=1)
    operation_status: TextureOperationStatus


class TextureOuterUnitReview(_StrictFrozenModel):
    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    disposition: Literal["accept", "reject", "revise"]
    rationale: str = Field(min_length=1)


class _TextureOuterReviewFields(_StrictFrozenModel):
    """Fields shared by an outer review input and its recorded packet."""

    outer_plan: ExecutionArtifactBinding
    generation: ExecutionArtifactBinding
    candidate_evidence: ExecutionArtifactBinding
    advisory_critique: ExecutionArtifactBinding | None = None
    candidate: ExecutionArtifactBinding
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    provided_images: tuple[TextureProvidedImageArtifact, ...] = ()
    unit_reviews: tuple[TextureOuterUnitReview, ...] = Field(min_length=1)
    inspected_visual_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(
        min_length=1
    )
    findings: tuple[str, ...] = Field(min_length=1)


class TextureOuterReviewInput(_TextureOuterReviewFields):
    """Outer-authored semantic decision before deterministic recording."""

    schema_version: Literal["content-agent-workflows.texture-outer-review-input.v1"] = (
        "content-agent-workflows.texture-outer-review-input.v1"
    )


class TextureOuterReviewPacket(_TextureOuterReviewFields):
    """Validated outer multimodal decision over exact reference and OVRTX evidence."""

    schema_version: Literal["content-agent-workflows.texture-outer-review.v1"] = (
        "content-agent-workflows.texture-outer-review.v1"
    )
    operation_status: TextureOperationStatus


class TexturePublicationReceipt(_StrictFrozenModel):
    """Final deterministic receipt for the exact outer-accepted candidate."""

    schema_version: Literal[
        "content-agent-workflows.texture-publication-receipt.v1"
    ] = "content-agent-workflows.texture-publication-receipt.v1"
    request: ExecutionArtifactBinding
    preparation: ExecutionArtifactBinding
    outer_plan: ExecutionArtifactBinding
    generation: ExecutionArtifactBinding
    candidate_evidence: ExecutionArtifactBinding
    outer_review: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    accepted_candidate: ExecutionArtifactBinding
    published_asset: ExecutionArtifactBinding
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    provided_images: tuple[TextureProvidedImageArtifact, ...] = ()
    verification_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    operation_status: TextureOperationStatus


class TextureScopeInspector(Protocol):
    def inspect(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_dir: Path,
    ) -> TextureInspectionResult: ...


class TextureProposalProvider(Protocol):
    def plan(self, request: TextureWorkflowRequest) -> TexturePlanDocument: ...

    def export_resume_state(self, plan: TexturePlanDocument) -> dict[str, Any]: ...


class TextureGeneratorLeaf(Protocol):
    provider_id: str
    capability_id: str

    def generate(
        self, request: TextureGeneratorLeafRequest
    ) -> TextureExecutionResult: ...


class TextureEvidenceCollector(Protocol):
    def collect_candidate_evidence(
        self,
        *,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        output_asset_path: str,
        unit_artifacts: Mapping[str, TextureUnitArtifact],
        unit_ids: tuple[str, ...],
        output_dir: Path,
        reference_artifacts: tuple[TextureReferenceArtifact, ...],
        provided_images: tuple[TextureProvidedImageArtifact, ...] = (),
    ) -> tuple[
        tuple[TextureUnitRenderEvidence, ...],
        tuple[ExecutionArtifactBinding, ...],
        dict[str, Any],
    ]: ...


class TextureCritiqueProvider(Protocol):
    provider_id: str
    capability_id: str

    def critique(
        self,
        *,
        intent: str,
        evidence: TextureCandidateEvidencePacket,
    ) -> tuple[TextureCritiqueFinding, ...]: ...


class AssessorTextureCritiqueProvider:
    """Adapter from the existing typed visual assessor to advisory critique."""

    def __init__(
        self,
        *,
        assessor: Any,
        provider_id: str,
        capability_id: str,
        unit_context_by_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.assessor = assessor
        self.provider_id = provider_id
        self.capability_id = capability_id
        self.unit_context_by_id = dict(unit_context_by_id)

    def critique(
        self,
        *,
        intent: str,
        evidence: TextureCandidateEvidencePacket,
    ) -> tuple[TextureCritiqueFinding, ...]:
        findings: list[TextureCritiqueFinding] = []
        for unit in evidence.unit_evidence:
            assessment = self.assessor.assess(
                intent=intent,
                unit_id=unit.unit_id,
                unit_context=self.unit_context_by_id[unit.unit_id],
                source_image_paths=tuple(item.path for item in unit.source_images),
                output_image_paths=tuple(item.path for item in unit.candidate_images),
            )
            findings.append(
                TextureCritiqueFinding(
                    unit_id=unit.unit_id,
                    status=assessment.status,
                    summary=assessment.summary,
                )
            )
        return tuple(findings)


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Texture artifact must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"Texture artifact is not a regular file: {resolved}")
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _verify_binding(binding: ExecutionArtifactBinding, *, label: str) -> None:
    if _binding(binding.path) != binding:
        raise ValueError(f"{label} bytes changed: {binding.path}")


def _verify_packet_binding(
    binding: ExecutionArtifactBinding,
    packet: BaseModel,
    *,
    label: str,
) -> None:
    _verify_binding(binding, label=label)
    persisted = load_json(binding.path)
    type(packet).model_validate(persisted)
    if persisted != packet.model_dump(mode="json"):
        raise ValueError(f"{label} binding contains another typed packet")


def _verify_references(references: Sequence[TextureReferenceArtifact]) -> None:
    for reference in references:
        _verify_binding(reference.artifact, label=f"Texture reference {reference.role}")


def _verify_provided_images(
    provided_images: Sequence[TextureProvidedImageArtifact],
) -> None:
    for image in provided_images:
        _verify_binding(
            image.artifact,
            label=f"Texture provided image {image.unit_id}/{image.channel}",
        )


def _verify_frozen_provider_inputs(preparation: TexturePreparationPacket) -> None:
    """Rebind every external input immediately before a provider can read it."""

    request = preparation.request
    _verify_binding(request.source, label="Texture source")
    verify_usd_dependency_closure(request.source.path, request.source_dependencies)
    _verify_references(request.reference_artifacts)


def _verify_generation_artifacts(generation: TextureGenerationPacket) -> None:
    """Rebind every exact artifact that produced one Texture candidate."""

    _verify_binding(generation.candidate, label="Texture candidate")
    _verify_references(generation.reference_artifacts)
    _verify_provided_images(generation.provided_images)
    for binding in generation.generator_artifacts:
        _verify_binding(binding, label="Texture generator result")


def _verify_candidate_evidence_artifacts(
    evidence: TextureCandidateEvidencePacket,
) -> None:
    """Rebind every artifact an outer critic or reviewer is about to inspect."""

    _verify_binding(evidence.source, label="Texture evidence source")
    verify_usd_dependency_closure(
        evidence.source.path,
        evidence.source_dependencies,
    )
    _verify_binding(evidence.candidate, label="Texture evidence candidate")
    _verify_references(evidence.reference_artifacts)
    _verify_provided_images(evidence.provided_images)
    for unit in evidence.unit_evidence:
        for binding in (*unit.source_images, *unit.candidate_images):
            _verify_binding(binding, label=f"Texture visual evidence {unit.unit_id}")
    for binding in evidence.static_evidence:
        _verify_binding(binding, label="Texture static candidate evidence")


def _packet_binding(
    path: str | Path, model: type[BaseModel]
) -> ExecutionArtifactBinding:
    resolved = Path(path).expanduser().resolve()
    model.model_validate(load_json(resolved))
    return _binding(resolved)


def _write_packet(path: Path, payload: BaseModel) -> ExecutionArtifactBinding:
    atomic_write_json(path, payload)
    return _packet_binding(path, type(payload))


def _request_digest(request: TextureCapabilityRequest) -> str:
    return canonical_json_digest(request)


def _initial_status(
    selection: TextureOperationSelection,
    *,
    inspect_artifact: ExecutionArtifactBinding,
) -> TextureOperationStatus:
    outcomes: list[TextureOperationOutcome] = [
        TextureOperationOutcome(
            operation="inspect",
            state="completed",
            artifact=inspect_artifact,
            detail="Provider-free source, scope, UV, reference, and render inspection.",
        )
    ]
    operations: tuple[TextureOperationName, ...] = (
        "propose",
        "generate",
        "evidence",
        "critique",
        "review",
        "publish",
    )
    for operation in operations:
        selected = selection.state_for(operation)
        outcomes.append(
            TextureOperationOutcome(
                operation=operation,
                state=("not_evaluated" if selected == "requested" else "not_requested"),
                detail=(
                    "Selected by the outer plan but not evaluated yet."
                    if selected == "requested"
                    else "Not selected by the outer plan; no success is claimed."
                ),
            )
        )
    return TextureOperationStatus(operations=tuple(outcomes))


def _completed_status(
    previous: TextureOperationStatus,
    *,
    operation: TextureOperationName,
    artifact: ExecutionArtifactBinding | None = None,
) -> TextureOperationStatus:
    current = previous.outcome(operation)
    if current.state == "not_requested":
        raise ValueError(f"Texture {operation} was not requested by the frozen plan")
    if current.state == "completed":
        raise ValueError(f"Texture {operation} is already completed")
    return TextureOperationStatus(
        operations=tuple(
            TextureOperationOutcome(
                operation=item.operation,
                state="completed",
                artifact=artifact,
                detail=f"Focused Texture {operation} operation completed.",
            )
            if item.operation == operation
            else item
            for item in previous.operations
        )
    )


def build_texture_capability_request(
    *,
    source_asset: str | Path,
    output_dir: str | Path,
    intent: str,
    operations: TextureOperationSelection,
    material_prim_paths: Sequence[str] = (),
    prim_paths: Sequence[str] = (),
    reference_artifacts: Sequence[tuple[str, str | Path]] = (),
    texture_size: int = 1024,
    metadata: Mapping[str, Any] | None = None,
) -> TextureCapabilityRequest:
    """Bind a caller-authored task plan without constructing any provider."""

    source = _binding(source_asset)
    return TextureCapabilityRequest(
        source=source,
        source_dependencies=tuple(bind_usd_dependency_closure(source.path)),
        output_dir=str(Path(output_dir).expanduser().resolve()),
        intent=intent,
        material_prim_paths=tuple(material_prim_paths),
        prim_paths=tuple(prim_paths),
        reference_artifacts=tuple(
            TextureReferenceArtifact(role=role, artifact=_binding(path))
            for role, path in reference_artifacts
        ),
        operations=operations,
        texture_size=texture_size,
        metadata=dict(metadata or {}),
    )


def _workflow_request(request: TextureCapabilityRequest) -> TextureWorkflowRequest:
    metadata = {
        **request.metadata,
        "explicit_material_paths": list(request.material_prim_paths),
        "explicit_prim_paths": list(request.prim_paths),
        "texture_size": request.texture_size,
        "texture_backend": request.metadata.get(
            "texture_backend",
            "not_requested",
        ),
        "auto_prompt_enabled": True,
        "detail_policy": "surface_only",
        "discovery_mode": "explicit",
        "unit_mode": "per_material",
        "source_asset_binding": request.source.model_dump(mode="json"),
        "source_dependencies": [
            item.model_dump(mode="json") for item in request.source_dependencies
        ],
    }
    return TextureWorkflowRequest(
        source_asset=request.source.path,
        output_dir=Path(request.output_dir),
        intent=request.intent,
        reference_artifacts=request.reference_artifacts,
        metadata=metadata,
    )


def _provider_free_scope_plan(request: TextureCapabilityRequest) -> TexturePlanDocument:
    """Reuse the existing deterministic Texture planner with no provider defaults."""

    from texture_agent.functions.material_discovery import (
        discover_effective_materials_from_file,
    )
    from texture_agent.planning.contracts import TexturePlanRequest, TexturePlanSource
    from texture_agent.planning.planner import build_texture_plan

    discovery = discover_effective_materials_from_file(
        request.source.path,
        material_prim_paths=request.material_prim_paths or None,
        prim_scope_paths=request.prim_paths or None,
    )
    planning_request = TexturePlanRequest(
        source=TexturePlanSource(
            source_asset=request.source.path,
            source_asset_sha256=request.source.sha256,
        ),
        discovery_mode="explicit",
        unit_mode="per_material",
        explicit_material_paths=request.material_prim_paths,
        explicit_prim_paths=request.prim_paths,
        detail_policy="surface_only",
        texture_size=request.texture_size,
        backend="not_requested",
    )
    plan = build_texture_plan(
        planning_request,
        discovered_materials=discovery.authored_materials,
        effective_discovery=discovery,
        auto_prompt_enabled=True,
    )
    payload = plan.model_dump(mode="python")
    materials_by_path = {
        str(material.prim_path): material for material in discovery.authored_materials
    }
    for unit in payload["selected_units"]:
        aliases = {
            str(alias)
            for material_path in unit.get("material_prim_paths", ())
            if (material := materials_by_path.get(str(material_path))) is not None
            for alias in (material.prim_path, *material.material_alias_paths)
        }
        unit["material_alias_paths"] = sorted(aliases)
    return TexturePlanDocument.model_validate(payload)


def prepare_texture_scope(
    request: TextureCapabilityRequest,
    *,
    inspector: TextureScopeInspector,
    output_dir: str | Path | None = None,
    output_dir_claimed: bool = False,
    resume_incomplete: bool = False,
) -> tuple[TexturePreparationPacket, ExecutionArtifactBinding]:
    """Inspect Texture scope without constructing or calling semantic providers."""

    _verify_binding(request.source, label="Texture source")
    verify_usd_dependency_closure(request.source.path, request.source_dependencies)
    _verify_references(request.reference_artifacts)
    package_route_preflight = getattr(inspector, "preflight_package_route", None)
    if callable(package_route_preflight):
        # The production usd-cli inspector owns this admission boundary. Pin
        # its exact route before claiming the operation root or writing the
        # capability request; the same pin is propagated into every session.
        package_route_preflight()
    requested_root = request.output_dir if output_dir is None else output_dir
    operation_root = Path(requested_root).expanduser()
    scope_plan: TexturePlanDocument | None = None
    if resume_incomplete and operation_root.exists():
        if output_dir_claimed:
            raise ValueError(
                "Texture incomplete preparation cannot use a preclaimed root"
            )
        scope_plan = _provider_free_scope_plan(request)
        if (
            not scope_plan.decision.execution_allowed
            or not scope_plan.selected_unit_ids
        ):
            raise ValueError(
                "deterministic Texture scope inspection selected no usable units"
            )
        request_binding = _adopt_incomplete_texture_preparation_root(
            operation_root,
            request=request,
            selected_unit_ids=scope_plan.selected_unit_ids,
        )
    else:
        operation_root = _claim_texture_operation_root(
            requested_root,
            standalone_root=request.output_dir,
            preclaimed=output_dir_claimed,
        )
        request_binding = _write_packet(
            operation_root / "capability_request.json",
            request,
        )
    request_path = operation_root / "capability_request.json"
    if scope_plan is None:
        scope_plan = _provider_free_scope_plan(request)
    if not scope_plan.decision.execution_allowed or not scope_plan.selected_unit_ids:
        raise ValueError(
            "deterministic Texture scope inspection selected no usable units"
        )
    workflow_request = _workflow_request(request)
    inspection = inspector.inspect(
        request=workflow_request,
        plan=scope_plan,
        output_dir=operation_root,
    )
    if tuple(unit.unit_id for unit in inspection.units) != scope_plan.selected_unit_ids:
        raise ValueError(
            "Texture inspection does not exactly cover deterministic scope"
        )
    if inspection.proposal_plan_digest != texture_plan_digest(scope_plan):
        raise ValueError("Texture inspection binds another deterministic scope plan")
    if inspection.source != request.source:
        raise ValueError("Texture inspection binds another source")
    if inspection.reference_artifacts != request.reference_artifacts:
        raise ValueError("Texture inspection changed reference identities")
    if not inspection.renderer_metadata:
        raise ValueError("Texture inspection requires renderer provenance")
    for artifact in (
        *inspection.before_render_artifacts,
        *inspection.inspection_artifacts,
    ):
        _verify_binding(artifact, label="Texture preparation evidence")
    inspection_path = operation_root / "texture_inspection.json"
    inspection_binding = _write_packet(inspection_path, inspection)
    packet = TexturePreparationPacket(
        request=request,
        request_digest=_request_digest(request),
        scope_plan=scope_plan,
        scope_plan_digest=texture_plan_digest(scope_plan),
        inspection=inspection,
        operation_status=_initial_status(
            request.operations,
            inspect_artifact=inspection_binding,
        ),
    )
    packet_path = operation_root / "texture_preparation.json"
    packet_binding = _write_packet(packet_path, packet)
    # Rebind request after every write so its immutable bytes are already durable.
    if _packet_binding(request_path, TextureCapabilityRequest) != request_binding:
        raise ValueError("Texture capability request changed during preparation")
    return packet, packet_binding


def _adopt_incomplete_texture_preparation_root(
    operation_root: Path,
    *,
    request: TextureCapabilityRequest,
    selected_unit_ids: tuple[str, ...],
) -> ExecutionArtifactBinding:
    """Adopt one request-bound root before any factual evidence was produced."""

    if not operation_root.is_absolute() or ".." in operation_root.parts:
        raise ValueError("Texture graph operation root must be canonical and absolute")
    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        # Public release/runtime support is Linux, Linux containers, and WSL2.
        # Do not silently fall back to path-based traversal or cleanup on a host
        # that cannot retain the exact admitted root through a descriptor path.
        raise ValueError(
            "Texture incomplete preparation requires descriptor-confined Linux "
            "filesystem inspection"
        )
    try:
        with open_confined_directory(operation_root) as root_descriptor:
            root_metadata = os.fstat(root_descriptor)
            effective_uid = getattr(os, "geteuid", lambda: root_metadata.st_uid)()
            if (
                not stat.S_ISDIR(root_metadata.st_mode)
                or root_metadata.st_uid != effective_uid
                or stat.S_IMODE(root_metadata.st_mode) != 0o700
            ):
                raise ValueError(
                    "Texture incomplete preparation root must be a mode-0700 directory"
                )
            return _adopt_held_incomplete_texture_preparation_root(
                operation_root,
                root_descriptor=root_descriptor,
                request=request,
                selected_unit_ids=selected_unit_ids,
                effective_uid=effective_uid,
            )
    except (ArtifactPathError, OSError) as exc:
        raise ValueError(
            "Texture incomplete preparation root has an unavailable or symlinked "
            f"directory chain: {operation_root}"
        ) from exc


def _adopt_held_incomplete_texture_preparation_root(
    operation_root: Path,
    *,
    root_descriptor: int,
    request: TextureCapabilityRequest,
    selected_unit_ids: tuple[str, ...],
    effective_uid: int,
) -> ExecutionArtifactBinding:
    """Inspect and reconstruct one root while its no-follow chain is held."""

    inspected_root = Path("/proc/self/fd") / str(root_descriptor)

    request_path = inspected_root / "capability_request.json"
    inspection_root = inspected_root / "usd_cli_inspection"
    inspection_source_root = inspection_root / "source"
    inspection_source_asset_root = inspection_root / "source_asset"
    usd_cli_raw_root = inspection_source_root / "raw"
    usd_cli_state_root = inspection_source_root / ".usd-cli"
    usd_cli_marker_path = usd_cli_state_root / ".workflow-owned"
    usd_cli_config_path = usd_cli_state_root / "config.toml"
    if (
        not selected_unit_ids
        or len(selected_unit_ids) != len(set(selected_unit_ids))
        or any(
            not unit_id
            or Path(unit_id).name != unit_id
            or unit_id in {".", "..", ".usd-cli"}
            for unit_id in selected_unit_ids
        )
    ):
        raise ValueError("Texture incomplete preparation selected unit IDs are unsafe")
    expected_unit_directories = {
        inspection_source_root / unit_id for unit_id in selected_unit_ids
    }
    allowed_directories = {
        inspection_root,
        inspection_source_root,
        inspection_source_asset_root,
        usd_cli_raw_root,
        usd_cli_state_root,
        *expected_unit_directories,
    }
    allowed_files = {
        request_path,
        usd_cli_marker_path,
        usd_cli_config_path,
    }
    observed_directories: set[Path] = set()
    observed_files: set[Path] = set()
    observed_source_asset_files: set[Path] = set()
    observed_unit_directories: set[Path] = set()

    def reject_walk_error(error: OSError) -> None:
        raise ValueError(
            f"Texture incomplete preparation root cannot be inspected: {operation_root}"
        ) from error

    for current, directory_names, file_names in os.walk(
        inspected_root,
        topdown=True,
        followlinks=False,
        onerror=reject_walk_error,
    ):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            metadata = candidate.lstat()
            if candidate not in allowed_directories:
                raise ValueError(
                    "Texture incomplete preparation root contains an unexpected "
                    f"artifact: {candidate}"
                )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != effective_uid
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ValueError(
                    "Texture incomplete preparation root contains an unsafe "
                    f"directory: {candidate}"
                )
            observed_directories.add(candidate)
            if candidate in expected_unit_directories:
                observed_unit_directories.add(candidate)
        for name in file_names:
            candidate = current_path / name
            metadata = candidate.lstat()
            is_source_asset = candidate.parent == inspection_source_asset_root
            if (
                (candidate not in allowed_files and not is_source_asset)
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != effective_uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ValueError(
                    "Texture incomplete preparation root contains an unexpected "
                    f"artifact: {candidate}"
                )
            observed_files.add(candidate)
            if is_source_asset:
                observed_source_asset_files.add(candidate)
    if not request_path.is_file() or request_path.is_symlink():
        raise ValueError(
            "Texture incomplete preparation root omitted its frozen capability request"
        )
    try:
        persisted = TextureCapabilityRequest.model_validate(load_json(request_path))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Texture incomplete preparation capability request is corrupt"
        ) from exc
    if persisted != request:
        raise ValueError(
            "Texture incomplete preparation capability request differs from resume"
        )
    request_binding = _packet_binding(request_path, TextureCapabilityRequest)
    if (
        inspection_source_root in observed_directories
        and observed_unit_directories != expected_unit_directories
    ):
        raise ValueError(
            "Texture incomplete preparation source omitted its exact selected-unit "
            "directory footprint"
        )
    marker_present = usd_cli_marker_path in observed_files
    config_present = usd_cli_config_path in observed_files
    if config_present and not marker_present:
        raise ValueError(
            "Texture incomplete preparation usd-cli config omitted its ownership marker"
        )
    if marker_present:
        try:
            marker = read_contained_artifact(
                inspected_root,
                usd_cli_marker_path.relative_to(inspected_root),
                max_bytes=64,
                capture_bytes=True,
            )
        except ValueError as exc:
            raise ValueError(
                "Texture incomplete preparation usd-cli ownership marker is unsafe"
            ) from exc
        if marker.data != b"texture-validation\n":
            raise ValueError(
                "Texture incomplete preparation usd-cli ownership marker differs "
                "from Texture validation"
            )
    if config_present:
        try:
            read_contained_artifact(
                inspected_root,
                usd_cli_config_path.relative_to(inspected_root),
                max_bytes=65536,
            )
        except ValueError as exc:
            raise ValueError(
                "Texture incomplete preparation usd-cli config is unsafe"
            ) from exc
    if inspection_source_asset_root in observed_directories:
        if len(observed_source_asset_files) != 1:
            raise ValueError(
                "Texture incomplete preparation source_asset directory must contain "
                "exactly one frozen source file"
            )
        staged_source = next(iter(observed_source_asset_files))
        try:
            staged = read_contained_artifact(
                inspected_root,
                staged_source.relative_to(inspected_root),
                max_bytes=request.source.size_bytes,
            )
        except ValueError as exc:
            raise ValueError(
                "Texture incomplete preparation source_asset is unsafe"
            ) from exc
        if (
            staged.sha256 != request.source.sha256
            or staged.size_bytes != request.source.size_bytes
        ):
            raise ValueError(
                "Texture incomplete preparation source_asset differs from the "
                "frozen source"
            )
    if inspection_root in observed_directories:
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError(
                "Texture incomplete preparation cannot safely reconstruct usd-cli "
                "state on this platform"
            )
        shutil.rmtree("usd_cli_inspection", dir_fd=root_descriptor)
    return request_binding


def validate_texture_preparation(
    preparation: TexturePreparationPacket,
    *,
    preparation_binding: ExecutionArtifactBinding,
    expected_request: TextureCapabilityRequest | None = None,
) -> None:
    """Replay every deterministic identity in one cached preparation packet."""

    _verify_packet_binding(
        preparation_binding,
        preparation,
        label="Texture preparation",
    )
    if expected_request is not None and preparation.request != expected_request:
        raise ValueError("Texture preparation request differs from the frozen request")
    _verify_binding(preparation.request.source, label="Texture preparation source")
    verify_usd_dependency_closure(
        preparation.request.source.path,
        preparation.request.source_dependencies,
    )
    _verify_references(preparation.request.reference_artifacts)
    if preparation.request_digest != _request_digest(preparation.request):
        raise ValueError("Texture preparation request digest is stale")
    if preparation.scope_plan_digest != texture_plan_digest(preparation.scope_plan):
        raise ValueError("Texture preparation scope-plan digest is stale")
    inspection = preparation.inspection
    if (
        inspection.source != preparation.request.source
        or inspection.reference_artifacts != preparation.request.reference_artifacts
        or inspection.proposal_plan_digest != preparation.scope_plan_digest
        or tuple(item.unit_id for item in inspection.units)
        != preparation.scope_plan.selected_unit_ids
    ):
        raise ValueError("Texture preparation inspection binds another request")
    if not inspection.renderer_metadata:
        raise ValueError("Texture preparation inspection requires renderer provenance")
    for artifact in (
        *inspection.before_render_artifacts,
        *inspection.inspection_artifacts,
    ):
        _verify_binding(artifact, label="Texture preparation evidence")
    inspect_outcome = preparation.operation_status.outcome("inspect")
    if inspect_outcome.state != "completed" or inspect_outcome.artifact is None:
        raise ValueError("Texture preparation inspection outcome is incomplete")
    _verify_packet_binding(
        inspect_outcome.artifact,
        inspection,
        label="Texture preparation inspection",
    )


def request_texture_provider_proposal(
    preparation: TexturePreparationPacket,
    *,
    preparation_binding: ExecutionArtifactBinding,
    provider: TextureProposalProvider,
    provider_id: str,
) -> tuple[TextureProviderProposalPacket, ExecutionArtifactBinding]:
    """Invoke one explicitly selected advisory proposal provider."""

    _verify_packet_binding(
        preparation_binding,
        preparation,
        label="Texture preparation",
    )
    if preparation.operation_status.outcome("propose").state == "not_requested":
        raise ValueError("Texture proposal was not requested by the frozen plan")
    provider_id = str(provider_id).strip()
    if not provider_id:
        raise ValueError("Texture proposal provider ID must not be empty")
    request = _workflow_request(preparation.request)
    _verify_frozen_provider_inputs(preparation)
    proposal = provider.plan(request)
    proposal_path = (
        Path(preparation.request.output_dir) / "texture_provider_proposal_payload.json"
    )
    proposal_binding = _write_packet(proposal_path, proposal)
    packet = TextureProviderProposalPacket(
        preparation=preparation_binding,
        provider=provider_id,
        proposal=proposal,
        proposal_digest=texture_plan_digest(proposal),
        provider_resume_state=provider.export_resume_state(proposal),
        operation_status=_completed_status(
            preparation.operation_status,
            operation="propose",
            artifact=proposal_binding,
        ),
    )
    path = Path(preparation.request.output_dir) / "texture_provider_proposal.json"
    binding = _write_packet(path, packet)
    return packet, binding


def validate_texture_outer_plan(
    outer_plan: TextureOuterPlan,
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
) -> None:
    """Validate outer-owned semantics without comparing them to a proposal."""

    _verify_packet_binding(
        preparation_binding,
        preparation,
        label="Texture preparation",
    )
    if outer_plan.preparation != preparation_binding:
        raise ValueError("outer Texture plan binds another preparation packet")
    if outer_plan.source != preparation.request.source:
        raise ValueError("outer Texture plan source differs from preparation")
    if outer_plan.scope_plan_digest != preparation.scope_plan_digest:
        raise ValueError("outer Texture plan binds stale deterministic scope")
    if outer_plan.reference_artifacts != preparation.request.reference_artifacts:
        raise ValueError("outer Texture plan reference identity differs from request")
    if outer_plan.operations != preparation.request.operations:
        raise ValueError("outer Texture plan changed the frozen operation selection")
    if (
        outer_plan.capability_constraints
        != preparation.inspection.capability_constraints
    ):
        raise ValueError("outer Texture plan changed capability constraints")
    if outer_plan.advisory_proposal is not None:
        _verify_binding(
            outer_plan.advisory_proposal,
            label="Texture advisory provider proposal",
        )
    inspected = {unit.unit_id: unit for unit in preparation.inspection.units}
    if outer_plan.target_unit_ids != tuple(inspected):
        raise ValueError("outer Texture plan must cover exact inspected unit order")
    for target in outer_plan.targets:
        unit = inspected[target.unit_id]
        if unit.uv_status != "ready":
            raise ValueError(
                f"outer Texture plan cannot generate unavailable UV scope: "
                f"{target.unit_id}={unit.uv_status}"
            )
        if (
            target.material_prim_paths != unit.material_prim_paths
            or target.member_prim_paths != unit.member_prim_paths
            or target.member_subset_paths != unit.member_subset_paths
        ):
            raise ValueError(f"outer Texture target drifted: {target.unit_id}")
        if target.generator_inputs.reference_artifacts != tuple(
            item.artifact for item in outer_plan.reference_artifacts
        ):
            raise ValueError(
                f"outer Texture generator references are incomplete: {target.unit_id}"
            )
        provided_images = target.generator_inputs.provided_images
        if outer_plan.execution_mode == "provider_generate":
            if provided_images:
                raise ValueError(
                    "provider-generated Texture plans cannot bind provided images"
                )
        elif (
            len(provided_images) != 1
            or provided_images[0].unit_id != target.unit_id
            or provided_images[0].channel != "albedo"
        ):
            raise ValueError(
                "apply_provided requires one ordered albedo image per target unit"
            )
    if outer_plan.execution_mode == "apply_provided":
        provided_unit_ids = tuple(item.unit_id for item in outer_plan.provided_images)
        if provided_unit_ids != outer_plan.target_unit_ids:
            raise ValueError(
                "provided Texture images must exactly match target unit order"
            )
        provided_paths = tuple(
            item.artifact.path for item in outer_plan.provided_images
        )
        if len(provided_paths) != len(set(provided_paths)):
            raise ValueError("provided Texture image paths must be unique")


def _claim_texture_operation_root(
    output_dir: str | Path | None,
    *,
    standalone_root: str | Path,
    preclaimed: bool = False,
) -> Path:
    if output_dir is None:
        # Standalone capability calls retain their caller-owned run directory.
        # It may already contain earlier operation artifacts, and its existing
        # permissions must not be changed here.
        operation_root = Path(standalone_root)
        operation_root.mkdir(parents=True, exist_ok=True)
        return operation_root

    operation_root = Path(output_dir).expanduser()
    if not operation_root.is_absolute() or ".." in operation_root.parts:
        raise ValueError("Texture graph operation root must be canonical and absolute")
    if preclaimed:
        if (
            operation_root.parent != Path("/proc/self/fd")
            or not operation_root.name.isdigit()
        ):
            raise ValueError(
                "Texture preclaimed operation root must be one held descriptor"
            )
        metadata = operation_root.stat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                f"Texture graph operation root is not a directory: {operation_root}"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ValueError(
                f"Texture graph operation root must have mode 0700: {operation_root}"
            )
        return operation_root
    # Explicit roots are graph-owned attempt/native directories. Claim the
    # exact path atomically: never resolve through or replace a symlink, and
    # never accept a root another process precreated.
    try:
        operation_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Texture graph operation root already exists: {operation_root}"
        ) from exc
    return operation_root


def invoke_texture_generator(
    outer_plan: TextureOuterPlan,
    *,
    outer_plan_binding: ExecutionArtifactBinding,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    generator: TextureGeneratorLeaf,
    proposal: TextureProviderProposalPacket | None = None,
    provider_proposal: ExecutionArtifactBinding | None = None,
    output_dir: str | Path | None = None,
    output_dir_claimed: bool = False,
) -> tuple[TextureGenerationPacket, ExecutionArtifactBinding]:
    """Invoke exactly one explicit novel-texture generator leaf."""

    validate_texture_outer_plan(
        outer_plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    _verify_packet_binding(
        outer_plan_binding,
        outer_plan,
        label="outer Texture plan",
    )
    _verify_references(outer_plan.reference_artifacts)
    _verify_provided_images(outer_plan.provided_images)
    if (proposal is None) != (provider_proposal is None):
        raise ValueError(
            "Texture proposal packet and binding must be supplied together"
        )
    if proposal is not None and preparation.request.operations.propose != "requested":
        raise ValueError("Texture proposal was not requested by the frozen plan")
    if preparation.request.operations.propose == "requested" and proposal is None:
        raise ValueError("requested Texture proposal must complete before generation")
    prior_status = preparation.operation_status
    if proposal is not None and provider_proposal is not None:
        _verify_packet_binding(
            provider_proposal,
            proposal,
            label="Texture provider proposal",
        )
        if proposal.preparation != preparation_binding:
            raise ValueError("Texture provider proposal binds another preparation")
        if outer_plan.advisory_proposal != provider_proposal:
            raise ValueError("outer Texture plan binds another advisory proposal")
        prior_status = proposal.operation_status
    elif outer_plan.advisory_proposal is not None:
        raise ValueError("outer Texture plan names an unavailable advisory proposal")
    provider_id = str(generator.provider_id).strip()
    capability_id = str(generator.capability_id).strip()
    if not provider_id or not capability_id:
        raise ValueError("Texture generator leaf must declare provider and capability")
    inputs = tuple(target.generator_inputs for target in outer_plan.targets)
    if any(item.backend != provider_id for item in inputs):
        raise ValueError("outer Texture generator backend differs from selected leaf")
    operation_root = _claim_texture_operation_root(
        output_dir,
        standalone_root=preparation.request.output_dir,
        preclaimed=output_dir_claimed,
    )
    leaf_request = TextureGeneratorLeafRequest(
        outer_plan=outer_plan_binding,
        preparation=preparation_binding,
        source=preparation.request.source,
        source_dependencies=preparation.request.source_dependencies,
        intent=preparation.request.intent,
        scope_plan=preparation.scope_plan,
        target_unit_ids=outer_plan.target_unit_ids,
        generator_inputs=inputs,
        reference_artifacts=outer_plan.reference_artifacts,
        output_dir=str(operation_root / "candidate"),
    )
    _verify_frozen_provider_inputs(preparation)
    _verify_provided_images(outer_plan.provided_images)
    execution = generator.generate(leaf_request)
    if execution.requested_unit_ids != outer_plan.target_unit_ids:
        raise ValueError("Texture generator returned another unit scope")
    if tuple(item.unit_id for item in execution.unit_artifacts) != (
        outer_plan.target_unit_ids
    ):
        raise ValueError("Texture generator artifacts differ from outer unit order")
    candidate = _binding(execution.output_asset_path)
    generator_artifacts = tuple(
        _binding(path)
        for artifact in execution.unit_artifacts
        for path in artifact.artifact_paths
    )
    _verify_provided_images(outer_plan.provided_images)
    packet = TextureGenerationPacket(
        preparation=preparation_binding,
        outer_plan=outer_plan_binding,
        provider_proposal=provider_proposal,
        generator_provider=provider_id,
        generator_capability=capability_id,
        execution_mode=outer_plan.execution_mode,
        generator_inputs=inputs,
        provided_images=outer_plan.provided_images,
        execution=execution,
        candidate=candidate,
        unit_artifacts=execution.unit_artifacts,
        generator_artifacts=generator_artifacts,
        reference_artifacts=outer_plan.reference_artifacts,
        operation_status=_completed_status(
            prior_status,
            operation="generate",
            artifact=candidate,
        ),
    )
    path = operation_root / "texture_generation.json"
    binding = _write_packet(path, packet)
    return packet, binding


def collect_texture_candidate_evidence(
    outer_plan: TextureOuterPlan,
    generation: TextureGenerationPacket,
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    outer_plan_binding: ExecutionArtifactBinding,
    generation_binding: ExecutionArtifactBinding,
    collector: TextureEvidenceCollector,
    output_dir: str | Path | None = None,
    output_dir_claimed: bool = False,
) -> tuple[TextureCandidateEvidencePacket, ExecutionArtifactBinding]:
    """Collect deterministic static and OVRTX evidence without semantic critique."""

    validate_texture_outer_plan(
        outer_plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    _verify_packet_binding(
        outer_plan_binding,
        outer_plan,
        label="outer Texture plan",
    )
    _verify_packet_binding(
        generation_binding,
        generation,
        label="Texture generation",
    )
    if generation.outer_plan != outer_plan_binding:
        raise ValueError("Texture generation binds another outer plan")
    if generation.preparation != preparation_binding:
        raise ValueError("Texture generation binds another preparation")
    if generation.reference_artifacts != outer_plan.reference_artifacts:
        raise ValueError("Texture generation changed reference identities")
    if (
        generation.execution_mode != outer_plan.execution_mode
        or generation.provided_images != outer_plan.provided_images
    ):
        raise ValueError("Texture generation changed provided-image identities")
    if generation.candidate.path != str(
        Path(generation.execution.output_asset_path).expanduser().resolve()
    ):
        raise ValueError("Texture generation candidate differs from executor output")
    if generation.unit_artifacts != generation.execution.unit_artifacts:
        raise ValueError("Texture generation artifacts differ from executor output")
    _verify_generation_artifacts(generation)
    verify_usd_dependency_closure(
        preparation.request.source.path,
        preparation.request.source_dependencies,
    )
    workflow_request = _workflow_request(preparation.request)
    unit_artifacts = {item.unit_id: item for item in generation.unit_artifacts}
    operation_root = _claim_texture_operation_root(
        output_dir,
        standalone_root=preparation.request.output_dir,
        preclaimed=output_dir_claimed,
    )
    collector_kwargs: dict[str, Any] = {
        "request": workflow_request,
        "plan": preparation.scope_plan,
        "output_asset_path": generation.candidate.path,
        "unit_artifacts": unit_artifacts,
        "unit_ids": outer_plan.target_unit_ids,
        "output_dir": operation_root,
        "reference_artifacts": outer_plan.reference_artifacts,
    }
    if outer_plan.provided_images:
        collector_kwargs["provided_images"] = outer_plan.provided_images
    unit_evidence, static_evidence, renderer_metadata = (
        collector.collect_candidate_evidence(**collector_kwargs)
    )
    if tuple(item.unit_id for item in unit_evidence) != outer_plan.target_unit_ids:
        raise ValueError("Texture candidate evidence does not cover exact target order")
    if not static_evidence:
        raise ValueError("Texture candidate evidence requires static evidence")
    _verify_frozen_provider_inputs(preparation)
    _verify_provided_images(outer_plan.provided_images)
    _verify_generation_artifacts(generation)
    prior = generation.operation_status
    packet = TextureCandidateEvidencePacket(
        preparation=preparation_binding,
        outer_plan=outer_plan_binding,
        generation=generation_binding,
        source=preparation.request.source,
        source_dependencies=preparation.request.source_dependencies,
        candidate=generation.candidate,
        reference_artifacts=outer_plan.reference_artifacts,
        provided_images=outer_plan.provided_images,
        unit_evidence=unit_evidence,
        static_evidence=static_evidence,
        renderer_metadata=renderer_metadata,
        operation_status=_completed_status(
            prior,
            operation="evidence",
            artifact=static_evidence[0],
        ),
    )
    _verify_candidate_evidence_artifacts(packet)
    path = operation_root / "texture_candidate_evidence.json"
    binding = _write_packet(path, packet)
    return packet, binding


def request_texture_critique(
    evidence: TextureCandidateEvidencePacket,
    *,
    evidence_binding: ExecutionArtifactBinding,
    intent: str,
    provider: TextureCritiqueProvider,
) -> tuple[TextureCritiquePacket, ExecutionArtifactBinding]:
    """Invoke an optional advisory critic over already-collected evidence."""

    _verify_packet_binding(
        evidence_binding,
        evidence,
        label="Texture candidate evidence",
    )
    if evidence.operation_status.outcome("critique").state == "not_requested":
        raise ValueError("Texture critique was not requested by the frozen plan")
    _verify_candidate_evidence_artifacts(evidence)
    _verify_provided_images(evidence.provided_images)
    provider_id = str(provider.provider_id).strip()
    capability_id = str(provider.capability_id).strip()
    if not provider_id or not capability_id:
        raise ValueError("Texture critique leaf must declare provider and capability")
    findings = provider.critique(intent=intent, evidence=evidence)
    if tuple(item.unit_id for item in findings) != tuple(
        item.unit_id for item in evidence.unit_evidence
    ):
        raise ValueError("Texture critique does not cover exact evidence unit order")
    findings_path = (
        Path(evidence.preparation.path).parent / "texture_critique_findings.json"
    )
    atomic_write_json(
        findings_path,
        {
            "schema_version": "content-agent-workflows.texture-critique-findings.v1",
            "provider": provider_id,
            "capability": capability_id,
            "findings": [item.model_dump(mode="json") for item in findings],
        },
    )
    packet = TextureCritiquePacket(
        candidate_evidence=evidence_binding,
        provider=provider_id,
        capability=capability_id,
        findings=findings,
        operation_status=_completed_status(
            evidence.operation_status,
            operation="critique",
            artifact=_binding(findings_path),
        ),
    )
    path = Path(evidence.preparation.path).parent / "texture_critique.json"
    binding = _write_packet(path, packet)
    return packet, binding


def record_texture_outer_review(
    review: TextureOuterReviewInput,
    *,
    outer_plan: TextureOuterPlan,
    generation: TextureGenerationPacket,
    evidence: TextureCandidateEvidencePacket,
    critique: TextureCritiquePacket | None = None,
    critique_binding: ExecutionArtifactBinding | None = None,
    output_dir: str | Path | None = None,
    output_dir_claimed: bool = False,
) -> tuple[TextureOuterReviewPacket, ExecutionArtifactBinding]:
    """Validate and persist one exact outer-authored visual decision."""

    _verify_packet_binding(
        review.outer_plan,
        outer_plan,
        label="outer Texture plan",
    )
    _verify_packet_binding(
        review.generation,
        generation,
        label="Texture generation",
    )
    _verify_packet_binding(
        review.candidate_evidence,
        evidence,
        label="Texture candidate evidence",
    )
    _verify_candidate_evidence_artifacts(evidence)
    if (
        review.candidate != generation.candidate
        or review.candidate != evidence.candidate
    ):
        raise ValueError("outer Texture review substituted another candidate")
    if review.reference_artifacts != outer_plan.reference_artifacts:
        raise ValueError("outer Texture review changed reference identities")
    if (
        review.provided_images != outer_plan.provided_images
        or evidence.provided_images != outer_plan.provided_images
        or generation.provided_images != outer_plan.provided_images
    ):
        raise ValueError("outer Texture review changed provided-image identities")
    _verify_references(review.reference_artifacts)
    _verify_provided_images(review.provided_images)
    if (
        tuple(item.unit_id for item in review.unit_reviews)
        != outer_plan.target_unit_ids
    ):
        raise ValueError("outer Texture review must cover every target in exact order")
    required_visuals = tuple(
        binding
        for unit in evidence.unit_evidence
        for binding in (*unit.source_images, *unit.candidate_images)
    )
    required_visuals = (
        *(item.artifact for item in review.reference_artifacts),
        *(item.artifact for item in review.provided_images),
        *required_visuals,
    )
    if review.inspected_visual_artifacts != required_visuals:
        raise ValueError("outer Texture review visual evidence is stale or partial")
    if review.advisory_critique is not None:
        if (
            critique is None
            or critique_binding is None
            or review.advisory_critique != critique_binding
        ):
            raise ValueError("outer Texture review binds an invalid critique")
        _verify_packet_binding(
            critique_binding,
            critique,
            label="Texture advisory critique",
        )
        if critique.candidate_evidence != review.candidate_evidence:
            raise ValueError("outer Texture critique binds another candidate evidence")
    elif critique is not None or critique_binding is not None:
        raise ValueError("outer Texture review omitted the supplied advisory critique")
    if outer_plan.operations.critique == "requested" and critique is None:
        raise ValueError("requested Texture critique must complete before outer review")
    if all(item.disposition == "accept" for item in review.unit_reviews) is False:
        # Rejection/refinement is a valid review result, but cannot flow to publish.
        pass
    prior_status = (
        critique.operation_status if critique is not None else evidence.operation_status
    )
    operation_root = _claim_texture_operation_root(
        output_dir,
        standalone_root=Path(review.outer_plan.path).parent,
        preclaimed=output_dir_claimed,
    )
    path = operation_root / "texture_outer_review.json"
    completed = TextureOuterReviewPacket(
        **review.model_dump(mode="python", exclude={"schema_version"}),
        operation_status=_completed_status(prior_status, operation="review"),
    )
    binding = _write_packet(path, completed)
    return completed, binding


def publish_texture_candidate(
    review: TextureOuterReviewPacket,
    *,
    review_binding: ExecutionArtifactBinding,
    request_binding: ExecutionArtifactBinding,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    outer_plan: TextureOuterPlan,
    outer_plan_binding: ExecutionArtifactBinding,
    generation: TextureGenerationPacket,
    generation_binding: ExecutionArtifactBinding,
    evidence: TextureCandidateEvidencePacket,
    evidence_binding: ExecutionArtifactBinding,
    publication_path: str | Path,
    output_dir: str | Path | None = None,
    output_dir_claimed: bool = False,
) -> tuple[TexturePublicationReceipt, ExecutionArtifactBinding]:
    """Publish and read back only the exact candidate accepted by the outer review."""

    for binding, input_packet, label in (
        (review_binding, review, "outer Texture review"),
        (preparation_binding, preparation, "Texture preparation"),
        (outer_plan_binding, outer_plan, "outer Texture plan"),
        (generation_binding, generation, "Texture generation"),
        (evidence_binding, evidence, "Texture candidate evidence"),
    ):
        _verify_packet_binding(binding, input_packet, label=label)
    _verify_binding(request_binding, label="Texture capability request")
    persisted_request = TextureCapabilityRequest.model_validate(
        load_json(request_binding.path)
    )
    if persisted_request != preparation.request:
        raise ValueError("Texture publication request differs from preparation")
    if review.operation_status.outcome("review").state != "completed":
        raise ValueError("Texture publication requires a completed outer review")
    if outer_plan.operations.publish != "requested":
        raise ValueError("Texture publication was not requested by the frozen plan")
    if any(item.disposition != "accept" for item in review.unit_reviews):
        raise ValueError("Texture publication requires outer acceptance of every unit")
    if (
        review.outer_plan != outer_plan_binding
        or review.generation != generation_binding
        or review.candidate_evidence != evidence_binding
    ):
        raise ValueError("Texture publication inputs differ from the outer review")
    if (
        generation.candidate != evidence.candidate
        or review.candidate != evidence.candidate
    ):
        raise ValueError("Texture publication candidate identity changed")
    operations: tuple[TextureOperationName, ...] = (
        "inspect",
        "propose",
        "generate",
        "evidence",
        "critique",
        "review",
    )
    for operation in operations:
        if (
            outer_plan.operations.state_for(operation) == "requested"
            and review.operation_status.outcome(operation).state != "completed"
        ):
            raise ValueError(
                f"requested Texture {operation} operation is not completed"
            )
    _verify_candidate_evidence_artifacts(evidence)
    _verify_binding(preparation.request.source, label="Texture source")
    verify_usd_dependency_closure(
        preparation.request.source.path,
        preparation.request.source_dependencies,
    )
    _verify_references(outer_plan.reference_artifacts)
    if (
        generation.provided_images != outer_plan.provided_images
        or evidence.provided_images != outer_plan.provided_images
        or review.provided_images != outer_plan.provided_images
    ):
        raise ValueError("Texture publication changed provided-image identities")
    _verify_generation_artifacts(generation)
    candidate_dependencies = tuple(bind_usd_dependency_closure(evidence.candidate.path))
    if (
        candidate_dependencies
        and Path(evidence.candidate.path).suffix.lower() != ".usdz"
    ):
        raise ValueError(
            "Texture publication requires root-only USD or byte-self-contained USDZ"
        )
    report = validate_texture_scope_invariants(
        source_asset_path=preparation.request.source.path,
        output_asset_path=evidence.candidate.path,
        plan=preparation.scope_plan,
    )
    if not report.passed:
        raise ValueError("Texture publication failed deterministic scope validation")
    operation_root = _claim_texture_operation_root(
        output_dir,
        standalone_root=preparation.request.output_dir,
        preclaimed=output_dir_claimed,
    )
    output = Path(publication_path).expanduser().resolve()
    usd_suffixes = {".usd", ".usda", ".usdc", ".usdz"}
    if Path(evidence.candidate.path).suffix.lower() not in usd_suffixes:
        raise ValueError("Texture candidate must be a USD or USDZ asset")
    if output.suffix.lower() not in usd_suffixes:
        raise ValueError("Texture publication path must be a USD or USDZ asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Texture publication path already exists: {output}")
    staging = output.with_name(f".{output.name}.texture-staging{output.suffix}")
    if staging.exists():
        raise FileExistsError(f"Texture publication staging path exists: {staging}")
    staging_created = False
    publication_created = False
    verification_path = operation_root / "publication_validation.json"
    receipt_path = operation_root / "texture_publication_receipt.json"
    verification_existed = verification_path.exists()
    receipt_existed = receipt_path.exists()
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with Path(evidence.candidate.path).open("rb") as source_stream:
            descriptor = os.open(staging, flags, 0o600)
            staging_created = True
            with os.fdopen(descriptor, "wb") as staging_stream:
                shutil.copyfileobj(source_stream, staging_stream)
        staged = _binding(staging)
        if staged.sha256 != evidence.candidate.sha256:
            raise ValueError(
                "saved Texture publication differs from accepted candidate"
            )
        readback = validate_texture_scope_invariants(
            source_asset_path=preparation.request.source.path,
            output_asset_path=staged.path,
            plan=preparation.scope_plan,
        )
        if not readback.passed:
            raise ValueError("saved Texture publication failed stage readback")
        staged_dependencies = tuple(bind_usd_dependency_closure(staged.path))
        if staged_dependencies and staging.suffix.lower() != ".usdz":
            raise ValueError("saved Texture publication has external dependencies")
        _verify_frozen_provider_inputs(preparation)
        _verify_provided_images(outer_plan.provided_images)
        _verify_generation_artifacts(generation)
        os.link(staging, output)
        publication_created = True
        staging.unlink()
        staging_created = False
        published = _binding(output)
        if published.sha256 != evidence.candidate.sha256:
            raise ValueError("published Texture bytes changed after atomic promotion")
        _verify_frozen_provider_inputs(preparation)
        _verify_provided_images(outer_plan.provided_images)
        _verify_generation_artifacts(generation)
        atomic_write_json(
            verification_path,
            {
                "schema_version": (
                    "content-agent-workflows.texture-publication-validation.v1"
                ),
                "candidate_sha256": evidence.candidate.sha256,
                "published_sha256": published.sha256,
                "source_sha256": preparation.request.source.sha256,
                "scope_plan_digest": preparation.scope_plan_digest,
                "reference_artifacts": [
                    item.model_dump(mode="json")
                    for item in outer_plan.reference_artifacts
                ],
                "provided_images": [
                    item.model_dump(mode="json") for item in outer_plan.provided_images
                ],
                "pre_copy_scope": report.model_dump(mode="json"),
                "saved_stage_readback": readback.model_dump(mode="json"),
            },
        )
        verification = _binding(verification_path)
        packet = TexturePublicationReceipt(
            request=request_binding,
            preparation=preparation_binding,
            outer_plan=outer_plan_binding,
            generation=generation_binding,
            candidate_evidence=evidence_binding,
            outer_review=review_binding,
            source=preparation.request.source,
            source_dependencies=preparation.request.source_dependencies,
            accepted_candidate=evidence.candidate,
            published_asset=published,
            reference_artifacts=outer_plan.reference_artifacts,
            provided_images=outer_plan.provided_images,
            verification_artifacts=(verification,),
            operation_status=_completed_status(
                review.operation_status,
                operation="publish",
                artifact=verification,
            ),
        )
        binding = _write_packet(receipt_path, packet)
    except BaseException:
        if publication_created:
            output.unlink(missing_ok=True)
            if not verification_existed:
                verification_path.unlink(missing_ok=True)
            if not receipt_existed:
                receipt_path.unlink(missing_ok=True)
        raise
    finally:
        if staging_created:
            staging.unlink(missing_ok=True)
    return packet, binding


def load_texture_packet(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load and validate one focused Texture packet for CLI adapters."""

    return model.model_validate(load_json(Path(path).expanduser().resolve()))


__all__ = [
    "AssessorTextureCritiqueProvider",
    "TextureCandidateEvidencePacket",
    "TextureCapabilityRequest",
    "TextureCritiqueFinding",
    "TextureCritiquePacket",
    "TextureCritiqueProvider",
    "TextureEvidenceCollector",
    "TextureGenerationPacket",
    "TextureGeneratorLeaf",
    "TextureGeneratorLeafRequest",
    "TextureOperationOutcome",
    "TextureOperationSelection",
    "TextureOperationStatus",
    "TextureOuterPlan",
    "TextureOuterReviewInput",
    "TextureOuterReviewPacket",
    "TextureOuterUnitReview",
    "TexturePreparationPacket",
    "TextureProposalProvider",
    "TextureProviderProposalPacket",
    "TexturePublicationReceipt",
    "TextureReferenceArtifact",
    "TextureScopeInspector",
    "TextureUnitRenderEvidence",
    "build_texture_capability_request",
    "collect_texture_candidate_evidence",
    "invoke_texture_generator",
    "load_texture_packet",
    "prepare_texture_scope",
    "publish_texture_candidate",
    "record_texture_outer_review",
    "request_texture_critique",
    "request_texture_provider_proposal",
    "validate_texture_outer_plan",
]
