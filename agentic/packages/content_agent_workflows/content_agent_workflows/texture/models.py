# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-facing contracts for bounded agentic texture generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.domain_execution import (
    DomainExecutionContext,
    ExecutionArtifactBinding,
    domain_execution_context_from_metadata,
)

TEXTURE_PLAN_SCHEMA_VERSION: Literal["texture-agent-plan.v1"] = "texture-agent-plan.v1"
TEXTURE_WORKFLOW_PROGRESS_SCHEMA_VERSION = "content-agent-workflows.texture-progress.v2"
TEXTURE_FINALIZER_INPUT_SCHEMA_VERSION = (
    "content-agent-workflows.texture-finalizer-input.v2"
)
TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.texture-finalization-result.v2"
)
TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-validation-evidence.v3"
] = "content-agent-workflows.texture-validation-evidence.v3"
TEXTURE_CODING_AGENT_COMPANION_BACKEND = "coding_agent_companion"

TextureWorkflowMode = Literal["interactive", "batch"]
TextureWorkflowPhase = Literal[
    "resuming",
    "planned",
    "executing",
    "validating",
    "refining",
    "finalizing",
    "reviewing_publication",
    "completed",
    "cancelled",
]
TextureValidationStatus = Literal["pass", "fail"]
TextureFinalizationStatus = Literal["pass", "conditional", "cancelled"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextureReferenceArtifact(_StrictModel):
    """One exact outer-visible reference and its semantic role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    artifact: ExecutionArtifactBinding


class TextureProvidedImageProducer(_StrictModel):
    """Outer-visible provenance for one already-generated candidate image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)


class TextureProvidedImageArtifact(_StrictModel):
    """One outer-generated image mapped to an exact prepared Texture unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    channel: Literal["albedo"]
    role: Literal["outer_generated_candidate"] = "outer_generated_candidate"
    artifact: ExecutionArtifactBinding
    producer: TextureProvidedImageProducer


class TextureInspectionUnit(_StrictModel):
    """Provider-neutral material and UV facts for one proposed Texture unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    material_prim_paths: tuple[str, ...] = Field(min_length=1)
    material_alias_paths: tuple[str, ...] = ()
    member_prim_paths: tuple[str, ...] = ()
    member_subset_paths: tuple[str, ...] = ()
    uv_status: Literal["ready", "missing", "repair_required", "unsupported"]
    uv_facts: dict[str, Any] = Field(default_factory=dict)
    proposed_generator_inputs: TextureGeneratorInputs

    @model_validator(mode="after")
    def _require_surface_scope(self) -> Self:
        if not self.member_prim_paths and not self.member_subset_paths:
            raise ValueError("Texture inspection requires a prim or subset target")
        return self


class TextureInspectionResult(_StrictModel):
    """Digest-bound inspection returned before the outer coordinator plans."""

    schema_version: Literal["content-agent-workflows.texture-inspection.v1"] = (
        "content-agent-workflows.texture-inspection.v1"
    )
    source: ExecutionArtifactBinding
    proposal_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    units: tuple[TextureInspectionUnit, ...] = Field(min_length=1)
    before_render_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    inspection_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    capability_constraints: tuple[str, ...] = Field(min_length=1)
    renderer_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_units(self) -> Self:
        unit_ids = tuple(unit.unit_id for unit in self.units)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Texture inspection unit IDs must be unique")
        return self


class TextureGeneratorInputs(_StrictModel):
    """Outer-authored generator inputs for one exact target unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_mode: Literal["provider_generate", "apply_provided"] = "provider_generate"
    backend: str = Field(min_length=1)
    engine: str | None = None
    prompt: str = Field(min_length=1)
    detail_policy: Literal["default", "surface_only"] = "surface_only"
    seed: int | None = None
    texture_size: int = Field(default=1024, ge=64, le=16384)
    reference_artifacts: tuple[ExecutionArtifactBinding, ...] = ()
    provided_images: tuple[TextureProvidedImageArtifact, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution_mode(self) -> Self:
        if self.execution_mode == "provider_generate":
            if self.provided_images:
                raise ValueError(
                    "provider_generate inputs cannot include provided candidate images"
                )
            return self
        if not self.provided_images:
            raise ValueError("apply_provided inputs require provided candidate images")
        if self.engine is not None or self.seed is not None or self.parameters:
            raise ValueError(
                "apply_provided inputs cannot include engine, seed, or provider parameters"
            )
        return self


class TexturePlanTarget(_StrictModel):
    """One exact outer-owned target and requested surface appearance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    material_prim_paths: tuple[str, ...] = Field(min_length=1)
    member_prim_paths: tuple[str, ...] = ()
    member_subset_paths: tuple[str, ...] = ()
    requested_appearance: str = Field(min_length=1)
    generator_inputs: TextureGeneratorInputs

    @model_validator(mode="after")
    def _require_surface_scope(self) -> Self:
        if not self.member_prim_paths and not self.member_subset_paths:
            raise ValueError("Texture plan target requires a prim or subset target")
        if self.generator_inputs.execution_mode == "apply_provided":
            provided_unit_ids = tuple(
                item.unit_id for item in self.generator_inputs.provided_images
            )
            if provided_unit_ids != (self.unit_id,):
                raise ValueError(
                    "apply_provided requires exactly one ordered image for its target unit"
                )
        return self


class TexturePreservationConstraints(_StrictModel):
    """Content the candidate generator and publisher may not reinterpret."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preserve_geometry: bool = True
    preserve_joint_graph: bool = True
    preserve_non_target_materials: bool = True
    preserve_non_target_content: bool = True
    preserve_label_legend_authoring: bool = True

    @model_validator(mode="after")
    def _require_fail_closed_preservation(self) -> Self:
        if not all(
            (
                self.preserve_geometry,
                self.preserve_joint_graph,
                self.preserve_non_target_materials,
                self.preserve_non_target_content,
                self.preserve_label_legend_authoring,
            )
        ):
            raise ValueError(
                "embedded Texture plans may not relax preservation constraints"
            )
        return self


class TextureAcceptanceCriteria(_StrictModel):
    """Outer-owned visual and deterministic completion policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    appearance_requirements: tuple[str, ...] = Field(min_length=1)
    require_fresh_before_after_renders: bool = True
    require_provider_neutral_renderer_metadata: bool = True
    require_uv_scope_match: bool = True
    require_material_scope_match: bool = True
    require_package_closure: bool = True
    require_non_target_preservation: bool = True

    @model_validator(mode="after")
    def _require_visual_and_publication_gates(self) -> Self:
        if not all(
            (
                self.require_fresh_before_after_renders,
                self.require_provider_neutral_renderer_metadata,
                self.require_uv_scope_match,
                self.require_material_scope_match,
                self.require_package_closure,
                self.require_non_target_preservation,
            )
        ):
            raise ValueError("embedded Texture acceptance criteria must fail closed")
        return self


class TextureCanonicalPlan(_StrictModel):
    """Canonical typed Texture plan authored only by the outer coordinator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-outer-plan.v1"] = (
        "content-agent-workflows.texture-outer-plan.v1"
    )
    source: ExecutionArtifactBinding
    proposal_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: tuple[TexturePlanTarget, ...] = Field(min_length=1)
    preservation: TexturePreservationConstraints
    acceptance: TextureAcceptanceCriteria
    capability_constraints: tuple[str, ...] = Field(min_length=1)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    capability_claims: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_targets(self) -> Self:
        unit_ids = tuple(target.unit_id for target in self.targets)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("canonical Texture target unit IDs must be unique")
        return self

    @property
    def target_unit_ids(self) -> tuple[str, ...]:
        return tuple(target.unit_id for target in self.targets)


class TextureUnitDisposition(_StrictModel):
    """Outer semantic disposition; VQA status is only supporting critique."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    disposition: Literal["accept", "reject", "revise"]
    rationale: str = Field(min_length=1)


class TextureCandidateReviewDecision(_StrictModel):
    """Outer review of one exact candidate and fresh visual evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-candidate-review.v1"] = (
        "content-agent-workflows.texture-candidate-review.v1"
    )
    candidate_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_asset: ExecutionArtifactBinding
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_dispositions: tuple[TextureUnitDisposition, ...] = Field(min_length=1)
    visual_evidence_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(
        min_length=1
    )
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    visual_evidence_accepted: bool
    findings: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dispositions(self) -> Self:
        unit_ids = tuple(item.unit_id for item in self.unit_dispositions)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("candidate review unit IDs must be unique")
        if (
            any(item.disposition == "accept" for item in self.unit_dispositions)
            and not self.visual_evidence_accepted
        ):
            raise ValueError("visual rejection cannot accept a Texture unit")
        return self


class TexturePublicationDecision(_StrictModel):
    """Exact accepted candidate identity consumed by deterministic publication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.texture-publication-decision.v1"
    ] = "content-agent-workflows.texture-publication-decision.v1"
    candidate_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_output: ExecutionArtifactBinding
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_unit_ids: tuple[str, ...] = Field(min_length=1)
    candidate_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    publication_path: str = Field(min_length=1)


class TexturePublicationReviewDecision(_StrictModel):
    """Outer review of exact deterministic publication proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-publication-review.v1"] = (
        "content-agent-workflows.texture-publication-review.v1"
    )
    publication_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_asset: ExecutionArtifactBinding
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    disposition: Literal["accept", "reject", "revise"]
    findings: tuple[str, ...] = Field(min_length=1)


def _require_unique_unit_ids(field_name: str, unit_ids: tuple[str, ...]) -> None:
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError(f"{field_name} must be unique")


class TextureWorkflowRequest(_StrictModel):
    """Inputs shared by interactive and batch texture workflow entry points."""

    source_asset: str = Field(min_length=1)
    output_dir: Path
    intent: str = "Generate bounded, portable textures for the selected units."
    target_runtime: str = "usd-cli"
    max_vqa_iterations: int = Field(default=2, ge=0, le=8)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_execution_context(self) -> Self:
        domain_execution_context_from_metadata(
            self.metadata,
            expected_domain="texture",
        )
        roles = tuple(item.role for item in self.reference_artifacts)
        paths = tuple(item.artifact.path for item in self.reference_artifacts)
        if len(roles) != len(set(roles)):
            raise ValueError("Texture reference artifact roles must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("Texture reference artifact paths must be unique")
        return self

    @property
    def execution_context(self) -> DomainExecutionContext | None:
        """Return the validated lifecycle boundary without serializing a new field."""

        return domain_execution_context_from_metadata(
            self.metadata,
            expected_domain="texture",
        )


class TexturePlanSelectedUnit(BaseModel):
    """Selected-unit view of the WP0 texture plan contract.

    Additional WP0 fields are preserved when the plan is serialized. This is a
    compatibility envelope, not a replacement for the Texture Agent contract.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")


class TexturePlanCounts(BaseModel):
    """Count view needed to validate workflow fanout."""

    model_config = ConfigDict(extra="allow", frozen=True)

    selected_unit_count: int = Field(ge=0, le=64)


class TexturePlanDecision(BaseModel):
    """Planner decision view needed before executor work may start."""

    model_config = ConfigDict(extra="allow", frozen=True)

    state: str = Field(min_length=1)
    execution_allowed: bool


class TexturePlanDocument(BaseModel):
    """Pass-through envelope for the immutable WP0 ``texture_plan.json``."""

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal["texture-agent-plan.v1"] = TEXTURE_PLAN_SCHEMA_VERSION
    counts: TexturePlanCounts
    selected_units: tuple[TexturePlanSelectedUnit, ...]
    decision: TexturePlanDecision

    @model_validator(mode="after")
    def _validate_selected_units(self) -> Self:
        unit_ids = self.selected_unit_ids
        if self.counts.selected_unit_count != len(unit_ids):
            raise ValueError(
                "selected_unit_count must equal the number of selected_units"
            )
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("selected_units must contain unique unit_id values")
        return self

    @property
    def selected_unit_ids(self) -> tuple[str, ...]:
        """Return selected IDs in immutable plan order."""

        return tuple(unit.unit_id for unit in self.selected_units)


class TextureUnitArtifact(_StrictModel):
    """Executor artifacts owned by one canonical selected-unit ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    artifact_paths: tuple[str, ...] = Field(min_length=1)
    generation: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextureExecutionResult(_StrictModel):
    """One bounded executor invocation and its per-unit artifacts."""

    requested_unit_ids: tuple[str, ...] = Field(min_length=1)
    unit_artifacts: tuple[TextureUnitArtifact, ...] = Field(min_length=1)
    output_asset_path: str = Field(min_length=1)
    cache_hit_unit_ids: tuple[str, ...] = ()
    retry_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_exact_execution_scope(self) -> Self:
        artifact_ids = tuple(item.unit_id for item in self.unit_artifacts)
        if len(self.requested_unit_ids) != len(set(self.requested_unit_ids)):
            raise ValueError("requested_unit_ids must be unique")
        if artifact_ids != self.requested_unit_ids:
            raise ValueError(
                "unit_artifacts must exactly match requested_unit_ids in plan order"
            )
        unknown_cache_ids = set(self.cache_hit_unit_ids) - set(self.requested_unit_ids)
        if unknown_cache_ids:
            raise ValueError(
                "cache_hit_unit_ids must be a subset of requested_unit_ids"
            )
        return self


class TextureValidationFinding(_StrictModel):
    """usd-cli VQA outcome for one selected unit."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    status: TextureValidationStatus
    summary: str = Field(min_length=1)
    evidence_artifact_paths: tuple[str, ...] = Field(min_length=1)


class TextureValidationResult(_StrictModel):
    """A bounded usd-cli validation pass over explicit unit IDs."""

    iteration: int = Field(ge=0)
    evaluated_unit_ids: tuple[str, ...] = Field(min_length=1)
    findings: tuple[TextureValidationFinding, ...] = Field(min_length=1)
    output_asset_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_exact_validation_scope(self) -> Self:
        finding_ids = tuple(finding.unit_id for finding in self.findings)
        if len(self.evaluated_unit_ids) != len(set(self.evaluated_unit_ids)):
            raise ValueError("evaluated_unit_ids must be unique")
        if finding_ids != self.evaluated_unit_ids:
            raise ValueError(
                "findings must exactly match evaluated_unit_ids in requested order"
            )
        return self

    @property
    def failed_unit_ids(self) -> tuple[str, ...]:
        """Return exact failing IDs in validation order."""

        return tuple(
            finding.unit_id for finding in self.findings if finding.status == "fail"
        )


class TextureWorkflowProgress(_StrictModel):
    """Progress snapshot with accepted and remaining selected units."""

    schema_version: str = TEXTURE_WORKFLOW_PROGRESS_SCHEMA_VERSION
    mode: TextureWorkflowMode
    phase: TextureWorkflowPhase
    iteration: int = Field(default=0, ge=0)
    selected_unit_ids: tuple[str, ...]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    selected_unit_count: int = Field(ge=0)
    accepted_unit_count: int = Field(ge=0)
    remaining_unit_count: int = Field(ge=0)
    message: str = Field(min_length=1)

    @classmethod
    def build(
        cls,
        *,
        mode: TextureWorkflowMode,
        phase: TextureWorkflowPhase,
        selected_unit_ids: tuple[str, ...],
        accepted_unit_ids: tuple[str, ...],
        remaining_unit_ids: tuple[str, ...],
        message: str,
        iteration: int = 0,
    ) -> TextureWorkflowProgress:
        return cls(
            mode=mode,
            phase=phase,
            iteration=iteration,
            selected_unit_ids=selected_unit_ids,
            accepted_unit_ids=accepted_unit_ids,
            remaining_unit_ids=remaining_unit_ids,
            selected_unit_count=len(selected_unit_ids),
            accepted_unit_count=len(accepted_unit_ids),
            remaining_unit_count=len(remaining_unit_ids),
            message=message,
        )

    @model_validator(mode="after")
    def _validate_partition_and_counts(self) -> Self:
        selected = self.selected_unit_ids
        accepted = self.accepted_unit_ids
        remaining = self.remaining_unit_ids
        _require_unique_unit_ids("selected_unit_ids", selected)
        _require_unique_unit_ids("accepted_unit_ids", accepted)
        _require_unique_unit_ids("remaining_unit_ids", remaining)
        if set(accepted) & set(remaining):
            raise ValueError(
                "accepted_unit_ids and remaining_unit_ids must be disjoint"
            )
        if set(accepted) | set(remaining) != set(selected):
            raise ValueError(
                "accepted_unit_ids and remaining_unit_ids must partition selected units"
            )
        if (
            self.selected_unit_count != len(selected)
            or self.accepted_unit_count != len(accepted)
            or self.remaining_unit_count != len(remaining)
        ):
            raise ValueError("progress counts must match their unit ID lists")
        return self


class TextureFinalizerInput(_StrictModel):
    """Deterministic finalizer input shared by all launch modes."""

    schema_version: str = TEXTURE_FINALIZER_INPUT_SCHEMA_VERSION
    mode: TextureWorkflowMode
    request: TextureWorkflowRequest
    plan: TexturePlanDocument
    terminal_status: TextureFinalizationStatus
    cancellation_reason: str | None = None
    executions: tuple[TextureExecutionResult, ...] = ()
    validations: tuple[TextureValidationResult, ...] = ()
    progress: tuple[TextureWorkflowProgress, ...] = Field(min_length=1)
    unit_artifacts: dict[str, TextureUnitArtifact]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    output_asset_path: str | None = None
    output_asset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    workflow_checkpoint_path: str | None = Field(default=None, min_length=1)
    decision_ledger_path: str | None = Field(default=None, min_length=1)
    embedded_decision_receipt_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_final_state(self) -> Self:
        selected = self.plan.selected_unit_ids
        _require_unique_unit_ids("accepted_unit_ids", self.accepted_unit_ids)
        _require_unique_unit_ids("remaining_unit_ids", self.remaining_unit_ids)
        artifact_ids = set(self.unit_artifacts)
        selected_ids = set(selected)
        if not artifact_ids <= selected_ids:
            raise ValueError("unit_artifacts must not contain IDs outside the plan")
        if self.terminal_status != "cancelled" and artifact_ids != selected_ids:
            raise ValueError(
                "non-cancelled finalization must cover every selected unit"
            )
        if any(
            key != artifact.unit_id for key, artifact in self.unit_artifacts.items()
        ):
            raise ValueError("unit_artifacts keys must match artifact unit_id values")
        if set(self.accepted_unit_ids) & set(self.remaining_unit_ids):
            raise ValueError("accepted and remaining unit IDs must be disjoint")
        if set(self.accepted_unit_ids) | set(self.remaining_unit_ids) != set(selected):
            raise ValueError("accepted and remaining unit IDs must partition the plan")
        if not set(self.accepted_unit_ids) <= artifact_ids:
            raise ValueError("accepted units must have preserved artifacts")
        if self.terminal_status == "pass" and self.remaining_unit_ids:
            raise ValueError("pass finalization cannot contain remaining unit IDs")
        if self.terminal_status == "conditional" and not self.remaining_unit_ids:
            raise ValueError("conditional finalization requires remaining unit IDs")
        if self.terminal_status == "cancelled" and not self.cancellation_reason:
            raise ValueError("cancelled finalization requires a reason")
        if self.terminal_status != "cancelled" and self.cancellation_reason:
            raise ValueError(
                "cancellation_reason is only valid for cancelled finalization"
            )
        if self.terminal_status != "cancelled":
            if not self.executions:
                raise ValueError(
                    "non-cancelled finalization requires execution evidence"
                )
            if not self.validations:
                raise ValueError(
                    "non-cancelled finalization requires validation evidence"
                )
            if not self.output_asset_path:
                raise ValueError(
                    "non-cancelled finalization requires an output asset path"
                )
        if self.output_asset_sha256 is not None and self.output_asset_path is None:
            raise ValueError("output_asset_sha256 requires output_asset_path")
        return self


class TextureFinalizationResult(_StrictModel):
    """Canonical artifact index returned by the texture finalizer."""

    schema_version: str = TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION
    success: bool
    status: TextureFinalizationStatus
    mode: TextureWorkflowMode
    output_dir: str
    output_asset_path: str | None = None
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    cancellation_reason: str | None = None
    request_path: str
    texture_plan_path: str
    execution_summary_path: str
    visual_quality_assessment_path: str
    validation_evidence_path: str
    workflow_progress_path: str
    workflow_checkpoint_path: str
    decision_ledger_path: str | None = None
    embedded_decision_receipt_path: str | None = None
    final_summary_path: str

    @model_validator(mode="after")
    def _validate_status(self) -> Self:
        expected_success = self.status == "pass"
        if self.success is not expected_success:
            raise ValueError("success must be true only for pass finalization")
        if self.status == "pass" and self.remaining_unit_ids:
            raise ValueError("pass finalization cannot contain remaining unit IDs")
        if self.status == "conditional" and not self.remaining_unit_ids:
            raise ValueError("conditional finalization requires remaining unit IDs")
        if self.status == "cancelled" and not self.cancellation_reason:
            raise ValueError("cancelled finalization requires a reason")
        if self.status != "cancelled" and self.cancellation_reason:
            raise ValueError(
                "cancellation_reason is only valid for cancelled finalization"
            )
        return self


class TextureWorkflowValidationEvidence(_StrictModel):
    """Normalized usd-cli VQA and bounded-execution evidence."""

    schema_version: Literal[
        "content-agent-workflows.texture-validation-evidence.v3"
    ] = TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION
    workflow: Literal["texture_generation"] = "texture_generation"
    target_runtime: str = Field(min_length=1)
    status: TextureFinalizationStatus
    selected_unit_ids: tuple[str, ...]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    selected_unit_count: int = Field(ge=0)
    backend_job_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    output_asset_path: str | None = None
    output_asset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    unit_artifact_paths: dict[str, tuple[str, ...]]
    visual_evidence_paths: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_evidence_scope(self) -> Self:
        if bool(self.output_asset_path) != bool(self.output_asset_sha256):
            raise ValueError(
                "output_asset_path and output_asset_sha256 must be set together"
            )
        selected = self.selected_unit_ids
        if self.selected_unit_count != len(selected):
            raise ValueError("selected_unit_count must match selected_unit_ids")
        if set(self.accepted_unit_ids) | set(self.remaining_unit_ids) != set(selected):
            raise ValueError(
                "accepted and remaining unit IDs must cover selected units"
            )
        if set(self.accepted_unit_ids) & set(self.remaining_unit_ids):
            raise ValueError("accepted and remaining unit IDs must be disjoint")
        artifact_ids = set(self.unit_artifact_paths)
        selected_ids = set(selected)
        if not artifact_ids <= selected_ids:
            raise ValueError(
                "unit_artifact_paths must not contain IDs outside selected units"
            )
        if self.status != "cancelled" and artifact_ids != selected_ids:
            raise ValueError("non-cancelled evidence must cover every selected unit")
        if not set(self.accepted_unit_ids) <= artifact_ids:
            raise ValueError("accepted units must have artifact paths")
        if self.status == "pass" and self.remaining_unit_ids:
            raise ValueError("pass evidence cannot contain remaining unit IDs")
        if self.status == "conditional" and not self.remaining_unit_ids:
            raise ValueError("conditional evidence requires remaining unit IDs")
        return self
