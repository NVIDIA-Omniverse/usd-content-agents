# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-facing contracts for bounded agentic texture generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TEXTURE_PLAN_SCHEMA_VERSION: Literal["texture-agent-plan.v1"] = "texture-agent-plan.v1"
TEXTURE_WORKFLOW_PROGRESS_SCHEMA_VERSION = "content-agent-workflows.texture-progress.v1"
TEXTURE_FINALIZER_INPUT_SCHEMA_VERSION = (
    "content-agent-workflows.texture-finalizer-input.v1"
)
TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION = (
    "content-agent-workflows.texture-finalization-result.v1"
)
TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION = (
    "content-agent-workflows.texture-validation-evidence.v1"
)

TextureWorkflowMode = Literal["interactive", "batch"]
TextureWorkflowPhase = Literal[
    "planned",
    "executing",
    "validating",
    "refining",
    "finalizing",
    "completed",
]
TextureValidationStatus = Literal["pass", "fail"]
TextureFinalizationStatus = Literal["pass", "conditional"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_unique_unit_ids(field_name: str, unit_ids: tuple[str, ...]) -> None:
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError(f"{field_name} must be unique")


class TextureWorkflowRequest(_StrictModel):
    """Inputs shared by interactive and batch texture workflow entry points."""

    source_asset: str = Field(min_length=1)
    output_dir: Path
    intent: str = "Generate bounded, portable textures for the selected units."
    target_runtime: str = "content-workbench"
    max_vqa_iterations: int = Field(default=2, ge=0, le=8)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Workbench VQA outcome for one selected unit."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    status: TextureValidationStatus
    summary: str = Field(min_length=1)
    evidence_artifact_paths: tuple[str, ...] = Field(min_length=1)


class TextureValidationResult(_StrictModel):
    """A bounded Workbench validation pass over explicit unit IDs."""

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
    executions: tuple[TextureExecutionResult, ...] = Field(min_length=1)
    validations: tuple[TextureValidationResult, ...] = Field(min_length=1)
    progress: tuple[TextureWorkflowProgress, ...] = Field(min_length=1)
    unit_artifacts: dict[str, TextureUnitArtifact]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    output_asset_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_final_state(self) -> Self:
        selected = self.plan.selected_unit_ids
        _require_unique_unit_ids("accepted_unit_ids", self.accepted_unit_ids)
        _require_unique_unit_ids("remaining_unit_ids", self.remaining_unit_ids)
        if set(self.unit_artifacts) != set(selected):
            raise ValueError("unit_artifacts must cover every selected unit")
        if any(
            key != artifact.unit_id for key, artifact in self.unit_artifacts.items()
        ):
            raise ValueError("unit_artifacts keys must match artifact unit_id values")
        if set(self.accepted_unit_ids) & set(self.remaining_unit_ids):
            raise ValueError("accepted and remaining unit IDs must be disjoint")
        if set(self.accepted_unit_ids) | set(self.remaining_unit_ids) != set(selected):
            raise ValueError("accepted and remaining unit IDs must partition the plan")
        return self


class TextureFinalizationResult(_StrictModel):
    """Canonical artifact index returned by the texture finalizer."""

    schema_version: str = TEXTURE_FINALIZATION_RESULT_SCHEMA_VERSION
    success: bool
    status: TextureFinalizationStatus
    mode: TextureWorkflowMode
    output_dir: str
    output_asset_path: str
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    request_path: str
    texture_plan_path: str
    execution_summary_path: str
    visual_quality_assessment_path: str
    validation_evidence_path: str
    workflow_progress_path: str
    final_summary_path: str

    @model_validator(mode="after")
    def _validate_status(self) -> Self:
        expected_success = not self.remaining_unit_ids
        expected_status = "pass" if expected_success else "conditional"
        if self.success is not expected_success or self.status != expected_status:
            raise ValueError(
                "success and status must reflect whether remaining_unit_ids is empty"
            )
        return self


class TextureWorkflowValidationEvidence(_StrictModel):
    """Normalized Workbench VQA and bounded-execution evidence."""

    schema_version: str = TEXTURE_VALIDATION_EVIDENCE_SCHEMA_VERSION
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
    output_asset_path: str = Field(min_length=1)
    unit_artifact_paths: dict[str, tuple[str, ...]]
    visual_evidence_paths: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_evidence_scope(self) -> Self:
        selected = self.selected_unit_ids
        if self.selected_unit_count != len(selected):
            raise ValueError("selected_unit_count must match selected_unit_ids")
        if set(self.accepted_unit_ids) | set(self.remaining_unit_ids) != set(selected):
            raise ValueError(
                "accepted and remaining unit IDs must cover selected units"
            )
        if set(self.accepted_unit_ids) & set(self.remaining_unit_ids):
            raise ValueError("accepted and remaining unit IDs must be disjoint")
        if set(self.unit_artifact_paths) != set(selected):
            raise ValueError("unit_artifact_paths must cover selected units")
        expected_status = "pass" if not self.remaining_unit_ids else "conditional"
        if self.status != expected_status:
            raise ValueError("status must reflect remaining_unit_ids")
        return self
