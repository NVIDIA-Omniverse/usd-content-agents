# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation evidence schema for agentic asset workflows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VALIDATION_EVIDENCE_SCHEMA_VERSION = "content-agent-workflows.validation-evidence.v1"
VALIDATION_TIERS = ("T1_basic_stability", "T2_simulation_match", "T3_real_comparison")
SIM_READY_STATUSES = ("pass", "conditional", "fail", "not_evaluated")
VALIDATION_CHECK_TAXONOMY = (
    "scale",
    "collisions",
    "visual_materials",
    "non_visual_materials",
    "physics_properties",
    "articulation",
    "runtime_loadability",
    "no_explosions",
    "no_penetration",
    "stable_at_max_speed",
    "momentum_energy_conservation",
    "simulation_visual_review",
)

ValidationTier = Literal[
    "T1_basic_stability",
    "T2_simulation_match",
    "T3_real_comparison",
]
SimReadyStatus = Literal["pass", "conditional", "fail", "not_evaluated"]
CheckStatus = Literal["pass", "fail", "warning", "not_evaluated"]


class EvidenceArtifact(BaseModel):
    """Artifact referenced by validation evidence."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationCheck(BaseModel):
    """One validation check result."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    status: CheckStatus
    summary: str = Field(min_length=1)
    evidence_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationEvidence(BaseModel):
    """Workflow-level validation evidence for a target runtime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = VALIDATION_EVIDENCE_SCHEMA_VERSION
    workflow: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    target_runtime: str = Field(min_length=1)
    validation_tier: ValidationTier
    checks: list[ValidationCheck] = Field(default_factory=list)
    evidence_artifacts: list[EvidenceArtifact] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)
    sim_ready_status: SimReadyStatus = "not_evaluated"
    metadata: dict[str, Any] = Field(default_factory=dict)


def material_assignment_validation_evidence(
    *,
    asset: str,
    target_runtime: str,
    visual_materials_status: CheckStatus,
    evidence_artifacts: list[EvidenceArtifact] | None = None,
    failures: list[str] | None = None,
    warnings: list[str] | None = None,
    unresolved_issues: list[str] | None = None,
) -> ValidationEvidence:
    """Build initial validation evidence for material-assignment output."""

    evidence = evidence_artifacts or []
    failure_items = failures or []
    warning_items = warnings or []
    unresolved_items = unresolved_issues or []
    if visual_materials_status == "fail" or failure_items:
        sim_ready_status: SimReadyStatus = "fail"
    elif visual_materials_status == "not_evaluated":
        sim_ready_status = "not_evaluated"
    elif visual_materials_status == "warning" or unresolved_items or warning_items:
        sim_ready_status = "conditional"
    else:
        sim_ready_status = "pass"
    return ValidationEvidence(
        workflow="material_assignment",
        asset=asset,
        target_runtime=target_runtime,
        validation_tier="T1_basic_stability",
        checks=[
            ValidationCheck(
                name="visual_materials",
                status=visual_materials_status,
                summary="Material assignment visual coverage and final render review.",
                evidence_artifacts=evidence,
                failures=failure_items,
                warnings=warning_items,
                repair_hints=[
                    "Refine material assignment with focused Workbench renders and VQA."
                ]
                if unresolved_items or failure_items
                else [],
            )
        ],
        evidence_artifacts=evidence,
        failures=failure_items,
        warnings=warning_items,
        unresolved_issues=unresolved_items,
        repair_hints=[
            "Run bounded material-assignment refinement for unresolved visual issues."
        ]
        if unresolved_items or failure_items
        else [],
        sim_ready_status=sim_ready_status,
    )


def physics_validation_evidence(
    *,
    asset: str,
    target_runtime: str,
    physics_properties_status: CheckStatus,
    runtime_loadability_status: CheckStatus = "not_evaluated",
    no_explosions_status: CheckStatus = "not_evaluated",
    simulation_visual_review_status: CheckStatus | None = None,
    validation_tier: ValidationTier = "T1_basic_stability",
    evidence_artifacts: list[EvidenceArtifact] | None = None,
    failures: list[str] | None = None,
    warnings: list[str] | None = None,
    unresolved_issues: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ValidationEvidence:
    """Build validation evidence for physics authoring workflow output."""

    evidence = evidence_artifacts or []
    failure_items = failures or []
    warning_items = warnings or []
    unresolved_items = unresolved_issues or []
    statuses = [
        physics_properties_status,
        runtime_loadability_status,
        no_explosions_status,
    ]
    if simulation_visual_review_status is not None:
        statuses.append(simulation_visual_review_status)
    if "fail" in statuses or failure_items:
        sim_ready_status: SimReadyStatus = "fail"
    elif all(status == "not_evaluated" for status in statuses):
        sim_ready_status = "not_evaluated"
    elif "warning" in statuses or unresolved_items or warning_items:
        sim_ready_status = "conditional"
    elif "not_evaluated" in statuses:
        sim_ready_status = "conditional"
    else:
        sim_ready_status = "pass"

    repair_hints = (
        [
            "Refine inferred density, mass, friction, restitution, and collider choices; rerun runtime validation."
        ]
        if unresolved_items or failure_items
        else []
    )
    checks = [
        ValidationCheck(
            name="physics_properties",
            status=physics_properties_status,
            summary="Physics schemas and authored physical properties are present.",
            evidence_artifacts=evidence,
            failures=failure_items if physics_properties_status == "fail" else [],
            warnings=warning_items if physics_properties_status == "warning" else [],
            repair_hints=repair_hints,
        ),
        ValidationCheck(
            name="runtime_loadability",
            status=runtime_loadability_status,
            summary="Authored asset loads in the requested physics runtime.",
            evidence_artifacts=evidence,
        ),
        ValidationCheck(
            name="no_explosions",
            status=no_explosions_status,
            summary="Runtime trajectory remains finite and bounded.",
            evidence_artifacts=evidence,
        ),
    ]
    if simulation_visual_review_status is not None:
        checks.append(
            ValidationCheck(
                name="simulation_visual_review",
                status=simulation_visual_review_status,
                summary=(
                    "Rendered runtime simulation frames were reviewed for "
                    "visually plausible physics behavior."
                ),
                evidence_artifacts=evidence,
                failures=failure_items
                if simulation_visual_review_status == "fail"
                else [],
                warnings=warning_items
                if simulation_visual_review_status == "warning"
                else [],
                repair_hints=repair_hints,
            )
        )

    return ValidationEvidence(
        workflow="physics_authoring",
        asset=asset,
        target_runtime=target_runtime,
        validation_tier=validation_tier,
        checks=checks,
        evidence_artifacts=evidence,
        failures=failure_items,
        warnings=warning_items,
        unresolved_issues=unresolved_items,
        repair_hints=repair_hints,
        sim_ready_status=sim_ready_status,
        metadata=metadata or {},
    )
