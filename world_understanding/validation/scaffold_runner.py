# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable per-template execution adapter for the Validation Agent scaffold."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from world_understanding.agentic.validation_scaffold import (
    DraftTemplateResult,
    DraftValidationIssue,
    DraftValidationRequest,
    RuntimeVisualEvidence,
    create_default_scaffold_registry,
    plan_validation,
    prepare_validation_scaffold_context,
    run_validation_scaffold_template,
    runtime_visual_evidence_from_render_result,
)
from world_understanding.validation.cli import (
    scaffold_metadata_from_request,
    scaffold_policy_from_request,
)
from world_understanding.validation.models import (
    ValidationIssue,
    ValidationPlan,
    ValidationRequest,
    ValidationTemplateResult,
)
from world_understanding.validation.runner import ValidationTemplateContext
from world_understanding.validation.scaffold_compat import (
    validation_plan_from_scaffold_plan,
    validation_template_result_from_scaffold_result,
)

SCAFFOLD_VALIDATION_TEMPLATE_VERSIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "render_valid": "validation-scaffold.render-valid.v1",
        "look_right": "validation-scaffold.look-right.v1",
        "physics_sane": "validation-scaffold.physics-sane.v1",
        "physical_behavior": "validation-scaffold.physical-behavior.v1",
    }
)


def _draft_request(
    request: ValidationRequest,
    *,
    config_base_dir: Path,
    working_dir: Path,
) -> DraftValidationRequest:
    return DraftValidationRequest(
        task_description=request.task_description,
        inputs=request.inputs,
        working_dir=working_dir,
        base_dir=config_base_dir,
        focus_prim_paths=request.focus.prim_paths,
        requested_templates=request.requested_templates,
        policy=scaffold_policy_from_request(request, base_dir=config_base_dir),
        dry_run=False,
        metadata=scaffold_metadata_from_request(request),
    )


def _draft_issue(issue: ValidationIssue) -> DraftValidationIssue:
    return DraftValidationIssue(
        code=issue.code,
        severity=issue.severity,
        message=issue.message,
        subject=issue.subject,
        details=issue.details,
    )


def _draft_template_result(
    result: ValidationTemplateResult,
) -> DraftTemplateResult:
    return DraftTemplateResult(
        template_name=result.template_name,
        status=result.status,
        issues=tuple(_draft_issue(issue) for issue in result.issues),
        metrics=result.metrics,
        evidence=result.evidence,
        metadata=result.metadata,
    )


def _runtime_render_payload(
    previous_template_results: tuple[ValidationTemplateResult, ...],
) -> Mapping[str, Any] | None:
    for result in reversed(previous_template_results):
        if result.template_name != "render_valid":
            continue
        payload = result.metadata.get("runtime_render")
        if isinstance(payload, Mapping):
            return payload
        return None
    return None


def _missing_render_handoff_result(reason: str) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.render_handoff_missing",
        severity="fail",
        message=reason,
        template_name="look_right",
    )
    return ValidationTemplateResult(
        template_name="look_right",
        status="failed",
        issues=(issue,),
        metrics={"issue_count": 1, "vlm_invoked": False},
        metadata={
            "executor": "validation-scaffold-step-runner",
            "render_handoff_present": False,
        },
    )


@dataclass(frozen=True)
class ScaffoldValidationStepExecutor:
    """Plan once and execute one stable Validation Agent template at a time."""

    config_base_dir: Path

    @property
    def template_versions(self) -> Mapping[str, str]:
        """Return implementation identities used by resumable workflows."""

        return SCAFFOLD_VALIDATION_TEMPLATE_VERSIONS

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        """Create a stable plan without executing templates or rendering."""

        registry = create_default_scaffold_registry()
        draft_request = _draft_request(
            request,
            config_base_dir=self.config_base_dir.resolve(),
            working_dir=working_dir.resolve(),
        )
        return validation_plan_from_scaffold_plan(
            plan_validation(draft_request, registry)
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        """Execute one template, reusing only an accepted render handoff."""

        registry = create_default_scaffold_registry()
        draft_request = _draft_request(
            context.request,
            config_base_dir=self.config_base_dir.resolve(),
            # Preserve /proc/self/fd/<fd> instead of dereferencing the pinned
            # attempt directory back to a mutable path.
            working_dir=Path(os.path.abspath(context.working_dir)),
        )
        draft_plan = plan_validation(draft_request, registry)
        # The scaffold planner canonicalizes its inventory path. Restore the
        # descriptor-backed path before any template consumes the inventory.
        draft_plan = replace(
            draft_plan,
            input_inventory=replace(
                draft_plan.input_inventory,
                working_dir=draft_request.working_dir,
            ),
        )
        planned_names = tuple(step.template_name for step in context.plan.steps)
        draft_names = tuple(step.template_name for step in draft_plan.steps)
        if draft_names != planned_names:
            raise ValueError(
                "Validation plan changed between planning and template execution"
            )

        previous_results = tuple(
            _draft_template_result(result)
            for result in context.previous_template_results
        )
        runtime_render = _runtime_render_payload(context.previous_template_results)
        if template_name == "look_right":
            render_result = next(
                (
                    result
                    for result in reversed(context.previous_template_results)
                    if result.template_name == "render_valid"
                ),
                None,
            )
            if render_result is None:
                return _missing_render_handoff_result(
                    "look_right requires the current run's accepted render_valid "
                    "result before VLM judging."
                )
            if not render_result.passed:
                return _missing_render_handoff_result(
                    "look_right requires a passed render_valid result before "
                    "VLM judging."
                )
            if runtime_render is None:
                return _missing_render_handoff_result(
                    "The accepted render_valid result has no runtime render "
                    "evidence to hand off to look_right."
                )
        runtime_evidence: RuntimeVisualEvidence | None = None
        if runtime_render is not None:
            runtime_evidence = runtime_visual_evidence_from_render_result(
                runtime_render
            )
        elif template_name not in {"render_valid", "look_right"}:
            runtime_evidence = RuntimeVisualEvidence()

        draft_context = prepare_validation_scaffold_context(
            draft_request,
            registry,
            plan=draft_plan,
            previous_template_results=previous_results,
            runtime_visual_evidence=runtime_evidence,
        )
        draft_result = run_validation_scaffold_template(
            template_name,
            draft_context,
            registry,
        )
        return validation_template_result_from_scaffold_result(draft_result)
