# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary validation scaffold for issue #78.

This module wires the already-merged input resolver into a rules-only planner
and maps provisional template adapters into temporary result models. PR4 adds
stable contracts under ``world_understanding.validation``; use
``world_understanding.validation.validation_result_from_scaffold_result`` when
callers need the V1 report model before the full CLI runner lands.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from world_understanding.functions.cv.look_right import (
    DEFAULT_LOOK_RIGHT_PROMPT_TEMPLATE,
    DEFAULT_LOOK_RIGHT_SYSTEM_PROMPT,
    VISUAL_EVIDENCE_MISSING,
    VISUAL_JUDGE_UNAVAILABLE,
    VISUAL_RENDER_PREFLIGHT_FAILED,
    LookRightFinalJudgeResult,
    LookRightIssue,
    LookRightJudgeInvocation,
    LookRightJudgePlan,
    build_look_right_judge_plan,
    invoke_look_right_judge,
    normalize_look_right_judgment,
)
from world_understanding.functions.graphics.render_valid_adapter import (
    run_render_valid_adapter,
)
from world_understanding.functions.models.chat_models import create_chat_model
from world_understanding.functions.models.vision_language_models import create_vlm
from world_understanding.functions.physics.physical_behavior_evidence import (
    BEHAVIOR_EVIDENCE_MISSING,
    BEHAVIOR_JUDGE_UNAVAILABLE,
    PhysicalBehaviorEvidence,
    PhysicalBehaviorEvidenceResolution,
    PhysicalBehaviorIssue,
    resolve_physical_behavior_evidence,
)
from world_understanding.functions.physics.physics_sane_adapter import (
    run_physics_sane_adapter,
)
from world_understanding.functions.physics.physics_sanity import infer_physics_expected
from world_understanding.utils.credentials import (
    API_KEY_ENV_VAR_MAP,
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
)
from world_understanding.utils.input_resolver import (
    InputInventory,
    resolve_input_inventory,
)
from world_understanding.validation.usd_rendering import (
    render_usd_visual_evidence,
)

V1_TEMPLATE_NAMES = (
    "look_right",
    "render_valid",
    "physics_sane",
    "physical_behavior",
)

_LOGGER = logging.getLogger(__name__)

IssueSeverity = Literal["info", "warn", "fail"]
TemplateStatus = Literal[
    "passed",
    "warn",
    "failed",
    "needs_refinement",
    "skipped",
    "error",
]
ValidationVerdict = Literal["pass", "warn", "fail", "needs_refinement", "planned"]

BEHAVIOR_NEEDS_REFINEMENT = "physics.behavior_needs_refinement"
BEHAVIOR_REJECTED = "physics.behavior_rejected"
BEHAVIOR_REFINE_LOOP_FAILED = "physics.behavior_refine_loop_failed"
BEHAVIOR_REFINER_UNAVAILABLE = "physics.behavior_refiner_unavailable"
CANONICAL_RUNTIME_RENDER_DISABLED = "render.canonical_runtime_render_disabled"
CANONICAL_RUNTIME_RENDER_MISSING = "render.canonical_runtime_render_missing"
CANONICAL_USD_INPUT_MISSING = "render.canonical_usd_input_missing"
MAX_BEHAVIOR_RENDER_EVIDENCE_FILES = 32


class DraftValidationError(ValueError):
    """Raised when the temporary scaffold cannot create or run a plan."""


@dataclass(frozen=True)
class DraftValidationRequest:
    """Temporary request model used before #45 defines final contracts."""

    task_description: str
    inputs: tuple[str | Path, ...]
    working_dir: Path
    base_dir: Path | None = None
    focus_prim_paths: tuple[str, ...] = ()
    requested_templates: tuple[str, ...] = ()
    policy: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "task_description": self.task_description,
            "inputs": [str(input_path) for input_path in self.inputs],
            "working_dir": str(self.working_dir),
            "base_dir": str(self.base_dir) if self.base_dir is not None else None,
            "focus_prim_paths": list(self.focus_prim_paths),
            "requested_templates": list(self.requested_templates),
            "policy": _to_json_compatible(self.policy),
            "dry_run": self.dry_run,
            "metadata": _to_json_compatible(self.metadata),
        }


@dataclass(frozen=True)
class DraftValidationStep:
    """One scaffold template invocation in a validation plan."""

    template_name: str
    reason: str
    inputs_needed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "reason": self.reason,
            "inputs_needed": list(self.inputs_needed),
        }


@dataclass(frozen=True)
class DraftValidationPlan:
    """Rules-only validation plan produced by the temporary planner."""

    steps: tuple[DraftValidationStep, ...]
    input_inventory: InputInventory
    reasoning_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "input_inventory": self.input_inventory.to_dict(),
            "reasoning_summary": self.reasoning_summary,
        }


@dataclass(frozen=True)
class DraftValidationIssue:
    """Temporary issue model with codes intended to map to #45 later."""

    code: str
    severity: IssueSeverity
    message: str
    subject: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }
        if self.subject is not None:
            data["subject"] = self.subject
        return data


@dataclass(frozen=True)
class RuntimeVisualEvidence:
    """Runtime-generated visual evidence made available to scaffold templates."""

    status: str | None = None
    backend: str | None = None
    image_paths: tuple[str | Path, ...] = ()
    render_response: Mapping[str, Any] | None = None
    render_output_dir: str | Path | None = None
    expected_cameras: tuple[str, ...] = ()
    issues: tuple[DraftValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def attempted(self) -> bool:
        return self.status is not None

    def to_runtime_render_metadata(self) -> dict[str, Any] | None:
        if not self.attempted:
            return None
        return {
            "status": self.status,
            "backend": self.backend,
            "image_paths": [str(path) for path in self.image_paths],
            "render_response": self.render_response,
            "render_output_dir": (
                str(self.render_output_dir)
                if self.render_output_dir is not None
                else None
            ),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DraftTemplateResult:
    """Result from one scaffold template."""

    template_name: str
    status: TemplateStatus
    issues: tuple[DraftValidationIssue, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and all(
            issue.severity != "fail" for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_name": self.template_name,
            "status": self.status,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DraftValidationResult:
    """Aggregated scaffold validation output."""

    verdict: ValidationVerdict
    request: DraftValidationRequest
    plan: DraftValidationPlan
    template_results: tuple[DraftTemplateResult, ...] = ()
    issues: tuple[DraftValidationIssue, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "request": self.request.to_dict(),
            "plan": self.plan.to_dict(),
            "template_results": [result.to_dict() for result in self.template_results],
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DraftValidationContext:
    """Execution context passed to temporary templates."""

    request: DraftValidationRequest
    plan: DraftValidationPlan
    input_inventory: InputInventory
    working_dir: Path
    runtime_visual_evidence: RuntimeVisualEvidence = field(
        default_factory=RuntimeVisualEvidence,
    )
    previous_template_results: tuple[DraftTemplateResult, ...] = ()


class DraftValidationTemplate(Protocol):
    """Protocol implemented by temporary scaffold templates."""

    @property
    def name(self) -> str:
        """Template registry name."""

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        """Run the template against a scaffold context."""


class TemplateRegistry:
    """Allowlist-backed registry for issue #78 V1 scaffold templates."""

    def __init__(
        self,
        templates: Iterable[DraftValidationTemplate] | None = None,
    ) -> None:
        self._templates: dict[str, DraftValidationTemplate] = {}
        if templates is not None:
            for template in templates:
                self.register(template)

    def register(self, template: DraftValidationTemplate) -> None:
        if template.name not in V1_TEMPLATE_NAMES:
            raise DraftValidationError(
                f"Unknown validation template {template.name!r}. "
                f"Known templates: {', '.join(V1_TEMPLATE_NAMES)}"
            )
        self._templates[template.name] = template

    def get(self, name: str) -> DraftValidationTemplate:
        self.validate_template_names((name,))
        return self._templates[name]

    def names(self) -> tuple[str, ...]:
        return tuple(name for name in V1_TEMPLATE_NAMES if name in self._templates)

    def validate_template_names(self, names: Iterable[str]) -> None:
        for name in names:
            if name not in V1_TEMPLATE_NAMES:
                raise DraftValidationError(
                    f"Unknown validation template {name!r}. "
                    f"Known templates: {', '.join(V1_TEMPLATE_NAMES)}"
                )
            if name not in self._templates:
                raise DraftValidationError(
                    f"Validation template {name!r} is not registered"
                )


@dataclass(frozen=True)
class _FakeTemplate:
    name: str
    issue_code: str
    message: str

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        inventory = context.input_inventory
        metrics = {
            "usd_path_count": len(inventory.usd_paths),
            "image_path_count": len(inventory.image_paths)
            + len(inventory.render_bundle_image_paths),
            "video_path_count": len(inventory.video_paths),
            "focus_prim_path_count": len(inventory.focus_prim_paths),
        }
        issue = DraftValidationIssue(
            code=self.issue_code,
            severity="warn",
            message=self.message,
            details={
                "scaffold_only": True,
                "replacement_expected_in_later_lane": True,
            },
        )
        return DraftTemplateResult(
            template_name=self.name,
            status="skipped",
            issues=(issue,),
            metrics=metrics,
            evidence={
                "working_dir": str(context.working_dir),
                "input_count": len(inventory.items),
            },
            metadata={"template_kind": "fake"},
        )


@dataclass(frozen=True)
class _RenderValidTemplate:
    name: str = "render_valid"

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        policy = context.request.policy
        runtime_evidence = context.runtime_visual_evidence
        canonical_usd_visual_evidence = _canonical_usd_visual_evidence_enabled(policy)
        expected_cameras = _render_valid_expected_cameras(
            policy,
            runtime_evidence,
        )
        adapter_result = run_render_valid_adapter(
            image_paths=_render_valid_image_paths(
                context.input_inventory,
                policy,
                runtime_evidence,
            ),
            animation_frame_paths=(
                ()
                if canonical_usd_visual_evidence
                else _optional_path_sequence(
                    policy,
                    "animation_frame_paths",
                )
            ),
            frame_ids=(
                None
                if canonical_usd_visual_evidence
                else _optional_sequence(policy, "frame_ids")
            ),
            render_response=_render_valid_render_response(
                policy,
                runtime_evidence,
                canonical_usd_visual_evidence=canonical_usd_visual_evidence,
            ),
            expected_cameras=expected_cameras,
            expected_frames=_optional_sequence(policy, "expected_frames"),
            render_output_dir=_render_valid_render_output_dir(
                policy,
                runtime_evidence,
                canonical_usd_visual_evidence=canonical_usd_visual_evidence,
            ),
            backend=_optional_string(policy, "render_backend"),
            detect_error_material_artifacts=_strict_policy_bool(
                policy,
                "render_detect_error_material_artifacts",
            ),
        )
        return _draft_result_from_adapter_result(
            template_name=self.name,
            adapter_result=adapter_result,
            status=_render_adapter_status(adapter_result),
            metadata={
                "template_kind": "adapter",
                "adapter": "render_valid",
                "expected_cameras": expected_cameras,
                "runtime_render": runtime_evidence.to_runtime_render_metadata(),
            },
        )


@dataclass(frozen=True)
class _PhysicsSaneTemplate:
    name: str = "physics_sane"

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        usd_paths = context.input_inventory.usd_paths
        if not usd_paths:
            issue = DraftValidationIssue(
                code="physics_sane.evidence_missing",
                severity="warn",
                message="No USD evidence was supplied for physics_sane checks.",
            )
            return DraftTemplateResult(
                template_name=self.name,
                status="skipped",
                issues=(issue,),
                metrics={"usd_path_count": 0, "issue_count": 1},
                evidence={"usd_paths": []},
                metadata={"template_kind": "adapter", "adapter": "physics_sane"},
            )

        adapter_results = [
            run_physics_sane_adapter(
                usd_path,
                task_description=context.request.task_description,
                policy=context.request.policy,
                asset_validator_report=_asset_validator_report(
                    context.request.policy,
                    str(usd_path),
                ),
            )
            for usd_path in usd_paths
        ]
        return _draft_result_from_adapter_results(
            template_name=self.name,
            adapter_results=adapter_results,
            metadata={"template_kind": "adapter", "adapter": "physics_sane"},
        )


@dataclass(frozen=True)
class _PhysicalBehaviorTemplate:
    name: str = "physical_behavior"

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        policy = context.request.policy
        behavior_evidence_required = _policy_bool(
            policy,
            "behavior_evidence_required",
        )
        evidence_specs = _physical_behavior_evidence_specs(context)
        resolution = resolve_physical_behavior_evidence(
            evidence_specs,
            base_dir=context.request.base_dir,
            behavior_evidence_required=behavior_evidence_required,
            default_required=behavior_evidence_required,
        )
        resolution_issues = tuple(
            _issue_from_physical_behavior_issue(issue) for issue in resolution.issues
        )
        refine_summary_results = _physical_behavior_refine_summary_results(
            resolution.available_evidence,
        )
        behavior_status, behavior_issues, behavior_summary = (
            _physical_behavior_semantic_result(
                resolution,
                refine_summary_results=refine_summary_results,
                behavior_evidence_required=behavior_evidence_required,
            )
        )
        issues = resolution_issues + behavior_issues
        status = _physical_behavior_status(behavior_status, issues)
        metrics = _physical_behavior_metrics(
            context,
            resolution,
            refine_summary_results=refine_summary_results,
            issues=issues,
            status=status,
        )
        evidence = {
            "resolution": resolution.to_dict(),
            "evidence": [item.to_dict() for item in resolution.evidence],
            "available_evidence": [
                item.to_dict() for item in resolution.available_evidence
            ],
            "refine_summaries": [
                summary["summary"] for summary in refine_summary_results
            ],
            "behavior_summary": behavior_summary,
        }
        return DraftTemplateResult(
            template_name=self.name,
            status=status,
            issues=issues,
            metrics=metrics,
            evidence=evidence,
            metadata={
                "template_kind": "evidence_contract",
                "helper": "resolve_physical_behavior_evidence",
                "status_semantics": "pr10_physical_behavior",
            },
        )


@dataclass(frozen=True)
class _LookRightTemplate:
    name: str = "look_right"

    def run(self, context: DraftValidationContext) -> DraftTemplateResult:
        policy = context.request.policy
        runtime_evidence = context.runtime_visual_evidence
        runtime_render_metadata = runtime_evidence.to_runtime_render_metadata()
        canonical_usd_visual_evidence = _canonical_usd_visual_evidence_enabled(policy)
        raw_judge_response = _optional_string(policy, "look_right_response")
        live_judge_config = _look_right_live_judge_config(policy)
        final_judge_config = _look_right_final_judge_config(policy)
        vlm_available = _look_right_vlm_available(
            policy,
            raw_judge_response=raw_judge_response,
            live_judge_config=live_judge_config,
        )
        render_valid_result = _previous_adapter_result(
            context.previous_template_results,
            "render_valid",
        )
        judge_plan = build_look_right_judge_plan(
            context.request.task_description,
            current_image_paths=_look_right_current_image_paths(
                context.input_inventory,
                policy,
                runtime_evidence,
            ),
            render_image_paths=_look_right_render_image_paths(
                context.input_inventory,
                policy,
                runtime_evidence,
            ),
            sampled_video_frame_paths=(
                ()
                if canonical_usd_visual_evidence
                else _optional_path_sequence(
                    policy,
                    "sampled_video_frame_paths",
                )
            ),
            reference_image_paths=_optional_path_sequence(
                policy,
                "reference_image_paths",
            ),
            focused_image_paths=(
                {}
                if canonical_usd_visual_evidence
                else _optional_focused_image_paths(policy)
            ),
            focus_prim_paths=context.input_inventory.focus_prim_paths,
            reference_guidance=_optional_string(policy, "reference_guidance"),
            render_valid_result=render_valid_result,
            vlm_available=vlm_available,
            prompt_template=(
                _optional_string(policy, "look_right_prompt_template")
                or _optional_string(policy, "prompt_template")
                or DEFAULT_LOOK_RIGHT_PROMPT_TEMPLATE
            ),
            system_prompt=(
                _optional_string(policy, "look_right_system_prompt")
                or _optional_string(policy, "system_prompt")
                or DEFAULT_LOOK_RIGHT_SYSTEM_PROMPT
            ),
            temperature=_optional_float(policy, "look_right_temperature", 0.1),
            max_tokens=_optional_int(policy, "look_right_max_tokens", 2048),
        )
        plan_issues = tuple(
            _issue_from_look_right_issue(issue) for issue in judge_plan.issues
        )
        if render_valid_result is None and runtime_render_metadata is not None:
            plan_issues += _runtime_render_issues_from_metadata(
                {"runtime_render": runtime_render_metadata}
            )
        evidence: dict[str, Any] = {
            "judge_plan": judge_plan.to_dict(),
            "image_caption_pairs": [
                {"caption": caption, "path": path}
                for caption, path in judge_plan.image_caption_pairs
            ],
            "evidence_images": [
                evidence_image.to_dict()
                for evidence_image in judge_plan.evidence_images
            ],
            "render_valid_handoff": (
                dict(render_valid_result) if render_valid_result is not None else None
            ),
        }
        metrics: dict[str, Any] = {
            "ready_for_judge": judge_plan.ready_for_judge,
            "evidence_mode": (
                "canonical_usd"
                if canonical_usd_visual_evidence
                else "provided_visual_evidence"
            ),
            "evidence_image_count": len(judge_plan.evidence_images),
            "current_image_count": sum(
                1
                for evidence_image in judge_plan.evidence_images
                if evidence_image.role == "current"
            ),
            "reference_image_count": sum(
                1
                for evidence_image in judge_plan.evidence_images
                if evidence_image.role == "reference"
            ),
            "focus_image_count": sum(
                1
                for evidence_image in judge_plan.evidence_images
                if evidence_image.role == "focus"
            ),
            "vlm_invoked": False,
            "precomputed_response": raw_judge_response is not None,
            "issue_count": len(plan_issues),
        }

        if not judge_plan.ready_for_judge:
            return DraftTemplateResult(
                template_name=self.name,
                status=_look_right_unready_status(plan_issues),
                issues=plan_issues,
                metrics=metrics,
                evidence=evidence,
                metadata={
                    "template_kind": "visual_judge_plan",
                    "helper": "build_look_right_judge_plan",
                    "vlm_invoked": False,
                    "runtime_render": runtime_render_metadata,
                },
            )

        judge_invocation: LookRightJudgeInvocation | None = None
        if raw_judge_response is None:
            judge_invocation, unavailable_issue = _invoke_live_look_right_judge(
                judge_plan,
                live_judge_config,
            )
            if unavailable_issue is not None:
                metrics["issue_count"] = len(plan_issues) + 1
                return DraftTemplateResult(
                    template_name=self.name,
                    status="skipped",
                    issues=plan_issues + (unavailable_issue,),
                    metrics=metrics,
                    evidence=evidence,
                    metadata={
                        "template_kind": "visual_judge_plan",
                        "helper": "build_look_right_judge_plan",
                        "vlm_invoked": False,
                        "precomputed_response": False,
                        "runtime_render": runtime_render_metadata,
                    },
                )
            assert judge_invocation is not None
            raw_judge_response = judge_invocation.raw_response
            evidence["judge_invocation"] = judge_invocation.to_dict()
            metrics.update(
                {
                    "vlm_invoked": True,
                    "precomputed_response": False,
                    "judge_backend": judge_invocation.backend_name,
                    "judge_model": judge_invocation.model_name,
                }
            )

        if raw_judge_response is None:
            unavailable_issue = DraftValidationIssue(
                code=VISUAL_JUDGE_UNAVAILABLE,
                severity="warn",
                message=(
                    "The look_right judge plan is ready, but no VLM executor or "
                    "precomputed look_right_response was supplied."
                ),
                details={"vlm_available": False, "plan_ready": True},
            )
            metrics["issue_count"] = len(plan_issues) + 1
            return DraftTemplateResult(
                template_name=self.name,
                status="skipped",
                issues=plan_issues + (unavailable_issue,),
                metrics=metrics,
                evidence=evidence,
                metadata={
                    "template_kind": "visual_judge_plan",
                    "helper": "build_look_right_judge_plan",
                    "vlm_invoked": False,
                    "runtime_render": runtime_render_metadata,
                },
            )

        pass_threshold = _optional_float(policy, "look_right_pass_threshold", 0.7)
        needs_refinement_threshold = _optional_float(
            policy,
            "look_right_needs_refinement_threshold",
            0.55,
        )
        final_judge_result = _normalize_live_look_right_judgment(
            raw_judge_response,
            judge_plan,
            final_judge_config,
            pass_threshold=pass_threshold,
            needs_refinement_threshold=needs_refinement_threshold,
        )
        judgment = final_judge_result.judgment
        judgment_issues = _issues_from_look_right_judgment(judgment.to_dict())
        issues = plan_issues + judgment_issues
        evidence["judgment"] = judgment.to_dict()
        evidence["final_judge"] = final_judge_result.to_dict()
        metrics.update(
            {
                "judgment_verdict": judgment.verdict,
                "judgment_score": judgment.score,
                "final_judge_method": final_judge_result.metadata.get("method"),
                "llm_final_judge_invoked": final_judge_result.metadata.get(
                    "llm_invoked"
                ),
                "final_judge_backend": final_judge_result.backend_name,
                "final_judge_model": final_judge_result.model_name,
                "issue_count": len(issues),
            }
        )
        return DraftTemplateResult(
            template_name=self.name,
            status=_look_right_judgment_status(judgment.verdict),
            issues=issues,
            metrics=metrics,
            evidence=evidence,
            metadata={
                "template_kind": "visual_judge_plan",
                "helper": "build_look_right_judge_plan",
                "result_helper": "normalize_look_right_judgment",
                "vlm_invoked": judge_invocation is not None,
                "precomputed_response": judge_invocation is None,
                "llm_final_judge_invoked": final_judge_result.metadata.get(
                    "llm_invoked"
                ),
                "final_judge_method": final_judge_result.metadata.get("method"),
                "judge_backend": (
                    judge_invocation.backend_name
                    if judge_invocation is not None
                    else None
                ),
                "judge_model": (
                    judge_invocation.model_name
                    if judge_invocation is not None
                    else None
                ),
                "runtime_render": runtime_render_metadata,
            },
        )


def create_default_scaffold_registry() -> TemplateRegistry:
    """Return the default scaffold template registry."""

    return TemplateRegistry(
        (
            _LookRightTemplate(),
            _RenderValidTemplate(),
            _PhysicsSaneTemplate(),
            _PhysicalBehaviorTemplate(),
        )
    )


def create_draft_validation_request(
    *,
    task_description: str,
    inputs: Sequence[str | Path],
    working_dir: str | Path,
    base_dir: str | Path | None = None,
    focus_prim_paths: Sequence[str] = (),
    requested_templates: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> DraftValidationRequest:
    """Create a normalized temporary validation request."""

    return DraftValidationRequest(
        task_description=task_description,
        inputs=tuple(inputs),
        working_dir=Path(working_dir),
        base_dir=Path(base_dir) if base_dir is not None else None,
        focus_prim_paths=tuple(focus_prim_paths),
        requested_templates=tuple(requested_templates),
        policy=dict(policy or {}),
        dry_run=dry_run,
        metadata=dict(metadata or {}),
    )


def plan_validation(
    request: DraftValidationRequest,
    registry: TemplateRegistry | None = None,
) -> DraftValidationPlan:
    """Create a rules-only validation plan from resolved inputs."""

    template_registry = registry or create_default_scaffold_registry()
    inventory = resolve_input_inventory(
        request.inputs,
        base_dir=request.base_dir,
        focus_prim_paths=request.focus_prim_paths,
        working_dir=request.working_dir,
        create_working_dir=True,
    )

    if request.requested_templates:
        selected_template_names = _dedupe_preserve_order(request.requested_templates)
        template_registry.validate_template_names(selected_template_names)
        selection_reason = "requested by caller"
    else:
        selected_template_names = _select_templates(request, inventory)
        template_registry.validate_template_names(selected_template_names)
        selection_reason = "selected by rules"
    if not selected_template_names:
        raise DraftValidationError("No validation templates were selected")

    steps = tuple(
        DraftValidationStep(
            template_name=template_name,
            reason=_step_reason(template_name, selection_reason),
            inputs_needed=_step_inputs_needed(template_name),
        )
        for template_name in selected_template_names
    )
    return DraftValidationPlan(
        steps=steps,
        input_inventory=inventory,
        reasoning_summary=(
            "Rules-only planner selected registered scaffold templates. "
            "Focus prim paths are copied only from caller input."
        ),
    )


def run_validation_scaffold(
    request: DraftValidationRequest,
    registry: TemplateRegistry | None = None,
    *,
    dry_run: bool | None = None,
) -> DraftValidationResult:
    """Plan, optionally run fake templates, and write scaffold artifacts."""

    template_registry = registry or create_default_scaffold_registry()
    plan = plan_validation(request, template_registry)
    if plan.input_inventory.working_dir is None:
        raise DraftValidationError("A working directory is required to write artifacts")
    working_dir = Path(plan.input_inventory.working_dir)
    write_plan_artifact(plan, working_dir=working_dir)

    should_dry_run = request.dry_run if dry_run is None else dry_run
    if should_dry_run:
        return DraftValidationResult(
            verdict="planned",
            request=request,
            plan=plan,
            metadata={
                "dry_run": True,
                "artifact_paths": {"plan": str(working_dir / "plan.json")},
            },
        )

    runtime_visual_evidence = _runtime_visual_evidence_for_request(
        request,
        input_inventory=plan.input_inventory,
        working_dir=working_dir,
        selected_template_names=tuple(step.template_name for step in plan.steps),
    )
    context = DraftValidationContext(
        request=request,
        plan=plan,
        input_inventory=plan.input_inventory,
        working_dir=working_dir,
        runtime_visual_evidence=runtime_visual_evidence,
    )
    template_results_list: list[DraftTemplateResult] = []
    for step in plan.steps:
        step_context = DraftValidationContext(
            request=context.request,
            plan=context.plan,
            input_inventory=context.input_inventory,
            working_dir=context.working_dir,
            runtime_visual_evidence=context.runtime_visual_evidence,
            previous_template_results=tuple(template_results_list),
        )
        template_results_list.append(
            _run_template_safely(
                template_registry.get(step.template_name),
                step_context,
            )
        )
    template_results = tuple(template_results_list)
    issues = tuple(issue for result in template_results for issue in result.issues)
    metrics: dict[str, Any] = {
        result.template_name: dict(result.metrics) for result in template_results
    }
    evidence = {
        result.template_name: dict(result.evidence) for result in template_results
    }
    result = DraftValidationResult(
        verdict=_aggregate_verdict(template_results),
        request=request,
        plan=plan,
        template_results=template_results,
        issues=issues,
        metrics=metrics,
        evidence=evidence,
        metadata={
            "dry_run": False,
            "artifact_paths": {
                "plan": str(working_dir / "plan.json"),
                "validation_result": str(working_dir / "validation_result.json"),
            },
        },
    )
    write_result_artifact(result, working_dir=working_dir)
    return result


def write_plan_artifact(
    plan: DraftValidationPlan,
    *,
    working_dir: str | Path,
    filename: str = "plan.json",
) -> Path:
    """Write a scaffold plan artifact and return the artifact path."""

    output_path = Path(working_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_pretty_json(plan.to_dict()), encoding="utf-8")
    return output_path


def write_result_artifact(
    result: DraftValidationResult,
    *,
    working_dir: str | Path,
    filename: str = "validation_result.json",
) -> Path:
    """Write a provisional validation result artifact."""

    output_path = Path(working_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_pretty_json(result.to_dict()), encoding="utf-8")
    return output_path


def _select_templates(
    request: DraftValidationRequest,
    inventory: InputInventory,
) -> tuple[str, ...]:
    selected: list[str] = []

    has_render_evidence = bool(
        inventory.image_paths
        or inventory.render_bundle_image_paths
        or inventory.usd_paths
    )
    has_behavior_evidence = (
        bool(inventory.video_paths)
        or _policy_bool(request.policy, "behavior_evidence_required")
        or _policy_has_behavior_evidence(request.policy)
    )

    explicit_expect_physics = _optional_policy_bool(request.policy, "expect_physics")
    if inventory.usd_paths and infer_physics_expected(
        task_text=request.task_description,
        expect_physics=explicit_expect_physics,
    ):
        selected.append("physics_sane")

    if has_render_evidence:
        selected.append("render_valid")

    if has_behavior_evidence:
        selected.append("physical_behavior")
    elif has_render_evidence:
        selected.append("look_right")

    return _dedupe_preserve_order(selected)


def _policy_has_behavior_evidence(policy: Mapping[str, Any]) -> bool:
    for key in (
        "physical_behavior_evidence",
        "behavior_evidence",
        "time_sampled_usd_paths",
        "animation_usd_paths",
        "behavior_video_paths",
        "video_paths",
        "simulation_json_paths",
        "trajectory_metrics_paths",
        "physical_behavior_refine_output_dir",
        "physics_refine_output_dir",
        "refine_output_dir",
        "physical_behavior_refine_summary_path",
        "refine_summary_path",
    ):
        if _policy_value_present(policy.get(key)):
            return True
    return False


def _policy_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_policy_value_present(item) for item in value)
    if isinstance(value, Mapping):
        return bool(value)
    return True


def _step_reason(template_name: str, selection_reason: str) -> str:
    reasons = {
        "look_right": "visual evidence is available",
        "render_valid": "render or USD evidence is available",
        "physics_sane": "physics expectations are present for USD input",
        "physical_behavior": "behavior evidence was provided or required",
    }
    base_reason = reasons.get(template_name, "template was selected")
    return f"{base_reason}; {selection_reason}"


def _step_inputs_needed(template_name: str) -> tuple[str, ...]:
    requirements = {
        "look_right": ("images_or_usd",),
        "render_valid": ("images_or_render_bundle_or_usd",),
        "physics_sane": ("usd",),
        "physical_behavior": ("video_or_policy",),
    }
    return requirements.get(template_name, ())


def _run_template_safely(
    template: DraftValidationTemplate,
    context: DraftValidationContext,
) -> DraftTemplateResult:
    try:
        return template.run(context)
    except Exception as exc:
        issue = DraftValidationIssue(
            code="agent.template_error",
            severity="fail",
            message=f"Template {template.name!r} failed: {exc}",
            details={"template_name": template.name},
        )
        return DraftTemplateResult(
            template_name=template.name,
            status="error",
            issues=(issue,),
            metadata={"exception_type": type(exc).__name__},
        )


def _aggregate_verdict(
    template_results: Sequence[DraftTemplateResult],
) -> ValidationVerdict:
    if any(
        result.status in {"failed", "error"}
        or any(issue.severity == "fail" for issue in result.issues)
        for result in template_results
    ):
        return "fail"
    if any(result.status == "needs_refinement" for result in template_results):
        return "needs_refinement"
    if any(
        result.status in {"warn", "skipped"}
        or any(issue.severity == "warn" for issue in result.issues)
        for result in template_results
    ):
        return "warn"
    return "pass"


def _render_image_paths(inventory: InputInventory) -> tuple[Path, ...]:
    return inventory.image_paths + inventory.render_bundle_image_paths


def _runtime_visual_evidence_for_request(
    request: DraftValidationRequest,
    *,
    input_inventory: InputInventory,
    working_dir: Path,
    selected_template_names: Sequence[str],
) -> RuntimeVisualEvidence:
    if _canonical_usd_input_missing(
        request,
        input_inventory=input_inventory,
        selected_template_names=selected_template_names,
    ):
        _LOGGER.warning(
            "Canonical USD visual evidence requires a USD input; visual "
            "validation will fail closed."
        )
        return _runtime_visual_evidence_with_missing_canonical_usd_input(
            request.policy,
        )

    if _canonical_runtime_render_disabled(
        request,
        selected_template_names=selected_template_names,
    ):
        _LOGGER.warning(
            "Canonical USD visual evidence requires runtime rendering; "
            "policy.runtime_render_usd=false will fail visual validation."
        )
        return _runtime_visual_evidence_with_disabled_canonical_runtime_render(
            request.policy,
        )

    if not _should_runtime_render_usd(
        request,
        input_inventory=input_inventory,
        selected_template_names=selected_template_names,
    ):
        return _runtime_visual_evidence_from_policy(request.policy)

    render_result = render_usd_visual_evidence(
        usd_paths=input_inventory.usd_paths,
        working_dir=working_dir,
        policy=request.policy,
    )
    canonical_usd_visual_evidence = _canonical_usd_visual_evidence_enabled(
        request.policy,
    )
    generated_paths = tuple(
        str(path) for path in render_result.get("image_paths", ()) if path
    )
    if canonical_usd_visual_evidence and not generated_paths:
        render_result = _canonical_runtime_render_result_with_missing_evidence(
            render_result,
        )
    return _runtime_visual_evidence_from_render_result(render_result)


def _should_runtime_render_usd(
    request: DraftValidationRequest,
    *,
    input_inventory: InputInventory,
    selected_template_names: Sequence[str],
) -> bool:
    if not input_inventory.usd_paths:
        return False
    canonical_usd_visual_evidence = _canonical_usd_visual_evidence_enabled(
        request.policy,
    )
    if _optional_policy_bool(request.policy, "runtime_render_usd") is False:
        return False
    if not _visual_templates_selected(selected_template_names):
        return False
    if canonical_usd_visual_evidence:
        return True
    return not _has_visual_current_or_render_evidence(input_inventory, request.policy)


def _canonical_usd_visual_evidence_enabled(policy: Mapping[str, Any]) -> bool:
    """Whether all visual scoring should be based only on canonical USD renders.

    This is intentionally opt-in so existing image-only validation remains
    possible. For artifact QA, enabling this avoids comparing scores produced
    from different caller-supplied image bundles for the same USD. Canonical
    mode requires generated runtime renders; pairing it with an explicit
    ``runtime_render_usd: false`` fails closed instead of falling back to
    caller-provided visual evidence.
    """

    return _canonical_usd_visual_evidence_requested(policy)


def _canonical_usd_visual_evidence_requested(policy: Mapping[str, Any]) -> bool:
    if _optional_policy_bool(policy, "canonical_visual_evidence") is True:
        return True
    mode = _optional_string(policy, "visual_evidence_mode")
    return mode is not None and mode.strip().lower() in {
        "canonical_usd",
        "canonical-usd",
    }


def _canonical_usd_input_missing(
    request: DraftValidationRequest,
    *,
    input_inventory: InputInventory,
    selected_template_names: Sequence[str],
) -> bool:
    return (
        _canonical_usd_visual_evidence_requested(request.policy)
        and not input_inventory.usd_paths
        and _visual_templates_selected(selected_template_names)
    )


def _canonical_runtime_render_disabled(
    request: DraftValidationRequest,
    *,
    selected_template_names: Sequence[str],
) -> bool:
    return (
        _canonical_usd_visual_evidence_enabled(request.policy)
        and _optional_policy_bool(request.policy, "runtime_render_usd") is False
        and _visual_templates_selected(selected_template_names)
    )


def _visual_templates_selected(selected_template_names: Sequence[str]) -> bool:
    return any(
        template_name in {"render_valid", "look_right"}
        for template_name in selected_template_names
    )


def _runtime_visual_evidence_with_missing_canonical_usd_input(
    policy: Mapping[str, Any],
) -> RuntimeVisualEvidence:
    issue = DraftValidationIssue(
        code=CANONICAL_USD_INPUT_MISSING,
        severity="fail",
        message=(
            "canonical_visual_evidence requires a USD input so the validator "
            "can generate trusted runtime render evidence."
        ),
        details={
            "canonical_visual_evidence": True,
            "usd_input_present": False,
        },
    )
    return RuntimeVisualEvidence(
        status="failed",
        backend=_optional_string(policy, "render_backend") or "runtime",
        issues=(issue,),
        metadata={
            "canonical_visual_evidence": True,
            "runtime_render_required": True,
        },
    )


def _runtime_visual_evidence_with_disabled_canonical_runtime_render(
    policy: Mapping[str, Any],
) -> RuntimeVisualEvidence:
    issue = DraftValidationIssue(
        code=CANONICAL_RUNTIME_RENDER_DISABLED,
        severity="fail",
        message=(
            "canonical_visual_evidence requires generated runtime USD renders, "
            "but policy.runtime_render_usd is false."
        ),
        details={
            "canonical_visual_evidence": True,
            "runtime_render_usd": False,
        },
    )
    return RuntimeVisualEvidence(
        status="failed",
        backend=_optional_string(policy, "render_backend") or "runtime",
        issues=(issue,),
        metadata={
            "canonical_visual_evidence": True,
            "runtime_render_required": True,
        },
    )


def _canonical_runtime_render_result_with_missing_evidence(
    render_result: Mapping[str, Any],
) -> dict[str, Any]:
    failed_result = dict(render_result)
    failed_result["status"] = "failed"
    failed_result["image_paths"] = []
    failed_result["render_response"] = None
    failed_result["render_output_dir"] = None
    issues = [dict(issue) for issue in _mapping_sequence(failed_result, "issues")]
    if not any(
        issue.get("code") == CANONICAL_RUNTIME_RENDER_MISSING for issue in issues
    ):
        issues.append(
            {
                "code": CANONICAL_RUNTIME_RENDER_MISSING,
                "severity": "fail",
                "message": (
                    "canonical_visual_evidence requires runtime USD renders, "
                    "but the runtime renderer produced no image paths."
                ),
                "details": {
                    "canonical_visual_evidence": True,
                    "runtime_render_status": render_result.get("status"),
                },
            }
        )
    failed_result["issues"] = issues
    return failed_result


def _runtime_visual_evidence_from_render_result(
    render_result: Mapping[str, Any],
) -> RuntimeVisualEvidence:
    render_response = render_result.get("render_response")
    if not isinstance(render_response, Mapping):
        render_response = None
    return RuntimeVisualEvidence(
        status=_optional_string(render_result, "status"),
        backend=_optional_string(render_result, "backend"),
        image_paths=tuple(
            str(path) for path in render_result.get("image_paths", ()) if path
        ),
        render_response=render_response,
        render_output_dir=(
            render_result.get("render_output_dir")
            if isinstance(render_result.get("render_output_dir"), str | Path)
            else None
        ),
        expected_cameras=(
            _render_response_camera_names(render_response)
            if render_response is not None
            else ()
        ),
        issues=tuple(
            _issue_from_adapter_issue(issue)
            for issue in _mapping_sequence(render_result, "issues")
        ),
        metadata=dict(_mapping_value(render_result, "metadata")),
    )


def _runtime_visual_evidence_from_policy(
    policy: Mapping[str, Any],
) -> RuntimeVisualEvidence:
    """Import legacy caller-supplied runtime metadata without mutating policy."""

    runtime_render = policy.get("runtime_render")
    if not isinstance(runtime_render, Mapping):
        return RuntimeVisualEvidence()
    return _runtime_visual_evidence_from_render_result(runtime_render)


def _canonical_runtime_render_image_paths(
    runtime_evidence: RuntimeVisualEvidence,
) -> tuple[str | Path, ...]:
    return runtime_evidence.image_paths


def _has_visual_current_or_render_evidence(
    inventory: InputInventory,
    policy: Mapping[str, Any],
) -> bool:
    return bool(
        inventory.image_paths
        or inventory.render_bundle_image_paths
        or _optional_path_sequence(policy, "current_image_paths")
        or _optional_path_sequence(policy, "render_image_paths")
        or _focused_image_path_values(_optional_focused_image_paths(policy))
        or policy.get("render_response") is not None
    )


def _render_valid_image_paths(
    inventory: InputInventory,
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
) -> tuple[str | Path, ...]:
    if _canonical_usd_visual_evidence_enabled(policy):
        return _canonical_runtime_render_image_paths(runtime_evidence)
    return (
        _render_image_paths(inventory)
        + (_optional_path_sequence(policy, "current_image_paths") or ())
        + (_optional_path_sequence(policy, "render_image_paths") or ())
        + runtime_evidence.image_paths
        + _focused_image_path_values(_optional_focused_image_paths(policy))
    )


def _render_valid_expected_cameras(
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
) -> tuple[str, ...] | None:
    if runtime_evidence.expected_cameras:
        return runtime_evidence.expected_cameras
    return _optional_string_sequence(policy, "expected_cameras")


def _render_valid_render_response(
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
    *,
    canonical_usd_visual_evidence: bool,
) -> Any | None:
    if canonical_usd_visual_evidence:
        return runtime_evidence.render_response
    if "render_response" in policy:
        return _optional_value(policy, "render_response")
    return runtime_evidence.render_response


def _render_valid_render_output_dir(
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
    *,
    canonical_usd_visual_evidence: bool,
) -> str | Path | None:
    if canonical_usd_visual_evidence:
        return runtime_evidence.render_output_dir
    policy_output_dir = _optional_path(policy, "render_output_dir")
    return policy_output_dir or runtime_evidence.render_output_dir


def _physical_behavior_evidence_specs(
    context: DraftValidationContext,
) -> tuple[Any, ...]:
    policy = context.request.policy
    required = _policy_bool(policy, "behavior_evidence_required")
    specs: list[Any] = []
    specs.extend(_policy_physical_behavior_evidence(policy))
    specs.extend(
        _inventory_physical_behavior_evidence(context.input_inventory, required)
    )
    specs.extend(_sampled_frame_evidence(policy, required))
    specs.extend(
        _refine_output_evidence(
            policy,
            base_dir=context.request.base_dir,
            default_required=required,
        )
    )
    return _dedupe_evidence_specs(specs)


def _policy_physical_behavior_evidence(policy: Mapping[str, Any]) -> list[Any]:
    specs: list[Any] = []
    for key in ("physical_behavior_evidence", "behavior_evidence"):
        value = policy.get(key)
        if value is not None:
            specs.extend(_normalize_evidence_policy_value(value))
    for key, kind, role in (
        ("time_sampled_usd_paths", "time_sampled_usd", "time_sampled_usd"),
        ("animation_usd_paths", "animation_usd", "animation_usd"),
        ("behavior_video_paths", "video", "video"),
        ("video_paths", "video", "video"),
        ("simulation_json_paths", "simulation_json", "simulation_json"),
        ("trajectory_metrics_paths", "trajectory_metrics", "trajectory_metrics"),
    ):
        for path in _optional_path_sequence(policy, key) or ():
            specs.append({"path": path, "kind": kind, "role": role})
    return specs


def _normalize_evidence_policy_value(value: Any) -> list[Any]:
    if isinstance(value, str | Path) or isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return list(value)
    return [value]


def _inventory_physical_behavior_evidence(
    inventory: InputInventory,
    required: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for usd_path in inventory.usd_paths:
        specs.append(
            {
                "path": usd_path,
                "kind": "time_sampled_usd",
                "role": "input_usd",
                "required": required,
            }
        )
    for video_path in inventory.video_paths:
        specs.append(
            {
                "path": video_path,
                "kind": "video",
                "role": "input_video",
                "required": required,
            }
        )
    return specs


def _sampled_frame_evidence(
    policy: Mapping[str, Any],
    required: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in _optional_path_sequence(policy, "sampled_video_frame_paths") or ():
        specs.append(
            {
                "path": path,
                "kind": "sampled_frame",
                "role": "sampled_video_frame",
                "required": required,
            }
        )
    return specs


def _refine_output_evidence(
    policy: Mapping[str, Any],
    *,
    base_dir: str | Path | None,
    default_required: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for key in (
        "physical_behavior_refine_output_dir",
        "physics_refine_output_dir",
        "refine_output_dir",
    ):
        path_value = policy.get(key)
        if path_value is None:
            continue
        output_dir = _resolve_policy_path(path_value, base_dir)
        specs.extend(
            _discover_refine_output_evidence(
                output_dir,
                required=default_required,
            )
        )

    for key in (
        "physical_behavior_refine_summary_path",
        "refine_summary_path",
    ):
        for path in _optional_path_sequence(policy, key) or ():
            specs.append(
                {
                    "path": path,
                    "kind": "simulation_json",
                    "role": "refine_summary",
                    "required": default_required,
                }
            )
    return specs


def _resolve_policy_path(value: Any, base_dir: str | Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    base_path = Path.cwd() if base_dir is None else Path(base_dir)
    return (base_path / path).resolve(strict=False)


def _discover_refine_output_evidence(
    output_dir: Path,
    *,
    required: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if not output_dir.exists():
        specs.append(
            {
                "path": output_dir / "refine_summary.json",
                "kind": "simulation_json",
                "role": "refine_summary",
                "required": required,
            }
        )
        return specs
    if output_dir.is_file():
        specs.append(
            {
                "path": output_dir,
                "kind": "simulation_json",
                "role": "refine_summary",
                "required": required,
            }
        )
        return specs

    summary_path = output_dir / "refine_summary.json"
    if summary_path.exists():
        specs.append(
            {
                "path": summary_path,
                "kind": "simulation_json",
                "role": "refine_summary",
                "required": required,
            }
        )

    for iter_dir in _iter_refine_dirs(output_dir):
        for filename, kind, role in (
            ("history.jsonl", "trajectory_metrics", "trial_history"),
            ("judge_result.json", "simulation_json", "judge_result"),
            ("refine_result.json", "simulation_json", "refine_result"),
            ("tune_results.json", "simulation_json", "tune_results"),
        ):
            path = iter_dir / filename
            if path.exists():
                specs.append(
                    {
                        "path": path,
                        "kind": kind,
                        "role": role,
                        "required": False,
                    }
                )
        render_dir = iter_dir / "render"
        specs.extend(_render_dir_evidence(render_dir))

    final_dir = output_dir / "final"
    if final_dir.exists():
        for filename, kind, role in (
            ("history.jsonl", "trajectory_metrics", "final_trial_history"),
            ("judge_result.json", "simulation_json", "final_judge_result"),
            ("refine_result.json", "simulation_json", "final_refine_result"),
            ("tune_results.json", "simulation_json", "final_tune_results"),
        ):
            path = final_dir / filename
            if path.exists():
                specs.append(
                    {
                        "path": path,
                        "kind": kind,
                        "role": role,
                        "required": False,
                    }
                )
        specs.extend(_render_dir_evidence(final_dir / "render"))

    return specs


def _iter_refine_dirs(output_dir: Path) -> tuple[Path, ...]:
    if not output_dir.exists():
        return ()
    iter_dirs: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        suffix = child.name.removeprefix("iter_")
        if suffix.isdigit():
            iter_dirs.append((int(suffix), child))
    return tuple(path for _, path in sorted(iter_dirs))


def _render_dir_evidence(render_dir: Path) -> list[dict[str, Any]]:
    if not render_dir.exists() or not render_dir.is_dir():
        return []
    specs: list[dict[str, Any]] = []
    files = sorted(child for child in render_dir.iterdir() if child.is_file())
    video_paths = [
        path for path in files if path.suffix.lower() in {".mp4", ".mov", ".webm"}
    ]
    frame_paths = [
        path
        for path in files
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    selected_paths = video_paths[:MAX_BEHAVIOR_RENDER_EVIDENCE_FILES]
    remaining_slots = MAX_BEHAVIOR_RENDER_EVIDENCE_FILES - len(selected_paths)
    if remaining_slots > 0:
        selected_paths.extend(frame_paths[:remaining_slots])
    for path in selected_paths:
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".mov", ".webm"}:
            specs.append(
                {
                    "path": path,
                    "kind": "video",
                    "role": "rendered_rollout",
                    "required": False,
                }
            )
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            specs.append(
                {
                    "path": path,
                    "kind": "sampled_frame",
                    "role": "sampled_frame",
                    "required": False,
                }
            )
    return specs


def _dedupe_evidence_specs(specs: Sequence[Any]) -> tuple[Any, ...]:
    deduped: list[Any] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for spec in specs:
        if isinstance(spec, Mapping):
            path = spec.get("path")
            key = (
                str(path) if path is not None else repr(spec),
                str(spec.get("kind")) if spec.get("kind") is not None else None,
                str(spec.get("role")) if spec.get("role") is not None else None,
            )
        else:
            key = (str(spec), None, None)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return tuple(deduped)


def _physical_behavior_refine_summary_results(
    available_evidence: Sequence[PhysicalBehaviorEvidence],
) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    for item in available_evidence:
        path = Path(item.path)
        if item.kind != "simulation_json":
            continue
        payload = _load_json_mapping(path)
        if payload is None:
            continue
        if _is_refine_summary(payload):
            summaries.append(
                {
                    "path": str(path),
                    "summary": _summarize_refine_summary(path, payload),
                }
            )
        elif _is_judge_result(item, payload):
            summaries.append(
                {
                    "path": str(path),
                    "summary": _summarize_judge_result(
                        path,
                        payload,
                        role=item.role,
                    ),
                }
            )
        elif _is_tune_results_with_judge(payload):
            summaries.append(
                {
                    "path": str(path),
                    "summary": _summarize_judge_result(
                        path,
                        _mapping_value(payload, "judge"),
                        role=item.role,
                    ),
                }
            )
    return tuple(summaries)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(data, Mapping):
        return data
    return None


def _is_refine_summary(payload: Mapping[str, Any]) -> bool:
    return _has_refine_summary_shape(payload)


def _has_refine_summary_shape(payload: Mapping[str, Any]) -> bool:
    iterations = payload.get("iterations")
    return "termination_reason" in payload or (
        isinstance(iterations, Sequence)
        and not isinstance(iterations, str | bytes | bytearray)
    )


def _is_judge_result(
    item: PhysicalBehaviorEvidence,
    payload: Mapping[str, Any],
) -> bool:
    if not _has_judge_result_shape(payload):
        return False
    return (
        item.role in {"judge_result", "final_judge_result"}
        or Path(item.path).name == "judge_result.json"
        or _allows_judge_shape_inference(item.role)
    )


def _allows_judge_shape_inference(role: str | None) -> bool:
    return role in {None, "simulation_json", "refine_summary"}


def _has_judge_result_shape(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.keys()
        & {
            "decision",
            "status",
            "score",
            "reasoning",
            "llm_unavailable",
            "error",
            "error_type",
            "cancelled",
        }
    )


def _is_tune_results_with_judge(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("judge"), Mapping) and _has_judge_result_shape(
        _mapping_value(payload, "judge")
    )


def _summarize_refine_summary(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_iterations = payload.get("iterations", ())
    if not isinstance(raw_iterations, Sequence) or isinstance(
        raw_iterations, str | bytes | bytearray
    ):
        raw_iterations = ()
    iterations = tuple(item for item in raw_iterations if isinstance(item, Mapping))
    final_record = _final_refine_record(payload, iterations)
    termination_reason = _optional_text(payload.get("termination_reason"))
    final_iteration = _optional_int_value(payload.get("final_iteration"))
    if final_iteration is None and final_record is not None:
        final_iteration = _optional_int_value(final_record.get("iteration"))
    judge_decision = (
        _optional_text(final_record.get("judge_decision"))
        if final_record is not None
        else None
    )
    judge_score = (
        _optional_float_value(final_record.get("judge_score"))
        if final_record is not None
        else None
    )
    error = (
        _optional_text(final_record.get("error")) if final_record is not None else None
    )
    cancelled = (
        bool(final_record.get("cancelled")) if final_record is not None else False
    )
    return {
        "kind": "refine_summary",
        "path": str(path),
        "termination_reason": termination_reason,
        "final_iteration": final_iteration,
        "iteration_count": len(iterations),
        "judge_decision": judge_decision,
        "judge_score": judge_score,
        "judge_reasoning": (
            _optional_text(final_record.get("judge_reasoning"))
            if final_record is not None
            else None
        ),
        "judge_llm_unavailable_count": sum(
            1 for record in iterations if record.get("judge_llm_unavailable") is True
        ),
        "refine_llm_unavailable_count": sum(
            1 for record in iterations if record.get("refine_llm_unavailable") is True
        ),
        "cancelled": cancelled,
        "error": error,
        "status": _behavior_status_from_refine_summary(
            termination_reason=termination_reason,
            judge_decision=judge_decision,
            error=error,
            cancelled=cancelled,
        ),
    }


def _final_refine_record(
    payload: Mapping[str, Any],
    iterations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not iterations:
        return None
    final_iteration = _optional_int_value(payload.get("final_iteration"))
    if final_iteration is not None:
        for record in iterations:
            if _optional_int_value(record.get("iteration")) == final_iteration:
                return record
    return iterations[-1]


def _summarize_judge_result(
    path: Path,
    payload: Mapping[str, Any],
    *,
    role: str | None = None,
) -> dict[str, Any]:
    decision = _optional_text(payload.get("decision"))
    score = _optional_float_value(payload.get("score"))
    status_text = _optional_text(payload.get("status"))
    error = _optional_text(payload.get("error")) or _optional_text(
        payload.get("error_type")
    )
    cancelled = status_text == "cancelled" or payload.get("cancelled") is True
    llm_unavailable = payload.get("llm_unavailable") is True
    return {
        "kind": "judge_result",
        "path": str(path),
        "role": role,
        "execution_status": status_text,
        "iteration": _iteration_from_refine_artifact_path(path),
        "final_iteration": None,
        "iteration_count": _optional_int_value(payload.get("iterations"))
        or _optional_int_value(payload.get("attempted_iterations")),
        "judge_decision": decision,
        "judge_score": score,
        "judge_reasoning": _optional_text(payload.get("reasoning")),
        "judge_llm_unavailable_count": 1 if llm_unavailable else 0,
        "refine_llm_unavailable_count": 0,
        "cancelled": cancelled,
        "error": error,
        "status": _behavior_status_from_judge_result(
            execution_status=status_text,
            judge_decision=decision,
            error=error,
            cancelled=cancelled,
            llm_unavailable=llm_unavailable,
        ),
    }


def _iteration_from_refine_artifact_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if not part.startswith("iter_"):
            continue
        suffix = part.removeprefix("iter_")
        if suffix.isdigit():
            return int(suffix)
    return None


def _behavior_status_from_refine_summary(
    *,
    termination_reason: str | None,
    judge_decision: str | None,
    error: str | None,
    cancelled: bool,
) -> TemplateStatus:
    if error or cancelled or termination_reason in {"error", "failed", "cancelled"}:
        return "failed"
    if termination_reason == "degraded":
        return "warn"
    if termination_reason == "approved":
        return "passed"
    if termination_reason == "max_iterations":
        return "needs_refinement"
    if judge_decision == "approve":
        return "passed"
    if judge_decision == "continue":
        return "needs_refinement"
    return "warn"


def _behavior_status_from_judge_result(
    *,
    execution_status: str | None,
    judge_decision: str | None,
    error: str | None,
    cancelled: bool,
    llm_unavailable: bool,
) -> TemplateStatus:
    if error or cancelled or execution_status in {"error", "failed", "cancelled"}:
        return "failed"
    if llm_unavailable or execution_status == "degraded":
        return "warn"
    if judge_decision == "approve":
        return "passed"
    if judge_decision == "continue":
        return "needs_refinement"
    return "warn"


def _physical_behavior_semantic_result(
    resolution: PhysicalBehaviorEvidenceResolution,
    *,
    refine_summary_results: Sequence[Mapping[str, Any]],
    behavior_evidence_required: bool,
) -> tuple[TemplateStatus, tuple[DraftValidationIssue, ...], dict[str, Any]]:
    available_evidence = resolution.available_evidence
    effective_required = behavior_evidence_required or any(
        item.required for item in resolution.evidence
    )
    if any(issue.severity == "fail" for issue in resolution.issues):
        return "failed", (), {"reason": "blocking_evidence_issue"}

    if not available_evidence:
        if effective_required:
            issue = DraftValidationIssue(
                code=BEHAVIOR_EVIDENCE_MISSING,
                severity="fail",
                message="Required physical behavior evidence is unavailable.",
                details={
                    "behavior_evidence_required": behavior_evidence_required,
                    "required_evidence_count": sum(
                        1 for item in resolution.evidence if item.required
                    ),
                    "evidence_count": len(resolution.evidence),
                },
            )
            return (
                "failed",
                (issue,),
                {"reason": "required_behavior_evidence_missing"},
            )
        return "skipped", (), {"reason": "no_behavior_evidence"}

    if not refine_summary_results:
        severity: IssueSeverity = "fail" if effective_required else "warn"
        issue = DraftValidationIssue(
            code=BEHAVIOR_JUDGE_UNAVAILABLE,
            severity=severity,
            message=(
                "Physical behavior evidence is present, but no supported "
                "Physics Agent judge/refine summary was supplied."
            ),
            details={
                "behavior_evidence_required": behavior_evidence_required,
                "required_evidence_count": sum(
                    1 for item in resolution.evidence if item.required
                ),
                "evidence_kinds": _physical_behavior_evidence_kinds(available_evidence),
            },
        )
        unavailable_status: TemplateStatus = "failed" if severity == "fail" else "warn"
        return unavailable_status, (issue,), {"reason": "behavior_judge_unavailable"}

    chosen = _choose_behavior_summary(refine_summary_results)
    status = _coerce_behavior_status(chosen.get("status"))
    issues = list(
        _issues_from_behavior_summary(
            chosen,
            behavior_evidence_required=effective_required,
        )
    )
    summaries = _behavior_summary_payloads(refine_summary_results)
    if any(
        int(summary.get("judge_llm_unavailable_count") or 0) > 0
        for summary in summaries
    ):
        issues.append(
            DraftValidationIssue(
                code=BEHAVIOR_JUDGE_UNAVAILABLE,
                severity="warn",
                message=(
                    "One or more Physics Agent behavior judge invocations "
                    "reported unavailable LLM evidence."
                ),
                details={
                    "judge_llm_unavailable_count": sum(
                        int(summary.get("judge_llm_unavailable_count") or 0)
                        for summary in summaries
                    )
                },
            )
        )
    if any(
        int(summary.get("refine_llm_unavailable_count") or 0) > 0
        for summary in summaries
    ):
        issues.append(
            DraftValidationIssue(
                code=BEHAVIOR_REFINER_UNAVAILABLE,
                severity="warn",
                message=(
                    "One or more Physics Agent scenario refiner invocations "
                    "reported unavailable LLM evidence."
                ),
                details={
                    "refine_llm_unavailable_count": sum(
                        int(summary.get("refine_llm_unavailable_count") or 0)
                        for summary in summaries
                    )
                },
            )
        )
    return status, tuple(issues), dict(chosen)


def _coerce_behavior_status(value: Any) -> TemplateStatus:
    if value in {"passed", "warn", "failed", "needs_refinement"}:
        return cast(TemplateStatus, value)
    return "warn"


def _choose_behavior_summary(
    refine_summary_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    statuses = {"failed": 0, "needs_refinement": 1, "warn": 2, "passed": 3}
    summaries = _behavior_summary_payloads(refine_summary_results)
    if not summaries:
        # Defensive fallback for callers such as metrics generation.
        return {"status": "warn", "reason": "summary_missing"}
    found_preferred_source = False
    for preferred in (
        lambda summary: summary.get("kind") == "refine_summary",
        lambda summary: str(summary.get("role") or "").startswith("final_"),
    ):
        preferred_summaries = [summary for summary in summaries if preferred(summary)]
        if preferred_summaries:
            summaries = preferred_summaries
            found_preferred_source = True
            break
    if not found_preferred_source:
        iteration_numbers = [
            iteration
            for summary in summaries
            if (iteration := _optional_int_value(summary.get("iteration"))) is not None
        ]
        if iteration_numbers:
            latest_iteration = max(iteration_numbers)
            summaries = [
                summary
                for summary in summaries
                if _optional_int_value(summary.get("iteration")) == latest_iteration
            ]
    # Within the selected source class, surface the most conservative outcome.
    return min(
        summaries, key=lambda summary: statuses.get(str(summary.get("status")), 2)
    )


def _behavior_summary_payloads(
    refine_summary_results: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        summary
        for result in refine_summary_results
        if isinstance((summary := result.get("summary")), Mapping)
    ]


def _issues_from_behavior_summary(
    summary: Mapping[str, Any],
    *,
    behavior_evidence_required: bool = False,
) -> tuple[DraftValidationIssue, ...]:
    status = summary.get("status")
    if status == "failed":
        return (
            DraftValidationIssue(
                code=BEHAVIOR_REFINE_LOOP_FAILED,
                severity="fail",
                message="Physics Agent behavior refinement did not complete.",
                subject=_optional_text(summary.get("path")),
                details=_summary_issue_details(summary),
            ),
        )
    judge_decision = str(summary.get("judge_decision") or "").strip().lower()
    if behavior_evidence_required and judge_decision in {"reject", "rejected"}:
        return (
            DraftValidationIssue(
                code=BEHAVIOR_REJECTED,
                severity="fail",
                message="Required physical behavior evidence was rejected by the judge.",
                subject=_optional_text(summary.get("path")),
                details=_summary_issue_details(summary),
            ),
        )
    if status == "needs_refinement":
        severity: IssueSeverity = "fail" if behavior_evidence_required else "warn"
        return (
            DraftValidationIssue(
                code=BEHAVIOR_NEEDS_REFINEMENT,
                severity=severity,
                message=(
                    "Physics Agent behavior judge requested another refinement "
                    "iteration or reached the iteration cap before approval."
                ),
                subject=_optional_text(summary.get("path")),
                details=_summary_issue_details(summary),
            ),
        )
    return ()


def _summary_issue_details(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key in (
            "termination_reason",
            "execution_status",
            "iteration",
            "final_iteration",
            "iteration_count",
            "judge_decision",
            "judge_score",
            "error",
            "cancelled",
        )
        if (value := summary.get(key)) is not None
    }


def _physical_behavior_status(
    behavior_status: TemplateStatus,
    issues: Sequence[DraftValidationIssue],
) -> TemplateStatus:
    if any(issue.severity == "fail" for issue in issues):
        return "failed"
    if behavior_status == "passed" and any(
        issue.severity == "warn" for issue in issues
    ):
        return "warn"
    return behavior_status


def _physical_behavior_metrics(
    context: DraftValidationContext,
    resolution: PhysicalBehaviorEvidenceResolution,
    *,
    refine_summary_results: Sequence[Mapping[str, Any]],
    issues: Sequence[DraftValidationIssue],
    status: TemplateStatus,
) -> dict[str, Any]:
    available = resolution.available_evidence
    chosen_summary = _choose_behavior_summary(refine_summary_results)
    metrics: dict[str, Any] = {
        "behavior_evidence_required": resolution.behavior_evidence_required,
        "status": status,
        "behavior_summary_kind": chosen_summary.get("kind"),
        "evidence_count": len(resolution.evidence),
        "available_evidence_count": len(available),
        "evidence_kinds": _physical_behavior_evidence_kinds(available),
        "usd_path_count": sum(
            1
            for item in available
            if item.kind in {"time_sampled_usd", "animation_usd"}
        ),
        "video_path_count": sum(1 for item in available if item.kind == "video"),
        "sampled_frame_path_count": sum(
            1 for item in available if item.kind == "sampled_frame"
        ),
        "simulation_json_count": sum(
            1 for item in available if item.kind == "simulation_json"
        ),
        "trajectory_metrics_count": sum(
            1 for item in available if item.kind == "trajectory_metrics"
        ),
        "refine_summary_count": len(refine_summary_results),
        "issue_count": len(issues),
        "input_video_path_count": len(context.input_inventory.video_paths),
        "input_usd_path_count": len(context.input_inventory.usd_paths),
    }
    for key in (
        "termination_reason",
        "execution_status",
        "role",
        "iteration",
        "final_iteration",
        "iteration_count",
        "judge_decision",
        "judge_score",
        "judge_llm_unavailable_count",
        "refine_llm_unavailable_count",
        "error",
        "cancelled",
    ):
        if (value := chosen_summary.get(key)) is not None:
            metrics[key] = value
    return metrics


def _physical_behavior_evidence_kinds(
    evidence: Sequence[PhysicalBehaviorEvidence],
) -> list[str]:
    return sorted({str(item.kind) for item in evidence if item.kind is not None})


def _issue_from_physical_behavior_issue(
    issue: PhysicalBehaviorIssue,
) -> DraftValidationIssue:
    return DraftValidationIssue(
        code=issue.code,
        severity=issue.severity,
        message=issue.message,
        subject=issue.subject,
        details=issue.details,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _look_right_current_image_paths(
    inventory: InputInventory,
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
) -> tuple[str | Path, ...]:
    _ = runtime_evidence
    if _canonical_usd_visual_evidence_enabled(policy):
        return ()
    return inventory.image_paths + (
        _optional_path_sequence(policy, "current_image_paths") or ()
    )


def _look_right_render_image_paths(
    inventory: InputInventory,
    policy: Mapping[str, Any],
    runtime_evidence: RuntimeVisualEvidence,
) -> tuple[str | Path, ...]:
    if _canonical_usd_visual_evidence_enabled(policy):
        return _canonical_runtime_render_image_paths(runtime_evidence)
    return (
        inventory.render_bundle_image_paths
        + (_optional_path_sequence(policy, "render_image_paths") or ())
        + runtime_evidence.image_paths
    )


def _look_right_vlm_available(
    policy: Mapping[str, Any],
    *,
    raw_judge_response: str | None,
    live_judge_config: Mapping[str, Any] | None,
) -> bool:
    explicit_value = _optional_policy_bool(policy, "look_right_vlm_available")
    if explicit_value is None:
        explicit_value = _optional_policy_bool(policy, "vlm_available")
    if explicit_value is not None:
        return explicit_value
    return raw_judge_response is not None or live_judge_config is not None


def _look_right_live_judge_config(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    config: dict[str, Any] = {}
    for key in ("look_right_vlm", "vlm"):
        nested_config = _mapping_value(policy, key)
        if nested_config:
            config.update(dict(nested_config))
            break

    _copy_policy_value(
        policy,
        config,
        target_key="backend",
        source_keys=("look_right_vlm_backend", "vlm_backend"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="model",
        source_keys=("look_right_vlm_model", "vlm_model"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="base_url",
        source_keys=("look_right_vlm_base_url", "vlm_base_url"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="api_key",
        source_keys=("look_right_vlm_api_key", "vlm_api_key"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="azure_endpoint",
        source_keys=("look_right_vlm_azure_endpoint", "vlm_azure_endpoint"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="endpoint",
        source_keys=("look_right_vlm_endpoint", "vlm_endpoint"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="api_name",
        source_keys=("look_right_vlm_api_name", "vlm_api_name"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="timeout",
        source_keys=("look_right_vlm_timeout", "vlm_timeout"),
    )

    if not config or config.get("enabled") is False:
        return None
    return config


def _look_right_final_judge_config(
    policy: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    config: dict[str, Any] = {}
    for key in ("look_right_llm_judge", "llm_judge"):
        nested_config = _mapping_value(policy, key)
        if nested_config:
            config.update(dict(nested_config))
            break

    _copy_policy_value(
        policy,
        config,
        target_key="backend",
        source_keys=("look_right_llm_judge_backend", "llm_judge_backend"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="model",
        source_keys=("look_right_llm_judge_model", "llm_judge_model"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="base_url",
        source_keys=("look_right_llm_judge_base_url", "llm_judge_base_url"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="api_key",
        source_keys=("look_right_llm_judge_api_key", "llm_judge_api_key"),
    )
    _copy_policy_value(
        policy,
        config,
        target_key="timeout",
        source_keys=("look_right_llm_judge_timeout", "llm_judge_timeout"),
    )

    if not config or config.get("enabled") is False:
        return None
    return config


def _copy_policy_value(
    policy: Mapping[str, Any],
    target: dict[str, Any],
    *,
    target_key: str,
    source_keys: Sequence[str],
) -> None:
    for source_key in source_keys:
        value = policy.get(source_key)
        if value is not None:
            target[target_key] = value
            return


def _invoke_live_look_right_judge(
    judge_plan: Any,
    live_judge_config: Mapping[str, Any] | None,
) -> tuple[LookRightJudgeInvocation | None, DraftValidationIssue | None]:
    if live_judge_config is None:
        return None, _live_look_right_unavailable_issue(
            "No look_right VLM executor config was supplied.",
            details={"configured": False},
        )

    try:
        vlm = _create_live_look_right_vlm(live_judge_config)
        invocation = invoke_look_right_judge(
            judge_plan,
            vlm,
            **_look_right_generation_kwargs(live_judge_config),
        )
    except Exception as exc:
        sanitized_config = _sanitize_model_config(live_judge_config)
        return None, _live_look_right_unavailable_issue(
            "The look_right VLM judge could not be invoked.",
            details={
                "configured": True,
                "model_config": sanitized_config,
                "error_type": type(exc).__name__,
                "error": _redact_sensitive_text(str(exc), live_judge_config),
            },
        )
    return invocation, None


class _UnavailableTextJudge:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, *_: Any, **__: Any) -> object:
        raise self.error


def _normalize_live_look_right_judgment(
    raw_judge_response: str,
    judge_plan: LookRightJudgePlan,
    final_judge_config: Mapping[str, Any] | None,
    *,
    pass_threshold: float,
    needs_refinement_threshold: float,
) -> LookRightFinalJudgeResult:
    if final_judge_config is None:
        return normalize_look_right_judgment(
            raw_judge_response,
            judge_plan=judge_plan,
            pass_threshold=pass_threshold,
            needs_refinement_threshold=needs_refinement_threshold,
        )

    try:
        llm_judge = _create_live_look_right_llm(final_judge_config)
    except Exception as exc:
        llm_judge = _UnavailableTextJudge(exc)

    return normalize_look_right_judgment(
        raw_judge_response,
        llm_judge=llm_judge,
        judge_plan=judge_plan,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
        temperature=_optional_float_value(final_judge_config.get("temperature")) or 0.0,
        max_tokens=_optional_int_value(final_judge_config.get("max_tokens")) or 512,
    )


def _create_live_look_right_llm(config: Mapping[str, Any]) -> Any:
    backend = _config_string(config, "backend") or _config_string(config, "provider")
    if backend is None:
        raise ValueError("look_right_llm_judge.backend is required")

    llm_kwargs = {
        str(key): value
        for key, value in config.items()
        if key
        not in {
            "backend",
            "provider",
            "enabled",
            "generation_kwargs",
            "max_tokens",
            "temperature",
        }
        and value is not None
    }
    api_key = _resolve_live_vlm_api_key(
        backend,
        base_url=llm_kwargs.get("base_url"),
        explicit_api_key=llm_kwargs.get("api_key"),
    )
    if api_key:
        llm_kwargs["api_key"] = api_key
    elif _backend_requires_api_key(backend):
        raise ValueError(
            f"API key required for look_right LLM judge backend {backend!r}; "
            "configure an endpoint-scoped key or use the documented environment "
            "variables for that backend."
        )
    else:
        llm_kwargs.pop("api_key", None)

    return create_chat_model(backend=backend, **llm_kwargs)


def _create_live_look_right_vlm(config: Mapping[str, Any]) -> Any:
    backend = _config_string(config, "backend") or _config_string(config, "provider")
    if backend is None:
        raise ValueError("look_right_vlm.backend is required for live judge execution")

    vlm_kwargs = {
        str(key): value
        for key, value in config.items()
        if key
        not in {
            "backend",
            "provider",
            "enabled",
            "generation_kwargs",
        }
        and value is not None
    }
    api_key = _resolve_live_vlm_api_key(
        backend,
        base_url=vlm_kwargs.get("base_url"),
        explicit_api_key=vlm_kwargs.get("api_key"),
    )
    if api_key:
        vlm_kwargs["api_key"] = api_key
    elif _backend_requires_api_key(backend):
        raise ValueError(
            f"API key required for look_right VLM backend {backend!r}; "
            "configure an endpoint-scoped key or use the documented environment "
            "variables for that backend."
        )
    else:
        vlm_kwargs.pop("api_key", None)

    return create_vlm(backend=backend, **vlm_kwargs)


def _resolve_live_vlm_api_key(
    backend: str,
    *,
    base_url: Any,
    explicit_api_key: Any,
) -> str | None:
    backend_name = backend.strip().lower()
    if backend_name == "nim":
        return get_nim_api_key_for_base_url(base_url, explicit_api_key)
    if backend_name == "openai":
        return get_openai_api_key_for_base_url(base_url, explicit_api_key)
    if backend_name in API_KEY_ENV_VAR_MAP:
        return get_env_api_key_for_backend(backend_name, explicit_api_key)
    if isinstance(explicit_api_key, str) and explicit_api_key.strip():
        return explicit_api_key.strip()
    return None


def _backend_requires_api_key(backend: str) -> bool:
    backend_name = backend.strip().lower()
    return backend_name in API_KEY_ENV_VAR_MAP


def _look_right_generation_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("generation_kwargs")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if key not in {"temperature", "max_tokens", "max_completion_tokens"}
    }


def _live_look_right_unavailable_issue(
    message: str,
    *,
    details: Mapping[str, Any],
) -> DraftValidationIssue:
    return DraftValidationIssue(
        code=VISUAL_JUDGE_UNAVAILABLE,
        severity="warn",
        message=message,
        details=details,
    )


def _config_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sanitize_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in config.items():
        key_text = str(key)
        if _is_sensitive_config_key(key_text):
            sanitized[key_text] = "<redacted>"
        else:
            sanitized[key_text] = _sanitize_model_config_value(value)
    return sanitized


def _sanitize_model_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_model_config(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_sanitize_model_config_value(item) for item in value]
    return _to_json_compatible(value)


def _is_sensitive_config_key(key: str) -> bool:
    key_lower = key.lower()
    return any(
        token in key_lower for token in ("api_key", "token", "secret", "password")
    )


def _redact_sensitive_text(text: str, config: Mapping[str, Any]) -> str:
    redacted = text
    for value in _sensitive_config_text_values(config):
        redacted = redacted.replace(value, "<redacted>")
    return redacted


def _sensitive_config_text_values(
    value: Any,
    *,
    sensitive_context: bool = False,
) -> list[str]:
    if isinstance(value, Mapping):
        values: list[str] = []
        for key, item in value.items():
            values.extend(
                _sensitive_config_text_values(
                    item,
                    sensitive_context=sensitive_context
                    or _is_sensitive_config_key(str(key)),
                )
            )
        return values
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        values = []
        for item in value:
            values.extend(
                _sensitive_config_text_values(
                    item,
                    sensitive_context=sensitive_context,
                )
            )
        return values
    if sensitive_context and isinstance(value, str) and value:
        return [value]
    return []


def _render_response_camera_names(render_response: Any) -> tuple[str, ...]:
    if not isinstance(render_response, Mapping):
        return ()
    raw_results = render_response.get("results")
    if not isinstance(raw_results, Sequence) or isinstance(
        raw_results,
        str | bytes | bytearray,
    ):
        return ()

    cameras: list[str] = []
    for entry in raw_results:
        if not isinstance(entry, Mapping):
            continue
        camera = (
            entry.get("camera") or entry.get("camera_path") or entry.get("camera_name")
        )
        if camera is not None:
            cameras.append(str(camera))
    return _dedupe_preserve_order(cameras)


def _optional_focused_image_paths(
    policy: Mapping[str, Any],
) -> Mapping[str, tuple[str | Path, ...]] | None:
    value = policy.get("focused_image_paths")
    if not isinstance(value, Mapping):
        return None
    focused_paths: dict[str, tuple[str | Path, ...]] = {}
    for prim_path, raw_paths in value.items():
        if not isinstance(prim_path, str):
            continue
        if isinstance(raw_paths, str | Path):
            focused_paths[prim_path] = (raw_paths,)
            continue
        if not isinstance(raw_paths, Sequence):
            continue
        paths = tuple(path for path in raw_paths if isinstance(path, str | Path))
        if paths:
            focused_paths[prim_path] = paths
    return focused_paths


def _focused_image_path_values(
    focused_image_paths: Mapping[str, Sequence[str | Path]] | None,
) -> tuple[str | Path, ...]:
    if focused_image_paths is None:
        return ()
    return tuple(
        path
        for prim_path in sorted(focused_image_paths)
        for path in focused_image_paths[prim_path]
    )


def _previous_adapter_result(
    template_results: Sequence[DraftTemplateResult],
    template_name: str,
) -> Mapping[str, Any] | None:
    for result in reversed(template_results):
        if result.template_name != template_name:
            continue
        adapter_result = result.metadata.get("adapter_result")
        if isinstance(adapter_result, Mapping):
            return adapter_result
        adapter_results = result.metadata.get("adapter_results")
        if isinstance(adapter_results, Sequence) and not isinstance(
            adapter_results,
            str,
        ):
            first_result = next(
                (item for item in adapter_results if isinstance(item, Mapping)),
                None,
            )
            if first_result is not None:
                return first_result
    return None


def _draft_result_from_adapter_result(
    *,
    template_name: str,
    adapter_result: Mapping[str, Any],
    status: TemplateStatus,
    metadata: Mapping[str, Any],
) -> DraftTemplateResult:
    issues = tuple(
        _issue_from_adapter_issue(issue)
        for issue in _mapping_sequence(adapter_result, "issues")
    )
    issues += _runtime_render_issues_from_metadata(metadata)
    metrics = dict(_mapping_value(adapter_result, "metrics"))
    metrics["issue_count"] = len(issues)
    final_status = _status_with_runtime_render_metadata(status, metadata)
    metadata_adapter_result = dict(adapter_result)
    if final_status != status:
        metadata_adapter_result["status"] = _adapter_status_from_template_status(
            final_status,
        )
        metadata_adapter_result["issues"] = [issue.to_dict() for issue in issues]
    return DraftTemplateResult(
        template_name=template_name,
        status=final_status,
        issues=issues,
        metrics=metrics,
        evidence=dict(_mapping_value(adapter_result, "evidence")),
        metadata={
            **dict(metadata),
            "adapter_result": metadata_adapter_result,
        },
    )


def _runtime_render_issues_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[DraftValidationIssue, ...]:
    runtime_render = metadata.get("runtime_render")
    if not isinstance(runtime_render, Mapping):
        return ()
    return tuple(
        _issue_from_adapter_issue(issue)
        for issue in _mapping_sequence(runtime_render, "issues")
    )


def _status_with_runtime_render_metadata(
    status: TemplateStatus,
    metadata: Mapping[str, Any],
) -> TemplateStatus:
    runtime_render = metadata.get("runtime_render")
    if not isinstance(runtime_render, Mapping):
        return status
    runtime_status = runtime_render.get("status")
    if runtime_status == "failed":
        return "failed"
    return status


def _adapter_status_from_template_status(status: TemplateStatus) -> str:
    if status == "passed":
        return "pass"
    if status == "failed":
        return "fail"
    return status


def _draft_result_from_adapter_results(
    *,
    template_name: str,
    adapter_results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> DraftTemplateResult:
    issues = tuple(
        _issue_from_adapter_issue(issue)
        for adapter_result in adapter_results
        for issue in _mapping_sequence(adapter_result, "issues")
    )
    status = _aggregate_adapter_statuses(adapter_results)
    metrics: dict[str, Any] = {
        "usd_path_count": len(adapter_results),
        "issue_count": len(issues),
        "passed_count": sum(
            1 for adapter_result in adapter_results if adapter_result.get("passed")
        ),
        "failed_count": sum(
            1
            for adapter_result in adapter_results
            if adapter_result.get("verdict") == "fail"
            or adapter_result.get("status") == "error"
        ),
        "warn_count": sum(
            1
            for adapter_result in adapter_results
            if adapter_result.get("verdict") == "warn"
        ),
    }
    if len(adapter_results) == 1:
        adapter_metrics = dict(_mapping_value(adapter_results[0], "metrics"))
        metrics["adapter_metrics"] = adapter_metrics
        for key, value in adapter_metrics.items():
            metrics.setdefault(key, value)

    usd_paths = [
        str(usd_path)
        for adapter_result in adapter_results
        if (usd_path := _mapping_value(adapter_result, "evidence").get("usd_path"))
    ]
    return DraftTemplateResult(
        template_name=template_name,
        status=status,
        issues=issues,
        metrics=metrics,
        evidence={"usd_paths": usd_paths},
        metadata={
            **dict(metadata),
            "adapter_results": [
                dict(adapter_result) for adapter_result in adapter_results
            ],
        },
    )


def _issue_from_adapter_issue(issue: Mapping[str, Any]) -> DraftValidationIssue:
    details = dict(_mapping_value(issue, "details"))
    for key, value in issue.items():
        if key not in {"code", "severity", "message", "subject", "details"}:
            details[key] = value
    return DraftValidationIssue(
        code=str(issue.get("code", "agent.template_issue")),
        severity=_issue_severity(issue.get("severity")),
        message=str(issue.get("message", "Validation template issue.")),
        subject=_optional_issue_subject(issue),
        details=details,
    )


def _issue_from_look_right_issue(issue: LookRightIssue) -> DraftValidationIssue:
    severity = _look_right_issue_severity(issue)
    return DraftValidationIssue(
        code=issue.code,
        severity=severity,
        message=issue.message,
        subject=issue.subject,
        details=dict(issue.details),
    )


def _look_right_issue_severity(issue: LookRightIssue) -> IssueSeverity:
    if issue.severity in {"warn", "warning"}:
        return "warn"
    if issue.severity == "info":
        return "info"
    if issue.code == VISUAL_EVIDENCE_MISSING:
        return "warn"
    return "fail"


def _look_right_unready_status(
    issues: Sequence[DraftValidationIssue],
) -> TemplateStatus:
    if any(issue.code == VISUAL_RENDER_PREFLIGHT_FAILED for issue in issues):
        return "failed"
    if any(
        issue.severity == "fail" and issue.code != VISUAL_EVIDENCE_MISSING
        for issue in issues
    ):
        return "failed"
    return "skipped"


def _issues_from_look_right_judgment(
    judgment: Mapping[str, Any],
) -> tuple[DraftValidationIssue, ...]:
    issue_codes = judgment.get("issue_codes")
    if not isinstance(issue_codes, Sequence) or isinstance(issue_codes, str):
        return ()

    verdict = judgment.get("verdict")
    severity: IssueSeverity
    if verdict == "fail":
        severity = "fail"
    elif verdict in {"warn", "needs_refinement"}:
        severity = "warn"
    else:
        severity = "info"

    return tuple(
        DraftValidationIssue(
            code=code,
            severity=severity,
            message=f"look_right visual judge reported {code}.",
            details={"judgment_verdict": verdict},
        )
        for code in issue_codes
        if isinstance(code, str) and code
    )


def _look_right_judgment_status(verdict: Any) -> TemplateStatus:
    if verdict == "pass":
        return "passed"
    if verdict == "fail":
        return "failed"
    if verdict == "needs_refinement":
        return "needs_refinement"
    if verdict == "warn":
        return "warn"
    return "error"


def _optional_issue_subject(issue: Mapping[str, Any]) -> str | None:
    subject = issue.get("subject")
    if subject is None:
        subject = issue.get("prim_path")
    return str(subject) if subject is not None else None


def _issue_severity(value: Any) -> IssueSeverity:
    if value == "info":
        return "info"
    if value in {"warn", "warning"}:
        return "warn"
    return "fail"


def _render_adapter_status(adapter_result: Mapping[str, Any]) -> TemplateStatus:
    status = adapter_result.get("status")
    if status == "pass":
        return "passed"
    if status == "skipped":
        return "skipped"
    if status == "fail":
        return "failed"
    return "error"


def _aggregate_adapter_statuses(
    adapter_results: Sequence[Mapping[str, Any]],
) -> TemplateStatus:
    if any(
        adapter_result.get("status") == "error" for adapter_result in adapter_results
    ):
        return "error"
    if any(
        adapter_result.get("verdict") == "fail" for adapter_result in adapter_results
    ):
        return "failed"
    if any(
        adapter_result.get("verdict") == "warn" for adapter_result in adapter_results
    ):
        return "warn"
    if all(
        adapter_result.get("status") == "skipped" for adapter_result in adapter_results
    ):
        return "skipped"
    return "passed"


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(
    mapping: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _optional_value(policy: Mapping[str, Any], key: str) -> Any | None:
    return policy.get(key)


def _optional_string(policy: Mapping[str, Any], key: str) -> str | None:
    value = policy.get(key)
    return value if isinstance(value, str) else None


def _optional_path(policy: Mapping[str, Any], key: str) -> str | Path | None:
    value = policy.get(key)
    return value if isinstance(value, str | Path) else None


def _optional_path_sequence(
    policy: Mapping[str, Any],
    key: str,
) -> tuple[str | Path, ...] | None:
    value = policy.get(key)
    if value is None:
        return None
    if isinstance(value, str | Path):
        return (value,)
    if not isinstance(value, Sequence):
        return None
    return tuple(item for item in value if isinstance(item, str | Path))


def _optional_string_sequence(
    policy: Mapping[str, Any],
    key: str,
) -> tuple[str, ...] | None:
    value = policy.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return None
    return tuple(item for item in value if isinstance(item, str))


def _optional_sequence(
    policy: Mapping[str, Any],
    key: str,
) -> tuple[Any, ...] | None:
    value = policy.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return None
    return tuple(value)


def _optional_float(
    policy: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = policy.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _optional_int(
    policy: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    value = policy.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _asset_validator_report(
    policy: Mapping[str, Any],
    usd_path: str,
) -> Mapping[str, Any] | None:
    report = policy.get("asset_validator_report")
    if not isinstance(report, Mapping):
        return None
    per_usd = report.get(usd_path)
    if isinstance(per_usd, Mapping):
        return per_usd
    return report


def _dedupe_preserve_order(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return tuple(deduped)


def _optional_policy_bool(policy: Mapping[str, Any], key: str) -> bool | None:
    value = policy.get(key)
    if isinstance(value, bool):
        return value
    return None


def _policy_bool(policy: Mapping[str, Any], key: str) -> bool:
    value = policy.get(key)
    return value if isinstance(value, bool) else False


def _strict_policy_bool(
    policy: Mapping[str, Any],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = policy.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"policy.{key} must be a bool, got {value!r}")


def _to_pretty_json(data: Mapping[str, Any]) -> str:
    return json.dumps(_to_json_compatible(data), indent=2, sort_keys=True) + "\n"


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _to_json_compatible(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_to_json_compatible(item) for item in value]
    return str(value)
