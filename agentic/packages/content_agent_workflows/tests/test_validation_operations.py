# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from world_understanding.validation import (
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
)

import content_agent_workflows.validation.embedded_assessment as validation_assessment
import content_agent_workflows.validation.operations as operations_module
import content_agent_workflows.validation.workflow as workflow_module
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.validation import (
    PROVIDED_OPERATION_ARTIFACT_NAMES,
    STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME,
    VALIDATION_OPERATION_INDEX_NAME,
    VALIDATION_TERMINAL_RECEIPT_NAME,
    EmbeddedValidationAssessmentError,
    ValidationAssessmentFinding,
    ValidationCoordinatorAssessment,
    ValidationCoordinatorReviewDraft,
    ValidationGateAssessment,
    ValidationOperationIndex,
    ValidationWorkflowError,
    assess_standalone_validation,
    finalize_validation_operations,
    load_validation_operation_preparation,
    prepare_standalone_validation_evidence,
    prepare_validation_operations,
    review_standalone_validation_assessment,
    run_validation_operation,
)


def test_execute_and_provided_modes_share_one_artifact_inventory() -> None:
    assert PROVIDED_OPERATION_ARTIFACT_NAMES == (
        "verified_operation_ingest_index.json",
        "verified_operation_evidence_index.json",
        "verified_operation_execution_index.json",
        "canonical_verified_operation_assessment.json",
        "verified_operations",
    )


class _FocusedExecutor:
    def __init__(self, *, skipped_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.skipped_on = skipped_on
        self.template_versions = {
            "render_valid": "test.render-valid.v1",
            "look_right": "test.look-right.v1",
            "physics_sane": "test.physics-sane.v1",
            "physical_behavior": "test.physical-behavior.v1",
        }

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(template_name=name, reason="Outer selected.")
                for name in request.requested_templates
            ),
            reasoning_summary="Validated an explicit outer-selected request.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        self.calls.append(template_name)
        if template_name == self.skipped_on:
            return ValidationTemplateResult(
                template_name=template_name,
                status="skipped",
            )
        if template_name == "look_right":
            raise AssertionError("optional external judge must not run implicitly")
        if template_name == "render_valid":
            image_path = context.working_dir / "render.png"
            Image.new("RGB", (8, 8), (20, 80, 140)).save(image_path, format="PNG")
            return ValidationTemplateResult(
                template_name=template_name,
                status="passed",
                metadata={
                    "runtime_render": {
                        "status": "available",
                        "image_paths": [str(image_path)],
                    }
                },
            )
        return ValidationTemplateResult(template_name=template_name, status="passed")


class _ReorderedExecutor(_FocusedExecutor):
    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        plan = super().plan(request, working_dir=working_dir)
        return plan.model_copy(update={"steps": tuple(reversed(plan.steps))})


class _QualifiedRenderExecutor(_FocusedExecutor):
    def __init__(self, image_path: Path, *, negative_physics: bool = False) -> None:
        super().__init__()
        self.image_path = image_path
        self.negative_physics = negative_physics

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        self.calls.append(template_name)
        if template_name == "physics_sane" and self.negative_physics:
            return ValidationTemplateResult(
                template_name=template_name,
                status="failed",
                issues=(
                    ValidationIssue(
                        code="physics.expected_negative",
                        severity="fail",
                        message="Frozen known-negative physics finding.",
                        template_name=template_name,
                    ),
                ),
            )
        if template_name == "render_valid":
            return ValidationTemplateResult(
                template_name=template_name,
                status="passed",
                evidence={"image_paths": [str(self.image_path)]},
            )
        return ValidationTemplateResult(template_name=template_name, status="passed")


def _request(source: Path, *templates: str) -> ValidationRequest:
    return ValidationRequest(
        task_description="Validate only the explicitly requested capabilities.",
        inputs=(str(source),),
        requested_templates=templates,
    )


def _qualified_render_policy(source: Path, image: Path) -> dict[str, object]:
    return {
        "visual_evidence_mode": "provided_evidence",
        "current_image_paths": [str(image)],
        "runtime_render_usd": False,
        "qualified_render_evidence": [
            {
                "path": str(image),
                "role": "qualified_render",
                "sha256": file_sha256(image),
                "source_sha256": file_sha256(source),
                "qualification": "Exact source-bound OVRTX evidence.",
                "ovrtx_render_metadata": {
                    "backend": "remote",
                    "renderer": "ovrtx",
                    "scene_tool": "usd-cli",
                    "scene_tool_source_revision": "test-revision",
                    "session_id": "test-session",
                    "workflow": "validation-canonical-visual-evidence",
                    "views": ["+x+y+z"],
                    "image_width": 8,
                    "image_height": 8,
                    "renderer_identities": [
                        {
                            "engine": "ovrtx",
                            "protocol_version": 4,
                            "status": "ready",
                        }
                    ],
                },
            }
        ],
    }


def test_prepare_is_provider_free_and_records_every_unselected_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")

    def unexpected_provider(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider construction is forbidden during preparation")

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        unexpected_provider,
    )
    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_chat_model",
        unexpected_provider,
    )
    preparation = prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
    )

    assert preparation.nested_agent_launched is False
    assert preparation.expansion.kind == "requested_templates"
    assert [item.template_name for item in preparation.capabilities] == [
        "physics_sane",
        "look_right",
        "render_valid",
        "physical_behavior",
    ]
    assert [item.selection_state for item in preparation.capabilities] == [
        "requested",
        "not_requested",
        "not_requested",
        "not_requested",
    ]
    assert {
        step.template_name: step.required_capabilities
        for step in preparation.plan.steps
    } == {"physics_sane": ("usd_schema_runtime",)}


def test_coordinator_projection_rechecks_required_capability_mandatory_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(
        ValidationWorkflowError,
        match="did not preserve required capabilities.*validation.physics_sane",
    ):
        operations_module._prepare_validation_operations_from_coordinator(
            _request(source, "physics_sane"),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            check_ids=("schema",),
            dependencies={"schema": ()},
            mandatory={"physics_sane": False},
            check_metadata={"schema": {"capability_id": "validation.physics_sane"}},
            required_capability_ids=("validation.physics_sane",),
            executor=_FocusedExecutor(),
        )

    assert not (output_dir / "validation_operation_preparation.json").exists()


def test_coordinator_projection_rejects_reordered_executor_plan(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(
        ValidationWorkflowError,
        match="did not preserve the required ordered work items",
    ):
        operations_module._prepare_validation_operations_from_coordinator(
            _request(source, "physics_sane", "physical_behavior"),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            check_ids=("schema", "behavior"),
            dependencies={"schema": (), "behavior": ("schema",)},
            mandatory={"physics_sane": True, "physical_behavior": True},
            check_metadata={
                "schema": {"capability_id": "validation.physics_sane"},
                "behavior": {"capability_id": "validation.physical_behavior"},
            },
            required_capability_ids=(),
            executor=_ReorderedExecutor(),
        )

    assert not (output_dir / "validation_operation_preparation.json").exists()


@pytest.mark.parametrize(
    "parameters",
    (
        {"backend": "remote"},
        {"views": ["front"]},
        {"image_width": 64, "image_height": 64},
    ),
)
def test_coordinator_projection_rejects_render_policy_parameters(
    tmp_path: Path, parameters: dict[str, object]
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(
        ValidationWorkflowError,
        match="cannot author render policy.*render",
    ):
        operations_module._prepare_validation_operations_from_coordinator(
            _request(source, "render_valid"),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            check_ids=("render",),
            dependencies={"render": ()},
            mandatory={"render_valid": True},
            check_metadata={"render": {"parameters": parameters}},
            required_capability_ids=(),
            executor=_FocusedExecutor(),
        )

    assert not (output_dir / "validation_operation_preparation.json").exists()


def test_prepare_fails_closed_without_a_template_capability_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    monkeypatch.setattr(workflow_module, "VALIDATION_TEMPLATE_CAPABILITIES", {})

    with pytest.raises(
        ValidationWorkflowError,
        match="physics_sane",
    ):
        prepare_validation_operations(
            _request(source, "physics_sane"),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=_FocusedExecutor(),
        )


def test_prepare_rejects_existing_provided_operation_mode(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "verified_operation_ingest_index.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationWorkflowError, match="mutually exclusive"):
        prepare_validation_operations(
            _request(source, "physics_sane"),
            output_dir=run_dir,
            config_base_dir=tmp_path,
            executor=_FocusedExecutor(),
        )


def test_execute_mode_rejects_late_provided_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    run_dir = tmp_path / "run"
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=run_dir,
        config_base_dir=tmp_path,
        executor=_FocusedExecutor(),
    )
    (run_dir / "verified_operations").mkdir()

    with pytest.raises(ValidationWorkflowError, match="mutually exclusive"):
        load_validation_operation_preparation(run_dir)


@pytest.mark.parametrize(
    ("requested_rules", "named_profile", "expected_kind", "expected_templates"),
    (
        (
            ("physics.usd_schema_sanity",),
            None,
            "requested_rules",
            ("physics_sane",),
        ),
        ((), "visual", "named_profile", ("render_valid",)),
    ),
)
def test_prepare_deterministically_expands_explicit_rule_or_profile(
    tmp_path: Path,
    requested_rules: tuple[str, ...],
    named_profile: str | None,
    expected_kind: str,
    expected_templates: tuple[str, ...],
) -> None:
    source = tmp_path / f"{expected_kind}.txt"
    source.write_text("asset\n", encoding="utf-8")
    executor = _FocusedExecutor()

    preparation = prepare_validation_operations(
        _request(source),
        output_dir=tmp_path / f"run-{expected_kind}",
        config_base_dir=tmp_path,
        requested_rules=requested_rules,
        named_profile=named_profile,
        executor=executor,
    )

    assert preparation.expansion.kind == expected_kind
    assert preparation.request.requested_templates == expected_templates
    assert executor.calls == []


def test_prepare_does_not_implicitly_add_render_for_advisory_critique(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")

    with pytest.raises(ValidationWorkflowError, match="requires render_valid"):
        prepare_validation_operations(
            _request(source, "look_right"),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=_FocusedExecutor(),
        )


def test_operation_rejects_undeclared_prior_result_without_burning_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "physics_sane", "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    physics = run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    physics_path = (
        output_dir / "operations" / physics.template_name / "operation_result.json"
    )

    with pytest.raises(ValidationWorkflowError, match="exact prior results"):
        run_validation_operation(
            output_dir,
            template_name="render_valid",
            prior_result_paths=(physics_path,),
            executor=executor,
        )

    assert not (output_dir / "operations" / "render_valid").exists()


def test_advisory_failed_render_blocks_dependent_visual_critique(
    tmp_path: Path,
) -> None:
    class _FailedRenderExecutor(_FocusedExecutor):
        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            if template_name == "render_valid":
                self.calls.append(template_name)
                return ValidationTemplateResult(
                    template_name=template_name,
                    status="failed",
                )
            return super().run(template_name, context)

    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    executor = _FailedRenderExecutor()
    operations_module._prepare_validation_operations_from_coordinator(
        _request(source, "render_valid", "look_right"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("render", "critique"),
        dependencies={"render": (), "critique": ("render",)},
        mandatory={"render_valid": False, "look_right": False},
        check_metadata={"render": {}, "critique": {}},
        required_capability_ids=(),
        executor=executor,
    )
    render = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    critique = run_validation_operation(
        output_dir,
        template_name="look_right",
        prior_result_paths=(
            output_dir / "operations" / "render_valid" / "operation_result.json",
        ),
        executor=executor,
    )

    assert render.template_result.status == "failed"
    assert critique.template_result.status == "skipped"
    assert critique.template_result.issues[0].code == (
        "validation.operation_dependency_not_satisfied"
    )
    assert executor.calls == ["render_valid"]


def test_advisory_render_keeps_outer_visual_evidence_advisory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    executor = _FocusedExecutor()
    operations_module._prepare_validation_operations_from_coordinator(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("render",),
        dependencies={"render": ()},
        mandatory={"render_valid": False},
        check_metadata={"render": {}},
        required_capability_ids=(),
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)

    evidence = prepare_standalone_validation_evidence(output_dir)
    visual = next(
        record
        for record in evidence.records
        if record.evidence_id == "validation-outer-visual-evidence"
    )

    assert visual.status == "available"
    assert visual.required is False


def test_advisory_critique_issue_codes_remain_evidence_bound(
    tmp_path: Path,
) -> None:
    class _CritiqueExecutor(_FocusedExecutor):
        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            if template_name == "look_right":
                self.calls.append(template_name)
                return ValidationTemplateResult(
                    template_name=template_name,
                    status="needs_refinement",
                    issues=(
                        ValidationIssue(
                            code="visual.low_confidence",
                            severity="warn",
                            message="The live judge needs stronger visual evidence.",
                            template_name=template_name,
                        ),
                    ),
                    metrics={"vlm_invoked": True},
                )
            return super().run(template_name, context)

    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _CritiqueExecutor()
    prepare_validation_operations(
        _request(source, "render_valid", "look_right"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="look_right",
        prior_result_paths=(
            output_dir / "operations" / "render_valid" / "operation_result.json",
        ),
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    evidence = prepare_standalone_validation_evidence(output_dir)
    critique = next(
        record
        for record in evidence.records
        if record.evidence_id == "validation-template-look_right"
    )
    assert critique.facts["issue_codes"] == ("visual.low_confidence",)

    assessment = ValidationCoordinatorAssessment(
        assessment_id="mismatched-critique-code",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-render_valid",),
                disposition="pass",
                rationale="The exact render operation passed.",
            ),
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=(
                    "validation-outer-visual-evidence",
                    "validation-template-look_right",
                ),
                disposition="pass",
                rationale="The bound render and advisory critique were reviewed.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The immutable package readback passed.",
            ),
        ),
        findings=(
            ValidationAssessmentFinding(
                finding_id="critique-finding",
                source_evidence_ids=("validation-template-look_right",),
                source_issue_codes=(
                    "visual.low_confidence",
                    "visual.prompt_mismatch",
                ),
                severity="warning",
                summary="The advisory critique requires attention.",
                disposition="accepted",
                rationale="The live result is explicitly dispositioned.",
            ),
        ),
        terminal_disposition="pass",
        summary="Required gates pass and the advisory result is dispositioned.",
    )
    assessment_path = tmp_path / "mismatched-assessment.json"
    assessment_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="issue codes absent from its source evidence.*visual.prompt_mismatch",
    ):
        assess_standalone_validation(output_dir, assessment_path=assessment_path)


def test_refinement_directory_keeps_dedicated_behavior_evidence_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    refine_dir = tmp_path / "refine"
    iteration_dir = refine_dir / "iter_1"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "history.jsonl").write_text(
        '{"trial_index":0}\n', encoding="utf-8"
    )
    (iteration_dir / "judge_result.json").write_text(
        '{"decision":"approve","score":0.9}\n', encoding="utf-8"
    )
    (refine_dir / "refine_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(refine_dir),
                "termination_reason": "approved",
                "final_iteration": 1,
                "final_dir": str(refine_dir / "final"),
                "user_prompt": "validate behavior",
                "iterations": [
                    {
                        "iteration": 1,
                        "iteration_dir": str(iteration_dir),
                        "scenario_yaml_path": str(iteration_dir / "scenario.yaml"),
                        "tune_output_dir": str(iteration_dir),
                        "best_params": {},
                        "best_score": 0.9,
                        "n_trials": 1,
                        "judge_decision": "approve",
                        "judge_score": 0.9,
                        "judge_reasoning": "approved",
                        "judge_llm_unavailable": False,
                        "refine_llm_unavailable": False,
                        "refine_reasoning": "approved",
                        "metric_name": "settle_distance",
                        "metric_value": 0.1,
                        "cancelled": False,
                        "error": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    request = _request(source, "physical_behavior").model_copy(
        update={
            "policy": {
                "physical_behavior_refine_output_dir": str(refine_dir),
            }
        }
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("behavior",),
        dependencies={"behavior": ()},
        mandatory={"physical_behavior": True},
        check_metadata={
            "behavior": {"parameters": {"evidence_paths": [str(refine_dir.resolve())]}}
        },
        required_capability_ids=(),
    )

    operation = run_validation_operation(
        output_dir,
        template_name="physical_behavior",
    )

    assert operation.template_result.status == "passed"
    assert operation.template_result.metrics["refine_summary_count"] >= 1
    assert "physics.behavior_evidence_unsupported" not in {
        issue.code for issue in operation.template_result.issues
    }


def test_coordinator_projection_preserves_typed_behavior_evidence_record(
    tmp_path: Path,
) -> None:
    class _BehaviorEvidenceExecutor(_FocusedExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.behavior_evidence: object = None

        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            if template_name == "physical_behavior":
                self.behavior_evidence = context.request.policy.get(
                    "physical_behavior_evidence"
                )
            return super().run(template_name, context)

    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    trajectory = tmp_path / "trajectory.dat"
    trajectory.write_text('{"settle_distance": 0.1}\n', encoding="utf-8")
    typed_record = {
        "path": str(trajectory),
        "kind": "trajectory_metrics",
        "role": "runtime_trajectory",
        "description": "Frozen runtime trajectory metrics.",
        "required": True,
    }
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    executor = _BehaviorEvidenceExecutor()
    request = _request(source, "physical_behavior").model_copy(
        update={"policy": {"physical_behavior_evidence": typed_record}}
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("behavior",),
        dependencies={"behavior": ()},
        mandatory={"physical_behavior": True},
        check_metadata={
            "behavior": {"parameters": {"evidence_paths": [str(trajectory.resolve())]}}
        },
        required_capability_ids=(),
        executor=executor,
    )

    run_validation_operation(
        output_dir,
        template_name="physical_behavior",
        executor=executor,
    )

    assert executor.behavior_evidence == [typed_record]


def test_relative_behavior_evidence_uses_config_base_dir_not_operation_dir(
    tmp_path: Path,
) -> None:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    source = inputs_dir / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    runtime_evidence = inputs_dir / "runtime-evidence.json"
    runtime_evidence.write_text(
        json.dumps(
            {
                "status": "completed",
                "decision": "approve",
                "score": 0.91,
                "reasoning": "The rollout remained stable.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    request = _request(source, "physical_behavior").model_copy(
        update={
            "policy": {
                "physical_behavior_evidence": {
                    "path": runtime_evidence.name,
                    "kind": "simulation_json",
                    "role": "simulation_json",
                },
                "behavior_evidence_required": True,
            }
        }
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=inputs_dir,
        check_ids=("behavior",),
        dependencies={"behavior": ()},
        mandatory={"physical_behavior": True},
        check_metadata={
            "behavior": {
                "parameters": {
                    "evidence_paths": [str(runtime_evidence.resolve())],
                }
            }
        },
        required_capability_ids=(),
    )

    operation = run_validation_operation(output_dir, template_name="physical_behavior")

    assert operation.template_result.status == "passed"
    assert "validation.accepted_evidence_missing" not in {
        issue.code for issue in operation.template_result.issues
    }
    resolved_records = operation.template_result.evidence["available_evidence"]
    runtime_record = next(
        record for record in resolved_records if record["role"] == "simulation_json"
    )
    assert runtime_record["original"] == runtime_evidence.name
    assert runtime_record["path"] == str(runtime_evidence.resolve())
    runtime_identity = next(
        artifact
        for artifact in operation.evidence_artifacts
        if artifact.path == str(runtime_evidence.resolve())
    )
    assert operation.source_before == operation.source_after
    source_identity = next(
        artifact
        for artifact in operation.source_after
        if artifact.path == str(source.resolve())
    )
    assert source_identity.sha256 == file_sha256(source)
    assert runtime_identity.sha256 == file_sha256(runtime_evidence)


def test_coordinator_projection_cannot_downgrade_required_behavior_evidence(
    tmp_path: Path,
) -> None:
    class _BehaviorEvidenceExecutor(_FocusedExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.behavior_evidence_required: object = None

        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            if template_name == "physical_behavior":
                self.behavior_evidence_required = context.request.policy.get(
                    "behavior_evidence_required"
                )
            return super().run(template_name, context)

    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text('{"settle_distance": 0.1}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    executor = _BehaviorEvidenceExecutor()
    request = _request(source, "physical_behavior").model_copy(
        update={
            "policy": {
                "physical_behavior_evidence": {"path": str(trajectory)},
                "behavior_evidence_required": True,
            }
        }
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("behavior",),
        dependencies={"behavior": ()},
        mandatory={"physical_behavior": False},
        check_metadata={
            "behavior": {"parameters": {"evidence_paths": [str(trajectory.resolve())]}}
        },
        required_capability_ids=(),
        executor=executor,
    )

    run_validation_operation(
        output_dir,
        template_name="physical_behavior",
        executor=executor,
    )

    assert executor.behavior_evidence_required is True


def test_coordinator_projection_limits_behavior_to_selected_policy_evidence(
    tmp_path: Path,
) -> None:
    class _BehaviorPolicyExecutor(_FocusedExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.policy: dict[str, object] = {}

        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            if template_name == "physical_behavior":
                self.policy = dict(context.request.policy)
            return super().run(template_name, context)

    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    selected = tmp_path / "selected.dat"
    selected.write_text('{"settle_distance": 0.1}\n', encoding="utf-8")
    unselected = tmp_path / "unselected.json"
    unselected.write_text('{"settle_distance": 9.9}\n', encoding="utf-8")
    unselected_refine_dir = tmp_path / "unselected-refine"
    unselected_refine_dir.mkdir()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    executor = _BehaviorPolicyExecutor()
    request = _request(source, "physical_behavior").model_copy(
        update={
            "policy": {
                "trajectory_metrics_paths": [str(selected), str(unselected)],
                "physical_behavior_refine_output_dir": str(unselected_refine_dir),
            }
        }
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("behavior",),
        dependencies={"behavior": ()},
        mandatory={"physical_behavior": True},
        check_metadata={
            "behavior": {"parameters": {"evidence_paths": [str(selected.resolve())]}}
        },
        required_capability_ids=(),
        executor=executor,
    )

    run_validation_operation(
        output_dir,
        template_name="physical_behavior",
        executor=executor,
    )

    assert executor.policy["physical_behavior_evidence"] == [
        {
            "path": str(selected),
            "kind": "trajectory_metrics",
            "role": "trajectory_metrics",
        }
    ]
    assert "trajectory_metrics_paths" not in executor.policy
    assert "physical_behavior_refine_output_dir" not in executor.policy


@pytest.mark.parametrize("digest_matches", (True, False))
def test_focused_render_accepts_only_explicitly_qualified_precomputed_evidence(
    tmp_path: Path,
    digest_matches: bool,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    source_digest = file_sha256(source)
    image_digest = file_sha256(image) if digest_matches else "0" * 64
    request = _request(source, "render_valid").model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "provided_evidence",
                "current_image_paths": [str(image)],
                "runtime_render_usd": False,
                "qualified_render_evidence": [
                    {
                        "path": str(image),
                        "role": "qualified_render",
                        "sha256": image_digest,
                        "source_sha256": source_digest,
                        "qualification": "Exact source-bound OVRTX evidence.",
                        "ovrtx_render_metadata": {
                            "backend": "remote",
                            "renderer": "ovrtx",
                            "scene_tool": "usd-cli",
                            "scene_tool_source_revision": "test-revision",
                            "session_id": "test-session",
                            "workflow": "validation-canonical-visual-evidence",
                            "views": ["+x+y+z"],
                            "image_width": 8,
                            "image_height": 8,
                            "renderer_identities": [
                                {
                                    "engine": "ovrtx",
                                    "protocol_version": 4,
                                    "status": "ready",
                                }
                            ],
                        },
                    }
                ],
            }
        }
    )
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    if digest_matches:
        assert operation.template_result.status == "passed"
        assert operation.template_result.metadata["render_evidence_origin"] == (
            "qualified_precomputed"
        )
        runtime_render = operation.template_result.metadata["runtime_render"]
        assert runtime_render["status"] == "completed"
        assert runtime_render["backend"] == "remote"
        assert runtime_render["image_paths"] == [str(image.resolve())]
        assert runtime_render["metadata"]["render_evidence_origin"] == (
            "qualified_precomputed"
        )
    else:
        assert operation.template_result.status == "failed"
        assert {issue.code for issue in operation.template_result.issues} == {
            "validation.render_evidence_missing"
        }


def test_qualified_render_cannot_alias_declared_reference_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    image = tmp_path / "qualified-and-reference.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    policy = _qualified_render_policy(source, image)
    policy["reference_image_paths"] = [str(image)]
    request = _request(source, "render_valid", "look_right").model_copy(
        update={"policy": policy}
    )
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    assert operation.template_result.status == "failed"
    assert {issue.code for issue in operation.template_result.issues} == {
        "validation.render_evidence_input_collision"
    }
    assert operation.template_result.metadata["render_evidence_accepted"] is False


def test_qualified_render_relative_path_uses_config_base_and_source_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    request = _request(source, "render_valid").model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "provided_evidence",
                "current_image_paths": [str(image)],
                "runtime_render_usd": False,
                "qualified_render_evidence": [
                    {
                        "path": image.name,
                        "role": "qualified_render",
                        "sha256": file_sha256(image),
                        "source_sha256": file_sha256(source),
                        "qualification": "Exact source-bound OVRTX evidence.",
                        "ovrtx_render_metadata": {
                            "backend": "remote",
                            "renderer": "ovrtx",
                            "scene_tool": "usd-cli",
                            "scene_tool_source_revision": "test-revision",
                            "session_id": "test-session",
                            "workflow": "validation-canonical-visual-evidence",
                            "views": ["+x+y+z"],
                            "image_width": 8,
                            "image_height": 8,
                            "renderer_identities": [
                                {
                                    "engine": "ovrtx",
                                    "protocol_version": 4,
                                    "status": "ready",
                                }
                            ],
                        },
                    }
                ],
            }
        }
    )
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    assert operation.template_result.status == "passed"


def test_qualified_render_rejects_usd_dependency_digest_as_source(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(
        '#usda 1.0\ndef Xform "Dependency" {}\n',
        encoding="utf-8",
    )
    source = tmp_path / "asset.usda"
    source.write_text(
        '#usda 1.0\n(\n    subLayers = [@dependency.usda@]\n)\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    policy = _qualified_render_policy(source, image)
    qualified = policy["qualified_render_evidence"]
    assert isinstance(qualified, list)
    assert isinstance(qualified[0], dict)
    qualified[0]["source_sha256"] = file_sha256(dependency)
    request = _request(source, "render_valid").model_copy(update={"policy": policy})
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)

    identity = workflow_module._workflow_identity(  # noqa: SLF001
        request,
        config_base_dir=tmp_path,
        executor=executor,
    )
    assert file_sha256(dependency) in {
        artifact.sha256 for artifact in identity.source_artifacts
    }
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    assert operation.template_result.status == "failed"
    assert {issue.code for issue in operation.template_result.issues} == {
        "validation.render_evidence_missing"
    }


def test_usd_dependency_closure_prefers_resolver_path_over_root_relative_decoy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import UsdUtils

    root = tmp_path / "root.usda"
    root.write_text('#usda 1.0\ndef Xform "Root" {}\n', encoding="utf-8")
    decoy = tmp_path / "texture.png"
    decoy.write_bytes(b"root-relative-decoy")
    resolved = tmp_path / "nested" / "texture.png"
    resolved.parent.mkdir()
    resolved.write_bytes(b"resolver-selected-texture")
    monkeypatch.setattr(
        UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            (),
            (
                SimpleNamespace(
                    path="texture.png",
                    resolvedPath=str(resolved.resolve()),
                ),
            ),
            (),
        ),
    )

    dependencies, runtime_modules = workflow_module._usd_dependency_closure(root)

    assert resolved.resolve() in dependencies
    assert decoy.resolve() not in dependencies
    assert runtime_modules == ()


def test_qualified_render_rejects_incomplete_multi_source_coverage(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.usda"
    first_source.write_text(
        '#usda 1.0\ndef Xform "First" {}\n',
        encoding="utf-8",
    )
    second_source = tmp_path / "second.usda"
    second_source.write_text(
        '#usda 1.0\ndef Xform "Second" {}\n',
        encoding="utf-8",
    )
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    request = _request(first_source, "render_valid").model_copy(
        update={
            "inputs": (str(first_source), str(second_source)),
            "policy": _qualified_render_policy(first_source, image),
        }
    )
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    assert operation.template_result.status == "failed"
    assert {issue.code for issue in operation.template_result.issues} == {
        "validation.render_evidence_missing"
    }


def test_qualified_render_accepts_exact_directory_source_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset"
    source.mkdir()
    source.joinpath("asset.usda").write_text(
        '#usda 1.0\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (40, 100, 160)).save(image, format="PNG")
    source_identity = workflow_module._artifact_identity(  # noqa: SLF001
        source,
        role="source",
        base_dir=tmp_path,
    )
    assert source_identity.sha256 is not None
    request = _request(source, "render_valid").model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "provided_evidence",
                "current_image_paths": [str(image)],
                "runtime_render_usd": False,
                "qualified_render_evidence": [
                    {
                        "path": str(image),
                        "role": "qualified_render",
                        "sha256": file_sha256(image),
                        "source_sha256": source_identity.sha256,
                        "qualification": "Exact directory-source-bound OVRTX evidence.",
                        "ovrtx_render_metadata": {
                            "backend": "remote",
                            "renderer": "ovrtx",
                            "scene_tool": "usd-cli",
                            "scene_tool_source_revision": "test-revision",
                            "session_id": "test-session",
                            "workflow": "validation-canonical-visual-evidence",
                            "views": ["+x+y+z"],
                            "image_width": 8,
                            "image_height": 8,
                            "renderer_identities": [
                                {
                                    "engine": "ovrtx",
                                    "protocol_version": 4,
                                    "status": "ready",
                                }
                            ],
                        },
                    }
                ],
            }
        }
    )
    output_dir = tmp_path / "run"
    executor = _QualifiedRenderExecutor(image)
    prepare_validation_operations(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )

    assert operation.template_result.status == "passed"
    assert operation.template_result.metadata["render_evidence_origin"] == (
        "qualified_precomputed"
    )
    assert operation.source_before[0].kind == "directory"
    assert operation.source_before[0].sha256 == source_identity.sha256


def test_advisory_negative_operation_can_be_reviewed_without_becoming_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    image = tmp_path / "render.png"
    Image.new("RGB", (8, 8), (20, 80, 140)).save(image, format="PNG")
    executor = _QualifiedRenderExecutor(image, negative_physics=True)
    request = _request(source, "physics_sane", "render_valid").model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "provided_evidence",
                "current_image_paths": [str(image)],
                "qualified_render_evidence": [
                    {
                        "path": str(image),
                        "role": "qualified_render",
                        "sha256": file_sha256(image),
                        "source_sha256": file_sha256(source),
                        "qualification": "Exact source-bound benchmark render.",
                        "ovrtx_render_metadata": {
                            "backend": "remote",
                            "renderer": "ovrtx",
                            "scene_tool": "usd-cli",
                            "scene_tool_source_revision": "test-revision",
                            "session_id": "test-session",
                            "workflow": "validation-canonical-visual-evidence",
                            "views": ["+x+y+z"],
                            "image_width": 8,
                            "image_height": 8,
                            "renderer_identities": [
                                {
                                    "engine": "ovrtx",
                                    "protocol_version": 4,
                                    "status": "ready",
                                }
                            ],
                        },
                    }
                ],
            }
        }
    )
    operations_module._prepare_validation_operations_from_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        check_ids=("known-negative", "render"),
        dependencies={"known-negative": (), "render": ("known-negative",)},
        mandatory={"physics_sane": False, "render_valid": True},
        check_metadata={"known-negative": {}, "render": {}},
        required_capability_ids=(),
        executor=executor,
    )
    physics = run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        prior_result_paths=(
            output_dir / "operations" / "physics_sane" / "operation_result.json",
        ),
        executor=executor,
    )
    run = finalize_validation_operations(output_dir)
    assert physics.template_result.status == "failed"
    assert run.result.verdict == "fail"
    assert (
        prepare_standalone_validation_evidence(output_dir).records[0].required is False
    )
    assessment = ValidationCoordinatorAssessment(
        assessment_id="accepted-known-negative",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=(
                    "validation-template-physics_sane",
                    "validation-template-render_valid",
                ),
                disposition="pass",
                rationale="The required render passed; the frozen negative was observed.",
            ),
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=("validation-outer-visual-evidence",),
                disposition="pass",
                rationale="The outer reasoner inspected the exact bound image.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The immutable package readback passed.",
            ),
        ),
        findings=(
            ValidationAssessmentFinding(
                finding_id="expected-physics-negative",
                source_evidence_ids=("validation-template-physics_sane",),
                source_issue_codes=("physics.expected_negative",),
                severity="warning",
                summary="The manifest-frozen negative finding was reproduced.",
                disposition="accepted",
                rationale="This advisory finding is the expected benchmark outcome.",
            ),
        ),
        terminal_disposition="pass",
        summary="Required gates pass and the advisory negative is truthful.",
    )
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assess_standalone_validation(output_dir, assessment_path=assessment_path)
    review = ValidationCoordinatorReviewDraft(
        created_at=datetime.now(UTC),
        disposition="accept",
        findings=("The known-negative finding and required gates match evidence.",),
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(review.model_dump_json(indent=2) + "\n", encoding="utf-8")

    receipt = review_standalone_validation_assessment(
        output_dir,
        review_path=review_path,
    )

    assert receipt.receipt_status == "completed"
    assert receipt.source_mutated is False


def test_finalize_fails_closed_when_a_mandatory_selected_operation_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=_FocusedExecutor(),
    )

    with pytest.raises(ValidationWorkflowError, match="physics_sane"):
        finalize_validation_operations(tmp_path / "run")

    index = ValidationOperationIndex.model_validate_json(
        (tmp_path / "run" / VALIDATION_OPERATION_INDEX_NAME).read_text(encoding="utf-8")
    )
    physics = next(
        item for item in index.operations if item.template_name == "physics_sane"
    )
    assert physics.state == "not_evaluated"
    assert physics.mandatory is True
    assert index.required_operations_complete is False


def test_optional_critique_can_remain_not_evaluated_and_outer_visual_can_pass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid", "look_right"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    render = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    run = finalize_validation_operations(output_dir)

    assert executor.calls == ["render_valid"]
    assert run.result.schema_version == "1.0"
    assert [item.template_name for item in run.result.template_results] == [
        "render_valid",
        "look_right",
    ]
    assert run.result.template_results[1].status == "skipped"
    assert run.result.template_results[1].metadata["selection_state"] == (
        "not_evaluated"
    )
    index = prepare_standalone_validation_evidence(output_dir)
    visual_records = [
        record
        for record in index.records
        if record.evidence_id == "validation-outer-visual-evidence"
    ]
    assert len(visual_records) == 1
    assert not visual_records[0].facts.get("verdict")
    assessment = ValidationCoordinatorAssessment(
        assessment_id="outer-visual-without-advisory-judge",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-render_valid",),
                disposition="pass",
                rationale="The render preparation fact is an explicit pass.",
            ),
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=("validation-outer-visual-evidence",),
                disposition="pass",
                rationale="The outer multimodal reasoner inspected the bound image.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The saved artifact readback is intact.",
            ),
        ),
        terminal_disposition="pass",
        summary="Required standalone gates pass without advisory judge output.",
    )
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assess_standalone_validation(output_dir, assessment_path=assessment_path)
    review = ValidationCoordinatorReviewDraft(
        created_at=datetime.now(UTC),
        disposition="accept",
        findings=("Exact saved assessment matches the outer-authored decision.",),
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = review_standalone_validation_assessment(
        output_dir,
        review_path=review_path,
    )

    assert render.template_name == "render_valid"
    assert receipt.mode == "standalone"
    assert receipt.gate_dispositions == {
        "static_validation": "pass",
        "runtime_validation": "not_evaluated",
        "visual_quality": "pass",
        "package_integrity": "pass",
        "cross_stage_integrity": "not_evaluated",
    }
    assert receipt.receipt_status == "completed"
    persisted = json.loads(
        (output_dir / VALIDATION_TERMINAL_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert persisted["nested_agent_launched"] is False


def test_required_visual_gate_cannot_pass_without_current_render_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    index = prepare_standalone_validation_evidence(output_dir)
    visual = next(
        record
        for record in index.records
        if record.evidence_id == "validation-outer-visual-evidence"
    )
    assert visual.required is False
    assert visual.status == "unsupported"
    assessment = ValidationCoordinatorAssessment(
        assessment_id="invalid-visual-pass",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-physics_sane",),
                disposition="pass",
                rationale="The static evidence passed.",
            ),
            ValidationGateAssessment(
                gate="visual_quality",
                evidence_ids=("validation-outer-visual-evidence",),
                disposition="pass",
                rationale="This must not pass without current render evidence.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The package readback passed.",
            ),
        ),
        terminal_disposition="pass",
        summary="This assessment is intentionally invalid.",
    )
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="without available required current-render evidence",
    ):
        assess_standalone_validation(output_dir, assessment_path=assessment_path)


def test_missing_selected_runtime_capability_is_an_explicit_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor(skipped_on="physical_behavior")
    prepare_validation_operations(
        _request(source, "physical_behavior"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    operation = run_validation_operation(
        output_dir,
        template_name="physical_behavior",
        executor=executor,
    )
    run = finalize_validation_operations(output_dir)

    assert operation.template_result.status == "failed"
    assert {issue.code for issue in operation.template_result.issues} == {
        "validation.required_capability_not_evaluated"
    }
    assert run.result.verdict == "fail"


def test_unavailable_optional_critique_remains_not_evaluated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor(skipped_on="look_right")
    prepare_validation_operations(
        _request(source, "render_valid", "look_right"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    render = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="look_right",
        prior_result_paths=(
            output_dir / "operations" / render.template_name / "operation_result.json",
        ),
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    index = ValidationOperationIndex.model_validate_json(
        (output_dir / VALIDATION_OPERATION_INDEX_NAME).read_text(encoding="utf-8")
    )
    critique = next(
        item for item in index.operations if item.template_name == "look_right"
    )

    assert critique.state == "not_evaluated"
    assert critique.result_path is None
    assert index.optional_critique_evaluated is False


def test_standalone_evidence_rejects_stale_render_bytes(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    image_path = Path(operation.evidence_artifacts[0].path)
    image_path.write_bytes(b"stale")

    with pytest.raises(EmbeddedValidationAssessmentError, match="stale"):
        prepare_standalone_validation_evidence(output_dir)


def test_standalone_evidence_rejects_operation_state_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    index_path = output_dir / VALIDATION_OPERATION_INDEX_NAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    render_status = next(
        item
        for item in payload["operations"]
        if item["template_name"] == "render_valid"
    )
    render_status.update(
        {
            "state": "not_requested",
            "mandatory": False,
            "result_path": None,
            "result_sha256": None,
        }
    )
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(EmbeddedValidationAssessmentError, match="state differs"):
        prepare_standalone_validation_evidence(output_dir)


def test_operation_index_rejects_required_state_downgrade(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    index_path = output_dir / VALIDATION_OPERATION_INDEX_NAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    physics = next(
        item
        for item in payload["operations"]
        if item["template_name"] == "physics_sane"
    )
    physics["mandatory"] = False
    payload["required_operations_complete"] = True
    index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValidationWorkflowError, match="mandatory state differs"):
        operations_module.load_validated_validation_operation_index(output_dir)


def test_standalone_evidence_rejects_mismatched_typed_template_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    operation_path = (
        output_dir / "operations" / "render_valid" / "operation_result.json"
    )
    operation_payload = json.loads(operation_path.read_text(encoding="utf-8"))
    operation_payload["template_result"]["status"] = "failed"
    operation_path.write_text(
        json.dumps(operation_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    index_path = output_dir / VALIDATION_OPERATION_INDEX_NAME
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    render_status = next(
        item
        for item in index_payload["operations"]
        if item["template_name"] == "render_valid"
    )
    render_status["result_sha256"] = file_sha256(operation_path)
    index_path.write_text(
        json.dumps(index_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EmbeddedValidationAssessmentError, match="bytes differ"):
        prepare_standalone_validation_evidence(output_dir)


def test_finalize_reuses_operation_timestamps_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    operation = run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )
    first = finalize_validation_operations(output_dir)
    artifact_names = (
        "validation_checkpoint.json",
        "validation_result.json",
        "validation_evidence.json",
        "final_summary.json",
        VALIDATION_OPERATION_INDEX_NAME,
    )
    first_bytes = {name: (output_dir / name).read_bytes() for name in artifact_names}

    second = finalize_validation_operations(output_dir)

    assert {name: (output_dir / name).read_bytes() for name in artifact_names} == (
        first_bytes
    )
    assert first.checkpoint == second.checkpoint
    assert first.checkpoint.records[0].started_at == operation.completed_at
    assert first.checkpoint.records[0].finished_at == operation.completed_at
    assert first.checkpoint.created_at == operation.completed_at
    assert first.checkpoint.updated_at == operation.completed_at


def test_standalone_evidence_write_once_rejects_symlink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    evidence_index = output_dir / STANDALONE_VALIDATION_EVIDENCE_INDEX_NAME
    evidence_index.symlink_to(outside)

    with pytest.raises(
        EmbeddedValidationAssessmentError, match="must not be a symlink"
    ):
        prepare_standalone_validation_evidence(output_dir)

    assert outside.read_text(encoding="utf-8") == "{}\n"


def test_deleted_operation_result_fails_through_typed_boundary(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    operation = run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    finalize_validation_operations(output_dir)
    Path(operation.template_result_path).unlink()

    with pytest.raises(EmbeddedValidationAssessmentError, match="binding is stale"):
        prepare_standalone_validation_evidence(output_dir)


def test_unknown_operation_template_fails_through_typed_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _FocusedExecutor()
    prepare_validation_operations(
        _request(source, "render_valid"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    run_validation_operation(
        output_dir,
        template_name="render_valid",
        executor=executor,
    )
    operation_path = (
        output_dir / "operations" / "render_valid" / "operation_result.json"
    )
    payload = json.loads(operation_path.read_text(encoding="utf-8"))
    payload["template_name"] = "unknown_template"
    payload["template_result"]["template_name"] = "unknown_template"
    operation_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValidationWorkflowError, match="unknown Validation template"):
        operations_module._load_operation_result(operation_path)


def test_source_drift_fails_through_typed_boundary(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    class _SourceMutatingExecutor(_FocusedExecutor):
        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            result = super().run(template_name, context)
            source.write_text("changed\n", encoding="utf-8")
            return result

    executor = _SourceMutatingExecutor()
    prepare_validation_operations(
        _request(source, "physics_sane"),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    with pytest.raises(ValidationWorkflowError, match="source identity changed"):
        run_validation_operation(
            output_dir,
            template_name="physics_sane",
            executor=executor,
        )


def test_legacy_embedded_receipt_index_fails_closed_without_migration(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "embedded_validation_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.embedded-validation-receipt-index.v1"
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EmbeddedValidationAssessmentError,
        match="intentionally invalidated",
    ):
        validation_assessment._load_embedded_validation_receipt_index(receipt_path)
