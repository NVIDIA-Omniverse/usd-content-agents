# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic Validation coordinator contract and adversarial tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image
from world_understanding.validation import (
    ValidationFocusConfig,
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationProject,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
)

from content_agent_workflows.common import canonical_json_digest
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.validation import (
    VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME,
    VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
    VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
    VALIDATION_COORDINATOR_PREPARATION_NAME,
    EmbeddedValidationAssessmentError,
    ValidationAssessmentFinding,
    ValidationCoordinatorAssessment,
    ValidationCoordinatorPlanPatch,
    ValidationCoordinatorReviewDraft,
    ValidationGateAssessment,
    ValidationWorkflowError,
    assess_standalone_validation,
    execute_validation_coordinator_plan,
    load_validation_coordinator_preparation,
    prepare_standalone_validation_evidence,
    prepare_validation_coordinator,
    review_standalone_validation_assessment,
)
from content_agent_workflows.validation import (
    accept_validation_coordinator_plan as _accept_validation_coordinator_plan,
)
from content_agent_workflows.validation.coordinator import _validate_plan_patch


class _CoordinatorExecutor:
    def __init__(
        self,
        *,
        skipped_on: str | None = None,
        nonpassing_on: str | None = None,
        nonpassing_status: Literal["warn", "needs_refinement"] = "warn",
    ) -> None:
        self.plan_calls = 0
        self.run_calls: list[tuple[str, tuple[str, ...]]] = []
        self.behavior_evidence_paths: tuple[str, ...] = ()
        self.skipped_on = skipped_on
        self.nonpassing_on = nonpassing_on
        self.nonpassing_status = nonpassing_status

    @property
    def template_versions(self) -> Mapping[str, str]:
        return {
            "render_valid": "fake.render.v1",
            "look_right": "fake.look.v1",
            "physics_sane": "fake.physics.v1",
            "physical_behavior": "fake.behavior.v1",
        }

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        self.plan_calls += 1
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=name,
                    reason="Exact coordinator-selected adapter projection.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Compatibility projection of the accepted child plan.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        self.run_calls.append(
            (
                template_name,
                tuple(
                    result.template_name for result in context.previous_template_results
                ),
            )
        )
        if template_name == "physical_behavior":
            evidence = context.request.policy.get("physical_behavior_evidence", ())
            if isinstance(evidence, list | tuple):
                self.behavior_evidence_paths = tuple(
                    str(item.get("path")) if isinstance(item, Mapping) else str(item)
                    for item in evidence
                )
        status: Literal["passed", "skipped", "warn", "needs_refinement"]
        if template_name == self.skipped_on:
            status = "skipped"
        elif template_name == self.nonpassing_on:
            status = self.nonpassing_status
        else:
            status = "passed"
        return ValidationTemplateResult(
            template_name=template_name,
            status=status,
            metadata={"executor": "coordinator-test"},
        )


class _PassedWithFailIssueExecutor(_CoordinatorExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "physical_behavior":
            return result
        return result.model_copy(
            update={
                "issues": (
                    ValidationIssue(
                        code="physics.behavior_failed",
                        severity="fail",
                        message="Behavior evidence contains a blocking issue.",
                        template_name=template_name,
                    ),
                )
            }
        )


def _with_runtime_render_evidence(
    result: ValidationTemplateResult,
    context: ValidationTemplateContext,
) -> ValidationTemplateResult:
    context.working_dir.mkdir(parents=True, exist_ok=True)
    image_path = (context.working_dir / "render.png").resolve()
    Image.new("RGB", (8, 8), color=(24, 48, 96)).save(image_path)
    metadata = dict(result.metadata)
    metadata["runtime_render"] = {
        "status": "passed",
        "backend": "fake",
        "image_paths": [str(image_path)],
        "render_response": None,
        "render_output_dir": str(context.working_dir),
        "issues": [],
        "metadata": {},
    }
    return result.model_copy(
        update={
            "evidence": {"image_paths": [str(image_path)]},
            "metadata": metadata,
        }
    )


class _DependencyUnavailablePolicyExecutor(_CoordinatorExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "render_valid":
            return _with_runtime_render_evidence(result, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "status": "skipped",
                "issues": (
                    ValidationIssue(
                        code="visual.judge_unavailable",
                        severity="warn",
                        message="The optional visual judge is unavailable.",
                        template_name=template_name,
                    ),
                ),
            }
        )


class _ExpectedNegativePolicyExecutor(_CoordinatorExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "render_valid":
            return _with_runtime_render_evidence(result, context)
        if template_name != "physics_sane":
            return result
        return result.model_copy(
            update={
                "status": "failed",
                "issues": (
                    ValidationIssue(
                        code="physics.no_physics_scene",
                        severity="fail",
                        message="The fixture has no physics scene.",
                        template_name=template_name,
                    ),
                    ValidationIssue(
                        code="physics.no_rigid_bodies",
                        severity="fail",
                        message="The fixture has no rigid bodies.",
                        template_name=template_name,
                    ),
                ),
            }
        )


def _request(source: Path, output_dir: Path) -> ValidationRequest:
    return ValidationRequest(
        task_description="Validate schema sanity and existing behavior evidence.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
        focus=ValidationFocusConfig(prim_paths=("/Asset",)),
        policy={"visual_evidence_mode": "canonical_usd"},
    )


def _patch_payload(preparation_digest: str, source: Path) -> dict[str, object]:
    return {
        "schema_version": "content-agent-workflows.validation-coordinator-plan-patch.v1",
        "plan_id": "provider-free-proof",
        "preparation_digest": preparation_digest,
        "producer": "codex",
        "child_session_id": "child-proof-1",
        "claims": [
            {
                "claim_id": "schema-and-runtime",
                "statement": "The asset is schema-valid and its behavior evidence is coherent.",
                "acceptance_criteria": [
                    "USD schema checks are evaluated.",
                    "Existing behavior evidence is evaluated after schema checks.",
                ],
                "check_ids": ["schema", "behavior"],
            }
        ],
        "selected_checks": [
            {
                "check_id": "schema",
                "capability_id": "validation.physics_sane",
                "template_name": "physics_sane",
                "rule_id": "physics.usd_schema_sanity",
                "targets": [str(source)],
                "focus_prim_paths": ["/Asset"],
                "parameters": {},
                "required": True,
                "depends_on": [],
                "evidence_requirements": ["source_identity", "usd_schema_result"],
                "completion_policy": "all_required_evidence",
            },
            {
                "check_id": "behavior",
                "capability_id": "validation.physical_behavior",
                "template_name": "physical_behavior",
                "rule_id": "physics.behavior_evidence",
                "targets": [str(source)],
                "focus_prim_paths": ["/Asset"],
                "parameters": {"evidence_paths": [str(source.parent / "runtime.json")]},
                "required": True,
                "depends_on": ["schema"],
                "evidence_requirements": [
                    "source_identity",
                    "runtime_evidence",
                    "behavior_result",
                ],
                "completion_policy": "all_required_evidence",
            },
        ],
        "decision_only": True,
        "source_mutated": False,
        "execution_performed": False,
        "publication_performed": False,
    }


def _prepare_patch(
    tmp_path: Path,
    *,
    executor: _CoordinatorExecutor,
    required_capability_ids: tuple[str, ...] = (),
    policy: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    runtime_evidence = tmp_path / "runtime.json"
    runtime_evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    request_policy: dict[str, Any] = {
        "visual_evidence_mode": "canonical_usd",
        "physical_behavior_evidence": {"path": str(runtime_evidence)},
    }
    request_policy.update(policy or {})
    request = _request(source, output_dir).model_copy(
        update={
            "policy": request_policy,
            "metadata": {
                "validation_required_capability_ids": list(required_capability_ids)
            },
        }
    )
    preparation = prepare_validation_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    payload = _patch_payload(preparation.preparation_digest, source.resolve())
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    raw = output_dir / "raw"
    raw.mkdir()
    child_output = raw / "codex_final.json"
    child_output.write_text('{"status":"plan_authored"}\n', encoding="utf-8")
    child_stream = raw / "validation_planner_output.jsonl"
    child_stream.write_text('{"type":"turn_finished"}\n', encoding="utf-8")
    preparation_path = output_dir / VALIDATION_COORDINATOR_PREPARATION_NAME
    preparation_identity = {
        "path": str(preparation_path.resolve()),
        "sha256": file_sha256(preparation_path),
        "size_bytes": preparation_path.stat().st_size,
    }
    launch_descriptor = {
        "schema_version": "content-agents.child-launch.v1",
        "profile_key": "validation.plan",
        "workflow": "validation.plan",
        "workflow_skill": "content-workflow-validation",
        "scene_backend": "none",
        "required_staged_skills": ["content-workflow-validation"],
        "skill_staging_mode": "required",
        "capability_inventory": preparation_identity,
        "domain_policy_bounds": preparation_identity,
        "network_policy": {
            "schema_version": "content-agents.child-launch-network-policy.v1",
            "mode": "reasoning_transport_only",
            "tool_network_access": False,
            "allowed_hosts": [],
        },
        "credential_policy": {
            "mode": "reasoning_transport_only",
            "forbidden_environment_names": [
                "NVCF_API_KEY",
                "VALIDATION_JUDGE_KEY",
            ],
        },
        "artifacts": {
            "schema_version": "content-agents.child-launch-artifacts.v1",
            "run_root": str(output_dir.resolve()),
            "child_output_path": str(child_stream.resolve()),
            "child_final_path": str(child_output.resolve()),
            "bridge_artifact_prefix": "validation_planner",
        },
        "runner_identity": {
            "runner": "codex",
            "model": "test-model",
            "model_reasoning_effort": None,
            "claude_execution_mode": None,
        },
    }
    (raw / "validation_planner_launch_descriptor.json").write_text(
        json.dumps(launch_descriptor, indent=2) + "\n",
        encoding="utf-8",
    )
    return source, output_dir, payload


def _policy_patch_checks(source: Path) -> list[dict[str, object]]:
    target = str(source.resolve())
    return [
        {
            "check_id": "render",
            "capability_id": "validation.render_valid",
            "template_name": "render_valid",
            "rule_id": "render.runtime_evidence",
            "targets": [target],
            "focus_prim_paths": ["/Asset"],
            "parameters": {},
            "required": True,
            "depends_on": [],
            "evidence_requirements": [
                "source_identity",
                "render_result",
                "qualified_render_evidence",
            ],
            "completion_policy": "all_required_evidence",
        },
        {
            "check_id": "physics",
            "capability_id": "validation.physics_sane",
            "template_name": "physics_sane",
            "rule_id": "physics.usd_schema_sanity",
            "targets": [target],
            "focus_prim_paths": ["/Asset"],
            "parameters": {},
            "required": False,
            "depends_on": ["render"],
            "evidence_requirements": ["source_identity", "usd_schema_result"],
            "completion_policy": "best_effort_advisory",
        },
    ]


def _write_policy_patch(
    output_dir: Path,
    payload: dict[str, object],
    *,
    checks: list[dict[str, object]],
) -> None:
    payload["claims"] = [
        {
            "claim_id": "policy-finalization",
            "statement": "The native facts and policy-derived verdict are coherent.",
            "acceptance_criteria": [
                "Raw adapter evidence remains available.",
                "The published verdict follows the frozen request policy.",
            ],
            "check_ids": [str(check["check_id"]) for check in checks],
        }
    ]
    payload["selected_checks"] = checks
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def accept_validation_coordinator_plan(
    output_dir: Path,
    *,
    plan_patch_path: str | Path,
    child_output_path: str | Path,
    producer: Literal["codex", "claude"],
    child_session_id: str | None = None,
    child_plan_id: str | None = None,
    executor: _CoordinatorExecutor | None = None,
) -> Any:
    descriptor_path = output_dir / "raw" / "validation_planner_launch_descriptor.json"
    try:
        descriptor_payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        descriptor_digest = "0" * 64
    else:
        descriptor_digest = canonical_json_digest(descriptor_payload)
    return _accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=plan_patch_path,
        child_output_path=child_output_path,
        producer=producer,
        expected_child_launch_descriptor_digest=descriptor_digest,
        child_session_id=child_session_id,
        child_plan_id=child_plan_id,
        executor=executor,
    )


def test_preparation_selects_nothing_and_invokes_no_adapter(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _CoordinatorExecutor()

    preparation = prepare_validation_coordinator(
        _request(source, output_dir),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert preparation.request.requested_templates == ()
    assert preparation.selected_checks == ()
    assert preparation.provider_invoked is False
    assert preparation.operation_invoked is False
    assert preparation.mandatory_constraints.minimum_provider_free_checks == 2
    assert preparation.mandatory_constraints.minimum_required_checks == 1
    assert preparation.mandatory_constraints.explicit_dependency_required is True
    assert set(preparation.mandatory_constraints.provider_free_capability_ids) == {
        "validation.render_valid",
        "validation.physics_sane",
        "validation.physical_behavior",
    }
    assert executor.plan_calls == 0
    assert executor.run_calls == []
    assert not (output_dir / "validation_plan.json").exists()
    assert not (output_dir / "validation_request.json").exists()


def test_preparation_binds_bare_omnipbr_as_public_runtime_dependency(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        """#usda 1.0

def Shader "Material"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
    uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
}
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    executor = _CoordinatorExecutor()

    preparation = prepare_validation_coordinator(
        _request(source, output_dir),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    external = tuple(
        artifact
        for artifact in preparation.source_dependency_closure
        if artifact.kind == "external"
    )
    assert len(external) == 1
    assert external[0].path == "mdl://runtime/OmniPBR.mdl"
    assert external[0].sha256 is not None
    assert executor.plan_calls == 0
    assert executor.run_calls == []
    assert (
        load_validation_coordinator_preparation(
            output_dir,
            executor=executor,
        )
        == preparation
    )


@pytest.mark.parametrize("authored_path", ("MissingCustom.mdl", "./OmniPBR.mdl"))
def test_preparation_rejects_unresolved_noncanonical_mdl_dependency(
    tmp_path: Path,
    authored_path: str,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        f"""#usda 1.0

def Shader "Material"
{{
    uniform asset info:mdl:sourceAsset = @{authored_path}@
}}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationWorkflowError, match="dependency closure is unresolved"
    ):
        prepare_validation_coordinator(
            _request(source, tmp_path / "run"),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=_CoordinatorExecutor(),
        )

    assert not (tmp_path / "run").exists()


def test_preparation_rejects_bare_omnipbr_as_a_missing_composition_layer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@OmniPBR.mdl@]\n)\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationWorkflowError, match="dependency closure is unresolved"
    ):
        prepare_validation_coordinator(
            _request(source, tmp_path / "run"),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=_CoordinatorExecutor(),
        )

    assert not (tmp_path / "run").exists()


def test_preparation_rejects_output_inside_source_before_creating_it(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_dir.joinpath("asset.txt").write_text("asset\n", encoding="utf-8")
    output_dir = source_dir / "validation-run"
    request = _request(source_dir, output_dir)

    with pytest.raises(ValidationWorkflowError, match="cannot be inside"):
        prepare_validation_coordinator(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=_CoordinatorExecutor(),
        )

    assert not output_dir.exists()


def test_preparation_freezes_each_declared_evidence_inventory_class(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    evidence_paths = {
        kind: tmp_path / f"{kind}.json"
        for kind in ("runtime", "render", "package", "upstream")
    }
    for kind, path in evidence_paths.items():
        path.write_text(json.dumps({"kind": kind}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = _request(source, output_dir).model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "canonical_usd",
                "physical_behavior_evidence": {"path": str(evidence_paths["runtime"])},
                "render_image_paths": [str(evidence_paths["render"])],
                "package_evidence_paths": [str(evidence_paths["package"])],
                "upstream_evidence_paths": [str(evidence_paths["upstream"])],
            }
        }
    )
    executor = _CoordinatorExecutor()

    preparation = prepare_validation_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert {item.evidence_kind for item in preparation.evidence_inventory} == {
        "runtime",
        "render",
        "package",
        "upstream",
    }
    evidence_paths["runtime"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValidationWorkflowError, match="evidence inventory changed"):
        load_validation_coordinator_preparation(output_dir, executor=executor)


def test_preparation_preserves_one_path_declared_for_multiple_evidence_roles(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    shared_evidence = tmp_path / "shared.png"
    shared_evidence.write_bytes(b"shared-evidence")
    output_dir = tmp_path / "run"
    request = _request(source, output_dir).model_copy(
        update={
            "policy": {
                "sampled_video_frame_paths": [str(shared_evidence)],
                "current_image_paths": [str(shared_evidence)],
            }
        }
    )

    preparation = prepare_validation_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=_CoordinatorExecutor(),
    )

    roles = {
        item.evidence_kind
        for item in preparation.evidence_inventory
        if item.artifact.path == str(shared_evidence.resolve())
    }
    assert roles == {"runtime", "render"}


@pytest.mark.parametrize("nested_render_key", ("focused_image_paths", "runtime_render"))
def test_preparation_freezes_nested_render_evidence_paths(
    tmp_path: Path,
    nested_render_key: str,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    image = tmp_path / "render.png"
    image.write_bytes(b"frozen-render")
    output_dir = tmp_path / "run"
    nested_value: object
    if nested_render_key == "focused_image_paths":
        nested_value = {"/Asset": [image.name]}
    else:
        nested_value = {"status": "available", "image_paths": [image.name]}
    request = _request(source, output_dir).model_copy(
        update={"policy": {nested_render_key: nested_value}}
    )
    executor = _CoordinatorExecutor()

    preparation = prepare_validation_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert any(
        item.evidence_kind == "render" and item.artifact.path == str(image.resolve())
        for item in preparation.evidence_inventory
    )
    image.write_bytes(b"changed-render")
    with pytest.raises(ValidationWorkflowError, match="evidence inventory changed"):
        load_validation_coordinator_preparation(output_dir, executor=executor)


def test_valid_child_plan_is_accepted_then_executes_exact_adapters(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)

    accepted = accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    assert accepted.ordered_check_ids == ("schema", "behavior")
    assert [check.template_name for check in accepted.selected_checks] == [
        "physics_sane",
        "physical_behavior",
    ]
    assert executor.plan_calls == 1
    assert (output_dir / "validation_plan.json").is_file()
    plan = json.loads((output_dir / "validation_plan.json").read_text("utf-8"))
    assert plan["steps"][1]["metadata"]["agentic_work_item"]["depends_on"] == [
        "validation:physics_sane"
    ]
    assert (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).is_file()

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )

    assert executor.run_calls == [
        ("physics_sane", ()),
        ("physical_behavior", ("physics_sane",)),
    ]
    assert executor.behavior_evidence_paths == (str(tmp_path / "runtime.json"),)
    assert run.result.verdict == "pass"
    assert receipt.execution_disposition == "completed"
    assert receipt.required_checks_successful is True
    assert receipt.source_mutated is False
    assert receipt.publication_kind == "validation_assessment"
    assert (output_dir / VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME).is_file()


def test_accept_rejects_missing_child_launch_descriptor(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    (output_dir / "raw" / "validation_planner_launch_descriptor.json").unlink()

    with pytest.raises(
        ValidationWorkflowError,
        match="child launch descriptor is missing or unsafe",
    ):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert not (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).exists()


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    (
        ("runner_identity", "model", "other-model"),
        ("runner_identity", "model_reasoning_effort", "low"),
        ("runner_identity", "claude_execution_mode", "sdk"),
        ("artifacts", "child_output_path", "redirect-to-final"),
    ),
)
def test_accept_rejects_child_rewrite_of_parent_launch_descriptor(
    tmp_path: Path,
    section: str,
    field: str,
    replacement: str,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    descriptor_path = output_dir / "raw" / "validation_planner_launch_descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    expected_digest = canonical_json_digest(descriptor)
    replacement_value = (
        str((output_dir / "raw" / "codex_final.json").resolve())
        if replacement == "redirect-to-final"
        else replacement
    )
    descriptor[section][field] = replacement_value
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="differs from the parent-owned pre-launch descriptor",
    ):
        _accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            expected_child_launch_descriptor_digest=expected_digest,
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert not (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).exists()


def test_accept_rejects_empty_child_launch_stream(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    (output_dir / "raw" / "validation_planner_output.jsonl").write_text(
        "", encoding="utf-8"
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="child launch output is missing or unsafe",
    ):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert not (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        pytest.param(
            lambda payload: payload["selected_checks"][0].update(
                {"capability_id": "validation.unknown"}
            ),
            "unknown capability",
            id="unknown-capability",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][0].update(
                {"template_name": "render_valid"}
            ),
            "adapter identity",
            id="unapproved-template",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][0].update(
                {"depends_on": ["missing"]}
            ),
            "undeclared dependencies",
            id="missing-dependency",
        ),
        pytest.param(
            lambda payload: (
                payload["selected_checks"][0].update({"depends_on": ["behavior"]}),
                payload["selected_checks"][1].update({"depends_on": ["schema"]}),
            ),
            "cycle",
            id="dependency-cycle",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][0].update(
                {"targets": ["/undeclared/asset.usd"]}
            ),
            "exact top-level request inputs",
            id="undeclared-target",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][0].update(
                {"parameters": {"invented": True}}
            ),
            "unknown parameters",
            id="invented-parameter",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][1].update({"parameters": {}}),
            "available frozen runtime evidence",
            id="missing-required-runtime-evidence",
        ),
        pytest.param(
            lambda payload: (
                payload.update({"selected_checks": [payload["selected_checks"][0]]}),
                payload["claims"][0].update({"check_ids": ["schema"]}),
            ),
            "at least two distinct provider-free checks",
            id="too-few-provider-free-checks",
        ),
        pytest.param(
            lambda payload: payload["selected_checks"][1].update({"depends_on": []}),
            "at least one explicit dependency",
            id="missing-explicit-dependency",
        ),
        pytest.param(
            lambda payload: [
                check.update(
                    {
                        "required": False,
                        "completion_policy": "best_effort_advisory",
                    }
                )
                for check in payload["selected_checks"]
            ],
            "at least one required check",
            id="all-checks-advisory",
        ),
    ),
)
def test_outer_acceptance_rejects_adversarial_child_plans_before_planning(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    mutation(payload)  # type: ignore[operator]
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationWorkflowError, match=message):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.run_calls == []
    assert not (output_dir / "validation_plan.json").exists()


def test_child_cannot_replace_top_level_source_with_usd_dependency(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "dependency.usda"
    dependency.write_text('#usda 1.0\ndef Xform "Dependency" {}\n', encoding="utf-8")
    source = tmp_path / "asset.usda"
    source.write_text(
        '#usda 1.0\n( subLayers = [@dependency.usda@] )\ndef Xform "Asset" {}\n',
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"status":"passed"}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _CoordinatorExecutor()
    request = _request(source, output_dir).model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "canonical_usd",
                "physical_behavior_evidence": {"path": str(runtime)},
            }
        }
    )
    preparation = prepare_validation_coordinator(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )
    assert str(dependency.resolve()) in {
        item.path for item in preparation.source_dependency_closure
    }
    payload = _patch_payload(preparation.preparation_digest, source.resolve())
    payload["selected_checks"][0]["targets"] = [str(dependency.resolve())]
    patch = ValidationCoordinatorPlanPatch.model_validate(payload)

    with pytest.raises(ValidationWorkflowError, match="top-level request inputs"):
        _validate_plan_patch(preparation, patch)

    assert executor.plan_calls == 0
    assert executor.run_calls == []


def test_operator_required_capability_cannot_be_downgraded(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, payload = _prepare_patch(
        tmp_path,
        executor=executor,
        required_capability_ids=("validation.physical_behavior",),
    )
    payload["selected_checks"][1].update(  # type: ignore[index,union-attr]
        {"required": False, "completion_policy": "best_effort_advisory"}
    )
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationWorkflowError, match="made required.*advisory"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.run_calls == []


@pytest.mark.parametrize(
    "parameters",
    (
        pytest.param({"backend": "remote"}, id="provider-backend"),
        pytest.param({"views": ["corner"]}, id="render-views"),
        pytest.param({"image_width": 512}, id="render-width"),
        pytest.param({"image_height": 512}, id="render-height"),
    ),
)
def test_child_cannot_author_outer_render_policy(
    tmp_path: Path,
    parameters: dict[str, object],
) -> None:
    executor = _CoordinatorExecutor()
    source, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    payload["claims"] = [
        {
            "claim_id": "render",
            "statement": "The exact frozen source has valid render evidence.",
            "acceptance_criteria": ["The frozen render check is evaluated."],
            "check_ids": ["render"],
        }
    ]
    payload["selected_checks"] = [
        {
            "check_id": "render",
            "capability_id": "validation.render_valid",
            "template_name": "render_valid",
            "rule_id": "render.runtime_evidence",
            "targets": [str(source.resolve())],
            "focus_prim_paths": ["/Asset"],
            "parameters": parameters,
            "required": True,
            "depends_on": [],
            "evidence_requirements": [
                "source_identity",
                "render_result",
                "qualified_render_evidence",
            ],
            "completion_policy": "all_required_evidence",
        }
    ]
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    preparation = load_validation_coordinator_preparation(
        output_dir,
        executor=executor,
    )
    render_capability = next(
        item
        for item in preparation.approved_capabilities
        if item.capability_id == "validation.render_valid"
    )
    assert render_capability.allowed_parameters == ()
    with pytest.raises(ValidationWorkflowError, match="unknown parameters"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.run_calls == []
    assert not (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).exists()


def test_preparation_rejects_required_advisory_capability_before_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    executor = _CoordinatorExecutor()
    request = _request(source, output_dir).model_copy(
        update={
            "metadata": {
                "validation_required_capability_ids": ["validation.look_right"]
            }
        }
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="look_right is advisory and cannot be a required capability",
    ):
        prepare_validation_coordinator(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert not output_dir.exists()
    assert executor.plan_calls == 0


def test_preparation_inventories_supported_refinement_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    summary = tmp_path / "refine_summary.json"
    summary.write_text('{"status":"completed"}\n', encoding="utf-8")
    refine_output = tmp_path / "refine-output"
    refine_output.mkdir()
    request = _request(source, tmp_path / "run").model_copy(
        update={
            "policy": {
                "physical_behavior_refine_summary_path": summary.name,
                "physical_behavior_refine_output_dir": refine_output.name,
            }
        }
    )

    preparation = prepare_validation_coordinator(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=_CoordinatorExecutor(),
    )
    runtime_inventory = {
        Path(item.artifact.path): item
        for item in preparation.evidence_inventory
        if item.evidence_kind == "runtime"
    }

    assert runtime_inventory[summary.resolve()].availability == "available"
    assert runtime_inventory[summary.resolve()].artifact.kind == "file"
    assert runtime_inventory[refine_output.resolve()].availability == "available"
    assert runtime_inventory[refine_output.resolve()].artifact.kind == "directory"


def test_operator_focus_cannot_be_dropped_by_child_plan(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    payload["selected_checks"][0]["focus_prim_paths"] = []  # type: ignore[index]
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationWorkflowError, match="preserve all requested"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )


def test_failed_dependency_is_recorded_without_invoking_dependent_adapter(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor(skipped_on="physics_sane")
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )

    assert executor.run_calls == [("physics_sane", ())]
    behavior = next(
        result
        for result in run.result.template_results
        if result.template_name == "physical_behavior"
    )
    assert behavior.status == "failed"
    assert {issue.code for issue in behavior.issues} == {
        "validation.operation_dependency_not_satisfied",
        "validation.required_capability_not_evaluated",
    }
    assert receipt.required_checks_successful is False


def test_stale_preparation_and_early_execution_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    source, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    source.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValidationWorkflowError, match="changed"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )

    source.write_text("asset\n", encoding="utf-8")
    (output_dir / "operations").mkdir()
    payload["preparation_digest"] = json.loads(
        (output_dir / "validation_coordinator_preparation.json").read_text("utf-8")
    )["preparation_digest"]
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationWorkflowError, match="decision-only"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            executor=executor,
        )


def test_outer_acceptance_rejects_changed_child_session(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)

    with pytest.raises(ValidationWorkflowError, match="child session"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            child_session_id="different-child-session",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert not (output_dir / "validation_plan.json").exists()


def test_outer_acceptance_rejects_changed_final_plan_id_before_publication(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)

    with pytest.raises(ValidationWorkflowError, match="changed its plan ID"):
        accept_validation_coordinator_plan(
            output_dir,
            plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            child_output_path=output_dir / "raw" / "codex_final.json",
            producer="codex",
            child_plan_id="different-plan",
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert not (output_dir / "validation_plan.json").exists()
    assert not (output_dir / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME).exists()


def test_required_skipped_check_is_non_successful(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor(skipped_on="physical_behavior")
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    assert receipt.execution_disposition == "required_checks_failed"
    assert receipt.required_checks_successful is False


def test_advisory_skipped_check_remains_not_evaluated(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor(skipped_on="physical_behavior")
    _, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    payload["selected_checks"][1].update(  # type: ignore[index,union-attr]
        {"required": False, "completion_policy": "best_effort_advisory"}
    )
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )

    index = json.loads(
        (output_dir / "validation_operation_index.json").read_text("utf-8")
    )
    behavior = next(
        item
        for item in index["operations"]
        if item["template_name"] == "physical_behavior"
    )
    assert behavior["state"] == "not_evaluated"
    assert behavior["mandatory"] is False
    behavior_result = next(
        item
        for item in run.result.template_results
        if item.template_name == "physical_behavior"
    )
    assert behavior_result.metrics == {}
    assert "authority" not in behavior_result.metadata
    assert behavior_result.metadata["executor"] == "coordinator-test"
    assert receipt.execution_disposition == "completed"


@pytest.mark.parametrize("nonpassing_status", ("warn", "needs_refinement"))
def test_nonpassing_required_check_blocks_advisory_with_truthful_receipt(
    tmp_path: Path,
    nonpassing_status: Literal["warn", "needs_refinement"],
) -> None:
    executor = _CoordinatorExecutor(
        nonpassing_on="physics_sane",
        nonpassing_status=nonpassing_status,
    )
    _, output_dir, payload = _prepare_patch(tmp_path, executor=executor)
    payload["selected_checks"][1].update(  # type: ignore[index,union-attr]
        {"required": False, "completion_policy": "best_effort_advisory"}
    )
    (output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )

    assert executor.run_calls == [("physics_sane", ())]
    prerequisite = next(
        item
        for item in run.result.template_results
        if item.template_name == "physics_sane"
    )
    assert prerequisite.status == nonpassing_status
    behavior = next(
        item
        for item in run.result.template_results
        if item.template_name == "physical_behavior"
    )
    assert behavior.status == "skipped"
    assert behavior.metadata["blocked_dependencies"] == ["physics_sane"]
    assert "authority" not in behavior.metadata
    assert behavior.metrics == {}
    assert receipt.execution_disposition == "required_checks_failed"
    assert receipt.required_checks_successful is False


def test_terminal_receipt_binds_complete_coordinator_chain(tmp_path: Path) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )
    execute_validation_coordinator_plan(output_dir, executor=executor)
    prepare_standalone_validation_evidence(output_dir)
    assessment = ValidationCoordinatorAssessment(
        assessment_id="coordinator-chain-assessment",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-physics_sane",),
                disposition="pass",
                rationale="The exact schema adapter result passed.",
            ),
            ValidationGateAssessment(
                gate="runtime_validation",
                evidence_ids=("validation-template-physical_behavior",),
                disposition="pass",
                rationale="The exact behavior-evidence adapter result passed.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The source readback and package bindings passed.",
            ),
        ),
        terminal_disposition="pass",
        summary="Independent required gates pass.",
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
        findings=("Exact coordinator chain and assessment read back unchanged.",),
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    terminal = review_standalone_validation_assessment(
        output_dir,
        review_path=review_path,
    )

    assert terminal.receipt_status == "completed"
    assert terminal.coordinator_planning_agent_launched is True
    assert terminal.coordinator_preparation is not None
    assert terminal.coordinator_plan_patch is not None
    assert terminal.accepted_coordinator_plan is not None
    assert terminal.coordinator_execution is not None
    assert terminal.operation_index is not None
    assert terminal.publication_kind == "validation_assessment"
    assert terminal.source_mutated is False
    assert terminal.cleanup_disposition == "not_required"


def test_collect_evidence_accepts_dependency_unavailable_policy_promotion(
    tmp_path: Path,
) -> None:
    executor = _DependencyUnavailablePolicyExecutor()
    source, output_dir, payload = _prepare_patch(
        tmp_path,
        executor=executor,
        policy={"gate_policy": {"dependency_unavailable": "block"}},
    )
    source_sha256 = file_sha256(source)
    checks = _policy_patch_checks(source)
    checks.append(
        {
            "check_id": "critique",
            "capability_id": "validation.look_right",
            "template_name": "look_right",
            "rule_id": "visual.optional_advisory_critique",
            "targets": [str(source.resolve())],
            "focus_prim_paths": ["/Asset"],
            "parameters": {},
            "required": False,
            "depends_on": ["render"],
            "evidence_requirements": [
                "render_result",
                "reference_inventory",
                "critique_result",
            ],
            "completion_policy": "best_effort_advisory",
        }
    )
    _write_policy_patch(output_dir, payload, checks=checks)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )
    index = prepare_standalone_validation_evidence(output_dir)

    raw_results = tuple(
        record.accepted_result.result
        for record in run.checkpoint.records
        if record.accepted_result is not None
    )
    assert [result.status for result in raw_results] == [
        "passed",
        "passed",
        "skipped",
    ]
    assert run.result.verdict == "fail"
    assert run.result.metadata["gate_policy_evaluation"] == {
        "blocked": True,
        "reason": "dependency_unavailable",
        "blocked_issue_codes": ["visual.judge_unavailable"],
    }
    assert "validation.dependency_unavailable" in {
        issue.code for issue in run.result.issues
    }
    assert receipt.required_checks_successful is True
    assert index.nested_agent_launched is False
    assert len(index.assessment_identity_sha256) == 64
    assert (output_dir / "standalone_validation_evidence.json").is_file()
    assert index.operation_index.sha256 == file_sha256(Path(index.operation_index.path))
    assert file_sha256(source) == source_sha256


def test_collect_evidence_accepts_matched_expected_negative_normalization(
    tmp_path: Path,
) -> None:
    executor = _ExpectedNegativePolicyExecutor()
    source, output_dir, payload = _prepare_patch(
        tmp_path,
        executor=executor,
        policy={
            "expected_verdict": "fail",
            "expected_issue_codes": [
                "physics.no_physics_scene",
                "physics.no_rigid_bodies",
            ],
        },
    )
    source_sha256 = file_sha256(source)
    _write_policy_patch(
        output_dir,
        payload,
        checks=_policy_patch_checks(source),
    )
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )

    run, _ = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )
    index = prepare_standalone_validation_evidence(output_dir)

    raw_physics = next(
        record.accepted_result.result
        for record in run.checkpoint.records
        if record.accepted_result is not None and record.template_name == "physics_sane"
    )
    published_physics = next(
        result
        for result in run.result.template_results
        if result.template_name == "physics_sane"
    )
    assert raw_physics.status == "failed"
    assert {issue.severity for issue in raw_physics.issues} == {"fail"}
    assert published_physics.status == "warn"
    assert {issue.severity for issue in published_physics.issues} == {"warn"}
    assert all(
        issue.details
        == {
            "original_severity": "fail",
            "expected_failure": True,
        }
        for issue in published_physics.issues
    )
    assert run.result.verdict == "warn"
    assert run.result.metadata["expected_result"]["matched"] is True
    assert index.nested_agent_launched is False
    assert len(index.assessment_identity_sha256) == 64
    assert (output_dir / "standalone_validation_evidence.json").is_file()
    assert index.operation_index.sha256 == file_sha256(Path(index.operation_index.path))
    assert file_sha256(source) == source_sha256

    result_path = output_dir / "validation_result.json"
    stale_result = json.loads(result_path.read_text(encoding="utf-8"))
    stale_result["verdict"] = "pass"
    result_path.write_text(
        json.dumps(stale_result, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EmbeddedValidationAssessmentError, match="consistent"):
        prepare_standalone_validation_evidence(output_dir)


def test_terminal_receipt_rejects_failed_required_coordinator_check(
    tmp_path: Path,
) -> None:
    executor = _PassedWithFailIssueExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )
    _, coordinator_receipt = execute_validation_coordinator_plan(
        output_dir,
        executor=executor,
    )
    assert coordinator_receipt.required_checks_successful is False

    prepare_standalone_validation_evidence(output_dir)
    assessment = ValidationCoordinatorAssessment(
        assessment_id="failed-required-check-assessment",
        created_at=datetime.now(UTC),
        gates=(
            ValidationGateAssessment(
                gate="static_validation",
                evidence_ids=("validation-template-physics_sane",),
                disposition="pass",
                rationale="The schema adapter status is passed.",
            ),
            ValidationGateAssessment(
                gate="runtime_validation",
                evidence_ids=("validation-template-physical_behavior",),
                disposition="pass",
                rationale="The behavior adapter status is passed.",
            ),
            ValidationGateAssessment(
                gate="package_integrity",
                evidence_ids=("validation-package-integrity",),
                disposition="pass",
                rationale="The package bindings passed.",
            ),
        ),
        findings=(
            ValidationAssessmentFinding(
                finding_id="accepted-fail-severity-issue",
                source_evidence_ids=("validation-template-physical_behavior",),
                source_issue_codes=("physics.behavior_failed",),
                severity="warning",
                summary="The outer assessment records the embedded issue.",
                disposition="accepted",
                rationale="This fixture isolates the terminal receipt guard.",
            ),
        ),
        terminal_disposition="pass",
        summary="Outer assessment passed despite the embedded fail-severity issue.",
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
        findings=("The passing outer assessment is accepted.",),
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(
        review.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    terminal = review_standalone_validation_assessment(
        output_dir,
        review_path=review_path,
    )

    assert terminal.receipt_status == "rejected"
    assert terminal.coordinator_execution is not None


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("raw/codex_final.json"),
        Path("raw/validation_planner_launch_descriptor.json"),
        Path("validation_request.json"),
        Path("operations/physics_sane/operation_result.json"),
    ),
)
def test_standalone_assessment_reverifies_nested_coordinator_bindings(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    _, output_dir, _ = _prepare_patch(tmp_path, executor=executor)
    accept_validation_coordinator_plan(
        output_dir,
        plan_patch_path=output_dir / VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
        child_output_path=output_dir / "raw" / "codex_final.json",
        producer="codex",
        executor=executor,
    )
    execute_validation_coordinator_plan(output_dir, executor=executor)
    with (output_dir / relative_path).open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(EmbeddedValidationAssessmentError, match="stale"):
        prepare_standalone_validation_evidence(output_dir)


def test_plan_patch_model_forbids_duplicate_capability_instances(
    tmp_path: Path,
) -> None:
    executor = _CoordinatorExecutor()
    _, _, payload = _prepare_patch(tmp_path, executor=executor)
    payload["selected_checks"][1].update(  # type: ignore[index,union-attr]
        {
            "capability_id": "validation.physics_sane",
            "template_name": "physics_sane",
            "rule_id": "physics.usd_schema_sanity",
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        ValidationCoordinatorPlanPatch.model_validate(payload)
