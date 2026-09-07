# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the stable Validation Agent scaffold step adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_understanding.agentic.validation_scaffold import (
    DraftTemplateResult,
    DraftValidationContext,
    RuntimeVisualEvidence,
    TemplateRegistry,
)
from world_understanding.validation import (
    ValidationIssue,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
    scaffold_runner,
)
from world_understanding.validation.scaffold_runner import (
    ScaffoldValidationStepExecutor,
)


def _request(source: Path) -> ValidationRequest:
    return ValidationRequest(
        task_description="Validate that this asset renders and looks right.",
        inputs=(str(source),),
        requested_templates=("render_valid", "look_right"),
    )


def test_look_right_fails_closed_without_a_render_result(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = _request(source)
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    prior_result = ValidationTemplateResult(
        template_name="physics_sane",
        status="warn",
        issues=(
            ValidationIssue(
                code="physics.test_warning",
                severity="warn",
                message="A prior template emitted a warning.",
                template_name="physics_sane",
                subject="/Asset",
                details={"source": "test"},
            ),
        ),
    )

    result = executor.run(
        "look_right",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "attempt",
            previous_template_results=(prior_result,),
        ),
    )

    assert result.status == "failed"
    assert result.issues[0].code == "validation.render_handoff_missing"
    assert "accepted render_valid result" in result.issues[0].message
    assert result.metrics == {"issue_count": 1, "vlm_invoked": False}
    assert result.metadata["render_handoff_present"] is False


def test_look_right_fails_closed_for_malformed_render_handoff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = _request(source)
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    render_result = ValidationTemplateResult(
        template_name="render_valid",
        status="passed",
        metadata={"runtime_render": "not-a-mapping"},
    )
    later_result = ValidationTemplateResult(
        template_name="physics_sane",
        status="passed",
    )

    result = executor.run(
        "look_right",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "attempt",
            previous_template_results=(render_result, later_result),
        ),
    )

    assert result.status == "failed"
    assert result.issues[0].code == "validation.render_handoff_missing"
    assert "no runtime render evidence" in result.issues[0].message


def test_look_right_fails_closed_for_failed_render_handoff(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = _request(source)
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    render_result = ValidationTemplateResult(
        template_name="render_valid",
        status="failed",
        issues=(
            ValidationIssue(
                code="render.test_failure",
                severity="fail",
                message="The render failed validation.",
                template_name="render_valid",
            ),
        ),
        metadata={
            "runtime_render": {
                "status": "completed",
                "image_paths": [str(tmp_path / "untrusted.png")],
            }
        },
    )

    result = executor.run(
        "look_right",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "attempt",
            previous_template_results=(render_result,),
        ),
    )

    assert result.status == "failed"
    assert result.issues[0].code == "validation.render_handoff_missing"
    assert "passed render_valid result" in result.issues[0].message


def test_runtime_render_payload_returns_none_without_render_result() -> None:
    unrelated_result = ValidationTemplateResult(
        template_name="physics_sane",
        status="passed",
    )

    assert scaffold_runner._runtime_render_payload((unrelated_result,)) is None


def test_nonvisual_step_reuses_accepted_render_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = ValidationRequest(
        task_description="Validate rendering and physics.",
        inputs=(str(source),),
        requested_templates=("render_valid", "physics_sane"),
    )
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    render_result = ValidationTemplateResult(
        template_name="render_valid",
        status="passed",
        metadata={
            "runtime_render": {
                "status": "passed",
                "backend": "test",
                "image_paths": [str(tmp_path / "render.png")],
            }
        },
    )
    captured_runtime_evidence: RuntimeVisualEvidence | None = None

    def fake_run(
        template_name: str,
        context: DraftValidationContext,
        registry: TemplateRegistry | None,
    ) -> DraftTemplateResult:
        nonlocal captured_runtime_evidence
        captured_runtime_evidence = context.runtime_visual_evidence
        return DraftTemplateResult(
            template_name=template_name,
            status="passed",
        )

    monkeypatch.setattr(
        scaffold_runner,
        "run_validation_scaffold_template",
        fake_run,
    )

    result = executor.run(
        "physics_sane",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "attempt",
            previous_template_results=(render_result,),
        ),
    )

    assert result.status == "passed"
    assert captured_runtime_evidence is not None
    assert captured_runtime_evidence.status == "passed"
    assert captured_runtime_evidence.backend == "test"


def test_nonvisual_step_before_render_does_not_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    request = ValidationRequest(
        task_description="Validate physics before rendering.",
        inputs=(str(source),),
        requested_templates=("physics_sane", "render_valid"),
    )
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    captured_runtime_evidence: RuntimeVisualEvidence | None = None

    def unexpected_render(*args: object, **kwargs: object) -> RuntimeVisualEvidence:
        del args, kwargs
        pytest.fail("nonvisual step unexpectedly prepared render evidence")

    def fake_run(
        template_name: str,
        context: DraftValidationContext,
        registry: TemplateRegistry | None,
    ) -> DraftTemplateResult:
        del registry
        nonlocal captured_runtime_evidence
        captured_runtime_evidence = context.runtime_visual_evidence
        return DraftTemplateResult(
            template_name=template_name,
            status="passed",
        )

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold."
        "_runtime_visual_evidence_for_request",
        unexpected_render,
    )
    monkeypatch.setattr(
        scaffold_runner,
        "run_validation_scaffold_template",
        fake_run,
    )

    result = executor.run(
        "physics_sane",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "attempt",
        ),
    )

    assert result.status == "passed"
    assert captured_runtime_evidence is not None
    assert captured_runtime_evidence.attempted is False


def test_run_preserves_descriptor_backed_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    real_attempt = tmp_path / "real-attempt"
    real_attempt.mkdir()
    descriptor_path = tmp_path / "descriptor-path"
    descriptor_path.symlink_to(real_attempt, target_is_directory=True)
    request = _request(source)
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "plan")
    captured_working_dir: Path | None = None

    def fake_run(
        template_name: str,
        context: DraftValidationContext,
        registry: TemplateRegistry | None,
    ) -> DraftTemplateResult:
        del registry
        nonlocal captured_working_dir
        captured_working_dir = context.input_inventory.working_dir
        return DraftTemplateResult(
            template_name=template_name,
            status="passed",
        )

    monkeypatch.setattr(
        scaffold_runner,
        "run_validation_scaffold_template",
        fake_run,
    )

    result = executor.run(
        "render_valid",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=descriptor_path,
        ),
    )

    assert result.status == "passed"
    assert captured_working_dir == descriptor_path.absolute()
    assert captured_working_dir != real_attempt.resolve()
