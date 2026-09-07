# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public batch-path tests for ``content-workflow-cli validate``."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import content_agent_workflows.validation as validation_module
import content_agent_workflows.validation.operations as operations_module
import content_agent_workflows.validation.workflow as workflow_module
import pytest
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_METADATA_KEY,
    ExecutionArtifactBinding,
)
from content_agent_workflows.validation import (
    VALIDATION_COORDINATOR_PREPARATION_NAME,
    VALIDATION_COORDINATOR_SAFE_RESTART_NAME,
    EmbeddedValidationReceiptIndex,
    ValidationTerminalReceipt,
)
from PIL import Image
from world_understanding.validation import (
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
)

import content_workflow_cli.cli as cli_module
from content_workflow_cli.cli import _validation_resume_command, main

TOOLBOX_PROMPT = (
    "Validate that this generated electrician's toolbox renders successfully "
    "and looks like the supplied reference image. Do not modify the asset; "
    "save a report with evidence and recommended actions."
)


class _FakeExecutor:
    """Stand in for render/VLM dependencies while the real workflow runs."""

    def __init__(
        self,
        base_dir: Path,
        *,
        interrupt_on: str | None = None,
        skip_on: str | None = None,
    ) -> None:
        del base_dir
        self.interrupt_on = interrupt_on
        self.skip_on = skip_on
        self.calls: list[str] = []

    @property
    def template_versions(self) -> Mapping[str, str]:
        return {
            "render_valid": "fake.render-valid.v1",
            "look_right": "fake.look-right.v1",
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
                ValidationPlanStep(
                    template_name=name,
                    reason="Selected by the visual validation workflow.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Fake deterministic visual validation plan.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        if self.interrupt_on == template_name:
            raise KeyboardInterrupt("interrupted validation execution")
        self.calls.append(template_name)
        if self.skip_on == template_name:
            return ValidationTemplateResult(
                template_name=template_name,
                status="skipped",
                metadata={"selection_state": "not_evaluated"},
            )
        context.working_dir.mkdir(parents=True, exist_ok=True)
        evidence = context.working_dir / f"{template_name}.png"
        Image.new("RGB", (8, 8), (230, 190, 20)).save(evidence, format="PNG")
        metadata: dict[str, Any] = {"executor": "fake"}
        if template_name == "render_valid":
            metadata["runtime_render"] = {
                "status": "passed",
                "backend": "fake",
                "image_paths": [str(evidence)],
                "render_response": None,
                "render_output_dir": str(context.working_dir),
                "issues": [],
                "metadata": {},
            }
            metadata["adapter_result"] = {
                "status": "pass",
                "verdict": "pass",
                "issues": [],
            }
        return ValidationTemplateResult(
            template_name=template_name,
            status="passed",
            metrics={"issue_count": 0, "vlm_invoked": template_name == "look_right"},
            evidence={"image_paths": [str(evidence)]},
            metadata=metadata,
        )


class _ExecutorRecorder:
    """Track every executor the workflow builds and stage interruptions."""

    def __init__(self) -> None:
        self.created: list[_FakeExecutor] = []
        self.interrupt_on: str | None = None
        self.skip_on: str | None = None

    def build(self, base_dir: Path) -> _FakeExecutor:
        executor = _FakeExecutor(
            base_dir,
            interrupt_on=self.interrupt_on,
            skip_on=self.skip_on,
        )
        self.created.append(executor)
        return executor

    @property
    def first_calls(self) -> list[str]:
        return self.created[0].calls

    @property
    def last_calls(self) -> list[str]:
        return self.created[-1].calls


@pytest.fixture
def executors(monkeypatch: pytest.MonkeyPatch) -> _ExecutorRecorder:
    """Run the real workflow while faking only the render/VLM dependency."""

    recorder = _ExecutorRecorder()
    monkeypatch.setattr(
        workflow_module,
        "ScaffoldValidationStepExecutor",
        recorder.build,
    )
    monkeypatch.setattr(
        operations_module,
        "ScaffoldValidationStepExecutor",
        recorder.build,
    )
    return recorder


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "toolbox.usda"
    source.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Toolbox"\n)\ndef Xform "Toolbox"\n{\n}\n',
        encoding="utf-8",
    )
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), (220, 180, 0)).save(reference, format="PNG")
    return source.resolve(), reference.resolve()


def _run_argv(source: Path, reference: Path, output_dir: Path) -> list[str]:
    return [
        "validate",
        "run",
        "--usd",
        str(source),
        "--task",
        TOOLBOX_PROMPT,
        "--reference-image",
        str(reference),
        "--output-dir",
        str(output_dir),
        "--direct-executor",
    ]


@pytest.mark.parametrize(
    ("subcommand", "typed_option", "typed_help"),
    (
        ("assess", "--assessment", "Outer-authored typed Validation assessment"),
        (
            "review-assessment",
            "--review",
            "Outer-authored review draft",
        ),
    ),
)
def test_validate_assessment_subcommands_document_outer_contract(
    subcommand: str,
    typed_option: str,
    typed_help: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", subcommand, "--help"]) == 0

    help_text = capsys.readouterr().out
    assert typed_option in help_text
    assert typed_help in help_text
    assert "asset_run.json" in help_text
    assert "omit for standalone" in help_text.lower()


def test_validate_run_documents_agentic_default_and_direct_compatibility(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", "run", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "--runner" in help_text
    assert "--direct-executor" in help_text
    assert "Decision-only planning child runner" in help_text


def test_validate_agentic_requires_explicit_runner_before_preparation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--reference-image",
                str(reference),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 2
    )

    assert "--runner is required for agentic Validation" in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("runner", "runner_args"),
    (
        ("codex", ()),
        ("claude", ("--claude-execution-mode", "cli")),
    ),
)
def test_validate_agentic_dry_run_prepares_without_selecting_or_executing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    runner: str,
    runner_args: tuple[str, ...],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / f"{runner}-run"

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--reference-image",
                str(reference),
                "--output-dir",
                str(output_dir),
                "--runner",
                runner,
                "--model",
                "test-model",
                *runner_args,
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "prepared"
    preparation = json.loads(
        (output_dir / "validation_coordinator_preparation.json").read_text("utf-8")
    )
    assert preparation["request"]["requested_templates"] == []
    assert preparation["selected_checks"] == []
    assert not (output_dir / "validation_plan.json").exists()
    assert not (output_dir / "validation_request.json").exists()
    prompt = (output_dir / "prompts" / "validation_coordinator_plan.md").read_text(
        "utf-8"
    )
    assert f"producer='{runner}'" in prompt


@pytest.mark.parametrize(
    ("runner_args", "message"),
    (
        (("--runner", "codex"), "--model is required"),
        (
            ("--runner", "claude", "--model", "test-model"),
            "--claude-execution-mode is required",
        ),
        (
            (
                "--runner",
                "codex",
                "--model",
                "test-model",
                "--claude-execution-mode",
                "cli",
            ),
            "may only be selected with --runner claude",
        ),
    ),
)
def test_validate_agentic_requires_explicit_provider_selection_before_preparation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    runner_args: tuple[str, ...],
    message: str,
) -> None:
    source, _reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(output_dir),
                *runner_args,
                "--dry-run",
            ]
        )
        == 2
    )

    assert message in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "coordinator_args",
    (
        ("--dry-run",),
        ("--repo-root", "."),
        ("--runner", "codex"),
        ("--model", "test-model"),
        ("--model-reasoning-effort", "high"),
        ("--codex-base-url", "https://example.test"),
        ("--codex-sandbox-mode", "workspace-write"),
        ("--codex-config-json", '{"model":"test"}'),
        ("--codex-config-file", "codex-config.json"),
        ("--claude-permission-mode", "default"),
        ("--claude-max-turns", "3"),
        ("--claude-execution-mode", "sdk"),
        ("--claude-config-json", '{"settings":{}}'),
        ("--claude-config-file", "claude-config.json"),
        ("--child-timeout", "30"),
        ("--agent-cwd", "."),
        ("--required-capability", "validation.render_valid"),
    ),
)
def test_validate_direct_executor_rejects_coordinator_only_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    coordinator_args: tuple[str, ...],
) -> None:
    source, reference = _inputs(tmp_path)

    assert (
        main(
            [
                *_run_argv(source, reference, tmp_path / "run"),
                *coordinator_args,
            ]
        )
        == 2
    )

    assert (
        "these options require the agentic coordinator path" in capsys.readouterr().err
    )


def test_validate_direct_executor_allows_ambient_codex_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executors: _ExecutorRecorder,
) -> None:
    source, reference = _inputs(tmp_path)
    monkeypatch.setenv("CONTENT_AGENT_CODEX_BASE_URL", "https://example.test")

    assert main(_run_argv(source, reference, tmp_path / "run")) == 0


def test_validate_policy_file_resolves_qualified_render_against_base_dir(
    tmp_path: Path,
) -> None:
    source, _ = _inputs(tmp_path)
    image = tmp_path / "qualified.png"
    Image.new("RGB", (8, 8), (20, 80, 140)).save(image, format="PNG")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "qualified_render_evidence": [
                    {
                        "path": image.name,
                        "role": "qualified_render",
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "source_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
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
                            "renderer_identities": [None],
                        },
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(output_dir),
                "--base-dir",
                str(tmp_path),
                "--policy-file",
                str(policy_path),
                "--runner",
                "codex",
                "--model",
                "test-model",
                "--dry-run",
            ]
        )
        == 0
    )

    preparation = json.loads(
        (output_dir / "validation_coordinator_preparation.json").read_text("utf-8")
    )
    qualified = preparation["request"]["policy"]["qualified_render_evidence"]
    assert qualified[0]["path"] == str(image.resolve())
    assert qualified[0]["ovrtx_render_metadata"]["renderer_identities"] == [None]


def test_validate_render_flags_have_same_scaffold_policy_with_empty_policy_file(
    tmp_path: Path,
) -> None:
    source, _ = _inputs(tmp_path)
    empty_policy = tmp_path / "empty-policy.json"
    empty_policy.write_text("{}\n", encoding="utf-8")
    policies: list[dict[str, object]] = []

    for index, policy_args in enumerate(((), ("--policy-file", str(empty_policy)))):
        output_dir = tmp_path / f"run-{index}"
        assert (
            main(
                [
                    "validate",
                    "run",
                    "--usd",
                    str(source),
                    "--task",
                    TOOLBOX_PROMPT,
                    "--output-dir",
                    str(output_dir),
                    "--render-backend",
                    "remote",
                    "--render-view",
                    "front",
                    "--render-width",
                    "320",
                    "--render-height",
                    "240",
                    "--runner",
                    "codex",
                    "--model",
                    "test-model",
                    *policy_args,
                    "--dry-run",
                ]
            )
            == 0
        )
        preparation = json.loads(
            (output_dir / "validation_coordinator_preparation.json").read_text("utf-8")
        )
        policies.append(preparation["request"]["policy"])

    assert policies[0] == policies[1]
    assert policies[0]["expected_cameras"] == ["front"]
    assert policies[0]["render_backend"] == "remote"
    assert policies[0]["render_image_height"] == 240
    assert policies[0]["render_image_width"] == 320


def test_validate_policy_file_rejects_explicit_flag_overlap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"reference_image_paths": [reference.name]}) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--reference-image",
                str(reference),
                "--output-dir",
                str(tmp_path / "run"),
                "--base-dir",
                str(tmp_path),
                "--policy-file",
                str(policy_path),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "cannot override policy keys" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("policy_key", "policy_value", "render_args"),
    (
        ("expected_cameras", ["back"], ("--render-view", "front")),
        ("render_view_directions", ["back"], ("--render-view", "front")),
        ("runtime_render_views", ["back"], ("--render-view", "front")),
        ("render_image_width", 640, ("--render-width", "320")),
        ("render_image_height", 480, ("--render-height", "240")),
    ),
)
def test_validate_policy_file_cannot_override_explicit_render_geometry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    policy_key: str,
    policy_value: object,
    render_args: tuple[str, str],
) -> None:
    source, _ = _inputs(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({policy_key: policy_value}) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(tmp_path / "run"),
                "--policy-file",
                str(policy_path),
                *render_args,
                "--dry-run",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "cannot override policy keys" in error
    assert policy_key in error


def test_validate_policy_file_rejects_null_outside_ovrtx_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _inputs(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"unexpected":null}\n', encoding="utf-8")

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(tmp_path / "run"),
                "--policy-file",
                str(policy_path),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "outside a qualified OVRTX renderer identity" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("subcommand", "required_option", "contract_text"),
    (
        (
            "ingest-verified-operation-result",
            "--envelope",
            "domain projector",
        ),
        (
            "produce-canonical-visual-evidence",
            "--render-backend",
            "shared OVRTX",
        ),
    ),
)
def test_validate_public_provided_result_leaves_are_documented(
    subcommand: str,
    required_option: str,
    contract_text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", subcommand, "--help"]) == 0

    help_text = capsys.readouterr().out
    assert required_option in help_text
    assert contract_text in help_text


def test_validate_ingest_verified_operation_result_is_independently_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text("{}\n", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []
    receipt = SimpleNamespace(
        envelope=SimpleNamespace(
            operation_id="physics.mass-properties.verified",
            native_status="pass",
        ),
        execution_mode="provided",
        model_dump=lambda **_kwargs: {
            "execution_mode": "provided",
            "operation_id": "physics.mass-properties.verified",
        },
    )

    def ingest(envelope: Path, *, output_dir: Path) -> object:
        calls.append((envelope, output_dir))
        return receipt

    monkeypatch.setattr(validation_module, "ingest_verified_operation_result", ingest)
    run_dir = tmp_path / "provided"
    assert (
        main(
            [
                "validate",
                "ingest-verified-operation-result",
                "--envelope",
                str(envelope_path),
                "--output-dir",
                str(run_dir),
                "--json",
            ]
        )
        == 0
    )
    assert calls == [(envelope_path, run_dir)]
    assert json.loads(capsys.readouterr().out)["execution_mode"] == "provided"


def test_validate_canonical_visual_evidence_leaf_is_independently_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    source_path = tmp_path / "source.usda"
    source_path.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    publication = SimpleNamespace(
        envelope=SimpleNamespace(path=str(tmp_path / "envelope.json")),
        result=SimpleNamespace(output=SimpleNamespace(sha256="a" * 64)),
        model_dump=lambda **_kwargs: {"schema_version": "publication.v1"},
    )

    def produce(**kwargs: object) -> object:
        calls.append(kwargs)
        return publication

    monkeypatch.setattr(validation_module, "produce_canonical_visual_evidence", produce)
    assert (
        main(
            [
                "validate",
                "produce-canonical-visual-evidence",
                "--usd",
                str(usd_path),
                "--source-usd",
                str(source_path),
                "--output-dir",
                str(tmp_path / "visual"),
                "--render-backend",
                "ovrtx",
                "--json",
            ]
        )
        == 0
    )
    assert calls[0]["post_mutation_usd"] == usd_path
    assert calls[0]["source_usd"] == source_path
    assert calls[0]["backend"] == "ovrtx"
    assert json.loads(capsys.readouterr().out)["schema_version"] == "publication.v1"


def test_validate_canonical_visual_evidence_requires_source_binding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    assert (
        main(
            [
                "validate",
                "produce-canonical-visual-evidence",
                "--usd",
                str(usd_path),
                "--output-dir",
                str(tmp_path / "visual"),
                "--render-backend",
                "ovrtx",
            ]
        )
        == 2
    )

    assert "--source-usd" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ("standalone", "embedded"))
def test_validate_review_assessment_json_returns_mode_neutral_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: Literal["standalone", "embedded"],
) -> None:
    output_dir = tmp_path / mode
    output_dir.mkdir()
    bound_artifact = ExecutionArtifactBinding(
        path=str(tmp_path / "bound.json"),
        sha256="a" * 64,
        size_bytes=1,
    )
    receipt = ValidationTerminalReceipt(
        mode=mode,
        assessment_identity_sha256="b" * 64,
        evidence_index=bound_artifact,
        canonical_assessment=bound_artifact,
        coordinator_review=bound_artifact,
        gate_dispositions={
            "static_validation": "pass",
            "runtime_validation": "not_evaluated",
            "visual_quality": "pass",
            "package_integrity": "pass",
            "cross_stage_integrity": "not_evaluated",
        },
        terminal_disposition="pass",
        review_disposition="accept",
        receipt_status="completed",
    )
    review_path = tmp_path / "review.json"
    review_path.write_text("{}\n", encoding="utf-8")
    args = [
        "validate",
        "review-assessment",
        "--output-dir",
        str(output_dir),
        "--review",
        str(review_path),
        "--json",
    ]
    if mode == "standalone":
        monkeypatch.setattr(
            validation_module,
            "review_standalone_validation_assessment",
            lambda *_args, **_kwargs: receipt,
        )
    else:
        terminal_path = output_dir / "validation_terminal_receipt.json"
        terminal_path.write_text(
            receipt.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        terminal_binding = ExecutionArtifactBinding(
            path=str(terminal_path),
            sha256=hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
            size_bytes=terminal_path.stat().st_size,
        )
        embedded_index = EmbeddedValidationReceiptIndex.model_construct(
            terminal_receipt=terminal_binding,
            receipt_status="completed",
        )
        monkeypatch.setattr(
            validation_module,
            "review_embedded_validation_assessment",
            lambda *_args, **_kwargs: embedded_index,
        )
        run_state = tmp_path / "asset_run.json"
        run_state.write_text("{}\n", encoding="utf-8")
        args.extend(("--embedded-run-state", str(run_state)))

    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == receipt.model_dump(mode="json")


def test_validate_focused_cli_prepares_runs_finalizes_and_collects_evidence(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _inputs(tmp_path)
    output_dir = tmp_path / "focused-run"

    assert (
        main(
            [
                "validate",
                "prepare",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(output_dir),
                "--profile",
                "visual",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                "check",
                "--output-dir",
                str(output_dir),
                "--template",
                "render_valid",
            ]
        )
        == 0
    )
    assert main(["validate", "finalize", "--output-dir", str(output_dir)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "validate",
                "collect-evidence",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["nested_agent_launched"] is False
    assert payload["operation_index"]["path"].endswith(
        "validation_operation_index.json"
    )
    assert executors.last_calls == ["render_valid"]


def test_validate_check_reports_optional_skip_as_process_success_not_semantic_pass(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _inputs(tmp_path)
    output_dir = tmp_path / "focused-run"
    assert (
        main(
            [
                "validate",
                "prepare",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(output_dir),
                "--template",
                "render_valid",
                "--template",
                "look_right",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                "check",
                "--output-dir",
                str(output_dir),
                "--template",
                "render_valid",
            ]
        )
        == 0
    )
    render_result = output_dir / "operations" / "render_valid" / "operation_result.json"
    capsys.readouterr()
    executors.skip_on = "look_right"

    assert (
        main(
            [
                "validate",
                "check",
                "--output-dir",
                str(output_dir),
                "--template",
                "look_right",
                "--prior-result",
                str(render_result),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "look_right: skipped" in output
    assert "Semantic outcome: not_evaluated (not passed)" in output
    operation = json.loads(
        (output_dir / "operations" / "look_right" / "operation_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert operation["template_result"]["status"] == "skipped"
    assert operation["template_result"]["metadata"]["selection_state"] == (
        "not_evaluated"
    )

    assert main(["validate", "finalize", "--output-dir", str(output_dir)]) == 0
    index = json.loads(
        (output_dir / "validation_operation_index.json").read_text(encoding="utf-8")
    )
    critique = next(
        item for item in index["operations"] if item["template_name"] == "look_right"
    )
    assert critique["state"] == "not_evaluated"
    assert index["optional_critique_evaluated"] is False


def test_validate_collect_evidence_returns_typed_error_before_finalize(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _inputs(tmp_path)
    output_dir = tmp_path / "focused-run"
    assert (
        main(
            [
                "validate",
                "prepare",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--output-dir",
                str(output_dir),
                "--profile",
                "visual",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "validate",
                "collect-evidence",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Invalid embedded Validation artifact" in captured.err


def test_validate_run_publishes_an_evidence_backed_report(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert main(_run_argv(source, reference, output_dir)) == 0

    assert (output_dir / "validation_request.json").is_file()
    assert (output_dir / "validation_plan.json").is_file()
    assert (output_dir / "validation_result.json").is_file()
    assert (output_dir / "validation_evidence.json").is_file()
    assert (output_dir / "final_summary.json").is_file()

    result = json.loads((output_dir / "validation_result.json").read_text("utf-8"))
    assert result["verdict"] == "pass"
    request = json.loads((output_dir / "validation_request.json").read_text("utf-8"))
    assert DOMAIN_EXECUTION_CONTEXT_METADATA_KEY not in request["metadata"]
    assert "embedded_execution" not in result["metadata"]
    assert "semantic_completion_authority" not in result["metadata"]
    assert [step["template_name"] for step in result["plan"]["steps"]] == [
        "render_valid",
        "look_right",
    ]
    # Validation never modifies the source asset.
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest

    out = capsys.readouterr().out
    assert "Verdict: pass (completed)" in out
    assert "render_valid: completed" in out
    assert "look_right: completed" in out


def test_completed_classic_embedded_run_automatically_prepares_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_state = tmp_path / "asset_run.json"
    run_state.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "validation"
    request = SimpleNamespace()
    run = SimpleNamespace(
        status=SimpleNamespace(value="completed"),
        output_dir=str(output_dir),
        result=SimpleNamespace(
            verdict="pass",
            model_dump=lambda **_kwargs: {"verdict": "pass"},
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "run_validation_workflow",
        lambda *_args, **_kwargs: run,
    )
    calls: list[tuple[object, Path, bool]] = []
    monkeypatch.setattr(
        cli_module,
        "_prepare_fixed_pipeline_embedded_validation_evidence",
        lambda observed_run, *, run_state_path, defer_until_cross_stage: calls.append(
            (observed_run, run_state_path, defer_until_cross_stage)
        ),
    )

    assert (
        cli_module._run_validation_workflow_command(
            SimpleNamespace(
                embedded_run_state=run_state,
                json=True,
                fail_on_warn=False,
            ),
            request=request,  # type: ignore[arg-type]
            output_dir=output_dir,
            base_dir=tmp_path,
            resume=False,
        )
        == 0
    )
    assert calls == [(run, run_state, True)]
    assert json.loads(capsys.readouterr().out) == {"verdict": "pass"}


def test_validate_resume_reuses_render_valid_and_runs_only_look_right(
    tmp_path: Path,
    executors: _ExecutorRecorder,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    executors.interrupt_on = "look_right"
    assert main(_run_argv(source, reference, output_dir)) == 130

    assert executors.first_calls == ["render_valid"]
    assert not (output_dir / "validation_result.json").exists()
    standalone_request_bytes = (output_dir / "validation_request.json").read_bytes()

    executors.interrupt_on = None
    assert (
        main(
            [
                "validate",
                "resume",
                "--output-dir",
                str(output_dir),
                "--recover-orphaned-claims",
            ]
        )
        == 0
    )

    # The accepted render_valid result is reused; only look_right re-executes.
    assert executors.last_calls == ["look_right"]
    assert (output_dir / "validation_request.json").read_bytes() == (
        standalone_request_bytes
    )
    result = json.loads((output_dir / "validation_result.json").read_text("utf-8"))
    assert result["verdict"] == "pass"
    assert "embedded_execution" not in result["metadata"]
    assert {entry["template_name"] for entry in result["template_results"]} == {
        "render_valid",
        "look_right",
    }


def test_validate_resume_refuses_to_run_beside_a_live_claim(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    executors.interrupt_on = "look_right"
    assert main(_run_argv(source, reference, output_dir)) == 130

    # Resume cannot distinguish a crashed runner from a live concurrent one, so
    # it refuses by default and points the operator at the recovery flag.
    executors.interrupt_on = None
    assert main(["validate", "resume", "--output-dir", str(output_dir)]) == 2
    err = capsys.readouterr().err
    assert "active validation work" in err
    assert "look_right" in err


def test_validate_resume_fails_closed_when_the_asset_changed(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    executors.interrupt_on = "look_right"
    assert main(_run_argv(source, reference, output_dir)) == 130

    source.write_text("#usda 1.0\n(\n)\n", encoding="utf-8")
    executors.interrupt_on = None

    assert (
        main(
            [
                "validate",
                "resume",
                "--output-dir",
                str(output_dir),
                "--recover-orphaned-claims",
            ]
        )
        == 2
    )
    assert "identity" in capsys.readouterr().err.lower()


def test_validate_resume_without_a_prior_run_reports_a_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "missing"
    output_dir.mkdir()

    assert main(["validate", "resume", "--output-dir", str(output_dir)]) == 2
    assert "no validation run to resume" in capsys.readouterr().err


def test_validate_coordinator_resume_writes_safe_restart_without_legacy_execution(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "scaffold.usda"
    source.write_text('#usda 1.0\ndef Xform "Scaffold" {}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    request = ValidationRequest(
        task_description="Validate saved behavior evidence.",
        inputs=(str(source),),
        requested_templates=("physics_sane", "physical_behavior"),
        policy={"visual_evidence_mode": "canonical_usd"},
    )
    (output_dir / "validation_request.json").write_text(
        request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / VALIDATION_COORDINATOR_PREPARATION_NAME).write_text(
        '{"schema_version":"test-coordinator-preparation"}\n',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "validate",
                "resume",
                "--output-dir",
                str(output_dir),
                "--recover-orphaned-claims",
                "--json",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    receipt_path = output_dir / VALIDATION_COORDINATOR_SAFE_RESTART_NAME
    first_receipt = receipt_path.read_bytes()

    assert payload["disposition"] == "safe_restart_required"
    assert payload["reason_code"] == "coordinator_resume_unsupported"
    assert payload["unsupported_capability_ids"] == [
        "validation.physics_sane",
        "validation.physical_behavior",
    ]
    assert payload["unsupported_template_names"] == [
        "physics_sane",
        "physical_behavior",
    ]
    assert payload["prior_run_preserved"] is True
    assert payload["restart_requires_fresh_run_identity"] is True
    assert payload["restart_requires_fresh_output_dir"] is True
    assert payload["prior_results_reused"] is False
    assert payload["stale_result_promotion_refused"] is True
    assert payload["legacy_executor_invoked"] is False
    assert payload["nested_agent_launched"] is False
    assert payload["source_mutated"] is False
    assert executors.created == []
    assert not (output_dir / "validation_checkpoint.json").exists()

    assert (
        main(
            [
                "validate",
                "resume",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 2
    )
    assert receipt_path.read_bytes() == first_receipt
    assert json.loads(capsys.readouterr().out) == payload


def test_validate_run_emits_json_when_requested(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    assert main([*_run_argv(source, reference, output_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "pass"
    assert payload["metadata"]["workflow_status"] == "completed"


def test_validate_run_rejects_a_second_run_in_the_same_directory(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    assert main(_run_argv(source, reference, output_dir)) == 0
    assert main(_run_argv(source, reference, output_dir)) == 2
    assert "already exists" in capsys.readouterr().err


def test_validate_interruption_points_at_the_recovery_command(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    executors.interrupt_on = "render_valid"
    # Ctrl-C is an interruption, not a configuration error: exit 130, no
    # traceback, and the exact command needed to continue.
    assert main(_run_argv(source, reference, output_dir)) == 130

    err = capsys.readouterr().err
    assert "interrupted" in err
    assert f"validate resume --output-dir {output_dir}" in err
    assert "--recover-orphaned-claims" in err


def test_validate_interruption_quotes_embedded_recovery_paths(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run with spaces"
    run_state = tmp_path / "outer state; keep.json"

    command = _validation_resume_command(output_dir, run_state)

    assert shlex.split(command) == [
        "content-workflow-cli",
        "validate",
        "resume",
        "--output-dir",
        str(output_dir),
        "--recover-orphaned-claims",
        "--embedded-run-state",
        str(run_state),
    ]


def test_validate_without_a_subcommand_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate"]) == 1
    assert "usage:" in capsys.readouterr().out


def test_validate_run_rejects_a_non_image_reference(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _ = _inputs(tmp_path)
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("not a reference image\n", encoding="utf-8")

    assert (
        main(
            [
                "validate",
                "run",
                "--usd",
                str(source),
                "--task",
                TOOLBOX_PROMPT,
                "--reference-image",
                str(not_an_image),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert "expects an image file" in capsys.readouterr().err
    # Rejected before any work started.
    assert executors.created == []


def test_validate_run_restricts_the_plan_to_requested_templates(
    tmp_path: Path,
    executors: _ExecutorRecorder,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    argv = [*_run_argv(source, reference, output_dir), "--template", "render_valid"]
    assert main(argv) == 0

    assert executors.first_calls == ["render_valid"]
    plan = json.loads((output_dir / "validation_plan.json").read_text("utf-8"))
    assert [step["template_name"] for step in plan["steps"]] == ["render_valid"]


def test_validate_resume_emits_json_when_requested(
    tmp_path: Path,
    executors: _ExecutorRecorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    executors.interrupt_on = "look_right"
    assert main(_run_argv(source, reference, output_dir)) == 130
    capsys.readouterr()

    executors.interrupt_on = None
    assert (
        main(
            [
                "validate",
                "resume",
                "--output-dir",
                str(output_dir),
                "--recover-orphaned-claims",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "pass"
