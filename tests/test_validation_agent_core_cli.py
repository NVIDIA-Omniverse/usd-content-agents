# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Validation Agent core config and policy helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import typer
from PIL import Image as PILImage
from PIL import ImageDraw
from rich.console import Console

from world_understanding.validation import (
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationRenderConfig,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateResult,
    ValidationVerdict,
)
from world_understanding.validation.cli import (
    DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION,
    PASS_EXIT_CODE,
    VALIDATION_CLI_ERROR_EXIT_CODE,
    VALIDATION_FAILURE_EXIT_CODE,
    ValidationCliError,
    ValidationCliRun,
    _apply_expected_result_policy,
    _apply_gate_policy,
    _dependency_unavailable_recommended_action,
    _downgrade_expected_failure_issue,
    _gate_policy_string_sequence,
    _scaffold_policy_from_request,
    build_validation_request_from_inputs,
    load_validation_request_config,
    print_validation_summary,
    run_validation_cli_command,
    run_validation_from_config,
    run_validation_from_inputs,
    run_validation_inputs_cli_command,
    run_validation_request,
    validation_exit_code,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeLookRightVLM:
    backend_name = "fake"
    model_name = "fake-vlm"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate_with_image_caption_pairs(
        self,
        *,
        image_caption_pairs: list[tuple[str, str]],
        final_prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs: object,
    ) -> str:
        self.calls.append(
            {
                "image_caption_pairs": image_caption_pairs,
                "final_prompt": final_prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self.response


def _write_valid_image(path: Path) -> Path:
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    for x in range(32):
        for y in range(32):
            image.putpixel((x, y), (255, 0, 0))
            image.putpixel((x + 32, y), (0, 255, 0))
            image.putpixel((x, y + 32), (0, 0, 255))
            image.putpixel((x + 32, y + 32), (255, 255, 0))
    image.save(path)
    return path


def _expected_failure_policy_result(
    *,
    policy: Mapping[str, Any],
    verdict: ValidationVerdict = "fail",
    issues: Sequence[ValidationIssue] | None = None,
    template_results: Sequence[ValidationTemplateResult] | None = None,
) -> ValidationResult:
    if issues is None:
        issues = (
            ValidationIssue(
                code="physics.no_physics_scene",
                severity="fail",
                message="No physics scene.",
                template_name="physics_sane",
            ),
        )
    if template_results is None:
        template_results = (
            ValidationTemplateResult(
                template_name="physics_sane",
                status="failed",
                issues=tuple(issues),
            ),
        )
    return ValidationResult(
        verdict=verdict,
        request=ValidationRequest(
            task_description="Validate known-negative policy.",
            inputs=("asset.usd",),
            requested_templates=("physics_sane",),
            policy=dict(policy),
        ),
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="physics_sane",
                    reason="test",
                ),
            ),
        ),
        template_results=tuple(template_results),
        issues=tuple(issues),
    )


def _fake_cli_run(
    *,
    verdict: ValidationVerdict = "pass",
    exit_code: int = PASS_EXIT_CODE,
    issues: Sequence[ValidationIssue] = (),
) -> ValidationCliRun:
    request = ValidationRequest(
        task_description="Validate CLI output.",
        inputs=("asset.usd",),
        requested_templates=("render_valid",),
    )
    template_result = ValidationTemplateResult(
        template_name="render_valid",
        status="failed" if issues else "passed",
        issues=tuple(issues),
    )
    result = ValidationResult(
        verdict=verdict,
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="render_valid",
                    reason="requested",
                ),
            )
        ),
        template_results=(template_result,),
        issues=tuple(issues),
    )
    return ValidationCliRun(
        request=request,
        result=result,
        output_dir=Path("/tmp/validation-agent-run"),
        artifact_paths={
            "validation_request": "/tmp/validation-agent-run/validation_request.json",
            "validation_plan": "/tmp/validation-agent-run/validation_plan.json",
            "validation_result": "/tmp/validation-agent-run/validation_result.json",
        },
        exit_code=exit_code,
    )


def test_validation_agent_builds_request_from_direct_inputs(tmp_path: Path) -> None:
    request = build_validation_request_from_inputs(
        task_description="  Validate that the generated asset looks correct.  ",
        inputs=("asset.usda",),
        output_dir="direct-run",
        template_overrides=("render_valid", "look_right"),
        focus_prim_overrides=("/World/Handle",),
        reference_image_paths=("reference.png",),
        render_backend="remote",
        render_views=("front", "right"),
        render_image_width=512,
        render_image_height=256,
        base_dir=tmp_path,
    )

    assert (
        request.task_description == "Validate that the generated asset looks correct."
    )
    assert request.inputs == ("asset.usda",)
    assert request.project.working_dir == str(
        (tmp_path / "direct-run").resolve(strict=False)
    )
    assert request.requested_templates == ("render_valid", "look_right")
    assert request.focus.prim_paths == ("/World/Handle",)
    assert request.render.backend == "remote"
    assert request.render.views == ("front", "right")
    assert request.render.image_width == 512
    assert request.render.image_height == 256
    assert request.policy == {"reference_image_paths": ["reference.png"]}


def test_validation_agent_builds_request_from_single_input_string(
    tmp_path: Path,
) -> None:
    request = build_validation_request_from_inputs(
        task_description="Validate one asset.",
        inputs="asset.usda",
        output_dir="direct-run",
        base_dir=tmp_path,
    )

    assert request.inputs == ("asset.usda",)


def test_validation_agent_direct_inputs_reject_blank_task(tmp_path: Path) -> None:
    with pytest.raises(ValidationCliError, match="task must not be empty"):
        build_validation_request_from_inputs(
            task_description="  ",
            inputs="asset.usda",
            output_dir="direct-run",
            base_dir=tmp_path,
        )


def test_validation_agent_direct_inputs_default_output_uses_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    request = build_validation_request_from_inputs(
        task_description="Validate default output.",
        inputs="asset.usda",
    )

    assert request.project.working_dir == str(
        (tmp_path / ".validation-runs" / "validation-agent").resolve(strict=False)
    )


def test_validation_agent_config_relative_output_uses_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "validation.json"
    config_path.write_text(
        json.dumps(
            {
                "task_description": "Validate relative CLI output.",
                "inputs": ["asset.usda"],
                "requested_templates": ["render_valid"],
            }
        ),
        encoding="utf-8",
    )

    run = run_validation_from_config(
        config_path,
        output_dir="relative-run",
        dry_run=True,
    )

    assert run.output_dir == (tmp_path / "relative-run").resolve(strict=False)


def test_validation_agent_config_command_rejects_unknown_output_format() -> None:
    console = Console(record=True, width=120)

    with pytest.raises(typer.Exit) as exc_info:
        run_validation_cli_command(
            config_path=Path("validation.yaml"),
            output_dir=None,
            dry_run=False,
            fail_on_warn=False,
            output_format="xml",
            console=console,
        )

    assert exc_info.value.exit_code == VALIDATION_CLI_ERROR_EXIT_CODE
    assert "Unknown format: xml" in console.export_text()


def test_validation_agent_config_command_prints_json_and_propagates_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "world_understanding.validation.cli.run_validation_from_config",
        lambda *args, **kwargs: _fake_cli_run(
            verdict="fail",
            exit_code=VALIDATION_FAILURE_EXIT_CODE,
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_validation_cli_command(
            config_path=Path("validation.yaml"),
            output_dir=Path("out"),
            dry_run=True,
            template_overrides=("render_valid",),
            focus_prim_overrides=("/World",),
            fail_on_warn=True,
            output_format="json",
            console=console,
        )

    assert exc_info.value.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert '"verdict": "fail"' in console.export_text()


def test_validation_agent_config_command_prints_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "world_understanding.validation.cli.run_validation_from_config",
        lambda *args, **kwargs: _fake_cli_run(),
    )

    run_validation_cli_command(
        config_path=Path("validation.yaml"),
        output_dir=None,
        dry_run=True,
        fail_on_warn=False,
        output_format="text",
        console=console,
    )

    assert "Validation Agent verdict" in console.export_text()


def test_validation_agent_inputs_command_rejects_unknown_output_format() -> None:
    console = Console(record=True, width=120)

    with pytest.raises(typer.Exit) as exc_info:
        run_validation_inputs_cli_command(
            task_description="Validate inputs.",
            inputs=(Path("asset.usd"),),
            output_dir=None,
            dry_run=False,
            fail_on_warn=False,
            output_format="xml",
            console=console,
        )

    assert exc_info.value.exit_code == VALIDATION_CLI_ERROR_EXIT_CODE
    assert "Unknown format: xml" in console.export_text()


def test_validation_agent_inputs_command_prints_json_and_propagates_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "world_understanding.validation.cli.run_validation_from_inputs",
        lambda **kwargs: _fake_cli_run(
            verdict="fail",
            exit_code=VALIDATION_FAILURE_EXIT_CODE,
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        run_validation_inputs_cli_command(
            task_description="Validate inputs.",
            inputs=(Path("asset.usd"),),
            output_dir=Path("out"),
            dry_run=True,
            template_overrides=("render_valid",),
            focus_prim_overrides=("/World",),
            fail_on_warn=True,
            output_format="json",
            console=console,
            reference_image_paths=(Path("reference.png"),),
            render_backend="warp",
            render_views=("front",),
            render_image_width=64,
            render_image_height=32,
        )

    assert exc_info.value.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert '"verdict": "fail"' in console.export_text()


def test_validation_agent_inputs_command_prints_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(
        "world_understanding.validation.cli.run_validation_from_inputs",
        lambda **kwargs: _fake_cli_run(),
    )

    run_validation_inputs_cli_command(
        task_description="Validate inputs.",
        inputs=(Path("asset.usd"),),
        output_dir=None,
        dry_run=True,
        fail_on_warn=False,
        output_format="text",
        console=console,
    )

    assert "Validation Agent verdict" in console.export_text()


def test_validation_agent_summary_prints_issue_and_artifact_tables() -> None:
    issue = ValidationIssue(
        code="render.no_image",
        severity="fail",
        message="No render image was produced.",
        template_name="render_valid",
    )
    console = Console(record=True, width=120)

    print_validation_summary(
        _fake_cli_run(
            verdict="fail",
            exit_code=VALIDATION_FAILURE_EXIT_CODE,
            issues=(issue,),
        ),
        console=console,
    )

    output = console.export_text()
    assert "Validation Agent Issues" in output
    assert "render.no_image" in output
    assert "validation_result" in output


def test_validation_agent_direct_inputs_dry_run_writes_stable_artifacts(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    output_dir = tmp_path / "direct-run"

    run = run_validation_from_inputs(
        task_description="Validate render evidence.",
        inputs=("render.png",),
        output_dir="direct-run",
        dry_run=True,
        template_overrides=("render_valid",),
        base_dir=tmp_path,
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "planned"
    assert run.result.request.inputs == ("render.png",)
    assert run.result.request.project.working_dir == str(
        output_dir.resolve(strict=False)
    )
    assert run.result.plan.steps[0].template_name == "render_valid"
    assert (output_dir / "validation_request.json").is_file()
    assert (output_dir / "validation_plan.json").is_file()
    assert (output_dir / "validation_result.json").is_file()


def test_validation_agent_dry_run_writes_stable_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate render evidence.",
                "inputs:",
                "  - render.png",
                "project:",
                "  name: local-render",
                "  working_dir: run-from-config",
                "requested_templates:",
                "  - render_valid",
                "",
            )
        ),
        encoding="utf-8",
    )

    run = run_validation_from_config(
        config_path,
        dry_run=True,
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.schema_version == "1.0"
    assert run.result.verdict == "planned"
    assert run.result.plan.steps[0].template_name == "render_valid"

    output_dir = tmp_path / "run-from-config"
    request_path = output_dir / "validation_request.json"
    plan_path = output_dir / "validation_plan.json"
    result_path = output_dir / "validation_result.json"
    assert request_path.is_file()
    assert plan_path.is_file()
    assert result_path.is_file()

    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_data["schema_version"] == "1.0"
    assert result_data["verdict"] == "planned"
    assert result_data["request"]["project"]["name"] == "local-render"
    assert result_data["request"]["project"]["working_dir"] == str(
        output_dir.resolve(strict=False)
    )
    assert result_data["plan"]["steps"][0]["template_name"] == "render_valid"
    assert result_data["artifact_paths"]["validation_request"] == str(request_path)
    assert result_data["artifact_paths"]["validation_plan"] == str(plan_path)
    assert result_data["artifact_paths"]["validation_result"] == str(result_path)


def test_validation_agent_targeted_template_overrides_return_ci_status(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded by look_right")
    config_path = tmp_path / "validation.json"
    output_dir = tmp_path / "targeted-output"
    config_path.write_text(
        json.dumps(
            {
                "task_description": "Validate that the handle looks correct.",
                "inputs": ["render.png"],
                "requested_templates": ["render_valid"],
            }
        ),
        encoding="utf-8",
    )

    run = run_validation_from_config(
        config_path,
        output_dir=output_dir,
        template_overrides=("look_right",),
        focus_prim_overrides=("/World/Handle",),
        fail_on_warn=True,
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    result_data = json.loads(
        (output_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["verdict"] == "warn"
    assert [step["template_name"] for step in result_data["plan"]["steps"]] == [
        "look_right"
    ]
    assert result_data["plan"]["focus_prim_paths"] == ["/World/Handle"]
    assert result_data["request"]["requested_templates"] == ["look_right"]


def test_validation_agent_unknown_template_exits_as_cli_error(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded")
    config_path = tmp_path / "validation.json"
    config_path.write_text(
        json.dumps(
            {
                "task_description": "Validate render evidence.",
                "inputs": [str(image_path)],
                "requested_templates": ["not_a_template"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationCliError, match="Unknown validation template"):
        run_validation_from_config(config_path)


def test_validation_agent_non_mapping_config_reports_top_level_type(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "validation.yaml"
    config_path.write_text("- not-a-request\n", encoding="utf-8")

    with pytest.raises(ValidationCliError, match="got list"):
        run_validation_from_config(config_path)


def test_validation_agent_unreadable_config_reports_cli_error(tmp_path: Path) -> None:
    config_path = tmp_path / "validation.yaml"
    config_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ValidationCliError, match="Unable to read validation config"):
        run_validation_from_config(config_path)


def test_validation_agent_setup_errors_surface_as_cli_error(tmp_path: Path) -> None:
    request = ValidationRequest(
        task_description="Validate render evidence.",
        inputs=("render.png",),
        requested_templates=("render_valid",),
    )
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("", encoding="utf-8")

    with pytest.raises(ValidationCliError, match="Validation Agent run failed"):
        run_validation_request(
            request,
            config_base_dir=tmp_path,
            output_dir=output_file,
            dry_run=True,
        )


def test_validation_agent_preserves_scalar_render_hints() -> None:
    request = ValidationRequest(
        task_description="Validate render response.",
        inputs=("render.png",),
        render=ValidationRenderConfig(
            views="front",
            animation_frames="1:3",
            image_width=320,
            image_height=240,
        ),
    )

    policy = _scaffold_policy_from_request(request)

    assert policy["expected_cameras"] == ["front"]
    assert policy["expected_frames"] == ["1:3"]
    assert policy["render_image_width"] == 320
    assert policy["render_image_height"] == 240


def test_validation_agent_typed_render_backend_is_authoritative() -> None:
    request = ValidationRequest(
        task_description="Validate render response.",
        inputs=("render.png",),
        render=ValidationRenderConfig(backend=" OVRTX "),
        policy={"render_backend": "ovrtx"},
    )

    policy = _scaffold_policy_from_request(request)

    assert policy["render_backend"] == " OVRTX "


def test_validation_agent_rejects_conflicting_render_backend_fields() -> None:
    request = ValidationRequest(
        task_description="Validate render response.",
        inputs=("render.png",),
        render=ValidationRenderConfig(backend="ovrtx"),
        policy={"render_backend": "remote"},
    )

    with pytest.raises(ValidationCliError, match="Conflicting rendering backends"):
        _scaffold_policy_from_request(request)


def test_validation_agent_rejects_backend_conflict_before_output_side_effects(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate render response.",
        inputs=("render.png",),
        render=ValidationRenderConfig(backend="ovrtx"),
        policy={"render_backend": "remote"},
    )
    reused_output_dir = tmp_path / "reused"
    reused_output_dir.mkdir()
    existing_artifacts = {
        reused_output_dir / "validation_request.json": "existing request",
        reused_output_dir / "validation_plan.json": "existing plan",
        reused_output_dir / "validation_result.json": "existing result",
    }
    for artifact_path, content in existing_artifacts.items():
        artifact_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValidationCliError, match="Conflicting rendering backends"):
        run_validation_request(
            request,
            config_base_dir=tmp_path,
            output_dir=reused_output_dir,
        )

    assert {
        artifact_path: artifact_path.read_text(encoding="utf-8")
        for artifact_path in existing_artifacts
    } == existing_artifacts

    new_output_dir = tmp_path / "new"
    with pytest.raises(ValidationCliError, match="Conflicting rendering backends"):
        run_validation_request(
            request,
            config_base_dir=tmp_path,
            output_dir=new_output_dir,
        )

    assert not new_output_dir.exists()


def test_validation_agent_preserves_tuple_frames_and_render_metadata() -> None:
    request = ValidationRequest(
        task_description="Validate sampled render response.",
        inputs=("render.png",),
        render=ValidationRenderConfig(
            animation_frames=(1, 3, 5),
            metadata={"renderer": "warp"},
        ),
    )

    policy = _scaffold_policy_from_request(request)

    assert policy["expected_frames"] == [1, 3, 5]
    assert policy["render_metadata"] == {"renderer": "warp"}


def test_validation_agent_policy_path_edge_values(tmp_path: Path) -> None:
    absolute_render_dir = (tmp_path / "renders").resolve(strict=False)
    request = ValidationRequest(
        task_description="Validate policy path edges.",
        inputs=("asset.usd",),
        policy={
            "reference_image_paths": None,
            "current_image_paths": "current.png",
            "physical_behavior_refine_summary_path": None,
            "render_output_dir": absolute_render_dir,
        },
    )

    policy = _scaffold_policy_from_request(request, base_dir=tmp_path)

    assert policy["reference_image_paths"] is None
    assert policy["current_image_paths"] == [
        str((tmp_path / "current.png").resolve(strict=False))
    ]
    assert policy["physical_behavior_refine_summary_path"] is None
    assert policy["render_output_dir"] == str(absolute_render_dir)


def test_validation_agent_resolves_non_visual_policy_paths_from_config_dir(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate render and behavior evidence.",
        inputs=("asset.usd",),
        policy={
            "animation_frame_paths": ["frames/frame_001.png"],
            "time_sampled_usd_paths": ["sim/rollout.usda"],
            "behavior_video_paths": ["videos/rollout.mp4"],
            "simulation_json_paths": ["metrics/sim.json"],
            "physical_behavior_refine_summary_path": "summaries/main.json",
            "refine_summary_path": ["summaries/extra.json"],
            "physical_behavior_refine_output_dir": "physical-refine",
            "physics_refine_output_dir": "physics-refine",
            "render_output_dir": "renders",
            "refine_output_dir": "refine",
        },
    )

    policy = _scaffold_policy_from_request(request, base_dir=tmp_path)

    assert policy["animation_frame_paths"] == [
        str((tmp_path / "frames" / "frame_001.png").resolve())
    ]
    assert policy["time_sampled_usd_paths"] == [
        str((tmp_path / "sim" / "rollout.usda").resolve())
    ]
    assert policy["behavior_video_paths"] == [
        str((tmp_path / "videos" / "rollout.mp4").resolve())
    ]
    assert policy["simulation_json_paths"] == [
        str((tmp_path / "metrics" / "sim.json").resolve())
    ]
    assert policy["physical_behavior_refine_summary_path"] == str(
        (tmp_path / "summaries" / "main.json").resolve()
    )
    assert policy["refine_summary_path"] == [
        str((tmp_path / "summaries" / "extra.json").resolve())
    ]
    assert policy["physical_behavior_refine_output_dir"] == str(
        (tmp_path / "physical-refine").resolve()
    )
    assert policy["physics_refine_output_dir"] == str(
        (tmp_path / "physics-refine").resolve()
    )
    assert policy["render_output_dir"] == str((tmp_path / "renders").resolve())
    assert policy["refine_output_dir"] == str((tmp_path / "refine").resolve())


def test_validation_agent_public_example_configs_load() -> None:
    examples_dir = REPO_ROOT / "apps" / "validation_agent" / "examples" / "configs"

    requests = {
        path.stem: load_validation_request_config(path)
        for path in sorted(examples_dir.glob("*.yaml"))
    }

    assert set(requests) == {
        "electricians_toolbox_visual",
        "steel_scaffold_behavior_refine_summary",
        "steel_scaffold_known_negative_physics",
    }
    visual = requests["electricians_toolbox_visual"]
    assert visual.requested_templates == ("render_valid", "look_right")
    assert visual.policy["look_right_vlm"] == {
        "backend": "nim",
        "model": "google/gemma-4-31b-it",
    }
    known_negative = requests["steel_scaffold_known_negative_physics"]
    assert known_negative.policy["expected_issue_codes"] == [
        "physics.no_physics_scene",
        "physics.no_rigid_bodies",
    ]


def test_validation_agent_public_behavior_example_runs(tmp_path: Path) -> None:
    config_path = (
        REPO_ROOT
        / "apps"
        / "validation_agent"
        / "examples"
        / "configs"
        / "steel_scaffold_behavior_refine_summary.yaml"
    )

    run = run_validation_from_config(config_path, output_dir=tmp_path / "run")

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "pass"
    behavior = run.result.template_results[0]
    assert behavior.template_name == "physical_behavior"
    assert behavior.status == "passed"
    assert behavior.metrics["judge_decision"] == "approve"


def test_validation_agent_required_behavior_at_iteration_cap_fails(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "behavior.mp4"
    video_path.write_bytes(b"not decoded by scaffold")
    summary_path = tmp_path / "refine_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "termination_reason": "max_iterations",
                "final_iteration": 1,
                "iterations": [
                    {
                        "iteration": 1,
                        "judge_decision": "continue",
                        "judge_score": 0.3,
                        "judge_reasoning": "The rollout still tips over.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    request = ValidationRequest(
        task_description="Validate required rollout behavior.",
        inputs=(str(video_path),),
        requested_templates=("physical_behavior",),
        policy={
            "behavior_evidence_required": True,
            "physical_behavior_refine_summary_path": str(summary_path),
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    behavior = run.result.template_results[0]
    assert behavior.status == "failed"
    assert behavior.issues[0].code == "physics.behavior_needs_refinement"
    assert behavior.issues[0].severity == "fail"
    assert behavior.metrics["termination_reason"] == "max_iterations"
    assert behavior.metrics["judge_decision"] == "continue"
    result_data = json.loads(
        (tmp_path / "run" / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["verdict"] == "fail"
    assert result_data["template_results"][0]["status"] == "failed"
    assert result_data["issues"][0]["severity"] == "fail"


def test_validation_agent_rejects_malformed_policy_path_sequences(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate malformed image evidence.",
        inputs=("asset.usd",),
        policy={"reference_image_paths": ["refs/good.png", 7]},
    )

    with pytest.raises(
        ValidationCliError,
        match=r"policy\.reference_image_paths\[1\] must be a path string",
    ):
        _scaffold_policy_from_request(request, base_dir=tmp_path)


def test_validation_agent_rejects_malformed_focused_image_paths(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate malformed focused evidence.",
        inputs=("asset.usd",),
        policy={"focused_image_paths": {"/World/Handle": ["focus.png", 7]}},
    )

    with pytest.raises(
        ValidationCliError,
        match=r"policy\.focused_image_paths\['/World/Handle'\]\[1\]",
    ):
        _scaffold_policy_from_request(request, base_dir=tmp_path)


def test_validation_agent_rejects_non_mapping_focused_image_paths(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate malformed focused evidence.",
        inputs=("asset.usd",),
        policy={"focused_image_paths": ["focus.png"]},
    )

    with pytest.raises(
        ValidationCliError,
        match=r"policy\.focused_image_paths must be a mapping",
    ):
        _scaffold_policy_from_request(request, base_dir=tmp_path)


def test_validation_agent_allows_current_directory_policy_value(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate current directory path.",
        inputs=("asset.usd",),
        policy={"render_output_dir": "."},
    )

    policy = _scaffold_policy_from_request(request, base_dir=tmp_path)

    assert policy["render_output_dir"] == str(tmp_path.resolve())


def test_validation_agent_rejects_empty_string_policy_value(
    tmp_path: Path,
) -> None:
    request = ValidationRequest(
        task_description="Validate empty path string.",
        inputs=("asset.usd",),
        policy={"render_output_dir": ""},
    )

    with pytest.raises(
        ValidationCliError,
        match=r"policy\.render_output_dir path must not be empty",
    ):
        _scaffold_policy_from_request(request, base_dir=tmp_path)


def test_validation_agent_renders_usd_inputs_for_visual_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate that the generated asset looks correct.",
                "inputs:",
                "  - asset.usda",
                "project:",
                "  working_dir: run",
                "render:",
                "  views:",
                "    - front",
                "  image_width: 128",
                "  image_height: 96",
                "requested_templates:",
                "  - render_valid",
                "  - look_right",
                "policy:",
                "  look_right_response: |",
                "    Critique: The render matches the requested asset.",
                "    Score: 8",
                "    Decision: PASS",
                "    Issue Codes: none",
                "",
            )
        ),
        encoding="utf-8",
    )

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        assert tuple(usd_paths) == (usd_path.resolve(),)
        assert policy["render_image_width"] == 128
        assert policy["render_image_height"] == 96
        render_dir = Path(working_dir) / "renders" / "asset"
        render_dir.mkdir(parents=True)
        render_path = _write_valid_image(render_dir / "asset_front_0000.png")
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "remote",
                "status": "completed",
                "results": [
                    {
                        "camera": "front",
                        "camera_path": "/ValidationAgentCameras/front",
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )

    run = run_validation_from_config(config_path)
    assert run.exit_code == PASS_EXIT_CODE
    result_path = tmp_path / "run" / "validation_result.json"
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    rendered_path = str(tmp_path / "run" / "renders" / "asset" / "asset_front_0000.png")
    assert result_data["verdict"] == "pass"
    assert result_data["template_results"][0]["evidence"]["image_paths"] == [
        rendered_path
    ]
    assert (
        result_data["template_results"][0]["metadata"]["runtime_render"]["status"]
        == "completed"
    )
    assert result_data["template_results"][1]["evidence"]["image_caption_pairs"] == [
        {"caption": "Current Render Output - View 1:", "path": rendered_path},
    ]


def test_validation_agent_unknown_render_backend_fails_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Validate render evidence.",
        inputs=(str(usd_path),),
        render=ValidationRenderConfig(backend="typo"),
        requested_templates=("render_valid",),
    )
    monkeypatch.setattr(
        "world_understanding.validation.usd_rendering."
        "_create_rendering_backend_from_factory",
        lambda *_args, **_kwargs: pytest.fail("factory must not be called"),
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=output_dir,
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    assert "render.backend_unknown" in {issue.code for issue in run.result.issues}
    runtime_render = run.result.template_results[0].metadata["runtime_render"]
    assert runtime_render["status"] == "failed"
    assert runtime_render["backend"] == "typo"
    assert not (output_dir / "renders").exists()


def test_validation_agent_live_look_right_mock_vlm_writes_invocation(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 31, 31), fill=(255, 0, 0))
    draw.rectangle((32, 0, 63, 31), fill=(0, 255, 0))
    draw.rectangle((0, 32, 31, 63), fill=(0, 0, 255))
    draw.rectangle((32, 32, 63, 63), fill=(255, 255, 0))
    image.save(image_path)
    request = ValidationRequest(
        task_description="Validate render evidence with a live mock VLM.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={"look_right_vlm": {"backend": "mock"}},
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "warn"
    look_right = run.result.template_results[1]
    assert look_right.template_name == "look_right"
    assert look_right.status == "warn"
    assert look_right.metadata["vlm_invoked"] is True
    assert look_right.evidence["judge_invocation"]["backend_name"] == "mock"
    assert look_right.evidence["judgment"]["raw_response"].startswith("<reasoning>")


def test_validation_agent_resolves_policy_image_paths_from_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    reference_path = _write_valid_image(tmp_path / "reference.png")
    focus_path = _write_valid_image(tmp_path / "focus.png")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task_description: Validate the render against a reference.",
                "inputs:",
                "  - render.png",
                "requested_templates:",
                "  - render_valid",
                "  - look_right",
                "policy:",
                "  reference_image_paths:",
                "    - reference.png",
                "  look_right_vlm:",
                "    backend: fake",
                "    model: fake-vlm",
                "  focused_image_paths:",
                "    /World/Handle:",
                "      - focus.png",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    class FakeVLM:
        backend_name = "fake"
        model_name = "fake-vlm"

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "Critique: match\nScore: 8\nDecision: PASS\nIssue Codes: none"

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        lambda **_: FakeVLM(),
    )

    run = run_validation_from_config(config_path, output_dir=tmp_path / "run")

    assert run.exit_code == PASS_EXIT_CODE
    assert calls
    assert calls[0]["image_caption_pairs"] == [
        ("Reference Image 1:", str(reference_path.resolve())),
        ("Current Asset Evidence - View 1:", str(image_path.resolve())),
        (
            "Focused Asset Evidence - /World/Handle - View 1:",
            str(focus_path.resolve()),
        ),
    ]


def test_validation_agent_expected_failure_policy_reports_known_negative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_physics_sane_adapter(*args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "template": "physics_sane",
            "status": "completed",
            "verdict": "fail",
            "passed": False,
            "issues": [
                {
                    "code": "physics.no_physics_scene",
                    "severity": "fail",
                    "message": "No physics scene.",
                    "subject": str(usd_path),
                    "details": {},
                },
                {
                    "code": "physics.no_rigid_bodies",
                    "severity": "fail",
                    "message": "No rigid bodies.",
                    "subject": str(usd_path),
                    "details": {},
                },
            ],
            "metrics": {"opened": True},
            "evidence": {"usd_path": str(usd_path)},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.run_physics_sane_adapter",
        fake_physics_sane_adapter,
    )
    request = ValidationRequest(
        task_description="Validate known-negative public PhysX asset.",
        inputs=(str(usd_path),),
        requested_templates=("physics_sane",),
        policy={
            "expect_physics": True,
            "expected_verdict": "fail",
            "expected_issue_codes": [
                "physics.no_physics_scene",
                "physics.no_rigid_bodies",
            ],
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "warn"
    assert run.result.metadata["expected_result"]["matched"] is True
    physics_sane = run.result.template_results[0]
    assert physics_sane.status == "warn"
    assert physics_sane.metadata["expected_failure_matched"] is True
    assert {issue.severity for issue in physics_sane.issues} == {"warn"}
    assert {issue.details["original_severity"] for issue in physics_sane.issues} == {
        "fail"
    }
    result_data = json.loads(
        (tmp_path / "run" / "validation_result.json").read_text(encoding="utf-8")
    )
    assert result_data["metadata"]["expected_result"]["matched"] is True
    assert result_data["verdict"] == "warn"


def test_validation_agent_expected_failure_mismatch_stays_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_physics_sane_adapter(*args: object, **kwargs: object) -> dict[str, Any]:
        return {
            "template": "physics_sane",
            "status": "completed",
            "verdict": "fail",
            "passed": False,
            "issues": [
                {
                    "code": "physics.no_physics_scene",
                    "severity": "fail",
                    "message": "No physics scene.",
                    "subject": str(usd_path),
                    "details": {},
                },
                {
                    "code": "physics.no_rigid_bodies",
                    "severity": "fail",
                    "message": "No rigid bodies.",
                    "subject": str(usd_path),
                    "details": {},
                },
            ],
            "metrics": {"opened": True},
            "evidence": {"usd_path": str(usd_path)},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.run_physics_sane_adapter",
        fake_physics_sane_adapter,
    )
    request = ValidationRequest(
        task_description="Validate known-negative public PhysX asset.",
        inputs=(str(usd_path),),
        requested_templates=("physics_sane",),
        policy={
            "expect_physics": True,
            "expected_verdict": "fail",
            "expected_issue_codes": ["physics.no_physics_scene"],
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    expected_result = run.result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "issue_code_mismatch"
    assert expected_result["unexpected_issue_codes"] == ["physics.no_rigid_bodies"]


def test_validation_agent_expected_failure_policy_accepts_single_code_string() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": "physics.no_physics_scene",
            },
        )
    )

    assert result.verdict == "warn"
    assert result.metadata["expected_result"]["matched"] is True
    assert result.issues[0].details["expected_failure"] is True


def test_validation_agent_expected_failure_issue_ignores_non_matching_issue() -> None:
    issue = ValidationIssue(
        code="render.no_image",
        severity="warn",
        message="No render image was produced.",
    )

    assert (
        _downgrade_expected_failure_issue(issue, {"physics.no_physics_scene"}) is issue
    )


def test_validation_agent_expected_failure_empty_code_string_is_missing() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": " ",
            },
        )
    )

    expected_result = result.metadata["expected_result"]
    assert result.verdict == "fail"
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "expected_issue_codes_missing"


def test_validation_agent_expected_failure_missing_codes_stays_failed() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(policy={"expected_verdict": "fail"})
    )

    assert result.verdict == "fail"
    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "expected_issue_codes_missing"
    assert result.issues[0].severity == "fail"
    assert result.issues[-1].code == "validation.expected_result_mismatch"


def test_validation_agent_expected_failure_missing_observed_fails_closed() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            verdict="pass",
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
            issues=(),
            template_results=(
                ValidationTemplateResult(
                    template_name="physics_sane",
                    status="passed",
                ),
            ),
        )
    )

    assert result.verdict == "fail"
    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "issue_code_mismatch"
    assert expected_result["missing_expected_issue_codes"] == [
        "physics.no_physics_scene"
    ]
    assert result.issues[-1].code == "validation.expected_result_mismatch"


def test_validation_agent_expected_failure_planned_result_is_unchanged() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            verdict="planned",
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
            issues=(),
            template_results=(),
        )
    )

    assert result.verdict == "planned"
    assert result.issues == ()
    assert "expected_result" not in result.metadata


def test_validation_agent_expected_failure_verdict_mismatch_not_matched() -> None:
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            verdict="warn",
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
        )
    )

    assert result.verdict == "fail"
    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "verdict_mismatch"
    assert result.issues[0].severity == "fail"


def test_validation_agent_expected_failure_unexpected_failed_template_blocks() -> None:
    expected_issue = ValidationIssue(
        code="physics.no_physics_scene",
        severity="fail",
        message="No physics scene.",
        template_name="physics_sane",
    )
    warning_issue = ValidationIssue(
        code="render.no_image",
        severity="warn",
        message="Render evidence missing.",
        template_name="render_valid",
    )
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
            issues=(expected_issue,),
            template_results=(
                ValidationTemplateResult(
                    template_name="physics_sane",
                    status="failed",
                    issues=(expected_issue,),
                ),
                ValidationTemplateResult(
                    template_name="render_valid",
                    status="failed",
                    issues=(warning_issue,),
                ),
            ),
        )
    )

    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "unexpected_failed_template"
    assert result.template_results[1].status == "failed"


def test_validation_agent_expected_failure_mixed_failed_template_blocks() -> None:
    expected_issue = ValidationIssue(
        code="physics.no_physics_scene",
        severity="fail",
        message="No physics scene.",
        template_name="physics_sane",
    )
    unexpected_issue = ValidationIssue(
        code="physics.no_rigid_bodies",
        severity="fail",
        message="No rigid bodies.",
        template_name="physics_sane",
    )
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
            issues=(expected_issue,),
            template_results=(
                ValidationTemplateResult(
                    template_name="physics_sane",
                    status="failed",
                    issues=(expected_issue, unexpected_issue),
                ),
            ),
        )
    )

    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "unexpected_failed_template"
    assert result.issues[-1].code == "validation.expected_result_mismatch"


def test_validation_agent_expected_failure_template_error_blocks() -> None:
    expected_issue = ValidationIssue(
        code="physics.no_physics_scene",
        severity="fail",
        message="No physics scene.",
        template_name="physics_sane",
    )
    result = _apply_expected_result_policy(
        _expected_failure_policy_result(
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["physics.no_physics_scene"],
            },
            issues=(expected_issue,),
            template_results=(
                ValidationTemplateResult(
                    template_name="physics_sane",
                    status="failed",
                    issues=(expected_issue,),
                ),
                ValidationTemplateResult(
                    template_name="render_valid",
                    status="error",
                ),
            ),
        )
    )

    expected_result = result.metadata["expected_result"]
    assert expected_result["matched"] is False
    assert expected_result["reason"] == "template_error"
    assert result.template_results[1].status == "error"


def test_validation_agent_gate_policy_blocks_unavailable_dependency(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with a required visual judge.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={"gate_policy": {"dependency_unavailable": "block"}},
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    assert run.result.recommended_action is not None
    assert [issue.code for issue in run.result.issues] == [
        "visual.judge_unavailable",
        "validation.dependency_unavailable",
    ]
    assert run.result.issues[-1].severity == "fail"
    assert run.result.issues[-1].details["blocked_issue_codes"] == [
        "visual.judge_unavailable"
    ]
    assert run.result.metadata["gate_policy_evaluation"] == {
        "blocked": True,
        "reason": "dependency_unavailable",
        "blocked_issue_codes": ["visual.judge_unavailable"],
    }

    result_path = tmp_path / "run" / "validation_result.json"
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_data["verdict"] == "fail"
    assert result_data["issues"][-1]["code"] == "validation.dependency_unavailable"


def test_validation_agent_gate_policy_blocks_look_right_only_renderer_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        return {
            "status": "unavailable",
            "backend": "remote",
            "image_paths": [],
            "render_response": None,
            "render_output_dir": None,
            "issues": [
                {
                    "code": "render.renderer_unavailable",
                    "severity": "warn",
                    "message": "Set RENDER_ENDPOINT before rendering USD inputs.",
                    "details": {"required_env": ["RENDER_ENDPOINT"]},
                }
            ],
            "metadata": {"base_url_configured": False},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = ValidationRequest(
        task_description="Validate render evidence with a required renderer.",
        inputs=(str(usd_path),),
        requested_templates=("look_right",),
        policy={
            "gate_policy": {"dependency_unavailable": "block"},
            "look_right_response": "\n".join(
                (
                    "Critique: This should not be used without current evidence.",
                    "Score: 9",
                    "Decision: PASS",
                    "Issue Codes: none",
                )
            ),
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    assert [issue.code for issue in run.result.issues] == [
        "visual.evidence_missing",
        "render.renderer_unavailable",
        "validation.dependency_unavailable",
    ]
    assert run.result.issues[-1].details["blocked_issue_codes"] == [
        "render.renderer_unavailable"
    ]


def test_validation_agent_gate_policy_blocks_renderer_after_live_look_right_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    image_path = _write_valid_image(tmp_path / "render.png")
    fake_vlm = _FakeLookRightVLM(
        "\n".join(
            (
                "Critique: The supplied image looks consistent with the prompt.",
                "Score: 9",
                "Decision: PASS",
                "Issue Codes: none",
            )
        )
    )
    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.create_vlm",
        lambda **_: fake_vlm,
    )
    request = ValidationRequest(
        task_description="Validate render evidence with a required renderer.",
        inputs=(str(usd_path), str(image_path)),
        requested_templates=("look_right",),
        policy={
            "gate_policy": {"dependency_unavailable": "block"},
            "look_right_vlm": {"backend": "fake"},
            "runtime_render": {
                "status": "unavailable",
                "backend": "remote",
                "image_paths": [],
                "render_response": None,
                "render_output_dir": None,
                "issues": [
                    {
                        "code": "render.renderer_unavailable",
                        "severity": "warn",
                        "message": "Set RENDER_ENDPOINT before rendering USD inputs.",
                        "details": {"required_env": ["RENDER_ENDPOINT"]},
                    }
                ],
                "metadata": {"base_url_configured": False},
            },
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert len(fake_vlm.calls) == 1
    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    assert [issue.code for issue in run.result.issues] == [
        "render.renderer_unavailable",
        "validation.dependency_unavailable",
    ]
    assert run.result.issues[-1].details["blocked_issue_codes"] == [
        "render.renderer_unavailable"
    ]


def test_validation_agent_gate_policy_preserves_existing_recommended_action() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence with multiple remediation hints.",
        inputs=("render.png",),
        requested_templates=("render_valid", "look_right"),
        policy={"gate_policy": {"dependency_unavailable": "block"}},
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="look_right",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="visual.judge_unavailable",
                severity="warn",
                message="The visual judge is unavailable.",
            ),
        ),
        recommended_action="Fix the authored physics metadata.",
    )

    gated = _apply_gate_policy(result, request=request)

    assert gated.recommended_action is not None
    assert DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION in gated.recommended_action
    assert "Fix the authored physics metadata." in gated.recommended_action


def test_validation_agent_gate_policy_leaves_existing_dependency_action_unchanged() -> (
    None
):
    existing = (
        f"{DEPENDENCY_UNAVAILABLE_RECOMMENDED_ACTION} Existing recommendation: "
        "review the fixture."
    )

    assert _dependency_unavailable_recommended_action(existing) == existing


def test_validation_agent_gate_policy_string_sequence_allows_none() -> None:
    assert (
        _gate_policy_string_sequence(
            None,
            field_name="gate_policy.dependency_unavailable_issue_codes",
        )
        == ()
    )


def test_validation_agent_gate_policy_honors_dependency_issue_override(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with a scoped dependency gate.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {
                "dependency_unavailable": "block",
                "dependency_unavailable_issue_codes": ["render.renderer_unavailable"],
            }
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "warn"
    assert [issue.code for issue in run.result.issues] == ["visual.judge_unavailable"]
    assert run.result.recommended_action is None
    assert "gate_policy_evaluation" not in run.result.metadata


def test_validation_agent_gate_policy_accepts_scalar_dependency_issue_code(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with a scalar dependency gate.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {
                "dependency_unavailable": "block",
                "dependency_unavailable_issue_codes": "visual.judge_unavailable",
            }
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == VALIDATION_FAILURE_EXIT_CODE
    assert run.result.verdict == "fail"
    assert [issue.code for issue in run.result.issues] == [
        "visual.judge_unavailable",
        "validation.dependency_unavailable",
    ]


def test_validation_agent_gate_policy_blocks_behavior_refiner_unavailable() -> None:
    request = ValidationRequest(
        task_description="Validate behavior evidence with a required refiner.",
        inputs=("behavior.mp4",),
        requested_templates=("physical_behavior",),
        policy={"gate_policy": {"dependency_unavailable": "block"}},
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="physical_behavior",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="physics.behavior_refiner_unavailable",
                severity="warn",
                message="The behavior refiner is unavailable.",
            ),
        ),
    )

    gated = _apply_gate_policy(result, request=request)

    assert gated.verdict == "fail"
    assert gated.issues[-1].code == "validation.dependency_unavailable"
    assert gated.issues[-1].details["blocked_issue_codes"] == [
        "physics.behavior_refiner_unavailable"
    ]


def test_validation_agent_gate_policy_rejects_bad_dependency_issue_codes() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence with malformed dependency codes.",
        inputs=("render.png",),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {
                "dependency_unavailable": "block",
                "dependency_unavailable_issue_codes": [123],
            }
        },
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="look_right",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="visual.judge_unavailable",
                severity="warn",
                message="The visual judge is unavailable.",
            ),
        ),
    )

    with pytest.raises(
        ValidationCliError,
        match="dependency_unavailable_issue_codes must contain only strings",
    ):
        _apply_gate_policy(result, request=request)


def test_validation_agent_gate_policy_rejects_invalid_dependency_issue_code() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence with malformed dependency codes.",
        inputs=("render.png",),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {
                "dependency_unavailable": "block",
                "dependency_unavailable_issue_codes": ["not-a-valid-code"],
            }
        },
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="look_right",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="visual.judge_unavailable",
                severity="warn",
                message="The visual judge is unavailable.",
            ),
        ),
    )

    with pytest.raises(
        ValidationCliError,
        match="contains invalid issue code",
    ):
        _apply_gate_policy(result, request=request)


def test_validation_agent_gate_policy_empty_dependency_override_blocks_nothing(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with no watched dependencies.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {
                "dependency_unavailable": "block",
                "dependency_unavailable_issue_codes": [],
            }
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "warn"
    assert [issue.code for issue in run.result.issues] == ["visual.judge_unavailable"]
    assert run.result.recommended_action is None
    assert "gate_policy_evaluation" not in run.result.metadata


def test_validation_agent_gate_policy_block_value_is_case_insensitive() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence with a trimmed gate value.",
        inputs=("render.png",),
        requested_templates=("render_valid", "look_right"),
        policy={"gate_policy": {"dependency_unavailable": "  BLOCK  "}},
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="look_right",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="visual.judge_unavailable",
                severity="warn",
                message="The visual judge is unavailable.",
            ),
        ),
    )

    gated = _apply_gate_policy(result, request=request)

    assert gated.verdict == "fail"
    assert gated.issues[-1].code == "validation.dependency_unavailable"


def test_validation_agent_gate_policy_reports_duplicate_blocked_issue_count() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence with duplicate dependency issues.",
        inputs=("asset.usda",),
        requested_templates=("look_right",),
        policy={"gate_policy": {"dependency_unavailable": "block"}},
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="look_right",
                    reason="requested",
                ),
            )
        ),
        issues=(
            ValidationIssue(
                code="render.renderer_unavailable",
                severity="warn",
                message="The renderer is unavailable for the first asset.",
            ),
            ValidationIssue(
                code="render.renderer_unavailable",
                severity="warn",
                message="The renderer is unavailable for the second asset.",
            ),
        ),
    )

    gated = _apply_gate_policy(result, request=request)

    assert gated.issues[-1].details["blocked_issue_codes"] == [
        "render.renderer_unavailable"
    ]
    assert gated.issues[-1].details["blocked_issue_count"] == 2


def test_validation_agent_gate_policy_requires_documented_block_value(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with a non-string gate value.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={"gate_policy": {"dependency_unavailable": True}},
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "warn"
    assert [issue.code for issue in run.result.issues] == ["visual.judge_unavailable"]
    assert run.result.recommended_action is None
    assert "gate_policy_evaluation" not in run.result.metadata


def test_validation_agent_gate_policy_does_not_block_without_dependency_issue(
    tmp_path: Path,
) -> None:
    image_path = _write_valid_image(tmp_path / "render.png")
    request = ValidationRequest(
        task_description="Validate render evidence with a passing visual judge.",
        inputs=(str(image_path),),
        requested_templates=("render_valid", "look_right"),
        policy={
            "gate_policy": {"dependency_unavailable": "block"},
            "look_right_response": "\n".join(
                (
                    "Critique: The render matches the requested asset.",
                    "Score: 9",
                    "Decision: PASS",
                    "Issue Codes: none",
                )
            ),
        },
    )

    run = run_validation_request(
        request,
        config_base_dir=tmp_path,
        output_dir=tmp_path / "run",
    )

    assert run.exit_code == PASS_EXIT_CODE
    assert run.result.verdict == "pass"
    assert run.result.issues == ()
    assert run.result.recommended_action is None
    assert "gate_policy_evaluation" not in run.result.metadata


def test_validation_exit_code_policy() -> None:
    assert validation_exit_code("pass") == PASS_EXIT_CODE
    assert validation_exit_code("planned") == PASS_EXIT_CODE
    assert validation_exit_code("warn") == PASS_EXIT_CODE
    assert (
        validation_exit_code("warn", fail_on_warn=True) == VALIDATION_FAILURE_EXIT_CODE
    )
    assert validation_exit_code("fail") == VALIDATION_FAILURE_EXIT_CODE
    assert validation_exit_code("needs_refinement") == VALIDATION_FAILURE_EXIT_CODE
