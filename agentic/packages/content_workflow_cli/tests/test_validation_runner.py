# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Codex and Claude request-path tests for Validation planning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from world_understanding.validation import ValidationProject, ValidationRequest

from content_workflow_cli import validation_runner
from content_workflow_cli.runner import (
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    _child_workflow_name,
)


def _stage_fake_validation_skill(repo_root: Path, run_dir: Path) -> None:
    source = (
        repo_root / "agentic" / ".agents" / "skills" / "content-workflow-validation"
    )
    source.mkdir(parents=True, exist_ok=True)
    source.joinpath("SKILL.md").write_text("# Validation\n", encoding="utf-8")
    for discovery_root in (".agents", ".claude"):
        destination = (
            run_dir / discovery_root / "skills" / "content-workflow-validation"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    plugin_metadata_root = run_dir / ".claude" / ".claude-plugin"
    plugin_metadata_root.mkdir()
    (plugin_metadata_root / "plugin.json").write_text(
        json.dumps(
            validation_runner.CLAUDE_CLI_TRUSTED_SKILLS_PLUGIN_MANIFEST,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_plan_patch(*, producer: str = "codex") -> dict[str, Any]:
    return {
        "schema_version": (
            "content-agent-workflows.validation-coordinator-plan-patch.v1"
        ),
        "plan_id": "p",
        "preparation_digest": "0" * 64,
        "producer": producer,
        "child_session_id": "fake-child-session",
        "decision_only": True,
        "source_mutated": False,
        "execution_performed": False,
        "publication_performed": False,
        "claims": [
            {
                "claim_id": "claim",
                "statement": "Validate the source.",
                "acceptance_criteria": ["The check passes."],
                "check_ids": ["check"],
            }
        ],
        "selected_checks": [
            {
                "check_id": "check",
                "capability_id": "validation.check",
                "template_name": "check",
                "rule_id": "validation.check",
                "targets": ["/source"],
                "evidence_requirements": ["result"],
            }
        ],
    }


def test_validation_child_schema_is_strict_and_parameter_bounded() -> None:
    schema = validation_runner._child_final_schema()

    def assert_strict(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                assert_strict(item)
            return
        if not isinstance(value, dict):
            return
        assert "default" not in value
        properties = value.get("properties")
        if value.get("type") == "object" and isinstance(properties, dict):
            assert value.get("additionalProperties") is False
            assert value.get("required") == list(properties)
        for child in value.values():
            assert_strict(child)

    assert_strict(schema)
    definitions = schema["$defs"]
    assert "JsonValue" not in definitions
    parameters = definitions["ValidationCoordinatorSelectedCheck"]["properties"][
        "parameters"
    ]
    assert parameters["anyOf"] == [
        {
            "additionalProperties": False,
            "properties": {},
            "required": [],
            "type": "object",
        },
        {
            "additionalProperties": False,
            "properties": {
                "evidence_paths": {
                    "items": {"type": "string"},
                    "type": "array",
                }
            },
            "required": ["evidence_paths"],
            "type": "object",
        },
    ]


def test_validation_discards_empty_claude_cli_project_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "run" / validation_runner.CLAUDE_CLI_PROJECT_SCRATCH
    scratch.mkdir(parents=True)
    scratch.chmod(0o700)

    validation_runner._discard_empty_claude_cli_project_scratch(tmp_path / "run")

    assert not scratch.exists()


def test_validation_accepts_rebuildable_child_cache_directory(tmp_path: Path) -> None:
    cache_root = tmp_path / "run" / validation_runner.CHILD_CACHE_RELPATH
    cache_entry = cache_root / "claude-cli-nodejs" / "cache.json"
    cache_entry.parent.mkdir(parents=True)
    cache_entry.write_text("{}\n", encoding="utf-8")

    validation_runner._validate_rebuildable_child_cache_root(tmp_path / "run")

    assert cache_entry.is_file()


def test_validation_rejects_non_directory_child_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "run" / validation_runner.CHILD_CACHE_RELPATH
    cache_root.parent.mkdir()
    cache_root.write_text("not a cache directory\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="rebuildable cache root.*[.]cache"):
        validation_runner._validate_rebuildable_child_cache_root(tmp_path / "run")


@pytest.mark.parametrize("scratch_kind", ("file", "symlink", "nonempty"))
def test_validation_rejects_untrusted_claude_cli_project_scratch(
    tmp_path: Path,
    scratch_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    scratch = run_dir / validation_runner.CLAUDE_CLI_PROJECT_SCRATCH
    scratch.parent.mkdir(parents=True)
    if scratch_kind == "file":
        scratch.write_text("untrusted\n", encoding="utf-8")
    elif scratch_kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            scratch.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
    else:
        scratch.mkdir()
        (scratch / "payload").write_text("untrusted\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"trusted staged skill tree.*cc-writes"):
        validation_runner._discard_empty_claude_cli_project_scratch(run_dir)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory symlink regression")
def test_validation_rejects_symlinked_claude_cli_project_parent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    (outside / ".cc-writes").mkdir(parents=True)
    (run_dir / ".claude").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match=r"trusted staged skill tree.*cc-writes"):
        validation_runner._discard_empty_claude_cli_project_scratch(run_dir)

    assert (outside / ".cc-writes").is_dir()


@pytest.mark.skipif(os.name != "nt", reason="requires a Windows directory junction")
def test_validation_rejects_junctioned_claude_cli_project_parent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    (outside / ".cc-writes").mkdir(parents=True)
    subprocess.run(
        (
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(run_dir / ".claude"),
            str(outside),
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match=r"trusted staged skill tree.*cc-writes"):
        validation_runner._discard_empty_claude_cli_project_scratch(run_dir)

    assert (outside / ".cc-writes").is_dir()


@pytest.mark.parametrize("runner", (RUNNER_CODEX, RUNNER_CLAUDE))
def test_validation_runner_uses_one_decision_only_child_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    task_description = (
        "Author plan_id validation-exact-v1 with check_id physics-static "
        "followed by check_id render-evidence."
    )
    request = ValidationRequest(
        task_description=task_description,
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
        policy={
            "visual_evidence_mode": "canonical_usd",
            "look_right_vlm": {"api_key_env": "${VALIDATION_JUDGE_KEY}"},
        },
    )
    child_calls: list[dict[str, Any]] = []
    acceptance_calls: list[dict[str, Any]] = []

    def fake_child(**kwargs: Any) -> int:
        child_calls.append(kwargs)
        _stage_fake_validation_skill(kwargs["config"].repo_root, kwargs["run_dir"])
        cache_entry = (
            kwargs["run_dir"]
            / validation_runner.CHILD_CACHE_RELPATH
            / "provider-cache"
            / "entry.json"
        )
        cache_entry.parent.mkdir(parents=True)
        cache_entry.write_text("{}\n", encoding="utf-8")
        kwargs["child_output_path"].write_text("planner output\n", encoding="utf-8")
        patch_path = kwargs["run_dir"] / "validation_coordinator_plan_patch.json"
        raw_dir = kwargs["run_dir"] / "raw"
        for name, content in (
            ("validation_planner_request.json", "{}\n"),
            ("validation_planner_items.json", "[]\n"),
            ("validation_planner_result.json", "{}\n"),
            ("validation_planner_observable_events.jsonl", ""),
            ("validation_planner_launch_descriptor.json", "{}\n"),
        ):
            (raw_dir / name).write_text(content, encoding="utf-8")
        kwargs["child_final_path"].write_text(
            json.dumps(
                {
                    "status": "plan_authored",
                    "plan_id": "p",
                    "plan_patch_path": str(patch_path),
                    "plan_patch": _fake_plan_patch(producer=kwargs["config"].runner),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    def fake_accept(*args: Any, **kwargs: Any) -> SimpleNamespace:
        acceptance_calls.append({"args": args, **kwargs})
        return SimpleNamespace(plan_id="p")

    fake_run = SimpleNamespace(result=SimpleNamespace(verdict="pass"))
    fake_receipt = SimpleNamespace(required_checks_successful=True)
    monkeypatch.setattr(validation_runner, "run_child_agent", fake_child)
    monkeypatch.setattr(
        validation_runner,
        "accept_validation_coordinator_plan",
        fake_accept,
    )
    monkeypatch.setattr(
        validation_runner,
        "execute_validation_coordinator_plan",
        lambda *_args, **_kwargs: (fake_run, fake_receipt),
    )

    result = validation_runner.run_validation_coordinator(
        validation_runner.ValidationCoordinatorRunConfig(
            repo_root=tmp_path,
            request=request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            runner=runner,
            model="test-model",
            claude_execution_mode="sdk" if runner == RUNNER_CLAUDE else None,
        )
    )

    assert result.status == "assessment_required"
    assert (
        output_dir
        / validation_runner.CHILD_CACHE_RELPATH
        / "provider-cache"
        / "entry.json"
    ).is_file()
    assert len(child_calls) == 1
    call = child_calls[0]
    assert call["stage_skills"] is True
    assert call["tools_disabled"] is True
    assert call["extra_allowed_hosts"] == []
    assert call["config"].runner == runner
    assert call["config"].scene_backend == "none"
    descriptor = call["launch_descriptor"]
    assert descriptor.runner_identity.runner == runner
    assert descriptor.runner_identity.model == "test-model"
    assert descriptor.runner_identity.model_reasoning_effort is None
    assert descriptor.runner_identity.claude_execution_mode == (
        "sdk" if runner == RUNNER_CLAUDE else None
    )
    assert call["config"].child_forbidden_environment_names == ("VALIDATION_JUDGE_KEY",)
    forbidden_environment_names = set(
        descriptor.credential_policy.forbidden_environment_names
    )
    assert {
        "NGC_API_KEY",
        "NVCF_API_KEY",
        "NVCF_RENDER_FUNCTION_ID",
        "VALIDATION_JUDGE_KEY",
    } <= forbidden_environment_names
    assert descriptor.artifacts.child_output_path == str(
        (output_dir / "raw" / "validation_planner_output.jsonl").resolve()
    )
    assert descriptor.artifacts.child_final_path == str(
        (output_dir / "raw" / "validation_planner_final.json").resolve()
    )
    assert call["output_schema"]["properties"]["status"] == {
        "const": "plan_authored",
        "title": "Status",
        "type": "string",
    }
    assert _child_workflow_name(call["config"]) == "validation.plan"
    assert call["config"].child_capability_inventory is not None
    assert (
        call["config"].child_domain_policy_bounds
        == call["config"].child_capability_inventory
    )
    assert "decision-only" in call["prompt"]
    assert "<validation_coordinator_preparation>" in call["prompt"]
    assert '"mandatory_constraints"' in call["prompt"]
    assert json.dumps(task_description) in call["prompt"]
    assert "Copy every plan_id and check_id" in call["prompt"]
    assert "structured final response's `plan_patch`" in call["prompt"]
    assert "Do not use a shell, file-write tool" in call["prompt"]
    assert not (output_dir / "validation_coordinator_plan_patch.json").is_symlink()
    assert (output_dir / "validation_coordinator_plan_patch.json").is_file()
    assert len(acceptance_calls) == 1
    assert acceptance_calls[0]["producer"] == runner
    assert acceptance_calls[0]["child_plan_id"] == "p"
    assert acceptance_calls[0][
        "expected_child_launch_descriptor_digest"
    ] == canonical_json_digest(descriptor.model_dump(mode="json"))
    child_session_id = acceptance_calls[0]["child_session_id"]
    assert isinstance(child_session_id, str) and child_session_id
    assert child_session_id in call["prompt"]


@pytest.mark.parametrize(
    ("execution_mode", "max_turns", "message"),
    (
        ("sdk", 0, "must be greater than 0"),
        ("sdk", -1, "must be greater than 0"),
        ("cli", 10, "is not supported with.*cli"),
    ),
)
def test_validation_runner_rejects_unsupported_claude_turn_limit(
    tmp_path: Path,
    execution_mode: Literal["sdk", "cli"],
    max_turns: int,
    message: str,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
    )

    with pytest.raises(ValueError, match=message):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner=RUNNER_CLAUDE,
                model="test-model",
                claude_execution_mode=execution_mode,
                claude_max_turns=max_turns,
            )
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("runner", "execution_mode", "config_field"),
    (
        (RUNNER_CODEX, None, "codex_config"),
        (RUNNER_CLAUDE, "sdk", "claude_config"),
    ),
)
def test_validation_runner_rejects_inline_agent_credentials_before_preparation(
    tmp_path: Path,
    runner: str,
    execution_mode: str | None,
    config_field: str,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
    )
    unsafe_config: dict[str, object] = {"api_key": "must-not-be-persisted"}

    with pytest.raises(ValueError, match="Validation agent configuration"):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner=runner,
                model="test-model",
                claude_execution_mode=execution_mode,
                codex_config=(
                    unsafe_config if config_field == "codex_config" else None
                ),
                claude_config=(
                    unsafe_config if config_field == "claude_config" else None
                ),
            )
        )

    assert not output_dir.exists()


def test_validation_runner_allows_only_digest_matched_claude_cli_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"exact-reference-image")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
        policy={"reference_image_paths": [str(reference)]},
    )

    def fake_child(**kwargs: Any) -> int:
        _stage_fake_validation_skill(kwargs["config"].repo_root, kwargs["run_dir"])
        raw_dir = kwargs["run_dir"] / "raw"
        staged = raw_dir / "validation_planner_reference_images"
        staged.mkdir()
        staged.joinpath("reference-000.png").write_bytes(reference.read_bytes())
        patch_path = kwargs["run_dir"] / "validation_coordinator_plan_patch.json"
        kwargs["child_output_path"].write_text("planner output\n", encoding="utf-8")
        kwargs["child_final_path"].write_text(
            json.dumps(
                {
                    "status": "plan_authored",
                    "plan_id": "p",
                    "plan_patch_path": str(patch_path),
                    "plan_patch": _fake_plan_patch(producer="claude"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for name, content in (
            ("validation_planner_request.json", "{}\n"),
            ("validation_planner_items.json", "[]\n"),
            ("validation_planner_result.json", "{}\n"),
            ("validation_planner_observable_events.jsonl", ""),
            ("validation_planner_launch_descriptor.json", "{}\n"),
        ):
            (raw_dir / name).write_text(content, encoding="utf-8")
        return 0

    fake_run = SimpleNamespace(result=SimpleNamespace(verdict="pass"))
    fake_receipt = SimpleNamespace(required_checks_successful=True)
    monkeypatch.setattr(validation_runner, "run_child_agent", fake_child)
    monkeypatch.setattr(
        validation_runner,
        "accept_validation_coordinator_plan",
        lambda *_args, **_kwargs: SimpleNamespace(plan_id="p"),
    )
    monkeypatch.setattr(
        validation_runner,
        "execute_validation_coordinator_plan",
        lambda *_args, **_kwargs: (fake_run, fake_receipt),
    )

    result = validation_runner.run_validation_coordinator(
        validation_runner.ValidationCoordinatorRunConfig(
            repo_root=tmp_path,
            request=request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            runner=RUNNER_CLAUDE,
            model="test-model",
            claude_execution_mode="cli",
        )
    )

    assert result.status == "assessment_required"


def test_validation_runner_rejects_nested_raw_child_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
        policy={"visual_evidence_mode": "canonical_usd"},
    )

    def fake_child(**kwargs: Any) -> int:
        _stage_fake_validation_skill(kwargs["config"].repo_root, kwargs["run_dir"])
        patch_path = kwargs["run_dir"] / "validation_coordinator_plan_patch.json"
        kwargs["child_output_path"].write_text("planner output\n", encoding="utf-8")
        kwargs["child_final_path"].write_text(
            json.dumps(
                {
                    "status": "plan_authored",
                    "plan_id": "p",
                    "plan_patch_path": str(patch_path),
                    "plan_patch": _fake_plan_patch(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        helper = kwargs["run_dir"] / "raw" / "helper.py"
        helper.write_text("raise SystemExit\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(validation_runner, "run_child_agent", fake_child)
    monkeypatch.setattr(
        validation_runner,
        "accept_validation_coordinator_plan",
        lambda *_args, **_kwargs: pytest.fail("unsafe child output reached acceptance"),
    )

    with pytest.raises(RuntimeError, match="decision-only raw boundary.*helper.py"):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner=RUNNER_CODEX,
                model="test-model",
            )
        )


@pytest.mark.parametrize(
    "unexpected_relative",
    (
        Path("prompts/helper.py"),
        Path(".agents/skills/content-workflow-validation/helper.py"),
        Path(".claude/skills/content-workflow-validation/helper.py"),
        Path(".claude/.claude-plugin/helper.py"),
    ),
)
def test_validation_runner_rejects_nested_support_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unexpected_relative: Path,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
        policy={"visual_evidence_mode": "canonical_usd"},
    )

    def fake_child(**kwargs: Any) -> int:
        _stage_fake_validation_skill(kwargs["config"].repo_root, kwargs["run_dir"])
        patch_path = kwargs["run_dir"] / "validation_coordinator_plan_patch.json"
        kwargs["child_output_path"].write_text("planner output\n", encoding="utf-8")
        kwargs["child_final_path"].write_text(
            json.dumps(
                {
                    "status": "plan_authored",
                    "plan_id": "p",
                    "plan_patch_path": str(patch_path),
                    "plan_patch": _fake_plan_patch(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unexpected = kwargs["run_dir"] / unexpected_relative
        unexpected.parent.mkdir(parents=True, exist_ok=True)
        unexpected.write_text("raise SystemExit\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(validation_runner, "run_child_agent", fake_child)
    monkeypatch.setattr(
        validation_runner,
        "accept_validation_coordinator_plan",
        lambda *_args, **_kwargs: pytest.fail("unsafe child output reached acceptance"),
    )

    with pytest.raises(
        RuntimeError,
        match="decision-only prompt boundary|trusted staged skill tree",
    ):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner=RUNNER_CODEX,
                model="test-model",
            )
        )


def test_validation_runner_rejects_unknown_child_runner(tmp_path: Path) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
    )

    with pytest.raises(ValueError, match="Unsupported Validation child runner"):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner="unknown",
                model="test-model",
                dry_run=True,
            )
        )


@pytest.mark.parametrize(
    ("runner", "model", "claude_execution_mode", "message"),
    (
        (RUNNER_CODEX, "", None, "explicit non-empty child model"),
        (RUNNER_CLAUDE, "test-model", None, "explicit Claude execution mode"),
        (
            RUNNER_CODEX,
            "test-model",
            "cli",
            "only be selected with runner=claude",
        ),
    ),
)
def test_validation_runner_rejects_implicit_provider_selection_before_preparation(
    tmp_path: Path,
    runner: str,
    model: str,
    claude_execution_mode: str | None,
    message: str,
) -> None:
    source = tmp_path / "asset.txt"
    source.write_text("asset\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    request = ValidationRequest(
        task_description="Select exact checks for this asset.",
        inputs=(str(source),),
        project=ValidationProject(working_dir=str(output_dir)),
    )

    with pytest.raises(ValueError, match=message):
        validation_runner.run_validation_coordinator(
            validation_runner.ValidationCoordinatorRunConfig(
                repo_root=tmp_path,
                request=request,
                output_dir=output_dir,
                config_base_dir=tmp_path,
                runner=runner,
                model=model,
                claude_execution_mode=claude_execution_mode,
                dry_run=True,
            )
        )

    assert not output_dir.exists()
