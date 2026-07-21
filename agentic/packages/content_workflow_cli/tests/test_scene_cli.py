# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public large-scene batch launcher."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from content_agent_workflows.large_scene import begin_phase, load_run_state

from content_workflow_cli import scene_runner
from content_workflow_cli.cli import main
from content_workflow_cli.runner import _build_codex_sdk_request
from content_workflow_cli.scene_runner import SceneRunConfig, run_scene_workflow
from content_workflow_cli.trace import UnsafeRunArtifactError, _request_input


def _write_scene_inputs(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    (repo_root / "agentic" / ".agents" / "skills").mkdir(parents=True)
    source = repo_root / "scene.usd"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    references = repo_root / "references"
    references.mkdir()
    (references / "b.png").write_text("b", encoding="utf-8")
    (references / "A.jpg").write_text("a", encoding="utf-8")
    (references / "notes.txt").write_text("notes", encoding="utf-8")
    (references / "ignored.bin").write_text("ignored", encoding="utf-8")
    accepted = repo_root / "accepted.png"
    accepted.write_text("accepted", encoding="utf-8")

    materials_usd = repo_root / "materials.usd"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    materials_yaml = repo_root / "materials.yaml"
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    guidance = repo_root / "material_guidance.md"
    guidance.write_text("Use blue panels and white frames.\n", encoding="utf-8")
    return {
        "repo_root": repo_root,
        "source": source,
        "references": references,
        "accepted": accepted,
        "materials_yaml": materials_yaml,
        "materials_usd": materials_usd,
        "guidance": guidance,
    }


def _scene_config(
    paths: dict[str, Path],
    run_dir: Path,
    *,
    dry_run: bool,
) -> SceneRunConfig:
    return SceneRunConfig(
        repo_root=paths["repo_root"],
        usd_path=paths["source"],
        requested_tasks=["material"],
        workbench_url="http://127.0.0.1:8088",
        reference_images=[paths["accepted"]],
        materials_yaml=paths["materials_yaml"],
        materials_usd=paths["materials_usd"],
        additional_instructions="Use blue panels and white frames.",
        additional_instruction_sources=[paths["guidance"]],
        output_dir=run_dir,
        run_id="scene-test",
        start_workbench=False,
        dry_run=dry_run,
    )


def _scene_config_with_request_size(
    paths: dict[str, Path],
    run_dir: Path,
    *,
    request_size: int,
) -> SceneRunConfig:
    config = _scene_config(paths, run_dir, dry_run=True)
    request = scene_runner._build_scene_request(
        replace(config, agent_cwd=run_dir, additional_instructions="x"),
        run_id="scene-test",
        run_dir=run_dir,
        run_state_path=run_dir / "large_scene_run.json",
    )
    serialized_size = len((request.model_dump_json(indent=2) + "\n").encode("utf-8"))
    instruction_size = request_size - serialized_size + 1
    assert instruction_size > 0
    return replace(config, additional_instructions="x" * instruction_size)


def test_scene_run_dry_run_writes_resolved_request_and_state(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"

    exit_code = main(
        [
            "scene",
            "run",
            "--usd",
            str(paths["source"]),
            "--task",
            "material",
            "--materials-yaml",
            str(paths["materials_yaml"]),
            "--reference-dir",
            str(paths["references"]),
            "--reference-image",
            str(paths["accepted"]),
            "--additional-instructions-file",
            str(paths["guidance"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--run-id",
            "scene-test",
            "--output-dir",
            str(run_dir),
            "--no-start-workbench",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request_path = run_dir / "request.json"
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy_path.parent == run_dir.parent
    assert policy_path.parent != run_dir
    assert policy_path.stat().st_mode & 0o777 == 0o600
    assert policy == {
        "schema_version": "content-agents.scene-launcher-policy.v1",
        "run_dir": str(run_dir),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    }
    assert request["schema_version"] == "content-agents.large-scene-request.v1"
    assert request["workflow"] == "scene.run"
    assert request["requested_tasks"] == ["material"]
    assert request["agent_workspace"] == str(paths["repo_root"] / "agentic")
    assert request["child_workspace"] == str(run_dir)
    assert request["runtime"]["child_timeout_seconds"] == 1800.0
    assert request["runtime"]["workbench_url"] == "http://127.0.0.1:8088"
    assert request["references"]["images"] == [
        str(paths["accepted"]),
        str(paths["references"] / "A.jpg"),
        str(paths["references"] / "b.png"),
    ]
    assert request["references"]["files"] == [str(paths["references"] / "notes.txt")]
    assert request["additional_instructions"] == "Use blue panels and white frames."
    assert request["tasks"] == [
        {
            "domain": "material",
            "inputs": {
                "materials_yaml": str(paths["materials_yaml"]),
                "materials_usd": str(paths["materials_usd"]),
            },
            "policy": {
                "appearance_evidence_policy": {
                    "default": "ignore",
                    "global_sources": [],
                    "schema_version": "content-agent-workflows.appearance-evidence-policy.v1",
                    "scopes": [],
                },
                "candidate_space": "source",
                "respect_existing_material_bindings": False,
            },
        }
    ]
    assert _request_input(request, "usd") == str(paths["source"])
    assert (
        _request_input(request, "reference_images") == request["references"]["images"]
    )
    assert _request_input(request, "materials_yaml") == str(paths["materials_yaml"])

    state = load_run_state(run_dir / "large_scene_run.json")
    assert state.current_phase == "decomposition"
    assert state.phases["decomposition"].status == "ready"
    assert str(run_dir / "request.json") in state.request_artifact_paths
    assert str(paths["guidance"]) in state.request_artifact_paths

    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`content-workflow-large-scene` skill" in prompt
    assert str(run_dir / "request.json") in prompt
    assert "Use blue panels" not in prompt


def test_scene_run_accepts_maximum_resumable_request_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scene_runner,
        "utc_now",
        lambda: "2026-07-18T00:00:00+00:00",
    )
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = _scene_config_with_request_size(
        paths,
        run_dir,
        request_size=scene_runner.MAX_SCENE_REQUEST_BYTES,
    )

    run_scene_workflow(config)

    assert (run_dir / "request.json").stat().st_size == (
        scene_runner.MAX_SCENE_REQUEST_BYTES
    )
    resumed = scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert resumed.returncode == 0
    assert resumed.request_path == run_dir / "request.json"


def test_scene_run_rejects_oversized_request_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scene_runner,
        "utc_now",
        lambda: "2026-07-18T00:00:00+00:00",
    )
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = _scene_config_with_request_size(
        paths,
        run_dir,
        request_size=scene_runner.MAX_SCENE_REQUEST_BYTES + 1,
    )

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="Scene request exceeds",
    ):
        run_scene_workflow(config)

    assert not (run_dir / "request.json").exists()
    assert not (run_dir / "large_scene_run.json").exists()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()


def test_scene_run_freezes_custom_remote_workbench_url(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    workbench_url = "https://workbench.example.test:8443"
    config = replace(
        _scene_config(paths, run_dir, dry_run=True),
        workbench_url=workbench_url,
        start_workbench=False,
    )

    result = run_scene_workflow(config)

    assert result.returncode == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["workbench_url"] == workbench_url


def test_scene_run_rejects_inspection_candidate_space(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    config = replace(
        _scene_config(paths, tmp_path / "run", dry_run=True),
        material_candidate_space="inspection",
    )

    with pytest.raises(ValueError, match="material-candidate-space=source"):
        run_scene_workflow(config)


def test_scene_run_requires_material_library_for_material_task(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_scene_inputs(tmp_path)

    exit_code = main(
        [
            "scene",
            "run",
            "--usd",
            str(paths["source"]),
            "--task",
            "material",
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--materials-yaml is required" in capsys.readouterr().err


def test_scene_run_rejects_unsafe_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_scene_inputs(tmp_path)

    exit_code = main(
        [
            "scene",
            "run",
            "--usd",
            str(paths["source"]),
            "--task",
            "material",
            "--materials-yaml",
            str(paths["materials_yaml"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--run-id",
            "../outside",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--run-id must start with an alphanumeric" in capsys.readouterr().err


def test_scene_run_rejects_zero_exit_when_phases_are_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        scene_runner, "wait_for_workbench", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    result = run_scene_workflow(_scene_config(paths, tmp_path / "run", dry_run=False))

    assert result.returncode == 1
    assert result.completed is False
    assert captured["config"].agent_cwd == (tmp_path / "run").resolve()  # type: ignore[union-attr]
    terminal = json.loads(
        result.terminal_validation_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    assert terminal["valid"] is False
    assert "current_phase is still decomposition" in terminal["errors"]


def test_scene_run_does_not_follow_child_planted_artifact_symlinks_after_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    outside_output = tmp_path / "outside-output.txt"
    outside_output.write_text("keep output\n", encoding="utf-8")
    outside_trace = tmp_path / "outside-trace.txt"
    outside_trace.write_text("keep trace\n", encoding="utf-8")

    def fake_child(**kwargs: object) -> int:
        child_output_path = Path(str(kwargs["child_output_path"]))
        child_output_path.symlink_to(outside_output)
        events_path = run_dir / "trace" / "events.jsonl"
        events_path.unlink()
        events_path.symlink_to(outside_trace)
        scene_runner._reject_unsafe_run_links(run_dir)
        return 0

    monkeypatch.setattr(
        scene_runner, "wait_for_workbench", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    with pytest.raises(UnsafeRunArtifactError, match="symlinks are not allowed"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert outside_output.read_text(encoding="utf-8") == "keep output\n"
    assert outside_trace.read_text(encoding="utf-8") == "keep trace\n"


def test_scene_resume_resets_interrupted_phase_without_repeating_predecessors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    prepared = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    begin_phase(prepared.run_state_path, "decomposition", actor="test")

    def fake_child(**kwargs: object) -> int:
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        scene_runner, "wait_for_workbench", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    result = scene_runner.resume_scene_workflow(run_dir)

    assert result.returncode == 1
    state = load_run_state(prepared.run_state_path)
    assert state.phases["decomposition"].status == "ready"
    assert any(
        transition.reason == "Batch launcher resumed an interrupted or failed phase."
        for transition in state.transitions
    )


def test_scene_resume_cli_dry_run_preserves_ready_state(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    prepared = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    exit_code = main(
        [
            "scene",
            "resume",
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "agent_resume_prompt.md").is_file()
    state = load_run_state(prepared.run_state_path)
    assert state.phases["decomposition"].status == "ready"


def test_scene_resume_cli_uses_skill_workspace_and_confined_child_cwd(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "scene",
                "run",
                "--usd",
                str(paths["source"]),
                "--task",
                "material",
                "--materials-yaml",
                str(paths["materials_yaml"]),
                "--repo-root",
                str(paths["repo_root"]),
                "--output-dir",
                str(run_dir),
                "--no-start-workbench",
                "--dry-run",
            ]
        )
        == 0
    )

    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["agent_workspace"] == str(paths["repo_root"] / "agentic")
    assert request["child_workspace"] == str(run_dir)
    assert main(["scene", "resume", "--run-dir", str(run_dir), "--dry-run"]) == 0


def test_scene_resume_rejects_missing_legacy_launcher_policy(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    scene_runner._scene_launcher_policy_path(run_dir).unlink()

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="policy is missing.*legacy or unverified run",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_request_config_rejects_untrusted_agent_workspace(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request = scene_runner.SceneRunRequest.model_validate_json(
        (run_dir / "request.json").read_bytes()
    ).model_copy(update={"agent_workspace": str(tmp_path / "untrusted")})

    with pytest.raises(ValueError, match="trusted repository workspace"):
        scene_runner._config_from_request(request, dry_run=True)


def test_scene_run_rejects_untrusted_agent_workspace(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    untrusted_workspace = tmp_path / "untrusted"
    (untrusted_workspace / ".agents" / "skills").mkdir(parents=True)
    config = replace(
        _scene_config(paths, tmp_path / "run", dry_run=True),
        agent_workspace=untrusted_workspace,
    )

    with pytest.raises(ValueError, match="trusted repository workspace"):
        run_scene_workflow(config)

    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("field_path", "tampered_value"),
    [
        (("runtime", "codex_sandbox_mode"), "danger-full-access"),
        (("repository_root",), "/"),
        (("runtime", "start_workbench"), True),
        (("runtime", "workbench_url"), "http://127.0.0.1:65534"),
        (("agent_workspace",), "/"),
        (("child_workspace",), "/"),
    ],
)
def test_scene_resume_rejects_tampered_privileged_request_fields(
    tmp_path: Path,
    field_path: tuple[str, ...],
    tampered_value: object,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request_path = run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    target = request
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = tampered_value
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="request digest does not match",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_run_refuses_existing_launcher_policy_symlink(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    symlink_target = tmp_path / "do-not-overwrite.json"
    symlink_target.write_text("unchanged\n", encoding="utf-8")
    policy_path.symlink_to(symlink_target)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert policy_path.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize("unsafe_path", ["request.json", "raw"])
def test_scene_run_rejects_preseeded_output_links(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    if unsafe_path == "raw":
        outside.mkdir()
        (run_dir / unsafe_path).symlink_to(outside, target_is_directory=True)
    else:
        (run_dir / unsafe_path).symlink_to(outside / "request.json")

    with pytest.raises(RuntimeError, match="symlinks are not allowed"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    if unsafe_path == "raw":
        assert list(outside.iterdir()) == []
    else:
        assert not outside.exists()


def test_scene_run_allows_fresh_output_directory(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "fresh-run"

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert result.run_dir == run_dir.resolve()
    assert run_dir.is_dir()


def test_scene_run_cli_rejects_symlinked_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_scene_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    lexical_run_dir = tmp_path / "run"
    lexical_run_dir.symlink_to(outside, target_is_directory=True)

    exit_code = main(
        [
            "scene",
            "run",
            "--usd",
            str(paths["source"]),
            "--task",
            "material",
            "--materials-yaml",
            str(paths["materials_yaml"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--output-dir",
            str(lexical_run_dir),
            "--no-start-workbench",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "must resolve without traversing symlinks" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


def test_scene_resume_cli_rejects_symlinked_run_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_scene_inputs(tmp_path)
    actual_run_dir = tmp_path / "actual-run"
    run_scene_workflow(_scene_config(paths, actual_run_dir, dry_run=True))
    lexical_run_dir = tmp_path / "run"
    lexical_run_dir.symlink_to(actual_run_dir, target_is_directory=True)

    exit_code = main(
        [
            "scene",
            "resume",
            "--run-dir",
            str(lexical_run_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "must resolve without traversing symlinks" in capsys.readouterr().err
    assert not (actual_run_dir / "agent_resume_prompt.md").exists()


def test_scene_resume_refuses_launcher_policy_symlink(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    symlink_target = tmp_path / "copied-policy.json"
    symlink_target.write_bytes(policy_path.read_bytes())
    policy_path.unlink()
    policy_path.symlink_to(symlink_target)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="read scene launcher policy safely",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_resume_refuses_hard_linked_launcher_policy(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    os.link(policy_path, tmp_path / "policy-alias.json")

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="single-link regular file",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_resume_rejects_request_symlink_before_policy_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request_path = run_dir / "request.json"
    outside_request = tmp_path / "outside-request.json"
    outside_request.write_bytes(request_path.read_bytes())
    request_path.unlink()
    request_path.symlink_to(outside_request)

    def fail_if_policy_is_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("policy verification must follow run-tree validation")

    monkeypatch.setattr(
        scene_runner,
        "_verify_scene_launcher_policy",
        fail_if_policy_is_read,
    )

    with pytest.raises(RuntimeError, match="symlinks are not allowed"):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_launcher_policy_reader_rejects_hard_linked_request(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request_path = run_dir / "request.json"
    os.link(request_path, tmp_path / "request-alias.json")

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="Protected scene request must be a single-link regular file",
    ):
        scene_runner._verify_scene_launcher_policy(
            run_dir,
            request_path=request_path,
        )


def test_scene_launcher_policy_reader_bounds_request_size(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request_path = run_dir / "request.json"
    request_path.write_bytes(b"x" * (scene_runner.MAX_SCENE_REQUEST_BYTES + 1))

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="Protected scene request exceeds",
    ):
        scene_runner._verify_scene_launcher_policy(
            run_dir,
            request_path=request_path,
        )


def test_scene_run_rejects_child_workspace_outside_run_dir(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    config = replace(
        _scene_config(paths, tmp_path / "run", dry_run=True),
        agent_cwd=paths["repo_root"] / "agentic",
    )

    with pytest.raises(ValueError, match="must resolve to the run directory"):
        run_scene_workflow(config)


def test_codex_sdk_request_uses_confined_run_directory(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    config = _scene_config(paths, tmp_path / "run", dry_run=False)

    request = _build_codex_sdk_request(
        config=config,
        prompt="run the scene",
        run_dir=tmp_path / "run",
        child_final_path=tmp_path / "run" / "child-final.md",
    )

    assert request["repo_root"] == str((tmp_path / "run").resolve())
