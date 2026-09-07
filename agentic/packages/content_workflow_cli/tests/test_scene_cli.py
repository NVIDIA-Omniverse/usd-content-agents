# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public large-scene batch launcher."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    seal_phase_result,
)
from content_agent_workflows.large_scene import begin_phase, load_run_state
from content_agent_workflows.scene_collection import (
    CollectionPhaseResult,
    DomainCollectionResult,
)

from content_workflow_cli import scene_runner
from content_workflow_cli import trace as trace_module
from content_workflow_cli.cli import main
from content_workflow_cli.runner import (
    _build_codex_sdk_request,
    bind_parent_usd_cli_capability,
)
from content_workflow_cli.scene_runner import SceneRunConfig, run_scene_workflow
from content_workflow_cli.trace import UnsafeRunArtifactError, _request_input


def _symlink_or_skip(
    link: Path,
    target: Path,
    *,
    target_is_directory: bool = False,
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise


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
        reference_images=[paths["accepted"]],
        materials_yaml=paths["materials_yaml"],
        materials_usd=paths["materials_usd"],
        additional_instructions="Use blue panels and white frames.",
        additional_instruction_sources=[paths["guidance"]],
        output_dir=run_dir,
        run_id="scene-test",
        dry_run=dry_run,
    )


def test_scene_parent_capability_route_is_declared(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    config = _scene_config(paths, tmp_path / "run", dry_run=True)
    capability = SimpleNamespace(
        identity_path=tmp_path / "identity.json",
        identity_sha256="a" * 64,
        server_url="http://127.0.0.1:43210",
    )

    bound = bind_parent_usd_cli_capability(config, capability)

    assert bound.usd_cli_server_url == capability.server_url
    assert asdict(bound)["usd_cli_server_url"] == capability.server_url


def _stub_usd_cli_scene_runtime(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> None:
    """Keep launcher tests CPU-only while preserving the usd-cli lifecycle."""

    target_path = paths["repo_root"] / "bin" / "usd-cli"
    telemetry_route = SimpleNamespace(
        active=True,
        target_path=target_path,
        reason=None,
        metadata=lambda: {
            "status": "active",
            "target_path": str(target_path),
        },
    )
    launches = 0

    def fake_start(**kwargs: object) -> SimpleNamespace:
        nonlocal launches
        launches += 1
        run_dir = Path(str(kwargs["run_dir"]))
        launch_id = f"scene-test-{launches}"
        identity_path = run_dir / "raw" / f"parent_usd_cli_session_{launch_id}.json"
        atomic_write_json(identity_path, {"launch_id": launch_id}, within=run_dir)
        return SimpleNamespace(
            route=telemetry_route,
            lease=object(),
            session=SimpleNamespace(
                workflow="large-scene",
                session_id="scene-test-session",
                open=lambda _path: None,
            ),
            readiness=SimpleNamespace(
                version="usd-cli test",
                artifact_path=None,
                probe={},
            ),
            identity_path=identity_path,
            identity_sha256=file_sha256(identity_path),
            server_url="http://127.0.0.1:43210",
        )

    monkeypatch.setattr(
        scene_runner,
        "start_parent_usd_cli_capability",
        fake_start,
    )
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda _capability: _released_scene_teardown_evidence(),
    )


def _released_scene_teardown_evidence() -> SimpleNamespace:
    return SimpleNamespace(
        process_released=True,
        descendants_released=True,
        sessions_released=True,
        listener_released=True,
        daemon_leases_released=True,
        state_directory_released=True,
        host="127.0.0.1",
        port=43210,
        daemon_log_path=None,
        daemon_log_sha256=None,
    )


def _scene_config_with_request_size(
    paths: dict[str, Path],
    run_dir: Path,
    *,
    request_size: int,
) -> SceneRunConfig:
    config = _scene_config(paths, run_dir, dry_run=True)
    staged_inputs_root = run_dir.parent / f".{run_dir.name}.scene-inputs"
    request = scene_runner._build_scene_request(
        replace(config, agent_cwd=run_dir, additional_instructions="x"),
        run_id="scene-test",
        run_dir=run_dir,
        run_state_path=run_dir / "large_scene_run.json",
        staged_source=SimpleNamespace(
            staged_usd_path=staged_inputs_root / "source" / paths["source"].name
        ),
        staged_material_library=SimpleNamespace(
            staged_usd_path=(
                staged_inputs_root / "material_library" / paths["materials_usd"].name
            )
        ),
        staged_materials_yaml=(
            staged_inputs_root / "material_library" / paths["materials_yaml"].name
        ),
    )
    serialized_size = len((request.model_dump_json(indent=2) + "\n").encode("utf-8"))
    instruction_size = request_size - serialized_size + 1
    assert instruction_size > 0
    return replace(config, additional_instructions="x" * instruction_size)


def test_scene_default_run_directory_is_at_repo_root(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    config = replace(
        _scene_config(paths, tmp_path / "unused", dry_run=True),
        output_dir=None,
        run_id="root-scene",
    )

    with scene_runner._prepare_scene_run_dir(config) as (run_id, run_dir):
        assert run_id == "root-scene"
        assert run_dir == paths["repo_root"] / "runs" / "root-scene"


@pytest.mark.parametrize(
    "command",
    ("phase", "decompose", "process", "material-task", "collect"),
)
def test_scene_workflow_operations_are_exposed_through_primary_cli(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["scene", command, "--help"]) == 0
    assert f"content-workflow-cli scene {command}" in capsys.readouterr().out


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
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request_path = run_dir / "request.json"
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    initialization_lock_path = scene_runner._scene_run_initialization_lock_path(run_dir)
    staged_inputs_root = run_dir.parent / f".{run_dir.name}.scene-inputs"
    assert policy_path.parent == run_dir.parent
    assert policy_path.parent != run_dir
    if os.name == "nt":
        assert stat.S_ISREG(policy_path.stat().st_mode)
        assert stat.S_ISREG(initialization_lock_path.stat().st_mode)
        assert policy_path.stat().st_nlink == 1
        assert initialization_lock_path.stat().st_nlink == 1
    else:
        assert policy_path.stat().st_mode & 0o777 == 0o600
        assert initialization_lock_path.stat().st_mode & 0o777 == 0o600
    assert policy == {
        "schema_version": "content-agents.scene-launcher-policy.v1",
        "run_dir": str(run_dir),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    }
    assert request["schema_version"] == "content-agents.large-scene-request.v2"
    assert request["workflow"] == "scene.run"
    assert request["requested_tasks"] == ["material"]
    assert request["agent_workspace"] == str(paths["repo_root"] / "agentic")
    assert request["child_workspace"] == str(run_dir)
    assert request["runtime"]["scene_backend"] == "usd-cli"
    assert request["runtime"]["child_timeout_seconds"] == 1800.0
    assert request["runtime"]["scene_tool_timeout_seconds"] == 60.0
    assert request["source_scene_original"] == str(paths["source"])
    assert request["source_scene"] == str(
        staged_inputs_root / "source" / paths["source"].name
    )
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
            "scene_backend": "usd-cli",
            "scene_session_scope": "per_asset",
            "inputs": {
                "materials_yaml": str(
                    staged_inputs_root
                    / "material_library"
                    / paths["materials_yaml"].name
                ),
                "materials_yaml_source": str(paths["materials_yaml"]),
                "materials_usd": str(
                    staged_inputs_root
                    / "material_library"
                    / paths["materials_usd"].name
                ),
                "materials_usd_source": str(paths["materials_usd"]),
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
    assert _request_input(request, "usd") == request["source_scene"]
    assert (
        _request_input(request, "reference_images") == request["references"]["images"]
    )
    staged_materials_yaml = (
        staged_inputs_root / "material_library" / paths["materials_yaml"].name
    )
    assert _request_input(request, "materials_yaml") == str(staged_materials_yaml)
    assert request["tasks"][0]["inputs"]["materials_yaml_source"] == str(
        paths["materials_yaml"]
    )
    staged_materials = yaml.safe_load(staged_materials_yaml.read_text(encoding="utf-8"))
    assert (
        staged_materials_yaml.parent / staged_materials["library_path"]
    ).resolve() == (
        staged_inputs_root / "material_library" / paths["materials_usd"].name
    )
    assert run_dir not in staged_inputs_root.parents
    assert stat.S_IMODE(staged_inputs_root.stat().st_mode) == 0o555
    assert stat.S_IMODE(Path(request["source_scene"]).stat().st_mode) == 0o444
    state = load_run_state(run_dir / "large_scene_run.json")
    assert state.schema_version == "content-agent-workflows.large-scene-run.v2"
    assert state.scene_backend == "usd-cli"
    assert state.current_phase == "decomposition"
    assert state.phases["decomposition"].status == "ready"
    assert str(run_dir / "request.json") in state.request_artifact_paths
    assert str(paths["source"]) in state.request_artifact_paths
    assert str(paths["guidance"]) in state.request_artifact_paths
    assert str(staged_materials_yaml) in state.request_artifact_paths
    assert state.source_scene == request["source_scene"]
    for label in ("source", "material_library"):
        staged_manifest = run_dir / "raw" / f"staged_input_{label}.json"
        assert staged_manifest.is_file()
        assert (
            json.loads(staged_manifest.read_text(encoding="utf-8"))["self_containment"][
                "status"
            ]
            == "verified"
        )

    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`content-workflow-large-scene` skill" in prompt
    assert str(run_dir / "request.json") in prompt
    assert "Use blue panels" not in prompt
    assert "Frozen scene backend: `usd-cli`" in prompt
    assert "independent session identity" in prompt
    assert "Use `content-workflow-cli scene ...` for every workflow operation" in prompt
    assert "Never invoke\n`content-workflow-cli scene run`" in prompt
    assert "`content-workflow-cli scene resume` here" in prompt
    assert "Report a policy blocker only after" in prompt
    assert "Never substitute a `*_source` provenance path" in prompt
    if os.name == "nt":
        assert "content-workflow-cli artifact write-json" in prompt
        assert "child file-editing tools" not in prompt
        assert "required parent directories" in prompt
        assert "`scene material-task run-batch` command MUST include" in prompt
        assert "`--fail-fast`" in prompt
    else:
        assert "child file-editing tools" in prompt
    prepared_manifest_path = run_dir / "workflow_run_manifest.json"
    assert prepared_manifest_path.is_file()
    manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    assert manifest["workflow"] == "scene.run"
    assert manifest["status"] == "blocked"
    assert manifest["source_sha256"] == file_sha256(paths["source"])
    assert manifest["backend"]["scene_backend"] == "usd-cli"
    assert manifest["backend"]["scene_session_scope"] == "per_asset"
    assert manifest["policy"]["input_sha256"][str(paths["materials_yaml"])] == (
        file_sha256(paths["materials_yaml"])
    )
    assert {artifact["logical_name"] for artifact in manifest["artifacts"]} >= {
        "request",
        "run_state",
        "prompt",
        "operation_trace",
    }
    assert manifest["checkpoints"][-1]["phase"] == "blocked"
    assert "run_state" not in manifest["checkpoints"][-1]["artifact_sha256"]


def test_scene_run_rewrites_relative_material_library_to_staged_copy(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    manifest_dir = paths["repo_root"] / "manifests"
    library_dir = paths["repo_root"] / "libs"
    manifest_dir.mkdir()
    library_dir.mkdir()
    paths["materials_yaml"] = manifest_dir / "materials.yaml"
    paths["materials_usd"] = library_dir / "materials.usd"
    paths["materials_usd"].write_text("#usda 1.0\n", encoding="utf-8")
    paths["materials_yaml"].write_text(
        'library_path: "../libs/materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    request = json.loads(result.request_path.read_text(encoding="utf-8"))
    staged_yaml = Path(request["tasks"][0]["inputs"]["materials_yaml"])
    staged_usd = Path(request["tasks"][0]["inputs"]["materials_usd"])
    payload = yaml.safe_load(staged_yaml.read_text(encoding="utf-8"))
    assert (staged_yaml.parent / payload["library_path"]).resolve() == staged_usd


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
    run_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 1
    assert result.completed is False
    assert captured["config"].agent_cwd == (tmp_path / "run").resolve()  # type: ignore[union-attr]
    terminal = json.loads(
        result.terminal_validation_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    assert terminal["valid"] is False
    assert "current_phase is still decomposition" in terminal["errors"]
    manifest = json.loads(result.workflow_run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "fail"
    assert manifest["failure"]["code"] == "large_scene_run_failed"
    artifact_by_name = {
        artifact["logical_name"]: artifact for artifact in manifest["artifacts"]
    }
    assert artifact_by_name["backend_probe"]["sha256"] == file_sha256(
        run_dir / artifact_by_name["backend_probe"]["path"]
    )
    assert result.terminal_validation_path is not None
    assert artifact_by_name["terminal_validation"]["sha256"] == file_sha256(
        result.terminal_validation_path
    )


@pytest.mark.parametrize("write_errno", [errno.ENOSPC, errno.EDQUOT])
def test_scene_run_finishes_validation_after_transient_trace_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    write_errno: int,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    original_append = trace_module.append_run_text

    def fake_child(**kwargs: object) -> int:
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    def fail_finished_event_once(
        artifact_run_dir: Path,
        path: Path,
        text: str,
    ) -> None:
        if '"event_type": "child_agent_finished"' in text:
            raise OSError(write_errno, "injected transient trace write failure")
        original_append(artifact_run_dir, path, text)

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(trace_module, "append_run_text", fail_finished_event_once)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 1
    assert result.terminal_validation_path is not None
    assert result.terminal_validation_path.is_file()
    assert (run_dir / "run_cost_metrics.json").is_file()
    assert any(
        "Unable to append trace event child_agent_finished" in record.message
        for record in caplog.records
    )
    events = [
        json.loads(line)
        for line in (run_dir / "trace" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(event["event_type"] == "child_agent_finished" for event in events)
    assert any(event["event_type"] == "terminal_validation" for event in events)


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
        _symlink_or_skip(child_output_path, outside_output)
        events_path = run_dir / "trace" / "events.jsonl"
        events_path.unlink()
        _symlink_or_skip(events_path, outside_trace)
        scene_runner._reject_unsafe_run_links(run_dir)
        return 0

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
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
    initial_manifest = json.loads(
        prepared.workflow_run_manifest_path.read_text(encoding="utf-8")
    )
    begin_phase(prepared.run_state_path, "decomposition", actor="test")

    def fake_child(**kwargs: object) -> int:
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    result = scene_runner.resume_scene_workflow(run_dir)

    assert result.returncode == 1
    state = load_run_state(prepared.run_state_path)
    assert state.phases["decomposition"].status == "ready"
    assert any(
        transition.reason == "Batch launcher resumed an interrupted or failed phase."
        for transition in state.transitions
    )
    resumed_manifest = json.loads(
        result.workflow_run_manifest_path.read_text(encoding="utf-8")
    )
    assert resumed_manifest["created_at"] == initial_manifest["created_at"]
    assert len(resumed_manifest["checkpoints"]) > len(initial_manifest["checkpoints"])
    assert resumed_manifest["status"] == "fail"


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


def test_scene_resume_rejects_missing_modern_manifest(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    prepared = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    prepared.workflow_run_manifest_path.unlink()

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="modern runs cannot be adopted",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    assert not prepared.workflow_run_manifest_path.exists()
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_resume_explicit_legacy_adoption_rejects_modern_run(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    prepared = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    request_sha256 = hashlib.sha256(prepared.request_path.read_bytes()).hexdigest()
    manifest_before = prepared.workflow_run_manifest_path.read_bytes()

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="mixed or partially modern run",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )

    assert not scene_runner._legacy_scene_adoption_receipt_path(run_dir).exists()
    assert prepared.workflow_run_manifest_path.read_bytes() == manifest_before
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
        (("runtime", "scene_tool_timeout_seconds"), 0.01),
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
    _symlink_or_skip(policy_path, symlink_target)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert policy_path.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "unchanged\n"


def test_scene_run_refuses_initialization_lock_symlink(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    lock_path = scene_runner._scene_run_initialization_lock_path(run_dir)
    symlink_target = tmp_path / "do-not-lock"
    symlink_target.write_text("unchanged\n", encoding="utf-8")
    _symlink_or_skip(lock_path, symlink_target)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="open scene initialization lock safely",
    ):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert lock_path.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "unchanged\n"


def test_scene_run_recreates_policy_after_run_directory_is_deleted(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = _scene_config(paths, run_dir, dry_run=True)
    run_scene_workflow(config)
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    first_policy = json.loads(policy_path.read_text(encoding="utf-8"))

    shutil.rmtree(run_dir)
    rerun = run_scene_workflow(
        replace(config, additional_instructions="Use a different palette.")
    )

    request_bytes = rerun.request_path.read_bytes()
    replacement_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert (
        replacement_policy["request_sha256"]
        == hashlib.sha256(request_bytes).hexdigest()
    )
    assert replacement_policy["request_sha256"] != first_policy["request_sha256"]
    resumed = scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert resumed.returncode == 0


def test_scene_run_refuses_invalid_orphan_launcher_policy(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    policy_path.write_text("not a launcher policy\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert policy_path.read_text(encoding="utf-8") == "not a launcher policy\n"


def test_scene_run_surfaces_orphan_policy_inspection_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    shutil.rmtree(run_dir)
    original_lstat = Path.lstat

    def reject_policy_inspection(path: Path) -> os.stat_result:
        if path == policy_path:
            raise PermissionError("policy inspection denied")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", reject_policy_inspection)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="Unable to inspect scene launcher policy.*inspection denied",
    ):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))


def test_scene_run_refuses_hard_linked_orphan_policy(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    policy_alias = tmp_path / "policy-alias.json"
    os.link(policy_path, policy_alias)
    shutil.rmtree(run_dir)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert policy_path.samefile(policy_alias)


def test_scene_run_preserves_policy_created_by_concurrent_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    policy_path = scene_runner._scene_launcher_policy_path(run_dir)
    concurrent_request = b"concurrent launcher request\n"
    original_reject_unsafe_run_links = scene_runner._reject_unsafe_run_links

    def inject_concurrent_policy(path: Path, *, allow_missing: bool = False) -> None:
        original_reject_unsafe_run_links(path, allow_missing=allow_missing)
        if not allow_missing and path == run_dir and not policy_path.exists():
            scene_runner._write_scene_launcher_policy(
                run_dir,
                request_bytes=concurrent_request,
            )

    monkeypatch.setattr(
        scene_runner,
        "_reject_unsafe_run_links",
        inject_concurrent_policy,
    )

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["request_sha256"] == hashlib.sha256(concurrent_request).hexdigest()


def test_scene_run_serializes_concurrent_orphan_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = _scene_config(paths, run_dir, dry_run=True)
    run_scene_workflow(config)
    shutil.rmtree(run_dir)

    first_reached_cleanup = threading.Event()
    release_first = threading.Event()
    second_reached_request_write = threading.Event()
    original_remove_stale_policy = scene_runner._remove_stale_scene_launcher_policy
    original_write_request = scene_runner._write_request

    def block_first_cleanup(*args: object, **kwargs: object) -> None:
        if threading.current_thread().name == "first-launcher":
            first_reached_cleanup.set()
            assert release_first.wait(5)
        original_remove_stale_policy(*args, **kwargs)  # type: ignore[arg-type]

    def observe_request_write(*args: object, **kwargs: object) -> bytes:
        if threading.current_thread().name == "second-launcher":
            second_reached_request_write.set()
        return original_write_request(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        scene_runner,
        "_remove_stale_scene_launcher_policy",
        block_first_cleanup,
    )
    monkeypatch.setattr(scene_runner, "_write_request", observe_request_write)

    outcomes: dict[str, object] = {}

    def launch(name: str, launcher_config: SceneRunConfig) -> None:
        try:
            outcomes[name] = run_scene_workflow(launcher_config)
        except Exception as exc:  # noqa: BLE001 - captured for cross-thread assertion
            outcomes[name] = exc

    first = threading.Thread(
        name="first-launcher",
        target=launch,
        args=("first", replace(config, additional_instructions="first")),
    )
    second = threading.Thread(
        name="second-launcher",
        target=launch,
        args=("second", replace(config, additional_instructions="second")),
    )
    try:
        first.start()
        assert first_reached_cleanup.wait(5)
        second.start()
        assert not second_reached_request_write.wait(0.2)
    finally:
        release_first.set()
        first.join(5)
        second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert isinstance(outcomes["first"], scene_runner.SceneRunResult)
    assert isinstance(outcomes["second"], FileExistsError)
    resumed = scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    assert resumed.returncode == 0


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
        _symlink_or_skip(run_dir / unsafe_path, outside, target_is_directory=True)
    else:
        _symlink_or_skip(run_dir / unsafe_path, outside / "request.json")

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
    _symlink_or_skip(lexical_run_dir, outside, target_is_directory=True)

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
    _symlink_or_skip(lexical_run_dir, actual_run_dir, target_is_directory=True)

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
    _symlink_or_skip(policy_path, symlink_target)

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
    _symlink_or_skip(request_path, outside_request)

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


_LEGACY_SCENE_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "legacy_scene_v1"


def test_legacy_scene_request_uses_current_timeout_when_old_value_is_unset() -> None:
    fixture = json.loads(
        (_LEGACY_SCENE_FIXTURE_DIR / "request.json").read_text(encoding="utf-8")
    )
    cases: list[object] = [None, 0.0, "missing"]

    for legacy_timeout in cases:
        request_payload = json.loads(json.dumps(fixture))
        if legacy_timeout == "missing":
            request_payload["runtime"].pop("workbench_timeout_seconds")
        else:
            request_payload["runtime"]["workbench_timeout_seconds"] = legacy_timeout

        request = scene_runner.SceneRunRequest.model_validate(request_payload)

        assert (
            request.runtime.scene_tool_timeout_seconds
            == scene_runner.DEFAULT_SCENE_TOOL_TIMEOUT_SECONDS
        )


def _replace_legacy_fixture_values(
    value: object,
    replacements: dict[str, str],
) -> object:
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [_replace_legacy_fixture_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_legacy_fixture_values(item, replacements)
            for key, item in value.items()
        }
    return value


def _write_completed_legacy_collection(run_dir: Path) -> tuple[Path, str, str]:
    collection_dir = run_dir / "03-collection"
    domain_dir = collection_dir / "domains" / "material"
    domain_dir.mkdir(parents=True)
    final_output = domain_dir / "output.usda"
    final_output.write_text("#usda 1.0\n", encoding="utf-8")
    collection_report = atomic_write_json(
        domain_dir / "collection_report.json", {"passed": True}
    )
    harmonization_report = atomic_write_json(
        domain_dir / "harmonization_report.json", {"passed": True}
    )
    validation_report = atomic_write_json(
        domain_dir / "validation_report.json", {"passed": True}
    )
    domain_artifacts = [
        str(final_output),
        str(collection_report),
        str(harmonization_report),
        str(validation_report),
    ]
    domain_result_path = domain_dir / "result.json"
    domain_result = seal_phase_result(
        DomainCollectionResult(
            domain="material",
            status="completed",
            output_paths=[str(final_output)],
            collection_report_path=str(collection_report),
            harmonization_report_path=str(harmonization_report),
            validation_report_path=str(validation_report),
            validation_passed=True,
            artifact_paths=domain_artifacts,
        ),
        domain_result_path,
    )
    assert domain_result.output_digest is not None

    topology_report = atomic_write_json(
        collection_dir / "topology_report.json", {"passed": True}
    )
    collection_input_digest = "legacy-processing-output-digest"
    collection_artifacts = [
        str(domain_result_path),
        *domain_artifacts,
        str(topology_report),
    ]
    collection_result_path = collection_dir / "result.json"
    collection_result = seal_phase_result(
        CollectionPhaseResult(
            success=True,
            input_digest=collection_input_digest,
            domain_result_paths=[str(domain_result_path)],
            required_domain_count=1,
            completed_required_domain_count=1,
            topology_report_path=str(topology_report),
            final_output_paths=[str(final_output)],
            artifact_paths=collection_artifacts,
            completion_policy_satisfied=True,
        ),
        collection_result_path,
    )
    assert collection_result.output_digest is not None
    return (
        collection_result_path,
        collection_input_digest,
        collection_result.output_digest,
    )


def _materialize_legacy_scene_fixture(
    paths: dict[str, Path],
    run_dir: Path,
    *,
    completed: bool = False,
) -> str:
    """Materialize checked-in pre-policy v1 bytes with test-local paths."""

    run_dir.mkdir()
    (run_dir / "raw").mkdir()
    (run_dir / "trace").mkdir()
    request_path = run_dir / "request.json"
    state_path = run_dir / "large_scene_run.json"
    replacements = {
        "__RUN_DIR__": str(run_dir),
        "__RUN_STATE__": str(state_path),
        "__REQUEST__": str(request_path),
        "__REPOSITORY_ROOT__": str(paths["repo_root"]),
        "__AGENT_WORKSPACE__": str(paths["repo_root"] / "agentic"),
        "__SOURCE_SCENE__": str(paths["source"]),
        "__REFERENCE_IMAGE__": str(paths["accepted"]),
        "__GUIDANCE__": str(paths["guidance"]),
        "__MATERIALS_YAML__": str(paths["materials_yaml"]),
        "__MATERIALS_USD__": str(paths["materials_usd"]),
    }
    request_fixture = json.loads(
        (_LEGACY_SCENE_FIXTURE_DIR / "request.json").read_text(encoding="utf-8")
    )
    request_payload = _replace_legacy_fixture_values(
        request_fixture,
        replacements,
    )
    request_path.write_text(
        json.dumps(request_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    collection_replacements: dict[str, str] = {}
    if completed:
        result_path, input_digest, output_digest = _write_completed_legacy_collection(
            run_dir
        )
        collection_replacements = {
            "__COLLECTION_RESULT__": str(result_path),
            "__COLLECTION_INPUT_DIGEST__": input_digest,
            "__COLLECTION_OUTPUT_DIGEST__": output_digest,
        }
    state_fixture_name = "completed_state.json" if completed else "active_state.json"
    state_fixture = json.loads(
        (_LEGACY_SCENE_FIXTURE_DIR / state_fixture_name).read_text(encoding="utf-8")
    )
    state_payload = _replace_legacy_fixture_values(
        state_fixture,
        {**replacements, **collection_replacements},
    )
    source_input_digest = scene_runner._source_input_digest(
        str(paths["source"]),
        [
            str(request_path),
            str(paths["guidance"]),
            str(paths["accepted"]),
            str(paths["materials_yaml"]),
            str(paths["materials_usd"]),
        ],
        ["material"],
        "Use blue panels and white frames.",
        schema_version="content-agent-workflows.large-scene-run.v1",
    )
    state_payload = _replace_legacy_fixture_values(
        state_payload,
        {"__SOURCE_INPUT_DIGEST__": source_input_digest},
    )
    state_path.write_text(
        json.dumps(state_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(request_path.read_bytes()).hexdigest()


def test_scene_run_usd_cli_backend_is_frozen_for_every_material_asset(
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
            "--repo-root",
            str(paths["repo_root"]),
            "--run-id",
            "scene-test",
            "--output-dir",
            str(run_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runtime"]["scene_backend"] == "usd-cli"
    assert request["tasks"][0]["scene_backend"] == "usd-cli"
    assert request["tasks"][0]["scene_session_scope"] == "per_asset"

    state = load_run_state(run_dir / "large_scene_run.json")
    assert state.scene_backend == "usd-cli"
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "Frozen scene backend: `usd-cli`" in prompt
    assert "independent session identity" in prompt
    assert "frozen domain task request's `processing_policy`" in prompt


@pytest.mark.parametrize("directory_name", ["raw", "trace"])
def test_scene_run_rejects_symlinked_private_directory(
    tmp_path: Path,
    directory_name: str,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / f"external-{directory_name}"
    external.mkdir()
    sentinel = external / "host-data.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    _symlink_or_skip(run_dir / directory_name, external, target_is_directory=True)

    with pytest.raises(
        UnsafeRunArtifactError, match="Child-created symlinks are not allowed"
    ):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"


def test_scene_run_rejects_dangling_request_symlink_without_creating_target(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "trace").mkdir()
    external = tmp_path / "external-request.json"
    _symlink_or_skip(run_dir / "request.json", external)

    with pytest.raises(
        UnsafeRunArtifactError, match="Child-created symlinks are not allowed"
    ):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    assert not external.exists()
    assert not (run_dir / "workflow_run_manifest.json").exists()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()


def test_scene_run_records_child_runner_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(
        scene_runner,
        "_run_child_agent",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("child launch failed")),
    )

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 2
    child_log = result.child_output_path.read_text(encoding="utf-8")
    assert "RuntimeError: child launch failed" in child_log


@pytest.mark.parametrize(
    ("child_outcome", "expected_returncode", "expected_boundary"),
    [
        ("failure", 2, "failed"),
        ("interrupt", 130, "cancelled"),
    ],
)
def test_scene_run_records_strict_teardown_after_child_failure_or_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_outcome: str,
    expected_returncode: int,
    expected_boundary: str,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)

    def child(**_kwargs: object) -> int:
        if child_outcome == "failure":
            raise RuntimeError("child failed")
        if child_outcome == "interrupt":
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(scene_runner, "_run_child_agent", child)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == expected_returncode
    receipt_paths = list((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "released"
    assert receipt["boundary"] == expected_boundary
    assert receipt["state_directory_released"] is True
    assert receipt["errors"] == []
    manifest = json.loads(result.workflow_run_manifest_path.read_text(encoding="utf-8"))
    teardown_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["logical_name"] == "usd_cli_teardown"
    )
    assert teardown_artifact["path"] == receipt_paths[0].relative_to(run_dir).as_posix()


def test_scene_live_resume_starts_after_prior_anchor_is_strictly_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    launches = 0

    def start(**kwargs: object) -> SimpleNamespace:
        nonlocal launches
        launches += 1
        observed_run_dir = Path(str(kwargs["run_dir"]))
        state_dir = observed_run_dir / ".usd-cli"
        if state_dir.exists() or state_dir.is_symlink():
            raise RuntimeError("Refusing pre-existing usd-cli run state")
        state_dir.mkdir()
        (state_dir / "config.toml").write_text("", encoding="utf-8")
        launch_id = f"resume-{launches}"
        identity_path = (
            observed_run_dir / "raw" / f"parent_usd_cli_session_{launch_id}.json"
        )
        atomic_write_json(identity_path, {"launch_id": launch_id}, within=run_dir)
        return SimpleNamespace(
            route=SimpleNamespace(
                metadata=lambda: {"status": "active", "target_path": "usd-cli"}
            ),
            lease=object(),
            session=SimpleNamespace(session_id="scene-session"),
            readiness=SimpleNamespace(version="usd-cli test", artifact_path=None),
            identity_path=identity_path,
            identity_sha256=file_sha256(identity_path),
            server_url="http://127.0.0.1:43210",
        )

    def stop(_capability: object) -> SimpleNamespace:
        state_dir = run_dir / ".usd-cli"
        assert state_dir.is_dir()
        shutil.rmtree(state_dir)
        return _released_scene_teardown_evidence()

    monkeypatch.setattr(scene_runner, "start_parent_usd_cli_capability", start)
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        stop,
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 1)

    first = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))
    second = scene_runner.resume_scene_workflow(run_dir)

    assert first.returncode == 1
    assert second.returncode == 1
    assert launches == 2
    assert not (run_dir / ".usd-cli").exists()
    receipts = sorted((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    assert len(receipts) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "released"
        for path in receipts
    )


def test_scene_teardown_retries_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    stop_calls = 0

    def stop(_capability: object) -> SimpleNamespace:
        nonlocal stop_calls
        stop_calls += 1
        if stop_calls == 1:
            raise KeyboardInterrupt
        return _released_scene_teardown_evidence()

    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        stop,
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 130
    assert stop_calls == 2
    receipt_path = next((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "released"
    assert receipt["boundary"] == "cancelled"


@pytest.mark.parametrize("terminal_valid", [False, True])  # type: ignore[misc]
def test_scene_failed_teardown_receipt_blocks_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_valid: bool,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda _capability: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 2
    receipt_path = next((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["state_directory_released"] is None
    assert "release failed" in " ".join(receipt["errors"])

    monkeypatch.setattr(
        scene_runner,
        "_validate_terminal_state",
        lambda _path: scene_runner.SceneTerminalValidation(
            checked_at="2026-08-26T00:00:00Z",
            valid=terminal_valid,
            current_phase=None,
            phase_statuses={
                "decomposition": "completed",
                "asset_task_processing": "completed",
                "collection": "completed",
            },
            errors=[] if terminal_valid else ["collection incomplete"],
            final_handoff={"valid": True} if terminal_valid else None,
        ),
    )
    monkeypatch.setattr(
        scene_runner,
        "start_parent_usd_cli_capability",
        lambda **_kwargs: pytest.fail("terminal resume must not launch a daemon"),
    )

    resumed = scene_runner.resume_scene_workflow(run_dir)

    assert resumed.returncode == 1
    assert resumed.completed is False
    manifest = json.loads(
        resumed.workflow_run_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["failure"]["code"] == "scene_usd_cli_release_unproven"


@pytest.mark.parametrize("terminal_valid", [False, True])  # type: ignore[misc]
def test_scene_missing_teardown_receipt_blocks_modern_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_valid: bool,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)
    monkeypatch.setattr(
        scene_runner,
        "_write_scene_usd_cli_teardown_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("receipt write failed")),
    )

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 2
    assert not list((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    monkeypatch.setattr(
        scene_runner,
        "_validate_terminal_state",
        lambda _path: scene_runner.SceneTerminalValidation(
            checked_at="2026-08-26T00:00:00Z",
            valid=terminal_valid,
            current_phase=None,
            phase_statuses={},
            errors=[] if terminal_valid else ["collection incomplete"],
            final_handoff={"valid": True} if terminal_valid else None,
        ),
    )
    monkeypatch.setattr(
        scene_runner,
        "start_parent_usd_cli_capability",
        lambda **_kwargs: pytest.fail("resume without release custody must not launch"),
    )

    resumed = scene_runner.resume_scene_workflow(run_dir)

    assert resumed.returncode == 1
    assert resumed.completed is False
    manifest = json.loads(
        resumed.workflow_run_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["failure"]["code"] == "scene_usd_cli_release_unproven"
    assert "missing its usd-cli teardown receipt" in " ".join(
        manifest["failure"]["terminal_errors"]
    )


def test_scene_partial_teardown_evidence_seals_failed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    release = _released_scene_teardown_evidence()
    release.state_directory_released = False
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda _capability: release,
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 2
    receipt_path = next((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["state_directory_released"] is False
    assert "state_directory_released" in " ".join(receipt["errors"])


def test_scene_teardown_receipt_survives_error_log_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda _capability: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)
    monkeypatch.setattr(
        scene_runner,
        "_append_child_runner_error",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("log failed")),
    )

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 2
    receipt_path = next((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "release failed" in " ".join(receipt["errors"])


@pytest.mark.parametrize("interrupt_stage", ["identity", "terminal", "receipt"])  # type: ignore[misc]
def test_scene_post_release_sealing_retries_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_stage: str,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", lambda **_kwargs: 0)
    target_name = {
        "identity": "_scene_parent_session_integrity_errors",
        "terminal": "_validate_terminal_state",
        "receipt": "_write_scene_usd_cli_teardown_receipt",
    }[interrupt_stage]
    original = getattr(scene_runner, target_name)
    calls = 0

    def interrupt_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(scene_runner, target_name, interrupt_once)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 130
    assert calls == 2
    receipt_path = next((run_dir / "raw").glob("scene_usd_cli_teardown_*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "released"
    assert receipt["boundary"] == "cancelled"
    assert receipt["interrupted"] is True


def test_scene_runner_symlinked_error_artifacts_abort_before_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    external_log = tmp_path / "external-child.log"
    external_log.write_text("host log\n", encoding="utf-8")
    external_probe = tmp_path / "external-probe.json"
    external_probe.write_text('{"host":true}\n', encoding="utf-8")

    def fail_after_retargeting_artifacts(**kwargs: object) -> int:
        child_output = Path(str(kwargs["child_output_path"]))
        _symlink_or_skip(child_output, external_log)
        probe = Path(str(kwargs["run_dir"])) / "raw" / "scene_backend_probe.json"
        probe.unlink()
        _symlink_or_skip(probe, external_probe)
        raise RuntimeError("primary child failure")

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(
        scene_runner,
        "_run_child_agent",
        fail_after_retargeting_artifacts,
    )

    with pytest.raises(
        UnsafeRunArtifactError,
        match="(?:singly linked regular file|changed to an unsafe path)",
    ):
        run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert external_log.read_text(encoding="utf-8") == "host log\n"
    assert external_probe.read_text(encoding="utf-8") == '{"host":true}\n'
    manifest = json.loads(
        (run_dir / "workflow_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "running"
    assert manifest["failure"] is None


def test_scene_run_usd_cli_preflight_starts_daemon_and_reaches_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    captured: dict[str, object] = {}
    capability_calls: list[dict[str, object]] = []
    lifecycle: list[str] = []
    telemetry_route = SimpleNamespace(
        active=True,
        target_path=paths["repo_root"] / "bin" / "usd-cli",
        reason=None,
        metadata=lambda: {
            "status": "active",
            "target_path": str(paths["repo_root"] / "bin" / "usd-cli"),
        },
    )

    capability = SimpleNamespace(
        route=telemetry_route,
        lease=object(),
        session=SimpleNamespace(workflow="large-scene", session_id="scene-session"),
        readiness=SimpleNamespace(
            version="usd-cli test",
            artifact_path=None,
            probe={},
        ),
        identity_path=(
            tmp_path / "run/raw/parent_usd_cli_session_scene-preflight.json"
        ),
        identity_sha256="",
        server_url="http://127.0.0.1:43210",
    )

    def fake_start_capability(**kwargs: object) -> SimpleNamespace:
        atomic_write_json(
            capability.identity_path,
            {"launch_id": "scene-preflight"},
            within=tmp_path / "run",
        )
        capability.identity_sha256 = file_sha256(capability.identity_path)
        capability_calls.append(kwargs)
        lifecycle.append("capability_start")
        return capability

    def fake_child(**kwargs: object) -> int:
        assert lifecycle == ["capability_start"]
        lifecycle.append("child")
        captured.update(kwargs)
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        scene_runner,
        "start_parent_usd_cli_capability",
        fake_start_capability,
    )
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda observed: (
            lifecycle.append("capability_stop") or _released_scene_teardown_evidence()
            if observed is capability
            else pytest.fail("scene cleanup did not use the shared capability")
        ),
    )
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    config = replace(
        _scene_config(paths, tmp_path / "run", dry_run=False),
        # A deliberately tiny launcher health budget: the OVRTX
        # readiness/render probe must not inherit it (PR #819 review pin —
        # a healthy cold OVRTX daemon start may legitimately take 60-600 s).
        scene_tool_timeout_seconds=0.25,
    )
    result = run_scene_workflow(config)

    assert result.returncode == 1
    # The probe budget covers the full OVRTX cold-start allowance
    # (OVRTX_DAEMON_START_TIMEOUT, 600 s) and stays independent of
    # ``--scene-tool-timeout``.
    assert scene_runner.USD_CLI_READINESS_TIMEOUT_SECONDS >= 600.0
    assert len(capability_calls) == 1
    capability_call = capability_calls[0]
    assert capability_call["run_dir"] == tmp_path / "run"
    assert capability_call["workflow"] == "scene.run"
    assert capability_call["session_workflow"] == "large-scene"
    staged_inputs_root = tmp_path / ".run.scene-inputs"
    assert capability_call["input_roots"] == (staged_inputs_root,)
    assert Path(capability_call["initial_scene"]).is_relative_to(staged_inputs_root)
    assert (
        capability_call["timeout_seconds"]
        == scene_runner.USD_CLI_READINESS_TIMEOUT_SECONDS
    )
    assert capability_call["required_capabilities"] == ("appearance.clear.v1",)
    assert lifecycle == [
        "capability_start",
        "child",
        "capability_stop",
    ]
    child_config = cast(SceneRunConfig, captured["config"])
    assert child_config.scene_tool_timeout_seconds == 0.25
    assert (
        child_config.parent_usd_cli_session_identity_sha256
        == capability.identity_sha256
    )


def test_scene_resume_usd_cli_leases_daemon_for_incomplete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config = _scene_config(paths, run_dir, dry_run=True)
    run_scene_workflow(config)

    lifecycle: list[str] = []
    target_path = paths["repo_root"] / "bin" / "usd-cli"
    telemetry_route = SimpleNamespace(
        active=True,
        target_path=target_path,
        reason=None,
        metadata=lambda: {
            "status": "active",
            "target_path": str(target_path),
        },
    )
    capability = SimpleNamespace(
        route=telemetry_route,
        lease=object(),
        session=SimpleNamespace(workflow="large-scene", session_id="scene-session"),
        readiness=SimpleNamespace(
            version="usd-cli test",
            artifact_path=None,
            probe={},
        ),
        identity_path=(run_dir / "raw/parent_usd_cli_session_scene-resume.json"),
        identity_sha256="",
        server_url="http://127.0.0.1:43210",
    )

    def fake_start_capability(**_kwargs: object) -> SimpleNamespace:
        atomic_write_json(
            capability.identity_path,
            {"launch_id": "scene-resume"},
            within=run_dir,
        )
        capability.identity_sha256 = file_sha256(capability.identity_path)
        lifecycle.append("capability_start")
        return capability

    monkeypatch.setattr(
        scene_runner,
        "start_parent_usd_cli_capability",
        fake_start_capability,
    )
    monkeypatch.setattr(
        scene_runner,
        "stop_parent_usd_cli_capability_strict",
        lambda _capability: lifecycle.append("capability_stop")
        or _released_scene_teardown_evidence(),
    )

    def fake_child(**kwargs: object) -> int:
        assert lifecycle == ["capability_start"]
        lifecycle.append("child")
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)
    result = scene_runner.resume_scene_workflow(run_dir)

    assert result.returncode == 1
    assert lifecycle == [
        "capability_start",
        "child",
        "capability_stop",
    ]


def test_scene_pre_staging_v2_resume_authorizes_verified_working_inputs(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = scene_runner._build_scene_request(
        replace(_scene_config(paths, run_dir, dry_run=False), agent_cwd=run_dir),
        run_id="scene-test",
        run_dir=run_dir,
        run_state_path=run_dir / "large_scene_run.json",
    )

    assert scene_runner._scene_parent_input_roots(run_dir, request=request) == (
        paths["source"],
        paths["materials_yaml"],
        paths["materials_usd"],
    )


def test_scene_resume_can_repeat_after_operation_trace_evolves(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    first = scene_runner.resume_scene_workflow(run_dir, dry_run=True)
    second = scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    assert first.returncode == 0
    assert second.returncode == 0
    manifest = json.loads(second.workflow_run_manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["checkpoints"]) == 3
    operation_trace_snapshots = [
        checkpoint["artifact_paths"]["operation_trace"]
        for checkpoint in manifest["checkpoints"]
    ]
    assert len(set(operation_trace_snapshots)) == 3


def test_scene_resume_rejects_symlinked_prompt_target(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    external = tmp_path / "external-prompt.md"
    external.write_text("host data\n", encoding="utf-8")
    _symlink_or_skip(run_dir / "agent_resume_prompt.md", external)
    manifest_before = (run_dir / "workflow_run_manifest.json").read_bytes()

    with pytest.raises(
        UnsafeRunArtifactError, match="Child-created symlinks are not allowed"
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    assert external.read_text(encoding="utf-8") == "host data\n"
    assert (run_dir / "workflow_run_manifest.json").read_bytes() == manifest_before


def test_scene_resume_legacy_request_and_state_migrate_to_usd_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    # Start from an actual checked-in pre-policy/pre-manifest v1 shape and bind
    # operator authorization to its exact materialized request bytes.
    legacy_request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    legacy_request_bytes = (run_dir / "request.json").read_bytes()
    legacy_state_bytes = (run_dir / "large_scene_run.json").read_bytes()

    captured: dict[str, object] = {}

    def fake_child(**kwargs: object) -> int:
        captured.update(kwargs)
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)

    result = scene_runner.resume_scene_workflow(
        run_dir,
        adopt_legacy_request_sha256=legacy_request_sha256,
    )

    assert result.returncode == 1
    resumed_config = captured["config"]
    assert resumed_config.scene_tool_timeout_seconds == 73.0  # type: ignore[union-attr]
    migrated_request = json.loads((run_dir / "request.json").read_text())
    migrated_state = load_run_state(run_dir / "large_scene_run.json")
    assert migrated_request["schema_version"] == (
        "content-agents.large-scene-request.v2"
    )
    assert migrated_request["child_workspace"] == str(run_dir)
    assert migrated_request["runtime"]["scene_backend"] == "usd-cli"
    assert migrated_request["runtime"]["scene_tool_timeout_seconds"] == 73.0
    assert not {
        "workbench_url",
        "start_workbench",
        "keep_workbench",
        "workbench_timeout_seconds",
    }.intersection(migrated_request["runtime"])
    assert migrated_request["tasks"][0]["scene_session_scope"] == "per_asset"
    assert migrated_state.schema_version == (
        "content-agent-workflows.large-scene-run.v2"
    )
    assert migrated_state.scene_backend == "usd-cli"
    assert "Frozen scene backend: `usd-cli`" in (
        run_dir / "agent_resume_prompt.md"
    ).read_text(encoding="utf-8")
    assert (run_dir / "workflow_run_manifest.json").is_file()
    receipt = json.loads(
        scene_runner._legacy_scene_adoption_receipt_path(run_dir).read_text(
            encoding="utf-8"
        )
    )
    legacy_request_path, legacy_state_path = scene_runner._legacy_scene_evidence_paths(
        run_dir
    )
    assert legacy_request_path.read_bytes() == legacy_request_bytes
    assert legacy_state_path.read_bytes() == legacy_state_bytes
    assert receipt["request_schema_before"].endswith("request.v1")
    assert receipt["request_schema_after"].endswith("request.v2")
    assert receipt["request_sha256_before"] == legacy_request_sha256
    assert receipt["request_sha256_after"] == file_sha256(run_dir / "request.json")
    assert (
        receipt["run_state_sha256_before"]
        == hashlib.sha256(legacy_state_bytes).hexdigest()
    )
    assert receipt["run_state_sha256_after"] == file_sha256(
        run_dir / "large_scene_run.json"
    )
    assert (
        receipt["source_input_digest_before"] != (receipt["source_input_digest_after"])
    )
    assert receipt["scene_backend"] == "usd-cli"
    policy = json.loads(
        scene_runner._scene_launcher_policy_path(run_dir).read_text(encoding="utf-8")
    )
    assert policy["legacy_adoption_receipt_sha256"] == file_sha256(
        scene_runner._legacy_scene_adoption_receipt_path(run_dir)
    )


def test_scene_resume_legacy_requires_explicit_digest(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _materialize_legacy_scene_fixture(paths, run_dir)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="Recognized a pre-policy v1 scene run.*--adopt-legacy-request-sha256",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    assert not scene_runner._legacy_scene_adoption_receipt_path(run_dir).exists()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()
    assert not (run_dir / "workflow_run_manifest.json").exists()
    assert not (run_dir / "agent_resume_prompt.md").exists()


def test_scene_resume_legacy_rejects_wrong_operator_digest(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _materialize_legacy_scene_fixture(paths, run_dir)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="does not match the operator-supplied",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256="0" * 64,
        )

    assert not scene_runner._legacy_scene_adoption_receipt_path(run_dir).exists()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()
    assert not (run_dir / "workflow_run_manifest.json").exists()


def test_scene_resume_legacy_rejects_mixed_v1_v2_state(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    legacy_request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    state_path = run_dir / "large_scene_run.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["schema_version"] = "content-agent-workflows.large-scene-run.v2"
    state_payload["scene_backend"] = "usd-cli"
    state_path.write_text(
        json.dumps(state_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="only genuine v1 state",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=legacy_request_sha256,
        )

    assert not scene_runner._legacy_scene_adoption_receipt_path(run_dir).exists()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()
    assert not (run_dir / "workflow_run_manifest.json").exists()


def test_scene_resume_legacy_rejects_unknown_request_version(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    _materialize_legacy_scene_fixture(paths, run_dir)
    request_path = run_dir / "request.json"
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    request_payload["schema_version"] = "content-agents.large-scene-request.v99"
    request_path.write_text(
        json.dumps(request_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="only a genuine v1 request",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_legacy_rejects_forged_source_input_digest(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    state_path = run_dir / "large_scene_run.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    forged = "0" * 64
    state_payload["source_input_digest"] = forged
    state_payload["phases"]["decomposition"]["input_digest"] = forged
    state_payload["transitions"][0]["input_digest"] = forged
    state_path.write_text(
        json.dumps(state_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="do not match the frozen v1 input digest",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_legacy_rejects_missing_required_input(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    paths["guidance"].unlink()

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="request input.*missing or cannot be inspected safely",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_legacy_rejects_symlinked_input(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    guidance_target = paths["repo_root"] / "guidance-target.md"
    paths["guidance"].replace(guidance_target)
    _symlink_or_skip(paths["guidance"], guidance_target)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="request input.*must not traverse a symlink",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_legacy_rejects_contradictory_partial_receipt(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    scene_runner._legacy_scene_adoption_receipt_path(run_dir).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="mixed or partially modern run.*migration intent is missing",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_legacy_adoption_is_interruption_safe_and_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    legacy_request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    original_write_policy = scene_runner._write_scene_launcher_policy
    monkeypatch.setattr(
        scene_runner,
        "_write_scene_launcher_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("interrupted after receipt")
        ),
    )

    with pytest.raises(RuntimeError, match="interrupted after receipt"):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=legacy_request_sha256,
        )

    assert scene_runner._legacy_scene_adoption_receipt_path(run_dir).is_file()
    assert not scene_runner._scene_launcher_policy_path(run_dir).exists()
    assert not (run_dir / "workflow_run_manifest.json").exists()

    monkeypatch.setattr(
        scene_runner,
        "_write_scene_launcher_policy",
        original_write_policy,
    )
    first = scene_runner.resume_scene_workflow(
        run_dir,
        dry_run=True,
        adopt_legacy_request_sha256=legacy_request_sha256,
    )
    second = scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert scene_runner._scene_launcher_policy_path(run_dir).is_file()
    manifest = json.loads(
        (run_dir / "workflow_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked"
    assert manifest["backend"]["scene_backend"] == "usd-cli"


def test_scene_resume_legacy_durably_syncs_parent_intent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    parent_directory_synced = False
    real_fsync_directory = scene_runner.fsync_directory

    def tracking_fsync_directory(path: Path) -> None:
        nonlocal parent_directory_synced
        if Path(path) == run_dir.parent:
            parent_directory_synced = True
        real_fsync_directory(path)

    monkeypatch.setattr(
        scene_runner,
        "fsync_directory",
        tracking_fsync_directory,
    )

    scene_runner.resume_scene_workflow(
        run_dir,
        dry_run=True,
        adopt_legacy_request_sha256=request_sha256,
    )

    assert parent_directory_synced is True


def test_scene_resume_legacy_rejects_tampered_recovery_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    original_write_policy = scene_runner._write_scene_launcher_policy
    monkeypatch.setattr(
        scene_runner,
        "_write_scene_launcher_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )
    monkeypatch.setattr(
        scene_runner,
        "_write_scene_launcher_policy",
        original_write_policy,
    )
    legacy_request_path, _legacy_state_path = scene_runner._legacy_scene_evidence_paths(
        run_dir
    )
    legacy_request_path.write_bytes(legacy_request_path.read_bytes() + b"\n")

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="does not match the operator-supplied",
    ):
        scene_runner.resume_scene_workflow(
            run_dir,
            dry_run=True,
            adopt_legacy_request_sha256=request_sha256,
        )


def test_scene_resume_migrated_run_rejects_tampered_parent_intent(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    scene_runner.resume_scene_workflow(
        run_dir,
        dry_run=True,
        adopt_legacy_request_sha256=request_sha256,
    )
    intent_path = scene_runner._legacy_scene_migration_intent_path(run_dir)
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["prepared_at"] = "2026-01-01T00:00:00Z"
    intent_path.write_text(json.dumps(intent, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="migration intent differs from the adoption receipt",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)


def test_scene_resume_migrated_run_rejects_tampered_receipt(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)
    scene_runner.resume_scene_workflow(
        run_dir,
        dry_run=True,
        adopt_legacy_request_sha256=request_sha256,
    )
    receipt_path = scene_runner._legacy_scene_adoption_receipt_path(run_dir)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["adopted_at"] = "2026-01-01T00:00:00Z"
    receipt_path.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scene_runner.SceneLauncherPolicyError,
        match="receipt differs from the parent-owned policy",
    ):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)


def test_scene_resume_cli_adopts_legacy_with_exact_request_digest(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    legacy_request_sha256 = _materialize_legacy_scene_fixture(paths, run_dir)

    exit_code = main(
        [
            "scene",
            "resume",
            "--run-dir",
            str(run_dir),
            "--adopt-legacy-request-sha256",
            legacy_request_sha256,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert scene_runner._legacy_scene_adoption_receipt_path(run_dir).is_file()
    assert scene_runner._scene_launcher_policy_path(run_dir).is_file()
    assert (run_dir / "workflow_run_manifest.json").is_file()


def test_scene_resume_completed_legacy_run_does_not_require_new_backend_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    legacy_request_sha256 = _materialize_legacy_scene_fixture(
        paths,
        run_dir,
        completed=True,
    )

    monkeypatch.setattr(
        scene_runner,
        "_run_child_agent",
        lambda **_kwargs: pytest.fail("A completed legacy run must not be replayed"),
    )

    result = scene_runner.resume_scene_workflow(
        run_dir,
        adopt_legacy_request_sha256=legacy_request_sha256,
    )

    assert result.returncode == 0
    assert result.completed is True
    manifest = json.loads(result.workflow_run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["backend"]["scene_backend"] == "usd-cli"
    assert manifest["backend"]["renderer_probe_required"] is True
    assert "legacy_adoption_receipt" in manifest["required_artifacts"]
    assert "legacy_request_evidence" in manifest["required_artifacts"]
    assert "legacy_run_state_evidence" in manifest["required_artifacts"]
    logical_names = {artifact["logical_name"] for artifact in manifest["artifacts"]}
    assert "legacy_request_evidence" in logical_names
    assert "legacy_run_state_evidence" in logical_names


def test_scene_run_success_seals_output_evidence_and_probe_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    collection_output = run_dir / "collection" / "result.json"
    validation_calls = 0

    def fake_child(**kwargs: object) -> int:
        collection_output.parent.mkdir(parents=True)
        staged_source = (
            run_dir.parent
            / f".{run_dir.name}.scene-inputs"
            / "source"
            / paths["source"].name
        )
        collection_output.write_text(
            json.dumps({"artifact_paths": [str(staged_source)]}) + "\n",
            encoding="utf-8",
        )
        state_path = run_dir / "large_scene_run.json"
        state = load_run_state(state_path)
        state.current_phase = None
        for phase in state.phases.values():
            phase.status = "completed"
        state.phases["collection"].result_path = str(collection_output)
        state_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    def validate_terminal(_path: Path) -> scene_runner.SceneTerminalValidation:
        nonlocal validation_calls
        validation_calls += 1
        return scene_runner.SceneTerminalValidation(
            checked_at="2026-07-27T00:00:00Z",
            valid=True,
            current_phase=None,
            phase_statuses={
                "decomposition": "completed",
                "asset_task_processing": "completed",
                "collection": "completed",
            },
            errors=[],
            final_handoff={"valid": True},
        )

    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(scene_runner, "_validate_terminal_state", validate_terminal)

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 0
    assert result.completed is True
    manifest = json.loads(result.workflow_run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["failure"] is None
    assert manifest["backend"]["renderer_probe_required"] is True
    assert manifest["backend"]["backend_readiness_evidence"] == ("ovrtx_render_probe")
    artifact_by_name = {
        artifact["logical_name"]: artifact for artifact in manifest["artifacts"]
    }
    assert set(manifest["required_artifacts"]) <= set(artifact_by_name)
    for logical_name in manifest["required_artifacts"]:
        artifact = artifact_by_name[logical_name]
        assert artifact["sha256"] == file_sha256(run_dir / artifact["path"])
    assert manifest["checkpoints"][-1]["phase"] == "validated"
    assert manifest["checkpoints"][-1]["sealed"] is True
    teardown_path = run_dir / artifact_by_name["usd_cli_teardown"]["path"]
    teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
    assert teardown["status"] == "released"
    assert teardown["boundary"] == "completed"
    assert teardown["state_directory_released"] is True
    assert validation_calls == 1

    resumed = scene_runner.resume_scene_workflow(run_dir)

    assert resumed.returncode == 0
    assert resumed.completed is True
    assert validation_calls == 2


def test_scene_run_never_reads_escaping_collection_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    external_result = tmp_path / "external-result.json"
    external_result.write_text('{"artifact_paths": []}\n', encoding="utf-8")
    collection_output = run_dir / "collection" / "result.json"

    def fake_child(**kwargs: object) -> int:
        collection_output.parent.mkdir(parents=True)
        _symlink_or_skip(collection_output, external_result)
        state_path = run_dir / "large_scene_run.json"
        state = load_run_state(state_path)
        state.current_phase = None
        for phase in state.phases.values():
            phase.status = "completed"
        state.phases["collection"].result_path = str(collection_output)
        state_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        Path(str(kwargs["child_output_path"])).write_text("child\n", encoding="utf-8")
        Path(str(kwargs["child_final_path"])).write_text("done\n", encoding="utf-8")
        return 0

    original_load_json = scene_runner.load_json

    def guarded_load_json(path: str | Path) -> dict[str, object]:
        if Path(path).resolve() == external_result.resolve():
            pytest.fail("The parent must not read an escaping collection output")
        return original_load_json(path)

    monkeypatch.setattr(scene_runner, "load_json", guarded_load_json)
    _stub_usd_cli_scene_runtime(monkeypatch, paths)
    monkeypatch.setattr(scene_runner, "_run_child_agent", fake_child)
    monkeypatch.setattr(
        scene_runner,
        "_validate_terminal_state",
        lambda _path: scene_runner.SceneTerminalValidation(
            checked_at="2026-07-29T00:00:00Z",
            valid=True,
            current_phase=None,
            phase_statuses={
                "decomposition": "completed",
                "asset_task_processing": "completed",
                "collection": "completed",
            },
            errors=[],
            final_handoff={"valid": True},
        ),
    )

    result = run_scene_workflow(_scene_config(paths, run_dir, dry_run=False))

    assert result.returncode == 1
    assert result.completed is False
    manifest = json.loads(result.workflow_run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "fail"
    assert "collection_output" in str(manifest["failure"]["artifact_errors"])


def test_scene_resume_rejects_source_and_policy_drift(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))

    paths["materials_yaml"].write_text(
        'library_path: "different.usd"\nentries: []\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy"):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)

    paths["materials_yaml"].write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    paths["source"].write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)


def test_scene_terminal_rejects_matching_staged_input_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    manifest_path = run_dir / "raw" / "staged_input_source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    staged_source = Path(manifest["staged_usd_path"])
    os.chmod(staged_source, staged_source.stat().st_mode | 0o200)
    staged_source.write_text("#usda 1.0\n# child replacement\n", encoding="utf-8")
    replacement_sha256 = file_sha256(staged_source)
    for item in manifest["files"]:
        if item["staged_path"] == str(staged_source):
            item["sha256"] = replacement_sha256
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    assert scene_runner._staged_input_integrity_errors(run_dir)
    terminal = scene_runner._validate_terminal_state(run_dir / "large_scene_run.json")

    assert terminal.valid is False
    assert any(
        "Frozen scene source or request input changed" in e for e in terminal.errors
    )


def test_scene_resume_rejects_checkpoint_artifact_tampering(
    tmp_path: Path,
) -> None:
    paths = _write_scene_inputs(tmp_path)
    run_dir = tmp_path / "run"
    prepared = run_scene_workflow(_scene_config(paths, run_dir, dry_run=True))
    manifest = json.loads(
        prepared.workflow_run_manifest_path.read_text(encoding="utf-8")
    )
    prompt_snapshot = run_dir / manifest["checkpoints"][-1]["artifact_paths"]["prompt"]
    prompt_snapshot.chmod(0o600)
    prompt_snapshot.write_text("tampered prompt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_artifacts"):
        scene_runner.resume_scene_workflow(run_dir, dry_run=True)


def test_scene_request_v2_requires_explicit_backend_contract(tmp_path: Path) -> None:
    paths = _write_scene_inputs(tmp_path)
    prepared = run_scene_workflow(_scene_config(paths, tmp_path / "run", dry_run=True))
    request_payload = json.loads(prepared.request_path.read_text(encoding="utf-8"))
    request_payload["runtime"].pop("scene_backend")

    with pytest.raises(ValueError, match="scene_backend"):
        scene_runner.SceneRunRequest.model_validate(request_payload)
