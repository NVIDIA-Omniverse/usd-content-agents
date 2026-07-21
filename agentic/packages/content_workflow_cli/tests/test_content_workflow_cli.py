# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI tests for content-workflow-cli."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_workflow_cli import runner as workflow_runner
from content_workflow_cli.cli import (
    _default_codex_sandbox_mode,
    _handle_auth_status,
    _load_claude_config,
    _parse_json_object,
    _resolve_materials_usd_from_manifest,
    _should_start_workbench,
    build_parser,
    main,
)
from content_workflow_cli.workbench_tools.snapshot_scene import (
    MaterialCandidatePolicy,
    _remap_instance_source_target,
    compact_summary,
    write_snapshot_artifacts,
)


class _FakeCodexProbeProcess:
    pid = 4321

    def __init__(
        self,
        *,
        returncode: int | None,
        stdout: str = "",
        stderr: str = "",
        timeout_once: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timeout_once = timeout_once
        self.communicate_timeouts: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_timeouts.append(timeout)
        if self.timeout_once and len(self.communicate_timeouts) == 1:
            raise subprocess.TimeoutExpired("codex-test", timeout=timeout or 0)
        if self.returncode is None:
            self.returncode = -signal.SIGKILL
        return self.stdout, self.stderr

    def wait(self) -> int:
        assert self.returncode is not None
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -signal.SIGKILL


def test_auth_status_probes_model_usability(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_calls: list[list[str]] = []
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    probe_process = _FakeCodexProbeProcess(returncode=0, stdout="OK\n")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        run_calls.append(command)
        return SimpleNamespace(returncode=0)

    def fake_popen(command: list[str], **kwargs: object) -> _FakeCodexProbeProcess:
        popen_calls.append((command, kwargs))
        return probe_process

    monkeypatch.setattr("content_workflow_cli.cli.subprocess.run", fake_run)
    monkeypatch.setattr("content_workflow_cli.cli.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "content_workflow_cli.cli._codex_executable", lambda: "codex-test"
    )

    assert _handle_auth_status(SimpleNamespace()) == 0
    assert run_calls == [["codex-test", "login", "status"]]
    probe_command, probe_kwargs = popen_calls[0]
    assert probe_command[0:2] == ["codex-test", "exec"]
    assert "--ephemeral" in probe_command
    assert "--ignore-user-config" in probe_command
    assert "--skip-git-repo-check" in probe_command
    assert ["--sandbox", "read-only"] == probe_command[
        probe_command.index("--sandbox") : probe_command.index("--sandbox") + 2
    ]
    probe_cwd = probe_command[probe_command.index("--cd") + 1]
    assert Path(probe_cwd).name.startswith("content-workflow-codex-auth-")
    assert probe_kwargs["stdin"] is subprocess.DEVNULL
    assert probe_kwargs["start_new_session"] is True
    assert probe_process.communicate_timeouts == [60]
    assert "Codex login is usable for model calls." in capsys.readouterr().out


def test_auth_status_fails_when_login_cannot_call_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe_process = _FakeCodexProbeProcess(
        returncode=1,
        stderr="HTTP 400: model calls are not supported for this account",
    )

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("content_workflow_cli.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "content_workflow_cli.cli.subprocess.Popen",
        lambda *args, **kwargs: probe_process,
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._codex_executable", lambda: "codex-test"
    )

    assert _handle_auth_status(SimpleNamespace()) == 1
    captured = capsys.readouterr()
    assert "cannot complete a model call" in captured.err
    assert "HTTP 400" in captured.err


def test_auth_status_fails_when_model_probe_times_out(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Model the Node launcher exiting while its native child still holds the
    # captured pipes open. Cleanup must signal the process group even though
    # the launcher itself already has a return code.
    probe_process = _FakeCodexProbeProcess(returncode=0, timeout_once=True)
    killpg_calls: list[tuple[int, int]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("content_workflow_cli.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "content_workflow_cli.cli.subprocess.Popen",
        lambda *args, **kwargs: probe_process,
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli.os.killpg",
        lambda pid, sig: killpg_calls.append((pid, sig)),
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._codex_executable", lambda: "codex-test"
    )

    assert _handle_auth_status(SimpleNamespace()) == 1
    assert "model usability probe timed out" in capsys.readouterr().err
    assert killpg_calls == [(probe_process.pid, signal.SIGKILL)]
    assert probe_process.communicate_timeouts == [60, None]


def test_materials_assign_dry_run_writes_contract(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usdc"
    reference = tmp_path / "reference.png"
    reference_pdf = tmp_path / "reference.pdf"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    codex_config = tmp_path / "codex-config.json"
    instructions = tmp_path / "material-guidance.md"
    for path in [usd, reference, reference_pdf, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    codex_config.write_text(
        json.dumps(
            {
                "model_provider": "proxy",
                "model_providers": {
                    "proxy": {
                        "name": "Proxy",
                        "base_url": "https://proxy.example.com/v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    instructions.write_text(
        "White structural frames.\nReserve yellow for safety accents.\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    exit_code = main(
        [
            "materials",
            "assign",
            "--usd",
            str(usd),
            "--reference-image",
            str(reference),
            "--reference",
            str(reference_pdf),
            "--materials-yaml",
            str(materials_yaml),
            "--workbench-url",
            "http://127.0.0.1:8088",
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(run_dir),
            "--model",
            "gpt-5.6-sol",
            "--model-reasoning-effort",
            "ultra",
            "--codex-base-url",
            "https://codex-proxy.example.com/v1",
            "--codex-sandbox-mode",
            "workspace-write",
            "--codex-config-file",
            str(codex_config),
            "--codex-config-json",
            '{"model_providers":{"proxy":{"wire_api":"responses"}}}',
            "--child-timeout",
            "120",
            "--additional-instructions-file",
            str(instructions),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["workflow"] == "materials.assign"
    assert request["dry_run"] is True
    assert request["workbench_optimize"] is True
    assert request["optimizer_options"] == {
        "flatten_prototypes": None,
        "enable_deinstance": None,
        "enable_split": None,
        "enable_deduplicate": None,
    }
    assert request["material_candidate_policy"] == {
        "material_candidate_space": "source",
        "root_prim_path": None,
        "skip_instances": True,
        "skip_prototypes": False,
        "skip_invisible": False,
    }
    assert request["runner"] == "codex"
    assert request["model"] == "gpt-5.6-sol"
    assert request["model_reasoning_effort"] == "ultra"
    assert request["codex_base_url"] == "https://codex-proxy.example.com/v1"
    assert request["codex_sandbox_mode"] == "workspace-write"
    assert request["child_timeout_seconds"] == 120
    assert request["vqa_refinement_max_iterations"] == 3
    assert request["codex_persistent_refinement"] is False
    assert request["additional_instructions"] == (
        "White structural frames.\nReserve yellow for safety accents."
    )
    assert "convergence" not in request
    assert request["inputs"]["materials_yaml"] == str(materials_yaml)
    assert request["inputs"]["materials_usd"] == str(materials_usd)
    assert request["codex_config"] == {
        "model_provider": "proxy",
        "model_providers": {
            "proxy": {
                "name": "Proxy",
                "base_url": "https://proxy.example.com/v1",
                "wire_api": "responses",
            }
        },
    }
    assert request["constraints"]["source_usd_edits_allowed"] is False
    assert request["constraints"]["material_candidate_policy"] == {
        "material_candidate_space": "source",
        "root_prim_path": None,
        "skip_instances": True,
        "skip_prototypes": False,
        "skip_invisible": False,
    }
    assert request["inputs"]["reference_images"] == [str(reference)]
    assert request["inputs"]["reference_files"] == [str(reference_pdf)]

    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "material_override" in prompt
    assert '"space":"<source-or-inspection>"' in prompt
    assert "Use only Workbench API calls" in prompt
    assert "quick navigation at or below 640x480" in prompt
    assert "evidence/final verification at or below 768x576" in prompt
    assert 'render_quality: "inspection"' in prompt
    assert "HDRI-600-only default" in prompt
    assert 'ovrtx_render_mode: "rt2"' in prompt
    assert "optimize: true" in prompt
    assert "content-workbench-snapshot-scene" in prompt
    assert '"material_candidate_space": "source"' in prompt
    assert "--include-instances" in prompt
    assert "/scene/snapshot" in prompt
    assert "--materials-yaml" in prompt
    assert "raw/material_authoring_context.md" in prompt
    assert "raw/material_assignment_seed.json" in prompt
    assert "raw/visible_candidate_prims.json" in prompt
    assert "raw/visible_candidate_table.tsv" in prompt
    assert "raw/material_palette.json" in prompt
    assert "Workbench API quick contract" in prompt
    assert "Pick uses the current session camera" in prompt
    assert "render-response camera JSON" in prompt
    assert "matching camera/view fields" not in prompt
    assert "closest opaque/surface-compatible visual proxy" in prompt
    assert "dominant gray/white/black mismatches" in prompt
    assert "Do not open, grep, `sed`, or `jq` saved Workbench docs" in prompt
    assert (
        "Do not read local skill docs, README files, or prior run summaries" in prompt
    )
    assert "Avoid broad `jq`, `sed`, or Python inspection" in prompt
    assert "In clean-slate mode, visible material candidates need explicit" in prompt
    assert "Literally iterate the candidate list during prediction" in prompt
    assert "Material assignments are not capped by prim count" in prompt
    assert "reference_files" in prompt
    assert str(reference_pdf) in prompt
    assert "12 pick calls" in prompt
    assert "/properties:batch" in prompt
    assert "/material-binding:batch" in prompt
    assert "/paths/translate:batch" in prompt
    assert "coverage_status" in prompt
    assert "preserved_existing" in prompt
    assert "ambiguous_unassigned" in prompt
    assert "Termination goal" in prompt
    assert "candidate_visible_prim_count == material_decision_prim_count" in prompt
    assert "raw/visible_candidate_prims.json" in prompt
    assert "final review/remediation pass" in prompt
    assert "VQA refinement iteration 1 of 3" in prompt
    assert (run_dir / "trace" / "operation_trace.json").exists()
    assert (run_dir / "trace" / "run_retrospective.json").exists()
    assert (run_dir / "trace" / "replay_manifest.json").exists()


def test_materials_assign_cli_rejects_symlinked_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usdc"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in (usd, reference, materials_usd):
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    lexical_run_dir = tmp_path / "run"
    lexical_run_dir.symlink_to(outside, target_is_directory=True)

    exit_code = main(
        [
            "materials",
            "assign",
            "--usd",
            str(usd),
            "--reference-image",
            str(reference),
            "--materials-yaml",
            str(materials_yaml),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(lexical_run_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "must resolve without traversing symlinks" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [
        [
            "materials",
            "assign",
            "--usd",
            "asset.usd",
            "--materials-yaml",
            "materials.yaml",
        ],
        [
            "physics",
            "apply",
            "--usd",
            "asset.usd",
            "--output-dir",
            "physics-output",
        ],
    ],
)
def test_model_and_reasoning_effort_are_provider_passthrough(args: list[str]) -> None:
    parsed = build_parser().parse_args(
        [
            *args,
            "--model",
            "provider-future-model",
            "--model-reasoning-effort",
            "provider-future-effort",
        ]
    )

    assert parsed.model == "provider-future-model"
    assert parsed.model_reasoning_effort == "provider-future-effort"


def test_physics_apply_dry_run_writes_visual_validation_contract(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    run_dir = tmp_path / "physics-run"

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(run_dir),
            "--workbench-url",
            "http://127.0.0.1:8088",
            "--visual-validation-max-iterations",
            "2",
            "--additional-instructions",
            "Treat the base as static.",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["workflow"] == "physics.apply"
    assert request["visual_validation_max_iterations"] == 2
    assert request["additional_instructions"] == "Treat the base as static."
    assert not (run_dir / "raw" / "physics_topology_plan.json").exists()
    assert "physics_behavior_assessment.json" in prompt
    assert "ovphysx/runtime metrics are authoritative" in prompt


def test_physics_apply_cli_rejects_symlinked_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    lexical_run_dir = tmp_path / "physics-run"
    lexical_run_dir.symlink_to(outside, target_is_directory=True)

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(lexical_run_dir),
            "--workbench-url",
            "http://127.0.0.1:8088",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "must resolve without traversing symlinks" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


def test_physics_preflight_closes_workbench_session_on_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[tuple[str, str, float]] = []

    monkeypatch.setattr(
        workflow_runner.workbench_client,
        "download_agent_api_docs",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        workflow_runner.workbench_client,
        "create_session",
        lambda *_args, **_kwargs: {"session_id": "session-one"},
    )

    def fail_inspect_components(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("inspection failed")

    monkeypatch.setattr(
        workflow_runner.workbench_client,
        "inspect_physics_components",
        fail_inspect_components,
    )

    def fake_close_session(
        workbench_url: str,
        session_id: str,
        *,
        timeout: float,
    ) -> None:
        closed.append((workbench_url, session_id, timeout))

    monkeypatch.setattr(workflow_runner, "close_workbench_session", fake_close_session)

    with pytest.raises(RuntimeError, match="inspection failed"):
        workflow_runner._prepare_physics_run_packet(
            workflow_runner.PhysicsApplyConfig(
                repo_root=tmp_path,
                usd_path=tmp_path / "asset.usda",
                workbench_url="http://127.0.0.1:8088",
                workbench_timeout_seconds=12.0,
            ),
            tmp_path / "run",
        )

    assert closed == [("http://127.0.0.1:8088", "session-one", 12.0)]


def test_physics_apply_json_dry_run_keeps_stdout_machine_readable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    run_dir = tmp_path / "physics-json-run"

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--output-dir",
            str(run_dir),
            "--simulation-engine",
            "fake",
            "--json",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["run_dir"] == str(run_dir)
    assert payload["returncode"] == 0
    assert captured.out.lstrip().startswith("{")
    assert "content-workflow-cli: run directory:" in captured.err


def test_model_reasoning_effort_rejects_max(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["scene", "run", "--model-reasoning-effort", "max"])

    assert error.value.code == 2
    assert (
        "unsupported model reasoning effort 'max'; use 'xhigh'"
        in capsys.readouterr().err
    )


def test_physics_deterministic_apply_failure_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows import physics as physics_workflows

    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    run_dir = tmp_path / "physics-run"

    def fake_run_physics_apply_workflow(
        _params: physics_workflows.PhysicsApplyWorkflowInput,
    ) -> physics_workflows.PhysicsApplyWorkflowResult:
        return physics_workflows.PhysicsApplyWorkflowResult(
            success=False,
            asset=str(usd),
            output_dir=str(run_dir),
            validation_status="fail",
            error="No mesh prims found.",
        )

    monkeypatch.setattr(
        physics_workflows,
        "run_physics_apply_workflow",
        fake_run_physics_apply_workflow,
    )

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--output-dir",
            str(run_dir),
            "--deterministic-workflow",
            "--no-simulation",
        ]
    )

    assert exit_code == 1


def test_physics_apply_rejects_session_id_without_deterministic_workflow(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--output-dir",
            str(tmp_path / "physics-run"),
            "--workbench-session-id",
            "existing-session",
            "--dry-run",
        ]
    )

    assert exit_code == 2


def test_convert_to_usd_defaults_output_to_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["convert-to-usd", str(source), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "passed"
    assert payload["source_format"] == "usd"
    assert payload["output_usd_path"] == str((tmp_path / "asset.usda").resolve())
    assert (tmp_path / "asset.usda").read_text(encoding="utf-8") == "#usda 1.0\n"


def test_convert_to_usd_installs_missing_dependencies_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import content_agent_workflows.convert_to_usd as convert_package
    from content_agent_workflows.convert_to_usd import (
        ConversionProbeArtifact,
        ConversionReport,
    )

    source = tmp_path / "robot.urdf"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    called: dict[str, bool] = {}

    def fake_convert_source_to_usd_file(
        source_asset: Path,
        output_usd_path: Path,
        *,
        output_format: str | None = None,
        install_missing: bool = False,
    ) -> tuple[ConversionReport, ConversionProbeArtifact]:
        called["install_missing"] = install_missing
        called["output_format"] = output_format == "usdc"
        output_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
        return (
            ConversionReport(
                status="passed",
                source_asset_path=str(source_asset),
                source_format="urdf",
                converter_skill="convert-to-usd",
                converter_reference="urdf-usd-converter",
                converter_tool="urdf_usd_converter",
                output_directory=str(output_usd_path.parent),
                output_usd_path=str(output_usd_path),
                generated_files=[output_usd_path.name],
            ),
            ConversionProbeArtifact(
                source_asset_path=str(source_asset),
                reference_order=[],
                selected_converter="urdf-usd-converter",
            ),
        )

    monkeypatch.setattr(
        convert_package,
        "convert_source_to_usd_file",
        fake_convert_source_to_usd_file,
    )

    exit_code = main(["convert-to-usd", str(source), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert called["install_missing"] is True
    assert called["output_format"] is False
    assert payload["status"] == "passed"


def test_convert_to_usd_output_format_defaults_output_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from content_agent_workflows.convert_to_usd import workflow as convert_workflow

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_export(_source_usd: Path, output_usd: Path) -> None:
        output_usd.write_bytes(b"PXR-USDC fake\n")

    monkeypatch.setattr(convert_workflow, "_export_usd_layer", fake_export)

    exit_code = main(
        ["convert-to-usd", str(source), "--output-format", "usdc", "--json"]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "passed"
    assert payload["output_usd_path"] == str((tmp_path / "asset.usdc").resolve())
    assert payload["output_format"] == "usdc"
    assert (tmp_path / "asset.usdc").read_bytes().startswith(b"PXR-USDC")


def test_convert_to_usd_output_format_rejects_mismatched_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    exit_code = main(
        [
            "convert-to-usd",
            str(source),
            str(tmp_path / "asset.usda"),
            "--output-format",
            "usdc",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "conflicts with requested output format" in captured.err


def test_convert_to_usd_preflight_cli_installs_inferred_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from content_agent_workflows.convert_to_usd import workflow as convert_workflow

    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    available = {"value": False}

    def fake_dependency_available(_converter_reference: str) -> bool:
        return available["value"]

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        available["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        convert_workflow,
        "_dependency_available",
        fake_dependency_available,
    )
    monkeypatch.setattr(convert_workflow.subprocess, "run", fake_run)

    exit_code = main(["preflight", "convert-to-usd", str(source), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "passed"
    assert payload["converter_reference"] == "usd-convert-cad"
    assert payload["install_attempted"] is True
    assert convert_workflow.USD_CONVERT_CAD_INSTALL_SPEC in payload["install_command"]


def test_convert_to_usd_output_dir_writes_artifacts_without_changing_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    cwd = tmp_path / "cwd"
    run_dir = tmp_path / "run"
    source_dir.mkdir()
    cwd.mkdir()
    source = source_dir / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    exit_code = main(
        [
            "convert-to-usd",
            str(source),
            "--output-dir",
            str(run_dir),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["output_usd_path"] == str((cwd / "asset.usda").resolve())
    assert (cwd / "asset.usda").exists()
    assert (run_dir / "request.json").exists()
    assert (run_dir / "conversion_report.json").exists()


def test_materials_assign_dry_run_supports_skill_routed_prompt(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usdc"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    run_dir = tmp_path / "run"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "materials",
            "assign",
            "--usd",
            str(usd),
            "--reference-image",
            str(reference),
            "--materials-yaml",
            str(materials_yaml),
            "--output-dir",
            str(run_dir),
            "--prompt-mode",
            "skill-routed",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["prompt_mode"] == "skill-routed"
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert "`content-workbench`" in prompt
    assert "`content-workflow-material`" in prompt
    assert "Structured task:" in prompt
    assert "Workbench API quick contract" not in prompt


def test_snapshot_scene_tool_writes_standard_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    materials_usd.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        json.dumps(
            {
                "library_path": "materials.usd",
                "entries": [
                    {
                        "name": "Plastic Orange",
                        "description": "Smooth orange plastic with light gloss",
                        "binding": "/World/Looks/Plastic_Orange",
                    },
                    {
                        "name": "Rubber Black Matte",
                        "description": "Matte black rubber surface",
                        "binding": "/World/Looks/Rubber_Black_Matte",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot = {
        "session_id": "session-1",
        "root_prim_path": "/World",
        "source_scene_path": "/tmp/source.usdc",
        "inspection_scene_path": "/tmp/inspection.usdc",
        "paths": ["/World", "/World/Step", "/World/Part", "/World/Part/Geometry"],
        "nodes": [
            {
                "path": "/World",
                "name": "World",
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "children": True,
                "child_paths": ["/World/Step", "/World/Part"],
            },
            {
                "path": "/World/Step",
                "name": "Step",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
            {
                "path": "/World/Part",
                "name": "Part",
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "children": True,
                "child_paths": ["/World/Part/Geometry"],
            },
            {
                "path": "/World/Part/Geometry",
                "name": "Geometry",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
        ],
        "properties": [{"prim_path": "/World/Step", "properties": {}}],
        "material_bindings": [
            {
                "prim_path": "/World/Step",
                "binding_type": "direct",
                "bound_material_path": "/World/Looks/Blue",
                "binding_source_path": "/World/Step",
                "relationship_path": "/World/Step.material:binding",
                "direct_targets": ["/World/Looks/Blue"],
                "material_override": None,
            }
        ],
        "path_translations": [
            {
                "session_id": "session-1",
                "input_path": "/World/Step",
                "source_space": "inspection",
                "target_space": "source",
                "source_paths": ["/World/Step"],
                "inspection_paths": ["/World/Step"],
                "ambiguous": False,
                "optimization": {"enabled": False, "status": "disabled"},
            }
        ],
        "candidates": [
            {
                "inspection_path": "/World",
                "source_paths": ["/World"],
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [1.0, 1.0, 0.0],
                "material_binding_type": "direct",
                "bound_material_path": "/World/Looks/Blue",
                "binding_source_path": "/World",
                "direct_targets": ["/World/Looks/Blue"],
                "material_override": None,
                "ambiguous_translation": False,
                "candidate_reason": "material_bound_container",
            },
            {
                "inspection_path": "/World/Part",
                "source_paths": ["/World/Part"],
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [0.5, 0.5, 0.5],
                "material_binding_type": "direct",
                "bound_material_path": None,
                "binding_source_path": "/World/Part",
                "direct_targets": ["/World/Looks/Rubber_Black_Matte"],
                "material_override": None,
                "ambiguous_translation": False,
                "candidate_reason": "material_bound_container",
            },
            {
                "inspection_path": "/World/Step",
                "source_paths": ["/World/Step"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [1.0, 1.0, 0.0],
                "material_binding_type": "direct",
                "bound_material_path": "/World/Looks/Blue",
                "binding_source_path": "/World/Step",
                "direct_targets": ["/World/Looks/Blue"],
                "material_override": None,
                "ambiguous_translation": False,
                "candidate_reason": "renderable_prim",
            },
            {
                "inspection_path": "/World/Part/Geometry",
                "source_paths": ["/World/Part"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [0.5, 0.5, 0.5],
                "material_binding_type": "none",
                "bound_material_path": None,
                "binding_source_path": "/World/Part",
                "direct_targets": [],
                "material_override": None,
                "ambiguous_translation": False,
                "candidate_reason": "renderable_prim",
            },
        ],
        "excluded_non_candidates": [],
        "summary": {
            "prim_count": 4,
            "candidate_count": 4,
            "ambiguous_translation_count": 0,
            "truncated": False,
        },
    }

    artifacts = write_snapshot_artifacts(
        snapshot,
        run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
    )

    assert set(artifacts) == {
        "scene_snapshot",
        "tree_paths",
        "properties",
        "material_bindings",
        "path_translations",
        "visible_candidates_preliminary",
        "visible_candidates",
        "material_authoring_context",
        "material_authoring_context_md",
        "visible_candidate_table",
        "material_palette",
        "material_assignment_seed",
    }
    assert (
        json.loads((run_dir / "raw" / "scene_snapshot.json").read_text())["session_id"]
        == "session-1"
    )
    tree = json.loads((run_dir / "raw" / "tree_paths.json").read_text())
    assert tree["nodes"][0]["children_count"] == 2
    candidates = json.loads(
        (run_dir / "raw" / "visible_candidate_prims_preliminary.json").read_text()
    )
    assert candidates["candidate_visible_prim_count"] == 4
    assert candidates["candidates"][0]["inspection_path"] == "/World"
    visible = json.loads((run_dir / "raw" / "visible_candidate_prims.json").read_text())
    assert visible["candidate_visible_prim_count"] == 2
    assert visible["path_space"] == "source"
    assert [candidate["source_path"] for candidate in visible["candidates"]] == [
        "/World/Part",
        "/World/Step",
    ]
    assert "/World" in visible["excluded_non_candidates"]
    assert "/World/Part" not in visible["excluded_non_candidates"]
    context = json.loads(
        (run_dir / "raw" / "material_authoring_context.json").read_text()
    )
    assert context["summary"]["candidate_count"] == 2
    assert context["summary"]["preliminary_candidate_count"] == 4
    assert (
        context["material_binding_policy"]["respect_existing_material_bindings"]
        is False
    )
    assert {group["grouping_basis"] for group in context["candidate_groups"]} == {
        "authoring_family"
    }
    assert {group["material_name"] for group in context["candidate_groups"]} == {None}
    assert {
        group["recommended_coverage_status"] for group in context["candidate_groups"]
    } == {"ambiguous_unassigned"}
    assert context["candidate_groups"][0]["requires_material_assignment"] is False
    assert (
        context["candidates"][1]["recommended_initial_status"] == "ambiguous_unassigned"
    )
    assert context["candidates"][1]["requires_material_assignment"] is False
    assert context["material_palette"]["material_count"] == 2
    assert context["material_palette"]["materials"][0]["name"] == "Plastic Orange"
    assert context["material_palette"]["materials"][0]["manifest_semantics"][
        "colors"
    ] == ["orange"]
    assert context["material_palette"]["materials"][0]["manifest_semantics"][
        "substances"
    ] == ["plastic"]
    candidate_table = (run_dir / "raw" / "visible_candidate_table.tsv").read_text()
    assert (
        "runtime_path\truntime_paths\truntime_space\tsource_path\tsource_paths\t"
        "original_source_paths\tinspection_path\tinspection_paths"
    ) in candidate_table
    assert "/World/Step" in candidate_table
    assert "/World/Part/Geometry" in candidate_table
    assignment_seed = json.loads(
        (run_dir / "raw" / "material_assignment_seed.json").read_text()
    )
    assert assignment_seed["coverage"]["candidate_visible_prim_count"] == 2
    assert assignment_seed["coverage"]["preserved_existing_prim_count"] == 0
    assert assignment_seed["coverage"]["ambiguous_unassigned_prim_count"] == 2
    assert {
        assignment["coverage_status"] for assignment in assignment_seed["assignments"]
    } == {"ambiguous_unassigned"}
    assert sorted(
        path
        for assignment in assignment_seed["assignments"]
        for path in assignment["prim_paths"]
    ) == ["/World/Part", "/World/Step"]
    context_md = (run_dir / "raw" / "material_authoring_context.md").read_text()
    assert "Material Authoring Context" in context_md
    assert (
        "Existing material bindings and authored display colors are cleared"
        in context_md
    )
    assert "Plastic Orange" in context_md
    assert "coverage evidence, not a material assignment plan" in context_md
    trace = (run_dir / "trace" / "events.jsonl").read_text()
    assert "POST /sessions/{session_id}/scene/snapshot" in trace
    summary = compact_summary(snapshot, artifacts)
    assert summary["prim_count"] == 4
    assert summary["candidate_count"] == 4
    assert summary["coverage_candidate_count"] == 2
    assert summary["agent_context"].endswith("material_authoring_context.json")
    assert summary["assignment_seed"].endswith("material_assignment_seed.json")

    respected_run_dir = tmp_path / "respected-run"
    write_snapshot_artifacts(
        snapshot,
        respected_run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        append_trace=False,
        respect_existing_material_bindings=True,
    )
    respected_context = json.loads(
        (respected_run_dir / "raw" / "material_authoring_context.json").read_text()
    )
    assert (
        respected_context["material_binding_policy"][
            "respect_existing_material_bindings"
        ]
        is True
    )
    assert {
        group["grouping_basis"] for group in respected_context["candidate_groups"]
    } == {"existing_material"}
    assert {
        group["recommended_coverage_status"]
        for group in respected_context["candidate_groups"]
    } == {"preserved_existing"}


def test_optimized_snapshot_suppresses_source_clones_covered_by_dedup(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    materials_usd.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        json.dumps({"library_path": "materials.usd", "entries": []}),
        encoding="utf-8",
    )
    snapshot = {
        "session_id": "session-optimized",
        "root_prim_path": "/World",
        "source_scene_path": "/tmp/source.usdc",
        "inspection_scene_path": "/tmp/optimized.usdc",
        "optimization": {"enabled": True, "status": "ready"},
        "paths": [
            "/World",
            "/World/BoltPrototype/Geometry",
            "/World/BoltB/Geometry",
            "/World/Panel/Geometry",
        ],
        "nodes": [
            {
                "path": "/World",
                "name": "World",
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "children": True,
                "child_paths": [
                    "/World/BoltPrototype/Geometry",
                    "/World/BoltB/Geometry",
                    "/World/Panel/Geometry",
                ],
            },
            {
                "path": "/World/BoltPrototype/Geometry",
                "name": "Geometry",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
            {
                "path": "/World/BoltB/Geometry",
                "name": "Geometry",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
            {
                "path": "/World/Panel/Geometry",
                "name": "Geometry",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
        ],
        "properties": [],
        "material_bindings": [],
        "path_translations": [],
        "candidates": [
            {
                "inspection_path": "/World/BoltPrototype/Geometry",
                "source_paths": ["/World/BoltA", "/World/BoltB"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": True,
                "direct_targets": [],
                "bound_material_path": None,
            },
            {
                "inspection_path": "/World/BoltB/Geometry",
                "source_paths": ["/World/BoltB/Geometry"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
            },
            {
                "inspection_path": "/World/Panel/Geometry",
                "source_paths": ["/World/Panel/Geometry"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
            },
        ],
    }

    write_snapshot_artifacts(
        snapshot,
        run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        append_trace=False,
        respect_existing_material_bindings=False,
        candidate_policy=MaterialCandidatePolicy(material_candidate_space="inspection"),
    )

    visible = json.loads((run_dir / "raw" / "visible_candidate_prims.json").read_text())
    assert visible["path_space"] == "inspection"
    assert visible["candidate_visible_prim_count"] == 2
    assert [candidate["runtime_path"] for candidate in visible["candidates"]] == [
        "/World/BoltPrototype/Geometry",
        "/World/Panel/Geometry",
    ]
    dedup_candidate = visible["candidates"][0]
    assert dedup_candidate["deduplicated"] is True
    assert dedup_candidate["source_paths"] == ["/World/BoltA", "/World/BoltB"]
    assert dedup_candidate["source_instance_count"] == 2

    assignment_seed = json.loads(
        (run_dir / "raw" / "material_assignment_seed.json").read_text()
    )
    assert assignment_seed["path_space"] == "inspection"
    assert sorted(
        path
        for assignment in assignment_seed["assignments"]
        for path in assignment["prim_paths"]
    ) == ["/World/BoltPrototype/Geometry", "/World/Panel/Geometry"]


def test_snapshot_collapses_instance_proxy_candidates_to_source_targets(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    usd_path = tmp_path / "instanced.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Prototype")
    UsdGeom.Mesh.Define(stage, "/World/Prototype/Mesh")
    for instance_path in ("/World/InstanceA", "/World/InstanceB"):
        instance = UsdGeom.Xform.Define(stage, instance_path).GetPrim()
        instance.GetReferences().AddInternalReference("/World/Prototype")
        instance.SetInstanceable(True)
    stage.GetRootLayer().Save()

    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    materials_usd.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        json.dumps({"library_path": "materials.usd", "entries": []}),
        encoding="utf-8",
    )
    snapshot = {
        "session_id": "session-instance",
        "root_prim_path": "/World",
        "source_scene_path": str(usd_path),
        "inspection_scene_path": str(usd_path),
        "optimization": {"enabled": False, "status": "disabled"},
        "paths": [
            "/World",
            "/World/InstanceA",
            "/World/InstanceA/Mesh",
            "/World/InstanceB",
            "/World/InstanceB/Mesh",
        ],
        "nodes": [],
        "properties": [],
        "material_bindings": [],
        "path_translations": [],
        "candidates": [
            {
                "inspection_path": "/World/InstanceA/Mesh",
                "source_paths": ["/World/InstanceA/Mesh"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [1.0, 1.0, 1.0],
            },
            {
                "inspection_path": "/World/InstanceB/Mesh",
                "source_paths": ["/World/InstanceB/Mesh"],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
                "bounds_center": [0.0, 0.0, 0.0],
                "bounds_size": [1.0, 1.0, 1.0],
            },
        ],
        "excluded_non_candidates": [],
        "summary": {"prim_count": 5, "candidate_count": 2, "truncated": False},
    }

    collapsed_run_dir = tmp_path / "collapsed"
    write_snapshot_artifacts(
        snapshot,
        collapsed_run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        append_trace=False,
    )

    collapsed = json.loads(
        (collapsed_run_dir / "raw" / "visible_candidate_prims.json").read_text()
    )
    assert collapsed["path_space"] == "source"
    assert collapsed["material_candidate_policy"]["skip_instances"] is True
    assert collapsed["candidate_visible_prim_count"] == 1
    assert collapsed["candidates"][0]["source_path"] == "/World/Prototype/Mesh"
    assert collapsed["candidates"][0]["original_source_paths"] == [
        "/World/InstanceA/Mesh",
        "/World/InstanceB/Mesh",
    ]
    assert collapsed["candidates"][0]["runtime_paths"] == [
        "/World/InstanceA/Mesh",
        "/World/InstanceB/Mesh",
    ]
    assert collapsed["candidates"][0]["source_instance_count"] == 2
    assert collapsed["candidates"][0]["instance_collapsed"] is True
    collapsed_context = json.loads(
        (collapsed_run_dir / "raw" / "material_authoring_context.json").read_text()
    )
    assert collapsed_context["candidate_groups"][0]["runtime_evidence_count"] == 2
    assert collapsed_context["candidate_groups"][0]["runtime_paths"] == [
        "/World/InstanceA/Mesh",
        "/World/InstanceB/Mesh",
    ]

    expanded_run_dir = tmp_path / "expanded"
    write_snapshot_artifacts(
        snapshot,
        expanded_run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        append_trace=False,
        candidate_policy=MaterialCandidatePolicy(skip_instances=False),
    )
    expanded = json.loads(
        (expanded_run_dir / "raw" / "visible_candidate_prims.json").read_text()
    )
    assert expanded["material_candidate_policy"]["skip_instances"] is False
    assert [candidate["source_path"] for candidate in expanded["candidates"]] == [
        "/World/InstanceA/Mesh",
        "/World/InstanceB/Mesh",
    ]
    assert all(
        not candidate["instance_collapsed"] for candidate in expanded["candidates"]
    )


def test_snapshot_keeps_unresolved_external_instance_candidates() -> None:
    source_path = "/World/ExternalInstance/Mesh"

    remapped, collapsed, skip = _remap_instance_source_target(
        source_path,
        {"/World/ExternalInstance": None},
    )

    assert remapped == source_path
    assert collapsed is False
    assert skip is False


def test_snapshot_ignores_display_color_as_preserved_material(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    materials_usd.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        json.dumps({"library_path": "materials.usd", "entries": []}),
        encoding="utf-8",
    )
    green_path = "/World/body__1/shape/mesh"
    orange_path = "/World/body__2/shape/mesh"
    snapshot = {
        "session_id": "session-display-color",
        "root_prim_path": "/World",
        "source_scene_path": "/tmp/source.usdc",
        "inspection_scene_path": "/tmp/source.usdc",
        "paths": ["/World", green_path, orange_path],
        "nodes": [
            {
                "path": "/World",
                "name": "World",
                "type_name": "Xform",
                "active": True,
                "loaded": True,
                "children": True,
                "child_paths": [green_path, orange_path],
            },
            {
                "path": green_path,
                "name": "mesh",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
            {
                "path": orange_path,
                "name": "mesh",
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "children": False,
                "child_paths": [],
            },
        ],
        "properties": [
            {
                "prim_path": green_path,
                "properties": {
                    "attributes": {
                        "primvars:displayColor": {
                            "type_name": "color3f[]",
                            "value": ["(0.2, 0.6, 0.2)"],
                        }
                    }
                },
            },
            {
                "prim_path": orange_path,
                "properties": {
                    "attributes": {
                        "primvars:displayColor": {
                            "type_name": "color3f[]",
                            "value": ["(0.8235, 0.4196, 0.2157)"],
                        }
                    }
                },
            },
        ],
        "material_bindings": [],
        "path_translations": [],
        "candidates": [
            {
                "inspection_path": green_path,
                "source_paths": [green_path],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
                "bounds_size": [1.0, 1.0, 1.0],
            },
            {
                "inspection_path": orange_path,
                "source_paths": [orange_path],
                "type_name": "Mesh",
                "active": True,
                "loaded": True,
                "effective_visible": True,
                "candidate_reason": "renderable_prim",
                "ambiguous_translation": False,
                "direct_targets": [],
                "bound_material_path": None,
                "bounds_size": [1.0, 1.0, 1.0],
            },
        ],
    }

    write_snapshot_artifacts(
        snapshot,
        run_dir,
        materials_yaml=materials_yaml,
        materials_usd=materials_usd,
        append_trace=False,
        respect_existing_material_bindings=False,
    )

    context = json.loads(
        (run_dir / "raw" / "material_authoring_context.json").read_text()
    )
    assert (
        context["material_binding_policy"]["respect_existing_material_bindings"]
        is False
    )
    assert context["summary"]["candidate_group_count"] == 2
    assert {
        next(iter(group["display_color_counts"]))
        for group in context["candidate_groups"]
    } == {"green_display_color", "orange_brown_display_color"}
    assert {
        group["recommended_coverage_status"] for group in context["candidate_groups"]
    } == {"ambiguous_unassigned"}

    assignment_seed = json.loads(
        (run_dir / "raw" / "material_assignment_seed.json").read_text()
    )
    assert {
        assignment["coverage_status"] for assignment in assignment_seed["assignments"]
    } == {"ambiguous_unassigned"}


def test_materials_assign_dry_run_supports_claude_runner(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usdc"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    claude_config = tmp_path / "claude-config.json"
    for path in [usd, reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    claude_config.write_text(
        json.dumps(
            {
                "maxBudgetUsd": 2.5,
                "settings": {"permissions": {"allow": ["Bash(curl*)"]}},
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "claude-run"
    exit_code = main(
        [
            "materials",
            "assign",
            "--usd",
            str(usd),
            "--reference-image",
            str(reference),
            "--materials-yaml",
            str(materials_yaml),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(run_dir),
            "--runner",
            "claude",
            "--model",
            "claude-sonnet-4-6",
            "--model-reasoning-effort",
            "high",
            "--claude-permission-mode",
            "default",
            "--claude-max-turns",
            "80",
            "--claude-config-file",
            str(claude_config),
            "--claude-config-json",
            '{"settings":{"permissions":{"deny":["Bash(rm*)"]}}}',
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["runner"] == "claude"
    assert request["model"] == "claude-sonnet-4-6"
    assert request["model_reasoning_effort"] == "high"
    assert request["claude_permission_mode"] == "default"
    assert request["claude_max_turns"] == 80
    assert request["claude_config"] == {
        "maxBudgetUsd": 2.5,
        "settings": {
            "permissions": {
                "allow": ["Bash(curl*)"],
                "deny": ["Bash(rm*)"],
            }
        },
    }


def test_load_claude_config_rejects_unsupported_top_level_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "claude-config.json"
    config_path.write_text(
        json.dumps({"mcpServers": {}, "settings": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported keys: mcpServers"):
        _load_claude_config(
            SimpleNamespace(
                claude_config_file=[config_path],
                claude_config_json=[],
            )
        )


def test_load_claude_config_rejects_non_object_settings() -> None:
    with pytest.raises(ValueError, match="settings must be a JSON object"):
        _load_claude_config(
            SimpleNamespace(
                claude_config_file=[],
                claude_config_json=['{"settings":"unsafe"}'],
            )
        )


def test_load_claude_config_accepts_supported_top_level_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "claude-config.json"
    config_path.write_text(
        json.dumps({"env": {"CLAUDE_CODE_USE_BEDROCK": "1"}}),
        encoding="utf-8",
    )

    assert _load_claude_config(
        SimpleNamespace(
            claude_config_file=[config_path],
            claude_config_json=['{"maxBudgetUsd":2.5,"settings":{"model":"x"}}'],
        )
    ) == {
        "env": {"CLAUDE_CODE_USE_BEDROCK": "1"},
        "maxBudgetUsd": 2.5,
        "settings": {"model": "x"},
    }


def test_default_codex_sandbox_mode_is_workspace_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTENT_AGENT_CODEX_SANDBOX_MODE", raising=False)

    assert _default_codex_sandbox_mode() == "workspace-write"


def test_default_codex_sandbox_mode_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_AGENT_CODEX_SANDBOX_MODE", "danger-full-access")

    with pytest.raises(ValueError, match="Invalid CONTENT_AGENT_CODEX_SANDBOX_MODE"):
        _default_codex_sandbox_mode()


def test_materials_assign_reports_missing_additional_instructions_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    usd = tmp_path / "asset.usdc"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    for path in [usd, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "materials",
            "assign",
            "--usd",
            str(usd),
            "--materials-yaml",
            str(materials_yaml),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--additional-instructions-file",
            str(tmp_path / "missing.md"),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--additional-instructions-file does not exist" in capsys.readouterr().err


def test_materials_assign_rejects_non_positive_vqa_refinement_iterations() -> None:
    with pytest.raises(SystemExit) as error:
        main(["materials", "assign", "--vqa-refinement-max-iterations", "0"])

    assert error.value.code == 2


def test_should_start_workbench_defaults_for_loopback_hosts() -> None:
    assert _should_start_workbench(
        SimpleNamespace(start_workbench=None, workbench_url="http://127.0.0.1:8088")
    )
    assert _should_start_workbench(
        SimpleNamespace(start_workbench=None, workbench_url="http://127.0.0.2:8088")
    )
    assert _should_start_workbench(
        SimpleNamespace(start_workbench=None, workbench_url="http://localhost:8088")
    )
    assert _should_start_workbench(
        SimpleNamespace(start_workbench=None, workbench_url="http://[::1]:8088")
    )
    assert not _should_start_workbench(
        SimpleNamespace(
            start_workbench=None, workbench_url="http://workbench-host:8088"
        )
    )
    assert not _should_start_workbench(
        SimpleNamespace(start_workbench=False, workbench_url="http://127.0.0.1:8088")
    )
    assert _should_start_workbench(
        SimpleNamespace(
            start_workbench=True, workbench_url="http://workbench-host:8088"
        )
    )


def test_resolve_materials_usd_from_yaml_manifest(tmp_path: Path) -> None:
    materials_yaml = tmp_path / "materials.yaml"
    materials_yaml.write_text(
        'library_path: "nested/materials.usd"\nentries: []\n',
        encoding="utf-8",
    )

    assert (
        _resolve_materials_usd_from_manifest(materials_yaml)
        == (tmp_path / "nested" / "materials.usd").resolve()
    )


def test_resolve_materials_usd_requires_manifest_library_path(
    tmp_path: Path,
) -> None:
    materials_yaml = tmp_path / "materials.yaml"
    materials_yaml.write_text("entries: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="library_path"):
        _resolve_materials_usd_from_manifest(materials_yaml)


def test_claude_config_validation_error_names_claude() -> None:
    with pytest.raises(ValueError, match="Claude config does not accept"):
        _parse_json_object(
            "null",
            "--claude-config-json",
            config_name="Claude config",
        )
