# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI tests for content-workflow-cli."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from content_agent_workflows.common import usd_cli as usd_cli_common
from world_understanding.utils.windows_process import windows_process_is_live

import content_workflow_cli.cli as cli_module
from content_workflow_cli.cli import (
    _codex_base_url_from_args,
    _default_codex_sandbox_mode,
    _handle_auth_status,
    _linux_codex_sandbox_prerequisite_error,
    _load_claude_config,
    _parse_json_object,
    _probe_codex_workspace_write_access,
    _read_exact_smoke_marker,
    _resolve_materials_usd_from_manifest,
    _sanitize_native_diagnostic,
    _smoke_write_command,
    build_parser,
    main,
)


def test_parser_rejects_invalid_vision_max_tokens_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_AGENT_VISION_MAX_TOKENS", "not-an-integer")

    with pytest.raises(
        SystemExit,
        match="CONTENT_AGENT_VISION_MAX_TOKENS must be an integer",
    ):
        build_parser()


def test_main_converts_argparse_system_exit_to_an_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--not-a-real-option"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_articulation_platform_preflight_uses_canonical_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "world_understanding.functions.physics.joint_rigger."
        "require_joint_rigger_authoring_platform",
        lambda: calls.append(True),
    )

    assert main(["preflight", "articulation-authoring-platform"]) == 0
    assert calls == [True]
    assert capsys.readouterr().out == ("articulation authoring platform preflight ok\n")


def test_articulation_platform_preflight_reports_intentional_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from world_understanding.functions.physics.joint_rigger import (
        JointRiggerBackendUnavailableError,
    )

    message = (
        "Descriptor-sealed Joint Rigger authoring requires Linux, a Linux "
        "container, or WSL2."
    )

    def reject() -> None:
        raise JointRiggerBackendUnavailableError(message)

    monkeypatch.setattr(
        "world_understanding.functions.physics.joint_rigger."
        "require_joint_rigger_authoring_platform",
        reject,
    )

    assert main(["preflight", "articulation-authoring-platform"]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_claude_sandbox_preflight_routes_to_confinement_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        "content_workflow_cli.runner.preflight_claude_windows_sandbox",
        lambda path: calls.append(path),
    )

    assert main(["preflight", "claude-sandbox"]) == 0
    assert calls == [Path.cwd()]
    assert "Claude sandbox preflight" in capsys.readouterr().out


def test_auth_status_probes_workspace_write_usability(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        run_calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("content_workflow_cli.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "content_workflow_cli.cli._linux_codex_sandbox_prerequisite_error",
        lambda: None,
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._probe_codex_workspace_write_access", lambda _exe: 0
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._codex_executable", lambda: "codex-test"
    )

    assert _handle_auth_status(SimpleNamespace()) == 0
    assert run_calls == [["codex-test", "login", "status"]]
    assert capsys.readouterr().out == ""


def test_auth_status_accepts_explicit_sandbox_smoke_flag() -> None:
    args = build_parser().parse_args(["auth", "status", "--sandbox-smoke"])

    assert args.sandbox_smoke is True
    assert args.handler is _handle_auth_status


def test_auth_status_fails_before_provider_when_linux_sandbox_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "content_workflow_cli.cli._codex_executable", lambda: "codex-test"
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._run_codex_command", lambda _command: 0
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._linux_codex_sandbox_prerequisite_error",
        lambda: "bubblewrap cannot create the Codex workspace-write sandbox",
    )

    assert _handle_auth_status(SimpleNamespace()) == 1
    assert "bubblewrap cannot create" in capsys.readouterr().err


def test_linux_sandbox_probe_uses_resolved_true_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr("content_workflow_cli.cli.sys.platform", "linux")
    monkeypatch.setattr(
        "content_workflow_cli.cli.shutil.which",
        lambda command: {
            "bwrap": "/usr/bin/bwrap",
            "true": "/nix/store/true/bin/true",
        }.get(command),
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli.subprocess.run",
        lambda command, **_kwargs: (
            commands.append(command) or subprocess.CompletedProcess(command, 0)
        ),
    )

    assert _linux_codex_sandbox_prerequisite_error() is None
    assert commands[0][-1] == "/nix/store/true/bin/true"


def test_workspace_write_probe_checks_direct_and_sdk_bridge(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "content_workflow_cli.cli.preflight_codex_windows_support",
        lambda: None,
    )

    def fake_probe(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "codex-test":
            cwd = Path(command[command.index("--cd") + 1])
            (cwd / ".content-workflow-codex-direct-smoke").write_text("direct-ok")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("content_workflow_cli.cli._run_codex_model_probe", fake_probe)

    assert _probe_codex_workspace_write_access("codex-test") == 0
    direct, bridge = calls
    assert direct[0:2] == ["codex-test", "exec"]
    assert ["--sandbox", "workspace-write"] == direct[
        direct.index("--sandbox") : direct.index("--sandbox") + 2
    ]
    assert bridge[-2] == "--sandbox-smoke"
    assert "SDK bridge workspace-write" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Job Objects")
def test_codex_model_probe_timeout_reaps_windows_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os,time; "
        f"open({str(child_pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(60)"
    )
    monkeypatch.setattr(cli_module, "_CODEX_MODEL_PROBE_TIMEOUT_SECONDS", 0.5)

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        cli_module._run_codex_model_probe([sys.executable, "-c", parent_code])

    assert time.monotonic() - started < 10
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert not windows_process_is_live(child_pid)


def test_windows_workspace_write_probe_rejects_before_provider_execution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[list[str]] = []

    def reject_windows() -> None:
        raise RuntimeError(
            "The Codex runner is not supported on native Windows in release 0.6. "
            "Run the Content Agent workflow inside WSL2 or on native Linux."
        )

    monkeypatch.setattr(
        "content_workflow_cli.cli.preflight_codex_windows_support", reject_windows
    )
    monkeypatch.setattr(
        "content_workflow_cli.cli._run_codex_model_probe",
        lambda command: calls.append(command),
    )

    assert _probe_codex_workspace_write_access("codex-test") == 1
    assert calls == []
    error = capsys.readouterr().err
    assert "not supported on native Windows" in error
    assert "WSL2" in error


def test_workspace_write_probe_uses_a_native_windows_powershell_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = Path(r"C:\Temp\sandbox marker")
    monkeypatch.setattr("content_workflow_cli.cli.sys.platform", "win32")

    command = _smoke_write_command(marker, "direct-ok")

    assert command == (
        "Set-Content -NoNewline -LiteralPath 'C:\\Temp\\sandbox marker' "
        "-Value 'direct-ok'"
    )


def test_exact_smoke_marker_rejects_symlinks_and_oversized_content(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker"
    marker.write_text("direct-ok-extra", encoding="utf-8")
    assert not _read_exact_smoke_marker(marker, "direct-ok")

    target = tmp_path / "target"
    target.write_text("direct-ok", encoding="utf-8")
    marker.unlink()
    try:
        marker.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    assert not _read_exact_smoke_marker(marker, "direct-ok")


def test_setup_script_quotes_child_runner_advice_heredoc() -> None:
    setup_script = Path(__file__).parents[4] / "scripts" / "setup_content_agent.sh"
    setup_text = setup_script.read_text(encoding="utf-8")

    assert (
        "if [[ \"$INSTALL_CHILD_RUNNERS\" -eq 1 ]]; then\n    cat <<'EOF'" in setup_text
    )
    assert 'bwrap_probe_executable="$(type -P true || true)"' in setup_text
    assert "require_command jq" in setup_text
    assert "content-workflow-cli auth status --sandbox-smoke" in setup_text


def test_bridge_smoke_cleanup_removes_all_artifacts(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Codex SDK bridge test")
    artifacts = [
        tmp_path / ".content-workflow-codex-sandbox-smoke",
        tmp_path / "bridge-final.txt",
        tmp_path / "bridge-items.json",
    ]
    for artifact in artifacts:
        artifact.write_text("smoke output", encoding="utf-8")
    bridge = Path(__file__).parents[1] / "content_workflow_cli" / "codex_sdk_bridge.mjs"
    script = (
        "import { cleanupSandboxSmokeArtifacts } from "
        f"{json.dumps(bridge.resolve().as_uri())}; "
        f"cleanupSandboxSmokeArtifacts({json.dumps([str(path) for path in artifacts])});"
    )

    subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert all(not artifact.exists() for artifact in artifacts)


def test_bridge_smoke_pins_windows_powershell_policy() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Codex SDK bridge test")
    bridge = Path(__file__).parents[1] / "content_workflow_cli" / "codex_sdk_bridge.mjs"
    script = (
        "import { sandboxSmokeHostExecutables } from "
        f"{json.dumps(bridge.resolve().as_uri())}; "
        "process.stdout.write(JSON.stringify(sandboxSmokeHostExecutables("
        "'win32', { SystemRoot: 'C:\\\\Windows' }, (value) => value, () => true)));"
    )

    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == [
        {
            "name": "powershell",
            "paths": [r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"],
        }
    ]


@pytest.mark.parametrize(
    "diagnostic",
    [
        "Authorization: Bearer test-secret",
        "authorization=Basic test-secret",
        "Bearer test-secret",
        "TOKEN: test-secret",
        "api_key=test-secret",
        "API key: test-secret",
        "API key=test-secret",
        "credential=test-secret",
        "access_key=test-secret",
        "Incorrect API key provided: test-secret",
        "token is test-secret",
        "API key provided was test-secret",
        "token provided is test-secret",
        "OPENAI_API_KEY='test secret'",
        "AWS_SECRET_ACCESS_KEY=test-secret",
        '{"access_token":"test-secret"}',
        '{"OPENAI_API_KEY":"test secret"}',
        'PASSWORD="ab\\"test-secret"',
        "PASSWORD='ab\\'test-secret'",
        "PASSWORD=`ab\\`test-secret`",
        'PASSWORD="test secret"',
    ],
)
def test_native_diagnostic_redacts_credentials(diagnostic: str) -> None:
    sanitized = _sanitize_native_diagnostic(diagnostic)

    assert "test-secret" not in sanitized
    assert "[redacted]" in sanitized


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
            str(_repo_root_with_usd_cli_source(tmp_path)),
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


def test_physics_apply_long_running_dry_run_writes_vomp_contract(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    vomp_root = tmp_path / "VoMP"
    vomp_root.mkdir()
    run_dir = tmp_path / "physics-vomp-run"

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--repo-root",
            str(_repo_root_with_usd_cli_source(tmp_path)),
            "--output-dir",
            str(run_dir),
            "--simulation-engine",
            "fake",
            "--vomp-root",
            str(vomp_root),
            "--vomp-target-prim",
            "/World/Body",
            "--vomp-num-views",
            "12",
            "--vomp-seed",
            "7",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (run_dir / "raw" / "physics_agentic_contract.json").read_text(encoding="utf-8")
    )
    prompt = (run_dir / "agent_prompt.md").read_text(encoding="utf-8")
    assert request["vomp_runtime"]["runtime_root"] == str(vomp_root.resolve())
    assert request["vomp_runtime"]["num_views"] == 12
    assert request["vomp_runtime"]["seed"] == 7
    assert contract["mass_properties"]["enabled"] is True
    assert contract["mass_properties"]["provider"] == "vomp"
    assert contract["mass_properties"]["target_prim_path"] == "/World/Body"
    assert contract["tuning"]["protected_parameters"] == ["mass_scale"]
    assert contract["tuning"]["allow_revise_patch"] is False
    assert "VoMP is the authoritative" in prompt
    assert "raw/physics_vomp_result.json" in prompt


@pytest.mark.parametrize("seed", ["-1", str(2**32)])
def test_physics_apply_rejects_vomp_seed_outside_uint32(
    seed: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--repo-root",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "run"),
            "--vomp-root",
            str(tmp_path / "VoMP"),
            "--vomp-seed",
            seed,
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "must be between 0 and 4294967295" in capsys.readouterr().err


def test_physics_apply_rejects_vomp_options_without_runtime_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
            "--vomp-target-prim",
            "/World/Body",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--vomp-target-prim require --vomp-root" in capsys.readouterr().err


def test_physics_deterministic_cli_preserves_lexical_output_dir(
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
            "--output-dir",
            str(lexical_run_dir),
            "--deterministic-workflow",
            "--no-simulation",
        ]
    )

    assert exit_code == 2
    assert "must resolve without traversing symlinks" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


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


def test_model_reasoning_effort_accepts_claude_max() -> None:
    parsed = build_parser().parse_args(
        [
            "materials",
            "assign",
            "--usd",
            "asset.usd",
            "--materials-yaml",
            "materials.yaml",
            "--runner",
            "claude",
            "--model-reasoning-effort",
            "max",
        ]
    )

    assert parsed.model_reasoning_effort == "max"


def test_physics_no_simulation_help_is_explicitly_non_passing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["physics", "apply", "--help"]) == 0

    help_text = capsys.readouterr().out
    no_simulation_block = re.search(
        r"(?ms)^\s+--no-simulation\s+(.*?)(?=^\s+--[a-z]|\Z)",
        help_text,
    )
    assert no_simulation_block is not None

    no_simulation_help = no_simulation_block.group(1)
    assert re.search(r"conditional\s+and\s+non-\s*passing", no_simulation_help)
    assert re.search(
        r"does not satisfy required runtime\s+and\s+visual evidence",
        no_simulation_help,
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


def test_physics_deterministic_workflow_rejects_agentic_refine_flags(
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
            "--deterministic-workflow",
            "--refine",
            "--behavior-prompt",
            "make it bouncy",
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
        timeout_s: float = 120.0,
    ) -> tuple[ConversionReport, ConversionProbeArtifact]:
        called["install_missing"] = install_missing
        called["output_format"] = output_format == "usdc"
        called["default_timeout"] = timeout_s == 120.0
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
    assert called["default_timeout"] is True
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
            "--converter-timeout",
            "300",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["converter_timeout_s"] == 300.0
    assert payload["output_usd_path"] == str((cwd / "asset.usda").resolve())
    assert (cwd / "asset.usda").exists()
    assert (run_dir / "request.json").exists()
    assert (run_dir / "conversion_report.json").exists()
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "workflow_run_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (run_dir / "conversion_report.json").read_text(encoding="utf-8")
    )
    assert request["converter_timeout_s"] == 300.0
    assert manifest["policy"]["converter_timeout_s"] == 300.0
    assert report["converter_timeout_s"] == 300.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_convert_to_usd_rejects_non_positive_or_non_finite_timeout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    exit_code = main(["convert-to-usd", str(source), "--converter-timeout", value])

    assert exit_code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_convert_to_usd_help_documents_converter_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["convert-to-usd", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "--converter-timeout SECONDS" in help_text
    assert "Defaults to 120" in help_text


def test_convert_to_usd_direct_output_reports_converter_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["convert-to-usd", str(source), "--converter-timeout", "300"])

    assert exit_code == 0
    assert "Converter timeout: 300s" in capsys.readouterr().out


def test_materials_assign_dry_run_writes_contract(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usda"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    reference.write_text("placeholder", encoding="utf-8")
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
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
            "--materials-yaml",
            str(materials_yaml),
            "--repo-root",
            str(_repo_root_with_usd_cli_source(tmp_path)),
            "--output-dir",
            str(run_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["workflow"] == "materials.assign"
    assert request["dry_run"] is True
    for relative_path in (
        "agent_prompt.md",
        "trace/events.jsonl",
        "trace/operation_trace.json",
        "trace/operation_trace.md",
        "trace/run_retrospective.json",
        "trace/replay_manifest.json",
    ):
        assert (run_dir / relative_path).is_file()


def test_materials_assign_accepts_exact_custom_responses_endpoint(
    tmp_path: Path,
) -> None:
    usd = tmp_path / "asset.usda"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    reference.write_text("placeholder", encoding="utf-8")
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    materials_yaml.write_text(
        'library_path: "materials.usd"\nentries: []\n',
        encoding="utf-8",
    )
    endpoint = "https://integrate.api.nvidia.com/v1/responses"
    run_dir = tmp_path / "custom-endpoint-run"

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
            str(_repo_root_with_usd_cli_source(tmp_path)),
            "--output-dir",
            str(run_dir),
            "--codex-responses-url",
            endpoint,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    assert request["codex_responses_url"] == endpoint
    assert request["codex_base_url"] == "https://integrate.api.nvidia.com/v1"


def test_materials_assign_help_documents_custom_responses_endpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["materials", "assign", "--help"]) == 0

    help_text = capsys.readouterr().out
    assert "--codex-responses-url CODEX_RESPONSES_URL" in help_text
    assert "NVIDIA Inference Hub" in help_text


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://provider.example/v1/responses",
        "https://user:password@provider.example/v1/responses",
        "https://provider.example/v1/chat/completions",
        "https://provider.example/v1/responses?tenant=secret",
    ],
)
def test_materials_assign_rejects_invalid_responses_endpoint_before_launch(
    endpoint: str,
) -> None:
    args = build_parser().parse_args(
        [
            "materials",
            "assign",
            "--usd",
            "asset.usda",
            "--reference-image",
            "reference.png",
            "--materials-yaml",
            "materials.yaml",
            "--codex-responses-url",
            endpoint,
        ]
    )

    with pytest.raises(ValueError, match="--codex-responses-url"):
        _codex_base_url_from_args(args)


def test_materials_assign_rejects_both_codex_endpoint_forms() -> None:
    args = build_parser().parse_args(
        [
            "materials",
            "assign",
            "--usd",
            "asset.usda",
            "--reference-image",
            "reference.png",
            "--materials-yaml",
            "materials.yaml",
            "--codex-base-url",
            "https://provider.example/v1",
            "--codex-responses-url",
            "https://provider.example/v1/responses",
        ]
    )

    with pytest.raises(ValueError, match="either --codex-base-url"):
        _codex_base_url_from_args(args)


def test_materials_assign_dry_run_supports_claude_runner(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usda"
    reference = tmp_path / "reference.png"
    materials_yaml = tmp_path / "materials.yaml"
    materials_usd = tmp_path / "materials.usd"
    claude_config = tmp_path / "claude-config.json"
    for path in [reference, materials_usd]:
        path.write_text("placeholder", encoding="utf-8")
    usd.write_text("#usda 1.0\n", encoding="utf-8")
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
            str(_repo_root_with_usd_cli_source(tmp_path)),
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
            str(_repo_root_with_usd_cli_source(tmp_path)),
            "--output-dir",
            str(tmp_path / "run"),
            "--additional-instructions-file",
            str(tmp_path / "missing.md"),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "--additional-instructions-file does not exist" in capsys.readouterr().err


def test_materials_assign_rejects_non_positive_vqa_refinement_iterations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["materials", "assign", "--vqa-refinement-max-iterations", "0"])

    assert exit_code == 2
    assert "must be at least 1" in capsys.readouterr().err


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


def test_resolve_materials_usd_reports_manifest_read_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Failed to read material YAML manifest"):
        _resolve_materials_usd_from_manifest(tmp_path)


def test_claude_config_validation_error_names_claude() -> None:
    with pytest.raises(ValueError, match="Claude config does not accept"):
        _parse_json_object(
            "null",
            "--claude-config-json",
            config_name="Claude config",
        )


def test_physics_apply_rejects_resume_without_deterministic_workflow(
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
            "--resume",
            "--dry-run",
        ]
    )

    assert exit_code == 2


def test_convert_to_usd_resume_requires_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    exit_code = main(["convert-to-usd", str(source), "--resume"])

    assert exit_code == 2
    assert "--resume requires --output-dir" in capsys.readouterr().err


def test_simready_resume_flags_parse() -> None:
    parser = build_parser()

    validation = parser.parse_args(
        ["simready", "validate-profile", "asset.usda", "--resume"]
    )
    conformance = parser.parse_args(
        [
            "simready",
            "conform-profile",
            "asset.usda",
            "--output-dir",
            "run",
            "--resume",
        ]
    )

    assert validation.resume
    assert conformance.resume


def _repo_root_with_usd_cli_source(tmp_path: Path) -> Path:
    if not usd_cli_common.USD_CLI_REQUIRED_SOURCE:
        pytest.skip("usd-cli backend source is not distributed in public staging")

    repo_root = tmp_path / "repo"
    source_root = repo_root / usd_cli_common.USD_CLI_SOURCE_PATH
    if (repo_root / ".git").exists():
        return repo_root

    source_root.mkdir(parents=True)
    for relative in usd_cli_common.USD_CLI_REQUIRED_SOURCE:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Content Workflow Tests",
            "-c",
            "user.email=content-workflow-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=repo_root,
        check=True,
    )
    return repo_root
