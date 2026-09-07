# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the public composed CAD-to-SimReady launcher."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from content_agent_workflows.physics import preflight as physics_preflight

from content_workflow_cli import cad_to_simready_runner
from content_workflow_cli.cli import main


def _selected_ovphysx_lock(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from world_understanding.functions.physics import ovphysx_daemon

    lock = repo / ovphysx_daemon._ovphysx_runtime_lock()
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(lock))
    return lock


def test_invoke_cli_records_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_argv: list[str], _log: io.StringIO) -> int:
        raise RuntimeError("unexpected launcher fault")

    monkeypatch.setattr(cad_to_simready_runner, "_run_cli", fail)
    log = io.StringIO()

    invocation = cad_to_simready_runner._invoke_cli(
        {"stage": "convert", "step": "convert-source", "argv": ["convert"]},
        log=log,
    )

    assert invocation.exit_code == 1
    assert "unexpected launcher fault" in log.getvalue()


def test_invoke_cli_records_argparse_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_argv: list[str], _log: io.StringIO) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(cad_to_simready_runner, "_run_cli", fail)
    log = io.StringIO()

    invocation = cad_to_simready_runner._invoke_cli(
        {"stage": "convert", "step": "convert-source", "argv": ["convert"]},
        log=log,
    )

    assert invocation.exit_code == 2
    assert "terminated while parsing" in log.getvalue()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            [
                "cad-to-simready",
                "run",
                "missing.stl",
                "--output-dir",
                "run",
                "--materials-yaml",
                "missing.yaml",
            ],
            "cad-to-simready failed:",
        ),
        (
            [
                "cad-to-simready",
                "flatten-for-physics",
                "missing.usd",
                "physics.usd",
                "--report",
                "flatten.json",
            ],
            "cad-to-simready flatten-for-physics failed:",
        ),
        (
            [
                "cad-to-simready",
                "render-final",
                "missing.usd",
                "--validation-report",
                "missing.json",
                "--output-dir",
                "renders",
                "--report",
                "render.json",
            ],
            "cad-to-simready render-final failed:",
        ),
    ],
)
def test_public_handlers_convert_validation_errors_to_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cad_to_simready_runner, "find_repo_root", lambda: tmp_path)

    exit_code = main(arguments)

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr


def _dry_run(
    tmp_path: Path, *extra: str
) -> tuple[dict[str, object], dict[str, object]]:
    source = tmp_path / "widget.stl"
    source.write_text("solid widget\nendsolid widget\n", encoding="utf-8")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    output = tmp_path / "run"

    exit_code = main(
        [
            "cad-to-simready",
            "run",
            str(source),
            "--output-dir",
            str(output),
            "--materials-yaml",
            str(materials),
            "--dry-run",
            *extra,
        ]
    )

    assert exit_code == 0
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
    return request, plan


@pytest.mark.parametrize("platform_name", ["darwin", "freebsd"])
def test_unsupported_native_host_fails_before_artifact_or_child_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    source = tmp_path / "widget.obj"
    source.write_text("o widget\n", encoding="utf-8")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(cad_to_simready_runner.sys, "platform", platform_name)
    monkeypatch.setattr(
        cad_to_simready_runner,
        "execute_cad_to_simready_workflow",
        lambda **_kwargs: pytest.fail("child workflow must not be invoked"),
    )

    config = cad_to_simready_runner.CadToSimReadyRunConfig(
        source_asset=source,
        output_dir=output,
        repo_root=tmp_path,
        materials_yaml=materials,
    )

    with pytest.raises(ValueError) as exc_info:
        cad_to_simready_runner.run_cad_to_simready(config)

    assert not output.exists()
    error = str(exc_info.value)
    assert "supports Linux, WSL2, and native Windows" in error
    assert "usd-cli remote configure" in error
    assert "usd-cli render-probe --require-engine ovrtx" in error
    assert "engine=ovrtx and protocol_version" in error
    assert "plain-text metrics /health endpoint is incompatible" in error


def test_native_windows_reaches_workflow_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "widget.glb"
    source.write_bytes(b"glTF")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(cad_to_simready_runner.sys, "platform", "win32")

    class ReachedWorkflow(RuntimeError):
        pass

    def reached_workflow(**_kwargs):  # noqa: ANN001
        raise ReachedWorkflow("reached composed workflow")

    monkeypatch.setattr(
        cad_to_simready_runner,
        "execute_cad_to_simready_workflow",
        reached_workflow,
    )

    config = cad_to_simready_runner.CadToSimReadyRunConfig(
        source_asset=source,
        output_dir=output,
        repo_root=tmp_path,
        materials_yaml=materials,
    )

    with pytest.raises(ReachedWorkflow, match="reached composed workflow"):
        cad_to_simready_runner.run_cad_to_simready(config)

    assert (output / "request.json").is_file()
    assert (output / "run_plan.json").is_file()


def test_dry_run_remains_available_on_unsupported_native_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "widget.obj"
    source.write_text("o widget\n", encoding="utf-8")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(cad_to_simready_runner.sys, "platform", "darwin")

    result = cad_to_simready_runner.run_cad_to_simready(
        cad_to_simready_runner.CadToSimReadyRunConfig(
            source_asset=source,
            output_dir=output,
            repo_root=tmp_path,
            materials_yaml=materials,
            dry_run=True,
        )
    )

    assert result.status == "planned"
    assert (output / "request.json").is_file()
    assert (output / "run_plan.json").is_file()
    assert (output / "result.json").is_file()


def test_unsupported_console_guard_precedes_workflow_imports() -> None:
    package_root = Path(__file__).parents[1]
    script = """
import builtins
import sys

real_import = builtins.__import__

def reject_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise AssertionError("POSIX dependency imported before Windows guard")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_fcntl
sys.platform = "darwin"
from content_workflow_cli.entrypoint import main
raise SystemExit(main(["cad-to-simready", "run", "widget.obj"]))
"""
    environment = {**os.environ, "PYTHONPATH": str(package_root)}

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert completed.returncode == 2
    assert "supports Linux, WSL2, and native Windows" in completed.stderr
    assert "POSIX dependency imported" not in completed.stderr
    assert 'content-workflow-cli = "content_workflow_cli.entrypoint:main"' in (
        package_root / "pyproject.toml"
    ).read_text(encoding="utf-8")


def test_default_model_is_inherited_and_agent_stages_are_skill_routed(
    tmp_path: Path,
) -> None:
    request, plan = _dry_run(tmp_path)

    assert request["model"] is None
    assert request["workflow_skill"] == "content-workflow-cad-to-simready"
    assert request["prompt_mode"] == "skill-routed"
    assert plan["model"] is None
    assert plan["workflow_skill"] == "content-workflow-cad-to-simready"
    assert plan["workflow_entrypoint"] == ["cad-to-simready", "run"]
    commands = plan["commands"]
    assert [command["step"] for command in commands[:3]] == [
        "convert-to-usd-preflight",
        "physics-runtime-preflight",
        "simready-foundation-preflight",
    ]
    assert [command["step"] for command in commands[3:]] == [
        "convert-to-usd",
        "canonicalize-usd",
        "assign-materials",
        "flatten-for-physics",
        "normalize-units-for-physics",
        "apply-physics",
        "initial-simready-validation",
        "conform-simready-profile",
        "final-simready-validation",
        "render-final-simready",
    ]
    for step in ("assign-materials", "apply-physics"):
        argv = next(command["argv"] for command in commands if command["step"] == step)
        assert argv[argv.index("--optimizer-selection") + 1] == "agent"
        assert argv[argv.index("--prompt-mode") + 1] == "skill-routed"
        assert "--model" not in argv
        assert argv[argv.index("--scene-tool-timeout") + 1] == "300.0"
        assert not any("workbench" in value for value in argv)
    render_argv = next(
        command["argv"]
        for command in commands
        if command["step"] == "render-final-simready"
    )
    assert render_argv[render_argv.index("--scene-tool-timeout") + 1] == "300.0"
    physics_argv = next(
        command["argv"] for command in commands if command["step"] == "apply-physics"
    )
    assert (
        physics_argv[physics_argv.index("--output-usd") + 1]
        .replace("\\", "/")
        .endswith("/physics/physics.usd")
    )
    initial_validation_argv = next(
        command["argv"]
        for command in commands
        if command["step"] == "initial-simready-validation"
    )
    assert (
        initial_validation_argv[3].replace("\\", "/").endswith("/physics/physics.usd")
    )
    assert "gpt-" not in json.dumps(plan)


@pytest.mark.parametrize("platform_name", ["win32", "linux"])
def test_converted_reference_transport_is_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    monkeypatch.setattr(
        cad_to_simready_runner,
        "sys",
        SimpleNamespace(platform=platform_name),
    )

    _request, plan = _dry_run(tmp_path)

    for step in ("assign-materials", "apply-physics"):
        argv = next(
            command["argv"] for command in plan["commands"] if command["step"] == step
        )
        reference = argv[argv.index("--reference") + 1].replace("\\", "/")
        assert reference.endswith("/convert/widget.usda")


def test_explicit_model_override_is_forwarded_to_both_agent_stages(
    tmp_path: Path,
) -> None:
    request, plan = _dry_run(tmp_path, "--model", "explicit-test-model")

    assert request["model"] == "explicit-test-model"
    for step in ("assign-materials", "apply-physics"):
        argv = next(
            command["argv"] for command in plan["commands"] if command["step"] == step
        )
        assert argv[argv.index("--model") + 1] == "explicit-test-model"


def test_explicit_source_units_are_recorded_and_applied_at_physics_handoff(
    tmp_path: Path,
) -> None:
    request, plan = _dry_run(tmp_path, "--source-meters-per-unit", "1.0")

    assert request["source_meters_per_unit"] == 1.0
    assert plan["source_meters_per_unit"] == 1.0
    flatten_argv = next(
        command["argv"]
        for command in plan["commands"]
        if command["step"] == "flatten-for-physics"
    )
    assert flatten_argv[flatten_argv.index("--source-meters-per-unit") + 1] == "1.0"


def test_physics_preflight_reports_exact_install_commands_in_check_only_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))
    runtime_python = runtime / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )

    report = physics_preflight.preflight_ovphysx_runtime(
        repo_root=repo,
        install_missing=False,
    )

    assert report.status == "BLOCKED"
    assert report.install_attempted is False
    assert report.install_commands[0][:4] == ["uv", "venv", "--python", "3.12"]
    assert report.install_commands[1][:3] == ["uv", "pip", "install"]
    assert str(lock) in report.install_commands[1]
    assert report.install_commands[2][0] == str(runtime_python)
    assert not (runtime.parent / f"{runtime.name}.provision.lock").exists()


def test_physics_preflight_installs_probes_then_marks_runtime_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))
    monkeypatch.setenv("PYTHONPATH", "/conflicting/checkout")
    monkeypatch.setattr(physics_preflight, "_uv_command", lambda: ["/usr/bin/uv"])
    runtime_python = runtime / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []
    lock_events: list[tuple[str, Path]] = []

    from usd_core import physics_runtime

    class ObservedProvisionLock:
        def __init__(self, venv_dir: Path) -> None:
            self.venv_dir = venv_dir

        def __enter__(self) -> None:
            lock_events.append(("enter", self.venv_dir))

        def __exit__(self, *_exc: object) -> None:
            lock_events.append(("exit", self.venv_dir))

    monkeypatch.setattr(
        physics_runtime,
        "ovphysx_provision_lock",
        ObservedProvisionLock,
    )

    def fake_run(command, **kwargs):  # noqa: ANN001
        assert lock_events == [("enter", runtime)]
        values = list(command)
        commands.append(values)
        environments.append(kwargs["env"])
        if values[:2] == ["/usr/bin/uv", "venv"]:
            runtime_python.parent.mkdir(parents=True)
            runtime_python.write_text("", encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(physics_preflight.subprocess, "run", fake_run)

    report = physics_preflight.preflight_ovphysx_runtime(repo_root=repo)

    assert report.status == "READY"
    assert report.install_attempted is True
    assert lock_events == [("enter", runtime), ("exit", runtime)]
    assert len(commands) == 3
    assert commands[0][:2] == ["/usr/bin/uv", "venv"]
    assert commands[1][:3] == ["/usr/bin/uv", "pip", "install"]
    assert commands[-1][0] == str(runtime_python)
    assert all("PYTHONPATH" not in environment for environment in environments)
    assert os.environ["PYTHONPATH"] == "/conflicting/checkout"
    marker = json.loads(
        (runtime / ".usd-cli-ovphysx-ready").read_text(encoding="utf-8")
    )
    assert marker["runtime_lock_sha256"] == report.runtime_lock_sha256
    assert report.ready_marker_path == str(runtime / ".usd-cli-ovphysx-ready")

    monkeypatch.setattr(
        physics_runtime, "_ovphysx_runtime_lock", lambda _venv_dir=None: lock
    )
    monkeypatch.setattr(physics_runtime, "_ovphysx_runtime_probe", lambda _python: True)

    def unexpected_reinstall() -> list[str]:
        raise AssertionError("the preflighted usd-cli runtime must be reused")

    monkeypatch.setattr(physics_runtime, "_uv_command", unexpected_reinstall)
    assert physics_runtime._ovphysx_python(runtime) == str(runtime_python)


def test_physics_preflight_timeout_returns_blocked_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))
    monkeypatch.setattr(physics_preflight, "_uv_command", lambda: ["/usr/bin/uv"])

    def time_out(command, **kwargs):  # noqa: ANN001
        assert kwargs["timeout"] == 900.0
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(physics_preflight.subprocess, "run", time_out)

    report = physics_preflight.preflight_ovphysx_runtime(repo_root=repo)

    assert report.status == "BLOCKED"
    assert report.install_attempted is True
    assert report.errors == ["OvPhysX setup command 1 timed out after 900 seconds."]
    assert not (runtime / ".usd-cli-ovphysx-ready").exists()


def test_physics_preflight_rejects_unbound_empty_ready_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o700)
    (runtime / ".usd-cli-ovphysx-ready").touch()
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(tmp_path / "ambient-runtime"))

    report = physics_preflight.preflight_ovphysx_runtime(
        repo_root=repo,
        install_missing=False,
        venv_dir=runtime,
    )

    assert report.status == "BLOCKED"
    assert report.passed is False
    assert report.runtime_ready is False
    assert report.install_attempted is False
    assert report.errors == [
        "OvPhysX runtime is not ready. Run the reported install commands "
        "or rerun preflight with dependency installation enabled."
    ]


def test_physics_preflight_accepts_usd_cli_lock_bound_ready_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    marker = runtime / ".usd-cli-ovphysx-ready"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "usd-cli.ovphysx-runtime-ready.v2",
                "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "python_path": str(python),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))

    report = physics_preflight.preflight_ovphysx_runtime(
        repo_root=repo,
        install_missing=False,
    )

    assert report.status == "READY"
    assert report.runtime_ready is True
    assert report.install_attempted is False
    assert report.ready_marker_path == str(marker)


def test_ready_physics_runtime_does_not_take_provision_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ready runtime remains usable below a read-only parent."""

    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "read-only-parent" / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    marker = runtime / ".usd-cli-ovphysx-ready"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "usd-cli.ovphysx-runtime-ready.v2",
                "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "python_path": str(python),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))

    from usd_core import physics_runtime

    monkeypatch.setattr(
        physics_runtime,
        "ovphysx_provision_lock",
        lambda _runtime: pytest.fail("ready runtime must not take a writable lock"),
    )

    report = physics_preflight.preflight_ovphysx_runtime(repo_root=repo)

    assert report.status == "READY"
    assert report.runtime_ready is True
    assert report.install_attempted is False


def test_physics_preflight_normalizes_provision_lock_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(physics_preflight.sys, "platform", "linux")
    runtime = tmp_path / "read-only-parent" / "runtime"
    repo = tmp_path / "repo"
    lock = _selected_ovphysx_lock(repo, monkeypatch)
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))

    from usd_core import physics_runtime

    class RejectedProvisionLock:
        def __enter__(self) -> None:
            raise PermissionError("read-only runtime parent")

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(
        physics_runtime,
        "ovphysx_provision_lock",
        lambda _runtime: RejectedProvisionLock(),
    )

    report = physics_preflight.preflight_ovphysx_runtime(repo_root=repo)

    assert report.status == "BLOCKED"
    assert report.runtime_ready is False
    assert report.install_attempted is False
    assert report.errors == [
        "OvPhysX runtime provisioning lock is unavailable: read-only runtime parent"
    ]


def test_physics_preflight_reports_local_windows_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from world_understanding.functions.physics import ovphysx_daemon

    repo = tmp_path / "repo"
    lock = repo / "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(lock))
    monkeypatch.setattr(ovphysx_daemon.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ovphysx_daemon.platform, "machine", lambda: "AMD64")

    report = physics_preflight.preflight_ovphysx_runtime(
        repo_root=repo,
        install_missing=False,
    )

    assert report.status == "BLOCKED"
    assert report.executor == "local"
    assert report.remote_url is None
    assert report.runtime_ready is False
    assert report.install_attempted is False
    assert report.lock_path == str(lock)
    assert str(lock) in report.install_commands[1]


def test_physics_preflight_honors_usd_cli_runtime_lock_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    repo = tmp_path / "repo"
    configured_lock = tmp_path / "locks" / "runtime.toml"
    configured_lock.parent.mkdir(parents=True)
    configured_lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(runtime))
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(configured_lock))

    report = physics_preflight.preflight_ovphysx_runtime(
        repo_root=repo,
        install_missing=False,
    )

    assert report.lock_path == str(configured_lock)
    assert str(configured_lock) in report.install_commands[1]


def test_physics_preflight_uv_pin_overrides_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pinned_uv = tmp_path / "trusted" / "uv"
    path_uv = tmp_path / "path" / "uv"
    pinned_uv.parent.mkdir()
    path_uv.parent.mkdir()
    for executable in (pinned_uv, path_uv):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o700)
    monkeypatch.setenv("USD_CLI_UV_EXECUTABLE", str(pinned_uv))
    monkeypatch.setenv("PATH", str(path_uv.parent))

    assert physics_preflight._uv_command() == [str(pinned_uv.resolve())]


def test_physics_preflight_defaults_to_usd_cli_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WU_OVPHYSX_VENV_DIR", raising=False)
    monkeypatch.setattr(physics_preflight.Path, "home", lambda: tmp_path)

    assert physics_preflight._runtime_venv_dir() == (
        tmp_path / ".cache/usd-cli/ovphysx_venv"
    )


def test_physics_preflight_uv_module_precedes_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("USD_CLI_UV_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        physics_preflight.importlib.util,
        "find_spec",
        lambda name: object() if name == "uv" else None,
    )
    monkeypatch.setattr(
        physics_preflight.shutil,
        "which",
        lambda _name: pytest.fail("PATH lookup must not precede the uv module"),
    )

    assert physics_preflight._uv_command() == [
        physics_preflight.sys.executable,
        "-m",
        "uv",
    ]


@pytest.mark.parametrize(
    "argv, expected_count",
    [
        (["simready", "validate-profile", "missing.usd"], 0),
        (
            [
                "simready",
                "validate-profile",
                "<conformed-output-usd>",
                "<conformed-output-usd>",
            ],
            2,
        ),
    ],
)
def test_placeholder_plan_mismatch_returns_typed_failed_invocation(
    argv: list[str],
    expected_count: int,
) -> None:
    command = {
        "stage": "validation",
        "step": "final-simready-validation",
        "argv": argv,
    }
    log = io.StringIO()

    resolved = cad_to_simready_runner._replace_placeholder(
        command,
        "<conformed-output-usd>",
        "/run/final.usd",
        log=log,
    )

    assert isinstance(resolved, cad_to_simready_runner.CadToSimReadyInvocation)
    assert resolved.exit_code == 1
    assert resolved.argv == ["content-workflow-cli", *argv]
    assert f"found {expected_count}" in log.getvalue()
    assert "final-simready-validation" in log.getvalue()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"output_usd_path": None},
        {"output_usd_path": ""},
        {"output_usd_path": []},
    ],
)
def test_conformed_output_resolution_rejects_invalid_report_value(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_dir / "conformance.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no non-empty output USD path"):
        cad_to_simready_runner._resolved_conformed_output_path(
            report_path=report,
            run_dir=run_dir,
        )


@pytest.mark.parametrize("candidate_kind", ["missing", "outside"])
def test_conformed_output_resolution_rejects_unavailable_candidate(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    candidate = (
        run_dir / "missing.usd"
        if candidate_kind == "missing"
        else tmp_path / "outside.usd"
    )
    if candidate_kind == "outside":
        candidate.write_text("#usda 1.0\n", encoding="utf-8")
    report = run_dir / "conformance.json"
    report.write_text(
        json.dumps({"output_usd_path": str(candidate)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output USD path"):
        cad_to_simready_runner._resolved_conformed_output_path(
            report_path=report,
            run_dir=run_dir,
        )


@pytest.mark.parametrize("valid_conformance_report", [True, False])
def test_launcher_delegates_order_and_completion_to_typed_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_conformance_report: bool,
) -> None:
    source = tmp_path / "widget.stl"
    source.write_text(
        "solid widget\n" + ("facet normal 0 0 1\n" * 4_000) + "endsolid widget\n",
        encoding="utf-8",
    )
    assert source.stat().st_size > 65_536
    monkeypatch.setattr(cad_to_simready_runner.sys, "platform", "linux")
    materials = tmp_path / "materials.yaml"
    materials.write_text("materials: []\n", encoding="utf-8")
    config = cad_to_simready_runner.CadToSimReadyRunConfig(
        source_asset=source,
        output_dir=tmp_path / "run",
        repo_root=tmp_path,
        materials_yaml=materials,
    )
    paths = cad_to_simready_runner._paths(config)
    calls: list[str] = []
    invoked_argv: dict[str, list[str]] = {}

    def write(path: Path, payload: object = "artifact") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(str(payload), encoding="utf-8")

    produced = {
        "convert-to-usd-preflight": ["converter_preflight"],
        "physics-runtime-preflight": ["physics_runtime_preflight"],
        "simready-foundation-preflight": ["simready_preflight"],
        "convert-to-usd": ["conversion_report", "converted_usd"],
        "canonicalize-usd": ["canonicalization_report", "canonical_usd"],
        "assign-materials": ["materialized_usd"],
        "flatten-for-physics": ["flatten_report", "flattened_physics_usd"],
        "normalize-units-for-physics": [
            "physics_units_report",
            "physics_input_usd",
        ],
        "apply-physics": ["physics_usd"],
        "initial-simready-validation": ["initial_validation_report"],
        "conform-simready-profile": ["conformance_report"],
        "final-simready-validation": ["final_validation_report"],
        "render-final-simready": [
            "final_render_manifest",
            "final_render_receipt",
            "final_render_receipt_checkpoint",
            "final_render_records",
            "final_hero_render",
            "final_multiview_render",
            "final_turntable_render",
        ],
    }

    def fake_invoke(command, *, log, argv=None):  # noqa: ANN001, ARG001
        step = command["step"]
        calls.append(step)
        invoked_argv[step] = list(argv or command["argv"])
        assert all(
            path.parent.is_dir()
            for name, path in paths.items()
            if name
            not in {
                "source",
                "final_render_receipt",
                "final_render_receipt_checkpoint",
            }
        )
        assert not paths["final_render_receipt"].parent.exists()
        for name in produced[step]:
            payload: object = "artifact"
            if name == "flatten_report":
                payload = {"default_prim_path": "/Widget"}
            if name == "conformance_report":
                final = config.output_dir / "simready/conform/staged/physics.usd"
                write(final)
                payload = (
                    {"output_usd_path": str(final)} if valid_conformance_report else {}
                )
            write(paths[name], payload)
        return cad_to_simready_runner.CadToSimReadyInvocation(
            stage=command["stage"],
            step=step,
            argv=["content-workflow-cli", *(argv or command["argv"])],
            exit_code=0,
        )

    monkeypatch.setattr(cad_to_simready_runner, "_invoke_cli", fake_invoke)

    result = cad_to_simready_runner.run_cad_to_simready(config)

    expected_calls = [
        "convert-to-usd-preflight",
        "physics-runtime-preflight",
        "simready-foundation-preflight",
        "convert-to-usd",
        "canonicalize-usd",
        "assign-materials",
        "flatten-for-physics",
        "normalize-units-for-physics",
        "apply-physics",
        "initial-simready-validation",
        "conform-simready-profile",
    ]
    if not valid_conformance_report:
        assert result.status == "failed"
        assert calls == expected_calls
        failed_invocation = result.stages["validation"].invocations[-1]
        assert failed_invocation.step == "conform-simready-profile"
        assert failed_invocation.exit_code == 1
        assert "conformance report has no non-empty output USD path" in paths[
            "log"
        ].read_text(encoding="utf-8")
        return

    assert result.status == "completed"
    assert calls == [
        *expected_calls,
        "final-simready-validation",
        "render-final-simready",
    ]
    assert result.output_asset is not None
    expected_reference = str(paths["converted_usd"])
    for step in ("assign-materials", "apply-physics"):
        argv = invoked_argv[step]
        assert argv[argv.index("--reference") + 1] == expected_reference
        assert str(source) not in argv
    conform_argv = invoked_argv["conform-simready-profile"]
    assert "<default-prim-from-flatten-report>" not in conform_argv
    assert conform_argv[conform_argv.index("--grasp-prim") + 1] == "/Widget"
    final_usd = str(config.output_dir / "simready/conform/staged/physics.usd")
    for step in ("final-simready-validation", "render-final-simready"):
        assert "<conformed-output-usd>" not in invoked_argv[step]
        assert final_usd in invoked_argv[step]
