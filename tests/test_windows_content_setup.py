# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-Windows content-agent setup contracts."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from content_agent_workflows.common import usd_cli

REPO_ROOT = Path(__file__).parents[1]


def test_powershell_setup_bridges_uv_pwd_and_checks_native_commands() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_content_agent.ps1").read_text(
        encoding="utf-8"
    )

    assert '$env:PWD = "/" + $RepoUri.AbsolutePath.TrimEnd("/")' in setup
    assert '"venv", $VenvDir, "--python", "3.12"' in setup
    assert ') "material-agent installation"' in setup
    assert ') "usd-cli installation"' in setup
    assert ') "content-workflow-cli installation"' in setup
    assert ') "Node SDK installation"' in setup
    assert "& $Executable @CommandArguments" in setup
    assert "[switch]$SkipBuildResources" in setup
    assert "[switch]$WithoutChildRunners" in setup
    assert "InstallClaudeSandbox" not in setup
    assert "@anthropic-ai/sandbox-runtime" not in setup
    assert 'Join-Path $NodeInstallDir "node.exe"' in setup
    assert 'Join-Path $NodeInstallDir "npm.cmd"' in setup
    assert '$env:PATH = "$NodeInstallDir;$env:PATH"' in setup
    assert (
        r'$MaterialAgentPackage = (Join-Path $RepoRoot "apps\material_agent") + "[all]"'
        in setup
    )
    assert (
        r'$UsdCliPackage = (Join-Path $RepoRoot "apps\usd_cli") + "[cli,server]"'
        in setup
    )
    assert (
        r'$UsdExchangeOverride = "apps\usd_cli\requirements\usd-exchange-override.txt"'
        in setup
    )
    assert r'Join-Path $RepoRoot "scripts\fetch_build_resources.ps1"' in setup
    assert "Resolve-Path -LiteralPath" in setup
    assert 'Join-Path $VenvDir "Scripts\\Activate.ps1"' in setup
    assert "Existing .venv has no Scripts\\python.exe" in setup
    assert "Rerun with -RecreateVenv to rebuild it" in setup
    assert setup.index("Existing .venv has no Scripts\\python.exe") < setup.index(
        "Reusing existing .venv"
    )


@pytest.mark.skipif(os.name != "nt", reason="native Windows installer regression")
def test_powershell_setup_preserves_paths_with_spaces(tmp_path: Path) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    profile = tmp_path / "User Profile With Spaces"
    temporary = profile / "Temporary Root With Spaces"
    checkout = temporary / "Checkout With Spaces"
    scripts = checkout / "scripts"
    fake_bin = temporary / "Fake Tool Bin"
    program_files = profile / "Program Files With Spaces"
    node_install = program_files / "nodejs"
    scripts.mkdir(parents=True)
    fake_bin.mkdir(parents=True)
    node_install.mkdir(parents=True)
    setup_script = scripts / "setup_content_agent.ps1"
    shutil.copy2(REPO_ROOT / "scripts" / "setup_content_agent.ps1", setup_script)

    capture = temporary / "uv arguments.txt"
    npm_capture = temporary / "npm arguments.txt"
    fake_uv = fake_bin / "uv.cmd"
    fake_uv.write_text(
        "@echo off\n"
        'set "SETUP_COMMAND=%~1"\n'
        'set "SETUP_VENV=%~2"\n'
        'echo -- invocation -->>"%SETUP_CAPTURE%"\n'
        'echo [PWD=%PWD%]>>"%SETUP_CAPTURE%"\n'
        ":capture\n"
        'if "%~1"=="" goto captured\n'
        'echo [%~1]>>"%SETUP_CAPTURE%"\n'
        "shift\n"
        "goto capture\n"
        ":captured\n"
        'if /I "%SETUP_COMMAND%"=="venv" goto makevenv\n'
        "exit /b 0\n"
        ":makevenv\n"
        'mkdir "%SETUP_VENV%\\Scripts" 2>nul\n'
        'type nul > "%SETUP_VENV%\\Scripts\\python.exe"\n'
        'type nul > "%SETUP_VENV%\\Scripts\\Activate.ps1"\n'
        "exit /b 0\n",
        encoding="utf-8",
    )
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is unavailable")
    shutil.copy2(node_executable, node_install / "node.exe")
    (node_install / "npm.cmd").write_text(
        "@echo off\n"
        ":capture\n"
        'if "%~1"=="" exit /b 0\n'
        'echo [%~1]>>"%NPM_CAPTURE%"\n'
        "shift\n"
        "goto capture\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "SETUP_CAPTURE": str(capture),
        "NPM_CAPTURE": str(npm_capture),
        "ProgramFiles": str(program_files),
        "USERPROFILE": str(profile),
        "APPDATA": str(profile / "App Data Roaming With Spaces"),
        "LOCALAPPDATA": str(profile / "App Data Local With Spaces"),
        "TEMP": str(temporary),
        "TMP": str(temporary),
    }
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(setup_script),
            "-SkipBuildResources",
            "-UvExecutable",
            str(fake_uv),
            "-NodeExecutable",
            str(node_install / "node.exe"),
            "-NpmExecutable",
            str(node_install / "npm.cmd"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8")
    expected_paths = (
        checkout / ".venv",
        checkout / ".venv" / "Scripts" / "python.exe",
        checkout / "apps" / "material_agent",
        checkout / "apps" / "usd_cli",
        checkout / "agentic" / "packages" / "content_workflow_cli",
    )
    normalized_arguments = arguments.replace("[all]", "").replace("[cli,server]", "")
    expected_pwd = checkout.resolve().as_uri().removeprefix("file://").rstrip("/")
    assert f"[PWD={expected_pwd}]" in arguments
    for expected in expected_paths:
        assert f"[{expected}]" in normalized_arguments
    assert "[apps\\usd_cli\\requirements\\usd-exchange-override.txt]" in arguments
    assert (
        f"[{checkout / 'apps' / 'usd_cli' / 'requirements' / 'usd-exchange-override.txt'}]"
        not in arguments
    )
    activation = checkout / ".venv" / "Scripts" / "Activate.ps1"
    assert f'. "{activation}"' in completed.stdout
    npm_arguments = npm_capture.read_text(encoding="utf-8")
    assert "[ci]" in npm_arguments
    assert "[--prefix]" in npm_arguments
    assert f"[{checkout / 'agentic' / 'packages' / 'content_workflow_cli'}]" in (
        npm_arguments
    )


def test_windows_build_resource_fetcher_is_native_and_transactional() -> None:
    fetcher = (REPO_ROOT / "scripts" / "fetch_build_resources.ps1").read_text(
        encoding="utf-8"
    )

    assert "windows-x86_64" in fetcher
    assert "Invoke-WebRequest" in fetcher
    assert "Get-FileHash" in fetcher
    assert "Expand-Archive" in fetcher
    assert fetcher.index("Get-FileHash") < fetcher.index("Expand-Archive")
    assert "$ActualSha256 -ne $TrustedSha256" in fetcher
    assert "archive SHA-256 mismatch" in fetcher
    assert "SO_CORE_SHA256" in fetcher
    assert "35a1e0992dd3c95ec56d93e282ecf6d47e635453cbac8ea243407cddda8fe253" in (
        fetcher
    )
    assert "scene_optimizer_core.backup.$Nonce" in fetcher
    assert "[System.IO.FileShare]::None" in fetcher
    assert "$OwnsPackageDir" in fetcher
    assert "destination exists before install attempt" in fetcher
    assert "partial destination remains" in fetcher
    assert "Test-ExpectedLayout $PackageDir" in fetcher
    install_loop = fetcher[fetcher.index("$Installed = $false") :]
    assert install_loop.index("destination exists before install attempt") < (
        install_loop.index("$OwnsPackageDir = $true")
    )
    assert install_loop.index("Test-ExpectedLayout $PackageDir") < install_loop.index(
        "$Installed = $true"
    )
    assert '"platform=$Platform"' in fetcher
    assert '"archive_sha256=$TrustedSha256"' in fetcher
    for expected_directory in ("python", "lib", "extraLibs", "usdpy"):
        assert f'"{expected_directory}"' in fetcher


def test_windows_setup_is_explicitly_development_only() -> None:
    """Keep the retained Windows helper out of the supported 0.6 route."""
    setup = (REPO_ROOT / "scripts" / "setup_content_agent.ps1").read_text(
        encoding="utf-8"
    )

    assert "Native Windows execution and local OVRTX rendering are unsupported" in setup
    assert "run supported workflows inside WSL2" in setup
    assert "development evidence and future qualification only" in setup
    assert "Local OVRTX rendering is supported" not in setup


def test_windows_git_resolution_uses_machine_program_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    git = tmp_path / "Git" / "cmd" / "git.exe"
    git.parent.mkdir(parents=True)
    git.write_bytes(b"trusted machine Git")
    monkeypatch.setattr(
        usd_cli,
        "_windows_program_files_directories",
        lambda: (tmp_path,),
    )

    assert usd_cli._windows_system_git_executable() == str(git.resolve())
