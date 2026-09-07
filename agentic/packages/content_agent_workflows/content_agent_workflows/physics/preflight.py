# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness and deterministic setup for the isolated OvPhysX runtime."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.artifacts import file_sha256

PHYSICS_RUNTIME_PREFLIGHT_SCHEMA_VERSION = (
    "content-agent-workflows.physics-runtime-preflight.v1"
)
_UV_EXECUTABLE_ENV = "USD_CLI_UV_EXECUTABLE"
_OVPHYSX_VENV_ENV = "WU_OVPHYSX_VENV_DIR"
_OVPHYSX_READY_MARKER = ".usd-cli-ovphysx-ready"
_OVPHYSX_READY_MARKER_SCHEMA_VERSION = "usd-cli.ovphysx-runtime-ready.v2"
_PHYSICS_RUNTIME_SETUP_TIMEOUT_SECONDS = 900.0


def _runtime_venv_dir() -> Path:
    configured = os.environ.get(_OVPHYSX_VENV_ENV)
    return Path(
        configured
        if configured
        else Path.home() / ".cache" / "usd-cli" / "ovphysx_venv"
    ).expanduser()


def _runtime_spec(
    repo_root: Path,
    *,
    venv_dir: Path | str | None = None,
):
    from world_understanding.functions.physics import resolve_ovphysx_runtime_spec

    return resolve_ovphysx_runtime_spec(
        repo_root,
        venv_dir=_runtime_venv_dir() if venv_dir is None else venv_dir,
    )


def _runtime_install_commands(
    repo_root: Path,
    *,
    venv_dir: Path | str | None = None,
) -> tuple[tuple[str, ...], ...]:
    from world_understanding.functions.physics import ovphysx_runtime_install_commands

    return ovphysx_runtime_install_commands(
        repo_root,
        venv_dir=_runtime_venv_dir() if venv_dir is None else venv_dir,
    )


def _runtime_ready_marker(venv_dir: Path) -> Path:
    return venv_dir / _OVPHYSX_READY_MARKER


def _uv_command() -> list[str]:
    """Resolve uv without leaving executable selection to child PATH lookup."""

    pinned_executable = os.environ.get(_UV_EXECUTABLE_ENV)
    if pinned_executable:
        executable_path = Path(pinned_executable).expanduser()
        if (
            not executable_path.is_absolute()
            or not executable_path.is_file()
            or not os.access(executable_path, os.X_OK)
        ):
            raise RuntimeError(
                f"{_UV_EXECUTABLE_ENV} does not identify an executable file: "
                f"{pinned_executable}"
            )
        return [str(executable_path.resolve(strict=True))]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    executable = shutil.which("uv")
    if executable:
        return [str(Path(executable).resolve(strict=True))]
    raise RuntimeError("uv is required to install the isolated OvPhysX runtime.")


class PhysicsRuntimePreflightReport(BaseModel):
    """Normalized readiness evidence for one selected simulation backend."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHYSICS_RUNTIME_PREFLIGHT_SCHEMA_VERSION
    status: Literal["READY", "BLOCKED"]
    passed: bool
    engine: Literal["ovphysx"] = "ovphysx"
    executor: Literal["local", "remote"] = "local"
    remote_url: str | None = None
    runtime_ready: bool
    venv_path: str
    python_path: str
    lock_path: str
    runtime_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ready_marker_path: str
    install_attempted: bool = False
    install_commands: list[list[str]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _blocked_report(
    *,
    repo_root: Path,
    install_attempted: bool,
    errors: list[str],
    include_install_commands: bool = True,
    venv_dir: Path | str | None = None,
) -> PhysicsRuntimePreflightReport:
    spec = _runtime_spec(repo_root, venv_dir=venv_dir)
    commands = _runtime_install_commands(repo_root, venv_dir=venv_dir)
    ready_marker = _runtime_ready_marker(spec.venv_dir)
    return PhysicsRuntimePreflightReport(
        status="BLOCKED",
        passed=False,
        runtime_ready=False,
        venv_path=str(spec.venv_dir),
        python_path=str(spec.python_path),
        lock_path=str(spec.lock_path),
        runtime_lock_sha256=(
            file_sha256(spec.lock_path) if spec.lock_path.is_file() else None
        ),
        ready_marker_path=str(ready_marker),
        install_attempted=install_attempted,
        install_commands=(
            [list(command) for command in commands] if include_install_commands else []
        ),
        errors=errors,
    )


def _preflight_ovphysx_runtime_impl(
    *,
    repo_root: Path | str,
    install_missing: bool = True,
    venv_dir: Path | str | None = None,
) -> PhysicsRuntimePreflightReport:
    """Verify or bootstrap; callers serialize the mutating path."""

    resolved_repo = Path(repo_root).expanduser().resolve()
    spec = _runtime_spec(resolved_repo, venv_dir=venv_dir)
    commands = _runtime_install_commands(resolved_repo, venv_dir=venv_dir)
    ready_marker = _runtime_ready_marker(spec.venv_dir)
    lock_sha256 = file_sha256(spec.lock_path) if spec.lock_path.is_file() else None

    def marker_matches_lock() -> bool:
        if lock_sha256 is None:
            return False
        try:
            marker_text = ready_marker.read_text(encoding="utf-8")
        except OSError:
            return False
        # The daemon's legacy readiness helper accepts a touched marker, but this
        # workflow publishes exact-lock evidence. An empty marker cannot prove
        # which lock provisioned the environment, so preflight must upgrade it by
        # reinstalling (when allowed) or fail closed in check-only mode.
        if not marker_text.strip():
            return False
        try:
            payload = json.loads(marker_text)
        except json.JSONDecodeError:
            return False
        return bool(
            isinstance(payload, dict)
            and payload.get("schema_version") == _OVPHYSX_READY_MARKER_SCHEMA_VERSION
            and payload.get("runtime_lock_sha256") == lock_sha256
            and payload.get("python_path") == str(spec.python_path)
        )

    if spec.python_path.is_file() and marker_matches_lock():
        return PhysicsRuntimePreflightReport(
            status="READY",
            passed=True,
            runtime_ready=True,
            venv_path=str(spec.venv_dir),
            python_path=str(spec.python_path),
            lock_path=str(spec.lock_path),
            runtime_lock_sha256=lock_sha256,
            ready_marker_path=str(ready_marker),
            install_commands=[list(command) for command in commands],
        )
    if not spec.lock_path.is_file():
        return _blocked_report(
            repo_root=resolved_repo,
            install_attempted=False,
            errors=[f"OvPhysX runtime lock is missing: {spec.lock_path}"],
            venv_dir=venv_dir,
        )
    if not install_missing:
        return _blocked_report(
            repo_root=resolved_repo,
            install_attempted=False,
            errors=[
                "OvPhysX runtime is not ready. Run the reported install commands "
                "or rerun preflight with dependency installation enabled."
            ],
            venv_dir=venv_dir,
        )
    try:
        uv_command = _uv_command()
    except RuntimeError as exc:
        return _blocked_report(
            repo_root=resolved_repo,
            install_attempted=False,
            errors=[str(exc)],
            venv_dir=venv_dir,
        )

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for index, command in enumerate(commands):
        executable_command = (
            [*uv_command, *command[1:]] if command[0] == "uv" else list(command)
        )
        try:
            completed = subprocess.run(
                executable_command,
                cwd=resolved_repo,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=_PHYSICS_RUNTIME_SETUP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return _blocked_report(
                repo_root=resolved_repo,
                install_attempted=True,
                errors=[
                    f"OvPhysX setup command {index + 1} timed out after "
                    f"{_PHYSICS_RUNTIME_SETUP_TIMEOUT_SECONDS:g} seconds."
                ],
                venv_dir=venv_dir,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            if len(detail) > 4000:
                detail = detail[-4000:]
            return _blocked_report(
                repo_root=resolved_repo,
                install_attempted=True,
                errors=[
                    f"OvPhysX setup command {index + 1} failed with exit code "
                    f"{completed.returncode}: {detail or 'no output'}"
                ],
                venv_dir=venv_dir,
            )

    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text(
        json.dumps(
            {
                "schema_version": _OVPHYSX_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": lock_sha256,
                "python_path": str(spec.python_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not spec.python_path.is_file() or not marker_matches_lock():
        return _blocked_report(
            repo_root=resolved_repo,
            install_attempted=True,
            errors=["OvPhysX setup completed but the readiness contract did not pass."],
            venv_dir=venv_dir,
        )
    return PhysicsRuntimePreflightReport(
        status="READY",
        passed=True,
        runtime_ready=True,
        venv_path=str(spec.venv_dir),
        python_path=str(spec.python_path),
        lock_path=str(spec.lock_path),
        runtime_lock_sha256=lock_sha256,
        ready_marker_path=str(ready_marker),
        install_attempted=True,
        install_commands=[list(command) for command in commands],
    )


def preflight_ovphysx_runtime(
    *,
    repo_root: Path | str,
    install_missing: bool = True,
    venv_dir: Path | str | None = None,
) -> PhysicsRuntimePreflightReport:
    """Verify or bootstrap the exact locked OvPhysX daemon environment."""

    from usd_core import physics_runtime

    if not install_missing:
        return _preflight_ovphysx_runtime_impl(
            repo_root=repo_root,
            install_missing=False,
            venv_dir=venv_dir,
        )
    # A lock is needed only for provisioning.  Checking an already-ready
    # runtime is read-only and must continue to work when the runtime's parent
    # is intentionally not writable.  The implementation rechecks readiness
    # under the lock below so concurrent installers remain serialized.
    ready_report = _preflight_ovphysx_runtime_impl(
        repo_root=repo_root,
        install_missing=False,
        venv_dir=venv_dir,
    )
    if ready_report.runtime_ready:
        return ready_report
    resolved_venv_dir = _runtime_spec(
        Path(repo_root).expanduser().resolve(),
        venv_dir=venv_dir,
    ).venv_dir
    with ExitStack() as stack:
        try:
            stack.enter_context(
                physics_runtime.ovphysx_provision_lock(resolved_venv_dir)
            )
        except (OSError, RuntimeError) as exc:
            return _blocked_report(
                repo_root=Path(repo_root).expanduser().resolve(),
                install_attempted=False,
                errors=[f"OvPhysX runtime provisioning lock is unavailable: {exc}"],
                venv_dir=resolved_venv_dir,
            )
        return _preflight_ovphysx_runtime_impl(
            repo_root=repo_root,
            install_missing=True,
            venv_dir=resolved_venv_dir,
        )
