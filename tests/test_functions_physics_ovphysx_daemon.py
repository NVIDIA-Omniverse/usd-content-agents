# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ovphysx daemon venv discovery and install hints."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

from world_understanding.functions.physics import ovphysx_daemon as daemon_mod
from world_understanding.functions.physics import ovphysx_process_limits as limit_mod


class _FakeResource:
    RLIMIT_AS = 9

    def __init__(self, limits: tuple[int, int]) -> None:
        self.limits = limits
        self.applied: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(self, resource_kind: int) -> tuple[int, int]:
        assert resource_kind == self.RLIMIT_AS
        return self.limits

    def setrlimit(self, resource_kind: int, limits: tuple[int, int]) -> None:
        self.applied.append((resource_kind, limits))


@pytest.mark.parametrize(
    ("system", "machine", "name"),
    [
        ("Linux", "AMD64", "pylock.ovphysx-runtime.toml"),
        ("Linux", "aarch64", "pylock.ovphysx-runtime.aarch64.toml"),
    ],
)
def test_runtime_spec_uses_packaged_reviewed_lock_without_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    name: str,
) -> None:
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(daemon_mod.platform, "system", lambda: system)
    monkeypatch.setattr(daemon_mod.platform, "machine", lambda: machine)

    spec = daemon_mod.resolve_ovphysx_runtime_spec(
        tmp_path / "not-a-checkout", venv_dir=tmp_path / "venv"
    )

    assert spec.lock_path == daemon_mod._THIS_DIR / name
    assert (
        spec.lock_path.read_bytes()
        == (
            daemon_mod._source_checkout_root() / "apps/physics_agent/runtime" / name
        ).read_bytes()
    )


def test_windows_runtime_spec_remains_checkout_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    lock = repo / "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
    lock.parent.mkdir(parents=True)
    lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(daemon_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(daemon_mod.platform, "machine", lambda: "AMD64")

    spec = daemon_mod.resolve_ovphysx_runtime_spec(repo, venv_dir=tmp_path / "venv")

    assert spec.lock_path == lock


def test_missing_packaged_lock_does_not_fall_back_to_site_packages_apps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "site-packages/world_understanding/functions/physics"
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(daemon_mod, "_THIS_DIR", package_dir)
    monkeypatch.setattr(daemon_mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(daemon_mod.platform, "machine", lambda: "AMD64")

    spec = daemon_mod.resolve_ovphysx_runtime_spec(
        tmp_path / "site-packages", venv_dir=tmp_path / "venv"
    )

    assert spec.lock_path == package_dir / "pylock.ovphysx-runtime.toml"
    assert "site-packages/apps" not in str(spec.lock_path)


def test_ovphysx_venv_python_path_uses_scripts_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon_mod._ovphysx_venv_python_path(tmp_path) == (
        tmp_path / "Scripts" / "python.exe"
    )


def test_ovphysx_venv_python_path_uses_bin_on_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon_mod.os, "name", "posix")

    assert daemon_mod._ovphysx_venv_python_path(tmp_path) == (
        tmp_path / "bin" / "python"
    )


def test_resolve_python_accepts_windows_venv_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = tmp_path / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    runtime_lock = (
        daemon_mod._source_checkout_root() / daemon_mod._ovphysx_runtime_lock()
    )
    (tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER).write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod.os, "name", "nt")
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: True)
    assert daemon._resolve_python() == python_path


def test_resolve_python_reuses_ready_runtime_without_provision_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_path = tmp_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    lock_path = daemon_mod._source_checkout_root() / daemon_mod._ovphysx_runtime_lock()
    (tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER).write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    lock_path.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon_mod,
        "_ovphysx_provision_lock",
        lambda *_args, **_kwargs: pytest.fail("ready runtime must not lock"),
    )
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: True)

    assert daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)._resolve_python() == python_path


def test_missing_daemon_venv_hint_targets_platform_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setenv("WU_OVPHYSX_AUTO_PROVISION", "0")
    monkeypatch.setattr(daemon_mod.os, "name", "nt")
    monkeypatch.setattr(
        daemon_mod, "_ovphysx_provision_lock", lambda *_args: nullcontext()
    )

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError) as exc_info:
        daemon._resolve_python()

    message = str(exc_info.value)
    assert "uv pip install --python" in message
    assert "uv venv --python 3.12 --allow-existing" in message
    assert str(tmp_path / "Scripts" / "python.exe") in message
    assert "exact reviewed runtime" in message
    assert (
        str(daemon_mod._THIS_DIR / daemon_mod._ovphysx_runtime_lock().name) in message
    )
    assert "--require-hashes --no-deps" in message
    assert "--no-config --no-sources" in message
    assert "from ovphysx import PhysX" in message
    assert str(tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER) in message
    assert "--extra-index-url" not in message


def test_resolve_python_auto_provisions_missing_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / "ovphysx-runtime"
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=venv_dir)
    commands: list[tuple[str, ...] | list[str]] = []

    monkeypatch.delenv("WU_OVPHYSX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(daemon_mod.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        daemon_mod,
        "resolve_ovphysx_runtime_spec",
        lambda _root, *, venv_dir: daemon_mod.OvPhysXRuntimeSpec(
            venv_dir=venv_dir,
            python_path=venv_dir / "bin" / "python",
            lock_path=tmp_path / "pylock.toml",
            ready_marker_path=venv_dir / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER,
        ),
    )
    (tmp_path / "pylock.toml").touch()
    monkeypatch.setattr(
        daemon_mod,
        "ovphysx_runtime_install_commands",
        lambda _root, *, venv_dir: (
            ("uv", "venv", str(venv_dir)),
            ("uv", "pip"),
            ("probe",),
        ),
    )

    def fake_run(
        command: tuple[str, ...] | list[str], check: bool, **kwargs: object
    ) -> None:
        commands.append(command)
        environment = kwargs["env"]
        assert isinstance(environment, Mapping)
        assert environment.get("PYTHONPATH") is None
        assert kwargs["timeout"] == daemon_mod._OVPHYSX_PROVISION_TIMEOUT_S
        if command[1] == "venv":
            (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
            (venv_dir / "bin" / "python").touch()

    monkeypatch.setattr(daemon_mod.subprocess, "run", fake_run)

    assert daemon._resolve_python() == venv_dir / "bin" / "python"
    assert commands == [
        ("/usr/bin/uv", "venv", str(venv_dir)),
        ("/usr/bin/uv", "pip"),
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()",
        ],
    ]
    assert (venv_dir / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER).is_file()


def test_native_windows_source_checkout_does_not_auto_provision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path / "runtime")
    monkeypatch.delenv("WU_OVPHYSX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(daemon_mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        daemon_mod, "_ovphysx_provision_lock", lambda *_args: nullcontext()
    )
    monkeypatch.setattr(
        daemon_mod,
        "_provision_ovphysx_runtime_lock_held",
        lambda *_args: pytest.fail("native Windows must not auto-provision"),
    )

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError):
        daemon._resolve_python()


def test_provision_rejects_active_parent_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = daemon_mod.OvPhysXRuntimeSpec(
        venv_dir=tmp_path,
        python_path=tmp_path / "bin/python",
        lock_path=tmp_path / "runtime.toml",
        ready_marker_path=tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER,
    )
    monkeypatch.setattr(daemon_mod.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        daemon_mod.shutil,
        "which",
        lambda *_args: pytest.fail("active environment must be rejected before uv"),
    )

    with pytest.raises(
        daemon_mod.OvPhysXDaemonUnavailableError,
        match="active parent environment",
    ):
        daemon_mod._provision_ovphysx_runtime_lock_held(spec)


def test_provision_lock_held_preserves_valid_runtime_without_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / "ovphysx-runtime"
    python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    runtime_lock = tmp_path / "runtime.toml"
    runtime_lock.write_text("locked", encoding="utf-8")
    ready_marker = venv_dir / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER
    ready_marker.write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    spec = daemon_mod.OvPhysXRuntimeSpec(
        venv_dir=venv_dir,
        python_path=python_path,
        lock_path=runtime_lock,
        ready_marker_path=ready_marker,
    )
    monkeypatch.setattr(daemon_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: True)

    assert daemon_mod._provision_ovphysx_runtime_lock_held(spec) == python_path
    assert ready_marker.is_file()


def test_provision_lock_held_repairs_marker_bound_runtime_after_probe_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / "ovphysx-runtime"
    python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    runtime_lock = tmp_path / "runtime.toml"
    runtime_lock.write_text("locked", encoding="utf-8")
    ready_marker = venv_dir / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER
    ready_marker.write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    spec = daemon_mod.OvPhysXRuntimeSpec(
        venv_dir=venv_dir,
        python_path=python_path,
        lock_path=runtime_lock,
        ready_marker_path=ready_marker,
    )
    commands: list[tuple[str, ...] | list[str]] = []
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: False)
    monkeypatch.setattr(daemon_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(
        daemon_mod,
        "ovphysx_runtime_install_commands",
        lambda _root, *, venv_dir: (
            ("uv", "venv", str(venv_dir)),
            ("uv", "pip", "install"),
            ("probe",),
        ),
    )

    def fake_run(command, **_kwargs):
        commands.append(command)

    monkeypatch.setattr(daemon_mod.subprocess, "run", fake_run)

    assert daemon_mod._provision_ovphysx_runtime_lock_held(spec) == python_path
    assert commands[0] == ("/usr/bin/uv", "venv", str(venv_dir))
    assert commands[1] == ("/usr/bin/uv", "pip", "install")
    assert commands[2][0] == str(python_path)
    assert ready_marker.is_file()


def test_start_fresh_runtime_acquires_provision_lock_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquisitions = 0
    lock_held = False
    python_path = tmp_path / "bin" / "python"

    @contextmanager
    def fake_provision_lock(_venv_dir: Path) -> Iterator[None]:
        nonlocal acquisitions, lock_held
        acquisitions += 1
        assert lock_held is False
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    class _FakeProcess:
        pid = 1234
        stderr: list[str] = []

        def poll(self) -> None:
            return None

    def fake_provision(spec: daemon_mod.OvPhysXRuntimeSpec) -> Path:
        assert lock_held is True
        python_path.parent.mkdir(parents=True)
        python_path.touch()
        return spec.python_path

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        assert lock_held is True
        return _FakeProcess()

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod, "_ovphysx_provision_lock", fake_provision_lock)
    monkeypatch.setattr(
        daemon_mod, "_provision_ovphysx_runtime_lock_held", fake_provision
    )
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)

    def ready_line(*_args: object) -> str:
        assert lock_held is True
        return '{"status": "ready"}\n'

    monkeypatch.setattr(daemon, "_read_stdout_line", ready_line)

    daemon._start()

    assert acquisitions == 1
    assert lock_held is False


def test_start_uses_prevalidated_read_only_baked_runtime_on_lock_eacces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = tmp_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    runtime_lock = tmp_path / "runtime.toml"
    runtime_lock.write_text("locked", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(runtime_lock))
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: True)
    (tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER).write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )

    @contextmanager
    def denied_lock(_venv_dir: Path) -> Iterator[None]:
        raise PermissionError(errno.EACCES, "read-only baked runtime")
        yield  # pragma: no cover

    class _FakeProcess:
        pid = 1234
        stderr: list[str] = []

        def poll(self) -> None:
            return None

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod, "_ovphysx_provision_lock", denied_lock)
    monkeypatch.setattr(daemon_mod.os, "access", lambda *_args: False)
    monkeypatch.setattr(
        daemon,
        "_resolve_python_lock_held",
        lambda _spec: pytest.fail("read-only fallback must use bound Python"),
    )
    monkeypatch.setattr(
        daemon_mod.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProcess()
    )
    monkeypatch.setattr(
        daemon, "_read_stdout_line", lambda *_args: '{"status": "ready"}\n'
    )

    daemon._start()

    assert daemon._is_running() is True


@pytest.mark.parametrize(
    "lock_error",
    [
        OSError(errno.EIO, "lock storage unavailable"),
        PermissionError(errno.EPERM, "lock operation rejected"),
    ],
)
def test_start_classifies_non_fallback_lock_entry_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_error: OSError,
) -> None:
    @contextmanager
    def broken_lock(_venv_dir: Path) -> Iterator[None]:
        raise lock_error
        yield  # pragma: no cover

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod, "_ovphysx_provision_lock", broken_lock)
    monkeypatch.setattr(
        daemon_mod.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("lock failure must prevent spawn"),
    )

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError) as exc_info:
        daemon._start()

    message = str(exc_info.value)
    assert message.startswith("Unable to lock OvPhysX runtime")
    assert "failed to spawn ovphysx daemon" not in message


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock interoperability guard")
def test_provision_lock_blocks_usd_cli_compatible_advisory_waiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "apps/usd_cli/src"))
    from usd_core.windows_files import advisory_file_lock

    lock_path = daemon_mod._ovphysx_provision_lock_path(tmp_path)
    waiter_started = threading.Event()
    waiter_entered = threading.Event()

    def wait_with_usd_cli_primitive() -> None:
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            waiter_started.set()
            with advisory_file_lock(descriptor):
                waiter_entered.set()
        finally:
            os.close(descriptor)

    with daemon_mod._ovphysx_provision_lock(tmp_path):
        inode = lock_path.stat().st_ino
        waiter = threading.Thread(target=wait_with_usd_cli_primitive)
        waiter.start()
        assert waiter_started.wait(timeout=1)
        assert waiter_entered.wait(timeout=0.1) is False

    assert waiter_entered.wait(timeout=1)
    waiter.join(timeout=1)
    assert waiter.is_alive() is False
    assert lock_path.stat().st_ino == inode


def test_resolve_python_reprovisions_incomplete_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_dir = tmp_path / "ovphysx-runtime"
    python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=venv_dir)
    provisioned: list[Path] = []

    monkeypatch.setattr(
        daemon_mod,
        "_provision_ovphysx_runtime",
        lambda path: provisioned.append(path) or python_path,
    )

    assert daemon._resolve_python() == python_path
    assert provisioned == [venv_dir]


def test_resolve_python_respects_disabled_auto_provisioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_OVPHYSX_AUTO_PROVISION", "0")

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError):
        daemon_mod._OvPhysXDaemon(venv_dir=tmp_path / "missing")._resolve_python()


def test_runtime_lock_selection_is_platform_and_architecture_specific() -> None:
    assert daemon_mod._ovphysx_runtime_lock("x86_64", "Linux") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.toml"
    )
    assert daemon_mod._ovphysx_runtime_lock("aarch64", "Linux") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    )
    assert daemon_mod._ovphysx_runtime_lock("arm64", "Linux") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    )
    assert daemon_mod._ovphysx_runtime_lock("AMD64", "Windows") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime-windows.toml"
    )
    with pytest.raises(ValueError, match="Unsupported architecture"):
        daemon_mod._ovphysx_runtime_lock("ppc64le", "Linux")
    with pytest.raises(ValueError, match="Unsupported operating system"):
        daemon_mod._ovphysx_runtime_lock("x86_64", "Darwin")


def test_unavailable_error_defers_architecture_selection_until_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon_mod.platform, "machine", lambda: "ppc64le")

    assert daemon_mod.OvPhysXDaemonUnavailableError.DEFAULT_MESSAGE == (
        "ovphysx daemon is not available."
    )
    with pytest.raises(ValueError, match="Unsupported architecture"):
        daemon_mod.OvPhysXDaemonUnavailableError()


def test_runtime_availability_requires_python_and_success_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WU_OVPHYSX_VENV_DIR", str(tmp_path))
    python_path = tmp_path / "bin" / "python"
    ready_marker = tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER

    assert daemon_mod.ovphysx_runtime_available() is False

    python_path.parent.mkdir(parents=True)
    python_path.touch()
    assert daemon_mod.ovphysx_runtime_available() is False

    lock_path = tmp_path / "lock.toml"
    lock_path.write_text("locked", encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(lock_path))
    ready_marker.write_text(
        json.dumps(
            {
                "schema_version": daemon_mod._OVPHYSX_RUNTIME_READY_MARKER_SCHEMA_VERSION,
                "runtime_lock_sha256": hashlib.sha256(
                    lock_path.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: False)
    assert daemon_mod.ovphysx_runtime_available() is False
    monkeypatch.setattr(daemon_mod, "_ovphysx_runtime_probe", lambda _path: True)
    assert daemon_mod.ovphysx_runtime_available() is True


def test_runtime_install_commands_allow_existing_isolated_venv(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    venv_dir = tmp_path / "ovphysx-runtime"
    venv_dir.mkdir()

    create, install, smoke_test = daemon_mod.ovphysx_runtime_install_commands(
        repo_root,
        venv_dir=venv_dir,
    )

    assert create == (
        "uv",
        "venv",
        "--python",
        "3.12",
        "--allow-existing",
        str(venv_dir),
    )
    assert install[:4] == (
        "uv",
        "pip",
        "install",
        "--python",
    )
    assert install[4] == str(daemon_mod._ovphysx_venv_python_path(venv_dir))
    assert install[-1] == "--no-sources"
    assert smoke_test[0] == str(daemon_mod._ovphysx_venv_python_path(venv_dir))


def test_runtime_lock_override_aligns_spec_and_install_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    venv_dir = tmp_path / "ovphysx-runtime"
    configured_lock = tmp_path / "locks" / "runtime.toml"
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(configured_lock))

    spec = daemon_mod.resolve_ovphysx_runtime_spec(
        repo_root,
        venv_dir=venv_dir,
    )
    _create, install, _smoke_test = daemon_mod.ovphysx_runtime_install_commands(
        repo_root,
        venv_dir=venv_dir,
    )

    assert spec.lock_path == configured_lock
    assert install[install.index("-r") + 1] == str(configured_lock)


def test_read_stdout_line_uses_threaded_pipe_reader_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PipeLikeStdout:
        def fileno(self) -> int:  # pragma: no cover - must not be used on Windows
            raise AssertionError("Windows pipe reader should not call fileno()")

        def readline(self) -> str:
            return '{"status": "ready"}\n'

    class _FakeProcess:
        stdout = _PipeLikeStdout()

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _FakeProcess()  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon._read_stdout_line(1.0, "startup") == '{"status": "ready"}\n'


def test_threaded_stdout_reader_timeout_kills_process_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowStdout:
        def readline(self) -> str:
            time.sleep(0.2)
            return ""

    class _FakeProcess:
        stdout = _SlowStdout()
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            return 0

    process = _FakeProcess()
    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = process  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="startup timed out"):
        daemon._read_stdout_line(0.01, "startup")

    assert process.killed is True


def test_threaded_stdout_reader_wraps_readline_errors_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingStdout:
        def readline(self) -> str:
            raise RuntimeError("pipe broke")

    class _FakeProcess:
        stdout = _FailingStdout()

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _FakeProcess()  # type: ignore[assignment]
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="stdout read failed"):
        daemon._read_stdout_line(1.0, "evaluate")


@pytest.mark.parametrize(
    ("relax_address_space_limit", "expects_marker"),
    [(False, False), (True, True)],
)
def test_daemon_start_scopes_environment_and_address_space_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relax_address_space_limit: bool,
    expects_marker: bool,
) -> None:
    seen_env: dict[str, str] = {}

    class _FakeProcess:
        pid = 1234
        stderr = ["", "ready on stderr\n"]

        def poll(self) -> None:
            return None

    def fake_popen(*args, **kwargs):
        seen_env.update(kwargs["env"])
        return _FakeProcess()

    daemon = daemon_mod._OvPhysXDaemon(
        venv_dir=tmp_path,
        device="cuda:0",
        relax_address_space_limit=relax_address_space_limit,
    )
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "ready"}\n',
    )
    monkeypatch.setenv("PYTHONPATH", "parent-path")
    monkeypatch.setenv(
        limit_mod.OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV,
        "ambient-value-must-not-authorize",
    )

    daemon._start()

    assert daemon._is_running() is True
    assert "PYTHONPATH" not in seen_env
    assert seen_env["WU_OVPHYSX_DEVICE"] == "cuda:0"
    assert (
        seen_env.get(limit_mod.OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV) == "1"
    ) is expects_marker


def test_daemon_consumes_marker_and_raises_only_soft_address_space_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _FakeResource((640 * 1024 * 1024, 4096 * 1024 * 1024))
    monkeypatch.setattr(limit_mod, "_resource", resource)
    environment = {limit_mod.OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV: "1"}

    assert limit_mod.relax_address_space_limit_from_environment(environment) is True

    assert environment == {}
    assert resource.applied == [
        (resource.RLIMIT_AS, (4096 * 1024 * 1024, 4096 * 1024 * 1024))
    ]


@pytest.mark.parametrize("marker", [None, "0", "ambient"])
def test_daemon_does_not_relax_address_space_without_one_shot_marker(
    monkeypatch: pytest.MonkeyPatch,
    marker: str | None,
) -> None:
    resource = _FakeResource((640, 4096))
    monkeypatch.setattr(limit_mod, "_resource", resource)
    environment = {}
    if marker is not None:
        environment[limit_mod.OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV] = marker

    assert limit_mod.relax_address_space_limit_from_environment(environment) is False

    assert environment == {}
    assert resource.applied == []


def test_daemon_start_wraps_spawn_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(
        daemon_mod.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("nope")),
    )

    with pytest.raises(
        daemon_mod.OvPhysXDaemonUnavailableError, match="failed to spawn"
    ):
        daemon._start()


def test_daemon_start_reports_missing_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(daemon_mod, "_DAEMON_SCRIPT_PATH", tmp_path / "missing.py")

    with pytest.raises(
        daemon_mod.OvPhysXDaemonUnavailableError, match="daemon script missing"
    ):
        daemon._start()


@pytest.mark.parametrize(
    ("ready_line", "expected_message"),
    [
        ("", "exited during start-up"),
        ("not-json\n", "ready line was not JSON"),
        ('{"status": "error", "error": "bad import"}\n', "bad import"),
        ('{"status": "weird"}\n', "unexpected start-up message"),
    ],
)
def test_daemon_start_ready_line_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready_line: str,
    expected_message: str,
) -> None:
    class _FakeProcess:
        pid = 99
        stderr: list[str] = []
        killed = False

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            return 13

        def kill(self) -> None:
            self.killed = True

    process = _FakeProcess()
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        daemon, "_read_stdout_line", lambda timeout_s, phase: ready_line
    )

    with pytest.raises(
        daemon_mod.OvPhysXDaemonUnavailableError, match=expected_message
    ):
        daemon._start()


def test_daemon_start_includes_stderr_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProcess:
        pid = 99
        stderr = [
            "Failed to preload USD: libX11.so.6: cannot open shared object file\n"
        ]

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float) -> int:
            return 1

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(
        daemon_mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess()
    )
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: (
            '{"status": "error", "error": "Failed to preload USD libraries"}\n'
        ),
    )

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError) as exc_info:
        daemon._start()

    message = str(exc_info.value)
    assert "Failed to preload USD libraries" in message
    assert "ovphysx stderr (tail)" in message
    assert "libX11.so.6" in message


def test_with_stderr_tail_waits_for_active_drain_thread() -> None:
    class _ActiveThread:
        def __init__(self) -> None:
            self.join_timeout: float | None = None

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    daemon = daemon_mod._OvPhysXDaemon()
    thread = _ActiveThread()
    daemon._stderr_thread = thread  # type: ignore[assignment]

    assert daemon._with_stderr_tail("startup failed") == "startup failed"
    assert thread.join_timeout == 1


def test_daemon_start_converts_read_timeout_to_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProcess:
        pid = 99
        stderr: list[str] = []

        def poll(self) -> None:
            return None

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(
        daemon, "_resolve_python_lock_held", lambda _spec: tmp_path / "bin" / "python"
    )
    monkeypatch.setattr(
        daemon_mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess()
    )
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: (_ for _ in ()).throw(
            daemon_mod.OvPhysXDaemonError("startup timed out")
        ),
    )

    with pytest.raises(
        daemon_mod.OvPhysXDaemonUnavailableError, match="startup timed out"
    ):
        daemon._start()


def test_ensure_running_starts_only_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon()
    starts = 0

    def fake_start() -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(daemon, "_start", fake_start)
    daemon.ensure_running()
    daemon._process = type("_Running", (), {"poll": lambda self: None})()  # type: ignore[assignment]
    daemon.ensure_running()

    assert starts == 1


def test_evaluate_and_reset_only_send_expected_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon()
    requests: list[tuple[dict[str, object], str]] = []

    def fake_send(request: dict[str, object], *, op_label: str) -> dict[str, object]:
        requests.append((request, op_label))
        return {"status": "ok"}

    monkeypatch.setattr(daemon, "_send_command", fake_send)

    assert daemon.evaluate(
        scene_usd=Path("scene.usd"),
        body_pattern="/World/*",
        duration_s=2,
        dt=0.25,
        sample_fps=12,
        initial_linear_velocity=(1, 2, 3),
        initial_angular_velocity=(4, 5, 6),
    ) == {"status": "ok"}
    assert daemon.reset_only() == {"status": "ok"}

    assert requests[0] == (
        {
            "command": "evaluate",
            "scene_usd": "scene.usd",
            "body_pattern": "/World/*",
            "duration_s": 2.0,
            "dt": 0.25,
            "sample_fps": 12,
            "initial_linear_velocity": [1, 2, 3],
            "initial_angular_velocity": [4, 5, 6],
        },
        "evaluate",
    )
    assert requests[1] == ({"command": "reset_only"}, "reset_only")


def test_shutdown_locked_handles_not_running_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = type("_Exited", (), {"poll": lambda self: 0})()  # type: ignore[assignment]
    daemon._shutdown_locked()
    assert daemon._process is None

    class _TimeoutProcess:
        stdin = io.StringIO()
        killed = False

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            raise daemon_mod.subprocess.TimeoutExpired("cmd", timeout)

        def kill(self) -> None:
            self.killed = True

    process = _TimeoutProcess()
    daemon._process = process  # type: ignore[assignment]
    daemon._shutdown_locked()

    assert process.killed is True
    assert daemon._process is None


def test_shutdown_locked_sends_shutdown_and_ignores_broken_pipe() -> None:
    class _RunningProcess:
        def __init__(self, stdin: object) -> None:
            self.stdin = stdin
            self.waited = False

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            self.waited = True
            return 0

    process = _RunningProcess(io.StringIO())
    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = process  # type: ignore[assignment]
    daemon.shutdown()

    assert process.stdin.getvalue() == '{"command": "shutdown"}\n'
    assert process.waited is True

    class _BrokenStdin:
        def write(self, _value: str) -> None:
            raise BrokenPipeError

    process = _RunningProcess(_BrokenStdin())
    daemon._process = process  # type: ignore[assignment]
    daemon.shutdown()
    assert process.waited is True


def test_send_command_success_error_and_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        stdin = io.StringIO()

        def poll(self) -> None:
            return None

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _Process()  # type: ignore[assignment]
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "ok", "value": 3}\n',
    )
    assert daemon._send_command({"command": "ping"}, op_label="ping")["value"] == 3

    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "error", "error": "bad"}\n',
    )
    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="ping error: bad"):
        daemon._send_command({"command": "ping"}, op_label="ping")

    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "strange"}\n',
    )
    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="unexpected response"):
        daemon._send_command({"command": "ping"}, op_label="ping")


def test_send_command_restarts_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Process:
        stdin = io.StringIO()

        def poll(self) -> None:
            return None

    daemon = daemon_mod._OvPhysXDaemon()

    def fake_start() -> None:
        daemon._process = _Process()  # type: ignore[assignment]

    monkeypatch.setattr(daemon, "_start", fake_start)
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "ok", "value": "started"}\n',
    )

    assert (
        daemon._send_command({"command": "ping"}, op_label="ping")["value"] == "started"
    )


def test_send_command_broken_pipe_empty_response_and_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenStdin:
        def write(self, _value: str) -> None:
            raise BrokenPipeError

        def flush(self) -> None:
            raise AssertionError("flush should not be reached")

    class _Process:
        def __init__(self, stdin: object, returncode: int | None = None) -> None:
            self.stdin = stdin
            self.returncode = returncode
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            self.returncode = 0
            return 0

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._process = _Process(_BrokenStdin(), returncode=None)  # type: ignore[assignment]
    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="pipe broke"):
        daemon._send_command({"command": "ping"}, op_label="ping")

    daemon._process = _Process(io.StringIO(), returncode=None)  # type: ignore[assignment]
    monkeypatch.setattr(daemon, "_read_stdout_line", lambda timeout_s, phase: "")
    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="died during ping"):
        daemon._send_command({"command": "ping"}, op_label="ping")

    daemon._process = _Process(io.StringIO(), returncode=None)  # type: ignore[assignment]
    monkeypatch.setattr(
        daemon, "_read_stdout_line", lambda timeout_s, phase: "not json\n"
    )
    with pytest.raises(daemon_mod.OvPhysXDaemonError, match="non-JSON response"):
        daemon._send_command({"command": "ping"}, op_label="ping")


@pytest.mark.skipif(os.name == "nt", reason="POSIX selector pipe regression")
def test_read_stdout_line_pop_buffer_zero_timeout_pipe_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stdout:
        def __init__(self, fd: int | None = None) -> None:
            self._fd = fd

        def fileno(self) -> int:
            assert self._fd is not None
            return self._fd

        def readline(self) -> str:
            return "tail\n"

    class _Process:
        def __init__(self, stdout: _Stdout) -> None:
            self.stdout = stdout
            self.killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            return 0

    daemon = daemon_mod._OvPhysXDaemon()
    daemon._stdout_buffer = b"first\nsecond"
    daemon._process = _Process(_Stdout())  # type: ignore[assignment]
    assert daemon._read_stdout_line(1.0, "phase") == "first\n"
    assert daemon._read_stdout_line(0, "phase") == "secondtail\n"
    assert daemon._read_stdout_line(0, "phase") == "tail\n"

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"ready\nremaining")
        daemon._process = _Process(_Stdout(read_fd))  # type: ignore[assignment]
        assert daemon._read_stdout_line(1.0, "phase") == "ready\n"
        assert daemon._stdout_buffer == b"remaining"
    finally:
        os.close(write_fd)
        os.close(read_fd)

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        daemon._stdout_buffer = b"partial"
        daemon._process = _Process(_Stdout(read_fd))  # type: ignore[assignment]
        assert daemon._read_stdout_line(1.0, "phase") == "partial"
    finally:
        os.close(read_fd)

    read_fd, write_fd = os.pipe()
    process = _Process(_Stdout(read_fd))
    try:
        daemon._process = process  # type: ignore[assignment]
        with pytest.raises(daemon_mod.OvPhysXDaemonError, match="phase timed out"):
            daemon._read_stdout_line(0.01, "phase")
        assert process.killed is True
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.skipif(os.name == "nt", reason="POSIX selector pipe regression")
def test_read_stdout_line_times_out_when_deadline_already_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()

    class _Stdout:
        def fileno(self) -> int:
            return read_fd

    class _Process:
        stdout = _Stdout()
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            return 0

    daemon = daemon_mod._OvPhysXDaemon()
    process = _Process()
    daemon._process = process  # type: ignore[assignment]
    monotonic_values = iter([1.0, 2.0])
    monkeypatch.setattr(daemon_mod.time, "monotonic", lambda: next(monotonic_values))

    try:
        with pytest.raises(daemon_mod.OvPhysXDaemonError, match="phase timed out"):
            daemon._read_stdout_line(0.1, "phase")
        assert process.killed is True
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_atexit_shutdown_calls_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = daemon_mod._OvPhysXDaemon()
    called = False

    def fake_shutdown() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(daemon, "shutdown", fake_shutdown)

    daemon._atexit_shutdown()

    assert called is True


def test_kill_process_handles_none_exited_and_kill_errors() -> None:
    daemon = daemon_mod._OvPhysXDaemon()
    daemon._kill_process()
    assert daemon._process is None

    daemon._process = type("_Exited", (), {"poll": lambda self: 2})()  # type: ignore[assignment]
    daemon._kill_process()
    assert daemon._process is None

    class _BadKill:
        def poll(self) -> None:
            return None

        def kill(self) -> None:
            raise RuntimeError("cannot kill")

    daemon._process = _BadKill()  # type: ignore[assignment]
    daemon._stdout_buffer = b"data"
    daemon._kill_process()
    assert daemon._process is None
    assert daemon._stdout_buffer == b""
