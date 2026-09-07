# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ovphysx platform gate — these run everywhere (no GPU needed).

The solver-backed black-box tests (test_physics_runtime.py) can only run on a
supported host, so the *unsupported*-platform behavior is pinned here by monkeypatching the
gate. Imports usd_core directly (run with `PYTHONPATH=src`), like test_materials.py.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest
from usd_core import physics_runtime


def test_default_runtime_lock_is_shipped_with_usd_cli(monkeypatch) -> None:
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")

    selected = physics_runtime._ovphysx_runtime_lock()

    assert selected == physics_runtime._PACKAGED_RUNTIME_LOCKS[
        ("linux", "amd64", sys.version_info[:2])
    ]
    assert selected.is_file()


@pytest.mark.parametrize(
    ("machine", "expected_name"),
    [
        ("AMD64", "pylock.ovphysx-runtime.py311.toml"),
        ("aarch64", "pylock.ovphysx-runtime.py311.aarch64.toml"),
    ],
)
def test_python311_selects_its_shipped_runtime_lock(
    monkeypatch, machine: str, expected_name: str
) -> None:
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(physics_runtime.sys, "version_info", (3, 11, 0))

    selected = physics_runtime._ovphysx_runtime_lock()

    assert selected.name == expected_name
    assert selected.is_file()


def test_existing_daemon_venv_selects_lock_for_its_python_minor(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(physics_runtime.sys, "version_info", (3, 12, 0))
    (tmp_path / "pyvenv.cfg").write_text(
        "version_info = 3.11.16.final.0\n", encoding="utf-8"
    )

    selected = physics_runtime._ovphysx_runtime_lock(tmp_path)

    assert selected.name == "pylock.ovphysx-runtime.py311.toml"


def test_daemon_holds_runtime_lock_through_ready_handshake(
    tmp_path: Path, monkeypatch
) -> None:
    lock_held = False

    @contextmanager
    def fake_provision_lock(_venv_dir: Path) -> Iterator[None]:
        nonlocal lock_held
        assert lock_held is False
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    class _FakeProcess:
        stdin = None
        stdout: list[str] = []

        def poll(self) -> None:
            return None

    monkeypatch.setattr(
        physics_runtime, "ovphysx_provision_lock", fake_provision_lock
    )
    monkeypatch.setattr(
        physics_runtime,
        "_ovphysx_python_lock_held",
        lambda _venv_dir: str(tmp_path / "bin/python"),
    )
    monkeypatch.setattr(
        physics_runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: _FakeProcess(),
    )

    def ready_status(_self, _timeout_s):
        assert lock_held is True
        return {"status": "ready"}

    monkeypatch.setattr(physics_runtime._OvPhysXDaemon, "_read_status", ready_status)

    daemon = physics_runtime._OvPhysXDaemon(venv_dir=tmp_path)

    assert daemon.alive is True
    assert lock_held is False


def test_daemon_uses_prevalidated_read_only_runtime_when_lock_is_denied(
    tmp_path: Path, monkeypatch
) -> None:
    @contextmanager
    def denied_lock(_venv_dir: Path) -> Iterator[None]:
        raise PermissionError(13, "root-owned image runtime")
        yield  # pragma: no cover

    class _FakeProcess:
        stdin = None
        stdout: list[str] = []

        def poll(self) -> None:
            return None

    bound_python = str(tmp_path / "bin/python")
    monkeypatch.setattr(physics_runtime, "ovphysx_provision_lock", denied_lock)
    monkeypatch.setattr(physics_runtime.os, "access", lambda *_args: False)
    monkeypatch.setattr(
        physics_runtime, "_ovphysx_ready_python", lambda _venv_dir: bound_python
    )
    monkeypatch.setattr(
        physics_runtime,
        "_ovphysx_python_lock_held",
        lambda _venv_dir: pytest.fail("read-only fallback must not provision"),
    )
    monkeypatch.setattr(
        physics_runtime.subprocess,
        "Popen",
        lambda command, **_kwargs: (
            _FakeProcess()
            if command[0] == bound_python
            else pytest.fail("daemon must use marker-bound Python")
        ),
    )
    monkeypatch.setattr(
        physics_runtime._OvPhysXDaemon,
        "_read_status",
        lambda *_args: {"status": "ready"},
    )

    daemon = physics_runtime._OvPhysXDaemon(venv_dir=tmp_path)

    assert daemon.alive is True


@pytest.mark.skipif(
    os.name not in {"nt", "posix"},
    reason="native cross-process file-lock regression",
)
def test_ovphysx_provision_lock_serializes_processes(tmp_path: Path) -> None:
    from usd_core.render.ovrtx import _file_lock

    lock_path = tmp_path / "ovphysx.provision.lock"
    started_path = tmp_path / "contender.started"
    entered_path = tmp_path / "contender.entered"
    contender = (
        "import sys\n"
        "from pathlib import Path\n"
        "from usd_core.render.ovrtx import _file_lock\n"
        "Path(sys.argv[2]).write_text('started', encoding='utf-8')\n"
        "with _file_lock(Path(sys.argv[1])):\n"
        "    Path(sys.argv[3]).write_text('entered', encoding='utf-8')\n"
    )

    with _file_lock(lock_path):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                contender,
                str(lock_path),
                str(started_path),
                str(entered_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not started_path.exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert started_path.exists(), process.communicate(timeout=1)[1]
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.25)
            assert not entered_path.exists()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
            raise

    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert entered_path.is_file()


def test_windows_x86_64_platform_uses_local_runtime(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(physics_runtime.sys, "version_info", (3, 12, 0))

    assert physics_runtime.ovphysx_platform_supported() is True
    selected = physics_runtime._ovphysx_runtime_lock()
    assert selected == physics_runtime._PACKAGED_RUNTIME_LOCKS[
        ("windows", "amd64", (3, 12))
    ]
    assert selected.is_file()


def test_wsl2_dxg_platform_routes_to_local_runtime(monkeypatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        physics_runtime.Path,
        "exists",
        lambda path: path.as_posix() == "/dev/dxg",
    )
    monkeypatch.setattr(glob, "glob", lambda _pattern: [])

    assert physics_runtime.ovphysx_platform_supported() is True


def test_linux_without_nvidia_device_routes_away_from_local_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(physics_runtime.Path, "exists", lambda _path: False)
    monkeypatch.setattr(glob, "glob", lambda _pattern: [])

    assert physics_runtime.ovphysx_platform_supported() is False


def test_remote_physics_backend_failure_redacts_url_credentials(monkeypatch):
    from usd_core import remote_protocol

    credential_url = (
        "https://user:password@ovrtx.example.test:8443/physics/signed-token"
        "?access_token=query-secret#fragment-secret"
    )

    def fail_protocol_check(url: str, **_kwargs: object) -> None:
        normalized_variant = url.replace("/physics/", "/physics//")
        raise RuntimeError(f"cannot reach '{normalized_variant}/live'")

    monkeypatch.setattr(remote_protocol, "check_remote_protocol", fail_protocol_check)

    with pytest.raises(RuntimeError) as raised:
        physics_runtime.resolve_remote_physics_backend(
            {"backends": [{"url": credential_url, "api_key": "header-secret"}]},
            verify=True,
        )

    message = str(raised.value)
    assert "https://ovrtx.example.test:8443" in message
    for secret in (
        "user",
        "password",
        "signed-token",
        "query-secret",
        "fragment-secret",
        "header-secret",
    ):
        assert secret not in message


def test_windows_x86_64_python311_routes_away_from_local_runtime(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(physics_runtime.sys, "version_info", (3, 11, 9))

    assert physics_runtime.ovphysx_platform_supported() is False


def test_windows_arm64_platform_is_not_locally_supported(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")

    assert physics_runtime.ovphysx_platform_supported() is False
    with pytest.raises(RuntimeError, match="No reviewed OvPhysX runtime"):
        physics_runtime._ovphysx_runtime_lock()


def test_ovphysx_limit_marker_is_explicit_and_consumed_before_native_imports(
    monkeypatch,
):
    calls: list[tuple[int, tuple[int, int]]] = []
    resource = ModuleType("resource")
    resource.RLIMIT_AS = 9
    resource.getrlimit = lambda _kind: (640, 4096)
    resource.setrlimit = lambda kind, limits: calls.append((kind, limits))
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setenv(
        physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV,
        "1",
    )

    exec(physics_runtime._DAEMON_ADDRESS_SPACE_LIMIT_PRELUDE, {})

    assert calls == [(resource.RLIMIT_AS, (4096, 4096))]
    assert (
        physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV
        not in physics_runtime.os.environ
    )
    assert physics_runtime._DAEMON_SCRIPT.index("setrlimit") < (
        physics_runtime._DAEMON_SCRIPT.index("import numpy")
    )


def test_ovphysx_limit_marker_is_not_honored_ambiently(monkeypatch):
    monkeypatch.setenv(
        physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV,
        "1",
    )
    environment = physics_runtime._ovphysx_daemon_environment(
        relax_address_space_limit=False,
    )
    assert physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV not in environment


def test_ovphysx_limit_marker_is_added_only_for_explicit_child_opt_in(monkeypatch):
    monkeypatch.delenv(
        physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV,
        raising=False,
    )
    environment = physics_runtime._ovphysx_daemon_environment(
        relax_address_space_limit=True,
    )
    assert environment[physics_runtime._OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV] == "1"


def test_local_simulation_forwards_child_only_address_space_opt_in(
    tmp_path,
    monkeypatch,
):
    from pxr import Usd, UsdGeom

    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Cube.Define(stage, "/World/Body")
    stage.GetRootLayer().Save()
    observed: list[bool] = []

    class FakeDaemon:
        def __init__(self, *, relax_address_space_limit=False):
            observed.append(relax_address_space_limit)

        def evaluate(self, **_kwargs):
            return {
                "trajectory": [
                    (0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6)
                ],
                "n_bodies": 1,
                "n_steps": 1,
            }

        def close(self):
            return None

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: True)
    monkeypatch.setattr(physics_runtime, "_OvPhysXDaemon", FakeDaemon)

    physics_runtime.simulate_scene(
        scene,
        str(tmp_path / "output"),
        body_path="/World/Body",
        rest_position=[0.0, 0.0, 0.0],
        world_up=[0.0, 0.0, 1.0],
        relax_address_space_limit=True,
    )

    assert observed == [True]


def test_unsupported_platform_raises(monkeypatch):
    # no local ovphysx and no remote backend → actionable error, no synthetic fallback
    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    with pytest.raises(RuntimeError, match="neither is available"):
        physics_runtime.simulate_scene(
            "/tmp/does_not_matter.usda",
            "/tmp/does_not_matter",
            body_path="/World/Body",
            rest_position=[0.0, 0.0, 0.0],
            world_up=[0.0, 0.0, 1.0],
            engine="ovphysx",
        )


def test_unknown_engine_raises(monkeypatch):
    # engine is validated before the platform gate, so this holds on every host.
    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: True)
    with pytest.raises(ValueError, match="only 'ovphysx' is supported"):
        physics_runtime.simulate_scene(
            "/tmp/does_not_matter.usda",
            "/tmp/does_not_matter",
            body_path="/World/Body",
            rest_position=[0.0, 0.0, 0.0],
            world_up=[0.0, 0.0, 1.0],
            engine="fake",
        )


def test_no_synthetic_fallback_exists():
    # The synthetic "fake" trajectory helper was removed — guard against it creeping back.
    assert not hasattr(physics_runtime, "_fake_trajectory")


def test_provisioning_fails_closed_without_a_pinned_runtime_lock(
    tmp_path, monkeypatch
):
    # Auto-provisioning must never resolve unpinned packages: with no lock
    # discoverable and no override, it fails closed before creating a venv.
    monkeypatch.delenv("WU_OVPHYSX_RUNTIME_LOCK", raising=False)
    monkeypatch.setattr(
        physics_runtime, "_WU_RUNTIME_LOCK_DEFAULT", tmp_path / "absent.toml"
    )
    monkeypatch.setattr(physics_runtime, "_WU_RUNTIME_LOCK_RELATIVE", {})
    monkeypatch.setattr(physics_runtime, "_PACKAGED_RUNTIME_LOCKS", {})
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="hash-pinned ovphysx runtime lock"):
        physics_runtime._ovphysx_python(tmp_path / "venv")
    assert not (tmp_path / "venv").exists()


def test_provisioning_rejects_an_unreadable_lock_override(tmp_path, monkeypatch):
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(tmp_path / "missing.toml"))
    with pytest.raises(RuntimeError, match="does not point to a readable"):
        physics_runtime._ovphysx_python(tmp_path / "venv")
    assert not (tmp_path / "venv").exists()


def test_uv_command_uses_parent_pinned_executable(tmp_path, monkeypatch):
    uv_executable = tmp_path / "uv"
    uv_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_executable.chmod(0o700)
    monkeypatch.setenv("USD_CLI_UV_EXECUTABLE", str(uv_executable))

    assert physics_runtime._uv_command() == [str(uv_executable.resolve())]


def test_uv_command_rejects_invalid_parent_pin(tmp_path, monkeypatch):
    missing = tmp_path / "missing-uv"
    monkeypatch.setenv("USD_CLI_UV_EXECUTABLE", str(missing))

    with pytest.raises(RuntimeError, match="USD_CLI_UV_EXECUTABLE"):
        physics_runtime._uv_command()


def test_provisioning_installs_only_from_the_pinned_lock_via_uv(
    tmp_path, monkeypatch
):
    lock = tmp_path / "pylock.ovphysx-runtime.toml"
    lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(lock))
    monkeypatch.setattr(
        physics_runtime, "_uv_command", lambda: ["/opt/fake/uv"]
    )
    commands: list[list[str]] = []

    def fake_run(command, check, **kwargs):
        assert check is True
        command = [str(part) for part in command]
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            physics_runtime._ovphysx_venv_python_path(
                Path(command[-1])
            ).parent.mkdir(parents=True, exist_ok=True)
        if command[0] == str(
            physics_runtime._ovphysx_venv_python_path(tmp_path / "venv")
        ):
            assert kwargs["env"].get("PYTHONPATH") is None

        class _Completed:
            returncode = 0

        return _Completed()

    monkeypatch.setattr(physics_runtime.subprocess, "run", fake_run)
    venv_dir = tmp_path / "venv"
    python_path = physics_runtime._ovphysx_python(venv_dir)
    expected_python = physics_runtime._ovphysx_venv_python_path(venv_dir)

    assert python_path == str(expected_python)
    assert commands[0][1:] == ["-m", "venv", str(venv_dir)]
    install = commands[1]
    assert install[0] == "/opt/fake/uv"
    assert install[1:3] == ["pip", "install"]
    assert "--require-hashes" in install
    assert "--no-deps" in install
    assert "--no-config" in install
    assert "--no-sources" in install
    assert install[-2:] == ["-r", str(lock)]
    assert commands[2] == [
        str(expected_python),
        "-c",
        "from ovphysx import PhysX; physics = PhysX(device='cpu'); physics.release()",
    ]
    # Nothing may fall back to bare pip or a live index resolution.
    flat = " ".join(" ".join(command) for command in commands[:2])
    assert "-m pip" not in flat
    assert "--extra-index-url" not in flat
    assert " ovphysx " not in f" {flat} "
    marker = json.loads(
        (venv_dir / ".usd-cli-ovphysx-ready").read_text(encoding="utf-8")
    )
    assert marker == {
        "schema_version": "usd-cli.ovphysx-runtime-ready.v2",
        "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python_path": str(expected_python),
    }


def test_provisioning_probe_failure_does_not_publish_ready_marker(
    tmp_path, monkeypatch
):
    runtime_lock = tmp_path / "pylock.ovphysx-runtime.toml"
    runtime_lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(runtime_lock))
    monkeypatch.setattr(physics_runtime, "_uv_command", lambda: ["/opt/fake/uv"])
    venv_dir = tmp_path / "venv"
    python_path = physics_runtime._ovphysx_venv_python_path(venv_dir)

    def fake_run(command, check, **_kwargs):
        assert check is True
        if command[1:3] == ["-m", "venv"]:
            python_path.parent.mkdir(parents=True, exist_ok=True)
        if command[0] == str(python_path):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(physics_runtime.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        physics_runtime._ovphysx_python(venv_dir)

    assert (venv_dir / ".usd-cli-ovphysx-ready").exists() is False


def test_marker_bound_runtime_with_failed_probe_is_repaired(tmp_path, monkeypatch):
    runtime_lock = tmp_path / "pylock.ovphysx-runtime.toml"
    runtime_lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(runtime_lock))
    monkeypatch.setattr(physics_runtime, "_uv_command", lambda: ["/opt/fake/uv"])
    venv_dir = tmp_path / "venv"
    python_path = physics_runtime._ovphysx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    marker = venv_dir / ".usd-cli-ovphysx-ready"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "usd-cli.ovphysx-runtime-ready.v2",
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        physics_runtime, "_ovphysx_runtime_probe", lambda _python: False
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([str(part) for part in command])

    monkeypatch.setattr(physics_runtime.subprocess, "run", fake_run)

    assert physics_runtime._ovphysx_python(venv_dir) == str(python_path)
    assert commands[0][0:3] == ["/opt/fake/uv", "pip", "install"]
    assert commands[1] == [
        str(python_path),
        "-c",
        physics_runtime._OVPHYSX_RUNTIME_PROBE,
    ]
    assert marker.is_file()


def test_legacy_preprobe_marker_is_not_ready(tmp_path, monkeypatch):
    runtime_lock = tmp_path / "pylock.ovphysx-runtime.toml"
    runtime_lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(runtime_lock))
    monkeypatch.setenv("WU_OVPHYSX_AUTO_PROVISION", "0")
    venv_dir = tmp_path / "venv"
    python_path = physics_runtime._ovphysx_venv_python_path(venv_dir)
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    (venv_dir / ".usd-cli-ovphysx-ready").write_text(
        json.dumps(
            {
                "schema_version": "usd-cli.ovphysx-runtime-ready.v1",
                "runtime_lock_sha256": hashlib.sha256(
                    runtime_lock.read_bytes()
                ).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="auto-provision disabled"):
        physics_runtime._ovphysx_python(venv_dir)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink canonicalization")
def test_ready_marker_matches_canonicalized_venv_path(tmp_path, monkeypatch):
    lock = tmp_path / "pylock.ovphysx-runtime.toml"
    lock.write_text('lock-version = "1.0"\n', encoding="utf-8")
    monkeypatch.setenv("WU_OVPHYSX_RUNTIME_LOCK", str(lock))
    monkeypatch.setenv("WU_OVPHYSX_AUTO_PROVISION", "0")
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_venv = alias_parent / "venv"
    canonical_venv = aliased_venv.resolve()
    python_path = physics_runtime._ovphysx_venv_python_path(canonical_venv)
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    marker = canonical_venv / ".usd-cli-ovphysx-ready"
    marker.write_text(
        json.dumps(
            {
                "schema_version": "usd-cli.ovphysx-runtime-ready.v2",
                "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "python_path": str(python_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        physics_runtime, "_ovphysx_runtime_probe", lambda _python: True
    )

    assert physics_runtime._ovphysx_python(aliased_venv) == str(python_path)
