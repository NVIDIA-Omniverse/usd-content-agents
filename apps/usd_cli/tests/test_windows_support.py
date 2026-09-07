# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Windows state-discovery primitives."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from usd_cli import daemon
from usd_core import windows_files
from usd_core.config import Config
from usd_core.render import ovrtx
from usd_core.session import Session
from usd_core.windows_files import (
    read_confined_regular_file,
    windows_process_parent_pid,
    windows_process_start_token,
    windows_process_state,
)
from usd_server import app as server_app
from usd_telemetry import main as telemetry_main

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only primitives")


def test_confined_directory_fallback_does_not_double_close_on_reopen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingFallbackApi:
        def __init__(self) -> None:
            self.absolute_opens = 0
            self.closed: list[int] = []

        def open_absolute_directory(self, _path: Path) -> int:
            self.absolute_opens += 1
            if self.absolute_opens == 2:
                raise OSError("fallback reopen failed")
            return 101

        def open_relative(
            self,
            _parent: int,
            _component: str,
            *,
            is_directory: bool,
        ) -> int:
            del is_directory
            raise PermissionError("ancestor open denied")

        def close_handle(self, handle: int) -> None:
            self.closed.append(handle)

    api = FailingFallbackApi()
    monkeypatch.setattr(windows_files, "_windows_api", lambda: api)

    with pytest.raises(OSError, match="fallback reopen failed"):
        with windows_files.open_confined_directory(tmp_path):
            raise AssertionError("fallback reopen should fail")

    assert api.closed == [101]


def test_windows_cli_help_overrides_legacy_code_page() -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1252"

    completed = subprocess.run(
        [sys.executable, "-m", "usd_cli.main", "physics", "simulate", "--help"],
        capture_output=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    help_text = completed.stdout.decode("utf-8")
    assert "Simulate an explicit scenario → time-sampled recording.usda" in help_text
    assert "local on supported Linux or" in help_text
    assert "Windows hosts" in help_text


def test_windows_fsync_path_uses_a_flushable_handle(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.usda"
    candidate.write_text("#usda 1.0\n", encoding="utf-8")

    Session._fsync_path(str(candidate))


def test_windows_ovrtx_provision_lock_uses_confined_handle(tmp_path: Path) -> None:
    lock_path = tmp_path / "ovrtx_venv.provision.lock"
    replacement = tmp_path / "replacement.lock"
    replacement.write_bytes(b"replacement")

    with ovrtx._file_lock(lock_path):
        assert lock_path.stat().st_size == 1
        with pytest.raises(OSError):
            os.replace(replacement, lock_path)

    os.replace(replacement, lock_path)


def test_windows_ovrtx_provision_lock_rejects_hardlink(tmp_path: Path) -> None:
    lock_path = tmp_path / "ovrtx_venv.provision.lock"
    lock_path.write_bytes(b"\0")
    os.link(lock_path, tmp_path / "second-link.lock")

    with pytest.raises(OSError, match="(?:single-link|exactly one link)"):
        with ovrtx._file_lock(lock_path):
            pytest.fail("hardlinked provision lock was accepted")


def test_windows_process_start_token_is_stable_for_live_process() -> None:
    first = windows_process_start_token(os.getpid())
    second = windows_process_start_token(os.getpid())

    assert first is not None
    assert first.startswith("w")
    assert second == first


def test_windows_startup_pipe_readiness_reports_bytes_eof_and_timeout() -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        started = time.monotonic()
        assert not daemon._startup_pipe_readable(read_descriptor, timeout=0.02)
        assert time.monotonic() - started < 1.0

        os.write(write_descriptor, b"1")
        assert daemon._startup_pipe_readable(read_descriptor, timeout=0.1)
        assert os.read(read_descriptor, 1) == b"1"

        os.close(write_descriptor)
        write_descriptor = -1
        assert daemon._startup_pipe_readable(read_descriptor, timeout=0.1)
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def test_windows_venv_launcher_child_identity_is_observable() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; print(os.getpid(), flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline())
    try:
        assert windows_process_parent_pid(child_pid) == process.pid
        assert windows_process_state(child_pid) == "alive"
    finally:
        os.kill(child_pid, signal.SIGTERM)
        process.wait(timeout=5)
    assert windows_process_state(child_pid) == "dead"


def test_windows_daemon_state_uses_confined_native_read(tmp_path: Path) -> None:
    state_dir = tmp_path / ".usd-cli"
    state_dir.mkdir()
    payload = b'{"host":"127.0.0.1","port":43210}\n'
    (state_dir / "server.json").write_bytes(payload)

    assert (
        read_confined_regular_file(state_dir, "server.json", max_bytes=1024)
        == payload
    )
    assert daemon._read_state(Config(project_dir=tmp_path)) == {
        "host": "127.0.0.1",
        "port": 43210,
    }


def test_windows_daemon_state_rejects_hardlinked_leaf(tmp_path: Path) -> None:
    state_dir = tmp_path / ".usd-cli"
    state_dir.mkdir()
    state_path = state_dir / "server.json"
    state_path.write_text('{"host":"127.0.0.1","port":43210}\n', encoding="utf-8")
    os.link(state_path, tmp_path / "second-link.json")

    assert daemon._read_state(Config(project_dir=tmp_path)) is None


def test_windows_drive_path_is_not_misclassified_as_a_uri(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    config = Config(
        project_dir=tmp_path,
        server={"allowed_roots": [str(tmp_path)]},
    )

    server_app._validate_request_policy(
        config,
        "open",
        {"file": str(asset)},
        shared=False,
    )


def test_windows_daemon_lifecycle_ledger_and_log_use_confined_handles(
    tmp_path: Path,
) -> None:
    config = Config(project_dir=tmp_path)

    with daemon._locked_start_lifecycle(config):
        with daemon._locked_ledger(config, create_state=True) as descriptor:
            daemon._write_ledger_locked(descriptor, [(4242, "t-owned")])
        log = daemon._open_daemon_log(config.state_dir)
        log.write(b"started\n")
        log.close()

    assert daemon._read_ledger(config) == [(4242, "t-owned")]
    assert (config.state_dir / "daemon.log").read_bytes() == b"started\n"


def test_windows_telemetry_action_lease_is_serialized(tmp_path: Path) -> None:
    lease = tmp_path / "agent-action.lock.json"

    telemetry_main._acquire_action_lease(lease, timeout_seconds=0.1)
    first = json.loads(lease.read_text(encoding="utf-8"))
    telemetry_main._acquire_action_lease(lease, timeout_seconds=0.1)

    assert first["pid"] == os.getppid()
    assert isinstance(first["start_token"], str)
    assert json.loads(lease.read_text(encoding="utf-8")) == first
