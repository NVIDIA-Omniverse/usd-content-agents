# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WU integration regressions for fail-closed usd-cli daemon ownership."""

from __future__ import annotations

import errno
import json
import os
import signal
import stat
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
import typer
from fastapi.testclient import TestClient
from usd_cli import daemon
from usd_core.config import Config
from usd_server import app as server_app


class _FakeSpawnedProcess:
    pid = 4242

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return 0 if self.terminated or self.killed else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def _open_test_directory(path: Path) -> int:
    if os.name != "nt":
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    from usd_core.windows_files import open_confined_directory

    with open_confined_directory(path) as descriptor:
        return os.dup(descriptor)


def _use_same_process_startup_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        monkeypatch.setattr(
            daemon,
            "_startup_descriptor_from_environment",
            lambda name, *, writable: int(os.environ[name]),
        )


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise


def test_record_spawn_identity_is_atomic_and_durable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    live_tokens = {123: "t-first", 456: "t-second"}
    monkeypatch.setattr(daemon, "_proc_start_token", live_tokens.get)
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")

    daemon._record_spawn_identity(config, 123, "t-first")
    first = config.state_dir / "daemon.pids"
    first_inode = first.stat().st_ino
    daemon._record_spawn_identity(config, 456, "t-second")

    metadata = first.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert metadata.st_ino != first_inode
    assert first.read_text(encoding="utf-8") == "123:t-first\n456:t-second\n"
    assert not list(config.state_dir.glob(".daemon.pids.*.tmp"))


def _run_child_handshake(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str | None = "t-owned",
) -> tuple[int, int, threading.Thread, list[tuple[int, str]], list[BaseException]]:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    project_descriptor = _open_test_directory(config.project_dir)
    state_descriptor = _open_test_directory(config.state_dir)
    release_read, release_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    monkeypatch.setenv(daemon.STARTUP_RELEASE_FD_ENV, str(release_read))
    monkeypatch.setenv(daemon.STARTUP_ACK_FD_ENV, str(acknowledgement_write))
    monkeypatch.setenv(daemon.STARTUP_NONCE_ENV, "nonce")
    monkeypatch.setenv(daemon.STARTUP_PROJECT_FD_ENV, str(project_descriptor))
    monkeypatch.setenv(daemon.STARTUP_STATE_FD_ENV, str(state_descriptor))
    monkeypatch.setenv(daemon.STARTUP_PROJECT_PATH_ENV, str(config.project_dir.resolve()))
    monkeypatch.setenv(
        daemon.STARTUP_STATE_PATH_ENV,
        os.path.abspath(config.state_dir),
    )
    monkeypatch.setattr(daemon.os, "getpid", lambda: 4242)
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: token)
    _use_same_process_startup_descriptors(monkeypatch)
    recorded: list[tuple[int, str]] = []
    monkeypatch.setattr(
        daemon,
        "_record_spawn_identity",
        lambda _config, pid, birth_token: recorded.append((pid, birth_token)),
    )
    errors: list[BaseException] = []

    def target() -> None:
        try:
            identity = daemon.child_startup_handshake(config)
            os.close(identity.state_descriptor)
            os.close(identity.project_descriptor)
        except BaseException as exc:  # noqa: BLE001 - test captures thread failure
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    return release_write, acknowledgement_read, thread, recorded, errors


def test_child_reports_identity_before_parent_releases_serving(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    release, acknowledgement, thread, recorded, errors = _run_child_handshake(
        config,
        monkeypatch,
    )
    try:
        payload = os.read(acknowledgement, daemon.MAX_DAEMON_REGISTRATION_BYTES)
        registration = json.loads(payload)
        assert registration == {
            "pid": 4242,
            "process_start_token": "t-owned",
            "nonce": "nonce",
        }
        assert recorded == []  # the stable-lock-holding parent records it
        assert thread.is_alive(), "child must remain blocked before parent release"

        os.write(release, b"1")
        thread.join(timeout=2)
    finally:
        os.close(acknowledgement)
        os.close(release)

    assert not thread.is_alive()
    assert not errors


def test_child_registration_gate_exits_on_parent_kill_eof(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    release, acknowledgement, thread, recorded, errors = _run_child_handshake(
        config,
        monkeypatch,
    )
    try:
        assert os.read(acknowledgement, daemon.MAX_DAEMON_REGISTRATION_BYTES)
        assert recorded == []
        assert thread.is_alive()
        # SIGKILL closes the starter's pipe descriptors in the kernel. EOF is the
        # relevant invariant: the child exits without crossing the serve gate.
        os.close(release)
        release = -1
        thread.join(timeout=2)
    finally:
        os.close(acknowledgement)
        if release >= 0:
            os.close(release)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert "before releasing" in str(errors[0])


def test_child_rejects_state_directory_replacement_before_server_import(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    release, acknowledgement, thread, _recorded, errors = _run_child_handshake(
        config,
        monkeypatch,
    )
    try:
        assert os.read(acknowledgement, daemon.MAX_DAEMON_REGISTRATION_BYTES)
        original_state = tmp_path / ".usd-cli.original"
        if os.name == "nt":
            with pytest.raises(OSError):
                config.state_dir.rename(original_state)
        else:
            config.state_dir.rename(original_state)
            (tmp_path / ".usd-cli").mkdir()
        os.write(release, b"1")
        thread.join(timeout=2)
    finally:
        os.close(acknowledgement)
        os.close(release)

    assert not thread.is_alive()
    if os.name == "nt":
        assert not errors
    else:
        assert len(errors) == 1
        assert "state-directory identity changed" in str(errors[0])


@pytest.mark.parametrize("failure_kind", ["birth-token", "ack-write"])
def test_child_registration_failure_closes_parent_acknowledgement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    project_descriptor = _open_test_directory(config.project_dir)
    state_descriptor = _open_test_directory(config.state_dir)
    release_read, release_write = os.pipe()
    acknowledgement_read, acknowledgement_write = os.pipe()
    monkeypatch.setenv(daemon.STARTUP_RELEASE_FD_ENV, str(release_read))
    monkeypatch.setenv(daemon.STARTUP_ACK_FD_ENV, str(acknowledgement_write))
    monkeypatch.setenv(daemon.STARTUP_NONCE_ENV, "nonce")
    monkeypatch.setenv(daemon.STARTUP_PROJECT_FD_ENV, str(project_descriptor))
    monkeypatch.setenv(daemon.STARTUP_STATE_FD_ENV, str(state_descriptor))
    monkeypatch.setenv(daemon.STARTUP_PROJECT_PATH_ENV, str(config.project_dir.resolve()))
    monkeypatch.setenv(
        daemon.STARTUP_STATE_PATH_ENV,
        os.path.abspath(config.state_dir),
    )
    monkeypatch.setattr(daemon.os, "getpid", lambda: 4242)
    _use_same_process_startup_descriptors(monkeypatch)
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda _pid: None if failure_kind == "birth-token" else "t-owned",
    )
    if failure_kind == "ack-write":
        monkeypatch.setattr(
            daemon,
            "_write_all",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("pipe failed")),
        )

    with pytest.raises((OSError, RuntimeError)):
        daemon.child_startup_handshake(config)

    assert os.read(acknowledgement_read, 1) == b""
    broken_pipe_error = OSError if os.name == "nt" else BrokenPipeError
    with pytest.raises(broken_pipe_error):
        os.write(release_write, b"1")
    os.close(acknowledgement_read)
    os.close(release_write)

def test_windows_breakaway_denial_accepts_native_access_denied() -> None:
    error = OSError(errno.EACCES, "access denied")
    error.winerror = 5

    assert daemon._is_windows_breakaway_denied(error)


def test_windows_daemon_spawn_fails_closed_when_breakaway_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    def fake_popen(_command, *, creationflags: int, **_kwargs):  # noqa: ANN001
        calls.append(creationflags)
        error = OSError(errno.EACCES, "access denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon.subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000, raising=False
    )
    flags = 0x00000200 | daemon.subprocess.CREATE_BREAKAWAY_FROM_JOB
    with pytest.raises(RuntimeError, match="denied job breakaway"):
        daemon._popen_windows_daemon(["daemon"], creationflags=flags)
    assert calls == [flags]


@pytest.mark.parametrize("registration_ok", [True, False])
@pytest.mark.parametrize(
    ("configured_host", "allocation_host"),
    [("localhost", "127.0.0.1"), ("::1", "::1")],
)
def test_spawn_releases_only_after_authenticated_child_registration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    registration_ok: bool,
    configured_host: str,
    allocation_host: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.server["host"] = configured_host
    config.render.update(
        {
            "remote_api_key": "primary-secret",
            "backends": [
                {"url": "https://a.example.test", "api_key": "pool-secret"}
            ],
        }
    )
    monkeypatch.setenv("USD_CLI_RENDER_REMOTE_API_KEY", "environment-secret")
    monkeypatch.setenv(
        "USD_CLI_RENDER_BACKEND_API_KEYS_JSON",
        '{"https://a.example.test":"environment-pool-secret"}',
    )
    process = _FakeSpawnedProcess()
    events: list[str] = []
    popen_argv: list[list[str]] = []
    popen_kwargs: list[dict] = []
    allocation_hosts: list[str] = []
    credential_payloads: list[dict[str, object]] = []

    def fake_popen(argv, **kwargs):  # noqa: ANN001,ANN003,ANN202
        popen_argv.append(argv)
        popen_kwargs.append(kwargs)
        return process

    def fake_await(*_args, **_kwargs) -> str:  # noqa: ANN002,ANN003
        events.append("durable-ack")
        if not registration_ok:
            raise RuntimeError("registration failed")
        return "t-owned"

    def recording_write(descriptor: int, payload: bytes) -> None:
        if payload == b"1":
            events.append("release")
            return
        credential_payloads.append(json.loads(payload))

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon,
        "_free_port",
        lambda host: allocation_hosts.append(host) or 4567,
    )
    monkeypatch.setattr(daemon, "_await_child_registration", fake_await)
    monkeypatch.setattr(
        daemon,
        "_record_spawn_identity",
        lambda _config, pid, token: events.append(f"durable-record:{pid}:{token}"),
    )
    monkeypatch.setattr(
        daemon,
        "_read_ledger",
        lambda _config: [(process.pid, "t-owned")],
    )
    monkeypatch.setattr(daemon, "_write_all", recording_write)
    monkeypatch.setattr(daemon, "_reap_orphans", lambda *_args, **_kwargs: 0)

    with daemon._locked_start_lifecycle(config):
        if registration_ok:
            assert daemon._spawn(config) == (process.pid, "t-owned")
            assert events == ["durable-ack", "durable-record:4242:t-owned", "release"]
            assert process.terminated is False
        else:
            with pytest.raises(RuntimeError, match="child registration"):
                daemon._spawn(config)
            assert events == ["durable-ack"]
            assert process.terminated is True
    assert len(popen_kwargs) == 1
    assert allocation_hosts == [allocation_host]
    assert popen_kwargs[0]["env"]["USD_CLI_SERVER_HOST"] == allocation_host
    assert popen_kwargs[0]["env"]["USD_CLI_SERVER_PORT"] == "4567"
    if os.name == "nt":
        assert "pass_fds" not in popen_kwargs[0]
        assert "start_new_session" not in popen_kwargs[0]
        assert popen_kwargs[0]["creationflags"] == (
            daemon.subprocess.DETACHED_PROCESS
            | daemon.subprocess.CREATE_NEW_PROCESS_GROUP
            | daemon.subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
        startupinfo = popen_kwargs[0]["startupinfo"]
        assert len(startupinfo.lpAttributeList["handle_list"]) == 5
    else:
        assert len(popen_kwargs[0]["pass_fds"]) == 5
        assert popen_kwargs[0]["start_new_session"] is True
        assert "creationflags" not in popen_kwargs[0]
    assert "USD_CLI_RENDER_REMOTE_API_KEY" not in popen_kwargs[0]["env"]
    assert "USD_CLI_RENDER_BACKEND_API_KEYS_JSON" not in popen_kwargs[0]["env"]
    assert credential_payloads == [
        {
            "remote_api_key": "primary-secret",
            "backend_api_keys": {"https://a.example.test": "pool-secret"},
        }
    ]
    assert popen_kwargs[0]["env"][daemon.STARTUP_PROJECT_PATH_ENV] == str(
        config.project_dir.resolve()
    )
    assert popen_kwargs[0]["env"][daemon.STARTUP_STATE_PATH_ENV] == os.path.abspath(
        config.state_dir
    )
    bootstrap = popen_argv[0][2]
    assert bootstrap.index("child_startup_handshake") < bootstrap.index(
        "from usd_server.app import serve"
    )


def test_daemon_child_reads_render_credentials_only_from_startup_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Config(project_dir=tmp_path)
    source.render.update(
        {
            "remote_api_key": "primary-secret",
            "backends": [
                {"url": "https://a.example.test/", "api_key": "pool-secret"}
            ],
        }
    )
    target = Config(project_dir=tmp_path)
    target.render["backends"] = [{"url": "https://a.example.test/"}]
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, daemon._startup_render_credentials_payload(source))
    os.close(write_descriptor)
    monkeypatch.setenv(
        daemon.STARTUP_RENDER_CREDENTIALS_FD_ENV,
        str(read_descriptor),
    )
    if os.name == "nt":
        monkeypatch.setattr(
            daemon,
            "_startup_descriptor_from_value",
            lambda raw, *, writable: int(raw),
        )

    daemon.apply_startup_render_credentials(target)

    assert target.render["remote_api_key"] == "primary-secret"
    assert target.render["backends"] == [
        {"url": "https://a.example.test/", "api_key": "pool-secret"}
    ]
    assert daemon.STARTUP_RENDER_CREDENTIALS_FD_ENV not in os.environ


def test_spawn_preserves_record_when_exact_child_cannot_be_confirmed_gone(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    process = _FakeSpawnedProcess()
    discarded: list[tuple[int, str]] = []
    monkeypatch.setattr(daemon.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(daemon, "_free_port", lambda _host: 4567)
    monkeypatch.setattr(
        daemon,
        "_await_child_registration",
        lambda *_args, **_kwargs: "t-owned",
    )
    monkeypatch.setattr(daemon, "_record_spawn_identity", lambda *_args: None)
    monkeypatch.setattr(
        daemon,
        "_read_ledger",
        lambda _config: [(process.pid, "t-owned")],
    )
    monkeypatch.setattr(
        daemon,
        "_write_all",
        lambda *_args: (_ for _ in ()).throw(OSError("release failed")),
    )
    monkeypatch.setattr(
        daemon,
        "_terminate_unrecorded_spawn",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        daemon,
        "_discard_spawn_identity",
        lambda _config, identity: discarded.append(identity),
    )

    with daemon._locked_start_lifecycle(config):
        with pytest.raises(RuntimeError, match="child registration"):
            daemon._spawn(config)

    assert discarded == []


def test_ensure_running_reaps_new_spawn_after_readiness_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    cleaned: list[tuple[int, str]] = []
    monkeypatch.setattr(daemon, "discover", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon, "_spawn", lambda _config: (4242, "t-owned"))
    monkeypatch.setattr(
        daemon,
        "_cleanup_failed_spawn",
        lambda _config, identity: cleaned.append(identity),
    )
    monkeypatch.setattr(daemon, "START_TIMEOUT_S", 0.0)

    with pytest.raises(RuntimeError, match="did not come up"):
        daemon.ensure_running(config)

    assert cleaned == [(4242, "t-owned")]


def test_ensure_running_reaps_new_spawn_when_lifecycle_exit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    discoveries = iter((None, None, ("http://127.0.0.1:4567", "secret")))
    cleaned: list[tuple[int, str]] = []

    @contextmanager
    def failing_lifecycle(_config):
        yield
        raise RuntimeError("state-directory identity changed on exit")

    monkeypatch.setattr(
        daemon, "discover", lambda *_args, **_kwargs: next(discoveries)
    )
    monkeypatch.setattr(daemon, "_locked_start_lifecycle", failing_lifecycle)
    monkeypatch.setattr(daemon, "_spawn", lambda _config: (4242, "t-owned"))
    monkeypatch.setattr(
        daemon,
        "_cleanup_failed_spawn",
        lambda _config, identity: cleaned.append(identity),
    )

    with pytest.raises(RuntimeError, match="identity changed on exit"):
        daemon.ensure_running(config)

    assert cleaned == [(4242, "t-owned")]


def test_failed_spawn_cleanup_preserves_evidence_when_identity_is_unverifiable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    ledger.write_text("4242:t-owned\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "_terminate", lambda *_args, **_kwargs: False)

    assert not daemon._cleanup_failed_spawn(config, (4242, "t-owned"))
    assert json.loads(config.server_state_path.read_text()) == state
    assert ledger.read_text() == "4242:t-owned\n"


def test_terminate_preserves_unverifiable_live_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: None)
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon.os,
        "kill",
        lambda _pid, sig: signaled.append(sig),
    )

    assert not daemon._terminate(4242, expected_token="t-owned")
    assert not signaled


def test_terminate_escalation_uses_platform_safe_force_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[int] = []
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(
        daemon,
        "_proc_state",
        lambda _pid: "dead" if len(signaled) >= 2 else "alive",
    )
    monkeypatch.setattr(
        daemon.os,
        "kill",
        lambda _pid, sent_signal: signaled.append(sent_signal),
    )

    assert daemon._terminate(
        4242,
        expected_token="t-owned",
        term_wait_s=0,
        kill_wait_s=0,
    )
    force_signal = signal.SIGTERM if os.name == "nt" else signal.SIGKILL
    assert signaled == [signal.SIGTERM, force_signal]


def test_permission_denied_cleanup_preserves_evidence_and_never_escalates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX signal-permission semantics")
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    ledger.write_text("4242:t-owned\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    signals: list[int] = []

    def deny_signal(_pid: int, sig: int) -> None:
        signals.append(sig)
        raise PermissionError("not signalable")

    monkeypatch.setattr(daemon.os, "kill", deny_signal)

    assert daemon._alive(4242)
    assert not daemon._cleanup_failed_spawn(config, (4242, "t-owned"))
    assert signals == [0, daemon.signal.SIGTERM]
    assert json.loads(config.server_state_path.read_text()) == state
    assert ledger.read_text(encoding="utf-8") == "4242:t-owned\n"


def test_process_identity_fallback_never_executes_caller_path_ps(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "hostile-ps-executed"
    hostile_ps = fake_bin / "ps"
    hostile_ps.write_text(
        f"#!/bin/sh\nprintf hostile > '{marker}'\nexit 0\n",
        encoding="utf-8",
    )
    hostile_ps.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    original_read_text = Path.read_text

    def force_ps_fallback(path: Path, *args, **kwargs):  # noqa: ANN002,ANN003,ANN202
        if str(path).startswith("/proc/"):
            raise OSError("force trusted ps fallback")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", force_ps_fallback)

    daemon._proc_start_token(os.getpid())
    daemon._proc_state(os.getpid())

    assert not marker.exists()


def test_process_start_token_rejects_failed_ps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no procfs")),
    )
    monkeypatch.setattr(daemon, "_system_ps_path", lambda: "/bin/ps")
    monkeypatch.setattr(
        daemon.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="Mon Aug  3 14:06:06 2026",
        ),
    )

    assert daemon._proc_start_token(4242) is None


def test_process_state_never_reaps_a_pid_from_mutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX process-state fallback semantics")
    monkeypatch.setattr(
        daemon.os,
        "waitpid",
        lambda *_args: pytest.fail("generic process inspection must not reap children"),
    )
    monkeypatch.setattr(daemon, "_alive", lambda _pid: True)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: "4242 (x) R")

    assert daemon._proc_state(4242) == "alive"


def test_reap_orphans_retains_live_identity_when_token_lookup_is_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    ledger = config.state_dir / "daemon.pids"
    ledger.write_text("4242:t-owned\n", encoding="utf-8")
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: None)
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")

    assert daemon._reap_orphans(config, keep_pid=None) == 0
    assert ledger.read_text(encoding="utf-8") == "4242:t-owned\n"


@pytest.mark.parametrize(
    "ledger_kind",
    ["symlink", "hardlink", "fifo", "oversize", "malformed", "concurrent-replace"],
)
def test_unsafe_daemon_ledgers_signal_nobody_and_are_not_rewritten(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_kind: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    external = tmp_path / "external-ledger"
    if ledger_kind == "symlink":
        external.write_text("4242:t-owned\n", encoding="utf-8")
        _symlink_or_skip(ledger, external)
    elif ledger_kind == "hardlink":
        external.write_text("4242:t-owned\n", encoding="utf-8")
        os.link(external, ledger)
    elif ledger_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO fixtures are unavailable on Windows")
        os.mkfifo(ledger)
    elif ledger_kind == "oversize":
        ledger.write_bytes(b"x" * (daemon.MAX_DAEMON_LEDGER_BYTES + 1))
    elif ledger_kind == "malformed":
        ledger.write_text("4242\n", encoding="utf-8")
    else:
        ledger.write_text("4242:t-owned\n", encoding="utf-8")
    original_lstat = ledger.lstat()
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )
    if ledger_kind == "concurrent-replace":
        original_reader = daemon._read_ledger_locked
        calls = 0

        def replacing_reader(directory_descriptor: int) -> list[tuple[int, str]]:
            nonlocal calls
            entries = original_reader(directory_descriptor)
            calls += 1
            if calls == 1:
                replacement = config.state_dir / "replacement-ledger"
                replacement.write_text("4242:t-other\n", encoding="utf-8")
                replacement.replace(ledger)
            return entries

        monkeypatch.setattr(daemon, "_read_ledger_locked", replacing_reader)

    assert daemon._reap_orphans(config, keep_pid=None) == 0
    assert not signaled
    if ledger_kind == "symlink":
        assert ledger.is_symlink()
        assert external.read_text(encoding="utf-8") == "4242:t-owned\n"
    elif ledger_kind == "hardlink":
        assert ledger.stat().st_ino == external.stat().st_ino
        assert ledger.stat().st_nlink == 2
    elif ledger_kind == "concurrent-replace":
        assert ledger.read_text(encoding="utf-8") == "4242:t-other\n"
    else:
        assert ledger.lstat().st_ino == original_lstat.st_ino


@pytest.mark.parametrize("state_kind", ["symlink", "hardlink", "fifo", "oversize"])
def test_unsafe_server_state_is_never_read(
    tmp_path,
    state_kind: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state_path = config.server_state_path
    external = tmp_path / "external-state"
    payload = b'{"pid":4242,"token":"external"}\n'
    if state_kind == "symlink":
        external.write_bytes(payload)
        _symlink_or_skip(state_path, external)
    elif state_kind == "hardlink":
        external.write_bytes(payload)
        os.link(external, state_path)
    elif state_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO fixtures are unavailable on Windows")
        os.mkfifo(state_path)
    else:
        state_path.write_bytes(b"x" * (daemon.MAX_DAEMON_STATE_BYTES + 1))

    assert daemon._read_state(config) is None
    if external.exists():
        assert external.read_bytes() == payload


def test_non_loopback_server_state_never_receives_authentication_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    config.server_state_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "process_start_token": "t-owned",
                "host": "192.0.2.10",
                "port": 4567,
                "token": "must-not-leak",
                "project_id": daemon._project_id(config),
                "instance_id": "instance",
            }
        ),
        encoding="utf-8",
    )
    (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        "4242:t-owned\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    probes: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda base, token, **_kwargs: probes.append((base, token)) or None,
    )

    assert daemon.discover(config) is None
    assert not probes


def test_daemon_health_bypasses_proxy_environment_and_keeps_token_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    calls: list[tuple[bool, str, dict[str, str]]] = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": True, "pid": 4242}

    class FakeClient:
        def __init__(self, *, trust_env: bool) -> None:
            self.trust_env = trust_env

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args) -> None:  # noqa: ANN002
            return None

        def get(self, url: str, *, headers: dict[str, str], timeout: float):
            del timeout
            calls.append((self.trust_env, url, headers))
            return FakeResponse()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    assert daemon._health("http://127.0.0.1:4567", "project-secret") == {
        "ok": True,
        "pid": 4242,
    }
    assert calls == [
        (
            False,
            "http://127.0.0.1:4567/health",
            {
                "x-usd-cli-token": "project-secret",
                "x-ov-token": "project-secret",
                "x-3dsc-token": "project-secret",
            },
        )
    ]


@pytest.mark.parametrize("payload", [["ok"], {"ok": "yes"}, {"ok": False}])
def test_daemon_health_requires_strict_boolean_object(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    import httpx

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> object:
            return payload

    class FakeClient:
        def __init__(self, *, trust_env: bool) -> None:
            assert trust_env is False

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_args) -> None:  # noqa: ANN002
            return None

        def get(self, *_args, **_kwargs):  # noqa: ANN002,ANN003,ANN201
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    assert daemon._health("http://127.0.0.1:4567", "secret") is None


def test_server_state_publication_rejects_a_preexisting_temporary_symlink(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    external = tmp_path / "external-state-target"
    external.write_text("unchanged", encoding="utf-8")
    temporary = config.state_dir / ".server.json.fixed.tmp"
    _symlink_or_skip(temporary, external)
    monkeypatch.setattr(server_app.secrets, "token_hex", lambda _size: "fixed")

    with pytest.raises(FileExistsError):
        server_app.write_server_state(
            config,
            "127.0.0.1",
            4567,
            "secret",
            "instance",
            daemon_pid=os.getpid(),
            process_start_token="t-child",
        )

    assert temporary.is_symlink()
    assert external.read_text(encoding="utf-8") == "unchanged"
    assert not config.server_state_path.exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink"])
def test_server_state_publication_replaces_unsafe_destination_without_following_it(
    tmp_path,
    unsafe_kind: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    external = tmp_path / "external-state-target"
    external.write_text("unchanged", encoding="utf-8")
    if unsafe_kind == "symlink":
        _symlink_or_skip(config.server_state_path, external)
    else:
        os.link(external, config.server_state_path)

    state_path = server_app.write_server_state(
        config,
        "127.0.0.1",
        4567,
        "secret",
        "instance",
        daemon_pid=os.getpid(),
        process_start_token="t-child",
    )

    assert external.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_ISREG(state_path.lstat().st_mode)
    assert state_path.stat().st_nlink == 1
    if os.name != "nt":
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert not list(config.state_dir.glob(".server.json.*.tmp"))


@pytest.mark.parametrize("unsafe_name", ["daemon.log", "server.json.lock"])
def test_daemon_log_and_start_lock_reject_symlinks(
    tmp_path,
    unsafe_name: str,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    external = tmp_path / "external"
    external.write_text("unchanged", encoding="utf-8")
    _symlink_or_skip(config.state_dir / unsafe_name, external)

    with pytest.raises((OSError, RuntimeError)):
        if unsafe_name == "daemon.log":
            daemon._open_daemon_log(config.state_dir)
        else:
            with daemon._locked_start_lifecycle(config):
                pass

    assert external.read_text(encoding="utf-8") == "unchanged"


def test_conflicting_daemon_tokens_never_authorize_a_signal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    ledger.write_text("4242:t-owned\n4242:t-external\n", encoding="utf-8")
    signaled: list[tuple[int, str | None]] = []
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, *, expected_token=None, **_kwargs: (
            signaled.append((pid, expected_token)) or True
        ),
    )

    assert daemon._reap_orphans(config, keep_pid=None) == 0
    assert not signaled
    assert ledger.read_text(encoding="utf-8") == ("4242:t-owned\n4242:t-external\n")


def test_additional_ledger_pid_invalidates_current_project_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        "4242:t-owned\n5252:t-other\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda pid: {4242: "t-owned", 5252: "t-other"}.get(pid),
    )
    probes: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: probes.append("health") or {"ok": True, **state},
    )
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    assert daemon.discover(config) is None
    with pytest.raises(daemon.DaemonStopRefused, match="malformed"):
        daemon.stop(config)
    assert not probes
    assert not signaled


def test_oversized_pid_is_rejected_before_process_or_network_use(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    oversized = daemon.MAX_SAFE_PID + 1
    state = {
        "pid": oversized,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        f"{oversized}:t-owned\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon.os,
        "kill",
        lambda *_args: pytest.fail("unsafe pid must not reach kill(2)"),
    )
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: pytest.fail("unsafe pid must not reach health"),
    )

    assert daemon.discover(config) is None
    with pytest.raises(daemon.DaemonStopRefused, match="malformed"):
        daemon.stop(config)


def test_authenticated_health_without_ledger_never_authorizes_stop_signal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: {"ok": True, **state},
    )
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    with pytest.raises(daemon.DaemonStopRefused, match="malformed"):
        daemon.stop(config)
    assert not signaled


def test_restart_never_starts_replacement_when_authenticated_stop_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    import usd_core
    from usd_cli import main as cli_main

    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        "4242:t-owned\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(usd_core, "load_config", lambda: config)
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    monkeypatch.setattr(
        daemon, "_health", lambda *_args, **_kwargs: {"ok": True, **state}
    )
    monkeypatch.setattr(daemon, "_terminate", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        daemon,
        "start",
        lambda _config: pytest.fail("restart must not start a replacement"),
    )

    # The invariant is that no replacement is started (the `start` patch above
    # fails the test if one is). The CLI renders the refusal as a clean exit-1
    # error rather than letting DaemonStopRefused escape as a traceback.
    with pytest.raises(typer.Exit) as excinfo:
        cli_main.server_restart()
    assert excinfo.value.exit_code == 1
    assert "refused restart" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("external_environment", "external_config"),
    [(True, False), (False, True)],
)
def test_cli_lifecycle_guard_refuses_either_external_owner_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    external_environment: bool,
    external_config: bool,
) -> None:
    import usd_core
    from usd_cli import main as cli_main

    config = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"} if external_config else {},
    )
    monkeypatch.setattr(usd_core, "load_config", lambda: config)
    if external_environment:
        monkeypatch.setenv("USD_CLI_LIFECYCLE_EXTERNALLY_OWNED", "1")
    else:
        monkeypatch.delenv("USD_CLI_LIFECYCLE_EXTERNALLY_OWNED", raising=False)
    monkeypatch.delenv("USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP", raising=False)

    with pytest.raises(typer.Exit) as excinfo:
        cli_main._require_child_lifecycle_authority("stop")

    assert excinfo.value.exit_code == 1


def test_cli_lifecycle_guard_allows_only_one_time_start_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import usd_core
    from usd_cli import main as cli_main

    config = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"},
    )
    monkeypatch.setattr(usd_core, "load_config", lambda: config)
    monkeypatch.setenv("USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP", "1")

    cli_main._require_child_lifecycle_authority("start")
    for action in ("stop", "restart"):
        with pytest.raises(typer.Exit) as excinfo:
            cli_main._require_child_lifecycle_authority(action)
        assert excinfo.value.exit_code == 1

    config.state_dir.mkdir(parents=True)
    config.server_state_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as excinfo:
        cli_main._require_child_lifecycle_authority("start")
    assert excinfo.value.exit_code == 1


def test_record_prunes_stale_duplicate_pid_records(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    ledger.write_text(
        "4242:t-old\n4242:t-old\n9999:t-dead\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda pid: "t-new" if pid == 4242 else None,
    )
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "dead")

    daemon._record_spawn_identity(config, 4242, "t-new")

    assert ledger.read_text(encoding="utf-8") == "4242:t-new\n"


def test_record_and_reap_are_serialized_by_the_ledger_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    ledger = config.state_dir / daemon.DAEMON_LEDGER_NAME
    ledger.write_text("111:t-old\n", encoding="utf-8")
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda pid: {111: "t-old", 222: "t-new"}.get(pid),
    )
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )
    entered_record = threading.Event()
    release_record = threading.Event()
    original_normalize = daemon._normalized_entries_for_record

    def blocking_normalize(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
        entered_record.set()
        assert release_record.wait(timeout=2)
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(daemon, "_normalized_entries_for_record", blocking_normalize)
    record_thread = threading.Thread(
        target=daemon._record_spawn_identity,
        args=(config, 222, "t-new"),
    )
    reap_result: list[int] = []
    reap_started = threading.Event()

    def reap() -> None:
        reap_started.set()
        reap_result.append(daemon._reap_orphans(config, keep_pid=222))

    reap_thread = threading.Thread(
        target=reap,
    )

    record_thread.start()
    assert entered_record.wait(timeout=2)
    reap_thread.start()
    assert reap_started.wait(timeout=2)
    assert reap_thread.is_alive(), "reap must block behind the record lock"
    assert not signaled
    release_record.set()
    record_thread.join(timeout=2)
    reap_thread.join(timeout=2)

    assert not record_thread.is_alive()
    assert not reap_thread.is_alive()
    assert reap_result == [0]
    assert signaled == []
    assert ledger.read_text(encoding="utf-8") == "111:t-old\n222:t-new\n"


def test_replacing_named_ledger_lock_cannot_split_writers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    monkeypatch.setattr(
        daemon,
        "_proc_start_token",
        lambda pid: {111: "t-first", 222: "t-second"}.get(pid),
    )
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    entered_first = threading.Event()
    release_first = threading.Event()
    original_normalize = daemon._normalized_entries_for_record

    def blocking_normalize(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
        if kwargs["new_pid"] == 111:
            entered_first.set()
            assert release_first.wait(timeout=2)
        return original_normalize(*args, **kwargs)

    monkeypatch.setattr(daemon, "_normalized_entries_for_record", blocking_normalize)
    first = threading.Thread(
        target=daemon._record_spawn_identity,
        args=(config, 111, "t-first"),
    )
    second = threading.Thread(
        target=daemon._record_spawn_identity,
        args=(config, 222, "t-second"),
    )
    first.start()
    assert entered_first.wait(timeout=2)
    named_lock = config.state_dir / daemon.DAEMON_LEDGER_LOCK_NAME
    if os.name == "nt":
        # Confined Windows handles deliberately omit FILE_SHARE_DELETE, so the
        # kernel prevents the replacement attack instead of relying on the
        # additional directory flock used by POSIX.
        with pytest.raises(PermissionError):
            named_lock.unlink()
    else:
        named_lock.unlink()
        named_lock.write_bytes(b"")
    second.start()
    assert second.is_alive(), "replacement lock must still block on the directory inode"
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert (config.state_dir / daemon.DAEMON_LEDGER_NAME).read_text() == (
        "111:t-first\n222:t-second\n"
    )


def test_replacing_named_start_lock_cannot_split_starters(tmp_path) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()

    def hold_first() -> None:
        with daemon._locked_start_lifecycle(config):
            entered_first.set()
            assert release_first.wait(timeout=2)

    def enter_second() -> None:
        with daemon._locked_start_lifecycle(config):
            entered_second.set()

    first = threading.Thread(target=hold_first)
    second = threading.Thread(target=enter_second)
    first.start()
    assert entered_first.wait(timeout=2)
    named_lock = config.state_dir / "server.json.lock"
    if os.name == "nt":
        with pytest.raises(PermissionError):
            named_lock.unlink()
    else:
        named_lock.unlink()
        named_lock.write_bytes(b"")
    second.start()
    assert not entered_second.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert entered_second.is_set()
    assert not first.is_alive() and not second.is_alive()


def test_replacing_whole_state_directory_cannot_overlap_starters(tmp_path) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
    errors: list[BaseException] = []

    def hold_first() -> None:
        try:
            with daemon._locked_start_lifecycle(config):
                entered_first.set()
                assert release_first.wait(timeout=2)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    def enter_second() -> None:
        with daemon._locked_start_lifecycle(config):
            entered_second.set()

    first = threading.Thread(target=hold_first)
    second = threading.Thread(target=enter_second)
    first.start()
    assert entered_first.wait(timeout=2)
    original_state = tmp_path / ".usd-cli.original"
    if os.name == "nt":
        # The held no-reparse state handle denies rename/delete sharing.
        with pytest.raises(PermissionError):
            config.state_dir.rename(original_state)
    else:
        config.state_dir.rename(original_state)
        (tmp_path / ".usd-cli").mkdir()
    second.start()
    assert not entered_second.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert entered_second.is_set()
    assert not first.is_alive() and not second.is_alive()
    if os.name == "nt":
        assert not errors
    else:
        assert len(errors) == 1
        assert "state-directory identity changed" in str(errors[0])


def test_foreign_project_state_and_ledger_never_discover_or_signal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_a = Config(project_dir=tmp_path / "project-a")
    config_b = Config(project_dir=tmp_path / "project-b")
    config_a.state_dir.mkdir(parents=True)
    config_b.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config_a),
        "instance_id": "instance-a",
    }
    for config in (config_a, config_b):
        config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
        (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
            "4242:t-owned\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    probes: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: probes.append("health") or {"ok": True, **state},
    )
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    assert daemon.discover(config_b) is None
    assert "unverified" in daemon.status(config_b)
    with pytest.raises(daemon.DaemonStopRefused, match="foreign-project"):
        daemon.stop(config_b)
    assert not probes
    assert not signaled
    assert config_b.server_state_path.exists()


def test_mismatched_health_identity_is_never_running_or_signal_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    state = {
        "pid": 4242,
        "process_start_token": "t-owned",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "secret",
        "project_id": daemon._project_id(config),
        "instance_id": "instance-a",
    }
    config.server_state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.state_dir / daemon.DAEMON_LEDGER_NAME).write_text(
        "4242:t-owned\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-owned")
    monkeypatch.setattr(daemon, "_proc_state", lambda _pid: "alive")
    monkeypatch.setattr(
        daemon,
        "_health",
        lambda *_args, **_kwargs: {
            "ok": True,
            **state,
            "instance_id": "spoofed-instance",
        },
    )
    signaled: list[int] = []
    monkeypatch.setattr(
        daemon,
        "_terminate",
        lambda pid, **_kwargs: signaled.append(pid) or True,
    )

    assert daemon.discover(config) is None
    assert "unverified daemon health identity" in daemon.status(config)
    with pytest.raises(daemon.DaemonStopRefused, match="unauthenticated"):
        daemon.stop(config)
    assert not signaled


def test_serve_rejects_a_changed_startup_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    config.server["port"] = 4567
    monkeypatch.setattr(daemon, "_proc_start_token", lambda _pid: "t-other")

    with pytest.raises(RuntimeError, match="startup identity changed"):
        server_app.serve(config, startup_identity=(os.getpid(), "t-captured"))


def test_serve_publishes_the_inherited_startup_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    config = Config(project_dir=tmp_path)
    config.server["port"] = 4567
    config.state_dir.mkdir()
    project_descriptor = _open_test_directory(tmp_path)
    state_descriptor = _open_test_directory(config.state_dir)
    identity = daemon.DaemonStartupIdentity(
        pid=os.getpid(),
        process_start_token="t-inherited",
        project_descriptor=project_descriptor,
        state_descriptor=state_descriptor,
        project_path=tmp_path.resolve(),
        state_path=config.state_dir.resolve(),
    )
    observed: dict[str, object] = {}

    class IdleThread:
        def __init__(self, *, target, daemon):
            del target, daemon

        def start(self) -> None:
            return None

    def fake_build_app(_config, **kwargs):
        observed["build_kwargs"] = kwargs
        return "built-app"

    def fake_uvicorn_run(app, **kwargs):
        observed["uvicorn_app"] = app
        observed["uvicorn_kwargs"] = kwargs
        observed["state"] = json.loads(config.server_state_path.read_text())

    monkeypatch.setattr(server_app.threading, "Thread", IdleThread)
    monkeypatch.setattr(server_app, "build_app", fake_build_app)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    try:
        server_app.serve(config, startup_identity=identity)
    finally:
        for descriptor in (project_descriptor, state_descriptor):
            try:
                os.close(descriptor)
            except OSError:
                pass

    state = observed["state"]
    assert isinstance(state, dict)
    assert state["pid"] == os.getpid()
    assert state["process_start_token"] == "t-inherited"
    build_kwargs = observed["build_kwargs"]
    assert isinstance(build_kwargs, dict)
    assert build_kwargs["daemon_pid"] == state["pid"]
    assert build_kwargs["process_start_token"] == state["process_start_token"]
    assert build_kwargs["instance_id"] == state["instance_id"]
    assert observed["uvicorn_app"] == "built-app"
    assert not config.server_state_path.exists()
    with pytest.raises(OSError):
        os.fstat(project_descriptor)
    with pytest.raises(OSError):
        os.fstat(state_descriptor)


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [("127.0.0.1", "127.0.0.1"), ("localhost", "127.0.0.1"), ("::1", "::1")],
)
def test_server_bind_host_is_numeric_and_discoverable(
    configured: str,
    normalized: str,
) -> None:
    assert server_app._normalized_bind_host(configured) == normalized


def test_daemon_server_is_loopback_only_in_every_start_path(
    tmp_path,
) -> None:
    config = Config(project_dir=tmp_path)
    config.server["host"] = "192.0.2.10"

    with pytest.raises(RuntimeError, match="loopback-only"):
        daemon._managed_bind_host(config)
    with pytest.raises(RuntimeError, match="loopback-only"):
        server_app._normalized_bind_host("192.0.2.10")


def test_daemon_state_host_requires_a_string() -> None:
    with pytest.raises(RuntimeError, match="numeric loopback"):
        daemon._base({"host": 2130706433, "port": 4567})
    config = Config()
    config.server["host"] = 2130706433
    with pytest.raises(RuntimeError, match="numeric IP"):
        daemon._managed_bind_host(config)
    with pytest.raises(RuntimeError, match="numeric IP"):
        server_app._normalized_bind_host(2130706433)


def test_server_state_cleanup_preserves_replacement_identity(tmp_path) -> None:
    config = Config(project_dir=tmp_path)
    config.state_dir.mkdir(parents=True)
    replacement = {
        "pid": 5252,
        "process_start_token": "t-replacement",
        "host": "127.0.0.1",
        "port": 4567,
        "token": "replacement-token",
        "project_id": daemon._project_id(config),
        "instance_id": "replacement-instance",
    }
    config.server_state_path.write_text(json.dumps(replacement), encoding="utf-8")

    server_app._remove_server_state_if_owned(
        config,
        daemon_pid=4242,
        process_start_token="t-original",
        instance_id="original-instance",
    )

    assert json.loads(config.server_state_path.read_text()) == replacement


def test_server_state_and_authenticated_health_share_child_process_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        project_dir=tmp_path,
        server={"lifecycle_owner": "external"},
    )
    process_start_token = "t-child"

    class FakeSession:
        _stage_path = None

        def __init__(self, _config, **_kwargs) -> None:  # noqa: ANN001
            pass

    monkeypatch.setattr(server_app, "Session", FakeSession)
    state_path = server_app.write_server_state(
        config,
        "127.0.0.1",
        4567,
        "secret",
        "instance",
        daemon_pid=os.getpid(),
        process_start_token=process_start_token,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    app = server_app.build_app(
        config,
        token="secret",
        instance_id="instance",
        daemon_pid=os.getpid(),
        process_start_token=process_start_token,
    )

    health = TestClient(app).get(
        "/health",
        headers={"x-usd-cli-token": "secret"},
    )

    assert health.status_code == 200
    for field in (
        "pid",
        "process_start_token",
        "project_id",
        "instance_id",
        "lifecycle_owner",
    ):
        assert health.json()[field] == state[field]


def test_server_state_records_unset_default_lifecycle_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(project_dir=tmp_path)
    process_start_token = "t-child"
    monkeypatch.setattr(server_app, "_project_id", lambda _config: "project-id")

    state_path = server_app.write_server_state(
        config,
        "127.0.0.1",
        4567,
        "secret",
        "instance",
        daemon_pid=os.getpid(),
        process_start_token=process_start_token,
    )

    assert json.loads(state_path.read_text(encoding="utf-8"))["lifecycle_owner"] is None
