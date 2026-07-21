# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ovphysx daemon venv discovery and install hints."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pytest

from world_understanding.functions.physics import ovphysx_daemon as daemon_mod


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

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    assert daemon._resolve_python() == python_path


def test_missing_daemon_venv_hint_targets_platform_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon_mod.os, "name", "nt")

    with pytest.raises(daemon_mod.OvPhysXDaemonUnavailableError) as exc_info:
        daemon._resolve_python()

    message = str(exc_info.value)
    assert "uv pip install --python" in message
    assert str(tmp_path / "Scripts" / "python.exe") in message
    assert "source checkout's repository root" in message
    assert "--require-hashes --no-deps" in message
    assert str(daemon_mod._ovphysx_runtime_lock()) in message
    assert "--no-config --no-sources" in message
    assert "from ovphysx import PhysX" in message
    assert str(tmp_path / daemon_mod._OVPHYSX_RUNTIME_READY_MARKER) in message
    assert "--extra-index-url" not in message


def test_runtime_lock_selection_is_architecture_specific() -> None:
    assert daemon_mod._ovphysx_runtime_lock("x86_64") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.toml"
    )
    assert daemon_mod._ovphysx_runtime_lock("aarch64") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    )
    assert daemon_mod._ovphysx_runtime_lock("arm64") == Path(
        "apps/physics_agent/runtime/pylock.ovphysx-runtime.aarch64.toml"
    )
    with pytest.raises(ValueError, match="Unsupported architecture"):
        daemon_mod._ovphysx_runtime_lock("ppc64le")


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

    ready_marker.touch()
    assert daemon_mod.ovphysx_runtime_available() is True


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


def test_daemon_start_success_strips_pythonpath_and_sets_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path, device="cuda:0")
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        daemon,
        "_read_stdout_line",
        lambda timeout_s, phase: '{"status": "ready"}\n',
    )
    monkeypatch.setenv("PYTHONPATH", "parent-path")

    daemon._start()

    assert daemon._is_running() is True
    assert "PYTHONPATH" not in seen_env
    assert seen_env["WU_OVPHYSX_DEVICE"] == "cuda:0"


def test_daemon_start_wraps_spawn_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = daemon_mod._OvPhysXDaemon(venv_dir=tmp_path)
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
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
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
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
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
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
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
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
    monkeypatch.setattr(daemon, "_resolve_python", lambda: tmp_path / "bin" / "python")
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
