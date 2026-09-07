# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OvRTX daemon stderr tail coverage for abnormal daemon exits."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from world_understanding.functions.graphics import render_ovrtx


class _FakeStdin:
    def write(self, data: str) -> None:
        return None

    def flush(self) -> None:
        return None


class _FakeProcess:
    pid = 88

    def __init__(self, stderr_lines: list[str], returncode: int | None = None) -> None:
        self.stdin = _FakeStdin()
        self.stdout = object()
        self.stderr = stderr_lines
        self.returncode = returncode
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


def _make_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> render_ovrtx._OvRTXDaemon:
    monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
    return render_ovrtx._OvRTXDaemon(
        ovrtx_python=str(tmp_path / "python"),
        daemon_script_path=str(tmp_path / "daemon.py"),
    )


def test_drain_stderr_keeps_bounded_tail_and_never_logs_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    lines = [f"line-{i}\n" for i in range(60)]
    process = _FakeProcess(lines)

    with caplog.at_level(logging.DEBUG, logger=render_ovrtx.__name__):
        daemon._drain_stderr(process)

    # Stderr content is retained in memory only; no log record carries it,
    # not even at DEBUG.
    assert "line-59" not in caplog.text
    assert not caplog.records

    # The ring buffer is bounded to the most recent lines.
    tail = list(daemon._stderr_tail)
    assert len(tail) == render_ovrtx._OVRTX_STDERR_TAIL_LINES
    assert tail[0] == "line-10"
    assert tail[-1] == "line-59"


def test_stderr_secrets_never_reach_any_log_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    # Includes shapes no scrubber can recognize: an unlabelled password and a
    # short bearer token.
    plain_password = "hun" + "ter2"
    short_bearer = "Bearer " + "abc"
    process = _FakeProcess(
        [
            f"login failed for {plain_password}\n",
            f"Authorization: {short_bearer}\n",
            "provider rejected OPENAI_API_KEY=sk-abc123\n",
            'payload was {"api_key": "json-tail-secret"}\n',
        ],
        returncode=3,
    )
    daemon._process = process

    with caplog.at_level(logging.DEBUG, logger=render_ovrtx.__name__):
        daemon._drain_stderr(process)
        daemon._warn_stderr_tail("died during render", 3)

    all_log_text = "\n".join(record.getMessage() for record in caplog.records)
    # No stderr content — recognizable or not — reaches any log surface.
    assert plain_password not in all_log_text
    assert short_bearer not in all_log_text
    assert "sk-abc123" not in all_log_text
    assert "json-tail-secret" not in all_log_text
    assert "login failed" not in all_log_text
    assert "OPENAI_API_KEY" not in all_log_text

    # The WARNING carries only value-free structured fields.
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "died during render" in warnings[0]
    assert "(exit code 3)" in warnings[0]
    assert "4 line(s)" in warnings[0]
    assert "withheld from logs" in warnings[0]


def test_daemon_death_during_render_reports_value_free_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    secret = "nvapi-daemon-crash-secret-713"
    process = _FakeProcess(
        [
            "Traceback (most recent call last):\n",
            f"RuntimeError: renderer init failed with api_key={secret}\n",
        ],
    )
    daemon._process = process
    daemon._drain_stderr(process)

    def read_stdout_dies(timeout: float, phase: str) -> str:
        # The daemon dies mid-request: EOF on stdout, then a crash returncode.
        process.returncode = 139
        return ""

    monkeypatch.setattr(daemon, "_write_stdin_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(daemon, "_read_stdout_line", read_stdout_dies)

    with caplog.at_level(logging.INFO, logger=render_ovrtx.__name__):
        with pytest.raises(RuntimeError, match=r"died during render \(exit code 139\)"):
            daemon.render(
                {
                    "cameras": ["/Camera"],
                    "usd_path": "/stage.usda",
                    "fps": 24.0,
                    "frames": [0],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "died during render" in record.getMessage()
    ]
    assert len(warnings) == 1
    warning_text = warnings[0].getMessage()
    # The WARNING is value-free: it reports counts and the exit code but
    # never the stderr text itself.
    assert "exit code 139" in warning_text
    assert "2 line(s)" in warning_text
    assert "withheld from logs" in warning_text
    assert "Traceback (most recent call last):" not in warning_text
    assert "renderer init failed" not in warning_text
    assert secret not in warning_text


def test_daemon_init_exit_surfaces_stderr_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    process = _FakeProcess(["fatal: missing GPU driver\n"], returncode=17)
    monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
    monkeypatch.setattr(
        render_ovrtx.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        render_ovrtx._OvRTXDaemon,
        "_read_stdout_line",
        lambda self, timeout_s, phase, **kwargs: "",
    )
    daemon = render_ovrtx._OvRTXDaemon(
        ovrtx_python=str(tmp_path / "python"),
        daemon_script_path=str(tmp_path / "daemon.py"),
    )

    with caplog.at_level(logging.INFO, logger=render_ovrtx.__name__):
        with pytest.raises(RuntimeError, match="exit code 17"):
            daemon.ensure_running()

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "exited during init" in record.getMessage()
    ]
    assert len(warning_messages) == 1
    # Value-free summary only: the stderr text never reaches the log.
    assert "fatal: missing GPU driver" not in warning_messages[0]
    assert "1 line(s)" in warning_messages[0]
    assert "withheld from logs" in warning_messages[0]


def test_drain_stderr_bounds_giant_newline_free_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    giant = "A1" * 500_000  # single ~1 MB record, no newline
    process = _FakeProcess([], returncode=9)
    process.stderr = io.StringIO(giant + "\ntrailing line\n")  # type: ignore[assignment]
    daemon._process = process

    daemon._drain_stderr(process)

    tail = list(daemon._stderr_tail)
    # The giant record is truncated to the per-record cap, with a marker, and
    # the total retained tail stays within the byte budget.
    assert any(line.endswith("[record truncated]") for line in tail)
    marker_len = len(render_ovrtx._OVRTX_STDERR_TRUNCATION_MARKER)
    assert all(
        len(line) <= render_ovrtx._OVRTX_STDERR_MAX_RECORD_CHARS + marker_len
        for line in tail
    )
    assert (
        sum(len(line) for line in tail)
        <= render_ovrtx._OVRTX_STDERR_MAX_TAIL_TOTAL_CHARS + marker_len
    )
    assert tail[-1] == "trailing line"

    # The WARNING publication of the tail is bounded accordingly.
    with caplog.at_level(logging.INFO, logger=render_ovrtx.__name__):
        daemon._warn_stderr_tail("died during render", 9)
    warning = next(
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert len(warning) < 2 * render_ovrtx._OVRTX_STDERR_MAX_TAIL_TOTAL_CHARS


def test_idle_daemon_crash_tail_surfaces_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    secret = "nvapi-idle-crash-secret-401"
    dead_process = _FakeProcess([], returncode=134)
    daemon._process = dead_process
    daemon._stderr_tail.extend(
        [
            "Traceback (most recent call last):",
            f"RuntimeError: idle crash with api_key={secret}",
        ]
    )

    new_process = _FakeProcess([])
    monkeypatch.setattr(
        render_ovrtx.subprocess,
        "Popen",
        lambda *args, **kwargs: new_process,
    )
    monkeypatch.setattr(
        render_ovrtx._OvRTXDaemon,
        "_read_stdout_line",
        lambda self, timeout_s, phase, **kwargs: '{"status": "ready"}',
    )

    with caplog.at_level(logging.INFO, logger=render_ovrtx.__name__):
        daemon._start()

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "exited while idle" in record.getMessage()
    ]
    # The idle crash is reported (value-free) before the tail reset.
    assert len(warnings) == 1
    assert "exit code 134" in warnings[0]
    assert "2 line(s)" in warnings[0]
    assert "withheld from logs" in warnings[0]
    assert "idle crash" not in warnings[0]
    assert secret not in warnings[0]
    # The replacement daemon starts with a fresh tail.
    assert daemon._process is new_process
    assert "idle crash" not in "\n".join(daemon._stderr_tail)


def test_warn_stderr_tail_join_is_clamped_to_active_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    join_timeouts: list[float | None] = []

    class _StuckDrainThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            join_timeouts.append(timeout)

    monkeypatch.setattr(daemon, "_stderr_thread", _StuckDrainThread())

    # Without a deadline the join keeps its default 0.5 s bound.
    daemon._warn_stderr_tail("died during render", 1)
    assert join_timeouts[-1] == 0.5

    # A distant deadline keeps the 0.5 s cap.
    daemon._warn_stderr_tail(
        "died during render", 1, deadline=render_ovrtx.time.monotonic() + 60.0
    )
    assert join_timeouts[-1] == 0.5

    # A nearly exhausted deadline clamps the join below the 0.5 s default.
    daemon._warn_stderr_tail(
        "died during render", 1, deadline=render_ovrtx.time.monotonic() + 0.1
    )
    assert join_timeouts[-1] is not None
    assert 0.0 <= join_timeouts[-1] <= 0.1

    # An already expired deadline never blocks.
    daemon._warn_stderr_tail(
        "exited during init", 17, deadline=render_ovrtx.time.monotonic() - 5.0
    )
    assert join_timeouts[-1] == 0.0


def test_daemon_death_without_captured_stderr_still_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    daemon = _make_daemon(tmp_path, monkeypatch)
    process = _FakeProcess([])
    daemon._process = process

    def read_stdout_dies(timeout: float, phase: str) -> str:
        process.returncode = 1
        return ""

    monkeypatch.setattr(daemon, "_write_stdin_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(daemon, "_read_stdout_line", read_stdout_dies)

    with caplog.at_level(logging.INFO, logger=render_ovrtx.__name__):
        with pytest.raises(RuntimeError, match="died during render"):
            daemon.render(
                {
                    "cameras": ["/Camera"],
                    "usd_path": "/stage.usda",
                    "fps": 24.0,
                    "frames": [0],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )

    assert any(
        record.levelno == logging.WARNING
        and "no stderr output was captured" in record.getMessage()
        for record in caplog.records
    )
