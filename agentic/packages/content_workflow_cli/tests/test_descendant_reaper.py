# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused descendant-reaper cleanup regressions."""

from __future__ import annotations

import os
import select
import signal
import threading

import pytest

from content_workflow_cli import descendant_reaper


def _supervisor_arguments(
    *, ready_fd: int, start_fd: int, command: list[str]
) -> list[str]:
    return [
        "--ready-fd",
        str(ready_fd),
        "--start-fd",
        str(start_fd),
        "--",
        *command,
    ]


@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        ([], "missing or invalid readiness protocol"),
        (
            ["--ready-fd", "bad", "--start-fd", "2", "--", "provider"],
            "invalid literal",
        ),
        (
            ["--ready-fd", "1", "--start-fd", "1", "--", "provider"],
            "invalid readiness file descriptors",
        ),
        (
            ["--ready-fd", "1", "--start-fd", "2", "--"],
            "missing command",
        ),
    ],
)
def test_supervisor_command_rejects_invalid_readiness_protocol(
    arguments: list[str],
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        descendant_reaper._supervisor_command(arguments)


def test_kill_and_reap_descendants_uses_only_nonblocking_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_scans = iter([{101}, set(), set()])
    wait_calls: list[tuple[int, int]] = []
    kill_calls: list[tuple[int, signal.Signals]] = []

    def fake_waitpid(pid: int, options: int) -> tuple[int, int]:
        wait_calls.append((pid, options))
        return (0, 0)

    monkeypatch.setattr(
        descendant_reaper,
        "_direct_child_pids",
        lambda: next(child_scans),
    )
    monkeypatch.setattr(descendant_reaper.os, "waitpid", fake_waitpid)
    monkeypatch.setattr(
        descendant_reaper.os,
        "kill",
        lambda pid, signum: kill_calls.append((pid, signum)),
    )
    monkeypatch.setattr(descendant_reaper.time, "sleep", lambda _seconds: None)

    remaining = descendant_reaper._kill_and_reap_descendants(timeout_seconds=1.0)

    assert remaining == set()
    assert kill_calls == [(101, signal.SIGKILL)]
    assert wait_calls
    assert all(options == os.WNOHANG for _pid, options in wait_calls)


def test_reap_exited_children_preserves_protected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        descendant_reaper,
        "_direct_child_pids",
        lambda: {301, 302, 303},
    )
    monkeypatch.setattr(
        descendant_reaper.os,
        "waitpid",
        lambda pid, options: wait_calls.append((pid, options)) or (pid, 0),
    )

    descendant_reaper._reap_exited_children(protected_pids={301})

    assert wait_calls == [(302, os.WNOHANG), (303, os.WNOHANG)]


def test_kill_and_reap_descendants_bounds_unreapable_child_and_continues_kills(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    child_pids = {201, 202}
    wait_calls: list[tuple[int, int]] = []
    kill_calls: list[int] = []

    def fake_waitpid(pid: int, options: int) -> tuple[int, int]:
        wait_calls.append((pid, options))
        return (0, 0)

    def fake_kill(pid: int, _signum: signal.Signals) -> None:
        kill_calls.append(pid)
        if pid == 201:
            raise PermissionError("not permitted")

    monkeypatch.setattr(
        descendant_reaper,
        "_direct_child_pids",
        lambda: set(child_pids),
    )
    monkeypatch.setattr(descendant_reaper.os, "waitpid", fake_waitpid)
    monkeypatch.setattr(descendant_reaper.os, "kill", fake_kill)

    remaining = descendant_reaper._kill_and_reap_descendants(timeout_seconds=0.0)

    assert remaining == child_pids
    assert set(kill_calls) == child_pids
    assert all(options == os.WNOHANG for _pid, options in wait_calls)
    assert "could not SIGKILL descendant 201" in capsys.readouterr().err


def test_unreaped_descendant_message_is_bounded_and_reports_actual_timeout() -> None:
    child_pids = set(
        range(100, 100 + descendant_reaper._DESCENDANT_PID_REPORT_LIMIT + 2)
    )

    message = descendant_reaper._unreaped_descendants_message(
        child_pids,
        timeout_seconds=5.0,
    )

    assert "18 descendant(s) remained after 5.0s" in message
    assert "100" in message
    assert "115" in message
    assert "116" not in message
    assert "2 more" in message


def test_print_cleanup_timeout_includes_exception_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    descendant_reaper._print_cleanup_timeout(
        {101},
        after=RuntimeError("cleanup failed"),
        timeout_seconds=5.0,
    )

    assert capsys.readouterr().err == (
        "descendant-reaper: cleanup timed out after cleanup failed: "
        "1 descendant(s) remained after 5.0s: 101\n"
    )


def test_main_fails_closed_when_reap_deadline_expires(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeTarget:
        pid = 301

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(descendant_reaper.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        descendant_reaper,
        "_configure_supervisor",
        lambda _parent_pid: None,
    )
    monkeypatch.setattr(
        descendant_reaper.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeTarget(),
    )
    monkeypatch.setattr(
        descendant_reaper,
        "_kill_and_reap_descendants",
        lambda: {302},
    )

    ready_read_fd, ready_write_fd = os.pipe()
    start_read_fd, start_write_fd = os.pipe()
    os.write(start_write_fd, descendant_reaper._SUPERVISOR_START_TOKEN)
    try:
        returncode = descendant_reaper.main(
            _supervisor_arguments(
                ready_fd=ready_write_fd,
                start_fd=start_read_fd,
                command=["provider"],
            )
        )
        assert os.read(ready_read_fd, 1) == descendant_reaper._SUPERVISOR_READY_TOKEN
    finally:
        for fd in (ready_read_fd, start_write_fd):
            descendant_reaper._close_fd(fd)

    assert returncode == descendant_reaper._SUPERVISOR_ERROR
    assert "cleanup timed out" in capsys.readouterr().err


def test_main_signals_ready_before_starting_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTarget:
        pid = 301

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    provider_started = threading.Event()
    result: list[int] = []
    monkeypatch.setattr(descendant_reaper.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        descendant_reaper,
        "_configure_supervisor",
        lambda _parent_pid: None,
    )
    monkeypatch.setattr(
        descendant_reaper.subprocess,
        "Popen",
        lambda *_args, **_kwargs: provider_started.set() or FakeTarget(),
    )
    monkeypatch.setattr(descendant_reaper, "_kill_and_reap_descendants", set)

    ready_read_fd, ready_write_fd = os.pipe()
    start_read_fd, start_write_fd = os.pipe()
    supervisor = threading.Thread(
        target=lambda: result.append(
            descendant_reaper.main(
                _supervisor_arguments(
                    ready_fd=ready_write_fd,
                    start_fd=start_read_fd,
                    command=["provider"],
                )
            )
        )
    )
    supervisor.start()
    try:
        readable, _, _ = select.select([ready_read_fd], [], [], 2.0)
        assert readable == [ready_read_fd]
        assert os.read(ready_read_fd, 1) == descendant_reaper._SUPERVISOR_READY_TOKEN
        assert not provider_started.is_set()

        os.write(start_write_fd, descendant_reaper._SUPERVISOR_START_TOKEN)
        supervisor.join(timeout=2.0)
    finally:
        for fd in (ready_read_fd, start_write_fd):
            descendant_reaper._close_fd(fd)

    assert not supervisor.is_alive()
    assert provider_started.is_set()
    assert result == [0]


def test_main_startup_failure_closes_readiness_without_starting_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider_started = False

    def fail_startup(_parent_pid: int) -> None:
        raise OSError("startup failed")

    def start_provider(*_args: object, **_kwargs: object) -> object:
        nonlocal provider_started
        provider_started = True
        return object()

    monkeypatch.setattr(descendant_reaper.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(descendant_reaper, "_configure_supervisor", fail_startup)
    monkeypatch.setattr(descendant_reaper.subprocess, "Popen", start_provider)

    ready_read_fd, ready_write_fd = os.pipe()
    start_read_fd, start_write_fd = os.pipe()
    try:
        returncode = descendant_reaper.main(
            _supervisor_arguments(
                ready_fd=ready_write_fd,
                start_fd=start_read_fd,
                command=["provider"],
            )
        )
        assert os.read(ready_read_fd, 1) == b""
    finally:
        for fd in (ready_read_fd, start_write_fd):
            descendant_reaper._close_fd(fd)

    assert returncode == descendant_reaper._SUPERVISOR_ERROR
    assert not provider_started
    assert "failed to start target: startup failed" in capsys.readouterr().err


def test_main_does_not_start_provider_without_runner_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider_started = False

    def start_provider(*_args: object, **_kwargs: object) -> object:
        nonlocal provider_started
        provider_started = True
        return object()

    monkeypatch.setattr(descendant_reaper.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        descendant_reaper,
        "_configure_supervisor",
        lambda _parent_pid: None,
    )
    monkeypatch.setattr(descendant_reaper.subprocess, "Popen", start_provider)

    ready_read_fd, ready_write_fd = os.pipe()
    start_read_fd, start_write_fd = os.pipe()
    os.close(start_write_fd)
    try:
        returncode = descendant_reaper.main(
            _supervisor_arguments(
                ready_fd=ready_write_fd,
                start_fd=start_read_fd,
                command=["provider"],
            )
        )
        assert os.read(ready_read_fd, 1) == descendant_reaper._SUPERVISOR_READY_TOKEN
    finally:
        descendant_reaper._close_fd(ready_read_fd)

    assert returncode == descendant_reaper._SUPERVISOR_ERROR
    assert not provider_started
    assert "runner did not confirm supervisor readiness" in capsys.readouterr().err
