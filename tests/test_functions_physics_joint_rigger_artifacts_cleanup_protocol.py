# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cleanup-ownership protocol tests for shared Joint Rigger artifacts."""

from collections.abc import Callable

import pytest

from world_understanding.functions.physics.joint_rigger import artifacts


def _failing_step(
    calls: list[str],
    name: str,
    error: BaseException,
) -> tuple[str, Callable[[], None]]:
    def fail() -> None:
        calls.append(name)
        raise error

    return name, fail


def test_cleanup_protocol_preserves_active_primary_and_runs_every_step() -> None:
    calls: list[str] = []
    primary = KeyboardInterrupt("active operation failure")
    ordinary = OSError("ordinary cleanup failure")
    ordinary.add_note("nested cleanup detail")
    fatal = SystemExit("fatal cleanup failure")

    artifacts._run_cleanup_steps(
        [
            _failing_step(calls, "ordinary", ordinary),
            _failing_step(calls, "fatal", fatal),
        ],
        primary_error=primary,
        label="test cleanup",
    )

    assert calls == ["ordinary", "fatal"]
    notes = "\n".join(primary.__notes__)
    assert "ordinary: OSError: ordinary cleanup failure" in notes
    assert "ordinary detail: nested cleanup detail" in notes
    assert "fatal: SystemExit: fatal cleanup failure" in notes


def test_cleanup_protocol_prefers_first_standalone_fatal() -> None:
    calls: list[str] = []
    ordinary = OSError("ordinary cleanup failure")
    first_fatal = SystemExit("first fatal cleanup failure")
    later_fatal = KeyboardInterrupt("later fatal cleanup failure")

    with pytest.raises(SystemExit) as raised:
        artifacts._run_cleanup_steps(
            [
                _failing_step(calls, "ordinary", ordinary),
                _failing_step(calls, "first fatal", first_fatal),
                _failing_step(calls, "later fatal", later_fatal),
            ],
            label="test cleanup",
        )

    assert raised.value is first_fatal
    assert calls == ["ordinary", "first fatal", "later fatal"]
    notes = "\n".join(first_fatal.__notes__)
    assert "ordinary: OSError: ordinary cleanup failure" in notes
    assert "later fatal: KeyboardInterrupt: later fatal cleanup failure" in notes


def test_cleanup_protocol_routes_ordinary_committed_failures() -> None:
    calls: list[str] = []
    first = OSError("first cleanup failure")
    second = RuntimeError("second cleanup failure")
    state = artifacts._PublicationCleanupState(committed=True)

    artifacts._run_cleanup_steps(
        [
            _failing_step(calls, "first", first),
            _failing_step(calls, "second", second),
        ],
        cleanup_state=state,
        label="test cleanup",
    )

    assert calls == ["first", "second"]
    assert state.errors == [first, second]


def test_cleanup_protocol_drains_prior_committed_errors_into_active_primary() -> None:
    prior = OSError("earlier committed cleanup failure")
    state = artifacts._PublicationCleanupState(committed=True, errors=[prior])
    primary = KeyboardInterrupt("active postcommit failure")

    artifacts._route_cleanup_failures(
        [],
        primary_error=primary,
        cleanup_state=state,
        label="test cleanup",
    )

    assert not state.errors
    assert "Earlier committed cleanup also failed: OSError" in "\n".join(
        primary.__notes__
    )


def test_cleanup_protocol_drains_prior_committed_errors_into_fatal_cleanup() -> None:
    calls: list[str] = []
    prior = OSError("earlier committed cleanup failure")
    state = artifacts._PublicationCleanupState(committed=True, errors=[prior])
    fatal = SystemExit("fatal cleanup failure")

    with pytest.raises(SystemExit) as raised:
        artifacts._run_cleanup_steps(
            [_failing_step(calls, "fatal", fatal)],
            cleanup_state=state,
            label="test cleanup",
        )

    assert raised.value is fatal
    assert calls == ["fatal"]
    assert not state.errors
    assert "Earlier committed cleanup also failed: OSError" in "\n".join(
        fatal.__notes__
    )


def test_cleanup_protocol_groups_multiple_standalone_ordinary_failures() -> None:
    calls: list[str] = []
    first = OSError("first cleanup failure")
    second = RuntimeError("second cleanup failure")

    with pytest.raises(ExceptionGroup) as raised:
        artifacts._run_cleanup_steps(
            [
                _failing_step(calls, "first", first),
                _failing_step(calls, "second", second),
            ],
            label="test cleanup",
        )

    assert calls == ["first", "second"]
    assert raised.value.exceptions == (first, second)
