# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable advisory descriptor locking."""

import os
from contextlib import contextmanager

import pytest

from world_understanding.utils import file_locking


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "state.lock"


def _open(path) -> int:
    return os.open(path, os.O_CREAT | os.O_RDWR, 0o600)


def test_lock_is_exclusive_against_a_second_descriptor(lock_path):
    """A held lock must be reported as contended, not silently granted."""
    first = _open(lock_path)
    second = _open(lock_path)
    try:
        with file_locking.exclusive_descriptor_lock(first):
            with pytest.raises(BlockingIOError):
                with file_locking.exclusive_descriptor_lock(second):
                    raise AssertionError("second lock should not have been granted")
    finally:
        os.close(first)
        os.close(second)


def test_lock_is_released_for_the_next_holder(lock_path):
    first = _open(lock_path)
    second = _open(lock_path)
    try:
        with file_locking.exclusive_descriptor_lock(first):
            pass
        with file_locking.exclusive_descriptor_lock(second):
            pass
    finally:
        os.close(first)
        os.close(second)


def test_lock_is_released_when_the_body_raises(lock_path):
    first = _open(lock_path)
    second = _open(lock_path)
    try:
        with pytest.raises(RuntimeError):
            with file_locking.exclusive_descriptor_lock(first):
                raise RuntimeError("body failed")
        with file_locking.exclusive_descriptor_lock(second):
            pass
    finally:
        os.close(first)
        os.close(second)


def test_descriptor_position_survives_locking(lock_path):
    """Callers keep using the descriptor for I/O, so the offset must be restored."""
    descriptor = _open(lock_path)
    try:
        os.write(descriptor, b"pipeline checkpoint lock")
        before = os.lseek(descriptor, 7, os.SEEK_SET)
        with file_locking.exclusive_descriptor_lock(descriptor):
            pass
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == before
    finally:
        os.close(descriptor)


def test_blocking_lock_retries_contention(monkeypatch, lock_path):
    attempts = 0
    sleeps: list[float] = []

    @contextmanager
    def contend_once(_descriptor: int):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BlockingIOError("busy")
        yield

    monkeypatch.setattr(file_locking, "exclusive_descriptor_lock", contend_once)
    monkeypatch.setattr(file_locking.time, "sleep", sleeps.append)

    descriptor = _open(lock_path)
    try:
        with file_locking.blocking_exclusive_descriptor_lock(
            descriptor,
            retry_interval_seconds=0.02,
        ):
            pass
    finally:
        os.close(descriptor)

    assert attempts == 2
    assert sleeps == [0.02]


def test_blocking_lock_rejects_nonpositive_retry_interval(lock_path):
    descriptor = _open(lock_path)
    try:
        with pytest.raises(ValueError, match="must be positive"):
            with file_locking.blocking_exclusive_descriptor_lock(
                descriptor,
                retry_interval_seconds=0,
            ):
                raise AssertionError("invalid lock interval should not enter")
    finally:
        os.close(descriptor)


@pytest.mark.skipif(os.name != "posix", reason="flock is the POSIX backend")
def test_posix_backend_takes_and_releases_an_exclusive_nonblocking_lock(
    monkeypatch, lock_path
):
    operations: list[int] = []
    monkeypatch.setattr(
        file_locking.fcntl,
        "flock",
        lambda _descriptor, operation: operations.append(operation),
    )

    descriptor = _open(lock_path)
    try:
        with file_locking.exclusive_descriptor_lock(descriptor):
            pass
    finally:
        os.close(descriptor)

    assert operations == [
        file_locking.fcntl.LOCK_EX | file_locking.fcntl.LOCK_NB,
        file_locking.fcntl.LOCK_UN,
    ]


@pytest.mark.skipif(os.name == "posix", reason="msvcrt is the non-POSIX backend")
def test_non_posix_backend_reports_contention_as_blocking_io_error(
    monkeypatch, lock_path
):
    """Windows raises EDEADLOCK/EACCES; callers retry on BlockingIOError."""

    def refuse(_descriptor: int, _mode: int, _length: int) -> None:
        raise OSError(36, "lock violation")

    monkeypatch.setattr(file_locking.msvcrt, "locking", refuse)

    descriptor = _open(lock_path)
    try:
        with pytest.raises(BlockingIOError):
            with file_locking.exclusive_descriptor_lock(descriptor):
                raise AssertionError("lock should not have been granted")
    finally:
        os.close(descriptor)
