# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable advisory locking for an already-open descriptor.

``fcntl`` is POSIX-only, so callers that lock a descriptor they already hold
route through here instead of importing it directly.

``fcntl.flock`` locks the whole open file description and ignores the
descriptor position. ``msvcrt.locking`` has no such call, so the non-POSIX
backend locks an agreed byte range instead and restores the position after
every call, leaving the descriptor usable for I/O either way.

The backend is chosen once at import: the non-POSIX implementation cannot run
on the hosts CI covers, so it is defined in its own excluded block rather than
scattered through per-call platform branches.
"""

import errno
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

# Locking one byte is enough for mutual exclusion because every participant
# agrees on the same range. Zero-length regions are rejected on Windows.
_LOCK_REGION_BYTES = 1

#: Errno values ``msvcrt.locking`` uses for an already-held lock. Every other
#: OSError from it means the descriptor or request is invalid, which no amount
#: of retrying fixes.
_CONTENDED_ERRNOS = frozenset({errno.EACCES, errno.EDEADLOCK, errno.EAGAIN})

if os.name == "posix":
    import fcntl

    def _lock_exclusive_nonblocking(descriptor: int) -> None:
        """Take an exclusive lock, raising ``BlockingIOError`` when contended."""
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(descriptor: int) -> None:
        """Release a lock taken by :func:`_lock_exclusive_nonblocking`."""
        fcntl.flock(descriptor, fcntl.LOCK_UN)

else:  # pragma: no cover - selected only on hosts without fcntl
    import msvcrt

    def _region_lock(descriptor: int, mode: int) -> None:
        """Lock or unlock the agreed byte, restoring the descriptor position."""
        position = os.lseek(descriptor, 0, os.SEEK_CUR)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, mode, _LOCK_REGION_BYTES)
        finally:
            os.lseek(descriptor, position, os.SEEK_SET)

    def _lock_exclusive_nonblocking(descriptor: int) -> None:
        """Take an exclusive lock, raising ``BlockingIOError`` when contended."""
        try:
            _region_lock(descriptor, msvcrt.LK_NBLCK)
        except OSError as error:
            # Windows reports contention as EDEADLOCK/EACCES rather than EAGAIN,
            # so normalize just those to the error POSIX callers already retry
            # on. Anything else (EBADF, ENOLCK, EINVAL) is a broken descriptor,
            # not a busy one: reporting it as contention makes the caller retry
            # until its deadline and then blame a timeout for a permanent fault.
            if error.errno not in _CONTENDED_ERRNOS:
                raise
            raise BlockingIOError(
                error.errno, "lock is held by another process"
            ) from None

    def _unlock(descriptor: int) -> None:
        """Release a lock taken by :func:`_lock_exclusive_nonblocking`."""
        _region_lock(descriptor, msvcrt.LK_UNLCK)


@contextmanager
def exclusive_descriptor_lock(descriptor: int) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``descriptor`` for the block.

    Raises:
        BlockingIOError: The lock is already held. Callers decide whether to
            retry within their own deadline.
    """
    _lock_exclusive_nonblocking(descriptor)
    try:
        yield
    finally:
        _unlock(descriptor)


@contextmanager
def blocking_exclusive_descriptor_lock(
    descriptor: int,
    *,
    retry_interval_seconds: float = 0.01,
) -> Iterator[None]:
    """Wait for and hold an exclusive advisory lock on ``descriptor``.

    This preserves blocking ``flock`` behavior for callers that do not own a
    timeout policy while sharing the same POSIX/Windows backend as
    :func:`exclusive_descriptor_lock`.
    """

    if retry_interval_seconds <= 0:
        raise ValueError("retry_interval_seconds must be positive")
    while True:
        lock = exclusive_descriptor_lock(descriptor)
        try:
            lock.__enter__()
        except BlockingIOError:
            time.sleep(retry_interval_seconds)
        else:
            break
    try:
        yield
    finally:
        lock.__exit__(None, None, None)
