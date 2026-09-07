# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light cooperative cancellation for camera-analysis work.

The daemon binds a :class:`threading.Event` to the worker thread that owns one
camera command.  Analysis code checks that event only at explicit safe boundaries;
the module deliberately imports neither USD, NumPy, Newton, nor Warp.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class CameraAnalysisCancelled(RuntimeError):
    """Raised when a bound camera-analysis request asks to stop."""


_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "usd_cli_camera_analysis_cancel_event", default=None
)


@contextmanager
def cancellation_scope(event: threading.Event | None) -> Iterator[None]:
    """Bind ``event`` to the current execution context for cooperative checks."""

    token = _CANCEL_EVENT.set(event)
    try:
        yield
    finally:
        _CANCEL_EVENT.reset(token)


@contextmanager
def defer_cancellation() -> Iterator[None]:
    """Temporarily defer checks while an indivisible operation finishes safely."""

    token = _CANCEL_EVENT.set(None)
    try:
        yield
    finally:
        _CANCEL_EVENT.reset(token)


def check_cancelled() -> None:
    """Raise at a safe boundary when the bound request has been cancelled."""

    event = _CANCEL_EVENT.get()
    if event is not None and event.is_set():
        raise CameraAnalysisCancelled("camera analysis cancelled")


__all__ = [
    "CameraAnalysisCancelled",
    "cancellation_scope",
    "check_cancelled",
    "defer_cancellation",
]
