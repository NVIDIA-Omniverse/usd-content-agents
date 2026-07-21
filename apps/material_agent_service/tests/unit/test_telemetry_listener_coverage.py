# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the telemetry event listener wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from ...service.events import telemetry_listener as telemetry_module
from ...service.events.telemetry_listener import TelemetryEventListener


class _InnerListener:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def info(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("info", args, kwargs))

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("debug", args, kwargs))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("warning", args, kwargs))

    def error(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("error", args, kwargs))

    def event(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("event", args, kwargs))


@pytest.mark.unit
def test_telemetry_listener_forwards_logs_and_records_step_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _InnerListener()
    listener = TelemetryEventListener(inner)
    timestamps = iter([100, 200, 300, 450])
    monkeypatch.setattr(telemetry_module.time, "time_ns", lambda: next(timestamps))

    listener.info("hello", a=1)
    listener.debug("debug")
    listener.warning("warning")
    listener.error("error")
    listener.event("task.started", {"task_name": "ignored"})
    listener.event("step.started", {"step_name": "render"})
    assert listener.get_step_timings() == []

    listener.event("step.completed", {"step_name": "render"})
    listener.event("step.completed", {"step_name": "missing"})
    listener.event("step.started", {"step_name": "predict"})
    listener.event("step.failed", {"step_name": "predict", "message": "bad"})
    listener.event("step.failed", {"step_name": "missing", "error": "ignored"})

    timings = listener.get_step_timings()
    assert timings == [
        {
            "name": "render",
            "started_at_ns": 100,
            "completed_at_ns": 200,
            "status": "completed",
            "error": None,
        },
        {
            "name": "predict",
            "started_at_ns": 300,
            "completed_at_ns": 450,
            "status": "failed",
            "error": "bad",
        },
    ]
    assert [call[0] for call in inner.calls[:5]] == [
        "info",
        "debug",
        "warning",
        "error",
        "event",
    ]
    assert inner.calls[-1][0] == "event"
