# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Joint Agent Service event listener progress mapping."""

from ...service.events.listener import FastAPIEventListener
from ...service.runtime.events import StepState


def test_predict_step_started_is_suppressed_until_prediction_progress():
    """Predict start events are suppressed until prediction progress arrives."""
    listener = FastAPIEventListener("session-1234")

    event = listener._map_event_to_progress("step.started", {"step_name": "predict"})

    assert event is None

    event = listener._map_event_to_progress("task.started", {"task_name": "CustomTask"})
    assert event is not None
    assert event.step == "CustomTask"
    assert event.state == StepState.RUNNING
    assert event.percent == 0
