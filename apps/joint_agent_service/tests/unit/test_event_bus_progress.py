# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for EventBus progress / completion semantics.

Covers the full default service pipeline (identify_asset → render →
prepare → predict → restore_usd) so the bus's weight/completion maps
can't drift away from the steps that actually run.
"""

import asyncio

import pytest

from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState

DEFAULT_STEP_ORDER = (
    "identify_asset",
    "build_dataset_usd",
    "build_dataset_prepare_dataset",
    "predict",
    "consistency_pass",
    "infer_articulation_candidates",
    "restore_usd",
)


def _completion_event(session_id: str, step: str) -> ProgressEvent:
    return ProgressEvent(
        session_id=session_id,
        step=step,
        state=StepState.COMPLETED,
        percent=100,
    )


def _pipeline_completed_event(session_id: str) -> ProgressEvent:
    return ProgressEvent(
        session_id=session_id,
        step="pipeline",
        state=StepState.COMPLETED,
        percent=100,
        extra={"pipeline_completed": True},
    )


def _running_event(session_id: str, step: str, percent: int) -> ProgressEvent:
    return ProgressEvent(
        session_id=session_id,
        step=step,
        state=StepState.RUNNING,
        percent=percent,
        message=f"running {step}",
    )


@pytest.mark.asyncio
async def test_seed_pending_session_resets_terminal_state_and_sse_queue() -> None:
    bus = EventBus()
    session_id = "regenerated"
    await bus.emit(_pipeline_completed_event(session_id))
    terminal_queue = bus.get_queue(session_id)
    assert not terminal_queue.empty()
    terminal_queue.get_nowait()
    waiting_subscriber = asyncio.create_task(terminal_queue.get())
    await asyncio.sleep(0)

    await bus.seed_pending_session(session_id)

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "pending"
    assert snapshot["completed_steps"] == []
    assert snapshot["overall_progress"]["percent"] == 0
    assert bus.get_queue(session_id) is terminal_queue
    assert terminal_queue.empty()

    regenerated_event = _running_event(session_id, "predict", 10)
    await bus.emit(regenerated_event)
    assert await asyncio.wait_for(waiting_subscriber, timeout=1) is regenerated_event


@pytest.mark.asyncio
async def test_status_stays_running_when_predict_completes_before_restore_usd() -> None:
    """predict completing must NOT flip the session to status=completed.

    restore_usd still has to run after predict and write restored predictions.
    If the bus marked the session completed at this point, clients would try
    to download the physics USD before it existed.
    """
    bus = EventBus()
    session_id = "s1"

    for step in (
        "identify_asset",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
    ):
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] != "completed", (
        "session must not be marked completed until restore_usd runs"
    )
    assert state["overall_progress"]["percent"] < 100


@pytest.mark.asyncio
async def test_status_flips_to_completed_after_pipeline_ready_event() -> None:
    """The service marks done only after result artifacts are ready."""
    bus = EventBus()
    session_id = "s2"

    for step in DEFAULT_STEP_ORDER:
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "running"
    assert state["overall_progress"]["percent"] == 98

    await bus.emit(_pipeline_completed_event(session_id))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["overall_progress"]["percent"] == 100


def test_completion_helper_preserves_legacy_terminal_percent() -> None:
    """Completion helper still finalizes state that is already at 100%."""
    bus = EventBus()
    state = {
        "status": "running",
        "completed_steps": [{"name": "custom"}],
        "overall_progress": {"current_step": 1, "total_steps": 1, "percent": 100},
    }

    bus._update_overall_progress_on_completion(state, "custom")

    assert state["status"] == "completed"
    assert "completed_at" in state


@pytest.mark.asyncio
async def test_identify_asset_has_display_name_and_weight() -> None:
    """Regression: identify_asset must advance overall progress off 0.

    Previously the weight map only covered build_dataset_usd / prepare /
    predict / apply_physics, so the ~15 seconds the default service
    pipeline spends in identify_asset showed percent=0 in /status.
    """
    bus = EventBus()
    session_id = "s3"

    await bus.emit(_running_event(session_id, "identify_asset", percent=50))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["current_step"] is not None
    assert state["current_step"]["display_name"] == "Identifying Asset"
    overall = state["overall_progress"]["percent"]
    assert 5 <= overall <= 10


@pytest.mark.asyncio
async def test_restore_usd_has_display_name_and_weight() -> None:
    """restore_usd must be registered in the bus's weight + display maps."""
    bus = EventBus()
    session_id = "s4"

    await bus.emit(_running_event(session_id, "restore_usd", percent=50))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["current_step"] is not None
    assert state["current_step"]["display_name"] == "Restoring Predictions"
    overall = state["overall_progress"]["percent"]
    assert 96 <= overall <= 100


@pytest.mark.asyncio
async def test_current_step_never_exceeds_total_steps_with_optimize_usd() -> None:
    """Enabling optimize_usd must not push current_step past total_steps.

    Regression: current_step used to be `len(completed_steps)`, so running
    the full 6-step sequence (optimize_usd + the 5 default steps) left the
    session at {"current_step": 6, "total_steps": 5, "percent": 100}.
    """
    from ...service.progress import SERVICE_DEFAULT_TOTAL_STEPS

    bus = EventBus()
    session_id = "s_opt"

    full_sequence = (
        "optimize_usd",
        "identify_asset",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "restore_usd",
        "restore_usd",
    )
    for step in full_sequence:
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    await bus.emit(_pipeline_completed_event(session_id))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["overall_progress"]["percent"] == 100
    progress = state["overall_progress"]
    assert progress["total_steps"] >= SERVICE_DEFAULT_TOTAL_STEPS
    assert progress["current_step"] <= progress["total_steps"]


@pytest.mark.asyncio
async def test_apply_joint_rigger_uses_final_progress_slot() -> None:
    """Opt-in Joint Rigger runs extend progress after restore_usd."""
    bus = EventBus()
    session_id = "s_rigger"

    for step in DEFAULT_STEP_ORDER:
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "running"
    assert state["overall_progress"] == {
        "current_step": 8,
        "total_steps": 8,
        "percent": 98,
    }

    await bus.emit(_running_event(session_id, "apply_joint_rigger", percent=50))
    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["overall_progress"] == {
        "current_step": 9,
        "total_steps": 9,
        "percent": 98,
    }

    await bus.emit(_completion_event(session_id, "apply_joint_rigger"))
    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "running"
    assert state["overall_progress"]["percent"] == 99

    await bus.emit(_pipeline_completed_event(session_id))
    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "completed"
    assert state["overall_progress"]["percent"] == 100


@pytest.mark.asyncio
async def test_predict_in_flight_progress_uses_weighted_range() -> None:
    """In-flight predict at percent=100 must not push overall to 100.

    Regression mirror on the bus side of the SessionManager issue: predict
    step-progress is weighted 60→90, so step_percent=100 should land overall
    at 88, leaving room for consistency, articulation, and restore steps.
    """
    bus = EventBus()
    session_id = "s_pred"

    for step in (
        "identify_asset",
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
    ):
        await bus.emit(_running_event(session_id, step, percent=50))
        await bus.emit(_completion_event(session_id, step))

    await bus.emit(_running_event(session_id, "predict", percent=100))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["status"] == "running"
    assert state["overall_progress"]["percent"] == 88


@pytest.mark.asyncio
async def test_total_steps_matches_visible_default_pipeline() -> None:
    """total_steps in /status must match the actual default pipeline length."""
    from ...service.progress import SERVICE_DEFAULT_TOTAL_STEPS

    bus = EventBus()
    session_id = "s5"
    await bus.emit(_running_event(session_id, "identify_asset", percent=0))

    state = bus.get_snapshot(session_id)
    assert state is not None
    assert state["overall_progress"]["total_steps"] == SERVICE_DEFAULT_TOTAL_STEPS
