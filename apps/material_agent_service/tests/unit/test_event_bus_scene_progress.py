# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EventBus large-scene progress handling."""

from unittest.mock import AsyncMock

import pytest

from ...service.runtime import EventBus, ProgressEvent, StepState


@pytest.mark.asyncio
async def test_known_scene_step_switches_to_scene_progress_mode() -> None:
    bus = EventBus()
    bus._persist_status = AsyncMock()  # type: ignore[method-assign]
    bus._save_event_to_log = AsyncMock()  # type: ignore[method-assign]

    await bus.emit(
        ProgressEvent(
            session_id="scene-known",
            step="scene_analyze",
            state=StepState.RUNNING,
            percent=50,
        )
    )

    snapshot = bus.get_snapshot("scene-known")
    assert snapshot is not None
    assert snapshot["overall_progress"]["total_steps"] == 9
    assert snapshot["overall_progress"]["current_step"] == 1
    assert snapshot["overall_progress"]["percent"] == 5


@pytest.mark.asyncio
async def test_unknown_scene_prefixed_step_does_not_switch_progress_mode() -> None:
    bus = EventBus()
    bus._persist_status = AsyncMock()  # type: ignore[method-assign]
    bus._save_event_to_log = AsyncMock()  # type: ignore[method-assign]

    await bus.emit(
        ProgressEvent(
            session_id="scene-typo",
            step="scene_typo",
            state=StepState.RUNNING,
            percent=50,
        )
    )

    snapshot = bus.get_snapshot("scene-typo")
    assert snapshot is not None
    assert snapshot["overall_progress"]["total_steps"] == 3
    assert snapshot["overall_progress"]["current_step"] == 0
    assert snapshot["overall_progress"]["percent"] == 0


@pytest.mark.asyncio
async def test_reused_session_resets_run_scoped_progress_after_completion() -> None:
    bus = EventBus()
    bus._persist_status = AsyncMock()  # type: ignore[method-assign]
    bus._save_event_to_log = AsyncMock()  # type: ignore[method-assign]

    await bus.emit(
        ProgressEvent(
            session_id="reused-session",
            step="scene_analyze",
            state=StepState.RUNNING,
            percent=100,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id="reused-session",
            step="scene_analyze",
            state=StepState.COMPLETED,
            percent=100,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id="reused-session",
            step="scene_pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True, "coverage": {}},
        )
    )

    completed_snapshot = bus.get_snapshot("reused-session")
    assert completed_snapshot is not None
    assert completed_snapshot["status"] == "completed"
    assert len(completed_snapshot["completed_steps"]) == 1

    await bus.emit(
        ProgressEvent(
            session_id="reused-session",
            step="predict",
            state=StepState.RUNNING,
            percent=10,
        )
    )

    snapshot = bus.get_snapshot("reused-session")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["completed_steps"] == []
    assert snapshot["step_timings"] == {}
    assert snapshot["overall_progress"]["total_steps"] == 3
    assert snapshot["overall_progress"]["percent"] == 48
    assert "completed_at" not in snapshot


@pytest.mark.asyncio
async def test_optional_render_after_apply_does_not_reset_same_run_progress() -> None:
    bus = EventBus()
    bus._persist_status = AsyncMock()  # type: ignore[method-assign]
    bus._save_event_to_log = AsyncMock()  # type: ignore[method-assign]

    await bus.emit(
        ProgressEvent(
            session_id="render-after-apply",
            step="apply",
            state=StepState.RUNNING,
            percent=100,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id="render-after-apply",
            step="apply",
            state=StepState.COMPLETED,
            percent=100,
        )
    )

    apply_snapshot = bus.get_snapshot("render-after-apply")
    assert apply_snapshot is not None
    assert apply_snapshot["status"] == "running"
    assert len(apply_snapshot["completed_steps"]) == 1

    await bus.emit(
        ProgressEvent(
            session_id="render-after-apply",
            step="render",
            state=StepState.RUNNING,
            percent=10,
        )
    )

    snapshot = bus.get_snapshot("render-after-apply")
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert len(snapshot["completed_steps"]) == 1
    assert snapshot["completed_steps"][0]["name"] == "apply"
    assert snapshot["current_step"]["name"] == "render"
    assert snapshot["overall_progress"]["percent"] == 95
