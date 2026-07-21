# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for /sessions/{sid} ↔ /pipeline/{sid}/status agreement.

Reading /sessions/{sid} straight from disk previously reported
current_step=None, completed_steps=[], elapsed_seconds=0, can_cancel=true for
an actively-running session because session.json is initialized with frozen
defaults and only "status" gets re-persisted on terminal transitions. The
sessions read endpoints must overlay the EventBus snapshot so all three
documented session-state read endpoints (/sessions list, /sessions/{sid},
/pipeline/{sid}/status) agree on every observable field.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from ...service.routers import pipeline_router, sessions_router
from ...service.runtime import bus as bus_module
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session.manager import SessionManager


def _wire_service_state(tmp_path: Path) -> tuple[SessionManager, bus_module.EventBus]:
    manager = SessionManager(tmp_path)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)

    bus_module._event_bus = None
    bus = bus_module.init_event_bus(manager)
    return manager, bus


async def _drive_progress(bus: bus_module.EventBus, session_id: str) -> None:
    """Run the session forward to mid-pipeline (generate_prompts in progress)."""
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="prepare_uvs", state=StepState.RUNNING
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="prepare_uvs", state=StepState.COMPLETED
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="discover_materials", state=StepState.RUNNING
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="discover_materials", state=StepState.COMPLETED
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="generate_prompts", state=StepState.RUNNING
        )
    )


async def test_get_session_overlays_active_progress_from_event_bus(
    tmp_path: Path,
) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "active-progress"
    manager.create_session(session_id)

    await _drive_progress(bus, session_id)

    detail = await sessions_router.get_session(session_id)
    pipeline_status = await pipeline_router.get_pipeline_status(session_id)

    assert detail.status == "running" == pipeline_status.status
    assert detail.current_step is not None
    assert detail.current_step.name == "generate_prompts"
    assert pipeline_status.current_step.name == "generate_prompts"

    completed_names = [s.name for s in detail.completed_steps]
    assert completed_names == ["prepare_uvs", "discover_materials"]
    assert [s.name for s in pipeline_status.completed_steps] == completed_names

    assert detail.can_cancel is True
    assert pipeline_status.can_cancel is True
    assert detail.elapsed_seconds >= 0
    # Both endpoints share the same percent / current_step counter from the
    # bus snapshot.
    assert detail.overall_progress is not None
    for key in ("percent", "current_step", "total_steps"):
        assert (
            getattr(detail.overall_progress, key)
            == pipeline_status.overall_progress.model_dump()[key]
        )


async def test_get_session_can_cancel_false_after_terminal_status(
    tmp_path: Path,
) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "terminal-cancel-flag"
    manager.create_session(session_id)

    await _drive_progress(bus, session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_prompts",
            state=StepState.CANCELLED,
        )
    )

    detail = await sessions_router.get_session(session_id)
    assert detail.status == "cancelled"
    assert detail.can_cancel is False


async def test_get_session_falls_back_to_disk_when_no_snapshot(
    tmp_path: Path,
) -> None:
    """Sessions whose worker never registered (or whose snapshot was cleared
    by /regenerate) still respond from disk metadata with sensible defaults."""
    manager, _bus = _wire_service_state(tmp_path)
    session_id = "no-bus-snapshot"
    manager.create_session(session_id)

    detail = await sessions_router.get_session(session_id)

    assert detail.session_id == session_id
    assert detail.status == "pending"
    assert detail.current_step is None
    assert detail.completed_steps == []
    assert detail.can_cancel is True


async def test_pipeline_status_uses_active_snapshot_without_disk_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "accepted-active"
    manager.create_session(session_id)
    await bus.seed_pending_session(session_id)

    class RunningRegistry:
        def is_running(self, session_id: str) -> bool:
            return True

    def fail_disk_view(*args, **kwargs):
        raise AssertionError("active status should not read disk view")

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: RunningRegistry())
    monkeypatch.setattr(sessions_router, "_build_session_view", fail_disk_view)

    pipeline_status = await pipeline_router.get_pipeline_status(session_id)

    assert pipeline_status.session_id == session_id
    assert pipeline_status.status == "pending"
    assert pipeline_status.overall_progress.percent == 0
    assert pipeline_status.can_cancel is True


async def test_get_session_surfaces_failed_step_diagnostics(
    tmp_path: Path,
) -> None:
    manager, bus = _wire_service_state(tmp_path)
    session_id = "failed-with-stats"
    manager.create_session(session_id)

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.RUNNING,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="generate_textures",
            state=StepState.FAILED,
            message="all 4 generation calls failed",
            extra={"textures_failed": 4, "errors": ["upstream timeout"]},
        )
    )

    detail = await sessions_router.get_session(session_id)
    assert detail.status == "failed"
    assert detail.failed_step == "generate_textures"
    assert detail.failed_step_stats == {
        "textures_failed": 4,
        "errors": ["upstream timeout"],
    }
    assert detail.can_cancel is False


async def test_list_sessions_overlays_live_progress(tmp_path: Path) -> None:
    manager, bus = _wire_service_state(tmp_path)

    active_id = "list-active"
    idle_id = "list-idle"
    manager.create_session(active_id)
    manager.create_session(idle_id)

    await _drive_progress(bus, active_id)

    listing = await sessions_router.list_sessions()
    by_id = {s.session_id: s for s in listing.sessions}

    assert by_id[active_id].status == "running"
    # elapsed_seconds is computed live from created_at, so any >=0 value is
    # acceptable -- the disk-only path returned 0 even after generate_prompts
    # had been running for 30s in the bug repro.
    assert by_id[active_id].elapsed_seconds >= 0
    assert by_id[active_id].updated_at is not None

    # Sessions with no bus snapshot still show up with their disk defaults.
    assert by_id[idle_id].status == "pending"


async def test_disk_terminal_status_wins_over_stale_bus_snapshot(
    tmp_path: Path,
) -> None:
    """If the executor's outer exception handler persists status="failed"
    directly via SessionManager.update_session() without emitting a FAILED
    event to the bus, the bus snapshot still carries the prior RUNNING/
    COMPLETED status. /sessions/{sid} must trust disk's terminal status
    rather than the stale bus snapshot, otherwise a failed session would
    report ``running`` and ``can_cancel: true`` while the actual run is
    finished.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "disk-failed-bus-running"
    manager.create_session(session_id)

    # Bus snapshot captures the session mid-pipeline (still RUNNING).
    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="generate_textures", state=StepState.RUNNING
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"

    # Disk gets directly written to "failed" by an outer exception handler
    # that bypassed the bus (no FAILED event emitted).
    manager.update_session(session_id, {"status": "failed", "error": "out of memory"})

    detail = await sessions_router.get_session(session_id)
    assert detail.status == "failed"
    assert detail.can_cancel is False
    assert detail.error == "out of memory"


async def test_all_three_read_endpoints_agree_under_disk_terminal_bypass(
    tmp_path: Path,
) -> None:
    """Cross-endpoint consistency under the disk-terminal-bus-running drift:
    /sessions/{sid}, /pipeline/{sid}/status, and /pipeline/{sid}/results
    must all dispatch on the same merged status. Without the shared
    ``_build_session_view`` plumbing, /pipeline/{sid}/results would
    return 202 ("still running") off disk metadata while /sessions and
    /pipeline/status report ``failed`` -- the very kind of cross-endpoint
    drift this MR is meant to eliminate.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "cross-endpoint-disk-failed"
    manager.create_session(session_id)

    await bus.emit(
        ProgressEvent(
            session_id=session_id, step="generate_textures", state=StepState.RUNNING
        )
    )
    manager.update_session(session_id, {"status": "failed", "error": "oom"})

    detail = await sessions_router.get_session(session_id)
    pipeline_status = await pipeline_router.get_pipeline_status(session_id)
    results = await pipeline_router.get_pipeline_results(session_id)

    assert detail.status == "failed" == pipeline_status.status == results.status
    assert detail.can_cancel is False is pipeline_status.can_cancel


async def test_pipeline_results_exposes_manifest_url_and_artifact_status(
    tmp_path: Path,
) -> None:
    manager, _bus = _wire_service_state(tmp_path)
    session_id = "completed-with-manifest"
    manager.create_session(session_id)
    manager.update_session(
        session_id,
        {
            "status": "completed",
            "results": {
                "manifest_available": True,
                "manifest_path": "cache/artifacts_manifest.json",
                "package_status": "succeeded",
                "uv_report_available": True,
            },
            "duration_seconds": 7,
            "completed_at": "2026-05-08T00:00:00+00:00",
        },
    )

    results = await pipeline_router.get_pipeline_results(session_id)

    assert results.status == "completed"
    assert results.download_urls["manifest"] == (f"/artifacts/{session_id}/manifest")
    assert results.stats["manifest_available"] is True
    assert results.stats["package_status"] == "succeeded"
    assert results.stats["uv_report_available"] is True


async def test_get_session_404_after_delete(tmp_path: Path) -> None:
    """Deleting the on-disk dir must 404 even if a stale snapshot is in the bus.
    Pairs with test_delete_session_lifecycle.py: session_exists() is the
    authoritative gate -- the bus snapshot alone never counts as 'exists'.
    """
    manager, bus = _wire_service_state(tmp_path)
    session_id = "stale-bus-after-delete"
    manager.create_session(session_id)
    await _drive_progress(bus, session_id)
    manager.delete_session(session_id)

    assert bus.get_snapshot(session_id) is not None

    with pytest.raises(HTTPException) as exc_info:
        await sessions_router.get_session(session_id)
    assert exc_info.value.status_code == 404
