# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ...service.runtime import bus as bus_module
from ...service.runtime.bus import EventBus, get_event_bus, init_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry


async def _sleep_forever() -> None:
    await asyncio.Event().wait()


def _raise_callback() -> None:
    raise RuntimeError("callback failed")


@pytest.mark.asyncio
async def test_job_registry_missing_done_prestart_and_heartbeat_edges() -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)

    assert await registry.cancel("missing") is False
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    registry._tasks["done"] = done
    assert await registry.cancel("done") is False

    async def never_started_job() -> None:
        return None

    await registry.register(
        "prestart",
        never_started_job(),
        on_never_started=_raise_callback,
        on_finished=_raise_callback,
    )
    task = registry.get_task("prestart")
    assert task is not None
    task.cancel()
    await asyncio.sleep(0)
    assert task.cancelled()
    registry._tasks.pop("prestart", None)

    await registry._semaphore.acquire()
    await registry.register(
        "queued",
        _sleep_forever(),
        on_never_started=_raise_callback,
        on_finished=_raise_callback,
    )
    await asyncio.sleep(0)
    queued_task = registry.get_task("queued")
    assert queued_task is not None
    queued_task.cancel()
    registry._semaphore.release()
    with pytest.raises(asyncio.CancelledError):
        await queued_task

    await registry._run_queued_heartbeat("hb-error", _raise_callback)

    heartbeat_calls = 0

    async def heartbeat() -> None:
        nonlocal heartbeat_calls
        heartbeat_calls += 1

    await registry._semaphore.acquire()
    acquire = asyncio.create_task(
        registry._acquire_slot_with_queued_heartbeat("hb", heartbeat, 0.01)
    )
    await asyncio.sleep(0.03)
    registry._semaphore.release()
    await acquire
    assert heartbeat_calls >= 1


@pytest.mark.asyncio
async def test_job_registry_cancel_swallows_task_failure_during_cancel() -> None:
    registry = JobRegistry(max_concurrent=1, cancel_wait_seconds=0.01)
    started = asyncio.Event()

    async def failing_cancel_job() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise RuntimeError("cleanup failed") from exc

    await registry.register("fails-on-cancel", failing_cancel_job())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await registry.cancel("fails-on-cancel") is True


@pytest.mark.asyncio
async def test_event_bus_clear_session_state_and_no_manager_noops() -> None:
    bus = EventBus()
    queue = bus.get_queue("sid")
    await queue.put(
        ProgressEvent(session_id="sid", step="render", state=StepState.FAILED)
    )
    task = asyncio.create_task(asyncio.sleep(60))
    bus._live_metadata_tasks["sid"] = task

    bus.clear_session_state("sid")

    assert queue.empty()
    assert task.cancelled() or task.cancelling()
    await bus._persist_status("sid", "running")
    await bus._persist_live_metadata("sid", {"status": "running"})
    await bus._save_event_to_log(
        ProgressEvent(session_id="sid", step="render", state=StepState.RUNNING)
    )
    assert await bus.cleanup_orphaned_sessions() == []


@pytest.mark.asyncio
async def test_event_bus_shared_manager_edges_with_low_loop_uptime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        def __init__(self) -> None:
            self.exists = True
            self.shared = True
            self.metadata = {"status": "completed"}
            self.updated: list[dict[str, Any]] = []
            self.events: list[dict[str, Any]] = []
            self.heartbeat_calls = 0

        def uses_shared_store(self) -> bool:
            return self.shared

        def session_exists(self, _session_id: str) -> bool:
            return self.exists

        def heartbeat_worker(self, _session_id: str, owner_token: str | None) -> bool:
            self.heartbeat_calls += 1
            return owner_token == "owner"

        def get_worker_reservation_owner_token(self, _session_id: str) -> str | None:
            return "owner"

        def get_session_metadata(self, _session_id: str) -> dict[str, Any]:
            return self.metadata

        def update_session(
            self,
            _session_id: str,
            updates: dict[str, Any],
            *,
            update_index: bool = True,
        ) -> None:
            self.updated.append({"updates": updates, "update_index": update_index})

        def append_event(self, _session_id: str, event: dict[str, Any]) -> None:
            self.events.append(event)

    manager = Manager()
    bus = EventBus(manager)
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: 1.0)
    assert await bus._session_exists_for_emit("sid") is True
    assert await bus._session_exists_for_emit("sid") is True

    await bus._heartbeat_worker_if_due("sid")
    await bus._heartbeat_worker_if_due("sid")
    assert manager.heartbeat_calls == 1

    await bus._persist_status("sid", "running")
    assert manager.updated == []

    await bus._persist_live_metadata("sid", {"status": "running"})
    assert manager.updated == [
        {"updates": {"status": "running"}, "update_index": False}
    ]

    await bus._save_event_to_log(
        ProgressEvent(session_id="sid", step="render", state=StepState.RUNNING)
    )
    assert manager.events

    manager.exists = False
    bus._state["gone"] = {"session_id": "gone"}
    bus.get_queue("gone")
    assert await bus.cleanup_orphaned_sessions() == ["gone"]


@pytest.mark.asyncio
async def test_event_bus_drops_deleted_session_events() -> None:
    bus = EventBus()
    bus._deleted_sessions.add("deleted")
    await bus.emit(
        ProgressEvent(session_id="deleted", step="render", state=StepState.RUNNING)
    )
    assert bus.get_snapshot("deleted") is None


@pytest.mark.asyncio
async def test_event_bus_additional_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SharedMissingManager:
        def uses_shared_store(self) -> bool:
            return True

        def session_exists(self, _session_id: str) -> bool:
            return False

    bus = EventBus()
    assert bus._uses_shared_store() is False
    await bus._heartbeat_worker_if_due("sid")

    shared_bus = EventBus(SharedMissingManager())
    assert await shared_bus._session_exists_for_emit("gone") is False

    deleted_bus = EventBus()
    deleted_bus._live_metadata_pending["sid"] = {"status": "running"}
    deleted_bus._deleted_sessions.add("sid")
    await deleted_bus._flush_live_metadata_after_delay("sid", 0)
    assert deleted_bus._live_metadata_tasks == {}

    empty_bus = EventBus()
    await empty_bus._flush_live_metadata_after_delay("sid", 0)

    no_state_bus = EventBus()

    def no_apply(
        event: ProgressEvent,
        pending_persists: list[tuple[str, str]],
    ) -> None:
        return None

    monkeypatch.setattr(no_state_bus, "_apply_event_to_state", no_apply)
    await no_state_bus.emit(
        ProgressEvent(session_id="sid", step="render", state=StepState.RUNNING)
    )
    event = await no_state_bus.get_queue("sid").get()
    assert event.overall_percent == 0

    terminal_bus = EventBus()
    pending: list[tuple[str, str]] = []
    terminal_bus._state["sid"] = {
        "session_id": "sid",
        "status": "completed",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_steps": [],
        "overall_progress": {},
        "step_timings": {},
    }
    terminal_bus._apply_event_to_state(
        ProgressEvent(session_id="sid", step="render", state=StepState.CANCELLING),
        pending,
    )
    assert terminal_bus._state["sid"]["status"] == "completed"
    assert pending == []

    task = asyncio.create_task(asyncio.sleep(60))
    cleanup_bus = EventBus()
    cleanup_bus._live_metadata_tasks["sid"] = task
    await cleanup_bus.cleanup_session("sid")
    assert task.cancelled() or task.cancelling()

    old_bus = bus_module._event_bus
    try:
        bus_module._event_bus = None
        assert get_event_bus() is get_event_bus()
        manager = object()
        assert init_event_bus(manager)._session_manager is manager
    finally:
        bus_module._event_bus = old_bus
