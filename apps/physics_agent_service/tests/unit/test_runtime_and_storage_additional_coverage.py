# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from ...service import utils as utils_module
from ...service.routers import sessions_router
from ...service.runtime import registry as registry_module
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry, JobReservation
from ...service.storage.local_store import LocalSessionStore
from ...service.utils import AccessLogFilter, get_version
from ...service.workers.executor import _extract_stats_from_result


async def _sleep_forever() -> None:
    await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_job_registry_reservation_register_cancel_and_task_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("reserved")
    assert reservation.session_id == "reserved"
    assert registry.get_task("reserved") is None
    assert registry.is_running("reserved") is True
    await reservation.release()
    assert registry.is_running("reserved") is False

    async def fail_start(self: JobReservation, coro) -> None:
        coro.close()
        raise RuntimeError("start failed")

    monkeypatch.setattr(JobReservation, "start", fail_start)
    with pytest.raises(RuntimeError, match="start failed"):
        await registry.register("start-fail", _sleep_forever())
    assert registry.is_running("start-fail") is False

    done_task = asyncio.create_task(asyncio.sleep(0))
    await done_task
    registry._tasks["done"] = done_task
    assert await registry.cancel("done") is False

    pending_task = asyncio.create_task(_sleep_forever())
    registry._tasks["pending"] = pending_task
    assert registry.get_task("pending") is pending_task
    assert registry.is_running("pending") is True
    pending_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_task


@pytest.mark.asyncio
async def test_job_registry_cancel_returns_while_worker_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    started = asyncio.Event()
    release_worker = asyncio.Event()

    async def draining_worker() -> None:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release_worker.wait()

    await registry.register("draining", draining_worker())
    await started.wait()
    monkeypatch.setattr(registry_module, "_CANCEL_WAIT_TIMEOUT_SECONDS", 0.01)

    assert await registry.cancel("draining") is True
    assert registry.is_running("draining") is True

    task = registry.get_task("draining")
    assert task is not None
    quiescence = asyncio.create_task(registry.wait_for_quiescence("draining"))
    await asyncio.sleep(0)
    assert quiescence.done() is False
    release_worker.set()
    await quiescence
    await task
    assert registry.is_running("draining") is False


@pytest.mark.asyncio
async def test_event_bus_emit_handles_no_snapshot_after_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    monkeypatch.setattr(bus, "_apply_event_to_state", lambda _event: None)
    event = ProgressEvent(session_id="sid", step="predict", state=StepState.RUNNING)
    await bus.emit(event)
    assert event.overall_percent == 0


@pytest.mark.asyncio
async def test_event_bus_cancelling_state_wins_over_late_step_completion() -> None:
    bus = EventBus()
    session_id = "draining"
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_physics",
            state=StepState.RUNNING,
        )
    )
    await bus.mark_cancelling(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_physics",
            state=StepState.COMPLETED,
            percent=100,
        )
    )

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelling"

    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="apply_physics",
            state=StepState.CANCELLED,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True},
        )
    )
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    assert snapshot["current_step"] is None
    assert snapshot["completed_at"]


@pytest.mark.asyncio
async def test_local_store_edges(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    assert await store.list_sessions() == []
    assert await store.list_keys("missing") == []

    sid = "sid"
    await store.init_session(sid)
    await store.put_bytes(sid, "cache/a.txt", b"a")
    assert await store.sync_to_local(sid, str(store._session_dir(sid))) == 0
    assert await store.sync_from_local(sid, str(store._session_dir(sid))) == 0
    assert await store.cleanup_stale_local_sessions(str(tmp_path), max_age_hours=0) == 0


def test_access_log_filter_and_version_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        utils_module,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("not package")),
    )
    assert get_version() == "0.0.1-dev"

    access_filter = AccessLogFilter()
    assert access_filter.filter(
        logging.LogRecord("x", 20, "", 1, "GET /ready", (), None)
    )
    assert not access_filter.filter(
        logging.LogRecord("x", 20, "", 1, "GET /health", (), None)
    )
    assert not access_filter.filter(
        logging.LogRecord("x", 20, "", 1, "GET /metrics", (), None)
    )


def test_executor_extract_stats_prepare_dataset_fallback() -> None:
    result = type(
        "Result",
        (),
        {
            "step_results": {},
            "raw_result": {
                "build_dataset_prepare_dataset_result": {"num_entries": 7},
            },
        },
    )()
    stats = _extract_stats_from_result(result, None)
    assert stats["prims_processed"] == 7


@pytest.mark.asyncio
async def test_sessions_router_missing_and_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        async def get_session_metadata(self, _session_id: str):
            return None

        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_session(self, _session_id: str) -> bool:
            return False

    class Registry:
        def is_running(self, _session_id: str) -> bool:
            return False

    sessions_router.set_session_manager(Manager())
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: Registry())

    with pytest.raises(sessions_router.HTTPException) as missing:
        await sessions_router.get_session("sid")
    assert missing.value.status_code == 404

    with pytest.raises(sessions_router.HTTPException) as failed:
        await sessions_router.delete_session("sid")
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_sessions_delete_waits_for_job_quiescence_before_deleting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    class Manager:
        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_session(self, _session_id: str) -> bool:
            operations.append("delete")
            return True

    class Registry:
        def is_running(self, _session_id: str) -> bool:
            return True

        async def cancel(self, _session_id: str) -> bool:
            operations.append("cancel")
            return True

        async def wait_for_quiescence(self, _session_id: str) -> None:
            operations.append("quiescent")

    sessions_router.set_session_manager(Manager())
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: Registry())

    assert await sessions_router.delete_session("sid") is None
    assert operations == ["cancel", "quiescent", "delete"]
