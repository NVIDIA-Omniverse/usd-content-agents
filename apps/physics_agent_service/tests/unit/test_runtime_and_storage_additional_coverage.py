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
from ...service.session.manager import SessionStoreDeletionError
from ...service.storage import local_store as local_store_module
from ...service.storage.local_store import LocalSessionStore
from ...service.utils import AccessLogFilter, derive_completed_step_names, get_version
from ...service.workers import refine_executor
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

    async def fail_start(
        self: JobReservation,
        coro,
        *,
        wait_heartbeat=None,
    ) -> None:
        coro.close()
        if hasattr(wait_heartbeat, "close"):
            wait_heartbeat.close()
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
async def test_job_registry_runs_heartbeat_while_job_waits_for_capacity() -> None:
    registry = JobRegistry(max_concurrent=0)
    heartbeat_started = asyncio.Event()
    heartbeat_stopped = asyncio.Event()
    worker_started = asyncio.Event()

    async def queued_worker() -> None:
        worker_started.set()

    async def waiting_heartbeat() -> None:
        heartbeat_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            heartbeat_stopped.set()

    await registry.register(
        "queued",
        queued_worker(),
        wait_heartbeat=waiting_heartbeat(),
    )

    await asyncio.wait_for(heartbeat_started.wait(), timeout=1)
    assert worker_started.is_set() is False
    assert await registry.cancel("queued") is True
    await asyncio.wait_for(heartbeat_stopped.wait(), timeout=1)
    assert registry.is_running("queued") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_error", [None, RuntimeError("renewal failed")])
async def test_job_registry_never_starts_queued_work_after_heartbeat_stops(
    heartbeat_error: RuntimeError | None,
) -> None:
    registry = JobRegistry(max_concurrent=0)
    heartbeat_started = asyncio.Event()
    stop_heartbeat = asyncio.Event()
    worker_started = asyncio.Event()

    async def queued_worker() -> None:
        worker_started.set()

    async def waiting_heartbeat() -> None:
        heartbeat_started.set()
        await stop_heartbeat.wait()
        if heartbeat_error is not None:
            raise heartbeat_error

    await registry.register(
        "lease-lost",
        queued_worker(),
        wait_heartbeat=waiting_heartbeat(),
    )
    await asyncio.wait_for(heartbeat_started.wait(), timeout=1)
    task = registry.get_task("lease-lost")
    assert task is not None
    stop_heartbeat.set()
    await task
    assert worker_started.is_set() is False
    assert registry.is_running("lease-lost") is False
    assert registry._semaphore._value == 0


@pytest.mark.asyncio
async def test_job_registry_discards_worker_when_capacity_and_lease_end_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    worker_started = asyncio.Event()

    async def queued_worker() -> None:
        worker_started.set()

    async def both_gate_tasks_done(tasks):
        await asyncio.gather(*tasks)
        return set(tasks), set()

    monkeypatch.setattr(registry_module, "_wait_for_queue_gate", both_gate_tasks_done)
    await registry.register(
        "simultaneous",
        queued_worker(),
        wait_heartbeat=asyncio.sleep(0),
    )
    task = registry.get_task("simultaneous")
    assert task is not None
    await task
    assert worker_started.is_set() is False
    assert registry._semaphore._value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owns_at_handoff", [False, True])
async def test_job_registry_requires_callable_lease_proof_at_capacity(
    owns_at_handoff: bool,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    worker_started = asyncio.Event()
    handoff_checked = asyncio.Event()

    async def queued_worker() -> None:
        worker_started.set()

    async def lease_guard(
        session_id: str,
        *,
        capacity_ready: asyncio.Event,
    ) -> bool:
        assert session_id == "lease-proof"
        await capacity_ready.wait()
        handoff_checked.set()
        return owns_at_handoff

    await registry.register(
        "lease-proof",
        queued_worker(),
        wait_heartbeat=lease_guard,
    )
    task = registry.get_task("lease-proof")
    assert task is not None
    await task
    assert handoff_checked.is_set()
    assert worker_started.is_set() is owns_at_handoff
    assert registry._semaphore._value == 1


@pytest.mark.asyncio
async def test_job_registry_discards_capacity_winner_when_lease_proof_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    worker_started = asyncio.Event()

    async def queued_worker() -> None:
        worker_started.set()

    async def failing_lease_guard(
        _session_id: str,
        *,
        capacity_ready: asyncio.Event,
    ) -> bool:
        await capacity_ready.wait()
        raise RuntimeError("ownership read failed")

    with caplog.at_level(logging.ERROR):
        await registry.register(
            "failed-lease-proof",
            queued_worker(),
            wait_heartbeat=failing_lease_guard,
        )
        task = registry.get_task("failed-lease-proof")
        assert task is not None
        await task

    assert worker_started.is_set() is False
    assert registry.is_running("failed-lease-proof") is False
    assert registry._semaphore._value == 1
    assert "Generation lease heartbeat failed" in caplog.text


@pytest.mark.asyncio
async def test_job_registry_releases_slot_when_cancelled_after_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    worker = asyncio.sleep(0)
    heartbeat = asyncio.Event().wait()

    async def cancel_after_acquire(tasks, *, return_when):
        assert return_when is asyncio.FIRST_COMPLETED
        semaphore_task = next(
            task for task in tasks if task.get_coro() is not heartbeat
        )
        await semaphore_task
        raise asyncio.CancelledError

    monkeypatch.setattr(registry_module.asyncio, "wait", cancel_after_acquire)
    with pytest.raises(asyncio.CancelledError):
        await registry._run_with_cleanup(
            "cancelled-after-acquire",
            worker,
            wait_heartbeat=heartbeat,
        )
    assert registry._semaphore._value == 1
    assert worker.cr_frame is None


@pytest.mark.asyncio
async def test_job_registry_closes_heartbeat_on_consumed_and_duplicate_paths() -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("consumed")
    await reservation.start(asyncio.sleep(0))

    duplicate_worker = asyncio.sleep(0)
    duplicate_heartbeat = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already consumed"):
        await reservation.start(
            duplicate_worker,
            wait_heartbeat=duplicate_heartbeat,
        )
    assert duplicate_worker.cr_frame is None
    assert duplicate_heartbeat.cr_frame is None

    held = await registry.reserve("duplicate")
    rejected_worker = asyncio.sleep(0)
    rejected_heartbeat = asyncio.sleep(0)
    with pytest.raises(ValueError):
        await registry.register(
            "duplicate",
            rejected_worker,
            wait_heartbeat=rejected_heartbeat,
        )
    assert rejected_worker.cr_frame is None
    assert rejected_heartbeat.cr_frame is None
    await held.release()
    await registry.wait_for_quiescence("consumed")


@pytest.mark.asyncio
async def test_job_registry_closes_unstarted_heartbeat_when_task_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    worker = asyncio.sleep(0)
    heartbeat = asyncio.sleep(0)

    def fail_create_task(_coro):
        raise RuntimeError("task creation failed")

    monkeypatch.setattr(registry_module.asyncio, "create_task", fail_create_task)
    with pytest.raises(RuntimeError, match="task creation failed"):
        await registry._run_with_cleanup(
            "sid",
            worker,
            wait_heartbeat=heartbeat,
        )
    assert worker.cr_frame is None
    assert heartbeat.cr_frame is None


@pytest.mark.asyncio
async def test_reservation_closes_heartbeat_when_wrapper_task_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    reservation = await registry.reserve("wrapper-create-failure")
    worker = asyncio.sleep(0)
    heartbeat = asyncio.sleep(0)

    def fail_create_task(_coro):
        raise RuntimeError("wrapper task creation failed")

    monkeypatch.setattr(registry_module, "_create_job_task", fail_create_task)
    with pytest.raises(RuntimeError, match="wrapper task creation failed"):
        await reservation.start(worker, wait_heartbeat=heartbeat)
    assert worker.cr_frame is None
    assert heartbeat.cr_frame is None
    await reservation.release()


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

    await store.delete_key("missing", "cache/a.txt")
    assert await store.get_event_log("missing") == []
    assert await store.sync_to_local("missing", str(tmp_path / "dest")) == 0
    assert await store.get_event_log("../escaped") == []
    assert await store.sync_to_local("../escaped", str(tmp_path / "dest")) == 0
    assert (
        store._copy_local_snapshot(
            tmp_path / "missing-source",
            tmp_path / "copy-destination",
            "",
        )
        == 0
    )


@pytest.mark.asyncio
async def test_local_store_delete_exhausts_bounded_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    attempts = 0

    def fail_delete(*_args, **_kwargs) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("delete failed")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(local_store_module, "remove_confined_tree", fail_delete)
    monkeypatch.setattr(local_store_module.asyncio, "sleep", no_sleep)
    with pytest.raises(OSError, match="delete failed"):
        await store.delete_session("sid")
    assert attempts == 3


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
    assert derive_completed_step_names(None, {"not": "a list"}) == []


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


def test_refine_failed_state_gets_a_durable_fallback_diagnostic() -> None:
    results = {"best_score": 0.5}
    updates: dict[str, object] = {}
    diagnostic = refine_executor._ensure_failed_refine_diagnostic(
        "failed",
        None,
        updates,
        results,
    )
    assert diagnostic is not None
    assert diagnostic.code == "physics_refine_terminal_state_invalid"
    assert updates["error"] == diagnostic.code
    assert updates["partial_results"] == results
    assert (
        refine_executor._ensure_failed_refine_diagnostic(
            "completed",
            None,
            {},
            results,
        )
        is None
    )


@pytest.mark.asyncio
async def test_sessions_router_missing_and_delete_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        async def get_session_metadata(self, _session_id: str):
            return None

        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_terminal_session(self, _session_id: str) -> bool:
            return False

    class Registry:
        def is_running(self, _session_id: str) -> bool:
            return False

    sessions_router.set_session_manager(Manager())
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: Registry())

    with pytest.raises(sessions_router.HTTPException) as missing:
        await sessions_router.get_session("sid")
    assert missing.value.status_code == 404

    with pytest.raises(sessions_router.HTTPException) as refused:
        await sessions_router.delete_session("sid")
    assert refused.value.status_code == 409

    async def failed_delete(_session_id: str) -> bool:
        raise SessionStoreDeletionError("failed")

    manager = sessions_router.get_session_manager()
    manager.delete_terminal_session = failed_delete
    with pytest.raises(sessions_router.HTTPException) as failed:
        await sessions_router.delete_session("sid")
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_sessions_delete_waits_for_job_quiescence_before_deleting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[str] = []

    class Manager:
        async def get_session_metadata(self, _session_id: str):
            return None

        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_terminal_session(self, _session_id: str) -> bool:
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


@pytest.mark.asyncio
async def test_sessions_delete_rejects_active_session_owned_by_another_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def get_session_metadata(self, _session_id: str):
            return {"status": "running"}

        async def delete_terminal_session(self, _session_id: str) -> bool:
            raise AssertionError("active remote session must not be deleted")

    class Registry:
        def is_running(self, _session_id: str) -> bool:
            return False

    sessions_router.set_session_manager(Manager())
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: Registry())

    with pytest.raises(sessions_router.HTTPException) as exc:
        await sessions_router.delete_session("sid")

    assert exc.value.status_code == 409
