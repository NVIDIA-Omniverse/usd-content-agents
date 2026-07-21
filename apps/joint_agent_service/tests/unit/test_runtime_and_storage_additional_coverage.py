# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import os
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from world_understanding.utils import artifacts as artifact_utils

from ...service import utils as utils_module
from ...service.routers import sessions_router
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry
from ...service.storage.local_store import LocalSessionStore
from ...service.utils import AccessLogFilter, get_version
from ...service.workers.executor import (
    _completed_step_records,
    _extract_stats_from_result,
)


async def _sleep_forever() -> None:
    await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_job_registry_register_cancel_and_task_edges() -> None:
    registry = JobRegistry(max_concurrent=1)
    assert await registry.cancel("missing") is False

    await registry.register("registered", _sleep_forever())
    await asyncio.sleep(0)
    assert registry.get_task("registered") is not None
    assert registry.is_running("registered") is True
    assert registry.registered_count == 1
    assert await registry.cancel("registered") is True
    assert registry.is_running("registered") is False

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
async def test_job_registry_rejects_duplicate_session_without_replacing_owner() -> None:
    registry = JobRegistry(max_concurrent=1)
    started = asyncio.Event()
    release = asyncio.Event()
    cleanup_calls: list[str] = []

    async def first_run() -> None:
        started.set()
        await release.wait()

    async def cleanup() -> None:
        cleanup_calls.append("first")

    await registry.register("same-session", first_run(), on_finish=cleanup)
    await started.wait()
    owner = registry.get_task("same-session")
    rejected = _sleep_forever()
    with pytest.raises(RuntimeError, match="already registered"):
        await registry.register("same-session", rejected)

    assert rejected.cr_frame is None
    assert registry.get_task("same-session") is owner
    release.set()
    assert owner is not None
    await owner
    assert registry.get_task("same-session") is None
    assert cleanup_calls == ["first"]


@pytest.mark.asyncio
async def test_job_registry_releases_job_cancelled_before_wrapper_starts() -> None:
    registry = JobRegistry(max_concurrent=1)
    cleanup_called = asyncio.Event()

    async def cleanup() -> None:
        cleanup_called.set()

    pipeline = _sleep_forever()
    await registry.register("cancelled-early", pipeline, on_finish=cleanup)
    task = registry.get_task("cancelled-early")
    assert task is not None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(cleanup_called.wait(), timeout=1)

    assert pipeline.cr_frame is None
    assert registry.get_task("cancelled-early") is None


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
async def test_event_bus_failure_cancel_completion_and_cleanup_edges() -> None:
    bus = EventBus()

    await bus.emit(
        ProgressEvent(
            session_id="failed",
            step="predict",
            state=StepState.FAILED,
            message="boom",
        )
    )
    failed = bus.get_snapshot("failed")
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"
    assert failed["failed_step"] == "predict"

    await bus.emit(
        ProgressEvent(
            session_id="cancelled",
            step="predict",
            state=StepState.CANCELLED,
        )
    )
    cancelled = bus.get_snapshot("cancelled")
    assert cancelled["status"] == "cancelled"

    bus._state["completed"] = {
        "session_id": "completed",
        "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "current_step": {
            "name": "restore_usd",
            "display_name": "Restore USD",
            "started_at": "2026-01-01T00:00:00+00:00",
            "progress": {},
            "elapsed_seconds": 0,
        },
        "completed_steps": [],
        "overall_progress": {"current_step": 0, "total_steps": 3, "percent": 0},
        "step_timings": {},
    }
    await bus.emit(
        ProgressEvent(
            session_id="completed",
            step="restore_usd",
            state=StepState.COMPLETED,
            timestamp="2026-01-01T00:00:05+00:00",
        )
    )
    completed = bus.get_snapshot("completed")
    assert completed["overall_progress"]["total_steps"] == 8
    assert completed["overall_progress"]["current_step"] == 8

    bus.get_queue("completed")
    bus.cleanup_session("completed")
    assert bus.get_snapshot("completed") is None
    assert "completed" not in bus._queues


@pytest.mark.asyncio
async def test_local_store_edges(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    assert await store.list_sessions() == []
    assert await store.list_keys("missing") == []

    sid = "sid"
    await store.init_session(sid)
    await store.put_bytes(sid, "cache/a.txt", b"a")
    await store.put_bytes(sid, "claim", b"one")
    assert await store.compare_and_swap_bytes(sid, "claim", b"one", b"renewed")
    assert not await store.compare_and_swap_bytes(sid, "claim", b"one", b"stale")
    assert await store.compare_and_swap_bytes(sid, "claim", b"renewed", None)
    assert not await store.compare_and_swap_bytes(sid, "claim", b"renewed", None)
    assert all(not key.endswith(".cas.lock") for key in await store.list_keys(sid))
    assert await store.sync_to_local(sid, str(store._session_dir(sid))) == 0
    assert (
        await store.sync_to_local(
            sid,
            str(store._session_dir(sid)),
            overwrite=True,
        )
        == 0
    )
    assert await store.sync_from_local(sid, str(store._session_dir(sid))) == 0
    equivalent_session_path = store._session_dir(sid) / ".." / sid
    assert await store.sync_from_local(sid, str(equivalent_session_path)) == 0
    assert await store.cleanup_stale_local_sessions(str(tmp_path), max_age_hours=0) == 0

    target = tmp_path / "target"
    await store.put_bytes(sid, "other.txt", b"skip")
    legacy_temp = (
        store._session_dir(sid) / "cache" / "nested" / ".pipeline_temp" / "config.yaml"
    )
    legacy_temp.parent.mkdir(parents=True)
    legacy_temp.write_bytes(b"api_key: sentinel")
    assert "cache/nested/.pipeline_temp/config.yaml" not in await store.list_keys(sid)
    assert "cache/nested/.pipeline_temp/config.yaml" not in await store.list_keys(
        sid, prefix="cache/"
    )
    (store._session_dir(sid) / "cache" / "nested").mkdir(exist_ok=True)
    assert await store.sync_to_local(sid, str(target), prefix="cache/") == 1
    assert (target / "cache" / "a.txt").read_bytes() == b"a"
    assert not (target / "cache" / "nested" / ".pipeline_temp" / "config.yaml").exists()
    assert not (target / "other.txt").exists()
    await store.put_bytes(sid, "cache/a.txt", b"new")
    assert await store.sync_to_local(sid, str(target), prefix="cache/") == 0
    assert (target / "cache" / "a.txt").read_bytes() == b"a"
    stale_file = target / "cache" / "stale" / "old.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_bytes(b"stale")
    assert (
        await store.sync_to_local(
            sid,
            str(target),
            prefix="cache/",
            overwrite=True,
        )
        == 1
    )
    assert (target / "cache" / "a.txt").read_bytes() == b"new"
    assert not stale_file.exists()
    assert not stale_file.parent.exists()

    empty_stale = target / "empty" / "stale.txt"
    empty_stale.parent.mkdir(parents=True)
    empty_stale.write_bytes(b"stale")
    assert (
        await store.sync_to_local(
            sid,
            str(target),
            prefix="empty/",
            overwrite=True,
        )
        == 0
    )
    assert not empty_stale.exists()
    assert not empty_stale.parent.exists()

    source = tmp_path / "source"
    (source / "cache").mkdir(parents=True)
    (source / "cache" / "b.txt").write_bytes(b"b")
    source_pipeline_temp = source / "cache" / "nested" / ".pipeline_temp"
    source_pipeline_temp.mkdir(parents=True)
    (source_pipeline_temp / "config.yaml").write_bytes(b"api_key: sentinel")
    (source / "other.txt").write_bytes(b"skip")
    assert await store.sync_from_local("copied", str(source), prefix="cache/") == 1
    assert (store._session_dir("copied") / "cache" / "b.txt").read_bytes() == b"b"
    assert not (
        store._session_dir("copied")
        / "cache"
        / "nested"
        / ".pipeline_temp"
        / "config.yaml"
    ).exists()
    assert not (store._session_dir("copied") / "other.txt").exists()


@pytest.mark.asyncio
async def test_local_store_overwrite_failure_preserves_stale_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    sid = "sid"
    await store.put_bytes(sid, "cache/current.txt", b"shared-new")
    target = tmp_path / "target"
    current = target / "cache" / "current.txt"
    obsolete = target / "cache" / "obsolete.txt"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"local-old")
    obsolete.write_bytes(b"stale")

    def partial_copy(_source, destination) -> None:
        destination.write(b"partial")
        raise RuntimeError("copy interrupted")

    monkeypatch.setattr(artifact_utils.shutil, "copyfileobj", partial_copy)

    with pytest.raises(RuntimeError, match="copy interrupted"):
        await store.sync_to_local(
            sid,
            str(target),
            prefix="cache/",
            overwrite=True,
        )

    assert current.read_bytes() == b"local-old"
    assert obsolete.read_bytes() == b"stale"
    assert list(current.parent.glob(f".{current.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_local_store_upload_overwrite_failure_preserves_store_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    sid = "sid"
    await store.put_bytes(sid, "cache/current.txt", b"store-old")
    source = tmp_path / "source"
    source_file = source / "cache" / "current.txt"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"local-new")
    destination = store._session_dir(sid) / "cache" / "current.txt"

    def partial_copy(_source, target) -> None:
        target.write(b"partial")
        raise RuntimeError("upload interrupted")

    monkeypatch.setattr(artifact_utils.shutil, "copyfileobj", partial_copy)

    with pytest.raises(RuntimeError, match="upload interrupted"):
        await store.sync_from_local(
            sid,
            str(source),
            prefix="cache/",
            overwrite=True,
        )

    assert destination.read_bytes() == b"store-old"
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.asyncio
async def test_local_store_sync_rejects_links_and_special_files(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")

    upload_source = tmp_path / "upload"
    upload_cache = upload_source / "cache"
    upload_cache.mkdir(parents=True)
    upload_link = upload_cache / "link.txt"
    upload_link.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlinked session artifact"):
        await store.sync_from_local("upload", str(upload_source))
    upload_link.unlink()
    upload_fifo = upload_cache / "artifact.fifo"
    os.mkfifo(upload_fifo)
    with pytest.raises(RuntimeError, match="special session artifact"):
        await store.sync_from_local("upload", str(upload_source))
    upload_fifo.unlink()

    await store.init_session("download")
    stored_cache = store._session_dir("download") / "cache"
    stored_cache.mkdir()
    stored_link = stored_cache / "link.txt"
    stored_link.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlinked session artifact"):
        await store.sync_to_local("download", str(tmp_path / "download"))
    stored_link.unlink()
    stored_fifo = stored_cache / "artifact.fifo"
    os.mkfifo(stored_fifo)
    with pytest.raises(RuntimeError, match="special session artifact"):
        await store.sync_to_local("download", str(tmp_path / "download"))
    stored_fifo.unlink()


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


def test_executor_completed_step_records_handles_non_dict_stats() -> None:
    result = type(
        "Result",
        (),
        {
            "completed_steps": ["predict"],
            "step_results": {"predict": "not a dict"},
        },
    )()

    records = _completed_step_records(result, None)
    assert records == [
        {
            "name": "predict",
            "display_name": "Running VLM Predictions",
            "started_at": records[0]["started_at"],
            "completed_at": records[0]["completed_at"],
            "duration_seconds": 0,
            "stats": {},
        }
    ]


@pytest.mark.asyncio
async def test_sessions_router_missing_and_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        async def get_session_metadata(self, _session_id: str):
            if _session_id == "present":
                return {"session_id": "present", "status": "ready"}
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

    assert await sessions_router.get_session("present") == {
        "session_id": "present",
        "status": "ready",
    }

    with pytest.raises(sessions_router.HTTPException) as missing:
        await sessions_router.get_session("sid")
    assert missing.value.status_code == 404

    with pytest.raises(sessions_router.HTTPException) as failed:
        await sessions_router.delete_session("sid")
    assert failed.value.status_code == 500


@pytest.mark.asyncio
async def test_sessions_router_keeps_session_while_cancelled_job_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        deleted = False

        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_session(self, _session_id: str) -> bool:
            self.deleted = True
            return True

    class Registry:
        cancelled = False

        def is_running(self, _session_id: str) -> bool:
            return True

        async def cancel(self, _session_id: str) -> bool:
            self.cancelled = True
            return True

    manager = Manager()
    registry = Registry()
    sessions_router.set_session_manager(manager)
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: registry)

    with pytest.raises(sessions_router.HTTPException) as draining:
        await sessions_router.delete_session("sid")

    assert draining.value.status_code == 409
    assert registry.cancelled is True
    assert manager.deleted is False


@pytest.mark.asyncio
async def test_sessions_router_keeps_session_while_finish_callback_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Manager:
        deleted = False

        async def session_exists(self, _session_id: str) -> bool:
            return True

        async def delete_session(self, _session_id: str) -> bool:
            self.deleted = True
            return True

    session_id = "finish-callback-draining"
    finish_started = asyncio.Event()
    release_finish = asyncio.Event()

    async def cleanup() -> None:
        finish_started.set()
        await release_finish.wait()

    manager = Manager()
    registry = JobRegistry(max_concurrent=1)
    sessions_router.set_session_manager(manager)
    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: registry)
    await registry.register(session_id, asyncio.sleep(0), on_finish=cleanup)
    task = registry.get_task(session_id)
    assert task is not None
    await finish_started.wait()

    with pytest.raises(sessions_router.HTTPException) as draining:
        await sessions_router.delete_session(session_id)

    assert draining.value.status_code == 409
    assert manager.deleted is False
    release_finish.set()
    await task
