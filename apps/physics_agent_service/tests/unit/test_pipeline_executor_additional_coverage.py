# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import builtins
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.workers import executor


class _Manager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        self.sync_calls: list[str] = []
        self.fail_sync = False
        self.update_calls = 0
        self.update_failures_remaining = 0
        self.cancel_next_update = False
        self.cancel_next_sync = False
        self.cancelled = False
        self.terminal_claim: str | None = None

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def get_session_metadata(self, _session_id: str):
        return dict(self.metadata)

    async def update_session(self, _session_id: str, updates: dict) -> None:
        self.update_calls += 1
        if self.cancel_next_update:
            self.cancel_next_update = False
            raise asyncio.CancelledError
        if self.update_failures_remaining:
            self.update_failures_remaining -= 1
            raise RuntimeError("metadata write failed")
        self.metadata.update(updates)

    async def sync_to_store(self, _session_id: str, *, prefix: str = "") -> int:
        if self.cancel_next_sync:
            self.cancel_next_sync = False
            raise asyncio.CancelledError
        if self.fail_sync:
            raise RuntimeError("sync failed")
        self.sync_calls.append(prefix)
        return 1

    async def is_cancelled(self, _session_id: str) -> bool:
        return self.cancelled

    async def claim_pipeline_terminal_state(
        self,
        _session_id: str,
        status: str,
    ) -> str:
        if self.terminal_claim is None:
            self.terminal_claim = status
        return self.terminal_claim


@pytest.mark.asyncio
async def test_execute_pipeline_success_failure_and_sync_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "physics-service-result-credential-713"
    caplog.set_level(logging.DEBUG, logger=executor.__name__)
    manager = _Manager(tmp_path)
    session_id = "pipeline"
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
            percent=10,
        )
    )

    async def good_pipeline(params):
        assert params.only_steps == ["predict"]
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["predict"],
            step_results={
                "predict": {
                    "predictions_count": 2,
                    "diagnostics": {"api_key": sentinel},
                }
            },
            raw_result={
                "build_dataset_usd_result": {"num_prims": 3, "num_images": 4},
                "config_dict": {"api_key": sentinel},
            },
        )

    monkeypatch.setattr(executor, "arun_pipeline", good_pipeline)
    await executor.execute_pipeline_async(
        session_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["predict"],
    )

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["results"]["predictions_made"] == 2
    assert sentinel not in caplog.text
    assert sentinel not in repr(manager.metadata)
    assert manager.sync_calls == [
        "cache/predictions/",
        "cache/dataset/dataset.jsonl",
        "cache/physics/",
    ]

    manager.fail_sync = True
    await executor.execute_pipeline_async(
        session_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["predict"],
    )
    assert manager.metadata["status"] == "completed"

    async def bad_pipeline(_params):
        return SimpleNamespace(
            success=False,
            error="sensitive-result-detail-713",
            completed_steps=[],
            step_results={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", bad_pipeline)
    failed_manager = _Manager(tmp_path / "bad")
    with pytest.raises(RuntimeError, match="physics_pipeline_execution_failed"):
        await executor.execute_pipeline_async(
            "bad",
            {"project": {"name": "test"}},
            failed_manager,
        )
    assert failed_manager.metadata["status"] == "failed"
    assert failed_manager.metadata["error"] == "physics_pipeline_execution_failed"
    assert failed_manager.metadata["failed_step"] == "pipeline"
    assert failed_manager.metadata["can_cancel"] is False
    assert failed_manager.metadata["completed_at"]
    assert failed_manager.metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "physics_pipeline_execution_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "sensitive-result-detail-713" not in caplog.text


@pytest.mark.asyncio
async def test_pipeline_exception_persists_failure_and_publishes_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "raised"
    manager = _Manager(tmp_path)
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="build_dataset_usd",
            state=StepState.RUNNING,
            percent=10,
        )
    )

    async def raised_pipeline(_params):
        raise RuntimeError("sensitive-exception-detail-713")

    monkeypatch.setattr(executor, "arun_pipeline", raised_pipeline)
    caplog.set_level(logging.ERROR, logger=executor.__name__)

    with pytest.raises(RuntimeError, match="physics_pipeline_execution_failed"):
        await executor.execute_pipeline_async(
            session_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert "sensitive-exception-detail-713" not in caplog.text
    assert "physics_pipeline_execution_failed" in caplog.text
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "pipeline"
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "physics_pipeline_execution_failed"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_pipeline_cancellation_persists_and_publishes_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "cancelled"
    manager = _Manager(tmp_path)
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
            percent=25,
        )
    )

    async def cancelled_pipeline(_params):
        raise asyncio.CancelledError

    monkeypatch.setattr(executor, "arun_pipeline", cancelled_pipeline)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(
            session_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["can_cancel"] is False
    assert manager.metadata["cancelled_at"]
    assert manager.metadata["completed_at"]
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["completion_metadata", "artifact_sync"])
async def test_pipeline_cancellation_covers_post_pipeline_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_phase: str,
) -> None:
    session_id = f"cancel-{cancel_phase}"
    manager = _Manager(tmp_path)

    async def completed_pipeline(_params):
        if cancel_phase == "completion_metadata":
            manager.cancel_next_update = True
        else:
            manager.cancel_next_sync = True
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["predict"],
            step_results={"predict": {"predictions_count": 1}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", completed_pipeline)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(
            session_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["can_cancel"] is False
    assert manager.metadata["completed_at"]


@pytest.mark.asyncio
async def test_pipeline_cancellation_waits_for_pipeline_worker_to_quiesce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_worker = asyncio.Event()

    async def cooperative_pipeline(params):
        started.set()
        while not params.cancel_event.is_set():
            await asyncio.sleep(0)
        cancellation_seen.set()
        await release_worker.wait()
        return SimpleNamespace(
            success=False,
            error="cancelled worker result",
            completed_steps=[],
            step_results={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", cooperative_pipeline)
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            "cooperative-cancel",
            {"project": {"name": "test"}},
            manager,
        )
    )
    await started.wait()

    task.cancel()
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)

    assert not task.done()
    assert manager.metadata["status"] == "pending"

    release_worker.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["completed_at"]


@pytest.mark.asyncio
async def test_cross_instance_cancel_marker_reaches_pipeline_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_worker = asyncio.Event()
    bus = get_event_bus()
    bus.cleanup_session("remote-cancel")
    await bus.emit(
        ProgressEvent(
            session_id="remote-cancel",
            step="build_dataset_usd",
            state=StepState.RUNNING,
        )
    )

    async def cooperative_pipeline(params):
        started.set()
        while not params.cancel_event.is_set():
            await asyncio.sleep(0)
        cancellation_seen.set()
        await release_worker.wait()
        return SimpleNamespace(
            success=False,
            cancelled=True,
            error="Pipeline cancelled",
            completed_steps=["build_dataset_usd"],
            step_results={"build_dataset_usd": {"num_prims": 3}},
        )

    monkeypatch.setattr(executor, "arun_pipeline", cooperative_pipeline)
    monkeypatch.setattr(executor, "_CANCELLATION_POLL_INTERVAL_SECONDS", 0)
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            "remote-cancel",
            {"project": {"name": "test"}},
            manager,
        )
    )
    await started.wait()

    manager.cancelled = True
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    snapshot = bus.get_snapshot("remote-cancel")
    assert snapshot is not None
    assert snapshot["status"] == "cancelling"
    release_worker.set()
    await task

    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["can_cancel"] is False
    assert manager.metadata["cancelled_at"]
    assert manager.metadata["completed_at"]
    assert manager.metadata["error"] is None
    assert manager.metadata["completed_step_names"] == ["build_dataset_usd"]
    assert manager.metadata["partial_results"] == {
        "build_dataset_usd": {"num_prims": 3}
    }
    snapshot = bus.get_snapshot("remote-cancel")
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    bus.cleanup_session("remote-cancel")


@pytest.mark.asyncio
async def test_completion_claim_rejects_cancel_during_terminal_metadata_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion_update_started = asyncio.Event()
    allow_completion_update = asyncio.Event()

    class BlockingManager(_Manager):
        async def update_session(self, session_id: str, updates: dict) -> None:
            if updates.get("status") == "completed":
                completion_update_started.set()
                await allow_completion_update.wait()
            await super().update_session(session_id, updates)

    manager = BlockingManager(tmp_path)

    async def completed_pipeline(_params):
        return SimpleNamespace(
            success=True,
            cancelled=False,
            error=None,
            completed_steps=["build_dataset_usd"],
            step_results={"build_dataset_usd": {"num_prims": 3}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", completed_pipeline)
    monkeypatch.setattr(executor, "_CANCELLATION_POLL_INTERVAL_SECONDS", 0)
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            "terminal-race",
            {"project": {"name": "test"}},
            manager,
        )
    )
    await completion_update_started.wait()

    manager.cancelled = True
    await asyncio.sleep(0)
    allow_completion_update.set()
    await task

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["can_cancel"] is False


@pytest.mark.asyncio
async def test_completion_claim_rejects_cancel_during_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_sync_started = asyncio.Event()
    allow_artifact_sync = asyncio.Event()

    class BlockingManager(_Manager):
        async def sync_to_store(self, session_id: str, *, prefix: str = "") -> int:
            if not artifact_sync_started.is_set():
                artifact_sync_started.set()
                await allow_artifact_sync.wait()
            return await super().sync_to_store(session_id, prefix=prefix)

    manager = BlockingManager(tmp_path)

    async def completed_pipeline(_params):
        return SimpleNamespace(
            success=True,
            cancelled=False,
            error=None,
            completed_steps=["build_dataset_usd"],
            step_results={"build_dataset_usd": {"num_prims": 3}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", completed_pipeline)
    monkeypatch.setattr(executor, "_CANCELLATION_POLL_INTERVAL_SECONDS", 0)
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            "artifact-race",
            {"project": {"name": "test"}},
            manager,
        )
    )
    await artifact_sync_started.wait()

    manager.cancelled = True
    allow_artifact_sync.set()
    await task

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["can_cancel"] is False


@pytest.mark.asyncio
async def test_failure_metadata_is_retried_before_event_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "persistence-failure"
    manager = _Manager(tmp_path)
    manager.update_failures_remaining = executor._TERMINAL_PERSIST_ATTEMPTS
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
        )
    )

    monkeypatch.setattr(executor, "_TERMINAL_PERSIST_RETRY_DELAY_SECONDS", 0)
    diagnostic = executor._pipeline_failure_diagnostic()
    with pytest.raises(
        RuntimeError,
        match="physics_pipeline_failure_metadata_failed",
    ):
        await executor._mark_failed(
            manager,
            session_id,
            diagnostic,
            "predict",
        )

    assert manager.update_calls == executor._TERMINAL_PERSIST_ATTEMPTS
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_failure_metadata_retry_succeeds_before_event_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "persistence-retry"
    manager = _Manager(tmp_path)
    manager.update_failures_remaining = executor._TERMINAL_PERSIST_ATTEMPTS - 1
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
        )
    )

    monkeypatch.setattr(executor, "_TERMINAL_PERSIST_RETRY_DELAY_SECONDS", 0)
    await executor._mark_failed(
        manager,
        session_id,
        executor._pipeline_failure_diagnostic(),
        "predict",
    )

    assert manager.update_calls == executor._TERMINAL_PERSIST_ATTEMPTS
    assert manager.metadata["status"] == "failed"
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_listener_failure_waits_for_durable_terminal_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "listener-persistence-failure"
    manager = _Manager(tmp_path)
    manager.update_failures_remaining = executor._TERMINAL_PERSIST_ATTEMPTS
    bus = get_event_bus()
    bus.cleanup_session(session_id)

    async def failed_pipeline(params):
        params.event_listener.event(
            "step.started",
            {"step_name": "predict"},
        )
        params.event_listener.event(
            "step.failed",
            {"step_name": "predict", "error": "sensitive-step-detail"},
        )
        params.event_listener.event(
            "workflow.failed",
            {"error": "sensitive-workflow-detail"},
        )
        await asyncio.sleep(0)
        return SimpleNamespace(
            success=False,
            error="sensitive-result-detail",
            completed_steps=[],
            step_results={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", failed_pipeline)
    monkeypatch.setattr(executor, "_TERMINAL_PERSIST_RETRY_DELAY_SECONDS", 0)

    with pytest.raises(
        RuntimeError,
        match="physics_pipeline_failure_metadata_failed",
    ):
        await executor.execute_pipeline_async(
            session_id,
            {"project": {"name": "test"}},
            manager,
        )
    await asyncio.sleep(0)

    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_pipeline_cancellation_preserves_cancelled_error_when_store_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.update_failures_remaining = executor._TERMINAL_PERSIST_ATTEMPTS

    async def cancelled_pipeline(_params):
        raise asyncio.CancelledError

    monkeypatch.setattr(executor, "arun_pipeline", cancelled_pipeline)
    monkeypatch.setattr(executor, "_TERMINAL_PERSIST_RETRY_DELAY_SECONDS", 0)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(
            "cancel-store-failure",
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.update_calls == executor._TERMINAL_PERSIST_ATTEMPTS


@pytest.mark.asyncio
async def test_failed_pipeline_persists_safe_partial_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)

    async def partial_pipeline(params):
        params.event_listener.event(
            "step.started",
            {"step_name": "build_dataset_usd"},
        )
        params.event_listener.event(
            "step.completed",
            {"step_name": "build_dataset_usd"},
        )
        params.event_listener.event(
            "task.started",
            {"task_name": "VLMInference"},
        )
        await asyncio.sleep(0)
        return SimpleNamespace(
            success=False,
            error="Pipeline failed",
            completed_steps=["build_dataset_usd"],
            step_results={"build_dataset_usd": {"num_prims": 3}},
        )

    monkeypatch.setattr(executor, "arun_pipeline", partial_pipeline)

    with pytest.raises(RuntimeError, match="physics_pipeline_execution_failed"):
        await executor.execute_pipeline_async(
            "partial-failure",
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["completed_step_names"] == ["build_dataset_usd"]
    assert manager.metadata["failed_step"] == "predict"
    assert manager.metadata["partial_results"] == {
        "build_dataset_usd": {"num_prims": 3}
    }


@pytest.mark.asyncio
async def test_failure_event_error_does_not_undo_persisted_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    manager.metadata["partial_results"] = {"stale": True}

    class _FailingEventBus:
        def get_snapshot(self, _session_id: str) -> dict:
            return {"status": "running"}

        async def emit(self, _event: ProgressEvent) -> None:
            raise RuntimeError("event publish failed")

    monkeypatch.setattr(executor, "get_event_bus", _FailingEventBus)
    caplog.set_level(logging.ERROR, logger=executor.__name__)

    await executor._mark_failed(
        manager,
        "event-failure",
        executor._pipeline_failure_diagnostic(),
        "predict",
    )

    assert manager.metadata["status"] == "failed"
    assert manager.metadata["partial_results"] is None
    assert "physics_pipeline_failure_event_failed" in caplog.text


def test_pipeline_authentication_failure_uses_distinct_value_free_code() -> None:
    diagnostic = executor._pipeline_failure_diagnostic(
        executor.MODEL_AUTHENTICATION_FAILURE_MESSAGE
    )

    assert diagnostic.code == "physics_pipeline_model_authentication_failed"


def test_pipeline_stats_file_fallbacks_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "sid"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    image = session_dir / "cache" / "dataset" / "renders" / "img.png"
    dataset.parent.mkdir(parents=True)
    predictions.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    dataset.write_text("{}\n{}\n", encoding="utf-8")
    predictions.write_text("{}\n\n{}\n", encoding="utf-8")
    image.write_bytes(b"png")

    result = SimpleNamespace(
        step_results={},
        raw_result={"dataset_info": {"num_entries": 5}},
    )
    assert (
        executor._extract_stats_from_result(result, session_dir)["prims_processed"] == 5
    )

    result = SimpleNamespace(step_results={}, raw_result={})
    stats = executor._extract_stats_from_result(result, session_dir)
    assert stats["prims_processed"] == 2
    assert stats["images_generated"] == 1
    assert stats["predictions_made"] == 2

    real_open = builtins.open

    def fail_open(path, *args, **kwargs):
        if Path(path) in {dataset, predictions}:
            raise OSError("read failed")
        return real_open(path, *args, **kwargs)

    real_glob = Path.glob

    def fail_glob(self: Path, pattern: str):
        if self == dataset.parent:
            raise OSError("glob failed")
        return real_glob(self, pattern)

    monkeypatch.setattr(builtins, "open", fail_open)
    monkeypatch.setattr(Path, "glob", fail_glob)
    executor._count_stats_from_files(
        session_dir,
        {"prims_processed": 0, "images_generated": 0, "predictions_made": 0},
    )
