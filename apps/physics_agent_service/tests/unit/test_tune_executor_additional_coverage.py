# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import math
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.workers import tune_executor as executor


class _Manager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata: dict[str, object] = {
            "created_at": (datetime.now(UTC) - timedelta(seconds=5)).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
        }
        self.cancelled = False
        self.sync_calls: list[str] = []
        self.fail_sync = False
        self.cancel_during_sync = False
        self.cancel_sync_task = False
        self.operations: list[str] = []

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def update_session(
        self, _session_id: str, updates: dict[str, object]
    ) -> None:
        if "status" in updates:
            self.operations.append(f"status:{updates['status']}")
        self.metadata.update(updates)

    async def get_session_metadata(self, _session_id: str) -> dict[str, object]:
        return dict(self.metadata)

    async def is_cancelled(self, _session_id: str) -> bool:
        return self.cancelled

    async def sync_to_store(self, _session_id: str, *, prefix: str = "") -> int:
        self.operations.append(f"sync:{prefix}")
        if self.cancel_during_sync:
            self.cancelled = True
        if self.cancel_sync_task:
            raise asyncio.CancelledError
        if self.fail_sync:
            raise RuntimeError("sync failed")
        self.sync_calls.append(prefix)
        return 1


def _result(
    *,
    success: bool = True,
    cancelled: bool = False,
    n_trials: int = 2,
    best_params: dict[str, float] | None = None,
    best_score: object = 0.2,
    artifacts: dict | None = None,
    error: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        success=success,
        cancelled=cancelled,
        best_params=best_params if best_params is not None else {"mass_scale": 1.2},
        best_score=best_score,
        n_trials=n_trials,
        optimizer_used="botorch",
        engine_used="fake",
        artifacts=artifacts,
        error=error,
    )


def test_tune_metadata_helpers_cover_edges() -> None:
    assert executor._finite_best_score(None) is None
    assert executor._finite_best_score("bad") is None
    assert executor._finite_best_score(float("inf")) is None
    assert executor._finite_best_score("1.5") == 1.5

    result = _result(best_score=float("nan"))
    metadata = executor._tune_results_metadata(result)
    assert metadata["best_score"] is None
    assert metadata["best_params"] == {"mass_scale": 1.2}

    assert executor._has_partial_tune_results(_result(n_trials=1)) is True
    assert (
        executor._has_partial_tune_results(_result(n_trials=0, best_params={"x": 1.0}))
        is True
    )
    assert (
        executor._has_partial_tune_results(
            _result(n_trials=0, best_params={}, artifacts={"report": "x"})
        )
        is True
    )
    assert (
        executor._has_partial_tune_results(
            _result(n_trials=0, best_params={}, artifacts=None)
        )
        is False
    )


@pytest.mark.asyncio
async def test_tune_event_listener_maps_runner_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = get_event_bus()
    bus.cleanup_session("tune-events")
    listener = executor._TuneEventListener("tune-events", max_trials=2)
    listener.info("hello")
    listener.debug("hello")
    listener.warning("SENTINEL_TUNE_WARNING")
    listener.error("SENTINEL_TUNE_ERROR")

    listener.event("tune.started", {"scenario": "drop_settle"})
    listener.event(
        "tune.trial.completed",
        {"trial_index": 0, "score": 0.3, "params": {"mass_scale": 1.1}},
    )
    listener.event("tune.trial.completed", {"score": 0.1, "failed": True})
    listener.event("tune.completed", {"best_score": 0.3})
    listener.event("tune.cancelled", {"reason": "user"})
    listener.event("tune.failed", {"error": "SENTINEL_TUNE_EVENT"})
    listener.event("unknown", {})
    await asyncio.sleep(0.01)

    queue = bus.get_queue("tune-events")
    events = []
    while not queue.empty():
        events.append(await queue.get())
    assert events
    assert all(event.state == StepState.RUNNING for event in events)
    assert any("publishing artifacts" in (event.message or "") for event in events)
    assert any(
        "publishing partial artifacts" in (event.message or "") for event in events
    )
    assert any(event.extra and event.extra.get("best_score") == 0.3 for event in events)
    assert "SENTINEL" not in repr(events)
    assert "SENTINEL" not in caplog.text


def test_tune_event_listener_emit_threadsafe_handles_missing_loop() -> None:
    listener = executor._TuneEventListener("no-loop", max_trials=1)
    listener.loop = None
    listener._emit_threadsafe(
        ProgressEvent(session_id="no-loop", step="tune", state=StepState.RUNNING)
    )


@pytest.mark.asyncio
async def test_tune_watch_for_cancel_sets_event_and_handles_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = 0

        async def is_cancelled(self, _session_id: str) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("SENTINEL_TRANSIENT_POLL")
            return True

    cancel_event = threading.Event()
    await executor._watch_for_cancel(
        Manager(),
        "sid",
        cancel_event,
        poll_interval=0,
    )
    assert cancel_event.is_set()
    assert "SENTINEL_TRANSIENT_POLL" not in caplog.text

    class NeverCancelled:
        async def is_cancelled(self, _session_id: str) -> bool:
            return False

    task = asyncio.create_task(
        executor._watch_for_cancel(
            NeverCancelled(),
            "sid",
            threading.Event(),
            poll_interval=10,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_tune_terminal_event_noops_and_handles_emit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await executor._emit_terminal_bus_event("missing", StepState.FAILED, "bad")

    class BadBus:
        def get_snapshot(self, _session_id: str) -> dict:
            return {"status": "running"}

        async def emit(self, _event: ProgressEvent) -> None:
            raise RuntimeError("emit failed")

    monkeypatch.setattr(executor, "get_event_bus", lambda: BadBus())
    await executor._emit_terminal_bus_event(
        "sid",
        StepState.FAILED,
        "bad",
        error="bad",
    )


async def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_or_exc,
    *,
    manager: _Manager | None = None,
) -> _Manager:
    manager = manager or _Manager(tmp_path)
    session_dir = manager.get_session_dir("sid")
    scenario = session_dir / "scenario.yaml"
    usd = session_dir / "physics.usda"
    scenario.write_text("name: drop_settle\n", encoding="utf-8")
    usd.write_text("#usda 1.0\n", encoding="utf-8")

    async def fake_arun_tune(tune_input):
        assert tune_input.physics_usd == usd
        assert tune_input.event_listener is not None
        if isinstance(result_or_exc, BaseException):
            raise result_or_exc
        (tune_input.output_dir / "best_params.json").write_text("{}", encoding="utf-8")
        return result_or_exc

    monkeypatch.setattr(executor, "arun_tune", fake_arun_tune)
    await executor.execute_tune_async(
        "sid",
        manager,
        scenario,
        usd,
        user_prompt="make it bouncy",
        engine="fake",
        optimizer="botorch",
        max_trials=2,
        seed=42,
    )
    return manager


@pytest.mark.asyncio
async def test_execute_tune_success_cancelled_failed_and_sync_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = await _execute(tmp_path / "success", monkeypatch, _result())
    assert manager.metadata["status"] == "completed"
    assert manager.metadata["results"]["best_score"] == 0.2
    assert manager.sync_calls == ["tune/"]
    assert manager.metadata["artifact_manifest"] == ["tune/best_params.json"]
    assert manager.operations.index("sync:tune/") < manager.operations.index(
        "status:completed"
    )

    manager = _Manager(tmp_path / "sync-fail")
    manager.fail_sync = True
    await _execute(tmp_path / "sync-fail", monkeypatch, _result(), manager=manager)
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "artifact_sync"
    assert manager.metadata["partial_results"]["best_score"] == 0.2
    assert manager.metadata["artifact_sync_error"] == (
        "physics_tune_artifact_sync_failed"
    )
    assert manager.metadata["artifact_sync_diagnostic"]["phase"] == "sync_upload"
    assert "sync failed" not in repr(manager.metadata)
    assert "sync failed" not in caplog.text

    manager = _Manager(tmp_path / "cancel-during-sync")
    manager.cancel_during_sync = True
    await _execute(
        tmp_path / "cancel-during-sync", monkeypatch, _result(), manager=manager
    )
    assert manager.metadata["status"] == "cancelled"

    manager = _Manager(tmp_path / "late-cancel")
    manager.cancelled = True
    manager.metadata["created_at"] = datetime.now().isoformat()
    await _execute(
        tmp_path / "late-cancel",
        monkeypatch,
        _result(cancelled=False, best_score=math.inf),
        manager=manager,
    )
    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["results"]["best_score"] is None

    manager = await _execute(
        tmp_path / "failed-partial",
        monkeypatch,
        _result(success=False, error="SENTINEL_JUDGE_FAILED", n_trials=1),
    )
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["error"] == "physics_tune_result_failed"
    assert manager.metadata["error_diagnostic"]["phase"] == "pipeline_execution"
    assert "SENTINEL_JUDGE_FAILED" not in repr(manager.metadata)
    assert manager.metadata["partial_results"]["n_trials"] == 1

    manager = await _execute(
        tmp_path / "failed-empty",
        monkeypatch,
        _result(
            success=False,
            error="SENTINEL_EARLY_FAILED",
            n_trials=0,
            best_params={},
        ),
    )
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["error"] == "physics_tune_result_failed"
    assert "SENTINEL_EARLY_FAILED" not in repr(manager.metadata)
    assert "partial_results" not in manager.metadata


@pytest.mark.asyncio
async def test_execute_tune_cancelled_and_failed_terminal_warning_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(session_id="sid", step="tune", state=StepState.RUNNING, percent=0)
    )

    async def fail_emit(_event: ProgressEvent) -> None:
        raise RuntimeError("emit failed")

    monkeypatch.setattr(bus, "emit", fail_emit)
    manager = _Manager(tmp_path / "cancel-warning")
    manager.cancelled = True
    manager.fail_sync = True
    await _execute(
        tmp_path / "cancel-warning",
        monkeypatch,
        _result(cancelled=False, n_trials=1, best_score=math.inf),
        manager=manager,
    )
    assert manager.metadata["status"] == "cancelled"

    monkeypatch.undo()
    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(session_id="sid", step="tune", state=StepState.RUNNING, percent=0)
    )
    manager = _Manager(tmp_path / "failed-warning")
    manager.fail_sync = True
    await _execute(
        tmp_path / "failed-warning",
        monkeypatch,
        _result(success=False, error="judge failed", n_trials=1),
        manager=manager,
    )
    assert manager.metadata["status"] == "failed"

    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(session_id="sid", step="tune", state=StepState.RUNNING, percent=0)
    )
    monkeypatch.setattr(bus, "emit", fail_emit)
    manager = _Manager(tmp_path / "failed-emit-warning")
    await _execute(
        tmp_path / "failed-emit-warning",
        monkeypatch,
        _result(success=False, error="judge failed", n_trials=1),
        manager=manager,
    )
    assert manager.metadata["status"] == "failed"


@pytest.mark.asyncio
async def test_execute_tune_runner_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for exc, expected_code in (
        (
            executor.BoTorchUnavailableError("SENTINEL_NO_BOTORCH"),
            "physics_tune_botorch_unavailable",
        ),
        (
            executor.OvPhysXUnavailableError("SENTINEL_NO_PHYSX"),
            "physics_tune_ovphysx_unavailable",
        ),
        (RuntimeError("SENTINEL_TUNE_EXCEPTION"), "physics_tune_execution_failed"),
    ):
        manager = _Manager(tmp_path / type(exc).__name__)
        with pytest.raises(type(exc), match=expected_code) as excinfo:
            await _execute(
                tmp_path / type(exc).__name__, monkeypatch, exc, manager=manager
            )
        assert manager.metadata["status"] == "failed"
        assert manager.metadata["failed_step"] == "tune"
        assert manager.metadata["error"] == expected_code
        assert manager.metadata["error_diagnostic"]["phase"] == "pipeline_execution"
        assert excinfo.value.__context__ is None
        assert "SENTINEL" not in repr(manager.metadata)
    assert "SENTINEL" not in caplog.text

    manager = _Manager(tmp_path / "outer-cancel")
    with pytest.raises(asyncio.CancelledError):
        await _execute(
            tmp_path / "outer-cancel",
            monkeypatch,
            asyncio.CancelledError(),
            manager=manager,
        )
    assert manager.metadata["status"] == "cancelled"

    manager = _Manager(tmp_path / "cooperative")
    await _execute(
        tmp_path / "cooperative",
        monkeypatch,
        executor.TuningCancelledError("cancelled"),
        manager=manager,
    )
    assert manager.metadata["status"] == "cancelled"

    manager = _Manager(tmp_path / "publication-cancel")
    manager.cancel_sync_task = True
    with pytest.raises(asyncio.CancelledError):
        await _execute(
            tmp_path / "publication-cancel",
            monkeypatch,
            _result(),
            manager=manager,
        )
    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["artifact_manifest"] == ["tune/best_params.json"]
    assert manager.metadata["results"]["n_trials"] == 2


@pytest.mark.asyncio
async def test_execute_tune_emits_terminal_events_when_snapshot_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(session_id="sid", step="tune", state=StepState.RUNNING, percent=0)
    )

    manager = await _execute(tmp_path / "events", monkeypatch, _result())

    assert manager.metadata["status"] == "completed"
    queue = bus.get_queue("sid")
    events = []
    while not queue.empty():
        events.append(await queue.get())
    assert any(event.extra and event.extra.get("tune_ready") for event in events)
