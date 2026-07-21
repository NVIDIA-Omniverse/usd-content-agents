# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.workers import refine_executor as executor


@dataclass
class _Iteration:
    iteration: object
    iteration_dir: Path
    judge_decision: str = "approve"
    judge_score: object = 0.95
    judge_reasoning: str = "ok"
    best_score: object = 0.12
    best_params: dict[str, object] | None = None
    error: str | None = None
    recording_error: str | None = None


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
            raise RuntimeError("sync boom")
        self.sync_calls.append(prefix)
        return 1


def _result(tmp_path: Path, *, success: bool = True, reason: str = "approved"):
    final_dir = tmp_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "best_params.json").write_text(
        json.dumps({"params": {"mass_scale": 1.4}, "best_score": 0.12}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        success=success,
        error=None if success else "judge rejected",
        termination_reason=reason,
        iteration_count=1,
        final_iteration=1,
        final_judge_score=0.94 if success else float("nan"),
        iterations=[
            _Iteration(
                iteration=1,
                iteration_dir=tmp_path / "iter_1",
                best_params={"mass_scale": 1.0},
            )
        ],
        final_dir=final_dir,
        output_dir=tmp_path,
        final_recording_usd=None,
        final_recording_error=None,
    )


def test_refine_metadata_helpers_cover_edge_cases(tmp_path: Path) -> None:
    assert executor._finite_or_none(None) is None
    assert executor._finite_or_none("bad") is None
    assert executor._finite_or_none(float("inf")) is None
    assert executor._finite_or_none("1.25") == 1.25

    assert executor._json_safe_metadata({"p": tmp_path, "t": (tmp_path, 1)}) == {
        "p": str(tmp_path),
        "t": [str(tmp_path), 1],
    }
    assert executor._json_safe_metadata([tmp_path]) == [str(tmp_path)]
    missing = tmp_path / "missing"
    assert executor._load_final_best_params(None) is None
    assert executor._load_final_best_params(missing) is None
    missing.mkdir()
    (missing / "best_params.json").write_text("[]", encoding="utf-8")
    assert executor._load_final_best_params(missing) is None
    (missing / "best_params.json").write_text(
        json.dumps({"best_score": 1.0}),
        encoding="utf-8",
    )
    assert executor._load_final_best_params(missing) is None
    (missing / "best_params.json").write_text(
        json.dumps({"params": {"ok": "2", "nan": math.nan}}),
        encoding="utf-8",
    )
    assert executor._load_final_best_params(missing) == {"ok": 2.0}
    (missing / "best_params.json").write_text(
        json.dumps({"params": {"ok": "2", "bad": "nope"}}),
        encoding="utf-8",
    )
    assert executor._load_final_best_params(missing) == {"ok": 2.0}
    (missing / "best_params.json").write_text(
        json.dumps({"mass_scale": 3.0}),
        encoding="utf-8",
    )
    assert executor._load_final_best_params(missing) == {"mass_scale": 3.0}

    result = _result(tmp_path)
    result.final_iteration = 99
    metadata = executor._refine_results_metadata(result)
    assert metadata["iterations"][-1]["best_params"] == {"mass_scale": 1.4}

    result.iterations[0].iteration = "bad"
    metadata = executor._refine_results_metadata(result)
    assert metadata["iterations"][-1]["best_params"] == {"mass_scale": 1.4}

    projected = executor._iteration_to_metadata(
        _Iteration(
            iteration=1,
            iteration_dir=tmp_path,
            error="SENTINEL_ITERATION_ERROR",
            recording_error="SENTINEL_RECORDING_ERROR",
        )
    )
    assert projected["error"] == "physics_refine_iteration_failed"
    assert projected["recording_error"] == ("physics_refine_iteration_recording_failed")
    assert projected["error_diagnostic"]["phase"] == "pipeline_execution"
    assert projected["recording_error_diagnostic"]["phase"] == "local_publication"
    assert "SENTINEL" not in repr(projected)


def test_build_refine_models_requires_chat_and_vlm_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.models.backends.registry as registry

    monkeypatch.setenv("PA_REFINE_BACKEND", "plugin-provider")
    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["plugin-provider"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: [])
    with pytest.raises(RuntimeError, match="not registered as both chat and VLM"):
        executor._build_refine_models(
            judge_max_tokens=None,
            judge_temperature=None,
        )


def test_build_refine_models_success_and_config_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_REFINE_BACKEND", "plugin-provider")
    monkeypatch.setenv("PA_REFINE_MODEL", "model")

    import physics_agent.tuning.visual_evidence as visual_evidence
    import world_understanding.agentic.config as agentic_config
    import world_understanding.functions.models.backends.registry as registry
    import world_understanding.functions.models.chat_models as chat_models
    import world_understanding.functions.models.vision_language_models as vlm_models
    import world_understanding.utils.credentials as credentials

    monkeypatch.setattr(registry, "list_chat_backends", lambda: ["plugin-provider"])
    monkeypatch.setattr(registry, "list_vlm_backends", lambda: ["plugin-provider"])
    monkeypatch.setattr(registry, "chat_backend_requires_api_key", lambda _name: True)
    monkeypatch.setattr(registry, "vlm_backend_requires_api_key", lambda _name: True)
    monkeypatch.setattr(credentials, "get_env_api_key_for_backend", lambda _backend: "")
    monkeypatch.setitem(
        credentials.API_KEY_ENV_VAR_MAP,
        "plugin-provider",
        ("PLUGIN_API_KEY",),
    )
    with pytest.raises(RuntimeError, match="API key"):
        executor._build_refine_models(
            judge_max_tokens=None,
            judge_temperature=None,
        )

    monkeypatch.setattr(
        credentials, "get_env_api_key_for_backend", lambda _backend: "key"
    )
    monkeypatch.setattr(
        agentic_config,
        "get_api_key_for_model_config",
        lambda _backend, _config, _kind: "vlm-key",
    )
    monkeypatch.setattr(
        chat_models,
        "create_chat_model",
        lambda **kwargs: ("chat", kwargs),
    )
    monkeypatch.setattr(
        vlm_models,
        "create_vlm",
        lambda **kwargs: ("vlm", kwargs),
    )
    monkeypatch.setattr(
        visual_evidence,
        "backend_supports_reasoning_effort",
        lambda _backend: False,
    )

    chat_model, vlm_model = executor._build_refine_models(
        judge_max_tokens=123,
        judge_temperature=0.25,
    )
    assert chat_model[0] == "chat"
    assert vlm_model[0] == "vlm"
    assert vlm_model[1]["api_key"] == "vlm-key"
    assert "reasoning_effort" not in vlm_model[1]


@pytest.mark.asyncio
async def test_refine_event_listener_maps_runner_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = get_event_bus()
    bus.cleanup_session("refine-events")
    listener = executor._RefineEventListener(
        "refine-events",
        max_iterations=2,
        max_trials=2,
    )
    listener.info("hello")
    listener.debug("hello")
    listener.warning("SENTINEL_RUNNER_WARNING")
    listener.error("SENTINEL_RUNNER_ERROR")

    listener.event("refine.iteration.started", {"iteration": 1})
    listener.event(
        "tune.trial.completed",
        {"trial_index": 0, "score": 0.2, "params": {"mass_scale": 1.1}},
    )
    listener.event("tune.trial.completed", {"score": 0.3, "failed": True})
    listener.event(
        "refine.iteration.tune_completed",
        {"iteration": 1, "n_trials": 2, "best_score": 0.2, "best_params": {"x": 1}},
    )
    listener.event("refine.iteration.judged", {"iteration": 1, "judge_score": 0.91})
    listener.event("refine.completed", {"ok": True})
    listener.event("tune.cancelled", {"why": "user"})
    listener.event("tune.failed", {"error": "SENTINEL_RUNNER_EVENT"})
    listener.event("unknown", {})
    await asyncio.sleep(0.01)

    queued = []
    queue = bus.get_queue("refine-events")
    while not queue.empty():
        queued.append(await queue.get())
    assert any(event.extra and event.extra.get("best_score") == 0.2 for event in queued)
    assert queued
    assert all(event.state == StepState.RUNNING for event in queued)
    assert any(
        "publishing partial artifacts" in (event.message or "") for event in queued
    )
    assert "SENTINEL" not in repr(queued)
    assert "SENTINEL" not in caplog.text


def test_refine_event_listener_emit_threadsafe_handles_missing_or_bad_loop() -> None:
    listener = executor._RefineEventListener("no-loop", 1, 1)
    listener.loop = None
    listener._emit_threadsafe(
        ProgressEvent(session_id="no-loop", step="refine", state=StepState.RUNNING)
    )

    class ClosedLoop:
        def is_closed(self) -> bool:
            return True

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(executor.asyncio, "get_running_loop", lambda: ClosedLoop())
        listener.loop = None
        listener._emit_threadsafe(
            ProgressEvent(session_id="no-loop", step="refine", state=StepState.RUNNING)
        )
    finally:
        monkeypatch.undo()

    class InlineLoop:
        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, callback):
            callback()

    monkeypatch = pytest.MonkeyPatch()
    try:

        def fail_create_task(coro):
            coro.close()
            raise RuntimeError("closed")

        monkeypatch.setattr(executor.asyncio, "create_task", fail_create_task)
        listener.loop = InlineLoop()
        listener._emit_threadsafe(
            ProgressEvent(session_id="no-loop", step="refine", state=StepState.RUNNING)
        )
    finally:
        monkeypatch.undo()

    class BadLoop:
        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, _callback):
            raise RuntimeError("closed")

    listener.loop = BadLoop()
    listener._emit_threadsafe(
        ProgressEvent(session_id="no-loop", step="refine", state=StepState.RUNNING)
    )


@pytest.mark.asyncio
async def test_refine_watch_for_cancel_sets_event_and_ignores_poll_errors() -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = 0

        async def is_cancelled(self, _session_id: str) -> bool:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return True

    cancel_event = threading.Event()
    await executor._watch_for_cancel(
        Manager(),
        "sid",
        cancel_event,
        poll_interval=0,
    )
    assert cancel_event.is_set()

    never_cancel = threading.Event()
    task = asyncio.create_task(
        executor._watch_for_cancel(
            Manager(),
            "sid",
            never_cancel,
            poll_interval=10,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_refine_terminal_event_noops_without_snapshot() -> None:
    await executor._emit_terminal_bus_event("missing", StepState.COMPLETED, "done")

    bus = get_event_bus()
    bus.cleanup_session("terminal")
    await bus.seed_pending_session("terminal")
    await executor._emit_terminal_bus_event(
        "terminal",
        StepState.FAILED,
        "failed",
        error="boom",
        extra={"reason": "test"},
    )
    snapshot = bus.get_snapshot("terminal")
    assert snapshot is not None
    assert snapshot["status"] == "failed"

    async def fail_emit(_event):
        raise RuntimeError("emit failed")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(bus, "emit", fail_emit)
        await executor._emit_terminal_bus_event("terminal", StepState.FAILED, "failed")
    finally:
        monkeypatch.undo()


async def _run_execute_refine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_or_exc,
    *,
    manager: _Manager | None = None,
) -> _Manager:
    manager = manager or _Manager(tmp_path)
    session_id = "sid"
    session_dir = manager.get_session_dir(session_id)
    scenario_path = session_dir / "scenario.yaml"
    physics_usd = session_dir / "physics.usda"
    scenario_path.write_text("name: drop_settle\n", encoding="utf-8")
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        executor,
        "_build_refine_models",
        lambda **_kwargs: ("chat", "vlm"),
    )

    async def fake_arun_refine(refine_input):
        assert refine_input.chat_model == "chat"
        assert refine_input.vlm_model == "vlm"
        assert refine_input.force_record_video == "off"
        assert refine_input.render_winning_trial is False
        if isinstance(result_or_exc, BaseException):
            raise result_or_exc
        (refine_input.output_dir / "refine_summary.json").write_text(
            "{}", encoding="utf-8"
        )
        return result_or_exc

    monkeypatch.setattr(executor, "arun_refine", fake_arun_refine)

    await executor.execute_refine_async(
        session_id,
        manager,
        scenario_path,
        physics_usd,
        user_prompt="make it bouncy",
        engine="fake",
        optimizer="botorch",
        max_trials=2,
        seed=42,
        max_iterations=1,
        score_threshold=0.9,
    )
    return manager


@pytest.mark.asyncio
async def test_execute_refine_success_and_sync_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = _result(tmp_path)
    manager = await _run_execute_refine(tmp_path, monkeypatch, result)
    assert manager.metadata["status"] == "completed"
    assert manager.metadata["results"]["final_best_params"] == {"mass_scale": 1.4}
    assert manager.sync_calls == ["refine/"]
    assert manager.metadata["artifact_manifest"] == ["refine/refine_summary.json"]
    assert manager.operations.index("sync:refine/") < manager.operations.index(
        "status:completed"
    )

    manager = _Manager(tmp_path)
    manager.fail_sync = True
    manager.metadata["created_at"] = datetime.now().isoformat()
    monkeypatch.setattr(executor, "_build_refine_models", lambda **_: ("c", "v"))

    async def fake_arun_refine(_input):
        return _result(tmp_path / "sync-fail")

    monkeypatch.setattr(executor, "arun_refine", fake_arun_refine)
    session_dir = manager.get_session_dir("sid2")
    scenario_path = session_dir / "scenario.yaml"
    physics_usd = session_dir / "physics.usda"
    scenario_path.write_text("name: drop_settle\n", encoding="utf-8")
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
    await executor.execute_refine_async(
        "sid2",
        manager,
        scenario_path,
        physics_usd,
        user_prompt="make it bouncy",
        engine="fake",
        optimizer="botorch",
        max_trials=2,
        seed=42,
        max_iterations=1,
        score_threshold=0.9,
    )
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "artifact_sync"
    assert manager.metadata["partial_results"]["termination_reason"] == "approved"
    assert manager.metadata["artifact_sync_error"] == (
        "physics_refine_artifact_sync_failed"
    )
    assert manager.metadata["artifact_sync_diagnostic"]["phase"] == "sync_upload"
    assert "sync boom" not in repr(manager.metadata)
    assert "sync boom" not in caplog.text

    manager = _Manager(tmp_path / "cancel-during-sync")
    manager.cancel_during_sync = True
    result = _result(tmp_path / "cancel-during-sync" / "result")
    await _run_execute_refine(
        tmp_path / "cancel-during-sync",
        monkeypatch,
        result,
        manager=manager,
    )
    assert manager.metadata["status"] == "cancelled"


@pytest.mark.asyncio
async def test_execute_refine_records_materialization_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = tmp_path / "published" / "recording.usd"
    recording.parent.mkdir()
    recording.write_bytes(b"recording")
    result = _result(tmp_path / "success-result")
    result.final_recording_usd = recording
    manager = await _run_execute_refine(
        tmp_path / "success-run",
        monkeypatch,
        result,
    )
    assert manager.metadata["results"]["final_recording_usd"] == str(recording)

    result = _result(tmp_path / "error-result")
    result.final_recording_error = "flatten failed"
    manager = await _run_execute_refine(
        tmp_path / "error-run",
        monkeypatch,
        result,
    )
    assert manager.metadata["results"]["final_recording_error"] == (
        "physics_refine_final_recording_failed"
    )
    assert (
        manager.metadata["results"]["final_recording_error_diagnostic"]["phase"]
        == "local_publication"
    )
    assert "flatten failed" not in repr(manager.metadata)


@pytest.mark.asyncio
async def test_execute_refine_failed_result_late_cancel_and_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failed = _result(tmp_path / "failed", success=False, reason="max_iterations")
    manager = await _run_execute_refine(tmp_path / "failed-run", monkeypatch, failed)
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "refine"
    assert manager.metadata["error"] == "physics_refine_result_failed"
    assert manager.metadata["error_diagnostic"]["phase"] == "pipeline_execution"
    assert "judge rejected" not in repr(manager.metadata)
    assert manager.metadata["partial_results"]["termination_reason"] == "max_iterations"

    late = _result(tmp_path / "late", success=True, reason="approved")
    manager = _Manager(tmp_path / "late-run")
    manager.cancelled = True
    monkeypatch.setattr(executor, "_build_refine_models", lambda **_: ("c", "v"))

    async def fake_late_refine(_input):
        return late

    monkeypatch.setattr(executor, "arun_refine", fake_late_refine)
    session_dir = manager.get_session_dir("late")
    scenario_path = session_dir / "scenario.yaml"
    physics_usd = session_dir / "physics.usda"
    scenario_path.write_text("name: drop_settle\n", encoding="utf-8")
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")
    await executor.execute_refine_async(
        "late",
        manager,
        scenario_path,
        physics_usd,
        user_prompt="make it bouncy",
        engine="fake",
        optimizer="botorch",
        max_trials=2,
        seed=42,
        max_iterations=1,
        score_threshold=0.9,
    )
    assert manager.metadata["status"] == "cancelled"

    exception_manager = _Manager(tmp_path / "raise-run")
    with pytest.raises(
        RuntimeError, match="physics_refine_execution_failed"
    ) as excinfo:
        await _run_execute_refine(
            tmp_path / "raise-run",
            monkeypatch,
            RuntimeError("SENTINEL_REFINE_EXCEPTION"),
            manager=exception_manager,
        )
    assert excinfo.value.__context__ is None
    assert exception_manager.metadata["error"] == "physics_refine_execution_failed"
    assert "SENTINEL_REFINE_EXCEPTION" not in repr(exception_manager.metadata)
    assert "SENTINEL_REFINE_EXCEPTION" not in caplog.text

    with pytest.raises(asyncio.CancelledError):
        await _run_execute_refine(
            tmp_path / "cancel-run",
            monkeypatch,
            asyncio.CancelledError(),
        )

    manager = _Manager(tmp_path / "publication-cancel")
    manager.cancel_sync_task = True
    with pytest.raises(asyncio.CancelledError):
        await _run_execute_refine(
            tmp_path / "publication-cancel",
            monkeypatch,
            _result(tmp_path / "publication-cancel-result"),
            manager=manager,
        )
    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["artifact_manifest"] == ["refine/refine_summary.json"]
    assert manager.metadata["results"]["termination_reason"] == "approved"
