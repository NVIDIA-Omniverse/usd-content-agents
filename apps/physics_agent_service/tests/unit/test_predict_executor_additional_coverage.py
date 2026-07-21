# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import builtins
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.workers import predict_executor as executor


class _Manager:
    def __init__(self, root: Path, *, fail_update: bool = False) -> None:
        self.root = root
        self.storage_path = root
        self.metadata: dict[str, object] = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "status": "pending",
        }
        self.fail_update = fail_update
        self.sync_from_calls: list[str] = []
        self.sync_to_calls: list[str] = []
        self.fail_sync_from = False
        self.fail_sync_to = False

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def sync_from_store(self, _session_id: str, *, prefix: str = "") -> int:
        self.sync_from_calls.append(prefix)
        if self.fail_sync_from:
            raise RuntimeError("pull failed")
        return 0

    async def sync_to_store(self, _session_id: str, *, prefix: str = "") -> int:
        self.sync_to_calls.append(prefix)
        if self.fail_sync_to:
            raise RuntimeError("push failed")
        return 1

    async def update_session(
        self, _session_id: str, updates: dict[str, object]
    ) -> None:
        if self.fail_update:
            raise RuntimeError("update failed")
        self.metadata.update(updates)

    async def get_session_metadata(self, _session_id: str) -> dict[str, object]:
        return dict(self.metadata)


def _config() -> dict:
    return {
        "steps": {
            "optimize_usd": {"enabled": False},
            "identify_asset": {"enabled": True},
            "build_dataset_usd": {"enabled": True},
            "build_dataset_prepare_dataset": {"enabled": True},
            "predict": {"enabled": False},
            "restore_usd": {"enabled": True},
        }
    }


def test_predict_image_helpers_cover_remaining_shapes(tmp_path: Path) -> None:
    assert executor._extract_image_paths(
        {"media": {"images": ["media.png", {"path": "dict.png"}, {"bad": "x"}]}}
    ) == ["media.png", "dict.png"]
    assert executor._extract_image_paths(
        {"images": [{"path": "list-dict.png"}, {"bad": "x"}, 3]}
    ) == ["list-dict.png"]

    jsonl = tmp_path / "dataset.jsonl"
    img = tmp_path / "img.png"
    img.write_text("x", encoding="utf-8")
    with jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(["not", "dict"]) + "\n")
        for idx in range(25):
            f.write(json.dumps({"id": idx, "images": ["missing.png"]}) + "\n")
        f.write(json.dumps({"id": "late", "images": ["img.png"]}) + "\n")
    assert executor._dataset_jsonl_has_resolvable_images(jsonl) is False

    dedup = tmp_path / "dedup.jsonl"
    dedup.write_text(
        json.dumps({"id": "dedup", "images": ["img.png", "img.png"]}) + "\n",
        encoding="utf-8",
    )
    assert executor._dataset_jsonl_has_resolvable_images(dedup) is True


@pytest.mark.asyncio
async def test_execute_predict_dataset_only_success_and_sync_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "predict-a"
    dataset_path = tmp_path / "external" / "dataset.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text(json.dumps({"id": "/World/Cube"}) + "\n", encoding="utf-8")

    async def fake_arun_predict(params):
        assert params.dataset_override == dataset_path
        assert (
            params.output_dir_override
            == manager.get_session_dir(session_id) / "cache" / "predictions"
        )
        out = params.output_dir_override
        out.mkdir(parents=True, exist_ok=True)
        (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error=None,
            predictions_count=1,
            failed_count=0,
            predictions_path=out / "predictions.jsonl",
            token_stats={"input": 4},
        )

    monkeypatch.setattr(executor, "arun_predict", fake_arun_predict)

    await executor.execute_predict_async(
        session_id,
        _config(),
        manager,
        dataset_path=dataset_path,
    )

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["predict_mode"] == "dataset_only"
    assert manager.metadata["predict_steps_run"] == ["predict"]
    assert manager.sync_to_calls == [
        "cache/predictions/",
        "cache/dataset/dataset.jsonl",
    ]

    manager.fail_sync_to = True
    await executor.execute_predict_async(
        session_id,
        _config(),
        manager,
        dataset_path=dataset_path,
    )
    assert manager.metadata["status"] == "completed"
    assert "push failed" not in caplog.text


@pytest.mark.asyncio
async def test_execute_predict_dataset_only_failure_and_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")

    async def failed_predict(_params):
        return SimpleNamespace(success=False, error="SENTINEL_BAD_VLM")

    manager = _Manager(tmp_path / "failed")
    monkeypatch.setattr(executor, "arun_predict", failed_predict)
    with pytest.raises(RuntimeError, match="physics_predict_result_failed") as excinfo:
        await executor.execute_predict_async(
            "predict-fail",
            _config(),
            manager,
            dataset_path=dataset_path,
        )
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "predict"
    assert manager.metadata["error"] == "physics_predict_result_failed"
    assert manager.metadata["error_diagnostic"]["phase"] == "pipeline_execution"
    assert excinfo.value.__context__ is None
    assert "SENTINEL_BAD_VLM" not in repr(manager.metadata)

    bus = get_event_bus()
    queued = []
    queue = bus.get_queue("predict-fail")
    while not queue.empty():
        queued.append(await queue.get())
    assert "SENTINEL_BAD_VLM" not in repr(queued)

    async def cancelled_predict(_params):
        raise asyncio.CancelledError

    manager = _Manager(tmp_path / "cancel")
    monkeypatch.setattr(executor, "arun_predict", cancelled_predict)
    with pytest.raises(asyncio.CancelledError):
        await executor.execute_predict_async(
            "predict-cancel",
            _config(),
            manager,
            dataset_path=dataset_path,
        )
    assert manager.metadata["status"] == "cancelled"


@pytest.mark.asyncio
async def test_execute_predict_full_mode_success_failure_and_unexpected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path / "full")
    manager.fail_sync_from = True
    session_id = "predict-b"

    async def good_pipeline(params):
        assert "predict" in params.only_steps
        session_dir = manager.get_session_dir(session_id)
        restored = session_dir / "restore" / "restored.jsonl"
        restored.parent.mkdir(parents=True, exist_ok=True)
        restored.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error=None,
            step_results={
                "predict": {
                    "predictions_count": 2,
                    "failed_count": 1,
                    "predictions_path": session_dir / "old.jsonl",
                    "token_stats": {"out": 2},
                },
                "restore_usd": {"restored_predictions_path": restored},
            },
            raw_result={"build_dataset_usd_result": {"num_prims": 3, "num_images": 4}},
        )

    monkeypatch.setattr(executor, "arun_pipeline", good_pipeline)
    await executor.execute_predict_async(session_id, _config(), manager)
    assert manager.metadata["status"] == "completed"
    assert manager.metadata["predict_mode"] == "full_predict"
    assert manager.metadata["predict_steps_run"][-1] == "predict"
    assert manager.sync_from_calls == ["cache/dataset/"]

    async def bad_pipeline(_params):
        return SimpleNamespace(success=False, error="SENTINEL_PIPELINE_BAD")

    manager = _Manager(tmp_path / "full-fail")
    monkeypatch.setattr(executor, "arun_pipeline", bad_pipeline)
    with pytest.raises(RuntimeError, match="physics_predict_result_failed"):
        await executor.execute_predict_async("predict-bad", _config(), manager)
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["error"] == "physics_predict_result_failed"
    assert "SENTINEL_PIPELINE_BAD" not in repr(manager.metadata)

    async def boom_pipeline(_params):
        raise RuntimeError("SENTINEL_PREDICT_EXCEPTION")

    manager = _Manager(tmp_path / "boom")
    monkeypatch.setattr(executor, "arun_pipeline", boom_pipeline)
    with pytest.raises(
        RuntimeError, match="physics_predict_execution_failed"
    ) as excinfo:
        await executor.execute_predict_async("predict-boom", _config(), manager)
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["error"] == "physics_predict_execution_failed"
    assert excinfo.value.__context__ is None
    assert "SENTINEL_PREDICT_EXCEPTION" not in repr(manager.metadata)
    assert "SENTINEL_PREDICT_EXCEPTION" not in caplog.text


@pytest.mark.asyncio
async def test_mark_failed_handles_update_and_bus_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = executor.durable_diagnostic(
        "physics_predict_execution_failed",
        phase=executor.FailurePhase.PIPELINE_EXECUTION,
        retryable=False,
    )
    manager = _Manager(tmp_path, fail_update=True)
    await executor._mark_failed(manager, "sid", diagnostic, "predict")

    class BadBus:
        def get_snapshot(self, _session_id: str) -> dict:
            return {"status": "running"}

        async def emit(self, _event: ProgressEvent) -> None:
            raise RuntimeError("emit failed")

    monkeypatch.setattr(executor, "get_event_bus", lambda: BadBus())
    manager = _Manager(tmp_path / "bus")
    await executor._mark_failed(manager, "sid", diagnostic, "predict")
    assert manager.metadata["status"] == "failed"


def test_extract_stats_from_pipeline_result_file_fallbacks_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "sid"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset.parent.mkdir(parents=True)
    predictions.parent.mkdir(parents=True)
    dataset.write_text("{}\n{}\n", encoding="utf-8")
    predictions.write_text("{}\n\n{}\n", encoding="utf-8")

    result = SimpleNamespace(
        step_results={},
        raw_result={"build_dataset_prepare_dataset_result": {"num_entries": 7}},
    )
    stats = executor._extract_stats_from_pipeline_result(result, session_dir)
    assert stats["prims_processed"] == 7
    assert stats["predictions_made"] == 2
    assert stats["predictions_path"] == str(predictions)

    restored = tmp_path / "restored.jsonl"
    target = session_dir / "cache" / "predictions" / "predictions.jsonl"
    restored.write_text("{}\n", encoding="utf-8")
    result = SimpleNamespace(
        step_results={"restore_usd": {"restored_predictions_path": restored}},
        raw_result={},
    )
    stats = executor._extract_stats_from_pipeline_result(result, session_dir)
    assert stats["predictions_path"] == str(target)
    assert target.read_text(encoding="utf-8") == "{}\n"

    def bad_copy(_src, _dst):
        raise OSError("copy failed")

    monkeypatch.setattr(shutil, "copyfile", bad_copy)
    executor._extract_stats_from_pipeline_result(result, session_dir)

    real_open = builtins.open

    def fail_open(path, *args, **kwargs):
        if Path(path) in {dataset, predictions}:
            raise OSError("read failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_open)
    result = SimpleNamespace(step_results={}, raw_result={})
    stats = executor._extract_stats_from_pipeline_result(result, session_dir)
    assert stats["prims_processed"] == 0
    assert stats["predictions_made"] == 0


@pytest.mark.asyncio
async def test_execute_predict_emits_terminal_events_when_snapshot_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "predict-events"
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
            percent=0,
        )
    )

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")

    async def fake_arun_predict(params):
        return SimpleNamespace(
            success=True,
            error=None,
            predictions_count=1,
            failed_count=0,
            predictions_path=None,
            token_stats=None,
        )

    monkeypatch.setattr(executor, "arun_predict", fake_arun_predict)
    await executor.execute_predict_async(
        session_id,
        _config(),
        manager,
        dataset_path=dataset_path,
    )

    queued = []
    queue = bus.get_queue(session_id)
    while not queue.empty():
        queued.append(await queue.get())
    assert any(event.extra and event.extra.get("pipeline_ready") for event in queued)
