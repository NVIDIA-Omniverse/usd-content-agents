# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import builtins
import json
import logging
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Never

import pytest

if TYPE_CHECKING:
    from joint_agent.api import PipelineInput

from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry
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
        self.sync_overwrite: list[bool] = []
        self.statuses_seen_during_sync: list[str] = []
        self.fail_sync = False
        self.current_run_id = "a" * 32
        self.run_claim_heartbeat_seconds = 60.0
        self.renew_current_run = True
        self.released_runs: list[str] = []

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def get_session_metadata(self, _session_id: str):
        return dict(self.metadata)

    async def update_session(self, _session_id: str, updates: dict) -> None:
        self.metadata.update(updates)

    async def is_run_current(self, _session_id: str, run_id: str) -> bool:
        return self.current_run_id == run_id

    async def is_cancellation_accepted(
        self,
        _session_id: str,
        _run_id: str,
    ) -> bool:
        return False

    async def renew_run(self, _session_id: str, run_id: str) -> bool:
        return self.renew_current_run and self.current_run_id == run_id

    async def release_run(self, _session_id: str, run_id: str) -> bool:
        if self.current_run_id != run_id:
            return False
        self.released_runs.append(run_id)
        self.current_run_id = ""
        return True

    async def terminalize_and_release_run(
        self,
        session_id: str,
        run_id: str,
        updates: dict,
    ) -> bool:
        if self.current_run_id != run_id:
            return False
        self.metadata.update(updates)
        return await self.release_run(session_id, run_id)

    async def update_session_for_run(
        self, _session_id: str, run_id: str, updates: dict
    ) -> bool:
        if self.current_run_id != run_id:
            return False
        self.metadata.update(updates)
        return True

    async def sync_to_store(
        self,
        _session_id: str,
        *,
        prefix: str = "",
        overwrite: bool = False,
    ) -> int:
        self.statuses_seen_during_sync.append(str(self.metadata.get("status")))
        if self.fail_sync:
            raise RuntimeError("sync failed")
        self.sync_calls.append(prefix)
        self.sync_overwrite.append(overwrite)
        return 1


def _executor_traceback_locals(error: BaseException) -> list[dict[str, object]]:
    """Copy only executor-frame locals exposed by a public exception."""

    frame_locals: list[dict[str, object]] = []
    current = error.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == executor.__name__:
            frame_locals.append(dict(current.tb_frame.f_locals))
        current = current.tb_next
    return frame_locals


def test_prerequisite_sync_is_limited_to_requested_producers() -> None:
    assert executor._produced_cache_namespaces(None) == ("dataset", "predictions")
    assert executor._produced_cache_namespaces(["predict"]) == ("predictions",)
    assert executor._produced_cache_namespaces(["build_dataset_prepare_dataset"]) == (
        "dataset",
    )
    assert executor._produced_cache_namespaces(["identify_asset"]) == ()
    assert executor._completed_cache_namespaces(
        None,
        {"build_dataset_usd", "build_dataset_prepare_dataset"},
    ) == ("dataset",)
    assert executor._completed_cache_namespaces(
        ["predict"],
        {"predict"},
    ) == ("predictions",)


@pytest.mark.asyncio
async def test_backend_failure_is_projected_to_value_free_durable_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "value-free-failure"
    sentinel = "sentinel-joint-backend-secret"
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.seed_pending_session(session_id)

    async def failed_pipeline(_params):
        return SimpleNamespace(success=False, error=sentinel, completed_steps=[])

    monkeypatch.setattr(executor, "arun_pipeline", failed_pipeline)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=executor.__name__):
        with pytest.raises(RuntimeError, match="^Pipeline failed$") as exc_info:
            await executor.execute_pipeline_async(
                session_id,
                manager.current_run_id,
                {"project": {"name": "test"}},
                manager,
            )

    expected_diagnostic = {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "joint_pipeline_execution_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert manager.metadata["error"] == "joint_pipeline_execution_failed"
    assert manager.metadata["error_diagnostic"] == expected_diagnostic
    failed_event = await bus.get_queue(session_id).get()
    assert failed_event.message == "joint_pipeline_execution_failed"
    assert failed_event.extra == {
        "failed_step": "pipeline",
        "error_diagnostic": expected_diagnostic,
    }
    assert "joint_pipeline_execution_failed" in caplog.text
    assert sentinel not in json.dumps(manager.metadata)
    assert sentinel not in failed_event.model_dump_json()
    assert sentinel not in caplog.text
    assert sentinel not in repr(_executor_traceback_locals(exc_info.value))
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_failed_run_does_not_publish_partial_prediction_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.metadata["cache_publications"] = {}
    session_id = "partial-predictions"

    async def partial_pipeline(_params):
        session_dir = manager.get_session_dir(session_id)
        dataset_dir = session_dir / "cache" / "dataset"
        predictions_dir = session_dir / "cache" / "predictions"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
        (predictions_dir / "predictions.jsonl").write_text(
            '{"partial": true}\n',
            encoding="utf-8",
        )
        (session_dir / "cache" / ".pipeline_state.json").write_text(
            json.dumps(
                {
                    "completed_steps": [
                        "build_dataset_usd",
                        "build_dataset_prepare_dataset",
                    ],
                    "failed_steps": ["predict"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            success=False,
            error="prediction failed",
            completed_steps=[],
        )

    monkeypatch.setattr(executor, "arun_pipeline", partial_pipeline)
    with pytest.raises(RuntimeError, match="^Pipeline failed$"):
        await executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["cache_publications"] == {"dataset": "a" * 32}
    assert manager.metadata["failed_step"] == "predict"
    assert not (
        manager.get_session_dir(session_id)
        / f"artifacts/run_cache/{'a' * 32}/cache/predictions"
    ).exists()


@pytest.mark.asyncio
async def test_selective_legacy_run_migrates_untouched_cache_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "legacy-migration"
    session_dir = manager.get_session_dir(session_id)
    dataset_dir = session_dir / "cache" / "dataset"
    predictions_dir = session_dir / "cache" / "predictions"
    dataset_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)
    (dataset_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    (predictions_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")

    async def identify_only(_params):
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["identify_asset"],
            step_results={},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", identify_only)
    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["identify_asset"],
    )

    assert manager.metadata["cache_publications"] == {
        "dataset": "a" * 32,
        "predictions": "a" * 32,
    }
    assert manager.sync_calls == [
        f"artifacts/run_cache/{'a' * 32}/cache/dataset/",
        f"artifacts/run_cache/{'a' * 32}/cache/predictions/",
    ]


def test_predict_snapshot_excludes_stale_downstream_outputs(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    predictions_dir = session_dir / "cache" / "predictions"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.jsonl").write_text("old\n", encoding="utf-8")
    (predictions_dir / "articulation_candidates.json").write_text(
        "old-candidates", encoding="utf-8"
    )
    (predictions_dir / "articulation_candidate_adjudications.json").write_text(
        "old-adjudication", encoding="utf-8"
    )

    executor._prepare_cache_namespaces_for_run(session_dir, ["predict"])
    (predictions_dir / "predictions.jsonl").write_text("new\n", encoding="utf-8")
    run_id = "b" * 32
    bindings = executor._materialize_cache_publications(
        session_dir,
        run_id,
        ("predictions",),
    )

    publication = session_dir / f"artifacts/run_cache/{run_id}/cache/predictions"
    assert bindings == {"predictions": run_id}
    assert (publication / "predictions.jsonl").read_text(encoding="utf-8") == "new\n"
    assert not (publication / "articulation_candidates.json").exists()
    assert not (publication / "articulation_candidate_adjudications.json").exists()


def test_remove_cache_path_preserves_missing_file_directory_and_symlink_behavior(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    cache_root = session_dir / "cache"

    executor._remove_cache_path(cache_root / "missing", cache_root)

    cache_root.mkdir(parents=True)
    stale_file = cache_root / ".pipeline_state.json"
    stale_file.write_text("stale", encoding="utf-8")
    executor._remove_cache_path(stale_file, cache_root)
    assert not stale_file.exists()

    stale_directory = cache_root / "dataset"
    stale_directory.mkdir()
    (stale_directory / "nested.txt").write_text("stale", encoding="utf-8")
    executor._remove_cache_path(stale_directory, cache_root)
    assert not stale_directory.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside", encoding="utf-8")
    alias = cache_root / "predictions"
    alias.symlink_to(outside, target_is_directory=True)
    executor._remove_cache_path(alias, cache_root)
    assert not alias.exists()
    assert not alias.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "outside"


def test_remove_cache_path_rejects_post_validation_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    cache_root = session_dir / "cache"
    target = cache_root / "dataset"
    target.mkdir(parents=True)
    owned_sentinel = target / "owned.txt"
    owned_sentinel.write_text("owned", encoding="utf-8")

    outside_session = tmp_path / "outside-session"
    outside_target = outside_session / "cache" / "dataset"
    outside_target.mkdir(parents=True)
    outside_sentinel = outside_target / "outside.txt"
    outside_sentinel.write_text("outside", encoding="utf-8")

    detached_session = tmp_path / "session.detached"
    original_remove = executor.remove_confined_tree
    swapped = False

    def swap_after_validation(
        working_dir: str | Path,
        allowed_root: str | Path,
    ) -> bool:
        nonlocal swapped
        if not swapped:
            session_dir.rename(detached_session)
            session_dir.symlink_to(outside_session, target_is_directory=True)
            swapped = True
        return original_remove(working_dir, allowed_root)

    monkeypatch.setattr(executor, "remove_confined_tree", swap_after_validation)

    with pytest.raises(
        executor.ArtifactPathError,
        match="Cache cleanup could not be confined",
    ) as exc_info:
        executor._remove_cache_path(target, cache_root)

    assert swapped is True
    assert outside_sentinel.read_text(encoding="utf-8") == "outside"
    assert (detached_session / "cache/dataset/owned.txt").read_text(
        encoding="utf-8"
    ) == "owned"
    assert "outside.txt" not in str(exc_info.value)
    assert "owned.txt" not in str(exc_info.value)


def test_clear_previous_joint_rigger_rejects_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    cache_root = session_dir / "cache"
    joint_rigger_dir = cache_root / "joint_rigger"
    joint_rigger_dir.mkdir(parents=True)
    owned_sentinel = joint_rigger_dir / "owned.usd"
    owned_sentinel.write_text("owned", encoding="utf-8")

    outside_cache = tmp_path / "outside-cache"
    outside_joint_rigger = outside_cache / "joint_rigger"
    outside_joint_rigger.mkdir(parents=True)
    outside_sentinel = outside_joint_rigger / "outside.usd"
    outside_sentinel.write_text("outside", encoding="utf-8")

    detached_cache = session_dir / "cache.detached"
    original_remove = executor.remove_confined_tree
    swapped = False

    def swap_cache_ancestor(
        working_dir: str | Path,
        allowed_root: str | Path,
    ) -> bool:
        nonlocal swapped
        if not swapped:
            cache_root.rename(detached_cache)
            cache_root.symlink_to(outside_cache, target_is_directory=True)
            swapped = True
        return original_remove(working_dir, allowed_root)

    monkeypatch.setattr(executor, "remove_confined_tree", swap_cache_ancestor)

    with pytest.raises(
        executor.ArtifactPathError,
        match="Cache cleanup could not be confined",
    ) as exc_info:
        executor._clear_previous_joint_rigger_artifacts(session_dir)

    assert swapped is True
    assert outside_sentinel.read_text(encoding="utf-8") == "outside"
    assert (detached_cache / "joint_rigger/owned.usd").read_text(
        encoding="utf-8"
    ) == "owned"
    assert "outside.usd" not in str(exc_info.value)
    assert "owned.usd" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_inflight_cache_upload_isolated_from_successor_workspace(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "cache-upload-fence"
    run_id = "c" * 32
    session_dir = manager.get_session_dir(session_id)
    predictions_dir = session_dir / "cache" / "predictions"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.jsonl").write_bytes(b"run-a")
    sync_started = asyncio.Event()
    allow_sync = asyncio.Event()
    uploaded: dict[str, bytes] = {}

    async def blocking_sync(
        _session_id: str,
        *,
        prefix: str = "",
        overwrite: bool = False,
    ) -> int:
        del overwrite
        sync_started.set()
        await allow_sync.wait()
        uploaded[prefix] = (session_dir / prefix / "predictions.jsonl").read_bytes()
        return 1

    manager.sync_to_store = blocking_sync  # type: ignore[method-assign]
    publication = asyncio.create_task(
        executor._publish_cache_publications(
            manager,
            session_id,
            run_id,
            ["predict"],
        )
    )
    await sync_started.wait()

    assert not predictions_dir.exists()
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.jsonl").write_bytes(b"run-b")
    allow_sync.set()
    produced, bindings = await publication

    prefix = f"artifacts/run_cache/{run_id}/cache/predictions/"
    assert produced == {"predictions"}
    assert bindings == {"predictions": run_id}
    assert uploaded == {prefix: b"run-a"}
    assert (predictions_dir / "predictions.jsonl").read_bytes() == b"run-b"


@pytest.mark.asyncio
async def test_execute_pipeline_success_failure_and_sync_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "joint-service-result-credential-713"
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
        predictions_dir = manager.get_session_dir(session_id) / "cache" / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (predictions_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["predict"],
            step_results={
                "predict": {
                    "predictions_count": 2,
                    "diagnostics": {"api_key": sentinel},
                },
                "apply_joint_rigger": {
                    "authored_joint_count": 0,
                    "joint_rigger_status": f"api_key={sentinel}",
                    "apply_joint_rigger_skipped": True,
                },
            },
            raw_result={
                "build_dataset_usd_result": {"num_prims": 3, "num_images": 4},
                "config_dict": {"api_key": sentinel},
            },
        )

    monkeypatch.setattr(executor, "arun_pipeline", good_pipeline)
    first_run_id = manager.current_run_id
    await executor.execute_pipeline_async(
        session_id,
        first_run_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["predict"],
    )

    assert manager.metadata["status"] == "completed"
    assert manager.metadata["results"]["predictions_made"] == 2
    assert manager.metadata["results"]["joint_rigger_artifact_keys"] == {}
    assert "joint_rigger_publication_id" not in manager.metadata["results"]
    assert sentinel not in caplog.text
    assert sentinel not in repr(manager.metadata)
    predictions_prefix = f"artifacts/run_cache/{first_run_id}/cache/predictions/"
    assert manager.sync_calls == [predictions_prefix]
    assert manager.sync_overwrite == [False]
    assert "completed" not in manager.statuses_seen_during_sync

    manager.sync_calls.clear()
    manager.sync_overwrite.clear()
    manager.statuses_seen_during_sync.clear()
    manager.current_run_id = "b" * 32
    manager.metadata["status"] = "pending"
    stale_rigger_dir = manager.get_session_dir(session_id) / "cache" / "joint_rigger"
    stale_rigger_dir.mkdir(parents=True)
    (stale_rigger_dir / "rigged.usd").write_text("#stale\n", encoding="utf-8")
    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["predict"],
    )
    assert manager.sync_calls == [f"artifacts/run_cache/{'b' * 32}/cache/predictions/"]
    assert manager.metadata["results"]["joint_rigger_artifacts"] == {
        "joint_rigger_output": False,
        "joint_rigger_diagnostics": False,
        "joint_rigger_validation": False,
    }
    assert manager.metadata["results"]["joint_rigger_artifact_keys"] == {}
    assert "joint_rigger_publication_id" not in manager.metadata["results"]
    assert not stale_rigger_dir.exists()
    assert "completed" not in manager.statuses_seen_during_sync

    async def rigger_pipeline(_params):
        rigger_dir = manager.get_session_dir(session_id) / "cache" / "joint_rigger"
        rigger_dir.mkdir(parents=True)
        (rigger_dir / "rigged.usdz").write_bytes(b"PK\x03\x04owned-core")
        (rigger_dir / "joint_rigger_diagnostics.json").write_text(
            "{}", encoding="utf-8"
        )
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["apply_joint_rigger"],
            step_results={
                "apply_joint_rigger": {
                    "joint_rigger_status": "authored",
                    "authored_joint_count": 2,
                }
            },
            raw_result={},
        )

    manager.sync_calls.clear()
    manager.sync_overwrite.clear()
    manager.statuses_seen_during_sync.clear()
    manager.current_run_id = "c" * 32
    manager.metadata["status"] = "pending"
    monkeypatch.setattr(executor, "arun_pipeline", rigger_pipeline)
    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
    )
    assert manager.metadata["results"]["joint_rigger_artifacts"] == {
        "joint_rigger_output": True,
        "joint_rigger_diagnostics": True,
        "joint_rigger_validation": False,
    }
    publication_id = manager.metadata["results"]["joint_rigger_publication_id"]
    assert len(publication_id) == 32
    assert int(publication_id, 16) >= 0
    publication_prefix = f"artifacts/joint_rigger/{publication_id}"
    assert manager.metadata["results"]["joint_rigger_artifact_keys"] == {
        "joint_rigger_output": f"{publication_prefix}/rigged.usdz",
        "joint_rigger_diagnostics": (
            f"{publication_prefix}/joint_rigger_diagnostics.json"
        ),
    }
    assert manager.sync_overwrite == [True]
    assert manager.sync_calls[-1] == f"{publication_prefix}/"
    assert "completed" not in manager.statuses_seen_during_sync

    manager.fail_sync = True
    manager.current_run_id = "d" * 32
    manager.metadata["status"] = "pending"
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
            percent=10,
        )
    )
    monkeypatch.setattr(executor, "arun_pipeline", good_pipeline)
    with pytest.raises(RuntimeError, match="Failed to sync result artifacts"):
        await executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
            only_steps=["predict"],
        )
    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "artifact_sync"
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["failed_step"] == "artifact_sync"

    async def bad_pipeline(_params):
        return SimpleNamespace(success=False, error="bad", completed_steps=[])

    monkeypatch.setattr(executor, "arun_pipeline", bad_pipeline)
    with pytest.raises(RuntimeError, match="Pipeline failed"):
        bad_manager = _Manager(tmp_path / "bad")
        await executor.execute_pipeline_async(
            "bad",
            bad_manager.current_run_id,
            {"project": {"name": "test"}},
            bad_manager,
        )


@pytest.mark.asyncio
async def test_completed_metadata_survives_event_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "completed-event-failure"

    async def successful_pipeline(_params):
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["identify_asset"],
            step_results={},
            raw_result={},
        )

    class FailingBus:
        @staticmethod
        def get_snapshot(_session_id):
            return {"status": "running"}

        @staticmethod
        async def emit(_event):
            raise RuntimeError("event transport failed")

    monkeypatch.setattr(executor, "arun_pipeline", successful_pipeline)
    monkeypatch.setattr(executor, "get_event_bus", lambda: FailingBus())

    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
        only_steps=["identify_asset"],
    )

    assert manager.metadata["status"] == "completed"
    assert manager.metadata.get("error") is None


@pytest.mark.asyncio
async def test_cancellation_after_completion_does_not_regress_terminal_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "cancel-after-completion"
    event_started = asyncio.Event()

    async def successful_pipeline(_params):
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["identify_asset"],
            step_results={},
            raw_result={},
        )

    class BlockingBus:
        @staticmethod
        def get_snapshot(_session_id):
            return {"status": "running"}

        @staticmethod
        async def emit(_event):
            event_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(executor, "arun_pipeline", successful_pipeline)
    monkeypatch.setattr(executor, "get_event_bus", lambda: BlockingBus())
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
            only_steps=["identify_asset"],
        )
    )
    await event_started.wait()
    assert manager.metadata["status"] == "completed"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.metadata["status"] == "completed"
    assert manager.metadata.get("error") is None


@pytest.mark.asyncio
async def test_execute_pipeline_binds_current_joint_output_across_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "reused"
    current_filename = "rigged.usdz"
    publication_ids: set[str] = set()

    async def rigger_pipeline(_params):
        rigger_dir = manager.get_session_dir(session_id) / "cache" / "joint_rigger"
        rigger_dir.mkdir(parents=True)
        output = rigger_dir / current_filename
        if output.suffix == ".usd":
            output.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
        else:
            output.write_bytes(current_filename.encode())
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["apply_joint_rigger"],
            step_results={
                "apply_joint_rigger": {
                    "joint_rigger_status": "authored",
                    "authored_joint_count": 1,
                }
            },
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", rigger_pipeline)
    for run_index, filename in enumerate(
        ("rigged.usdz", "rigged.usd", "rigged.usdz"),
        start=10,
    ):
        current_filename = filename
        manager.current_run_id = f"{run_index:x}" * 32
        manager.metadata["status"] = "pending"
        await executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )
        publication_id = manager.metadata["results"]["joint_rigger_publication_id"]
        publication_ids.add(publication_id)
        published_filename = "rigged.usdz" if filename == "rigged.usd" else filename
        assert manager.metadata["results"]["joint_rigger_artifact_keys"] == {
            "joint_rigger_output": (
                f"artifacts/joint_rigger/{publication_id}/{published_filename}"
            )
        }
        rigger_dir = manager.get_session_dir(session_id) / "cache" / "joint_rigger"
        expected_files = (
            ["rigged.usd", "rigged.usdz"] if filename == "rigged.usd" else [filename]
        )
        assert sorted(path.name for path in rigger_dir.iterdir()) == expected_files
    assert len(publication_ids) == 3

    async def no_output_pipeline(_params):
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["predict"],
            step_results={"predict": {"predictions_count": 1}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", no_output_pipeline)
    manager.current_run_id = "d" * 32
    manager.metadata["status"] = "pending"
    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
    )
    assert manager.metadata["results"]["joint_rigger_artifact_keys"] == {}
    assert "joint_rigger_publication_id" not in manager.metadata["results"]
    assert manager.metadata["results"]["joint_rigger_artifacts"] == {
        "joint_rigger_output": False,
        "joint_rigger_diagnostics": False,
        "joint_rigger_validation": False,
    }
    assert not (manager.get_session_dir(session_id) / "cache" / "joint_rigger").exists()

    current_filename = "rigged.usd"
    monkeypatch.setattr(executor, "arun_pipeline", rigger_pipeline)
    manager.current_run_id = "e" * 32
    manager.metadata.update({"status": "pending", "results": {}})
    await executor.execute_pipeline_async(
        session_id,
        manager.current_run_id,
        {"project": {"name": "test"}},
        manager,
    )

    async def failed_pipeline(_params):
        return SimpleNamespace(success=False, error="failed rerun", completed_steps=[])

    monkeypatch.setattr(executor, "arun_pipeline", failed_pipeline)
    manager.current_run_id = "f" * 32
    manager.metadata.update({"status": "pending", "results": {}})
    with pytest.raises(RuntimeError, match="^Pipeline failed$"):
        await executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )
    assert manager.metadata["results"] == {}
    assert not (manager.get_session_dir(session_id) / "cache" / "joint_rigger").exists()


@pytest.mark.asyncio
async def test_stale_executor_cannot_clear_or_complete_newer_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.current_run_id = "b" * 32
    session_id = "reused"
    output = manager.get_session_dir(session_id) / "cache/joint_rigger/rigged.usd"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"new-run")

    async def should_not_run(_params):
        raise AssertionError("stale executor reached the pipeline")

    monkeypatch.setattr(executor, "arun_pipeline", should_not_run)
    with pytest.raises(executor.PipelineRunSupersededError):
        await executor.execute_pipeline_async(
            session_id,
            "a" * 32,
            {"project": {"name": "test"}},
            manager,
        )

    assert output.read_bytes() == b"new-run"
    assert manager.metadata["status"] == "pending"


@pytest.mark.asyncio
async def test_run_claim_guard_fails_closed_on_lost_lease(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    manager.run_claim_heartbeat_seconds = 0.01
    manager.renew_current_run = False
    with pytest.raises(executor.PipelineRunSupersededError, match="lost its lease"):
        await asyncio.wait_for(
            executor.maintain_run_claim(
                manager,
                "lease-lost",
                manager.current_run_id,
            ),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_run_claim_guard_retries_cancel_poll_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.run_claim_heartbeat_seconds = 0.03
    manager.renew_current_run = False
    cancellation_polls = 0

    async def flaky_cancellation_poll(_session_id: str, _run_id: str) -> bool:
        nonlocal cancellation_polls
        cancellation_polls += 1
        if cancellation_polls == 1:
            raise RuntimeError("store temporarily unavailable")
        return False

    monkeypatch.setattr(
        executor,
        "_RUN_CANCELLATION_POLL_SECONDS",
        0.005,
    )
    monkeypatch.setattr(
        manager,
        "is_cancellation_accepted",
        flaky_cancellation_poll,
    )
    with pytest.raises(executor.PipelineRunSupersededError, match="lost its lease"):
        await asyncio.wait_for(
            executor.maintain_run_claim(
                manager,
                "lease-lost",
                manager.current_run_id,
            ),
            timeout=1,
        )
    assert cancellation_polls > 1


@pytest.mark.asyncio
async def test_cancelled_executor_emits_cancelled_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "cancelled-run"
    pipeline_started = asyncio.Event()
    pipeline_release = asyncio.Event()
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
        )
    )

    async def blocked_pipeline(_params):
        pipeline_started.set()
        await pipeline_release.wait()
        session_dir = manager.get_session_dir(session_id)
        dataset_dir = session_dir / "cache" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
        (session_dir / "cache" / ".pipeline_state.json").write_text(
            json.dumps(
                {
                    "completed_steps": [
                        "build_dataset_usd",
                        "build_dataset_prepare_dataset",
                    ],
                    "failed_steps": [],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(executor, "arun_pipeline", blocked_pipeline)
    task = asyncio.create_task(
        executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )
    )
    await pipeline_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    pipeline_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["cache_publications"] == {"dataset": "a" * 32}
    assert manager.sync_calls == [f"artifacts/run_cache/{'a' * 32}/cache/dataset/"]
    assert manager.sync_overwrite == [False]
    assert manager.statuses_seen_during_sync == ["pending"]
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


@pytest.mark.asyncio
async def test_failure_sync_error_does_not_mask_primary_pipeline_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.fail_sync = True

    async def failed_pipeline(_params):
        return SimpleNamespace(success=False, error="primary pipeline failure")

    monkeypatch.setattr(executor, "arun_pipeline", failed_pipeline)
    with pytest.raises(RuntimeError, match="^Pipeline failed$"):
        await executor.execute_pipeline_async(
            "primary-failure",
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["status"] == "failed"
    assert manager.metadata["error"] == "joint_pipeline_execution_failed"
    assert manager.metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "joint_pipeline_execution_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "primary pipeline failure" not in json.dumps(manager.metadata)
    assert manager.sync_calls == []


@pytest.mark.asyncio
async def test_pipeline_exception_is_replaced_without_secret_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "never-render-this-joint-pipeline-exception-credential"
    caplog.set_level(logging.DEBUG, logger=executor.__name__)

    async def rejected_pipeline(_params: PipelineInput) -> Never:
        rejected_value = sentinel
        raise RuntimeError(f"upstream rejected {rejected_value}")

    monkeypatch.setattr(executor, "arun_pipeline", rejected_pipeline)
    with pytest.raises(RuntimeError, match="^Pipeline failed$") as exc_info:
        await executor.execute_pipeline_async(
            "secret-graph",
            manager.current_run_id,
            {"vlm": {"api_key": sentinel}},
            manager,
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    observable = repr(
        (
            caplog.text,
            manager.metadata,
            _executor_traceback_locals(exc_info.value),
        )
    )
    assert sentinel not in observable
    assert manager.metadata["error"] == "joint_pipeline_execution_failed"
    assert manager.metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "joint_pipeline_execution_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_artifact_sync_exception_uses_fixed_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "never-render-this-joint-sync-exception-credential"
    caplog.set_level(logging.DEBUG, logger=executor.__name__)

    async def successful_pipeline(_params: PipelineInput) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["apply_joint_rigger"],
            step_results={"apply_joint_rigger": {}},
            raw_result={},
        )

    async def publish_cache(
        _session_manager: object,
        _session_id: str,
        _run_id: str,
        _only_steps: list[str] | None,
        namespaces: tuple[str, ...] | None = None,
    ) -> tuple[set[str], dict[str, str]]:
        del namespaces
        return set(), {}

    async def rejected_sync(
        _session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> Never:
        del prefix, overwrite
        rejected_value = sentinel
        raise RuntimeError(f"store rejected {rejected_value}")

    monkeypatch.setattr(executor, "arun_pipeline", successful_pipeline)
    monkeypatch.setattr(executor, "_publish_cache_publications", publish_cache)
    monkeypatch.setattr(
        executor,
        "_materialize_joint_rigger_download_usdz",
        lambda _session_dir: None,
    )
    monkeypatch.setattr(
        executor,
        "_joint_rigger_artifact_keys",
        lambda _session_dir: {"joint_rigger_output": "cache/joint_rigger/rigged.usdz"},
    )
    monkeypatch.setattr(
        executor,
        "_publish_joint_rigger_artifacts",
        lambda *_args: {
            "joint_rigger_output": "artifacts/joint_rigger/publication/rigged.usdz"
        },
    )
    monkeypatch.setattr(manager, "sync_to_store", rejected_sync)

    with pytest.raises(
        RuntimeError,
        match="^Failed to sync result artifacts$",
    ) as exc_info:
        await executor.execute_pipeline_async(
            "sync-secret-graph",
            manager.current_run_id,
            {"vlm": {"api_key": sentinel}},
            manager,
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    observable = repr(
        (
            caplog.text,
            manager.metadata,
            _executor_traceback_locals(exc_info.value),
        )
    )
    assert sentinel not in observable
    assert manager.metadata["error"] == "joint_pipeline_artifact_sync_failed"
    assert manager.metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "joint_pipeline_artifact_sync_failed",
        "phase": "sync_upload",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_failure_prerequisite_sync_is_fenced_after_supersession(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)

    async def superseded_failure(_params):
        manager.current_run_id = "b" * 32
        return SimpleNamespace(success=False, error="stale pipeline failure")

    monkeypatch.setattr(executor, "arun_pipeline", superseded_failure)
    with pytest.raises(RuntimeError, match="^Pipeline failed$"):
        await executor.execute_pipeline_async(
            "superseded-failure",
            "a" * 32,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.sync_calls == []
    assert manager.metadata["status"] == "pending"


@pytest.mark.asyncio
async def test_registry_holds_run_until_thread_backed_pipeline_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.run_claim_heartbeat_seconds = 0.01
    session_id = "thread-backed-cancel"
    run_id = manager.current_run_id
    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_finished = threading.Event()

    def blocking_pipeline():
        worker_started.set()
        assert worker_release.wait(timeout=2)
        worker_finished.set()
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=[],
            step_results={},
            raw_result={},
        )

    async def threaded_pipeline(_params):
        return await asyncio.to_thread(blocking_pipeline)

    original_release_run = manager.release_run

    async def observed_release(session_id: str, run_id: str) -> bool:
        assert worker_finished.is_set()
        return await original_release_run(session_id, run_id)

    monkeypatch.setattr(executor, "arun_pipeline", threaded_pipeline)
    monkeypatch.setattr(manager, "release_run", observed_release)
    registry = JobRegistry(max_concurrent=1)

    async def finish_run() -> None:
        await executor.finalize_pipeline_run(
            manager,
            session_id,
            run_id,
        )

    await registry.register(
        session_id,
        executor.execute_pipeline_async(
            session_id,
            run_id,
            {"project": {"name": "test"}},
            manager,
        ),
        run_id=run_id,
        liveness_guard=executor.maintain_run_claim(
            manager,
            session_id,
            run_id,
        ),
        on_finish=finish_run,
    )
    await asyncio.to_thread(worker_started.wait, 1)

    cancellation = asyncio.create_task(registry.cancel(session_id))
    await asyncio.sleep(0.05)
    assert registry.is_running(session_id)
    assert manager.released_runs == []

    worker_release.set()
    assert await cancellation is True
    assert worker_finished.is_set()
    assert manager.metadata["status"] == "cancelled"
    assert manager.released_runs == ["a" * 32]


@pytest.mark.asyncio
async def test_finalize_pipeline_run_cancels_nonterminal_owner_before_release(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "queued-cancel"
    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.seed_pending_session(session_id)

    await executor.finalize_pipeline_run(
        manager,
        session_id,
        manager.current_run_id,
    )

    assert manager.metadata["status"] == "cancelled"
    assert manager.released_runs == ["a" * 32]
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


@pytest.mark.asyncio
async def test_finalize_pipeline_run_preserves_terminal_metadata(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    manager.metadata["status"] = "completed"

    await executor.finalize_pipeline_run(
        manager,
        "completed-run",
        manager.current_run_id,
    )

    assert manager.metadata["status"] == "completed"
    assert manager.released_runs == ["a" * 32]


@pytest.mark.asyncio
async def test_publication_failure_is_recorded_as_terminal_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "publication-failure"

    async def pipeline_with_invalid_raw_output(_params):
        output = manager.get_session_dir(session_id) / "cache/joint_rigger/rigged.usd"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"not-usd")
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["apply_joint_rigger"],
            step_results={"apply_joint_rigger": {}},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", pipeline_with_invalid_raw_output)
    with pytest.raises(RuntimeError, match="Failed to publish result artifacts"):
        await executor.execute_pipeline_async(
            session_id,
            manager.current_run_id,
            {"project": {"name": "test"}},
            manager,
        )

    assert manager.metadata["status"] == "failed"
    assert manager.metadata["failed_step"] == "artifact_publication"


def test_publish_joint_rigger_artifacts_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cache/joint_rigger/rigged.usdz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"published bytes")

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(executor.os, "link", fail_link)
    published = executor._publish_joint_rigger_artifacts(
        tmp_path,
        {"joint_rigger_output": "cache/joint_rigger/rigged.usdz"},
        "a" * 32,
    )

    published_key = f"artifacts/joint_rigger/{'a' * 32}/rigged.usdz"
    assert published == {"joint_rigger_output": published_key}
    assert (tmp_path / published_key).read_bytes() == b"published bytes"


def test_publish_joint_rigger_raw_output_includes_validated_dependency_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    from pxr import Sdf, Usd, UsdShade, UsdUtils

    source = tmp_path / "cache/joint_rigger/rigged.usd"
    texture = source.parent / "rigged_assets/textures/albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"fake-png")
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("rigged_assets/textures/albedo.png")
    )
    stage.GetRootLayer().Save()

    executor._materialize_joint_rigger_download_usdz(tmp_path)
    source_keys = executor._joint_rigger_artifact_keys(tmp_path)
    assert source_keys == {"joint_rigger_output": "cache/joint_rigger/rigged.usdz"}

    publication_id = "b" * 32
    published = executor._publish_joint_rigger_artifacts(
        tmp_path,
        source_keys,
        publication_id,
    )

    publication = tmp_path / f"artifacts/joint_rigger/{publication_id}"
    assert published == {
        "joint_rigger_output": (f"artifacts/joint_rigger/{publication_id}/rigged.usdz")
    }
    assert (publication / "rigged_assets/textures/albedo.png").read_bytes() == (
        b"fake-png"
    )
    package = publication / "rigged.usdz"
    assert Usd.Stage.Open(str(package))
    assert not UsdUtils.ComputeAllDependencies(str(package))[2]
    with zipfile.ZipFile(package) as archive:
        texture_members = [
            name for name in archive.namelist() if name.endswith("albedo.png")
        ]
        assert len(texture_members) == 1
        assert archive.read(texture_members[0]) == b"fake-png"


def test_publish_joint_rigger_raw_output_rejects_missing_dependency(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    sentinel = "never-render-this-joint-closure-credential"
    session_dir = tmp_path / f"client_secret={sentinel}"
    source = session_dir / "cache/joint_rigger/rigged.usd"
    source.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(source))
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("rigged_assets/missing.png")
    )
    stage.GetRootLayer().Save()

    with pytest.raises(
        RuntimeError,
        match="^Could not validate Joint Rigger raw USD closure$",
    ) as exc_info:
        executor._materialize_joint_rigger_download_usdz(session_dir)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(_executor_traceback_locals(exc_info.value))
    assert not (source.parent / "rigged.usdz").exists()


@pytest.mark.asyncio
async def test_emit_pipeline_failed_without_snapshot_does_not_seed_state() -> None:
    session_id = "no-snapshot"
    bus = get_event_bus()
    bus.cleanup_session(session_id)

    await executor._emit_pipeline_failed(session_id, "artifact_sync", "sync failed")

    assert bus.get_snapshot(session_id) is None


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
