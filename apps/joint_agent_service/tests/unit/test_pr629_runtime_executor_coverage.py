# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ...service.runtime.registry import JobRegistry
from ...service.session.manager import SessionManager
from ...service.workers import executor


@pytest.mark.asyncio
async def test_registry_job_completion_survives_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = JobRegistry(max_concurrent=1)

    async def pipeline() -> None:
        return None

    async def fail_cleanup() -> None:
        raise RuntimeError("normal cleanup failed")

    with caplog.at_level(logging.ERROR):
        await registry.register("normal-cleanup", pipeline(), on_finish=fail_cleanup)
        task = registry.get_task("normal-cleanup")
        assert task is not None
        await task

    assert registry.get_task("normal-cleanup") is None
    assert registry.active_count == 0
    cleanup_record = next(
        record
        for record in caplog.records
        if "Pipeline cleanup failed for normal-c" in record.getMessage()
    )
    assert isinstance(cleanup_record.exc_info[1], RuntimeError)


@pytest.mark.asyncio
async def test_registry_prestart_cancellation_survives_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    cleanup_attempted = asyncio.Event()

    async def pipeline() -> None:
        await asyncio.Event().wait()

    async def fail_cleanup() -> None:
        cleanup_attempted.set()
        raise RuntimeError("prestart cleanup failed")

    pipeline_coro = pipeline()
    with caplog.at_level(logging.ERROR):
        await registry.register(
            "prestart-cancel",
            pipeline_coro,
            on_finish=fail_cleanup,
        )
        task = registry.get_task("prestart-cancel")
        assert task is not None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(cleanup_attempted.wait(), timeout=1)

    assert pipeline_coro.cr_frame is None
    assert registry.get_task("prestart-cancel") is None
    cleanup_record = next(
        record
        for record in caplog.records
        if "Pipeline cleanup failed for prestart" in record.getMessage()
    )
    assert isinstance(cleanup_record.exc_info[1], RuntimeError)


@pytest.mark.asyncio
async def test_registry_prestart_cleanup_consumes_cancellation() -> None:
    registry = JobRegistry(max_concurrent=1)
    cleanup_attempted = asyncio.Event()

    async def pipeline() -> None:
        await asyncio.Event().wait()

    async def cancelled_cleanup() -> None:
        cleanup_attempted.set()
        raise asyncio.CancelledError

    pipeline_coro = pipeline()
    await registry.register(
        "prestart-cancelled-cleanup",
        pipeline_coro,
        on_finish=cancelled_cleanup,
    )
    task = registry.get_task("prestart-cancelled-cleanup")
    assert task is not None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(cleanup_attempted.wait(), timeout=1)
    await asyncio.sleep(0)

    assert pipeline_coro.cr_frame is None
    assert registry.get_task("prestart-cancelled-cleanup") is None
    assert registry._cleanup_events == {}


@pytest.mark.asyncio
async def test_repeated_cancel_during_finish_drains_terminal_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    run_id = "a" * 32
    await manager.create_session(session_id)
    assert await manager.reserve_run(session_id, run_id)
    assert await manager.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )
    terminalize_started = asyncio.Event()
    finish_terminalize = asyncio.Event()
    pipeline_started = asyncio.Event()
    original_terminalize = manager.terminalize_and_release_run

    async def slow_terminalize(
        selected_session_id: str,
        selected_run_id: str,
        updates: dict,
    ) -> bool:
        terminalize_started.set()
        await finish_terminalize.wait()
        return await original_terminalize(
            selected_session_id,
            selected_run_id,
            updates,
        )

    async def pipeline() -> None:
        pipeline_started.set()
        await asyncio.Event().wait()

    async def finish_run() -> None:
        await executor.finalize_pipeline_run(manager, session_id, run_id)

    monkeypatch.setattr(manager, "terminalize_and_release_run", slow_terminalize)
    registry = JobRegistry(max_concurrent=1)
    await registry.register(session_id, pipeline(), on_finish=finish_run)
    task = registry.get_task(session_id)
    assert task is not None
    await pipeline_started.wait()

    task.cancel()
    await terminalize_started.wait()
    task.cancel()
    finish_terminalize.set()
    await asyncio.gather(task, return_exceptions=True)

    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["error"] == "Pipeline run was cancelled before completion"
    assert "active_run_id" not in metadata
    assert registry._cleanup_events == {}
    assert registry.registered_count == 0
    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_repeated_cancel_during_record_failure_drains_terminal_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    run_id = "a" * 32
    await manager.create_session(session_id)
    assert await manager.reserve_run(session_id, run_id)
    assert await manager.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )
    pipeline_started = asyncio.Event()
    finish_pipeline = asyncio.Event()
    terminalize_started = asyncio.Event()
    finish_terminalize = asyncio.Event()
    original_terminalize = manager.terminalize_and_release_run

    async def slow_pipeline(_params):
        pipeline_started.set()
        await finish_pipeline.wait()

    async def slow_terminalize(
        selected_session_id: str,
        selected_run_id: str,
        updates: dict,
    ) -> bool:
        terminalize_started.set()
        await finish_terminalize.wait()
        return await original_terminalize(
            selected_session_id,
            selected_run_id,
            updates,
        )

    async def finish_run() -> None:
        await executor.finalize_pipeline_run(manager, session_id, run_id)

    monkeypatch.setattr(executor, "arun_pipeline", slow_pipeline)
    monkeypatch.setattr(manager, "terminalize_and_release_run", slow_terminalize)
    registry = JobRegistry(max_concurrent=1)
    await registry.register(
        session_id,
        executor.execute_pipeline_async(
            session_id,
            run_id,
            {"project": {"name": "repeated-cancel"}},
            manager,
        ),
        on_finish=finish_run,
    )
    task = registry.get_task(session_id)
    assert task is not None
    await pipeline_started.wait()

    task.cancel()
    finish_pipeline.set()
    await terminalize_started.wait()
    task.cancel()
    finish_terminalize.set()
    await asyncio.gather(task, return_exceptions=True)

    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["error"] == "Pipeline run was cancelled"
    assert metadata["failed_step"] == "cancelled"
    assert "active_run_id" not in metadata
    assert "active_run_expires_at" not in metadata
    assert registry._cleanup_events == {}
    assert registry.registered_count == 0
    assert registry.active_count == 0


@pytest.mark.asyncio
async def test_registry_stale_generation_does_not_cancel_successor() -> None:
    registry = JobRegistry(max_concurrent=1)
    successor_started = asyncio.Event()
    release_successor = asyncio.Event()

    async def successor() -> None:
        successor_started.set()
        await release_successor.wait()

    session_id = "generation-fenced-cancel"
    successor_run_id = "b" * 32
    await registry.register(
        session_id,
        successor(),
        run_id=successor_run_id,
    )
    await successor_started.wait()

    assert not await registry.cancel(session_id, run_id="a" * 32)
    assert registry.is_running(session_id)

    release_successor.set()
    task = registry.get_task(session_id)
    assert task is not None
    await task
    assert not registry.is_running(session_id)


@pytest.mark.asyncio
async def test_registry_admission_is_generation_scoped_and_transfers() -> None:
    registry = JobRegistry(max_concurrent=1)
    run_a = "a" * 32
    run_b = "b" * 32

    assert await registry.reserve_admission("mismatch", run_a)
    assert registry.is_running("mismatch")
    assert not await registry.reserve_admission("mismatch", run_b)
    rejected = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="belongs to another run"):
        await registry.register("mismatch", rejected, run_id=run_b)
    assert rejected.cr_frame is None
    assert not await registry.release_admission("mismatch", run_b)
    assert await registry.release_admission("mismatch", run_a)

    release = asyncio.Event()

    async def pipeline() -> None:
        await release.wait()

    assert await registry.reserve_admission("transfer", run_a)
    await registry.register("transfer", pipeline(), run_id=run_a)
    await asyncio.sleep(0)
    assert registry.is_running("transfer")
    assert not await registry.reserve_admission("transfer", run_b)
    assert not await registry.release_admission("transfer", run_a)
    release.set()
    task = registry.get_task("transfer")
    assert task is not None
    await task


def test_prepare_selective_cache_run_removes_only_invalidated_files(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    dataset_dir = session_dir / "cache" / "dataset"
    predictions_dir = session_dir / "cache" / "predictions"
    dataset_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)
    checkpoint_path = session_dir / "cache" / ".pipeline_state.json"
    checkpoint_path.write_text("{}\n", encoding="utf-8")

    stale_dataset = dataset_dir / "dataset.jsonl"
    retained_dataset = dataset_dir / "images.jsonl"
    stale_dataset.write_text("stale\n", encoding="utf-8")
    retained_dataset.write_text("retained\n", encoding="utf-8")

    stale_prediction_files = {
        predictions_dir / "articulation_candidates.json",
        predictions_dir / "articulation_candidates.html",
        predictions_dir / "articulation_candidate_adjudications.json",
        predictions_dir / "predictions.stats.json",
    }
    for path in stale_prediction_files:
        path.write_text("stale\n", encoding="utf-8")
    retained_predictions = predictions_dir / "predictions.jsonl"
    retained_predictions.write_text("retained\n", encoding="utf-8")

    executor._prepare_cache_namespaces_for_run(
        session_dir,
        ["build_dataset_prepare_dataset", "consistency_pass"],
    )

    assert not stale_dataset.exists()
    assert not checkpoint_path.exists()
    assert retained_dataset.read_text(encoding="utf-8") == "retained\n"
    assert all(not path.exists() for path in stale_prediction_files)
    assert retained_predictions.read_text(encoding="utf-8") == "retained\n"


def test_pipeline_checkpoint_progress_handles_invalid_documents(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "checkpoint"
    checkpoint_path = session_dir / "cache" / ".pipeline_state.json"
    assert executor._pipeline_checkpoint_progress(session_dir) == (set(), None)

    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("not-json", encoding="utf-8")
    assert executor._pipeline_checkpoint_progress(session_dir) == (set(), None)

    checkpoint_path.write_text("[]", encoding="utf-8")
    assert executor._pipeline_checkpoint_progress(session_dir) == (set(), None)

    checkpoint_path.write_text(
        json.dumps({"completed_steps": "bad", "failed_steps": "bad"}),
        encoding="utf-8",
    )
    assert executor._pipeline_checkpoint_progress(session_dir) == (set(), None)

    checkpoint_path.write_text(
        json.dumps(
            {
                "completed_steps": ["predict", None],
                "failed_steps": [None, "restore_usd"],
            }
        ),
        encoding="utf-8",
    )
    assert executor._pipeline_checkpoint_progress(session_dir) == (
        {"predict"},
        "restore_usd",
    )


@pytest.mark.asyncio
async def test_await_inflight_work_propagates_inner_cancellation() -> None:
    async def cancelled_work() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await executor._await_inflight_work(cancelled_work())


@pytest.mark.asyncio
async def test_await_inflight_work_drains_repeated_cancellation_and_inner_failure() -> (
    None
):
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def failing_work() -> None:
        started.set()
        try:
            await release.wait()
            raise RuntimeError("inflight work failed while draining")
        finally:
            finished.set()

    outer_task = asyncio.create_task(executor._await_inflight_work(failing_work()))
    await started.wait()

    outer_task.cancel()
    await asyncio.sleep(0)
    assert not outer_task.done()
    outer_task.cancel()
    await asyncio.sleep(0)
    assert not outer_task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await outer_task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_failure_cache_publication_is_not_bound_after_ownership_loss(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "ownership-loss"
    run_id = "a" * 32
    session_dir = tmp_path / session_id
    predictions_dir = session_dir / "cache" / "predictions"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    manager = SimpleNamespace(
        get_session_dir=lambda _session_id: session_dir,
        is_run_current=AsyncMock(side_effect=[True, False]),
        sync_to_store=AsyncMock(return_value=1),
        get_session_metadata=AsyncMock(
            side_effect=AssertionError("superseded publication must not be bound")
        ),
    )

    with caplog.at_level(logging.INFO):
        publications = await executor._persist_failure_prerequisites_without_masking(
            manager,
            session_id,
            run_id,
            ["predict"],
            {"predict"},
            False,
        )

    publication_prefix = f"artifacts/run_cache/{run_id}/cache/predictions/"
    publication_file = session_dir / publication_prefix / "predictions.jsonl"
    assert publications is None
    assert publication_file.read_text(encoding="utf-8") == "{}\n"
    manager.sync_to_store.assert_awaited_once_with(
        session_id,
        prefix=publication_prefix,
        overwrite=False,
    )
    manager.get_session_metadata.assert_not_awaited()
    assert "superseded during immutable cache publication" in caplog.text


@pytest.mark.asyncio
async def test_additional_cancellation_waits_for_failure_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    publication_started = asyncio.Event()
    publication_release = asyncio.Event()
    publication_finished = asyncio.Event()
    manager = SimpleNamespace(
        is_run_current=AsyncMock(return_value=True),
        get_session_metadata=AsyncMock(return_value={"cache_publications": {}}),
        get_session_dir=lambda _session_id: tmp_path,
    )

    async def publish_cache(*_args, **_kwargs):
        publication_started.set()
        await publication_release.wait()
        publication_finished.set()
        return {"predictions"}, {"predictions": "b" * 32}

    monkeypatch.setattr(executor, "_publish_cache_publications", publish_cache)

    with caplog.at_level(logging.DEBUG):
        preservation = asyncio.create_task(
            executor._persist_failure_prerequisites_without_masking(
                manager,
                "additional-cancel",
                "b" * 32,
                ["predict"],
                {"predict"},
                False,
            )
        )
        await publication_started.wait()
        preservation.cancel()
        await asyncio.sleep(0)
        assert not preservation.done()
        publication_release.set()
        assert await preservation is None

    assert publication_finished.is_set()
    assert "Additional cancellation arrived while preserving failure artifacts" in (
        caplog.text
    )


def test_joint_rigger_publication_failure_removes_partial_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cache" / "joint_rigger" / "rigged.usd"
    source.parent.mkdir(parents=True)
    source.write_text("#usda 1.0\n", encoding="utf-8")
    publication_id = "c" * 32
    publication_dir = tmp_path / "artifacts" / "joint_rigger" / publication_id

    with pytest.raises(RuntimeError, match="must be packaged as self-contained USDZ"):
        executor._publish_joint_rigger_artifacts(
            tmp_path,
            {"joint_rigger_output": "cache/joint_rigger/rigged.usd"},
            publication_id,
        )

    assert source.read_text(encoding="utf-8") == "#usda 1.0\n"
    assert not publication_dir.exists()
