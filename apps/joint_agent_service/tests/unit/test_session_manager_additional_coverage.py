# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ...service.routers import artifacts_router
from ...service.session import manager as manager_mod
from ...service.session.cache_publications import (
    PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD,
    PREDICTION_REPORT_PUBLICATION_ID_FIELD,
    cache_publication_path,
    cache_publication_prefix,
    prediction_report_publication_key,
)
from ...service.session.manager import SessionManager
from ...service.storage.base import METADATA_KEY
from ...service.storage.local_store import LocalSessionStore


class _FailingDeleteStore(LocalSessionStore):
    async def delete_session(self, session_id: str) -> None:
        raise RuntimeError("store delete failed")


class _StreamStore(LocalSessionStore):
    @property
    def kind(self) -> str:
        return "s3"

    async def open_read(self, session_id: str, key: str) -> io.BytesIO:
        return io.BytesIO((self._session_dir(session_id) / key).read_bytes())


def _sid() -> str:
    return str(uuid4())


def test_session_manager_suffix_helpers_cover_config_shapes() -> None:
    sid = _sid()
    assert manager_mod._validate_session_id(sid) == sid
    with pytest.raises(manager_mod.InvalidSessionIdError):
        manager_mod._validate_session_id("../bad")
    with pytest.raises(ValueError):
        prediction_report_publication_key("bad")

    total_steps = manager_mod._observed_total_steps(
        {
            "overall_progress": {"total_steps": 3},
            "completed_steps": ["bad", {"name": "predict"}],
            "current_step": {"name": "restore_usd"},
        }
    )
    assert total_steps == 8


@pytest.mark.asyncio
async def test_session_manager_missing_metadata_paths(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()

    await manager.update_session(sid, {"status": "running"})
    await manager.update_step_progress(sid, "predict", {"percent": 50})
    await manager.mark_step_completed(sid, "predict")
    await manager.add_preview_image(sid, "preview.png")
    await manager.update_preview_images(sid, ["preview.png"])
    await manager.request_cancellation(sid)
    assert not await manager.is_cancellation_accepted(sid, "bad")

    assert await manager.sync_to_store(sid) == 0
    orphaned_output = manager.get_session_dir(sid) / "cache/joint_rigger/rigged.usdz"
    orphaned_output.parent.mkdir(parents=True)
    orphaned_output.write_bytes(b"orphaned")
    assert await manager.get_artifact_path(sid, "joint_rigger_output") is None


@pytest.mark.asyncio
async def test_session_manager_can_overwrite_reused_store_keys(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    relative_path = Path("cache/joint_rigger/rigged.usdz")
    local_output = session_dir / relative_path
    local_output.parent.mkdir(parents=True, exist_ok=True)
    local_output.write_bytes(b"new")
    await store.put_bytes(sid, relative_path.as_posix(), b"old")

    assert await manager.sync_to_store(sid, prefix="cache/joint_rigger/") == 0
    assert (store._session_dir(sid) / relative_path).read_bytes() == b"old"
    assert (
        await manager.sync_to_store(
            sid,
            prefix="cache/joint_rigger/",
            overwrite=True,
        )
        == 1
    )
    assert (store._session_dir(sid) / relative_path).read_bytes() == b"new"


@pytest.mark.asyncio
async def test_session_manager_naive_datetimes_and_unknown_step(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    await manager.create_session(sid)

    metadata = await manager.get_session_metadata(sid)
    metadata["created_at"] = datetime.now().isoformat()
    metadata["current_step"] = {
        "name": "custom",
        "display_name": "custom",
        "started_at": datetime.now().isoformat(),
        "progress": {},
        "elapsed_seconds": 0,
    }
    metadata.pop("completed_steps", None)
    await manager.store.put_json(sid, METADATA_KEY, metadata)

    await manager.update_session(sid, {"status": "running"})
    await manager.update_step_progress(sid, "custom", {"percent": 42})
    await manager.mark_step_completed(sid, "custom")

    updated = await manager.get_session_metadata(sid)
    assert updated["elapsed_seconds"] >= 0
    assert updated["overall_progress"]["percent"] == 0
    assert updated["completed_steps"][0]["name"] == "custom"

    updated.pop("preview_images", None)
    await manager.store.put_json(sid, METADATA_KEY, updated)
    await manager.add_preview_image(sid, "preview.png")
    updated = await manager.get_session_metadata(sid)
    assert updated["preview_images"] == ["preview.png"]

    manager_mod.STEP_NUMBERS["unweighted"] = 9
    try:
        await manager.update_step_progress(sid, "unweighted", {"percent": 42})
        updated = await manager.get_session_metadata(sid)
        assert updated["overall_progress"]["percent"] == 0
        assert updated["overall_progress"]["total_steps"] == 9
    finally:
        manager_mod.STEP_NUMBERS.pop("unweighted", None)


@pytest.mark.asyncio
async def test_session_manager_completion_progress_expands_and_finishes(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)

    late_sid = _sid()
    await manager.create_session(late_sid)
    late_metadata = await manager.get_session_metadata(late_sid)
    late_metadata["overall_progress"]["total_steps"] = 3
    late_metadata["current_step"] = {
        "name": "late",
        "display_name": "Late",
        "started_at": datetime.now(UTC).isoformat(),
        "progress": {},
        "elapsed_seconds": 0,
    }
    await manager.store.put_json(late_sid, METADATA_KEY, late_metadata)

    manager_mod.STEP_NUMBERS["late"] = 9
    try:
        await manager.mark_step_completed(late_sid, "late")
        updated = await manager.get_session_metadata(late_sid)
        assert updated["overall_progress"]["total_steps"] == 9
        assert updated["overall_progress"]["current_step"] == 9
    finally:
        manager_mod.STEP_NUMBERS.pop("late", None)

    done_sid = _sid()
    await manager.create_session(done_sid)
    done_metadata = await manager.get_session_metadata(done_sid)
    done_metadata["overall_progress"]["total_steps"] = 1
    done_metadata["current_step"] = {
        "name": "custom",
        "display_name": "Custom",
        "started_at": datetime.now(UTC).isoformat(),
        "progress": {},
        "elapsed_seconds": 0,
    }
    await manager.store.put_json(done_sid, METADATA_KEY, done_metadata)

    await manager.mark_step_completed(done_sid, "custom")
    updated = await manager.get_session_metadata(done_sid)
    assert updated["overall_progress"]["percent"] == 100


@pytest.mark.asyncio
async def test_session_manager_artifact_suffix_fallbacks(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    legacy_metadata = await manager.get_session_metadata(sid)
    assert legacy_metadata is not None
    legacy_metadata.pop("cache_publications")
    await manager.store.put_json(sid, METADATA_KEY, legacy_metadata)
    assert await manager.get_artifact_path(sid, "predictions") is None

    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    candidates = session_dir / "cache" / "predictions" / "articulation_candidates.json"
    report = session_dir / "cache" / "predictions" / "articulation_candidates.html"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    rigged_usdz = session_dir / "cache" / "joint_rigger" / "rigged.usdz"
    rigged_usd = session_dir / "cache" / "joint_rigger" / "rigged.usd"
    predictions.write_text("{}\n", encoding="utf-8")
    candidates.write_text("{}", encoding="utf-8")
    report.write_text("<html></html>", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    rigged_usdz.parent.mkdir(parents=True)
    rigged_usdz.write_bytes(b"PK\x03\x04owned-core")
    rigged_usd.write_text("#usda 1.0\n", encoding="utf-8")

    assert await manager.get_artifact_path(sid, "predictions") is None
    await manager.update_session(sid, {"status": "completed"})
    assert await manager.get_artifact_path(sid, "predictions") == predictions
    assert await manager.get_artifact_path(sid, "articulation_candidates") == candidates
    assert await manager.get_artifact_path(sid, "articulation_report") == report
    assert await manager.get_artifact_path(sid, "dataset") == dataset
    assert await manager.get_artifact_path(sid, "joint_rigger_output") == rigged_usdz
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") == (
        "rigged.usdz"
    )
    rigged_usdz.unlink()
    assert await manager.get_artifact_path(sid, "joint_rigger_output") == rigged_usd
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") == (
        "rigged.usd"
    )
    assert await manager.get_artifact_path(sid, "missing") is None
    assert await manager.has_artifact(sid, "predictions") is True
    assert await manager.has_artifact(sid, "missing") is False


@pytest.mark.asyncio
async def test_session_manager_artifact_stream_paths(tmp_path: Path) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    legacy_metadata = await manager.get_session_metadata(sid)
    assert legacy_metadata is not None
    legacy_metadata.pop("cache_publications")
    await manager.store.put_json(sid, METADATA_KEY, legacy_metadata)
    await manager.update_session(sid, {"status": "completed"})
    await store.put_bytes(sid, "cache/predictions/predictions.jsonl", b"{}\n")
    stream = await manager.get_artifact_stream(sid, "predictions")
    assert stream is not None
    assert stream.read() == b"{}\n"

    preferred_key = "cache/joint_rigger/rigged.usdz"
    legacy_key = "cache/joint_rigger/rigged.usd"
    await store.put_bytes(sid, preferred_key, b"PK\x03\x04owned-core")
    await store.put_bytes(sid, legacy_key, b"#usda 1.0\n")
    stream = await manager.get_artifact_stream(sid, "joint_rigger_output")
    assert stream is not None
    assert stream.read() == b"PK\x03\x04owned-core"
    selected = await manager.get_artifact_stream_with_filename(
        sid,
        "joint_rigger_output",
    )
    assert selected is not None
    selected_stream, selected_filename = selected
    assert selected_stream.read() == b"PK\x03\x04owned-core"
    assert selected_filename == "rigged.usdz"
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") == (
        "rigged.usdz"
    )

    (store._session_dir(sid) / preferred_key).unlink()
    stream = await manager.get_artifact_stream(sid, "joint_rigger_output")
    assert stream is not None
    assert stream.read().startswith(b"#usda")
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") == (
        "rigged.usd"
    )
    assert await manager.has_artifact(sid, "joint_rigger_output") is True

    await store.put_bytes(sid, preferred_key, b"PK\x03\x04current-owned-core")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": legacy_key,
                }
            },
        },
    )
    selected = await manager.get_artifact_stream_with_filename(
        sid,
        "joint_rigger_output",
    )
    assert selected is not None
    selected_stream, selected_filename = selected
    assert selected_stream.read().startswith(b"#usda")
    assert selected_filename == "rigged.usd"

    await manager.update_session(
        sid,
        {
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": preferred_key,
                }
            }
        },
    )
    selected = await manager.get_artifact_stream_with_filename(
        sid,
        "joint_rigger_output",
    )
    assert selected is not None
    selected_stream, selected_filename = selected
    assert selected_stream.read() == b"PK\x03\x04current-owned-core"
    assert selected_filename == "rigged.usdz"

    await manager.update_session(
        sid,
        {"results": {"joint_rigger_artifact_keys": {}}},
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_path(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") is None
    assert await manager.has_artifact(sid, "joint_rigger_output") is False

    await manager.update_session(
        sid,
        {
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": "cache/joint_rigger/not-current.usdz",
                }
            }
        },
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_path(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") is None
    assert await manager.has_artifact(sid, "joint_rigger_output") is False

    await manager.update_session(
        sid,
        {"results": {"joint_rigger_artifact_keys": "invalid"}},
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None

    await manager.update_session(
        sid,
        {"results": {"joint_rigger_publication_id": "a" * 32}},
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None

    publication_id = "a" * 32
    published_raw_key = f"artifacts/joint_rigger/{publication_id}/rigged.usd"
    await store.put_bytes(sid, published_raw_key, b"#usda 1.0\n")
    await manager.update_session(
        sid,
        {
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": published_raw_key,
                },
                "joint_rigger_publication_id": publication_id,
            }
        },
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None

    await manager.update_session(
        sid,
        {
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": preferred_key,
                },
                "joint_rigger_publication_id": None,
            }
        },
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_path(sid, "joint_rigger_output") is None
    assert await manager.get_artifact_filename(sid, "joint_rigger_output") is None
    assert await manager.has_artifact(sid, "joint_rigger_output") is False

    await manager.update_session(
        sid,
        {
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": preferred_key,
                },
                "joint_rigger_publication_id": "invalid",
            }
        },
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None

    await manager.update_session(
        sid,
        {
            "status": "failed",
            "results": {
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": preferred_key,
                }
            },
        },
    )
    assert await manager.get_artifact_stream(sid, "joint_rigger_output") is None

    assert await manager.get_artifact_stream(sid, "unknown") is None


@pytest.mark.asyncio
async def test_cache_artifact_lookup_uses_bound_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    run_id = "a" * 32
    stable_path = session_dir / "cache/predictions/predictions.jsonl"
    bound_path = (
        session_dir
        / f"artifacts/run_cache/{run_id}/cache/predictions/predictions.jsonl"
    )
    bound_report = bound_path.with_name("report.html")
    standalone_report_id = "b" * 32
    standalone_report = session_dir / prediction_report_publication_key(
        standalone_report_id
    )
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    bound_path.parent.mkdir(parents=True)
    standalone_report.parent.mkdir(parents=True)
    stable_path.write_bytes(b"stale")
    bound_path.write_bytes(b"bound")
    bound_report.write_text("<html>bound</html>", encoding="utf-8")
    standalone_report.write_text("<html>older standalone</html>", encoding="utf-8")
    initial_metadata = await manager.get_session_metadata(sid)
    assert initial_metadata is not None
    assert initial_metadata["cache_publications"] == {}
    assert await manager.get_artifact_path(sid, "predictions") is None
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "cache_publications": {"predictions": run_id},
            PREDICTION_REPORT_PUBLICATION_ID_FIELD: standalone_report_id,
        },
    )

    assert await manager.get_artifact_path(sid, "predictions") == bound_path
    assert await manager.get_artifact_path(sid, "prediction_report") == bound_report
    monkeypatch.setattr(artifacts_router, "get_session_manager", lambda: manager)
    response = await artifacts_router.view_prediction_report(sid)
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/html"
    assert response.background is not None
    await response.background()
    await manager.update_session(sid, {"status": "running"})
    assert await manager.get_artifact_path(sid, "predictions") is None

    await manager.update_session(
        sid,
        {"status": "completed", "cache_publications": None},
    )
    assert await manager.get_artifact_path(sid, "predictions") is None

    metadata = await manager.get_session_metadata(sid)
    assert metadata is not None
    metadata.pop("cache_publications")
    metadata.pop(PREDICTION_REPORT_PUBLICATION_ID_FIELD)
    await manager.store.put_json(sid, METADATA_KEY, metadata)
    assert await manager.get_artifact_path(sid, "predictions") == stable_path

    assert await manager.reserve_run(sid, "b" * 32)
    assert await manager.get_artifact_path(sid, "predictions") is None
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 404

    assert await manager.release_run(sid, "b" * 32)
    await manager.update_session(sid, {"status": "pending"})
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 404

    await manager.update_session(sid, {"status": "completed"})
    assert await manager.reserve_legacy_cache_run(sid, "e" * 32)
    assert await manager.release_run(sid, "e" * 32)
    await manager.update_session(sid, {"cache_publications": None})
    assert not await manager.reserve_legacy_cache_run(sid, "f" * 32)


@pytest.mark.asyncio
async def test_split_store_legacy_report_generation_publishes_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / "store"
    manager = SessionManager(
        tmp_path / "local",
        store=LocalSessionStore(str(store_root)),
    )
    sid = _sid()
    session_dir = await manager.create_session(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    await manager.sync_to_store(sid, prefix="cache/")
    metadata = await manager.get_session_metadata(sid)
    assert metadata is not None
    metadata["status"] = "completed"
    metadata.pop("cache_publications")
    await manager.store.put_json(sid, METADATA_KEY, metadata)

    async def generate_snapshot(
        snapshot_dir: Path,
        snapshot_predictions: Path,
        snapshot_dataset: Path,
    ) -> None:
        assert snapshot_predictions.parent == snapshot_dir
        assert snapshot_dataset.parent == snapshot_dir
        (snapshot_dir / "report.html").write_text(
            "<html>snapshot</html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(artifacts_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        generate_snapshot,
    )

    response = await artifacts_router.view_prediction_report(sid)
    assert isinstance(response, StreamingResponse)
    assert (predictions.parent / "report.html").read_text(encoding="utf-8") == (
        "<html>snapshot</html>"
    )
    completed = await manager.get_session_metadata(sid)
    assert completed is not None
    publication_id = completed[PREDICTION_REPORT_PUBLICATION_ID_FIELD]
    publication_key = prediction_report_publication_key(publication_id)
    assert (session_dir / publication_key).read_text(encoding="utf-8") == (
        "<html>snapshot</html>"
    )
    assert (store_root / sid / publication_key).read_text(encoding="utf-8") == (
        "<html>snapshot</html>"
    )
    assert not (store_root / sid / "cache" / "predictions" / "report.html").exists()


@pytest.mark.asyncio
async def test_new_session_generates_report_from_bound_cache_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_root = tmp_path / "store"
    manager = SessionManager(
        tmp_path / "local",
        store=LocalSessionStore(str(store_root)),
    )
    sid = _sid()
    session_dir = await manager.create_session(sid)
    cache_run_id = "a" * 32
    cache_publications = {
        "dataset": cache_run_id,
        "predictions": cache_run_id,
    }

    stable_predictions = session_dir / "cache/predictions/predictions.jsonl"
    stable_dataset = session_dir / "cache/dataset/dataset.jsonl"
    stable_predictions.write_text('{"source":"stale"}\n', encoding="utf-8")
    stable_dataset.write_text('{"source":"stale"}\n', encoding="utf-8")

    bound_predictions = (
        cache_publication_path(session_dir, cache_run_id, "predictions")
        / "predictions.jsonl"
    )
    bound_dataset = (
        cache_publication_path(session_dir, cache_run_id, "dataset") / "dataset.jsonl"
    )
    bound_predictions.parent.mkdir(parents=True)
    bound_dataset.parent.mkdir(parents=True)
    bound_predictions.write_text('{"source":"bound-predictions"}\n', encoding="utf-8")
    bound_dataset.write_text('{"source":"bound-dataset"}\n', encoding="utf-8")
    for namespace in ("dataset", "predictions"):
        await manager.sync_to_store(
            sid,
            prefix=cache_publication_prefix(cache_run_id, namespace),
        )
    shutil.rmtree(stable_predictions.parent)
    shutil.rmtree(stable_dataset.parent)
    shutil.rmtree(session_dir / "artifacts/run_cache")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "cache_publications": cache_publications,
        },
    )

    async def generate_snapshot(
        snapshot_dir: Path,
        snapshot_predictions: Path,
        snapshot_dataset: Path,
    ) -> None:
        assert snapshot_predictions.read_text(encoding="utf-8") == (
            '{"source":"bound-predictions"}\n'
        )
        assert snapshot_dataset.read_text(encoding="utf-8") == (
            '{"source":"bound-dataset"}\n'
        )
        (snapshot_dir / "report.html").write_text(
            "<html>bound report</html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(artifacts_router, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        generate_snapshot,
    )

    response = await artifacts_router.view_prediction_report(sid)
    assert isinstance(response, StreamingResponse)
    completed = await manager.get_session_metadata(sid)
    assert completed is not None
    publication_id = completed[PREDICTION_REPORT_PUBLICATION_ID_FIELD]
    assert completed[PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD] == cache_publications
    publication_path = session_dir / prediction_report_publication_key(publication_id)
    assert publication_path.read_text(encoding="utf-8") == ("<html>bound report</html>")
    assert await manager.get_artifact_path(sid, "prediction_report") == publication_path
    assert "active_run_id" not in completed
    assert "active_run_expires_at" not in completed
    assert response.background is not None
    await response.background()

    await manager.update_session(
        sid,
        {PREDICTION_REPORT_PUBLICATION_ID_FIELD: None},
    )
    assert await manager.get_artifact_path(sid, "prediction_report") is None
    assert await manager.get_artifact_stream(sid, "prediction_report") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_mode", "expected_detail"),
    [
        ("missing_metadata", "Session not found"),
        ("incomplete_binding", "Prediction report not available"),
        ("missing_predictions", "Predictions not available yet"),
        ("missing_dataset", "Dataset not available"),
    ],
)
async def test_bound_report_generation_rejects_incomplete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_detail: str,
) -> None:
    manager = SessionManager(tmp_path / failure_mode)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    cache_run_id = "a" * 32
    cache_publications = {
        "dataset": cache_run_id,
        "predictions": cache_run_id,
    }
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "cache_publications": cache_publications,
        },
    )
    original_get_metadata = manager.get_session_metadata

    if failure_mode == "missing_metadata":

        async def missing_metadata(_sid: str):
            return None

        monkeypatch.setattr(manager, "get_session_metadata", missing_metadata)
    elif failure_mode == "incomplete_binding":

        async def incomplete_metadata(_sid: str):
            return {
                "status": "completed",
                "cache_publications": {"predictions": cache_run_id},
            }

        monkeypatch.setattr(manager, "get_session_metadata", incomplete_metadata)
    elif failure_mode == "missing_dataset":
        predictions = (
            cache_publication_path(session_dir, cache_run_id, "predictions")
            / "predictions.jsonl"
        )
        predictions.parent.mkdir(parents=True)
        predictions.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(artifacts_router, "get_session_manager", lambda: manager)
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 404
    assert exc.value.detail == expected_detail

    monkeypatch.setattr(manager, "get_session_metadata", original_get_metadata)
    metadata = await manager.get_session_metadata(sid)
    assert metadata is not None
    assert "active_run_id" not in metadata
    assert "active_run_expires_at" not in metadata


@pytest.mark.asyncio
async def test_cache_stream_rejects_binding_change_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    run_a = "a" * 32
    run_b = "b" * 32
    key_a = f"artifacts/run_cache/{run_a}/cache/predictions/predictions.jsonl"
    key_b = f"artifacts/run_cache/{run_b}/cache/predictions/predictions.jsonl"
    await store.put_bytes(sid, key_a, b"run-a")
    await store.put_bytes(sid, key_b, b"run-b")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "cache_publications": {"predictions": run_a},
        },
    )
    original_open_read = store.open_read
    opened_streams: list[io.BytesIO] = []

    async def replace_during_open(session_id: str, artifact_key: str) -> io.BytesIO:
        if artifact_key != key_a:
            return await original_open_read(session_id, artifact_key)
        stream = await original_open_read(session_id, artifact_key)
        opened_streams.append(stream)
        await manager.update_session(
            sid,
            {
                "status": "completed",
                "cache_publications": {"predictions": run_b},
            },
        )
        return stream

    monkeypatch.setattr(store, "open_read", replace_during_open)

    assert await manager.get_artifact_stream(sid, "predictions") is None
    assert len(opened_streams) == 1
    assert opened_streams[0].closed is True


@pytest.mark.asyncio
async def test_legacy_cache_lookup_rejects_claim_installed_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    key = "cache/predictions/predictions.jsonl"
    local_path = session_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"legacy")
    await store.put_bytes(sid, key, b"legacy")
    metadata = await manager.get_session_metadata(sid)
    assert metadata is not None
    metadata["status"] = "completed"
    metadata.pop("cache_publications")
    await store.put_json(sid, METADATA_KEY, metadata)

    path_run_id = "c" * 32
    original_lookup = manager._artifact_lookup_for_session
    first_lookup = True

    async def claim_after_selection(
        session_id: str,
        artifact_type: str,
    ) -> tuple[tuple[str, ...], tuple[str, str, str] | None]:
        nonlocal first_lookup
        selection = await original_lookup(session_id, artifact_type)
        if first_lookup:
            first_lookup = False
            assert await manager.reserve_run(sid, path_run_id)
        return selection

    monkeypatch.setattr(manager, "_artifact_lookup_for_session", claim_after_selection)
    assert await manager.get_artifact_path(sid, "predictions") is None
    assert await manager.release_run(sid, path_run_id)
    monkeypatch.setattr(manager, "_artifact_lookup_for_session", original_lookup)

    stream_run_id = "d" * 32
    original_open_read = store.open_read
    opened_streams: list[io.BytesIO] = []
    stream_claim_installed = False

    async def claim_during_open(session_id: str, artifact_key: str) -> io.BytesIO:
        nonlocal stream_claim_installed
        stream = await original_open_read(session_id, artifact_key)
        if artifact_key != key or stream_claim_installed:
            return stream
        stream_claim_installed = True
        opened_streams.append(stream)
        assert await manager.reserve_run(sid, stream_run_id)
        return stream

    monkeypatch.setattr(store, "open_read", claim_during_open)
    assert await manager.get_artifact_stream(sid, "predictions") is None
    assert len(opened_streams) == 1
    assert opened_streams[0].closed is True


@pytest.mark.asyncio
async def test_remote_joint_output_uses_bound_store_bytes(tmp_path: Path) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    session_dir = await manager.create_session(sid)
    publication_id = "a" * 32
    key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    stale_local = session_dir / key
    stale_local.parent.mkdir(parents=True)
    stale_local.write_bytes(b"stale-local")
    await store.put_bytes(sid, key, b"current-store")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifacts": {"joint_rigger_output": True},
                "joint_rigger_artifact_keys": {
                    "joint_rigger_output": key,
                },
                "joint_rigger_publication_id": publication_id,
            },
        },
    )

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "joint_rigger_output",
        "model/vnd.usdz+zip",
        "rigged.usdz",
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert b"".join(chunks) == b"current-store"
    assert response.headers["content-disposition"] == (
        'attachment; filename="rigged.usdz"'
    )
    assert response.background is not None
    await response.background()


@pytest.mark.asyncio
async def test_remote_joint_output_streams_newline_free_payload_in_fixed_chunks(
    tmp_path: Path,
) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    publication_id = "a" * 32
    key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    payload = b"x" * (artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE * 2 + 17)
    await store.put_bytes(sid, key, payload)
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifacts": {"joint_rigger_output": True},
                "joint_rigger_artifact_keys": {"joint_rigger_output": key},
                "joint_rigger_publication_id": publication_id,
            },
        },
    )

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "joint_rigger_output",
        "model/vnd.usdz+zip",
        "rigged.usdz",
    )
    chunks = [chunk async for chunk in response.body_iterator]

    assert [len(chunk) for chunk in chunks] == [
        artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE,
        artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE,
        17,
    ]
    assert b"".join(chunks) == payload
    assert response.background is not None
    await response.background()


@pytest.mark.asyncio
async def test_joint_stream_rejects_publication_change_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    publication_a = "a" * 32
    publication_b = "b" * 32
    key_a = f"artifacts/joint_rigger/{publication_a}/rigged.usdz"
    key_b = f"artifacts/joint_rigger/{publication_b}/rigged.usdz"
    await store.put_bytes(sid, key_a, b"run-a")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifact_keys": {"joint_rigger_output": key_a},
                "joint_rigger_publication_id": publication_a,
            },
        },
    )
    original_open_read = store.open_read
    opened_streams: list[io.BytesIO] = []

    async def replace_during_open(session_id: str, artifact_key: str) -> io.BytesIO:
        if artifact_key != key_a:
            return await original_open_read(session_id, artifact_key)
        await store.put_bytes(session_id, key_b, b"run-b")
        stream = await original_open_read(session_id, artifact_key)
        opened_streams.append(stream)
        await manager.update_session(
            sid,
            {
                "status": "completed",
                "results": {
                    "joint_rigger_artifact_keys": {
                        "joint_rigger_output": key_b,
                    },
                    "joint_rigger_publication_id": publication_b,
                },
            },
        )
        return stream

    monkeypatch.setattr(store, "open_read", replace_during_open)

    assert (
        await manager.get_artifact_stream_with_filename(sid, "joint_rigger_output")
        is None
    )
    assert len(opened_streams) == 1
    assert opened_streams[0].closed is True


@pytest.mark.asyncio
async def test_joint_stream_ignores_next_upload_before_metadata_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _StreamStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await manager.create_session(sid)
    publication_a = "a" * 32
    publication_b = "b" * 32
    key_a = f"artifacts/joint_rigger/{publication_a}/rigged.usdz"
    key_b = f"artifacts/joint_rigger/{publication_b}/rigged.usdz"
    await store.put_bytes(sid, key_a, b"run-a")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifact_keys": {"joint_rigger_output": key_a},
                "joint_rigger_publication_id": publication_a,
            },
        },
    )
    original_open_read = store.open_read

    async def upload_next_before_open(
        session_id: str,
        artifact_key: str,
    ) -> io.BytesIO:
        await store.put_bytes(session_id, key_b, b"run-b")
        return await original_open_read(session_id, artifact_key)

    monkeypatch.setattr(store, "open_read", upload_next_before_open)

    selected = await manager.get_artifact_stream_with_filename(
        sid,
        "joint_rigger_output",
    )
    assert selected is not None
    stream, filename = selected
    assert stream.read() == b"run-a"
    assert filename == "rigged.usdz"
    stream.close()


@pytest.mark.asyncio
async def test_local_joint_response_uses_immutable_file_across_rerun(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "local")
    sid = _sid()
    session_dir = await manager.create_session(sid)
    publication_a = "a" * 32
    publication_b = "b" * 32
    key_a = f"artifacts/joint_rigger/{publication_a}/rigged.usdz"
    key_b = f"artifacts/joint_rigger/{publication_b}/rigged.usdz"
    output = session_dir / key_a
    output.parent.mkdir(parents=True)
    output.write_bytes(b"run-a")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifacts": {"joint_rigger_output": True},
                "joint_rigger_artifact_keys": {"joint_rigger_output": key_a},
                "joint_rigger_publication_id": publication_a,
            },
        },
    )

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "joint_rigger_output",
        "model/vnd.usdz+zip",
        "rigged.usdz",
    )
    await manager.update_session(sid, {"status": "pending", "results": {}})
    next_output = session_dir / key_b
    next_output.parent.mkdir(parents=True)
    next_output.write_bytes(b"run-b")
    assert isinstance(response, FileResponse)
    detached_output = output.with_name("rigged.safe.usdz")
    output.rename(detached_output)
    secret = session_dir / "cache" / ".pipeline_temp" / "rigged.usdz"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"joint-secret-sentinel")
    output.symlink_to(secret)

    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await response(
        {"type": "http", "method": "GET", "headers": [], "extensions": {}},
        receive,
        send,
    )
    body_chunks = [
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    ]
    response_body = b"".join(body_chunks)
    assert response_body == b"run-a"
    assert b"joint-secret-sentinel" not in response_body
    assert all(len(chunk) <= response.chunk_size for chunk in body_chunks)


@pytest.mark.asyncio
async def test_legacy_local_joint_output_uses_fixed_descriptor_chunks(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "local")
    sid = _sid()
    session_dir = await manager.create_session(sid)
    key = "cache/joint_rigger/rigged.usd"
    output = session_dir / key
    output.parent.mkdir(parents=True)
    payload = b"x" * (artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE * 2 + 17)
    output.write_bytes(payload)
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifacts": {"joint_rigger_output": True},
                "joint_rigger_artifact_keys": {"joint_rigger_output": key},
            },
        },
    )

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "joint_rigger_output",
        "application/octet-stream",
        "rigged.usd",
    )
    assert isinstance(response, StreamingResponse)
    chunks = [chunk async for chunk in response.body_iterator]
    assert [len(chunk) for chunk in chunks] == [
        artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE,
        artifacts_router.ARTIFACT_STREAM_CHUNK_SIZE,
        17,
    ]
    assert b"".join(chunks) == payload
    assert response.background is not None
    await response.background()


@pytest.mark.asyncio
async def test_joint_stream_handles_file_removed_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "local")
    sid = _sid()
    session_dir = await manager.create_session(sid)
    publication_id = "a" * 32
    key = f"artifacts/joint_rigger/{publication_id}/rigged.usdz"
    output = session_dir / key
    output.parent.mkdir(parents=True)
    output.write_bytes(b"run-a")
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifact_keys": {"joint_rigger_output": key},
                "joint_rigger_publication_id": publication_id,
            },
        },
    )

    async def removed_before_open(_session_id: str, _key: str):
        raise FileNotFoundError

    monkeypatch.setattr(manager.store, "open_read", removed_before_open)

    assert (
        await manager.get_artifact_stream_with_filename(sid, "joint_rigger_output")
        is None
    )


@pytest.mark.asyncio
async def test_session_manager_delete_failures_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = SessionManager(
        tmp_path / "failing", store=_FailingDeleteStore(str(tmp_path / "s"))
    )
    sid = _sid()
    await failing.create_session(sid)
    assert await failing.delete_session(sid) is False

    manager = SessionManager(
        tmp_path / "retry-local",
        store=LocalSessionStore(str(tmp_path / "retry-store")),
    )
    sid = _sid()
    await manager.create_session(sid)
    calls = {"count": 0}
    real_remove = manager_mod.remove_confined_tree

    def flaky_remove(working_dir: Path, allowed_root: Path) -> bool:
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("busy")
        return real_remove(working_dir, allowed_root)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(manager_mod, "remove_confined_tree", flaky_remove)
    monkeypatch.setattr(manager_mod.asyncio, "sleep", no_sleep)
    assert await manager.delete_session(sid) is True
    assert calls["count"] == 2

    sid = _sid()
    await manager.create_session(sid)
    assert await manager.delete_session(sid) is True

    split_store = LocalSessionStore(str(tmp_path / "delete-store"))
    split = SessionManager(tmp_path / "delete-local", store=split_store)
    sid = _sid()
    await split.create_session(sid)
    assert await split.delete_session(sid) is True


@pytest.mark.asyncio
async def test_session_manager_cleanup_expired_sessions(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    expired = _sid()
    fresh = _sid()
    await manager.create_session(expired)
    await manager.create_session(fresh)
    expired_meta = await manager.get_session_metadata(expired)
    expired_meta["ttl_expires_at"] = (
        datetime.now(UTC) - timedelta(hours=1)
    ).isoformat()
    await manager.store.put_json(expired, METADATA_KEY, expired_meta)
    fresh_meta = await manager.get_session_metadata(fresh)
    fresh_meta["ttl_expires_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    await manager.store.put_json(fresh, METADATA_KEY, fresh_meta)
    await manager.store.init_session(_sid())

    assert await manager.cleanup_expired_sessions() == 1
    assert not await manager.session_exists(expired)
    assert await manager.session_exists(fresh)

    manager = SessionManager(tmp_path / "cleanup-extra")
    missing = _sid()
    naive = _sid()
    await manager.create_session(naive)
    naive_meta = await manager.get_session_metadata(naive)
    naive_meta["ttl_expires_at"] = (datetime.now() - timedelta(seconds=1)).isoformat()
    await manager.store.put_json(naive, METADATA_KEY, naive_meta)

    async def list_extra_sessions() -> list[str]:
        return [missing, naive]

    manager.list_sessions = list_extra_sessions  # type: ignore[method-assign]
    assert await manager.cleanup_expired_sessions() == 1


@pytest.mark.asyncio
async def test_session_manager_sync_from_store_creates_local_dir(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    manager = SessionManager(tmp_path / "local", store=store)
    sid = _sid()
    await store.put_bytes(sid, "cache/dataset/dataset.jsonl", b"{}\n")

    assert await manager.sync_from_store(sid, prefix="cache/dataset/") == 1
    assert (
        tmp_path / "local" / sid / "cache" / "dataset" / "dataset.jsonl"
    ).read_bytes() == b"{}\n"
    await store.put_bytes(sid, "cache/dataset/dataset.jsonl", b"shared-new\n")
    assert await manager.sync_from_store(sid, prefix="cache/dataset/") == 0
    assert (
        tmp_path / "local" / sid / "cache" / "dataset" / "dataset.jsonl"
    ).read_bytes() == b"{}\n"
    assert (
        await manager.sync_from_store(
            sid,
            prefix="cache/dataset/",
            overwrite=True,
        )
        == 1
    )
    assert (
        tmp_path / "local" / sid / "cache" / "dataset" / "dataset.jsonl"
    ).read_bytes() == b"shared-new\n"
