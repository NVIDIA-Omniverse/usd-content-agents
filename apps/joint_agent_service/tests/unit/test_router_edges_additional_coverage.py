# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import HTTPException

from ...service.models.requests import PipelineStep, RegenerateRequest
from ...service.routers import artifacts_router, pipeline_router
from ...service.session.cache_publications import (
    PIPELINE_CONFIG_PUBLICATION_ID_FIELD,
    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD,
    pipeline_config_publication_path,
)
from ...service.session.manager import SessionManager

_MIB = 1024 * 1024


def _bind_pipeline_config_publication(
    manager: _Manager,
    session_id: str,
    config_path: Path,
    *,
    run_id: str,
) -> None:
    publication_path = pipeline_config_publication_path(
        manager.get_session_dir(session_id),
        run_id,
    )
    publication_path.parent.mkdir(parents=True, exist_ok=True)
    publication_bytes = config_path.read_bytes()
    publication_path.write_bytes(publication_bytes)
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                publication_bytes
            ).hexdigest(),
        }
    )


class _Upload:
    def __init__(self, filename: str | None, chunks: list[bytes]) -> None:
        self.filename = filename
        self._chunks = list(chunks)

    async def read(self, _chunk_size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events = [{"type": "ok"}]
        self.fail_events = False

    async def get_event_log(self, _session_id: str) -> list[dict[str, Any]]:
        if self.fail_events:
            raise RuntimeError("sentinel-event-store-secret")
        return self.events

    async def exists(self, _session_id: str, _key: str) -> bool:
        return True

    async def open_read(self, session_id: str, key: str):
        return (self.root / session_id / key).open("rb")

    async def list_keys(self, session_id: str, prefix: str = "") -> list[str]:
        session_dir = self.root / session_id
        if not session_dir.exists():
            return []
        return sorted(
            path.relative_to(session_dir).as_posix()
            for path in session_dir.rglob("*")
            if path.is_file()
            and path.relative_to(session_dir).as_posix().startswith(prefix)
        )

    async def delete_key(self, session_id: str, key: str) -> None:
        (self.root / session_id / key).unlink(missing_ok=True)


class _Manager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.storage_path = root
        self.sessions: set[str] = set()
        self.metadata: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.synced: list[str] = []
        self.pulled: list[str] = []
        self.cancelled: list[str] = []
        self.active_runs: dict[str, str] = {}
        self.store = _Store(root)
        self.sync_to_store_error: Exception | None = None
        self.sync_from_store_creates_input = False
        self.sync_from_store_count = 0
        self.stream_artifacts: set[str] = {"streamed"}
        self.run_claim_heartbeat_seconds = 60.0

    async def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        self.sessions.add(session_id)
        session_dir = self.get_session_dir(session_id)
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        self.metadata[session_id] = {
            "session_id": session_id,
            "status": "ready",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "overall_progress": {"percent": 0, "total_steps": 7, "current_step": 0},
            "completed_steps": [],
            "preview_images": [],
            "config": config or {},
        }
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        return self.root / session_id

    async def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    async def delete_session(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        self.sessions.discard(session_id)
        return True

    async def sync_to_store(
        self,
        session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        self.synced.append(prefix)
        if self.sync_to_store_error is not None:
            raise self.sync_to_store_error
        return 1

    async def sync_from_store(
        self,
        session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        del overwrite
        self.pulled.append(prefix)
        if self.sync_from_store_count:
            return self.sync_from_store_count
        if self.sync_from_store_creates_input:
            (self.get_session_dir(session_id) / "input").mkdir(
                parents=True, exist_ok=True
            )
            (self.get_session_dir(session_id) / "input" / "scene.usda").write_text(
                "#usda 1.0\n",
                encoding="utf-8",
            )
            return 1
        return 0

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        return self.metadata.get(session_id)

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        self.metadata.setdefault(
            session_id,
            {
                "session_id": session_id,
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "overall_progress": {},
                "completed_steps": [],
            },
        ).update(updates)

    async def reserve_run(self, session_id: str, run_id: str) -> bool:
        if session_id in self.active_runs:
            return False
        self.active_runs[session_id] = run_id
        await self.update_session(session_id, {"active_run_id": run_id})
        return True

    async def reserve_legacy_cache_run(self, session_id: str, run_id: str) -> bool:
        return await self.reserve_run(session_id, run_id)

    async def update_session_for_run(
        self, session_id: str, run_id: str, updates: dict[str, Any]
    ) -> bool:
        if self.active_runs.get(session_id) != run_id:
            return False
        await self.update_session(session_id, updates)
        return True

    async def renew_run(self, session_id: str, run_id: str) -> bool:
        return self.active_runs.get(session_id) == run_id

    async def release_run(self, session_id: str, run_id: str) -> bool:
        if self.active_runs.get(session_id) != run_id:
            return False
        del self.active_runs[session_id]
        self.metadata.get(session_id, {}).pop("active_run_id", None)
        return True

    async def terminalize_and_release_run(
        self,
        session_id: str,
        run_id: str,
        updates: dict[str, Any],
    ) -> bool:
        if not await self.update_session_for_run(session_id, run_id, updates):
            return False
        return await self.release_run(session_id, run_id)

    async def request_cancellation(self, session_id: str, run_id: str) -> bool:
        if self.active_runs.get(session_id) != run_id:
            return False
        self.cancelled.append(session_id)
        await self.update_session(session_id, {"status": "cancelling"})
        return True

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        path = self.get_session_dir(session_id) / f"{artifact_type}.txt"
        return path if path.exists() else None

    async def get_immutable_local_artifact_path_with_filename(
        self, _session_id: str, _artifact_type: str
    ) -> tuple[Path, str] | None:
        return None

    async def get_immutable_local_artifact_stream_with_filename(
        self, _session_id: str, _artifact_type: str
    ) -> None:
        return None

    async def get_artifact_stream(
        self, session_id: str, artifact_type: str
    ) -> io.BytesIO | None:
        if artifact_type in self.stream_artifacts:
            return io.BytesIO(b"stream")
        if artifact_type == "prediction_report":
            path = (
                self.get_session_dir(session_id)
                / "cache"
                / "predictions"
                / "report.html"
            )
        else:
            path = self.get_session_dir(session_id) / f"{artifact_type}.txt"
        if path.is_file():
            return io.BytesIO(path.read_bytes())
        return None

    async def get_artifact_stream_with_filename(
        self, _session_id: str, artifact_type: str
    ) -> tuple[io.BytesIO, str] | None:
        stream = await self.get_artifact_stream(_session_id, artifact_type)
        if stream is None:
            return None
        filename = (
            "rigged.usdz"
            if artifact_type == "joint_rigger_output"
            else f"{artifact_type}.txt"
        )
        return stream, filename

    async def has_artifact(self, session_id: str, artifact_type: str) -> bool:
        if await self.get_artifact_path(session_id, artifact_type):
            return True
        return artifact_type in self.stream_artifacts


class _Registry:
    def __init__(self) -> None:
        self.running = False
        self.registered: list[tuple[str, Any]] = []
        self.cancelled: list[str] = []
        self.cancelled_run_ids: list[str | None] = []
        self.admissions: dict[str, str] = {}

    async def reserve_admission(self, session_id: str, run_id: str) -> bool:
        if self.running or session_id in self.admissions:
            return False
        self.admissions[session_id] = run_id
        return True

    async def release_admission(self, session_id: str, run_id: str) -> bool:
        if self.admissions.get(session_id) != run_id:
            return False
        del self.admissions[session_id]
        return True

    async def register(
        self,
        session_id: str,
        coro: Any,
        *,
        run_id: str | None = None,
        liveness_guard=None,
        on_finish=None,
    ) -> None:
        if run_id is not None:
            self.admissions.pop(session_id, None)
        self.registered.append((session_id, coro))
        if hasattr(coro, "close"):
            coro.close()
        if hasattr(liveness_guard, "stop"):
            await liveness_guard.stop()
        elif hasattr(liveness_guard, "close"):
            liveness_guard.close()
        if on_finish is not None:
            await on_finish()

    def is_running(self, session_id: str) -> bool:
        return self.running or session_id in self.admissions

    async def cancel(self, session_id: str, *, run_id: str | None = None) -> bool:
        self.cancelled.append(session_id)
        self.cancelled_run_ids.append(run_id)
        return True


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Manager:
    mgr = _Manager(tmp_path)
    pipeline_router.set_session_manager(mgr)  # type: ignore[arg-type]
    artifacts_router.set_session_manager(mgr)  # type: ignore[arg-type]
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)
    monkeypatch.setattr(pipeline_router.config, "vlm_backend", "mock")
    monkeypatch.setattr(pipeline_router.config, "vlm_model", "mock")
    monkeypatch.setattr(pipeline_router.config, "llm_backend", "mock")
    monkeypatch.setattr(pipeline_router.config, "llm_model", "mock")
    return mgr


def test_router_manager_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_router, "session_manager", None)
    with pytest.raises(RuntimeError):
        pipeline_router.get_session_manager()
    monkeypatch.setattr(artifacts_router, "session_manager", None)
    with pytest.raises(RuntimeError):
        artifacts_router.get_session_manager()


def test_render_limit_and_s3_download_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.get_global_remote_render_limit",
        lambda: None,
    )
    pipeline_router._apply_render_request_limit({"steps": {"build_dataset_usd": {}}})

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.get_global_remote_render_limit",
        lambda: 3,
    )
    config = {
        "steps": {
            "build_dataset_usd": {
                "num_workers": "bad",
                "max_concurrent_requests": object(),
            }
        }
    }
    pipeline_router._apply_render_request_limit(config)
    assert config["steps"]["build_dataset_usd"] == {
        "num_workers": 3,
        "max_concurrent_requests": 3,
    }
    pipeline_router._apply_render_request_limit({"steps": {"predict": {}}})

    for uri, status in [
        ("http://bucket/scene.usda", 400),
        ("s3://bucket/path/model.obj", 400),
    ]:
        with pytest.raises(HTTPException) as exc:
            pipeline_router._download_s3_to_session(uri, tmp_path)
        assert exc.value.status_code == status

    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    def raise_not_found(_uri: str, _path: Path) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(pipeline_router, "download_file_from_s3", raise_not_found)
    with pytest.raises(HTTPException) as exc:
        pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert exc.value.status_code == 404

    def raise_denied(_uri: str, _path: Path) -> None:
        raise PermissionError

    monkeypatch.setattr(pipeline_router, "download_file_from_s3", raise_denied)
    with pytest.raises(HTTPException) as exc:
        pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert exc.value.status_code == 403

    def raise_other(_uri: str, _path: Path) -> None:
        raise RuntimeError("net")

    monkeypatch.setattr(pipeline_router, "download_file_from_s3", raise_other)
    with pytest.raises(HTTPException) as exc:
        pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert exc.value.status_code == 502

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    monkeypatch.setattr(
        pipeline_router,
        "download_file_from_s3",
        lambda _uri, path: path.write_bytes(b"x"),
    )
    local_path = pipeline_router._download_s3_to_session(
        "s3://bucket/scene.usda",
        tmp_path,
    )
    assert local_path.read_bytes() == b"x"
    assert not list((tmp_path / "input").glob(".*.download"))

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)
    monkeypatch.setattr(
        pipeline_router,
        "download_file_from_s3",
        lambda _uri, path: path.write_bytes(b"x" * (_MIB + 1)),
    )
    with pytest.raises(HTTPException) as exc:
        pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert exc.value.status_code == 413
    assert local_path.read_bytes() == b"x"
    assert not list((tmp_path / "input").glob(".*.download"))

    monkeypatch.setattr(
        pipeline_router,
        "download_file_from_s3",
        lambda _uri, path: path.write_bytes(b"x"),
    )
    local_path = pipeline_router._download_s3_to_session(
        "s3://bucket/scene.usda",
        tmp_path,
    )
    assert local_path.name == "scene.usda"


def test_prediction_limit_caps_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_router.config, "vlm_max_workers", 3)

    for existing_workers, expected_workers in (
        (64, 3),
        (2, 2),
        ("bad", 3),
        (None, 3),
    ):
        pipeline_config = {"steps": {"predict": {"max_workers": existing_workers}}}
        pipeline_router._apply_prediction_request_limit(pipeline_config)
        assert pipeline_config["steps"]["predict"]["max_workers"] == expected_workers

    missing_workers = {"steps": {"predict": {}}}
    pipeline_router._apply_prediction_request_limit(missing_workers)
    assert missing_workers["steps"]["predict"]["max_workers"] == 3

    unrelated_config = {"steps": {"build_dataset_usd": {}}}
    pipeline_router._apply_prediction_request_limit(unrelated_config)
    assert unrelated_config == {"steps": {"build_dataset_usd": {}}}


@pytest.mark.asyncio
async def test_upload_usd_with_native_local_store_confinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    monkeypatch.setattr(pipeline_router, "session_manager", manager)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)

    uploaded = await pipeline_router.upload_usd_immediate(
        _Upload("scene.usda", [b"#usda 1.0\n"]),
        None,
    )

    assert uploaded.status == "ready"
    assert await manager.sync_from_store(uploaded.session_id, prefix="input/") == 0
    assert (
        manager.get_session_dir(uploaded.session_id)
        .joinpath("input", "scene.usda")
        .read_bytes()
        == b"#usda 1.0\n"
    )


@pytest.mark.asyncio
async def test_upload_usd_immediate_edges(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(None, None)
    assert exc.value.status_code == 400

    upload = _Upload("scene.usda", [b"#usda\n"])
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(upload, "s3://bucket/scene.usda")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(_Upload("scene.obj", [b"x"]), None)
    assert exc.value.status_code == 400

    manager.sync_to_store_error = RuntimeError("sync")
    result = await pipeline_router.upload_usd_immediate(
        _Upload("scene.usda", [b"#usda\n"]),
        None,
    )
    assert result.status == "ready"
    manager.sync_to_store_error = None

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    result = await pipeline_router.upload_usd_immediate(
        _Upload("scene.usda", [b"#usda\n"]),
        None,
    )
    assert result.status == "ready"

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(
            _Upload("scene.usda", [b"x" * _MIB, b"x"]),
            None,
        )
    assert exc.value.status_code == 413
    rejected_session_dir = manager.get_session_dir(manager.deleted[-1])
    assert not list((rejected_session_dir / "input").glob("scene.*"))
    assert not list((rejected_session_dir / "input").glob(".*.upload"))

    def fake_s3_download(_uri: str, session_dir: Path) -> Path:
        path = session_dir / "input" / "scene.usda"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#usda\n")
        return path

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fake_s3_download)
    result = await pipeline_router.upload_usd_immediate(None, "s3://bucket/scene.usda")
    assert result.status == "ready"

    def fail_s3(_uri: str, _session_dir: Path) -> Path:
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_s3)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(None, "s3://bucket/missing.usda")
    assert exc.value.status_code == 404

    class BadUpload(_Upload):
        async def read(self, _chunk_size: int) -> bytes:
            raise RuntimeError("read failed")

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.upload_usd_immediate(BadUpload("scene.usda", []), None)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_create_pipeline_and_result_edges(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    prediction_limit_calls: list[dict[str, Any]] = []
    apply_prediction_limit = pipeline_router._apply_prediction_request_limit

    def record_prediction_limit(pipeline_config: dict[str, Any]) -> None:
        prediction_limit_calls.append(pipeline_config)
        apply_prediction_limit(pipeline_config)

    monkeypatch.setattr(
        pipeline_router,
        "_apply_prediction_request_limit",
        record_prediction_limit,
    )
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **kwargs: {
            "project": {
                "name": "demo",
                "session_id": kwargs["session_id"],
                "working_dir": kwargs["working_dir"],
            },
            "input": {"usd_path": kwargs["usd_path"]},
            "steps": {"build_dataset_usd": {}},
        },
    )

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(None, None, None, "", "")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(None, "missing", None, "", "")
    assert exc.value.status_code == 404

    session_id = "existing"
    await manager.create_session(session_id)
    manager.sync_from_store_creates_input = True
    created = await pipeline_router.create_pipeline(
        None,
        session_id,
        None,
        "  prompt  ",
        " remote ",
    )
    assert created.status == "pending"
    assert registry.registered
    assert len(prediction_limit_calls) == 1

    manager.sync_to_store_error = RuntimeError("sync")
    registered_count = len(registry.registered)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(
            None,
            session_id,
            None,
            "prompt",
            "remote",
        )
    assert exc.value.status_code == 503
    assert len(registry.registered) == registered_count
    assert session_id not in manager.active_runs
    manager.sync_to_store_error = None

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(
            _Upload("scene.obj", [b"x"]), None, None, "", ""
        )
    assert exc.value.status_code == 400

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    created_unbounded = await pipeline_router.create_pipeline(
        _Upload("scene.usda", [b"#usda\n"]),
        None,
        None,
        "",
        "",
    )
    assert created_unbounded.status == "pending"

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(
            _Upload("scene.usda", [b"x" * _MIB, b"x"]),
            None,
            None,
            "",
            "",
        )
    assert exc.value.status_code == 413
    rejected_session_dir = manager.get_session_dir(manager.deleted[-1])
    assert not list((rejected_session_dir / "input").glob("scene.*"))
    assert not list((rejected_session_dir / "input").glob(".*.upload"))

    def fail_if_downloaded(_uri: str, _session_dir: Path) -> Path:
        raise AssertionError("S3 download should not run after option validation")

    sessions_before_invalid_options = set(manager.sessions)
    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_if_downloaded)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(
            usd_file=None,
            session_id=None,
            s3_uri="s3://bucket/scene.usda",
            user_prompt="",
            render_backend="",
            apply_joint_rigger=False,
            joint_rigger_adapter="usd_joint_rigger",
        )
    assert exc.value.status_code == 400
    assert manager.sessions == sessions_before_invalid_options

    def fake_s3_download(_uri: str, session_dir: Path) -> Path:
        path = session_dir / "input" / "scene.usda"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"#usda\n")
        return path

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fake_s3_download)
    created_from_s3 = await pipeline_router.create_pipeline(
        None,
        None,
        "s3://bucket/scene.usda",
        "",
        "",
    )
    assert created_from_s3.status == "pending"

    def fail_s3(_uri: str, _session_dir: Path) -> Path:
        raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_s3)
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(
            None,
            None,
            "s3://bucket/missing.usda",
            "",
            "",
        )
    assert exc.value.status_code == 404

    no_input_sid = "no-input"
    await manager.create_session(no_input_sid)
    manager.sync_from_store_creates_input = False
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.create_pipeline(None, no_input_sid, None, "", "")
    assert exc.value.status_code == 400

    manager.metadata["done"] = {
        "status": "completed",
        "results": {"predictions_made": 1},
        "duration_seconds": 2,
        "completed_at": "now",
    }

    async def fail_has_artifact(_session_id: str, _artifact_type: str) -> bool:
        raise AssertionError("/results must not probe artifact storage")

    monkeypatch.setattr(manager, "has_artifact", fail_has_artifact)
    results = await pipeline_router.get_pipeline_results("done")
    assert results.status == "completed"
    assert "joint_rigger_output" not in results.download_urls

    manager.metadata["rigger-done"] = {
        "status": "completed",
        "results": {
            "joint_rigger_status": "authored",
            "joint_rigger_artifacts": {
                "joint_rigger_output": True,
                "joint_rigger_diagnostics": True,
                "joint_rigger_validation": False,
            },
        },
        "duration_seconds": 2,
        "completed_at": "now",
    }
    rigger_results = await pipeline_router.get_pipeline_results("rigger-done")
    assert "joint_rigger_output" in rigger_results.download_urls
    assert "joint_rigger_diagnostics" in rigger_results.download_urls
    assert "joint_rigger_validation" not in rigger_results.download_urls

    manager.metadata["legacy-rigger-done"] = {
        "status": "completed",
        "results": {"joint_rigger_status": "authored"},
        "duration_seconds": 2,
        "completed_at": "now",
    }
    legacy_rigger_results = await pipeline_router.get_pipeline_results(
        "legacy-rigger-done"
    )
    assert "joint_rigger_output" not in legacy_rigger_results.download_urls
    assert "joint_rigger_diagnostics" not in legacy_rigger_results.download_urls
    assert "joint_rigger_validation" not in legacy_rigger_results.download_urls

    manager.metadata["failed"] = {
        "status": "failed",
        "error": "bad",
        "failed_step": "predict",
        "completed_steps": [{"name": "build_dataset_usd"}],
    }
    failed = await pipeline_router.get_pipeline_results("failed")
    assert failed.status == "failed"
    manager.metadata["cancelled"] = {
        "status": "cancelled",
        "completed_steps": [{"name": "build_dataset_usd"}],
    }
    cancelled = await pipeline_router.get_pipeline_results("cancelled")
    assert cancelled.model_dump() == {
        "session_id": "cancelled",
        "status": "cancelled",
        "error_message": "Pipeline run was cancelled",
        "failed_step": "cancelled",
        "completed_steps": ["build_dataset_usd"],
        "partial_results": None,
    }
    manager.metadata["running"] = {"status": "running"}
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.get_pipeline_results("running")
    assert exc.value.status_code == 202

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.get_pipeline_results("missing")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_status_cancel_events_regenerate_and_event_log_edges(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    prediction_limit_calls: list[dict[str, Any]] = []
    apply_prediction_limit = pipeline_router._apply_prediction_request_limit

    def record_prediction_limit(pipeline_config: dict[str, Any]) -> None:
        prediction_limit_calls.append(pipeline_config)
        apply_prediction_limit(pipeline_config)

    monkeypatch.setattr(
        pipeline_router,
        "_apply_prediction_request_limit",
        record_prediction_limit,
    )

    sid = "sid"
    await manager.create_session(sid)
    manager.metadata[sid]["status"] = "running"
    manager.metadata[sid]["created_at"] = datetime.now().isoformat()

    status = await pipeline_router.get_pipeline_status(sid)
    assert status.session_id == sid

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.get_pipeline_status("missing")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.cancel_pipeline("missing", "f" * 32)
    assert exc.value.status_code == 404

    manager.metadata[sid]["status"] = "completed"
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.cancel_pipeline(sid, "f" * 32)
    assert exc.value.status_code == 400

    manager.metadata[sid]["status"] = "running"
    run_id = "a" * 32
    assert await manager.reserve_run(sid, run_id)
    registry.running = True
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.cancel_pipeline(sid, "b" * 32)
    assert exc.value.status_code == 409
    assert manager.cancelled == []

    cancelled = await pipeline_router.cancel_pipeline(sid, run_id)
    assert cancelled.status == "cancelling"
    assert cancelled.run_id == run_id
    assert registry.cancelled == [sid]
    assert registry.cancelled_run_ids == [run_id]
    assert await manager.release_run(sid, run_id)

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.stream_progress_events("missing")
    assert exc.value.status_code == 404

    manager.metadata[sid]["status"] = "running"
    registry.running = False
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.stream_progress_events(sid)
    assert exc.value.status_code == 503

    manager.metadata[sid]["status"] = "completed"
    response = await pipeline_router.stream_progress_events(sid)
    assert response is not None

    session_dir = manager.get_session_dir(sid)
    config_path = session_dir / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "demo",
                    "session_id": sid,
                    "working_dir": str(session_dir / "cache"),
                },
                "input": {"usd_path": str(session_dir / "input" / "scene.usda")},
                "steps": {"build_dataset_prepare_dataset": {}},
            }
        ),
        encoding="utf-8",
    )
    _bind_pipeline_config_publication(
        manager,
        sid,
        config_path,
        run_id="c" * 32,
    )

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.regenerate_pipeline(
            "missing", RegenerateRequest(steps=[])
        )
    assert exc.value.status_code == 404

    manager.metadata[sid]["status"] = "running"
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.regenerate_pipeline(
            sid,
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    assert exc.value.status_code == 409
    assert (
        exc.value.detail
        == "Regeneration prerequisite is unavailable: cache/dataset/dataset.jsonl"
    )

    manager.metadata[sid]["status"] = "completed"
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.regenerate_pipeline(
            sid,
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    assert exc.value.status_code == 409
    assert (
        exc.value.detail
        == "Regeneration prerequisite is unavailable: cache/dataset/dataset.jsonl"
    )

    config_path.write_text("[]\n", encoding="utf-8")
    _bind_pipeline_config_publication(
        manager,
        sid,
        config_path,
        run_id="d" * 32,
    )
    with pytest.raises(HTTPException) as exc:
        await pipeline_router.regenerate_pipeline(
            sid,
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    assert exc.value.status_code == 409
    assert (
        exc.value.detail == "Original config publication failed integrity verification"
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "demo",
                    "session_id": sid,
                    "working_dir": str(session_dir / "cache"),
                },
                "input": {"usd_path": str(session_dir / "input" / "scene.usda")},
                "steps": {"build_dataset_prepare_dataset": {}},
            }
        ),
        encoding="utf-8",
    )
    _bind_pipeline_config_publication(
        manager,
        sid,
        config_path,
        run_id="e" * 32,
    )
    dataset_dir = session_dir / "cache" / "dataset"
    render_path = dataset_dir / "renders" / "component.png"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    render_path.write_bytes(b"rendered-component")
    (dataset_dir / "dataset.jsonl").write_text(
        json.dumps(
            {
                "id": "/Root/Component",
                "media": {"images": [{"path": "renders/component.png"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    regenerated = await pipeline_router.regenerate_pipeline(
        sid,
        RegenerateRequest(steps=[PipelineStep.PREDICT], user_prompt="new prompt"),
    )
    assert regenerated.status == "pending"
    assert prediction_limit_calls

    assert await pipeline_router.get_event_log(sid) == {
        "events": [{"type": "ok"}],
        "total": 1,
    }
    manager.store.fail_events = True
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        assert await pipeline_router.get_event_log(sid) == {"events": []}
    assert "event_log_store_read_failed" in caplog.text
    assert "phase=persistence_verification" in caplog.text
    assert "sentinel-event-store-secret" not in caplog.text
    log_file = manager.get_session_dir(sid) / "event_log.jsonl"
    log_file.write_text(json.dumps({"type": "local"}) + "\n", encoding="utf-8")
    assert (await pipeline_router.get_event_log(sid))["total"] == 1
    local_sentinel = "sentinel-event-local-secret"
    log_file.write_text(f"{{{local_sentinel}\n", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as exc:
            await pipeline_router.get_event_log(sid)
    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to load event log"
    assert "event_log_local_read_failed" in caplog.text
    assert local_sentinel not in str(exc.value.detail)
    assert local_sentinel not in caplog.text

    with pytest.raises(HTTPException) as exc:
        await pipeline_router.get_event_log("missing")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_artifact_router_edges(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "artifacts"
    await manager.create_session(sid)
    local = manager.get_session_dir(sid) / "predictions.txt"
    local.write_text("{}", encoding="utf-8")
    (manager.get_session_dir(sid) / "dataset.txt").write_text("{}", encoding="utf-8")

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "predictions",
        "application/json",
        "predictions.jsonl",
    )
    assert response.media_type == "application/json"

    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "streamed",
        "text/plain",
        "streamed.txt",
    )
    assert response.media_type == "text/plain"

    with pytest.raises(HTTPException) as exc:
        await artifacts_router._serve_artifact(
            manager,
            "missing",
            "joint_rigger_output",
            "model/vnd.usdz+zip",
            "rigged.usdz",
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == "Session not found"

    manager.metadata[sid].update(
        {
            "status": "completed",
            "results": {
                "joint_rigger_artifacts": {"joint_rigger_output": True},
            },
        }
    )
    with pytest.raises(HTTPException) as exc:
        await artifacts_router._serve_artifact(
            manager,
            sid,
            "joint_rigger_output",
            "model/vnd.usdz+zip",
            "rigged.usdz",
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == "Joint_rigger_output not available"

    manager.stream_artifacts.add("joint_rigger_output")
    response = await artifacts_router._serve_artifact(
        manager,
        sid,
        "joint_rigger_output",
        "model/vnd.usdz+zip",
        "rigged.usdz",
    )
    assert response.media_type == "model/vnd.usdz+zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="rigged.usdz"'
    )

    with pytest.raises(HTTPException) as exc:
        await artifacts_router._serve_artifact(
            manager,
            sid,
            "missing",
            "text/plain",
            "missing.txt",
        )
    assert exc.value.status_code == 404

    for func in (
        artifacts_router.download_predictions,
        artifacts_router.download_articulation_candidates,
        artifacts_router.view_articulation_report,
        artifacts_router.view_prediction_report,
        artifacts_router.download_dataset,
        artifacts_router.download_joint_rigger_output,
        artifacts_router.download_joint_rigger_diagnostics,
        artifacts_router.download_joint_rigger_validation,
    ):
        with pytest.raises(HTTPException) as exc:
            await func("missing")
        assert exc.value.status_code == 404

    await artifacts_router.download_predictions(sid)
    await artifacts_router.download_dataset(sid)

    report = manager.get_session_dir(sid) / "cache" / "predictions" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("<html></html>", encoding="utf-8")
    assert (
        await artifacts_router.view_prediction_report(sid)
    ).media_type == "text/html"

    report.unlink()
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 404

    predictions = (
        manager.get_session_dir(sid) / "cache" / "predictions" / "predictions.jsonl"
    )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    manager.sync_from_store_count = 1
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 404
    manager.sync_from_store_count = 0

    predictions = (
        manager.get_session_dir(sid) / "cache" / "predictions" / "predictions.jsonl"
    )
    dataset = manager.get_session_dir(sid) / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    reporting = ModuleType("joint_agent.tasks.reporting")

    class Task:
        def run(self, context: dict[str, Any], _store: Any) -> None:
            Path(context["output_dir"], "report.html").write_text(
                "<html>generated</html>",
                encoding="utf-8",
            )

    reporting.GeneratePredictionReportTask = Task  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent.tasks.reporting", reporting)
    await artifacts_router.view_prediction_report(sid)

    report.unlink()

    sentinel = "joint-report-publication-sentinel-727"

    class BadTask:
        def run(self, _context: dict[str, Any], _store: Any) -> None:
            raise RuntimeError(sentinel)

    reporting.GeneratePredictionReportTask = BadTask  # type: ignore[attr-defined]
    with caplog.at_level(logging.ERROR, logger=artifacts_router.__name__):
        with pytest.raises(HTTPException) as exc:
            await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 500
    assert exc.value.detail == "Report generation failed"
    assert "joint_prediction_report_publication_failed" in caplog.text
    assert "phase=local_publication" in caplog.text
    assert sentinel not in caplog.text

    local_articulation = manager.get_session_dir(sid) / "articulation_report.txt"
    if local_articulation.exists():
        local_articulation.unlink()
    manager.stream_artifacts.add("articulation_report")
    response = await artifacts_router.view_articulation_report(sid)
    assert response.media_type == "text/html"
    manager.stream_artifacts.remove("articulation_report")
    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_articulation_report(sid)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_prediction_report_generation_discards_output_after_claim_loss(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "report-claim-loss"
    session_dir = await manager.create_session(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    async def lose_claim_after_generation(
        _session_dir: Path,
        snapshot_predictions: Path,
        _snapshot_dataset: Path,
    ) -> None:
        snapshot_predictions.with_name("report.html").write_text(
            "<html>stale</html>",
            encoding="utf-8",
        )
        manager.active_runs[sid] = "f" * 32

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        lose_claim_after_generation,
    )

    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 409
    assert not (session_dir / "cache" / "predictions" / "report.html").exists()
    assert not list(session_dir.glob(".prediction-report-*"))


@pytest.mark.asyncio
async def test_prediction_report_upload_after_claim_loss_remains_unbound(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "report-upload-claim-loss"
    session_dir = await manager.create_session(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    async def generate_report(
        _session_dir: Path,
        snapshot_predictions: Path,
        _snapshot_dataset: Path,
    ) -> None:
        snapshot_predictions.with_name("report.html").write_text(
            "<html>stale upload</html>",
            encoding="utf-8",
        )

    synced_prefixes: list[str] = []
    successor_run = "f" * 32

    async def lose_claim_during_sync(
        _session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        assert overwrite is False
        synced_prefixes.append(prefix)
        manager.active_runs[sid] = successor_run
        return 1

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        generate_report,
    )
    monkeypatch.setattr(manager, "sync_to_store", lose_claim_during_sync)

    with pytest.raises(HTTPException) as exc:
        await artifacts_router.view_prediction_report(sid)
    assert exc.value.status_code == 409
    assert synced_prefixes
    assert all(
        prefix.startswith("artifacts/prediction_reports/") for prefix in synced_prefixes
    )
    assert "prediction_report_publication_id" not in manager.metadata[sid]
    assert manager.active_runs[sid] == successor_run


@pytest.mark.asyncio
async def test_prediction_report_cancellation_drains_before_claim_release(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "report-cancel-drain"
    session_dir = await manager.create_session(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    generation_started = asyncio.Event()
    finish_generation = asyncio.Event()

    async def slow_generation(
        _session_dir: Path,
        snapshot_predictions: Path,
        _snapshot_dataset: Path,
    ) -> None:
        generation_started.set()
        await finish_generation.wait()
        snapshot_predictions.with_name("report.html").write_text(
            "<html>cancelled</html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        slow_generation,
    )

    request = asyncio.create_task(artifacts_router.view_prediction_report(sid))
    await generation_started.wait()
    assert sid in manager.active_runs
    request.cancel()
    await asyncio.sleep(0)
    assert not request.done()
    assert sid in manager.active_runs

    finish_generation.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert sid not in manager.active_runs
    assert not (session_dir / "cache" / "predictions" / "report.html").exists()
    assert not list(session_dir.glob(".prediction-report-*"))


@pytest.mark.asyncio
async def test_prediction_report_cancellation_during_cleanup_releases_claim(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "report-cancel-cleanup"
    session_dir = await manager.create_session(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    async def generate_report(
        _session_dir: Path,
        snapshot_predictions: Path,
        _snapshot_dataset: Path,
    ) -> None:
        snapshot_predictions.with_name("report.html").write_text(
            "<html>complete</html>",
            encoding="utf-8",
        )

    release_started = asyncio.Event()
    finish_release = asyncio.Event()
    original_release = manager.release_run

    async def slow_release(session_id: str, run_id: str) -> bool:
        release_started.set()
        await finish_release.wait()
        return await original_release(session_id, run_id)

    async def fail_publication_sync(
        _session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        del prefix, overwrite
        raise RuntimeError("publication sync failed")

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        generate_report,
    )
    monkeypatch.setattr(manager, "release_run", slow_release)
    monkeypatch.setattr(manager, "sync_to_store", fail_publication_sync)

    request = asyncio.create_task(artifacts_router.view_prediction_report(sid))
    await release_started.wait()
    assert sid in manager.active_runs
    request.cancel()
    await asyncio.sleep(0)
    assert not request.done()
    assert sid in manager.active_runs
    request.cancel()
    await asyncio.sleep(0)
    assert not request.done()
    assert sid in manager.active_runs

    finish_release.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert sid not in manager.active_runs
    assert (session_dir / "cache" / "predictions" / "report.html").is_file()


@pytest.mark.asyncio
async def test_report_generation_adds_support_paths_to_sys_path(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "sys-path"
    await manager.create_session(sid)
    session_dir = manager.get_session_dir(sid)
    predictions = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")

    service_dir = Path(artifacts_router.__file__).parent.parent.parent
    apps_dir = service_dir.parent
    repo_root = apps_dir.parent
    old_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if p not in {str(apps_dir), str(repo_root)}]

    reporting = ModuleType("joint_agent.tasks.reporting")

    class Task:
        def run(self, _context: dict[str, Any], _store: Any) -> None:
            return None

    reporting.GeneratePredictionReportTask = Task  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "joint_agent.tasks.reporting", reporting)
    try:
        await artifacts_router._generate_report_on_demand(
            session_dir,
            predictions,
            dataset,
        )
        assert str(apps_dir) in sys.path
        assert str(repo_root) in sys.path
    finally:
        sys.path[:] = old_path


def test_cache_namespace_swap_preserves_backup_when_restore_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_dir = tmp_path / "publication"
    publication_dir.mkdir()
    (publication_dir / "new.jsonl").write_text("new\n", encoding="utf-8")
    cache_dir = tmp_path / "cache" / "dataset"
    cache_dir.mkdir(parents=True)
    (cache_dir / "old.jsonl").write_text("old\n", encoding="utf-8")
    original_rename = pipeline_router.os.rename

    def fail_swap_and_restore(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if source.endswith(".staging") and target == cache_dir.name:
            raise OSError("swap failed")
        if source.endswith(".backup") and target == cache_dir.name:
            raise OSError("restore failed")
        original_rename(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(pipeline_router.os, "rename", fail_swap_and_restore)
    with pytest.raises(OSError, match="restore failed"):
        pipeline_router._replace_cache_namespace_from_publication(
            publication_dir,
            cache_dir,
        )

    backups = list(cache_dir.parent.glob(".dataset.*.backup"))
    assert len(backups) == 1
    assert (backups[0] / "old.jsonl").read_text(encoding="utf-8") == "old\n"
    assert not list(cache_dir.parent.glob(".dataset.*.staging"))


@pytest.mark.asyncio
async def test_missing_cache_binding_unlinks_stale_symlink(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "missing-cache-binding"
    session_dir = await manager.create_session(session_id)
    manager.metadata[session_id]["cache_publications"] = {"predictions": "a" * 32}
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    cache_dir = session_dir / "cache" / "dataset"
    cache_dir.parent.mkdir(parents=True)
    cache_dir.symlink_to(external_dir, target_is_directory=True)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_cache_namespace(
            manager,
            session_id,
            session_dir,
            "dataset",
        )

    assert exc_info.value.status_code == 409
    assert not cache_dir.is_symlink()
    assert external_dir.is_dir()


def test_explicit_null_cache_binding_fails_closed() -> None:
    assert pipeline_router.parse_cache_publications({}) is None
    assert pipeline_router.parse_cache_publications({"cache_publications": None}) == {}


@pytest.mark.asyncio
async def test_legacy_regeneration_restores_all_cache_namespaces(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    session_id = "legacy-cache-migration"
    session_dir = await manager.create_session(session_id)

    await pipeline_router._restore_regeneration_prerequisites(
        manager,
        session_id,
        session_dir,
        {"project": {"working_dir": str(session_dir / "cache")}},
        ["identify_asset"],
    )

    assert manager.pulled == ["cache/dataset/", "cache/predictions/"]


@pytest.mark.asyncio
async def test_lost_claim_drains_inflight_thread_before_returning() -> None:
    thread_started = threading.Event()
    thread_release = threading.Event()
    thread_finished = threading.Event()
    lose_claim = asyncio.Event()

    def blocked_swap() -> None:
        thread_started.set()
        assert thread_release.wait(timeout=2)
        thread_finished.set()

    async def stopped_guard() -> None:
        await lose_claim.wait()
        raise RuntimeError("claim lost")

    class Guard:
        task = asyncio.create_task(stopped_guard())

    operation = asyncio.create_task(
        pipeline_router._await_with_run_claim(
            asyncio.to_thread(blocked_swap),
            Guard(),
        )
    )
    while not thread_started.is_set():
        await asyncio.sleep(0)
    lose_claim.set()
    await asyncio.sleep(0.01)
    assert not operation.done()

    thread_release.set()
    with pytest.raises(HTTPException) as exc_info:
        await operation
    assert exc_info.value.status_code == 409
    assert thread_finished.is_set()


def test_pipeline_config_integrity_and_cache_path_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def integrity_failure(*_args: object, **_kwargs: object):
        raise pipeline_router._PipelineConfigIntegrityError("injected")
        yield

    with pytest.raises(
        pipeline_router._PipelineConfigIntegrityError,
        match="could not be verified",
    ):
        with pipeline_router._open_storage_root(tmp_path / "missing-root"):
            pass

    monkeypatch.setattr(
        pipeline_router,
        "_directory_open_flags",
        lambda: (_ for _ in ()).throw(
            pipeline_router._PipelineConfigIntegrityError("flags")
        ),
    )
    with pytest.raises(pipeline_router._PipelineConfigIntegrityError, match="flags"):
        with pipeline_router._open_storage_root(tmp_path):
            pass

    monkeypatch.setattr(
        pipeline_router,
        "pipeline_config_publication_key",
        lambda _publication_id: (_ for _ in ()).throw(ValueError("bad key")),
    )
    with pytest.raises(
        pipeline_router._PipelineConfigIntegrityError,
        match="not canonical",
    ):
        pipeline_router._pipeline_config_parent_components("sid", "publication")

    directory = tmp_path / "directory"
    directory.mkdir()
    descriptor = pipeline_router.os.open(directory, pipeline_router.os.O_RDONLY)
    try:
        pipeline_router._remove_descriptor_entry(descriptor, "missing")
    finally:
        pipeline_router.os.close(descriptor)

    monkeypatch.setattr(pipeline_router, "_open_storage_root", integrity_failure)
    with pytest.raises(pipeline_router._PipelineConfigIntegrityError, match="injected"):
        pipeline_router._validate_pipeline_config_publication(
            tmp_path,
            "sid",
            "publication",
            "0" * 64,
        )

    with pytest.raises(HTTPException) as invalid_config:
        pipeline_router._load_rebased_pipeline_config(
            tmp_path / "missing.yaml",
            "sid",
            tmp_path / "sid",
        )
    assert invalid_config.value.status_code == 400

    with pytest.raises(RuntimeError, match="share one session root"):
        pipeline_router._replace_cache_namespace_from_publication(
            tmp_path / "outside" / "publication",
            tmp_path / "session" / "cache" / "dataset",
        )
    pipeline_router._remove_cache_namespace(
        tmp_path / "missing-session" / "cache" / "dataset"
    )


@pytest.mark.asyncio
async def test_pipeline_restore_and_quarantine_generic_failure_edges(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "restore-errors"
    session_dir = await manager.create_session(session_id)
    publication_id = "a" * 32
    manager.store.kind = "s3"

    async def fail_delete(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("delete failed")

    monkeypatch.setattr(manager.store, "delete_key", fail_delete)
    await pipeline_router._quarantine_pipeline_config_publication(
        manager,
        session_id,
        publication_id,
    )

    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: publication_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: "invalid",
        }
    )

    async def no_op_sync(*_args: object, **_kwargs: object) -> int:
        return 0

    monkeypatch.setattr(manager, "sync_from_store", no_op_sync)
    with pytest.raises(HTTPException) as digest_failure:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            session_dir / "input" / "config.yaml",
        )
    assert digest_failure.value.status_code == 409
    manager.metadata[session_id][PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD] = "0" * 64

    async def fail_sync(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(manager, "sync_from_store", fail_sync)
    with pytest.raises(HTTPException) as restore_failure:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            session_dir / "input" / "config.yaml",
        )
    assert restore_failure.value.status_code == 409

    async def successful_sync(*_args: object, **_kwargs: object) -> int:
        return 0

    monkeypatch.setattr(manager, "sync_from_store", successful_sync)
    monkeypatch.setattr(
        pipeline_router,
        "_replace_cache_namespace_from_publication",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("invalid publication")),
    )
    manager.metadata[session_id]["cache_publications"] = {"dataset": publication_id}
    with pytest.raises(HTTPException) as cache_failure:
        await pipeline_router._restore_cache_namespace(
            manager,
            session_id,
            session_dir,
            "dataset",
        )
    assert cache_failure.value.status_code == 409

    manager.metadata[session_id].pop("cache_publications")
    monkeypatch.setattr(manager, "sync_from_store", fail_sync)
    with pytest.raises(HTTPException) as prerequisite_failure:
        await pipeline_router._restore_regeneration_prerequisites(
            manager,
            session_id,
            session_dir,
            {"project": {"working_dir": str(session_dir / "cache")}},
            ["identify_asset"],
        )
    assert prerequisite_failure.value.status_code == 503


@pytest.mark.asyncio
async def test_pipeline_s3_and_upload_generic_failures(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    def fail_download(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("download failed")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_download)
    with pytest.raises(HTTPException) as immediate_failure:
        await pipeline_router.upload_usd_immediate(
            None,
            "s3://bucket/scene.usda",
        )
    assert immediate_failure.value.status_code == 500
    assert immediate_failure.value.detail == "Failed to download USD from S3"
    assert "download failed" not in immediate_failure.value.detail

    def successful_download(_uri: str, session_dir: Path) -> Path:
        path = session_dir / "input" / "scene.usda"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n", encoding="utf-8")
        return path

    monkeypatch.setattr(
        pipeline_router,
        "_download_s3_to_session",
        successful_download,
    )
    manager.sync_to_store_error = RuntimeError("sync failed")
    assert (
        await pipeline_router.upload_usd_immediate(
            None,
            "s3://bucket/scene.usda",
        )
    ).status == "ready"
    manager.sync_to_store_error = None

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_download)

    with pytest.raises(HTTPException) as pipeline_failure:
        await pipeline_router.create_pipeline(
            None,
            None,
            "s3://bucket/scene.usda",
            "",
            "",
        )
    assert pipeline_failure.value.status_code == 500
    assert pipeline_failure.value.detail == "Failed to download USD from S3"
    assert "download failed" not in pipeline_failure.value.detail

    async def fail_upload(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(pipeline_router, "_stream_copy", fail_upload)
    with pytest.raises(HTTPException) as upload_failure:
        await pipeline_router.create_pipeline(
            _Upload("scene.usda", [b"#usda\n"]),
            None,
            None,
            "",
            "",
        )
    assert upload_failure.value.status_code == 500
    assert upload_failure.value.detail == "Failed to save USD file"
    assert "upload failed" not in upload_failure.value.detail

    existing_session = "nonmapping-config"
    existing_dir = await manager.create_session(existing_session)
    manager.metadata[existing_session]["config"] = "invalid"  # type: ignore[assignment]
    (existing_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **kwargs: {
            "project": {
                "name": "test",
                "session_id": kwargs["session_id"],
                "working_dir": kwargs["working_dir"],
            },
            "input": {"usd_path": kwargs["usd_path"]},
            "steps": {},
        },
    )

    async def persisted(*_args: object, **_kwargs: object) -> str:
        return "0" * 64

    monkeypatch.setattr(pipeline_router, "_persist_pipeline_config", persisted)
    assert (
        await pipeline_router.create_pipeline(
            None,
            existing_session,
            None,
            "",
            "",
        )
    ).status == "pending"


def _pipeline_config_publication_file(
    storage: Path,
    session_id: str,
    publication_id: str,
) -> Path:
    return (
        storage
        / session_id
        / pipeline_router.pipeline_config_publication_key(publication_id)
    )


def test_pipeline_config_quarantine_race_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_id = "a" * 32

    missing_storage = tmp_path / "missing"
    _pipeline_config_publication_file(
        missing_storage, "sid", publication_id
    ).parent.mkdir(parents=True)
    pipeline_router._move_pipeline_config_to_quarantine(
        missing_storage, "sid", publication_id
    )

    existing_storage = tmp_path / "existing"
    existing_source = _pipeline_config_publication_file(
        existing_storage, "sid", publication_id
    )
    existing_source.parent.mkdir(parents=True)
    existing_source.write_text("project: {}\n", encoding="utf-8")
    quarantine = (
        existing_storage / ".quarantine" / "sid" / "pipeline_configs" / publication_id
    )
    quarantine.mkdir(parents=True)
    (quarantine / "config.yaml").write_text("older", encoding="utf-8")
    pipeline_router._move_pipeline_config_to_quarantine(
        existing_storage, "sid", publication_id
    )
    assert len(list(quarantine.glob("config.*.yaml"))) == 1

    open_failure_storage = tmp_path / "open-failure"
    open_failure_source = _pipeline_config_publication_file(
        open_failure_storage, "sid", publication_id
    )
    open_failure_source.parent.mkdir(parents=True)
    open_failure_source.write_text("project: {}\n", encoding="utf-8")
    real_open = pipeline_router.os.open
    config_open_count = 0

    def fail_second_config_open(path, *args, **kwargs):
        nonlocal config_open_count
        if path == "config.yaml":
            config_open_count += 1
            if config_open_count == 2:
                raise OSError("quarantine open failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(pipeline_router.os, "open", fail_second_config_open)
    with pytest.raises(OSError, match="quarantine open failed"):
        pipeline_router._move_pipeline_config_to_quarantine(
            open_failure_storage,
            "sid",
            publication_id,
        )
    monkeypatch.setattr(pipeline_router.os, "open", real_open)

    drift_storage = tmp_path / "drift"
    drift_source = _pipeline_config_publication_file(
        drift_storage, "sid", publication_id
    )
    drift_source.parent.mkdir(parents=True)
    drift_source.write_text("project: {}\n", encoding="utf-8")
    real_fstat = pipeline_router.os.fstat
    file_stat_count = 0

    def drift_second_file_stat(descriptor: int):
        nonlocal file_stat_count
        result = real_fstat(descriptor)
        if pipeline_router.stat.S_ISREG(result.st_mode):
            file_stat_count += 1
            if file_stat_count == 2:
                return SimpleNamespace(
                    st_mode=result.st_mode,
                    st_dev=result.st_dev,
                    st_ino=result.st_ino + 1,
                )
        return result

    monkeypatch.setattr(pipeline_router.os, "fstat", drift_second_file_stat)
    with pytest.raises(OSError, match="source changed during move"):
        pipeline_router._move_pipeline_config_to_quarantine(
            drift_storage,
            "sid",
            publication_id,
        )


def test_pipeline_config_replace_and_cache_cleanup_defensive_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "storage"
    (storage / "sid" / "input").mkdir(parents=True)
    monkeypatch.setattr(
        pipeline_router.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen failed")),
    )
    with pytest.raises(OSError, match="fdopen failed"):
        pipeline_router._replace_pipeline_config_from_validated_bytes(
            storage,
            "sid",
            b"project: {}\n",
        )

    session_dir = tmp_path / "cache-session"
    publication_dir = session_dir / "publication"
    publication_dir.mkdir(parents=True)
    (publication_dir / "new.jsonl").write_text("new\n", encoding="utf-8")
    cache_dir = session_dir / "cache" / "dataset"
    cache_dir.parent.mkdir(parents=True)
    staging_paths: list[Path] = []

    def replace_staging_directory(_descriptor: int):
        candidates = list(cache_dir.parent.glob(".dataset.*.staging"))
        assert len(candidates) == 1
        staging_path = candidates[0]
        staging_paths.append(staging_path)
        pipeline_router.shutil.rmtree(staging_path)
        staging_path.write_text("changed type", encoding="utf-8")
        raise RuntimeError("staging changed type")
        yield

    monkeypatch.setattr(
        pipeline_router,
        "iter_open_regular_files",
        replace_staging_directory,
    )
    with pytest.raises(RuntimeError, match="staging changed type"):
        pipeline_router._replace_cache_namespace_from_publication(
            publication_dir,
            cache_dir,
        )
    assert staging_paths and not staging_paths[0].exists()

    class _UnrelatedCache:
        @property
        def parent(self):
            return self

        def absolute(self):
            return self

        def relative_to(self, _other):
            raise ValueError("outside")

    with pytest.raises(RuntimeError, match="inside its session root"):
        pipeline_router._remove_cache_namespace(_UnrelatedCache())  # type: ignore[arg-type]
