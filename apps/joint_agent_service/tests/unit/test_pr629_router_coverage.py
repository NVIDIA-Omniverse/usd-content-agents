# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused behavioral coverage for PR 629 router ownership and restore paths."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi import HTTPException, UploadFile
from world_understanding.utils.credentials import InlineSecretError

from ...service.models.requests import PipelineStep, RegenerateRequest
from ...service.routers import artifacts_router, pipeline_router
from ...service.runtime.registry import JobRegistry
from ...service.session.cache_publications import (
    PIPELINE_CONFIG_PUBLICATION_ID_FIELD,
    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD,
    pipeline_config_publication_path,
)
from .test_router_edges_additional_coverage import _Manager, _Registry


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Manager:
    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)  # type: ignore[arg-type]
    artifacts_router.set_session_manager(manager)  # type: ignore[arg-type]
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)
    monkeypatch.setattr(pipeline_router.config, "vlm_backend", "mock")
    monkeypatch.setattr(pipeline_router.config, "vlm_model", "mock")
    monkeypatch.setattr(pipeline_router.config, "llm_backend", "mock")
    monkeypatch.setattr(pipeline_router.config, "llm_model", "mock")
    return manager


def _pipeline_config(session_id: str, session_dir: Path) -> dict[str, Any]:
    return {
        "project": {
            "name": "coverage",
            "session_id": session_id,
            "working_dir": str(session_dir / "cache"),
        },
        "input": {"usd_path": str(session_dir / "input" / "scene.usda")},
        "steps": {},
    }


def _bind_pipeline_config_publication(
    manager: _Manager,
    session_id: str,
    config_path: Path,
    *,
    run_id: str = "d" * 32,
) -> None:
    """Bind a test session to an immutable copy of its current config."""
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


async def _create_existing_pipeline(session_id: str) -> None:
    await pipeline_router.create_pipeline(
        usd_file=None,
        session_id=session_id,
        s3_uri=None,
        user_prompt="",
        render_backend="",
        apply_joint_rigger=False,
        joint_rigger_adapter="",
        joint_rigger_on_missing_dependency="",
        joint_rigger_on_unready_candidates="",
        joint_rigger_template="",
        joint_rigger_apply_masses=None,
        joint_rigger_apply_collision=None,
    )


@pytest.mark.asyncio
async def test_report_heartbeat_stops_after_claim_loss(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager.run_claim_heartbeat_seconds = 0
    renewals = iter((True, False))

    async def renew_then_lose(_session_id: str, _run_id: str) -> bool:
        return next(renewals)

    monkeypatch.setattr(manager, "renew_run", renew_then_lose)

    await artifacts_router._maintain_report_claim(manager, "session", "lost-run")

    with pytest.raises(StopIteration):
        next(renewals)


@pytest.mark.asyncio
async def test_report_returns_not_found_when_published_stream_is_missing(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "report-stream-missing"
    session_dir = await manager.create_session(session_id)
    report_path = session_dir / "cache" / "predictions" / "report.html"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("<html>ready</html>", encoding="utf-8")

    async def missing_stream(_session_id: str, _artifact_type: str) -> None:
        return None

    monkeypatch.setattr(manager, "get_artifact_stream", missing_stream)

    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Prediction report not available"
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_completed_claim_guard_is_rejected_before_admission() -> None:
    guard_task = asyncio.create_task(asyncio.sleep(0))
    await guard_task
    guard = SimpleNamespace(task=guard_task)
    admission = asyncio.get_running_loop().create_future()
    admission.set_result("unused")

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._await_with_run_claim(admission, guard)

    assert exc_info.value.status_code == 409
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_cancelled_admission_drains_and_observes_inner_failure() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_after_release() -> None:
        started.set()
        await release.wait()
        raise RuntimeError("admission failed while draining")

    operation = asyncio.create_task(
        pipeline_router._drain_admission_operation(fail_after_release())
    )
    await started.wait()

    operation.cancel()
    await asyncio.sleep(0)
    operation.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await operation


@pytest.mark.asyncio
async def test_pipeline_config_persistence_requires_store_confirmation(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_config(_session_id: str, _key: str) -> bool:
        return False

    monkeypatch.setattr(manager.store, "exists", missing_config)
    session_id = "not-persisted"
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input" / "scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "a" * 32,
            _pipeline_config(session_id, session_dir),
            input_usd_path,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Pipeline config could not be persisted"


@pytest.mark.asyncio
async def test_pipeline_config_persistence_bounds_tampered_store_verification(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "oversized-store-publication"
    run_id = "e" * 32
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input" / "scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    class OversizedStoreStream:
        def __init__(self) -> None:
            self.remaining = pipeline_router._MAX_PIPELINE_CONFIG_PUBLICATION_BYTES + 2
            self.read_sizes: list[int] = []
            self.closed = False

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            returned = min(size, self.remaining)
            self.remaining -= returned
            return b"x" * returned

        def close(self) -> None:
            self.closed = True

    stream = OversizedStoreStream()

    async def open_tampered_store_stream(
        _session_id: str,
        _key: str,
    ) -> OversizedStoreStream:
        return stream

    monkeypatch.setattr(manager.store, "open_read", open_tampered_store_stream)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            run_id,
            _pipeline_config(session_id, session_dir),
            input_usd_path,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Pipeline config could not be persisted"
    assert sum(stream.read_sizes) == (
        pipeline_router._MAX_PIPELINE_CONFIG_PUBLICATION_BYTES + 1
    )
    assert max(stream.read_sizes) <= 2 * 1024 * 1024
    assert stream.closed is True
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    assert PIPELINE_CONFIG_PUBLICATION_ID_FIELD not in manager.metadata[session_id]
    assert PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD not in manager.metadata[session_id]


@pytest.mark.asyncio
async def test_pipeline_config_persistence_backend_failure_is_value_free(
    manager: _Manager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "backend-failure"
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input/scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    sentinel = "never-log-joint-store-signature"
    manager.sync_to_store_error = RuntimeError(sentinel)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "b" * 32,
            _pipeline_config(session_id, session_dir),
            input_usd_path,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Pipeline config could not be persisted"
    assert exc_info.value.__cause__ is None
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_pipeline_config_persistence_publication_failure_is_value_free(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "publication-failure"
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input/scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    sentinel = "never-log-joint-publication-signature"

    def fail_publication(*_args: Any, **_kwargs: Any) -> Path:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        pipeline_router,
        "_write_pipeline_config_publication",
        fail_publication,
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "d" * 32,
            _pipeline_config(session_id, session_dir),
            input_usd_path,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Pipeline config could not be persisted"
    assert exc_info.value.__cause__ is None
    assert manager.synced == []
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_pipeline_config_persistence_typed_backend_failure_is_not_a_client_error(
    manager: _Manager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "typed-backend-failure"
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input/scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    sentinel = "never-misclassify-joint-store-signature"
    manager.sync_to_store_error = InlineSecretError(sentinel)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "c" * 32,
            _pipeline_config(session_id, session_dir),
            input_usd_path,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Pipeline config could not be persisted"
    assert exc_info.value.__cause__ is None
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_pipeline_config_persistence_rejects_inline_credentials_before_write(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "inline-credential"
    run_id = "c" * 32
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input" / "scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    pipeline_config = _pipeline_config(session_id, session_dir)
    sentinel = "sentinel-joint-key"
    pipeline_config["steps"] = {
        "predict": {"vlm": {"api_key": sentinel}},
    }
    sync_to_store = AsyncMock()
    monkeypatch.setattr(manager, "sync_to_store", sync_to_store)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            run_id,
            pipeline_config,
            input_usd_path,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pipeline configuration is invalid"
    assert exc_info.value.__cause__ is None
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    sync_to_store.assert_not_awaited()
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_pipeline_config_persistence_rejects_url_credentials_before_write(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "url-credential"
    run_id = "d" * 32
    session_dir = await manager.create_session(session_id)
    input_usd_path = session_dir / "input" / "scene.usda"
    input_usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    pipeline_config = _pipeline_config(session_id, session_dir)
    sentinel = "sentinel-joint-signature"
    pipeline_config["steps"] = {
        "predict": {
            "vlm": {
                "base_url": (f"https://vlm.example.test/v1?X-Amz-Signature={sentinel}")
            }
        }
    }
    sync_to_store = AsyncMock()
    monkeypatch.setattr(manager, "sync_to_store", sync_to_store)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            run_id,
            pipeline_config,
            input_usd_path,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pipeline configuration is invalid"
    assert exc_info.value.__cause__ is None
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    sync_to_store.assert_not_awaited()
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_create_pipeline_rejects_inline_credentials_before_local_write(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    session_id = "create-inline-credential"
    session_dir = await manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    sentinel = "sentinel-create-pipeline-key"
    unsafe_config = _pipeline_config(session_id, session_dir)
    unsafe_config["steps"] = {"predict": {"vlm": {"api_key": sentinel}}}
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **_kwargs: unsafe_config,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pipeline configuration is invalid"
    assert not (session_dir / "input" / "config.yaml").exists()
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs
    assert session_id in manager.sessions
    assert session_id not in manager.deleted
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text


@pytest.mark.asyncio
async def test_create_pipeline_rejects_oversized_config_before_publication(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    run_id = "1" * 32
    monkeypatch.setattr(
        pipeline_router.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=run_id),
    )
    session_id = "oversized-initial-config"
    session_dir = await manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    oversized_config = _pipeline_config(session_id, session_dir)
    oversized_config["padding"] = "x" * (
        pipeline_router._MAX_PIPELINE_CONFIG_PUBLICATION_BYTES + 1
    )
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **_kwargs: oversized_config,
    )

    async def allow_async_scan(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipeline_router, "aensure_no_inline_secrets", allow_async_scan)
    monkeypatch.setattr(
        pipeline_router,
        "ensure_no_inline_secrets",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pipeline configuration is invalid"
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    assert not (session_dir / "input" / "config.yaml").exists()
    assert PIPELINE_CONFIG_PUBLICATION_ID_FIELD not in manager.metadata[session_id]
    assert PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD not in manager.metadata[session_id]
    assert manager.synced == []
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_create_pipeline_rejects_symlinked_publication_ancestor(
    manager: _Manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    run_id = "2" * 32
    monkeypatch.setattr(
        pipeline_router.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=run_id),
    )
    session_id = "aliased-initial-publication"
    session_dir = await manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    pipeline_config = _pipeline_config(session_id, session_dir)
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **_kwargs: pipeline_config,
    )
    outside_artifacts = tmp_path.parent / f"{tmp_path.name}-outside-artifacts"
    outside_artifacts.mkdir()
    (session_dir / "artifacts").symlink_to(
        outside_artifacts,
        target_is_directory=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 503
    assert list(outside_artifacts.rglob("config.yaml")) == []
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    assert not (session_dir / "input" / "config.yaml").exists()
    assert PIPELINE_CONFIG_PUBLICATION_ID_FIELD not in manager.metadata[session_id]
    assert PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD not in manager.metadata[session_id]
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_create_pipeline_rejects_publication_ancestor_swap(
    manager: _Manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    run_id = "3" * 32
    monkeypatch.setattr(
        pipeline_router.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=run_id),
    )
    session_id = "swapped-initial-publication"
    session_dir = await manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    pipeline_config = _pipeline_config(session_id, session_dir)
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **_kwargs: pipeline_config,
    )
    artifacts_root = session_dir / "artifacts"
    publication_root = artifacts_root / "publications" / "pipeline_config"
    publication_root.mkdir(parents=True)
    detached_artifacts = session_dir / "artifacts.detached"
    outside_artifacts = tmp_path.parent / f"{tmp_path.name}-swap-target"
    outside_artifacts.mkdir()
    original_open = pipeline_router.os.open
    swapped = False

    def swap_ancestor_before_leaf_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and path == "config.yaml"
            and flags & pipeline_router.os.O_WRONLY
            and flags & pipeline_router.os.O_CREAT
        ):
            artifacts_root.rename(detached_artifacts)
            artifacts_root.symlink_to(outside_artifacts, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(pipeline_router.os, "open", swap_ancestor_before_leaf_open)

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 503
    assert swapped is True
    assert artifacts_root.is_symlink()
    assert list(outside_artifacts.rglob("config.yaml")) == []
    assert list(detached_artifacts.rglob("config.yaml")) == []
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    assert not (session_dir / "input" / "config.yaml").exists()
    assert PIPELINE_CONFIG_PUBLICATION_ID_FIELD not in manager.metadata[session_id]
    assert PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD not in manager.metadata[session_id]
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_create_pipeline_rejects_symlinked_mutable_config_parent(
    manager: _Manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    run_id = "4" * 32
    monkeypatch.setattr(
        pipeline_router.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=run_id),
    )
    session_id = "aliased-initial-input"
    session_dir = await manager.create_session(session_id)
    input_root = session_dir / "input"
    input_root.rmdir()
    outside_input = tmp_path.parent / f"{tmp_path.name}-outside-input"
    outside_input.mkdir()
    (outside_input / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    outside_config = outside_input / "config.yaml"
    outside_config.write_bytes(b"outside-must-remain\n")
    input_root.symlink_to(outside_input, target_is_directory=True)
    pipeline_config = _pipeline_config(session_id, session_dir)
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **_kwargs: pipeline_config,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 503
    assert outside_config.read_bytes() == b"outside-must-remain\n"
    assert not pipeline_config_publication_path(session_dir, run_id).exists()
    assert PIPELINE_CONFIG_PUBLICATION_ID_FIELD not in manager.metadata[session_id]
    assert PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD not in manager.metadata[session_id]
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_pipeline_config_persistence_rejects_external_input(
    manager: _Manager,
) -> None:
    session_id = "external-input"
    session_dir = await manager.create_session(session_id)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "a" * 32,
            _pipeline_config(session_id, session_dir),
            session_dir.parent / "outside.usda",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Input USD is outside the session"

    contained_non_input = session_dir / "cache" / "scene.usda"
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._persist_pipeline_config(
            manager,
            session_id,
            "b" * 32,
            _pipeline_config(session_id, session_dir),
            contained_non_input,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Input USD is outside the session"


@pytest.mark.asyncio
async def test_pipeline_config_restore_requires_session_metadata(
    manager: _Manager,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            "missing-config-session",
            manager.get_session_dir("missing-config-session") / "input/config.yaml",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session not found"


@pytest.mark.asyncio
async def test_pipeline_config_restore_uses_exact_bound_publication(
    manager: _Manager,
) -> None:
    session_id = "bound-config"
    session_dir = await manager.create_session(session_id)
    run_id = "a" * 32
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.write_text("source: accepted\n", encoding="utf-8")
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: stale\n", encoding="utf-8")
    manager.metadata[session_id][PIPELINE_CONFIG_PUBLICATION_ID_FIELD] = run_id
    manager.metadata[session_id][PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD] = (
        hashlib.sha256(publication_path.read_bytes()).hexdigest()
    )

    await pipeline_router._restore_pipeline_config(
        manager,
        session_id,
        config_path,
    )

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {
        "source": "accepted"
    }
    assert manager.pulled[-1].endswith(f"/{run_id}/config.yaml")


@pytest.mark.asyncio
async def test_pipeline_config_restore_publishes_the_validated_bytes_once(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "config-replacement-race"
    session_dir = await manager.create_session(session_id)
    run_id = "9" * 32
    accepted_bytes = b"source: accepted\n"
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.write_bytes(accepted_bytes)
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: stale\n", encoding="utf-8")
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                accepted_bytes
            ).hexdigest(),
        }
    )
    original_validate = pipeline_router._validate_pipeline_config_publication

    def validate_then_replace(
        storage_path: Path,
        target_session_id: str,
        publication_id: str,
        expected_sha256: str,
    ) -> bytes:
        validated = original_validate(
            storage_path,
            target_session_id,
            publication_id,
            expected_sha256,
        )
        publication_path.write_text(
            "api_key: replacement-must-not-be-published\n",
            encoding="utf-8",
        )
        return validated

    monkeypatch.setattr(
        pipeline_router,
        "_validate_pipeline_config_publication",
        validate_then_replace,
    )

    await pipeline_router._restore_pipeline_config(
        manager,
        session_id,
        config_path,
    )

    assert config_path.read_bytes() == accepted_bytes
    assert b"replacement-must-not-be-published" not in config_path.read_bytes()


@pytest.mark.asyncio
async def test_pipeline_config_restore_quarantines_digest_mismatch(
    manager: _Manager,
) -> None:
    session_id = "tampered-config"
    session_dir = await manager.create_session(session_id)
    run_id = "b" * 32
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.write_text("source: tampered\n", encoding="utf-8")
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: accepted\n", encoding="utf-8")
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                b"source: accepted\n"
            ).hexdigest(),
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == pipeline_router._PIPELINE_CONFIG_INTEGRITY_DETAIL
    assert config_path.read_text(encoding="utf-8") == "source: accepted\n"
    assert not publication_path.exists()
    assert list(
        (manager.storage_path / ".quarantine" / session_id).rglob("config.yaml")
    )


@pytest.mark.asyncio
async def test_pipeline_config_quarantine_unlinks_symlink_without_following(
    manager: _Manager,
    tmp_path: Path,
) -> None:
    session_id = "symlink-config"
    session_dir = await manager.create_session(session_id)
    run_id = "e" * 32
    outside_target = tmp_path.parent / f"{tmp_path.name}-outside-config.yaml"
    outside_target.write_text("source: outside\n", encoding="utf-8")
    outside_target.chmod(0o644)
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.symlink_to(outside_target)
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: accepted\n", encoding="utf-8")
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                outside_target.read_bytes()
            ).hexdigest(),
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert not publication_path.exists()
    assert not publication_path.is_symlink()
    assert outside_target.read_text(encoding="utf-8") == "source: outside\n"
    assert outside_target.stat().st_mode & 0o777 == 0o644
    assert config_path.read_text(encoding="utf-8") == "source: accepted\n"


@pytest.mark.asyncio
async def test_pipeline_config_restore_rejects_symlinked_publication_ancestor(
    manager: _Manager,
    tmp_path: Path,
) -> None:
    session_id = "symlink-config-ancestor"
    session_dir = await manager.create_session(session_id)
    run_id = "8" * 32
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.parent.mkdir(parents=True)
    outside_run_dir = tmp_path.parent / f"{tmp_path.name}-outside-run"
    outside_run_dir.mkdir()
    outside_publication = outside_run_dir / "config.yaml"
    outside_bytes = b"source: external\n"
    outside_publication.write_bytes(outside_bytes)
    outside_publication.chmod(0o644)
    publication_path.parent.symlink_to(outside_run_dir, target_is_directory=True)
    config_path = session_dir / "input" / "config.yaml"
    canonical_bytes = b"source: canonical\n"
    config_path.write_bytes(canonical_bytes)
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                outside_bytes
            ).hexdigest(),
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == pipeline_router._PIPELINE_CONFIG_INTEGRITY_DETAIL
    assert publication_path.parent.is_symlink()
    assert outside_publication.read_bytes() == outside_bytes
    assert outside_publication.stat().st_mode & 0o777 == 0o644
    assert config_path.read_bytes() == canonical_bytes


@pytest.mark.asyncio
async def test_pipeline_config_quarantine_uses_held_parent_during_ancestor_swap(
    manager: _Manager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-validation ancestor swap cannot redirect quarantine outside."""

    session_id = "quarantine-ancestor-race"
    session_dir = await manager.create_session(session_id)
    run_id = "7" * 32
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    rejected_bytes = b"source: rejected\n"
    publication_path.write_bytes(rejected_bytes)

    config_path = session_dir / "input" / "config.yaml"
    canonical_bytes = b"source: canonical\n"
    config_path.write_bytes(canonical_bytes)
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                canonical_bytes
            ).hexdigest(),
        }
    )

    outside_run_dir = tmp_path.parent / f"{tmp_path.name}-outside-race-run"
    outside_run_dir.mkdir()
    outside_publication = outside_run_dir / "config.yaml"
    outside_bytes = b"source: outside-must-remain\n"
    outside_publication.write_bytes(outside_bytes)
    outside_publication.chmod(0o640)
    outside_mode = outside_publication.stat().st_mode & 0o777

    publication_parent = publication_path.parent
    detached_parent = publication_parent.with_name(f"{run_id}.detached")
    original_replace = pipeline_router.os.replace
    swapped = False

    def swap_ancestor_then_replace(
        source: Any,
        destination: Any,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and source == "config.yaml"
            and src_dir_fd is not None
            and dst_dir_fd is not None
        ):
            swapped = True
            publication_parent.rename(detached_parent)
            publication_parent.symlink_to(
                outside_run_dir,
                target_is_directory=True,
            )
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(pipeline_router.os, "replace", swap_ancestor_then_replace)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == pipeline_router._PIPELINE_CONFIG_INTEGRITY_DETAIL
    assert swapped is True
    assert publication_parent.is_symlink()
    assert outside_publication.read_bytes() == outside_bytes
    assert outside_publication.stat().st_mode & 0o777 == outside_mode
    assert config_path.read_bytes() == canonical_bytes
    quarantined = list(
        (manager.storage_path / ".quarantine" / session_id).rglob("config*.yaml")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == rejected_bytes
    assert quarantined[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_pipeline_config_quarantine_rejects_aliased_quarantine_root(
    manager: _Manager,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = "aliased-quarantine"
    session_dir = await manager.create_session(session_id)
    run_id = "f" * 32
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.write_text("source: rejected\n", encoding="utf-8")
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: accepted\n", encoding="utf-8")
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                b"source: accepted\n"
            ).hexdigest(),
        }
    )
    outside_quarantine = tmp_path.parent / f"{tmp_path.name}-outside-quarantine"
    outside_quarantine.mkdir()
    (manager.storage_path / ".quarantine").symlink_to(
        outside_quarantine,
        target_is_directory=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert "code=joint_pipeline_config_quarantine_failed" in caplog.text
    assert list(outside_quarantine.iterdir()) == []
    assert (manager.storage_path / ".quarantine").is_symlink()
    assert not publication_path.exists()
    assert config_path.read_text(encoding="utf-8") == "source: accepted\n"


@pytest.mark.asyncio
async def test_pipeline_config_restore_rejects_secret_even_with_forged_hash(
    manager: _Manager,
) -> None:
    session_id = "forged-secret-config"
    session_dir = await manager.create_session(session_id)
    run_id = "c" * 32
    sentinel = "forged-config-secret-727"
    publication_path = pipeline_config_publication_path(session_dir, run_id)
    publication_path.parent.mkdir(parents=True)
    publication_path.write_text(f"api_key: {sentinel}\n", encoding="utf-8")
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: accepted\n", encoding="utf-8")
    manager.metadata[session_id].update(
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: run_id,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: hashlib.sha256(
                publication_path.read_bytes()
            ).hexdigest(),
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert sentinel not in str(exc_info.value)
    assert config_path.read_text(encoding="utf-8") == "source: accepted\n"
    assert not publication_path.exists()


@pytest.mark.asyncio
async def test_pipeline_config_restore_fails_closed_without_integrity_binding(
    manager: _Manager,
) -> None:
    session_id = "legacy-unbound-config"
    session_dir = await manager.create_session(session_id)
    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text("source: legacy\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            config_path,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == pipeline_router._PIPELINE_CONFIG_INTEGRITY_DETAIL
    assert config_path.read_text(encoding="utf-8") == "source: legacy\n"


@pytest.mark.asyncio
async def test_pipeline_config_restore_rejects_explicit_null_binding(
    manager: _Manager,
) -> None:
    session_id = "null-config-binding"
    session_dir = await manager.create_session(session_id)
    manager.metadata[session_id][PIPELINE_CONFIG_PUBLICATION_ID_FIELD] = None

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_pipeline_config(
            manager,
            session_id,
            session_dir / "input" / "config.yaml",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Original config publication binding is invalid"


def test_stored_config_helpers_reject_malformed_and_non_durable_values(
    tmp_path: Path,
) -> None:
    original_session = tmp_path / "old" / "session"
    current_session = tmp_path / "new" / "session"
    external_path = str(tmp_path / "external" / "artifact.json")
    assert (
        pipeline_router._rebase_session_paths(
            external_path,
            original_session,
            current_session,
        )
        == external_path
    )

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._step_config({"steps": []}, "predict")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._step_config({"steps": {"predict": []}}, "predict")
    assert exc_info.value.status_code == 400

    session_dir = tmp_path / "session"
    dataset_dir = session_dir / "cache" / "dataset"
    predictions_dir = session_dir / "cache" / "predictions"
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._require_durable_path(
            session_dir / "cache" / "scratch" / "temporary.json",
            session_dir=session_dir,
            dataset_dir=dataset_dir,
            predictions_dir=predictions_dir,
            field_name="steps.predict.temporary_path",
        )
    assert exc_info.value.status_code == 409
    assert "non-durable session path" in exc_info.value.detail

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._required_config_path(
            {},
            "predictions_path",
            fallback=None,
            session_dir=session_dir,
            field_name="steps.predict.predictions_path",
        )
    assert exc_info.value.status_code == 409

    fallback = predictions_dir / "predictions.jsonl"
    assert (
        pipeline_router._required_config_path(
            {},
            "predictions_path",
            fallback=fallback,
            session_dir=session_dir,
            field_name="steps.predict.predictions_path",
        )
        == fallback
    )

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._regeneration_prerequisite_plan(
            {"project": []},
            ["predict"],
            session_dir,
        )
    assert exc_info.value.status_code == 400


def test_regeneration_plan_tracks_consistency_and_candidate_inputs(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    cache_dir = session_dir / "cache"
    canonical_predictions = cache_dir / "predictions" / "predictions.jsonl"
    consistency_config = {
        "project": {"working_dir": str(cache_dir)},
        "steps": {"consistency_pass": {}},
    }

    prefixes, _, _, prediction_paths, additional_paths = (
        pipeline_router._regeneration_prerequisite_plan(
            consistency_config,
            ["consistency_pass"],
            session_dir,
        )
    )
    assert prefixes == {"cache/predictions/"}
    assert prediction_paths == {canonical_predictions}
    assert additional_paths == set()

    missing_dataset_config = {
        "project": {"working_dir": str(cache_dir)},
        "steps": {
            "infer_articulation_candidates": {
                "predictions_path": str(canonical_predictions),
                "adjudication": {"require_source_images": True},
            }
        },
    }
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._regeneration_prerequisite_plan(
            missing_dataset_config,
            ["infer_articulation_candidates"],
            session_dir,
        )
    assert exc_info.value.status_code == 409
    assert "dataset_path" in exc_info.value.detail

    prim_metadata = session_dir / "input" / "prim_metadata.json"
    candidate_config = {
        "project": {"working_dir": str(cache_dir)},
        "steps": {
            "infer_articulation_candidates": {
                "predictions_path": str(canonical_predictions),
                "prim_metadata_path": str(prim_metadata),
            }
        },
    }
    prefixes, _, _, prediction_paths, additional_paths = (
        pipeline_router._regeneration_prerequisite_plan(
            candidate_config,
            ["infer_articulation_candidates"],
            session_dir,
        )
    )
    assert prefixes == {"cache/predictions/", "input/"}
    assert prediction_paths == {canonical_predictions}
    assert additional_paths == {prim_metadata}


def test_load_rebased_pipeline_config_rejects_cross_session_paths(
    tmp_path: Path,
) -> None:
    session_id = "selected-session"
    original_session_dir = tmp_path / "original" / session_id
    valid_config = {
        "project": {
            "session_id": session_id,
            "working_dir": str(original_session_dir / "cache"),
        },
        "input": {"usd_path": str(original_session_dir / "input" / "scene.usda")},
        "steps": {},
    }
    invalid_configs = [
        {
            **valid_config,
            "project": {**valid_config["project"], "session_id": "other-session"},
        },
        {
            **valid_config,
            "project": {
                **valid_config["project"],
                "working_dir": str(tmp_path / "original" / "other-session" / "cache"),
            },
        },
        {
            **valid_config,
            "input": {"usd_path": str(tmp_path / "outside.usda")},
        },
    ]
    config_path = tmp_path / "config.yaml"

    for invalid_config in invalid_configs:
        config_path.write_text(yaml.safe_dump(invalid_config), encoding="utf-8")
        with pytest.raises(HTTPException) as exc_info:
            pipeline_router._load_rebased_pipeline_config(
                config_path,
                session_id,
                tmp_path / "current" / session_id,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Original config is invalid for session"


def test_prerequisite_validation_rejects_invalid_render_metadata(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    dataset_dir = session_dir / "cache" / "dataset"
    usd_dir = dataset_dir / "usd"
    usd_dir.mkdir(parents=True)
    (usd_dir / "dataset.json").write_text("{}\n", encoding="utf-8")
    prims_path = usd_dir / "prims.jsonl"

    blank_line_jsonl = session_dir / "input" / "rows.jsonl"
    blank_line_jsonl.parent.mkdir(parents=True)
    blank_line_jsonl.write_text("\n{}\n", encoding="utf-8")
    assert list(pipeline_router._iter_jsonl_rows(blank_line_jsonl, session_dir)) == [{}]

    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._validate_referenced_file(
            None,
            base_dir=usd_dir,
            session_dir=session_dir,
            field_name="render path",
        )
    assert exc_info.value.status_code == 409

    prims_path.write_text(json.dumps({"renders": "invalid"}) + "\n", encoding="utf-8")
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._validate_render_dataset(dataset_dir, session_dir)
    assert exc_info.value.detail == (
        "Regeneration prerequisite contains invalid render metadata"
    )

    prims_path.write_text(json.dumps({"renders": [{}]}) + "\n", encoding="utf-8")
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._validate_render_dataset(dataset_dir, session_dir)
    assert exc_info.value.detail == (
        "Regeneration prerequisite contains invalid render metadata"
    )

    assert pipeline_router._dataset_image_paths(
        {
            "images": [{"path": "one.png"}, "two.png"],
            "image_path": "three.png",
        }
    ) == ["one.png", "two.png", "three.png"]


def test_cache_swap_restores_previous_directory_before_reraising(
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

    def fail_staging_swap(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if source.endswith(".staging") and target == cache_dir.name:
            raise OSError("swap failed")
        original_rename(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(pipeline_router.os, "rename", fail_staging_swap)

    with pytest.raises(OSError, match="swap failed"):
        pipeline_router._replace_cache_namespace_from_publication(
            publication_dir,
            cache_dir,
        )

    assert (cache_dir / "old.jsonl").read_text(encoding="utf-8") == "old\n"
    assert not list(cache_dir.parent.glob(".dataset.*.backup"))

    cache_without_backup = tmp_path / "cache" / "not-created"

    monkeypatch.setattr(pipeline_router.os, "rename", original_rename)

    def fail_before_backup(*_args: object, **_kwargs: object) -> bool:
        raise OSError("copy failed")

    monkeypatch.setattr(
        pipeline_router,
        "copy_open_file_to_confined",
        fail_before_backup,
    )
    with pytest.raises(OSError, match="copy failed"):
        pipeline_router._replace_cache_namespace_from_publication(
            publication_dir,
            cache_without_backup,
        )
    assert not cache_without_backup.exists()


def test_remove_cache_namespace_deletes_regular_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "dataset"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.jsonl").write_text("stale\n", encoding="utf-8")

    pipeline_router._remove_cache_namespace(cache_dir)

    assert not cache_dir.exists()


def test_cache_swap_holds_session_root_across_ancestor_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    publication_dir = session_dir / "artifacts" / "run_cache" / ("a" * 32)
    publication_dir = publication_dir / "cache" / "dataset"
    publication_dir.mkdir(parents=True)
    (publication_dir / "new.jsonl").write_text("new\n", encoding="utf-8")
    cache_dir = session_dir / "cache" / "dataset"
    cache_dir.mkdir(parents=True)
    (cache_dir / "old.jsonl").write_text("old\n", encoding="utf-8")
    detached = tmp_path / "session-held"
    outside = tmp_path / "outside-session"
    outside.mkdir()
    outside_sentinel = outside / "sentinel.txt"
    outside_sentinel.write_text("outside-cache-sentinel", encoding="utf-8")
    original_copy = pipeline_router.copy_open_file_to_confined
    swapped = False

    def swap_then_copy(*args: object, **kwargs: object) -> bool:
        nonlocal swapped
        if not swapped:
            swapped = True
            session_dir.rename(detached)
            session_dir.symlink_to(outside, target_is_directory=True)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_router,
        "copy_open_file_to_confined",
        swap_then_copy,
    )

    pipeline_router._replace_cache_namespace_from_publication(
        publication_dir,
        cache_dir,
    )

    assert (detached / "cache" / "dataset" / "new.jsonl").read_text(
        encoding="utf-8"
    ) == "new\n"
    assert not (detached / "cache" / "dataset" / "old.jsonl").exists()
    assert outside_sentinel.read_text(encoding="utf-8") == ("outside-cache-sentinel")


@pytest.mark.asyncio
async def test_cache_restore_requires_session_metadata(
    manager: _Manager,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_cache_namespace(
            manager,
            "missing-session",
            manager.get_session_dir("missing-session"),
            "dataset",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Session not found"


@pytest.mark.asyncio
async def test_regeneration_restore_syncs_direct_and_validates_additional_inputs(
    manager: _Manager,
) -> None:
    session_id = "restore-additional-input"
    session_dir = await manager.create_session(session_id)
    predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True)
    predictions_path.write_text("{}\n", encoding="utf-8")
    prim_metadata_path = session_dir / "input" / "prim_metadata.json"
    prim_metadata_path.write_text("{}\n", encoding="utf-8")
    pipeline_config = {
        "project": {"working_dir": str(session_dir / "cache")},
        "steps": {
            "infer_articulation_candidates": {
                "predictions_path": str(predictions_path),
                "prim_metadata_path": str(prim_metadata_path),
            }
        },
    }

    await pipeline_router._restore_regeneration_prerequisites(
        manager,
        session_id,
        session_dir,
        pipeline_config,
        ["infer_articulation_candidates"],
    )

    assert "input/" in manager.pulled
    assert "cache/predictions/" in manager.pulled


@pytest.mark.asyncio
async def test_regeneration_restore_preserves_prerequisite_http_error(
    manager: _Manager,
) -> None:
    session_id = "restore-missing-binding"
    session_dir = await manager.create_session(session_id)
    manager.metadata[session_id]["cache_publications"] = {}
    pipeline_config = {
        "project": {"working_dir": str(session_dir / "cache")},
        "steps": {},
    }

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._restore_regeneration_prerequisites(
            manager,
            session_id,
            session_dir,
            pipeline_config,
            ["predict"],
        )

    assert exc_info.value.status_code == 409
    assert "no dataset cache publication" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_pipeline_rejects_owned_and_superseded_runs(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        lambda **kwargs: _pipeline_config(
            kwargs["session_id"],
            Path(kwargs["working_dir"]).parent,
        ),
    )
    monkeypatch.setattr(pipeline_router, "_apply_render_request_limit", lambda _: None)

    owned_session = "create-owned"
    owned_dir = await manager.create_session(owned_session)
    (owned_dir / "input" / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    manager.active_runs[owned_session] = "a" * 32

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(owned_session)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "A pipeline run is already active for this session"

    superseded_session = "create-superseded"
    superseded_dir = await manager.create_session(superseded_session)
    (superseded_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )

    async def reject_update(
        _session_id: str,
        _run_id: str,
        _updates: dict[str, Any],
    ) -> bool:
        return False

    monkeypatch.setattr(manager, "update_session_for_run", reject_update)

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(superseded_session)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Pipeline run was superseded"
    assert superseded_session not in manager.active_runs


@pytest.mark.asyncio
async def test_fresh_joint_pipeline_failure_removes_credential_free_orphan(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(pipeline_router, "_apply_render_request_limit", lambda _: None)

    async def fail_publication(*_args: Any, **_kwargs: Any) -> str:
        raise HTTPException(status_code=503, detail="publication unavailable")

    monkeypatch.setattr(
        pipeline_router,
        "_persist_pipeline_config",
        fail_publication,
    )
    sessions_before = set(manager.sessions)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            usd_file=UploadFile(
                filename="scene.usda",
                file=io.BytesIO(b"#usda 1.0\n"),
            ),
            session_id=None,
            s3_uri=None,
            user_prompt="",
            render_backend="warp",
            apply_joint_rigger=False,
            joint_rigger_adapter="",
            joint_rigger_on_missing_dependency="",
            joint_rigger_on_unready_candidates="",
            joint_rigger_template="",
            joint_rigger_apply_masses=None,
            joint_rigger_apply_collision=None,
        )

    assert exc_info.value.status_code == 503
    assert manager.sessions == sessions_before
    assert manager.deleted
    assert registry.registered == []
    assert registry.admissions == {}
    assert manager.active_runs == {}


@pytest.mark.asyncio
async def test_fresh_joint_reservation_failure_removes_owned_session(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_reservation(*_args: Any, **_kwargs: Any) -> Any:
        raise HTTPException(status_code=409, detail="admission unavailable")

    monkeypatch.setattr(
        pipeline_router,
        "_reserve_run_admission",
        fail_reservation,
    )
    sessions_before = set(manager.sessions)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            usd_file=UploadFile(
                filename="scene.usda",
                file=io.BytesIO(b"#usda 1.0\n"),
            ),
            session_id=None,
            s3_uri=None,
            user_prompt="",
            render_backend="warp",
            apply_joint_rigger=False,
            joint_rigger_adapter="",
            joint_rigger_on_missing_dependency="",
            joint_rigger_on_unready_candidates="",
            joint_rigger_template="",
            joint_rigger_apply_masses=None,
            joint_rigger_apply_collision=None,
        )

    assert exc_info.value.status_code == 409
    assert manager.sessions == sessions_before
    assert manager.deleted
    assert manager.active_runs == {}


@pytest.mark.asyncio
async def test_create_pipeline_rejects_draining_local_task_before_reservation(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    registry.running = True
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    session_id = "create-local-drain"
    session_dir = await manager.create_session(session_id)
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(HTTPException) as exc_info:
        await _create_existing_pipeline(session_id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "A pipeline run is already active or still draining for this session"
    )
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_local_admission_releases_when_distributed_reservation_fails(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    async def fail_reservation(_session_id: str, _run_id: str) -> bool:
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(manager, "reserve_run", fail_reservation)

    with pytest.raises(RuntimeError, match="store unavailable"):
        await pipeline_router._reserve_run_admission(
            manager,
            "reservation-error",
            "a" * 32,
        )

    assert registry.admissions == {}


@pytest.mark.asyncio
async def test_local_admission_blocks_reclaim_while_old_work_drains(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    session_id = "draining-admission"
    await manager.create_session(session_id)
    owner_run = "a" * 32

    await pipeline_router._reserve_run_admission(manager, session_id, owner_run)
    assert await manager.release_run(session_id, owner_run)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._reserve_run_admission(
            manager,
            session_id,
            "b" * 32,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "A pipeline run is already active or still draining for this session"
    )
    assert session_id not in manager.active_runs
    assert await registry.release_admission(session_id, owner_run)


@pytest.mark.asyncio
async def test_cancel_rejects_a_stale_run_claim(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    session_id = "cancel-stale"
    await manager.create_session(session_id)
    manager.metadata[session_id].update(
        {"status": "running", "active_run_id": "b" * 32}
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.cancel_pipeline(session_id, "a" * 32)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Pipeline run changed before cancellation could be accepted"
    )
    assert registry.cancelled == []


@pytest.mark.asyncio
async def test_regenerate_pipeline_rejects_owned_and_superseded_runs(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(pipeline_router, "_apply_render_request_limit", lambda _: None)
    request = RegenerateRequest(steps=[PipelineStep.IDENTIFY_ASSET])

    owned_session = "regenerate-owned"
    await manager.create_session(owned_session)
    manager.active_runs[owned_session] = "c" * 32
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(owned_session, request)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "A pipeline run is already active for this session"

    superseded_session = "regenerate-superseded"
    superseded_dir = await manager.create_session(superseded_session)
    manager.metadata[superseded_session]["cache_publications"] = {}
    config_path = superseded_dir / "input" / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_pipeline_config(superseded_session, superseded_dir)),
        encoding="utf-8",
    )
    _bind_pipeline_config_publication(
        manager,
        superseded_session,
        config_path,
    )

    async def reject_update(
        _session_id: str,
        _run_id: str,
        _updates: dict[str, Any],
    ) -> bool:
        return False

    monkeypatch.setattr(manager, "update_session_for_run", reject_update)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(superseded_session, request)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Pipeline run was superseded"
    assert superseded_session not in manager.active_runs


@pytest.mark.asyncio
async def test_regenerate_pipeline_rejects_malformed_input_config(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(pipeline_router, "_apply_render_request_limit", lambda _: None)
    session_id = "regenerate-invalid-input"
    session_dir = await manager.create_session(session_id)
    manager.metadata[session_id]["cache_publications"] = {}
    config_path = session_dir / "input/config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"working_dir": str(session_dir / "cache")},
                "input": [],
                "steps": {},
            }
        ),
        encoding="utf-8",
    )
    _bind_pipeline_config_publication(manager, session_id, config_path)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(steps=[PipelineStep.IDENTIFY_ASSET]),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Original config is invalid for session"
    assert session_id not in manager.active_runs


@pytest.mark.asyncio
async def test_regenerate_pipeline_rejects_inline_prompt_before_publication(
    manager: _Manager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = _Registry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(pipeline_router, "_apply_render_request_limit", lambda _: None)
    monkeypatch.setattr(
        pipeline_router,
        "_apply_prediction_request_limit",
        lambda _: None,
    )
    session_id = "regenerate-inline-prompt"
    session_dir = await manager.create_session(session_id)
    manager.metadata[session_id]["cache_publications"] = {}
    input_path = session_dir / "input/scene.usda"
    input_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = session_dir / "input/config.yaml"
    config_path.write_text(
        yaml.safe_dump(_pipeline_config(session_id, session_dir)),
        encoding="utf-8",
    )
    _bind_pipeline_config_publication(manager, session_id, config_path)
    sentinel = "never-persist-regenerate-bearer"

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            session_id,
            RegenerateRequest(
                steps=[PipelineStep.IDENTIFY_ASSET],
                user_prompt=f"Authorization: Bearer {sentinel}",
            ),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Pipeline configuration is invalid"
    assert exc_info.value.__cause__ is None
    assert manager.synced == []
    assert (
        manager.metadata[session_id][PIPELINE_CONFIG_PUBLICATION_ID_FIELD] == "d" * 32
    )
    expected_publication = pipeline_config_publication_path(session_dir, "d" * 32)
    publications = list((session_dir / "artifacts/pipeline_configs").rglob("*.yaml"))
    assert publications == [expected_publication]
    assert registry.registered == []
    assert registry.admissions == {}
    assert session_id not in manager.active_runs
    observable_text = f"{exc_info.value.detail}\n{caplog.text}"
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in observable_text
