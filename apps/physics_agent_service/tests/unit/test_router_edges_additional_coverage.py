# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi import HTTPException, UploadFile
from world_understanding.utils.credentials import InlineSecretError
from world_understanding.utils.held_file_response import open_held_artifact_file

from ...service import config_persistence
from ...service.artifact_contract import REFINE_ARTIFACT_SPECS
from ...service.routers import (
    pipeline_router,
    predict_router,
    refine_router,
    tune_router,
)
from ...service.runtime import get_event_bus
from ...service.runtime.events import ProgressEvent, StepState


def _assert_rejected_exception_graph_severed(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


def _assert_config_persistence_traceback_safe(
    error: BaseException,
    *,
    sentinel: str,
    forbidden_locals: set[str],
) -> None:
    """Verify public-error frames retain no rejected request values."""
    traceback = error.__traceback__
    owned_frames = []
    while traceback is not None:
        frame = traceback.tb_frame
        if Path(frame.f_code.co_filename) == Path(config_persistence.__file__):
            owned_frames.append(frame)
            assert sentinel not in repr(frame.f_locals)
            assert forbidden_locals.isdisjoint(frame.f_locals)
        traceback = traceback.tb_next
    assert owned_frames


class _Store:
    def __init__(self) -> None:
        self.events = [{"type": "stored"}]
        self.fail_events = False

    async def get_event_log(self, _session_id: str) -> list[dict]:
        if self.fail_events:
            raise RuntimeError("sentinel-event-store-secret")
        return self.events


class _Manager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata: dict[str, object] | None = {
            "session_id": "sid",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "completed",
            "results": {"predictions_made": 2, "best_score": float("inf")},
            "config": {},
            "completed_steps": [
                {
                    "name": "predict",
                    "display_name": "Predict",
                    "started_at": datetime.now(UTC).isoformat(),
                    "completed_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": 1,
                    "stats": {},
                }
            ],
            "overall_progress": {"current_step": 1, "total_steps": 1, "percent": 100},
            "duration_seconds": 3,
            "completed_at": "done",
        }
        self.storage_path = root
        self.exists = True
        self.cancel_requested = False
        self.cancellation_cleared: list[str] = []
        self.terminal_claim_cleared: list[str] = []
        self.terminal_claim: str | None = None
        self.deleted: list[str] = []
        self.sync_from_calls: list[str] = []
        self.sync_to_calls: list[tuple[str, str]] = []
        self.updated: list[dict] = []
        self.store = _Store()
        self.delete_result = True

    def get_session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def session_exists(self, _session_id: str) -> bool:
        return self.exists

    async def get_session_metadata(self, _session_id: str):
        return self.metadata

    async def create_session(self, session_id: str) -> Path:
        return self.get_session_dir(session_id)

    async def delete_session(self, session_id: str) -> bool:
        self.deleted.append(session_id)
        return self.delete_result

    async def sync_from_store(self, _session_id: str, *, prefix: str = "") -> int:
        self.sync_from_calls.append(prefix)
        return 0

    async def sync_to_store(self, session_id: str, prefix: str = "") -> int:
        self.sync_to_calls.append((session_id, prefix))
        return 0

    async def list_store_keys(self, session_id: str, prefix: str = "") -> list[str]:
        session_dir = self.get_session_dir(session_id)
        return sorted(
            path.relative_to(session_dir).as_posix()
            for path in session_dir.rglob("*")
            if path.is_file()
            and path.relative_to(session_dir).as_posix().startswith(prefix)
        )

    async def open_local_artifact_key(self, session_id: str, key: str):
        try:
            return open_held_artifact_file(self.root, f"{session_id}/{key}")
        except (OSError, RuntimeError, ValueError):
            return None

    async def update_session(self, _session_id: str, updates: dict) -> None:
        self.updated.append(updates)
        if self.metadata is not None:
            self.metadata.update(updates)

    async def request_cancellation(self, _session_id: str) -> None:
        self.cancel_requested = True

    async def request_pipeline_cancellation(self, _session_id: str) -> bool:
        self.cancel_requested = True
        if self.terminal_claim is None:
            self.terminal_claim = "cancelled"
        return self.terminal_claim == "cancelled"

    async def claim_pipeline_terminal_state(
        self,
        _session_id: str,
        status: str,
    ) -> str:
        if self.terminal_claim is None:
            self.terminal_claim = status
        return self.terminal_claim

    async def clear_cancellation(self, session_id: str) -> None:
        self.cancellation_cleared.append(session_id)

    async def clear_pipeline_terminal_claim(self, session_id: str) -> None:
        self.terminal_claim_cleared.append(session_id)
        self.terminal_claim = None


async def _response_body(response, *, method: str = "GET") -> bytes:
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await response(
        {
            "type": "http",
            "method": method,
            "path": "/artifact",
            "headers": [],
        },
        receive,
        send,
    )
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message.get("type") == "http.response.body"
    )


class _Registry:
    def __init__(
        self,
        *,
        running: bool = False,
        cancel_result: bool = True,
        running_after_cancel: bool | None = None,
    ) -> None:
        self.running = running
        self.cancel_result = cancel_result
        self.running_after_cancel = running_after_cancel
        self.cancelled: list[str] = []

    def is_running(self, _session_id: str) -> bool:
        return self.running

    async def cancel(self, session_id: str) -> bool:
        self.cancelled.append(session_id)
        if self.running_after_cancel is not None:
            self.running = self.running_after_cancel
        return self.cancel_result


class _ClosingReservation:
    def __init__(self, registry: _ClosingRegistry, session_id: str) -> None:
        self.registry = registry
        self.session_id = session_id

    async def __aenter__(self) -> _ClosingReservation:
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        return None

    async def start(self, coro) -> None:
        self.registry.started.append(self.session_id)
        if hasattr(coro, "close"):
            coro.close()


class _ClosingRegistry(_Registry):
    def __init__(self) -> None:
        super().__init__(running=False)
        self.registered: list[str] = []
        self.reserved: list[str] = []
        self.started: list[str] = []

    async def register(self, session_id: str, coro) -> None:
        self.registered.append(session_id)
        if hasattr(coro, "close"):
            coro.close()

    async def reserve(self, session_id: str) -> _ClosingReservation:
        self.reserved.append(session_id)
        return _ClosingReservation(self, session_id)


def _upload(name: str, data: bytes = b"data") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


def _pipeline_create_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "usd_file": None,
        "session_id": None,
        "s3_uri": None,
        "user_prompt": "",
        "render_backend": "",
        "optimize_usd": False,
        "enable_deinstance": True,
        "enable_split": False,
        "enable_deduplicate": False,
    }
    kwargs.update(overrides)
    return kwargs


def _predict_create_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "usd_file": None,
        "session_id": None,
        "s3_uri": None,
        "dataset_path": None,
        "user_prompt": "",
        "render_backend": "",
        "optimize_usd": False,
        "enable_deinstance": True,
        "enable_split": False,
        "enable_deduplicate": False,
    }
    kwargs.update(overrides)
    return kwargs


_REFINE_SCENARIO_YAML = """
name: drop_settle
metric: settle_distance
target:
  drop_height_m: 0.5
  duration_s: 2.0
  gravity: -9.81
parameters:
  - name: mass_scale
    min: 0.5
    max: 2.0
"""

_VALID_SOURCE_SESSION_ID = "00000000-0000-4000-8000-000000000000"


def test_pipeline_render_limit_and_s3_download_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.graphics.render_remote_async as remote

    monkeypatch.setattr(remote, "get_global_remote_render_limit", lambda: None)
    cfg = {"steps": {"build_dataset_usd": {"num_workers": 9}}}
    pipeline_router._apply_render_request_limit(cfg)
    assert cfg["steps"]["build_dataset_usd"]["num_workers"] == 9

    monkeypatch.setattr(remote, "get_global_remote_render_limit", lambda: 4)
    cfg = {
        "steps": {
            "build_dataset_usd": {"num_workers": "bad", "max_concurrent_requests": 9}
        }
    }
    pipeline_router._apply_render_request_limit(cfg)
    assert cfg["steps"]["build_dataset_usd"]["num_workers"] == 4
    assert cfg["steps"]["build_dataset_usd"]["max_concurrent_requests"] == 4
    cfg = {
        "steps": {
            "build_dataset_usd": {
                "num_workers": 2,
                "max_concurrent_requests": "bad",
            }
        }
    }
    pipeline_router._apply_render_request_limit(cfg)
    assert cfg["steps"]["build_dataset_usd"]["max_concurrent_requests"] == 4
    pipeline_router._apply_render_request_limit({"steps": {}})

    with pytest.raises(HTTPException, match="Invalid S3 URI"):
        pipeline_router._download_s3_to_session("bad", tmp_path)
    with pytest.raises(HTTPException, match="Invalid USD file type"):
        pipeline_router._download_s3_to_session("s3://bucket/", tmp_path)
    with pytest.raises(HTTPException, match="Invalid USD file type"):
        pipeline_router._download_s3_to_session("s3://bucket/file.txt", tmp_path)

    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")

    def write_file(_uri: str, path: Path) -> None:
        path.write_bytes(b"usd")

    monkeypatch.setattr(pipeline_router, "download_file_from_s3", write_file)
    assert pipeline_router._download_s3_to_session(
        "s3://bucket/scene.usda", tmp_path
    ).exists()
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as too_large:
        pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert too_large.value.status_code == 413

    for exc, status in (
        (FileNotFoundError(), 404),
        (PermissionError(), 403),
        (RuntimeError("network"), 502),
    ):

        def raise_exc(_uri: str, _path: Path, exc=exc) -> None:
            raise exc

        monkeypatch.setattr(pipeline_router, "download_file_from_s3", raise_exc)
        with pytest.raises(HTTPException) as err:
            pipeline_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
        assert err.value.status_code == status


def test_pipeline_config_write_rejects_inline_credentials_before_disk(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input" / "config.yaml"
    sentinel = "sentinel-physics-key"

    with pytest.raises(ValueError, match="steps.predict.vlm.api_key") as exc_info:
        config_persistence.write_pipeline_config(
            config_path,
            {"steps": {"predict": {"vlm": {"api_key": sentinel}}}},
        )

    assert not config_path.parent.exists()
    error_text = str(exc_info.value)
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in error_text


def test_pipeline_config_write_rejects_url_credentials_before_disk(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input" / "config.yaml"
    sentinel = "sentinel-url-password"
    url = f"https://user:{sentinel}@vlm.example.test/v1"

    with pytest.raises(ValueError, match="steps.predict.vlm.base_url") as exc_info:
        config_persistence.write_pipeline_config(
            config_path,
            {"steps": {"predict": {"vlm": {"base_url": url}}}},
        )

    assert not config_path.parent.exists()
    assert sentinel not in str(exc_info.value)


def test_pipeline_config_write_rejects_plural_credentials_before_disk(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input" / "config.yaml"
    sentinel = "sentinel-plural-physics-key"

    with pytest.raises(ValueError, match="steps.predict.vlm.api_keys") as exc_info:
        config_persistence.write_pipeline_config(
            config_path,
            {"steps": {"predict": {"vlm": {"api_keys": [sentinel]}}}},
        )

    assert not config_path.exists()
    assert sentinel not in str(exc_info.value)


def test_pipeline_config_write_preserves_existing_file_on_serialization_failure(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("existing: valid\n", encoding="utf-8")

    with pytest.raises(yaml.representer.RepresenterError):
        config_persistence.write_pipeline_config(
            config_path,
            {"steps": {}, "unsupported": object()},
        )

    assert config_path.read_text(encoding="utf-8") == "existing: valid\n"
    assert list(config_path.parent.glob(f".{config_path.name}.*.tmp")) == []


def test_durable_yaml_rejects_unhashable_mapping_keys() -> None:
    with pytest.raises(HTTPException) as exc_info:
        config_persistence.validate_durable_request_content(
            {},
            yaml_documents={"scenario": "? [unhashable]\n: value\n"},
            context="physics scenario",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_DURABLE_INPUT_DETAIL
    _assert_rejected_exception_graph_severed(exc_info.value)


def test_durable_content_rejection_severs_inline_secret_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "api_key=durable-content-error-713"
    caplog.set_level(logging.WARNING, logger=config_persistence.__name__)

    with pytest.raises(HTTPException) as exc_info:
        config_persistence.validate_durable_request_content(
            {"prompt": sentinel},
            context="physics durable request",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_DURABLE_INPUT_DETAIL
    assert sentinel not in repr(exc_info.value)
    assert "durable_input_rejected field_path=content.prompt" in caplog.text
    assert sentinel not in caplog.text
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"content", "yaml_documents", "documents", "context"},
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "pass Authorization: Bearer <your-token>",
        "See https://user@example.com/path for the public example",
        "https://example.com/object?sig=%3Csignature%3E",
    ],
)
def test_durable_content_accepts_explicit_documentation_placeholders(
    prompt: str,
) -> None:
    assert (
        config_persistence.validate_durable_request_content(
            {"prompt": prompt},
            context="physics durable request",
        )
        == {}
    )


def test_durable_raw_yaml_rejection_removes_traceback_locals() -> None:
    sentinel = "durable-yaml-error-713"

    with pytest.raises(HTTPException) as exc_info:
        config_persistence.validate_durable_request_content(
            {},
            yaml_documents={"scenario": f"api_key: {sentinel}\n"},
            context="physics durable request",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_DURABLE_INPUT_DETAIL
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"content", "yaml_documents", "documents", "context"},
    )


def test_durable_parsed_content_rejection_severs_inline_secret_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "api_key=durable-parsed-error-713"
    calls = 0

    def reject_parsed_content(_value: Any, *, context: str) -> None:
        nonlocal calls
        del context
        calls += 1
        if calls == 2:
            raise InlineSecretError(sentinel)

    monkeypatch.setattr(
        config_persistence,
        "ensure_no_inline_secrets",
        reject_parsed_content,
    )

    with pytest.raises(HTTPException) as exc_info:
        config_persistence.validate_durable_request_content(
            {},
            yaml_documents={"scenario": "safe: value\n"},
            context="physics durable request",
        )

    assert calls == 2
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_DURABLE_INPUT_DETAIL
    assert sentinel not in repr(exc_info.value)
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"content", "yaml_documents", "documents", "context"},
    )


@pytest.mark.parametrize("non_finite", [".nan", ".inf", "-.inf"])
def test_durable_yaml_rejects_non_finite_floats(non_finite: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        config_persistence.validate_durable_request_content(
            {},
            yaml_documents={"scenario": f"value: {non_finite}\n"},
            context="physics scenario",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_DURABLE_INPUT_DETAIL


def test_durable_yaml_preserves_finite_float() -> None:
    canonical = config_persistence.validate_durable_request_content(
        {},
        yaml_documents={"scenario": "value: 1.25\n"},
        context="physics scenario",
    )

    assert yaml.safe_load(canonical["scenario"]) == {"value": 1.25}


@pytest.mark.asyncio
async def test_pipeline_invalid_upload_extension_cleans_request_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _ClosingRegistry())

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(usd_file=_upload("scene.obj"))
        )

    assert exc_info.value.status_code == 400
    assert len(manager.deleted) == 1
    assert manager.updated == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route_name", ["pipeline", "predict"])
@pytest.mark.parametrize("session_created_here", [False, True])
async def test_config_rejection_cleans_only_request_owned_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
    session_created_here: bool,
) -> None:
    manager = _Manager(tmp_path)
    registry = _ClosingRegistry()
    sentinel = "sentinel-inline-physics-key"

    def unsafe_config(**_kwargs: Any) -> dict[str, Any]:
        return {"vlm": {"api_key": sentinel}, "steps": {}}

    invoke: Any
    if route_name == "pipeline":
        pipeline_router.set_session_manager(manager)
        monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
        monkeypatch.setattr(
            pipeline_router, "build_default_pipeline_config", unsafe_config
        )
        invoke = pipeline_router.create_pipeline
        kwargs = _pipeline_create_kwargs(
            usd_file=_upload("scene.usda") if session_created_here else None,
            session_id=None if session_created_here else "existing",
        )
        config_name = "config.yaml"
    else:
        predict_router.set_session_manager(manager)
        monkeypatch.setattr(predict_router, "get_job_registry", lambda: registry)
        monkeypatch.setattr(
            predict_router, "build_default_pipeline_config", unsafe_config
        )
        invoke = predict_router.create_predict
        kwargs = _predict_create_kwargs(
            usd_file=_upload("scene.usda") if session_created_here else None,
            session_id=None if session_created_here else "existing",
        )
        config_name = "predict_config.yaml"

    if not session_created_here:
        input_path = manager.get_session_dir("existing") / "input" / "scene.usda"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("#usda\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        await invoke(**kwargs)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_PIPELINE_CONFIG_DETAIL
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"config_factory", "pipeline_config"},
    )
    for secret_fragment in (sentinel, sentinel[:8], sentinel[-8:]):
        assert secret_fragment not in str(exc_info.value.detail)
    if session_created_here:
        assert len(manager.deleted) == 1
        rejected_session_id = manager.deleted[0]
    else:
        assert manager.deleted == []
        rejected_session_id = "existing"
    assert not (
        manager.get_session_dir(rejected_session_id) / "input" / config_name
    ).exists()
    assert manager.updated == []
    assert registry.registered == []
    assert len(registry.reserved) == (1 if route_name == "pipeline" else 0)
    assert registry.started == []


@pytest.mark.asyncio
@pytest.mark.parametrize("session_created_here", [False, True])
async def test_config_persistence_contains_unexpected_write_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_created_here: bool,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "sentinel-physics-write-error"

    def fail_write(_path: Path, _config: dict[str, Any]) -> None:
        raise OSError(sentinel)

    monkeypatch.setattr(config_persistence, "write_pipeline_config", fail_write)

    with pytest.raises(HTTPException) as exc_info:
        await config_persistence.build_and_write_pipeline_config(
            config_factory=lambda: {"steps": {}},
            config_path=tmp_path / "config.yaml",
            session_manager=manager,
            session_id="request-session",
            session_created_here=session_created_here,
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail == config_persistence.PIPELINE_CONFIG_WRITE_FAILED_DETAIL
    )
    assert sentinel not in str(exc_info.value.detail)
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"config_factory", "pipeline_config", "writer"},
    )
    assert manager.deleted == (["request-session"] if session_created_here else [])


@pytest.mark.asyncio
async def test_config_persistence_runs_scanner_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    event_loop_thread = threading.get_ident()
    scanner_threads: list[int] = []

    def record_guard(_config: dict[str, Any]) -> None:
        scanner_threads.append(threading.get_ident())

    monkeypatch.setattr(config_persistence, "validate_pipeline_config", record_guard)

    result = await config_persistence.build_and_validate_pipeline_config(
        config_factory=lambda: {"steps": {}},
        session_manager=manager,
        session_id="request-session",
        session_created_here=False,
    )

    assert result == {"steps": {}}
    assert scanner_threads
    assert scanner_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_config_persistence_contains_unexpected_factory_errors(
    tmp_path: Path,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "sentinel-physics-factory-error"

    def fail_build() -> dict[str, Any]:
        raise RuntimeError(sentinel)

    with pytest.raises(HTTPException) as exc_info:
        await config_persistence.build_and_validate_pipeline_config(
            config_factory=fail_build,
            session_manager=manager,
            session_id="request-session",
            session_created_here=True,
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail == config_persistence.PIPELINE_CONFIG_WRITE_FAILED_DETAIL
    )
    assert sentinel not in str(exc_info.value.detail)
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"config_factory", "pipeline_config"},
    )
    assert manager.deleted == ["request-session"]


@pytest.mark.asyncio
async def test_config_persistence_contains_unexpected_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "api_key=physics-validation-error-713"

    def fail_validation(_config: dict[str, Any]) -> None:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        config_persistence,
        "validate_pipeline_config",
        fail_validation,
    )

    with pytest.raises(HTTPException) as exc_info:
        await config_persistence.build_and_validate_pipeline_config(
            config_factory=lambda: {"steps": {}},
            session_manager=manager,
            session_id="request-session",
            session_created_here=True,
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail == config_persistence.PIPELINE_CONFIG_WRITE_FAILED_DETAIL
    )
    assert sentinel not in repr(exc_info.value)
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"config_factory", "pipeline_config"},
    )
    assert manager.deleted == ["request-session"]


@pytest.mark.asyncio
@pytest.mark.parametrize("session_created_here", [False, True])
async def test_predict_late_config_failure_deletes_owned_session_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_created_here: bool,
) -> None:
    manager = _Manager(tmp_path)

    def fail_write(_path: Path, _config: dict[str, Any]) -> None:
        raise OSError("sentinel-contained-late-write-error")

    monkeypatch.setattr(config_persistence, "write_pipeline_config", fail_write)

    with pytest.raises(HTTPException) as exc_info:
        await predict_router._persist_predict_inputs_transactionally(
            predict_config={"steps": {}},
            config_path=tmp_path / "session" / "input" / "predict_config.yaml",
            dataset_source=None,
            dataset_target=tmp_path / "session" / "cache" / "dataset.jsonl",
            manager=manager,
            session_id="request-session",
            session_created_here=session_created_here,
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail == config_persistence.PIPELINE_CONFIG_WRITE_FAILED_DETAIL
    )
    assert manager.deleted == (["request-session"] if session_created_here else [])


@pytest.mark.asyncio
async def test_config_persistence_revalidates_at_write_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "sentinel-physics-late-validation-error"

    def reject_write(_path: Path, _config: dict[str, Any]) -> None:
        raise InlineSecretError(sentinel)

    monkeypatch.setattr(config_persistence, "write_pipeline_config", reject_write)

    with pytest.raises(HTTPException) as exc_info:
        await config_persistence.build_and_write_pipeline_config(
            config_factory=lambda: {"steps": {}},
            config_path=tmp_path / "config.yaml",
            session_manager=manager,
            session_id="request-session",
            session_created_here=True,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_PIPELINE_CONFIG_DETAIL
    assert sentinel not in str(exc_info.value.detail)
    _assert_rejected_exception_graph_severed(exc_info.value)
    assert manager.deleted == ["request-session"]


@pytest.mark.asyncio
async def test_config_persistence_does_not_misclassify_writer_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    sentinel = "sentinel-unrelated-writer-value-error"

    def reject_write(_path: Path, _config: dict[str, Any]) -> None:
        raise ValueError(sentinel)

    monkeypatch.setattr(config_persistence, "write_pipeline_config", reject_write)

    with pytest.raises(HTTPException) as exc_info:
        await config_persistence.build_and_write_pipeline_config(
            config_factory=lambda: {"steps": {}},
            config_path=tmp_path / "config.yaml",
            session_manager=manager,
            session_id="request-session",
            session_created_here=True,
        )

    assert exc_info.value.status_code == 500
    assert (
        exc_info.value.detail == config_persistence.PIPELINE_CONFIG_WRITE_FAILED_DETAIL
    )
    assert sentinel not in str(exc_info.value.detail)
    _assert_rejected_exception_graph_severed(exc_info.value)
    assert manager.deleted == ["request-session"]


@pytest.mark.asyncio
async def test_config_persistence_reports_owned_session_cleanup_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    manager.delete_result = False
    sentinel = "sentinel-invalid-config"

    def invalid_config() -> dict[str, Any]:
        raise ValueError(sentinel)

    with caplog.at_level(logging.ERROR, logger=config_persistence.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await config_persistence.build_and_write_pipeline_config(
                config_factory=invalid_config,
                config_path=tmp_path / "config.yaml",
                session_manager=manager,
                session_id="request-session",
                session_created_here=True,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_PIPELINE_CONFIG_DETAIL
    _assert_rejected_exception_graph_severed(exc_info.value)
    _assert_config_persistence_traceback_safe(
        exc_info.value,
        sentinel=sentinel,
        forbidden_locals={"config_factory", "pipeline_config"},
    )
    assert manager.deleted == ["request-session"]
    assert "Failed to clean up rejected pipeline session" in caplog.text
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_pipeline_config_factory_failure_cleans_request_owned_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    registry = _ClosingRegistry()
    sentinel = "sentinel-invalid-render-backend"

    def reject_config_factory(**_kwargs: Any) -> dict[str, Any]:
        raise ValueError(sentinel)

    pipeline_router.set_session_manager(manager)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    monkeypatch.setattr(
        pipeline_router,
        "build_default_pipeline_config",
        reject_config_factory,
    )

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(
                usd_file=_upload("scene.usda"),
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == config_persistence.INVALID_PIPELINE_CONFIG_DETAIL
    assert sentinel not in str(exc_info.value.detail)
    assert len(manager.deleted) == 1
    assert manager.updated == []
    assert registry.registered == []


@pytest.mark.asyncio
async def test_predict_invalid_optimizer_flags_reject_before_session_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    registry = _ClosingRegistry()

    predict_router.set_session_manager(manager)
    monkeypatch.setattr(predict_router, "get_job_registry", lambda: registry)

    with pytest.raises(HTTPException) as exc_info:
        await predict_router.create_predict(
            **_predict_create_kwargs(
                usd_file=_upload("scene.usda"),
                optimize_usd=True,
                enable_deinstance=False,
                enable_split=False,
                enable_deduplicate=False,
            )
        )

    assert exc_info.value.status_code == 400
    assert "At least one optimization operation must be enabled" in str(
        exc_info.value.detail
    )
    assert manager.deleted == []
    assert list(tmp_path.iterdir()) == []
    assert manager.updated == []
    assert registry.reserved == []
    assert registry.started == []


@pytest.mark.asyncio
async def test_pipeline_stream_copy_status_and_terminal_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "input" / "scene.usda"
    assert await pipeline_router._stream_copy(_upload("scene.usda", b"usd"), dest) == 3

    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry(False))
    manager.metadata["created_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    status = await pipeline_router.get_pipeline_status("sid")
    assert status.elapsed_seconds >= 0

    bus = get_event_bus()
    bus.cleanup_session("terminal-pipeline")
    manager.exists = True
    manager.metadata = {"status": "completed"}
    assert await pipeline_router.stream_progress_events("terminal-pipeline")


@pytest.mark.asyncio
async def test_pipeline_status_results_cancel_events_and_event_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)
    registry = _Registry(running=False)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.seed_pending_session("sid")

    status = await pipeline_router.get_pipeline_status("sid")
    assert status.status == "completed"

    manager.metadata = None
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.get_pipeline_status("sid")
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.get_pipeline_results("sid")
    manager.metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "error": "boom",
        "failed_step": "predict",
        "completed_step_names": ["build_dataset_usd"],
        "partial_results": {"build_dataset_usd": {"num_prims": 3}},
        "completed_steps": [
            {
                "name": "render",
                "display_name": "Render",
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "duration_seconds": 1,
                "stats": {},
            }
        ],
    }
    failed_results = await pipeline_router.get_pipeline_results("sid")
    assert failed_results.status == "failed"
    assert failed_results.completed_steps == ["build_dataset_usd"]
    assert failed_results.partial_results == {"build_dataset_usd": {"num_prims": 3}}
    manager.metadata["completed_step_names"] = []
    assert (await pipeline_router.get_pipeline_results("sid")).completed_steps == []
    manager.metadata["status"] = "running"
    with pytest.raises(HTTPException) as pending:
        await pipeline_router.get_pipeline_results("sid")
    assert pending.value.status_code == 202

    manager.metadata["status"] = "cancelled"
    manager.metadata["completed_at"] = datetime.now(UTC).isoformat()
    cancelled_results = await pipeline_router.get_pipeline_results("sid")
    assert cancelled_results.status == "cancelled"

    manager.metadata["status"] = "completed"
    manager.metadata["results"] = {"x": 1}
    assert (await pipeline_router.get_pipeline_results("sid")).status == "completed"

    manager.metadata = None
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.cancel_pipeline("sid")
    manager.metadata = {"status": "completed"}
    with pytest.raises(HTTPException, match="Cannot cancel"):
        await pipeline_router.cancel_pipeline("sid")
    manager.metadata = {"status": "running"}
    registry.running = True
    response = await pipeline_router.cancel_pipeline("sid")
    assert response["status"] == "cancelling"
    assert manager.cancel_requested
    assert registry.cancelled == ["sid"]
    assert manager.metadata["status"] == "running"
    cancelling_snapshot = bus.get_snapshot("sid")
    assert cancelling_snapshot is not None
    assert cancelling_snapshot["status"] == "cancelling"
    cancelling_status = await pipeline_router.get_pipeline_status("sid")
    assert cancelling_status.status == "cancelling"
    assert cancelling_status.can_cancel is False

    bus.cleanup_session("cross")
    manager.exists = False
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.stream_progress_events("cross")
    manager.exists = True
    manager.metadata = {"status": "running"}
    with pytest.raises(HTTPException) as unavailable:
        await pipeline_router.stream_progress_events("cross")
    assert unavailable.value.status_code == 503

    manager.exists = False
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.get_event_log("sid")
    manager.exists = True
    assert await pipeline_router.get_event_log("sid") == {
        "events": [{"type": "stored"}],
        "total": 1,
    }
    manager.store.fail_events = True
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        assert await pipeline_router.get_event_log("sid") == {"events": []}
    assert "event_log_store_read_failed" in caplog.text
    assert "sentinel-event-store-secret" not in caplog.text
    log_file = manager.get_session_dir("sid") / "event_log.jsonl"
    log_file.write_text(json.dumps({"type": "local"}) + "\n", encoding="utf-8")
    assert (await pipeline_router.get_event_log("sid"))["total"] == 1
    local_sentinel = "sentinel-event-local-secret"
    log_file.write_text(f"{{{local_sentinel}\n", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as local_error:
            await pipeline_router.get_event_log("sid")
    assert local_error.value.status_code == 500
    assert local_error.value.detail == "Failed to load event log"
    assert "event_log_local_read_failed" in caplog.text
    assert local_sentinel not in str(local_error.value.detail)
    assert local_sentinel not in caplog.text


@pytest.mark.asyncio
async def test_pipeline_cancel_terminalizes_job_queued_before_executor_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    manager.metadata = {
        "session_id": "queued",
        "status": "pending",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    pipeline_router.set_session_manager(manager)
    registry = _Registry(
        running=True,
        cancel_result=True,
        running_after_cancel=False,
    )
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    bus = get_event_bus()
    bus.cleanup_session("queued")
    await bus.seed_pending_session("queued")

    response = await pipeline_router.cancel_pipeline("queued")

    assert response["status"] == "cancelling"
    assert registry.cancelled == ["queued"]
    assert manager.metadata["status"] == "cancelled"
    assert manager.metadata["can_cancel"] is False
    assert manager.metadata["completed_at"]
    snapshot = bus.get_snapshot("queued")
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"
    bus.cleanup_session("queued")


@pytest.mark.asyncio
async def test_regenerate_resets_prior_terminal_state_and_event_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "regen-reset"
    manager = _Manager(tmp_path)
    manager.metadata.update(
        {
            "status": "cancelled",
            "error": "old-error",
            "failed_step": "predict",
            "completed_step_names": ["predict"],
            "partial_results": {"predict": {"old": True}},
        }
    )
    pipeline_router.set_session_manager(manager)
    registry = _ClosingRegistry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    async def valid_config(**_kwargs) -> dict[str, Any]:
        return {"project": {"name": "regenerated"}}

    monkeypatch.setattr(
        pipeline_router,
        "build_and_validate_pipeline_config",
        valid_config,
    )
    config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n", encoding="utf-8")

    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
        )
    )
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.COMPLETED,
        )
    )
    assert bus.get_snapshot(session_id)["completed_steps"]

    response = await pipeline_router.regenerate_pipeline(
        session_id,
        pipeline_router.RegenerateRequest(steps=[]),
    )

    assert response.status == "pending"
    assert manager.cancellation_cleared == [session_id]
    assert manager.metadata["completed_steps"] == []
    assert manager.metadata["completed_step_names"] == []
    assert manager.metadata["partial_results"] is None
    assert manager.metadata["error"] is None
    assert manager.metadata["results"] == {}
    assert manager.metadata["duration_seconds"] == 0
    assert manager.metadata["cancelled_at"] is None
    snapshot = bus.get_snapshot(session_id)
    assert snapshot is not None
    assert snapshot["status"] == "pending"
    assert snapshot["completed_steps"] == []
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_pipeline_create_and_upload_validation_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)
    registry = _ClosingRegistry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    def write_download(_uri: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "scene.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", write_download)
    uploaded = await pipeline_router.upload_usd_immediate(
        usd_file=None,
        s3_uri="s3://bucket/scene.usda",
    )
    assert uploaded.status == "ready"
    assert manager.updated[-1]["config"]["original_filename"] == "scene.usda"

    queued = await pipeline_router.create_pipeline(
        **_pipeline_create_kwargs(s3_uri="s3://bucket/scene.usda")
    )
    assert queued.status == "pending"
    assert registry.started[-1] == queued.session_id

    with pytest.raises(HTTPException, match="Invalid USD file type"):
        await pipeline_router.upload_usd_immediate(
            usd_file=_upload("bad.txt"),
            s3_uri=None,
        )

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as too_large:
        await pipeline_router.upload_usd_immediate(
            usd_file=_upload("scene.usda", b"x"),
            s3_uri=None,
        )
    assert too_large.value.status_code == 413
    assert manager.deleted


@pytest.mark.asyncio
async def test_pipeline_reuse_reserves_before_mutating_live_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "already-running"
    manager = _Manager(tmp_path)
    manager.metadata = {
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "config": {"preserved": True},
    }
    input_path = manager.get_session_dir(session_id) / "input" / "scene.usda"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("#usda\n", encoding="utf-8")
    config_path = input_path.parent / "config.yaml"
    config_path.write_text("preserved: true\n", encoding="utf-8")

    registry = _ClosingRegistry()

    async def reject_reservation(_session_id: str):
        raise ValueError("Session already has a running job")

    monkeypatch.setattr(registry, "reserve", reject_reservation)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)
    pipeline_router.set_session_manager(manager)

    bus = get_event_bus()
    bus.cleanup_session(session_id)
    await bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="predict",
            state=StepState.RUNNING,
        )
    )

    with pytest.raises(HTTPException) as conflict:
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(session_id=session_id)
        )

    assert conflict.value.status_code == 409
    assert manager.updated == []
    assert manager.cancellation_cleared == []
    assert manager.terminal_claim_cleared == []
    assert config_path.read_text(encoding="utf-8") == "preserved: true\n"
    assert bus.get_snapshot(session_id)["status"] == "running"
    bus.cleanup_session(session_id)


@pytest.mark.asyncio
async def test_pipeline_create_cleanup_and_store_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "s3_allowed_buckets", "bucket")
    manager = _Manager(tmp_path)
    pipeline_router.set_session_manager(manager)
    registry = _ClosingRegistry()
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    def fail_http(_uri: str, _session_dir: Path) -> Path:
        raise HTTPException(status_code=413, detail="too large")

    monkeypatch.setattr(pipeline_router, "_download_s3_to_session", fail_http)
    with pytest.raises(HTTPException) as s3_err:
        await pipeline_router.upload_usd_immediate(
            usd_file=None,
            s3_uri="s3://bucket/scene.usda",
        )
    assert s3_err.value.status_code == 413
    assert manager.deleted

    with pytest.raises(HTTPException) as pipeline_s3_err:
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(s3_uri="s3://bucket/scene.usda")
        )
    assert pipeline_s3_err.value.status_code == 413

    manager.exists = False
    with pytest.raises(HTTPException, match="Session not found"):
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(session_id="missing")
        )

    manager.exists = True
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as upload_too_large:
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(usd_file=_upload("scene.usda", b"x"))
        )
    assert upload_too_large.value.status_code == 413

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1024)

    async def sync_without_input(_session_id: str, *, prefix: str = "") -> int:
        manager.sync_from_calls.append(prefix)
        return 1

    manager.sync_from_store = sync_without_input
    with pytest.raises(HTTPException, match="Input USD not found"):
        await pipeline_router.create_pipeline(
            **_pipeline_create_kwargs(session_id="existing")
        )
    assert manager.sync_from_calls[-1] == "input/"

    async def sync_with_input(_session_id: str, *, prefix: str = "") -> int:
        manager.sync_from_calls.append(prefix)
        dest = manager.get_session_dir(_session_id) / "input" / "scene.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return 1

    manager.sync_from_store = sync_with_input
    response = await pipeline_router.create_pipeline(
        **_pipeline_create_kwargs(session_id="existing")
    )
    assert response.session_id == "existing"
    assert registry.started[-1] == "existing"
    assert manager.cancellation_cleared[-1] == "existing"
    assert manager.metadata["results"] == {}
    assert manager.metadata["duration_seconds"] == 0

    manager.metadata = {"status": "completed"}
    with pytest.raises(HTTPException, match="Original config not found"):
        await pipeline_router.regenerate_pipeline(
            "no-config",
            pipeline_router.RegenerateRequest(steps=[]),
        )

    config_path = manager.get_session_dir("no-config") / "input" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "steps": {
                    "build_dataset_prepare_dataset": {
                        "prompts": {"user": "safe prompt"}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registered_before = list(registry.registered)
    with pytest.raises(HTTPException) as inline_secret:
        await pipeline_router.regenerate_pipeline(
            "no-config",
            pipeline_router.RegenerateRequest(
                steps=[],
                user_prompt="api_key=" + "A" * 32,
            ),
        )
    assert inline_secret.value.status_code == 400
    assert inline_secret.value.detail == "Pipeline configuration is invalid"
    assert registry.registered == registered_before


def test_predict_helper_validation_and_s3_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HTTPException, match="absolute"):
        predict_router._resolve_dataset_path_safely("relative.jsonl", manager)
    with pytest.raises(HTTPException, match="does not exist"):
        predict_router._resolve_dataset_path_safely(
            str(tmp_path / "missing.jsonl"), manager
        )
    wrong = tmp_path / "not_dataset.txt"
    wrong.write_text("x", encoding="utf-8")
    with pytest.raises(HTTPException, match="dataset.jsonl"):
        predict_router._resolve_dataset_path_safely(str(wrong), manager)
    with pytest.raises(HTTPException, match="regular file"):
        predict_router._resolve_dataset_path_safely(str(tmp_path), manager)
    outside = tmp_path / "outside" / "dataset.jsonl"
    outside.parent.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    manager.storage_path = tmp_path / "sessions"
    monkeypatch.setenv("PA_DATASET_ALLOWED_ROOTS", "")
    with pytest.raises(HTTPException) as forbidden:
        predict_router._resolve_dataset_path_safely(str(outside), manager)
    assert forbidden.value.status_code == 403
    manager.storage_path = tmp_path
    assert predict_router._resolve_dataset_path_safely(str(dataset), manager) == dataset

    bad_root = tmp_path / "bad-root"
    manager.storage_path = bad_root
    monkeypatch.setenv("PA_DATASET_ALLOWED_ROOTS", str(tmp_path))
    real_resolve = Path.resolve

    def flaky_resolve(self: Path, *args, **kwargs):
        if self == bad_root:
            raise OSError("bad root")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", flaky_resolve)
    assert predict_router._resolve_dataset_path_safely(str(dataset), manager) == dataset
    monkeypatch.setattr(Path, "resolve", real_resolve)

    manager.storage_path = tmp_path / "other-root"
    monkeypatch.setenv(
        "PA_DATASET_ALLOWED_ROOTS", f"{tmp_path / 'other-root'}:{tmp_path}"
    )
    real_commonpath = predict_router.os.path.commonpath
    calls = {"count": 0}

    def flaky_commonpath(paths):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("different drives")
        return real_commonpath(paths)

    monkeypatch.setattr(predict_router.os.path, "commonpath", flaky_commonpath)
    assert predict_router._resolve_dataset_path_safely(str(dataset), manager) == dataset

    import world_understanding.utils.s3_utils as s3_utils

    def raise_http(_uri: str):
        raise HTTPException(status_code=418, detail="teapot")

    monkeypatch.setattr(s3_utils, "_parse_s3_path", raise_http)
    with pytest.raises(HTTPException) as teapot:
        predict_router._preflight_s3_object_size("s3://bucket/key.usd", 10)
    assert teapot.value.status_code == 418

    monkeypatch.setattr(s3_utils, "_parse_s3_path", lambda _uri: ("bucket", "key"))
    monkeypatch.setattr(
        s3_utils,
        "_create_s3_client",
        lambda: SimpleNamespace(head_object=lambda **_: {"ContentLength": 11}),
    )
    with pytest.raises(HTTPException) as too_large:
        predict_router._preflight_s3_object_size("s3://bucket/key.usd", 10)
    assert too_large.value.status_code == 413
    monkeypatch.setattr(
        s3_utils,
        "_create_s3_client",
        lambda: SimpleNamespace(head_object=lambda **_: {"ContentLength": None}),
    )
    predict_router._preflight_s3_object_size("s3://bucket/key.usd", 10)
    monkeypatch.setattr(
        s3_utils,
        "_create_s3_client",
        lambda: (_ for _ in ()).throw(RuntimeError("skip")),
    )
    predict_router._preflight_s3_object_size("s3://bucket/key.usd", 10)


@pytest.mark.asyncio
async def test_predict_create_validation_download_and_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predict_router.config, "s3_allowed_buckets", "bucket")
    manager = _Manager(tmp_path)
    predict_router.set_session_manager(manager)
    registry = _ClosingRegistry()
    monkeypatch.setattr(predict_router, "get_job_registry", lambda: registry)

    manager.exists = False
    with pytest.raises(HTTPException, match="Session not found"):
        await predict_router.create_predict(
            **_predict_create_kwargs(session_id="missing")
        )

    manager.exists = True
    manager.metadata = {"status": "running"}
    with pytest.raises(HTTPException) as conflict:
        await predict_router.create_predict(**_predict_create_kwargs(session_id="busy"))
    assert conflict.value.status_code == 409

    manager.metadata = {"status": "completed", "config": {}}

    async def sync_without_input(_session_id: str, *, prefix: str = "") -> int:
        manager.sync_from_calls.append(prefix)
        return 1 if prefix == "input/" else 0

    manager.sync_from_store = sync_without_input
    with pytest.raises(HTTPException, match="No input USD"):
        await predict_router.create_predict(
            **_predict_create_kwargs(session_id="no-input")
        )
    assert manager.sync_from_calls[-2:] == ["cache/dataset/", "input/"]

    async def sync_stale_dataset_then_input(
        session_id: str, *, prefix: str = ""
    ) -> int:
        manager.sync_from_calls.append(prefix)
        session_dir = manager.get_session_dir(session_id)
        if prefix == "cache/dataset/":
            dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
            dataset.parent.mkdir(parents=True, exist_ok=True)
            dataset.write_text(
                json.dumps({"images": {"prim_only": "missing.png"}}) + "\n",
                encoding="utf-8",
            )
        elif prefix == "input/":
            usd = session_dir / "input" / "scene.usda"
            usd.parent.mkdir(parents=True, exist_ok=True)
            usd.write_text("#usda\n", encoding="utf-8")
        return 1

    manager.sync_from_store = sync_stale_dataset_then_input
    fallback = await predict_router.create_predict(
        **_predict_create_kwargs(session_id="stale-remote-dataset")
    )
    assert fallback.status == "pending"
    assert manager.sync_from_calls[-2:] == ["cache/dataset/", "input/"]
    manager.metadata = {"status": "completed", "config": {}}

    input_path = manager.get_session_dir("invalid-options") / "input" / "scene.usda"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("#usda\n", encoding="utf-8")
    with pytest.raises(HTTPException, match="At least one optimization operation"):
        await predict_router.create_predict(
            **_predict_create_kwargs(
                session_id="invalid-options",
                optimize_usd=True,
                enable_deinstance=False,
                enable_split=False,
                enable_deduplicate=False,
            )
        )

    def write_download(_uri: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "scene.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    persisted: dict[str, object] = {}

    def write_config(config_path: Path, pipeline_config: dict[str, Any]) -> None:
        assert config_path.parents[1].name in registry.reserved
        persisted["path"] = config_path
        persisted["config"] = pipeline_config

    monkeypatch.setattr(predict_router, "_download_s3_to_session", write_download)
    monkeypatch.setattr(config_persistence, "write_pipeline_config", write_config)
    response = await predict_router.create_predict(
        **_predict_create_kwargs(s3_uri="s3://bucket/scene.usda")
    )
    assert response.status == "pending"
    assert registry.started[-1] == response.session_id
    assert persisted["path"] == (
        manager.get_session_dir(response.session_id) / "input" / "predict_config.yaml"
    )
    assert isinstance(persisted["config"], dict)


@pytest.mark.asyncio
async def test_predict_stream_copy_oversize_cleans_partial_file(tmp_path: Path) -> None:
    dest = tmp_path / "upload" / "scene.usda"
    with pytest.raises(HTTPException) as too_large:
        await predict_router._stream_copy(
            _upload("scene.usda", b"abcdef"),
            dest,
            max_bytes=4,
            chunk_size=3,
        )
    assert too_large.value.status_code == 413
    assert not dest.exists()


@pytest.mark.asyncio
async def test_tune_upload_reference_and_source_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "upload" / "big.bin"
    with pytest.raises(HTTPException) as too_large:
        await tune_router._stream_copy(
            _upload("big.bin", b"abcdef"),
            dest,
            chunk_size=3,
            max_bytes=4,
            too_large_detail="too big",
        )
    assert too_large.value.status_code == 413
    assert not dest.exists()

    assert tune_router._parse_reference_descriptions("", "refs") is None
    assert tune_router._nonempty_uploads(None) == []
    with pytest.raises(HTTPException) as huge_descriptions:
        tune_router._parse_reference_descriptions("[" + ("x" * 20_000) + "]", "refs")
    assert huge_descriptions.value.status_code == 413
    with pytest.raises(HTTPException, match="JSON array"):
        tune_router._parse_reference_descriptions("{bad", "refs")
    with pytest.raises(HTTPException, match="JSON array"):
        tune_router._parse_reference_descriptions(json.dumps({"not": "a list"}), "refs")
    with pytest.raises(HTTPException) as huge_description:
        tune_router._parse_reference_descriptions(json.dumps(["x" * 3000]), "refs")
    assert huge_description.value.status_code == 413

    with pytest.raises(HTTPException, match="Invalid reference image"):
        await tune_router._copy_reference_uploads(
            uploads=[_upload("bad.txt")],
            session_dir=tmp_path,
            subdir="refs",
            file_prefix="ref",
            valid_extensions={".png"},
            label="reference image",
        )
    with pytest.raises(HTTPException, match="batch too large"):
        await tune_router._copy_reference_uploads(
            uploads=[_upload("a.png")],
            session_dir=tmp_path,
            subdir="refs",
            file_prefix="ref",
            valid_extensions={".png"},
            label="reference image",
            current_batch_bytes=2,
            max_batch_bytes=1,
        )
    copied, batch_bytes = await tune_router._copy_reference_uploads(
        uploads=[_upload("ok.png", b"abc")],
        session_dir=tmp_path,
        subdir="refs",
        file_prefix="ref",
        valid_extensions={".png"},
        label="reference image",
    )
    assert copied[0].name == "ref_01.png"
    assert batch_bytes == 3

    with pytest.raises(HTTPException, match="Invalid S3 URI"):
        tune_router._download_s3_to_session("not-s3", tmp_path)
    with pytest.raises(HTTPException, match="Invalid USD file type"):
        tune_router._download_s3_to_session("s3://bucket/file.txt", tmp_path)

    monkeypatch.setattr(tune_router.config, "s3_allowed_buckets", "bucket")

    def write_file(_uri: str, path: Path) -> None:
        path.write_bytes(b"usd")

    monkeypatch.setattr(tune_router, "download_file_from_s3", write_file)
    assert tune_router._download_s3_to_session(
        "s3://bucket/physics.usda", tmp_path
    ).exists()

    for exc, status in (
        (FileNotFoundError(), 404),
        (PermissionError(), 403),
        (RuntimeError("network"), 502),
    ):

        def raise_exc(_uri: str, _path: Path, exc=exc) -> None:
            raise exc

        monkeypatch.setattr(tune_router, "download_file_from_s3", raise_exc)
        with pytest.raises(HTTPException) as err:
            tune_router._download_s3_to_session("s3://bucket/physics.usda", tmp_path)
        assert err.value.status_code == status

    monkeypatch.setattr(tune_router.config, "max_upload_size_mb", 0)
    monkeypatch.setattr(tune_router, "download_file_from_s3", write_file)
    with pytest.raises(HTTPException) as s3_too_large:
        tune_router._download_s3_to_session("s3://bucket/physics.usda", tmp_path)
    assert s3_too_large.value.status_code == 413

    class SourceManager:
        def __init__(self, exists: bool, artifact: Path | None) -> None:
            self.exists = exists
            self.artifact = artifact
            self.synced: list[str] = []

        async def session_exists(self, _session_id: str) -> bool:
            return self.exists

        async def get_artifact_path(
            self, _session_id: str, _artifact_type: str
        ) -> Path | None:
            return self.artifact

        async def sync_from_store(self, _session_id: str, *, prefix: str = "") -> int:
            self.synced.append(prefix)
            return 0

    with pytest.raises(HTTPException, match="not found"):
        await tune_router._copy_from_source_session(
            SourceManager(False, None), "src", tmp_path
        )
    with pytest.raises(HTTPException, match="no apply_physics"):
        await tune_router._copy_from_source_session(
            SourceManager(True, None), "src", tmp_path
        )
    src = tmp_path / "source.usda"
    src.write_text("#usda\n", encoding="utf-8")
    copied_path = await tune_router._copy_from_source_session(
        SourceManager(True, src),
        "src",
        tmp_path / "target",
    )
    assert copied_path.read_text(encoding="utf-8") == "#usda\n"

    assert tune_router._find_input_physics(tmp_path / "none") is None
    (tmp_path / "target" / "input" / "physics.usd").write_text("usd", encoding="utf-8")
    assert tune_router._find_input_physics(tmp_path / "target").suffix == ".usd"
    assert (
        tune_router._scenario_param_names_from_mapping({"parameters": "bad"}) == set()
    )
    assert tune_router._scenario_param_names_from_mapping(
        {"parameters": ["bad", {"name": "mass_scale"}, {"name": 3}]}
    ) == {"mass_scale"}
    tune_router._validate_engine_supports_param_names_for_request("fake", set())
    with pytest.raises(HTTPException, match="Unknown engine"):
        tune_router._validate_engine_name_for_request("missing")


async def _call_create_refine(**overrides: Any):
    kwargs: dict[str, Any] = {
        "physics_usd": _upload("physics.usda", b"#usda\n"),
        "s3_uri": None,
        "source_session_id": None,
        "reference_images": [],
        "reference_videos": [],
        "reference_descriptions": "",
        "reference_video_descriptions": "",
        "reference_video_frames": 8,
        "judge_reference_frames": 8,
        "judge_generated_frames": 16,
        "scenario_yaml": _REFINE_SCENARIO_YAML,
        "user_prompt": "make it settle",
        "optimizer": "botorch",
        "engine": "fake",
        "max_trials": 30,
        "max_iterations": 5,
        "score_threshold": 0.9,
        "seed": 42,
        "judge_max_tokens": None,
        "judge_temperature": None,
        "visual_evidence_enabled": True,
        "llm_timeout_seconds": 180.0,
    }
    kwargs.update(overrides)
    return await refine_router.create_refine(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_trials": 0}, "max_trials"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 13}, "max_iterations"),
        ({"score_threshold": float("inf")}, "score_threshold"),
        ({"judge_max_tokens": 0}, "judge_max_tokens"),
        ({"judge_temperature": -0.1}, "judge_temperature"),
        ({"llm_timeout_seconds": float("nan")}, "llm_timeout_seconds"),
        ({"reference_video_frames": 0}, "reference_video_frames"),
        ({"judge_reference_frames": 65}, "judge_reference_frames"),
        ({"judge_generated_frames": 0}, "judge_generated_frames"),
        (
            {"scenario_yaml": "name: drop_settle\n" + ("# filler\n" * 9000)},
            "scenario_yaml",
        ),
        ({"user_prompt": "x" * (17 * 1024)}, "user_prompt"),
        (
            {"reference_images": [_upload(f"ref{i}.png") for i in range(17)]},
            "Too many reference media",
        ),
    ],
)
async def test_refine_create_rejects_scalar_and_payload_limits(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(HTTPException, match=match):
        await _call_create_refine(**overrides)


@pytest.mark.asyncio
async def test_refine_create_rejects_reference_description_mismatches() -> None:
    with pytest.raises(HTTPException, match="reference_descriptions"):
        await _call_create_refine(
            reference_images=[_upload("ref.png")],
            reference_descriptions=json.dumps(["one", "two"]),
        )
    with pytest.raises(HTTPException, match="reference_video_descriptions"):
        await _call_create_refine(
            reference_videos=[_upload("ref.mp4")],
            reference_video_descriptions=json.dumps([]),
        )


@pytest.mark.asyncio
async def test_refine_create_input_cleanup_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(tune_router.config, "s3_allowed_buckets", "bucket")
    manager = _Manager(tmp_path)
    refine_router.set_session_manager(manager)
    monkeypatch.setattr(refine_router, "get_job_registry", lambda: _ClosingRegistry())
    monkeypatch.setattr(
        refine_router, "_validate_scenario_yaml_for_refine", lambda *_: None
    )

    with pytest.raises(HTTPException, match="Invalid USD file type"):
        await _call_create_refine(physics_usd=_upload("physics.txt"))
    assert manager.deleted

    with pytest.raises(HTTPException, match="physics_usd is empty"):
        await _call_create_refine(physics_usd=_upload("physics.usda", b""))

    def write_download(_uri: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "physics.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(refine_router, "_download_s3_to_session", write_download)
    s3_response = await _call_create_refine(
        physics_usd=None,
        s3_uri="s3://bucket/physics.usda",
    )
    assert s3_response.status == "pending"

    sentinel = "sentinel-refine-backend-secret"

    def fail_download(_uri: str, _session_dir: Path) -> Path:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(refine_router, "_download_s3_to_session", fail_download)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=refine_router.__name__):
        with pytest.raises(HTTPException) as provision_error:
            await _call_create_refine(
                physics_usd=None,
                s3_uri="s3://bucket/physics.usda",
            )
    assert provision_error.value.status_code == 500
    assert provision_error.value.detail == "Failed to provision input physics USD"
    assert "refine_input_provision_failed" in caplog.text
    assert sentinel not in str(provision_error.value.detail)
    assert sentinel not in caplog.text

    async def copy_source(_manager, _source_session_id: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "physics.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(refine_router, "_copy_from_source_session", copy_source)
    source_response = await _call_create_refine(
        physics_usd=None,
        source_session_id=_VALID_SOURCE_SESSION_ID,
    )
    assert source_response.status == "pending"

    def no_download(_uri: str, _session_dir: Path) -> Path:
        return _session_dir / "input" / "missing.usda"

    monkeypatch.setattr(refine_router, "_download_s3_to_session", no_download)
    with pytest.raises(HTTPException, match="Failed to provision input"):
        await _call_create_refine(
            physics_usd=None,
            s3_uri="s3://bucket/missing.usda",
        )

    async def fail_reference_copy(**_kwargs):
        raise HTTPException(status_code=400, detail="bad reference")

    monkeypatch.setattr(refine_router, "_copy_reference_uploads", fail_reference_copy)
    with pytest.raises(HTTPException, match="bad reference"):
        await _call_create_refine(reference_images=[_upload("ref.png")])


async def _call_create_tune(**overrides: Any):
    kwargs: dict[str, Any] = {
        "physics_usd": _upload("physics.usda", b"#usda\n"),
        "s3_uri": None,
        "source_session_id": None,
        "reference_images": [],
        "reference_videos": [],
        "reference_descriptions": "",
        "reference_video_descriptions": "",
        "reference_video_frames": 8,
        "judge_reference_frames": 8,
        "judge_generated_frames": 16,
        "scenario_yaml": "",
        "user_prompt": "make it bouncy",
        "optimizer": "auto",
        "engine": "fake",
        "max_trials": 30,
        "seed": 42,
        "enable_judge": True,
        "judge_max_iterations": 3,
        "judge_max_tokens": None,
        "judge_temperature": None,
    }
    kwargs.update(overrides)
    return await tune_router.create_tune(**kwargs)


@pytest.mark.asyncio
async def test_tune_create_validation_and_input_cleanup_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(tune_router.config, "s3_allowed_buckets", "bucket")
    with pytest.raises(HTTPException, match="reference_descriptions"):
        await _call_create_tune(
            reference_images=[_upload("ref.png")],
            reference_descriptions=json.dumps(["one", "two"]),
        )
    with pytest.raises(HTTPException, match="reference_video_descriptions"):
        await _call_create_tune(
            reference_videos=[_upload("ref.mp4")],
            reference_video_descriptions=json.dumps([]),
        )
    with pytest.raises(HTTPException, match="parse to a mapping"):
        await _call_create_tune(scenario_yaml="- item\n", user_prompt="")
    with pytest.raises(HTTPException, match="reference_video_frames"):
        await _call_create_tune(reference_video_frames=0)
    with pytest.raises(HTTPException, match="judge_reference_frames"):
        await _call_create_tune(judge_reference_frames=65)
    with pytest.raises(HTTPException, match="judge_generated_frames"):
        await _call_create_tune(judge_generated_frames=0)

    manager = _Manager(tmp_path)
    tune_router.set_session_manager(manager)
    monkeypatch.setattr(tune_router, "get_job_registry", lambda: _ClosingRegistry())

    monkeypatch.setattr(tune_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as too_large:
        await _call_create_tune(physics_usd=_upload("physics.usda", b"x"))
    assert too_large.value.status_code == 413

    monkeypatch.setattr(tune_router.config, "max_upload_size_mb", 1024)

    def write_download(_uri: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "physics.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(tune_router, "_download_s3_to_session", write_download)
    s3_response = await _call_create_tune(
        physics_usd=None,
        s3_uri="s3://bucket/physics.usda",
    )
    assert s3_response.status == "pending"

    sentinel = "sentinel-tune-backend-secret"

    def fail_download(_uri: str, _session_dir: Path) -> Path:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(tune_router, "_download_s3_to_session", fail_download)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=tune_router.__name__):
        with pytest.raises(HTTPException) as provision_error:
            await _call_create_tune(
                physics_usd=None,
                s3_uri="s3://bucket/physics.usda",
            )
    assert provision_error.value.status_code == 500
    assert provision_error.value.detail == "Failed to provision input physics USD"
    assert "tune_input_provision_failed" in caplog.text
    assert "phase=local_publication" in caplog.text
    assert sentinel not in str(provision_error.value.detail)
    assert sentinel not in caplog.text

    async def copy_source(_manager, _source_session_id: str, session_dir: Path) -> Path:
        dest = session_dir / "input" / "physics.usda"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("#usda\n", encoding="utf-8")
        return dest

    monkeypatch.setattr(tune_router, "_copy_from_source_session", copy_source)
    source_response = await _call_create_tune(
        physics_usd=None,
        source_session_id=_VALID_SOURCE_SESSION_ID,
    )
    assert source_response.status == "pending"

    def no_download(_uri: str, _session_dir: Path) -> Path:
        return _session_dir / "input" / "missing.usda"

    monkeypatch.setattr(tune_router, "_download_s3_to_session", no_download)
    with pytest.raises(HTTPException, match="Failed to provision input"):
        await _call_create_tune(
            physics_usd=None,
            s3_uri="s3://bucket/missing.usda",
        )

    async def fail_reference_copy(**_kwargs):
        raise HTTPException(status_code=400, detail="bad reference")

    monkeypatch.setattr(tune_router, "_copy_reference_uploads", fail_reference_copy)
    with pytest.raises(HTTPException, match="bad reference"):
        await _call_create_tune(reference_images=[_upload("ref.png")])


def test_refine_validation_helpers(tmp_path: Path) -> None:
    valid_uuid = "00000000-0000-4000-8000-000000000000"
    refine_router._validate_optimizer_name_for_request("botorch")
    with pytest.raises(HTTPException, match="Unknown optimizer"):
        refine_router._validate_optimizer_name_for_request("nope")
    assert refine_router._coerce_finite_score(None) is None
    assert refine_router._coerce_finite_score("bad") is None
    assert refine_router._coerce_finite_score("0.5") == 0.5
    assert refine_router._first_present(None, "x") == "x"
    assert any(
        spec.logical_name == "final_report" and spec.key == "refine/final/report.md"
        for spec in REFINE_ARTIFACT_SPECS
    )
    refine_router._validate_source_session_id("")
    refine_router._validate_source_session_id(valid_uuid)
    with pytest.raises(HTTPException, match="source_session_id"):
        refine_router._validate_source_session_id("not-a-uuid")
    refine_router._validate_route_session_id(valid_uuid)
    with pytest.raises(HTTPException, match="session_id"):
        refine_router._validate_route_session_id("not-a-uuid")

    tune_router._validate_ovphysx_runtime_for_request("fake")

    with pytest.raises(HTTPException, match="Invalid scenario YAML"):
        refine_router._validate_scenario_yaml_for_refine("[", "fake")
    with pytest.raises(HTTPException, match="mapping"):
        refine_router._validate_scenario_yaml_for_refine("- item", "fake")
    with pytest.raises(HTTPException, match="Invalid scenario"):
        refine_router._validate_scenario_yaml_for_refine("name: missing\n", "fake")
    assert (
        refine_router._metadata_elapsed_seconds(
            {"created_at": datetime.now(UTC).replace(tzinfo=None).isoformat()}
        )
        >= 0
    )


@pytest.mark.asyncio
async def test_ovphysx_request_preflight_uses_shared_runtime_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tune_router, "ovphysx_runtime_available", lambda: False)
    with pytest.raises(HTTPException) as unavailable:
        tune_router._validate_ovphysx_runtime_for_request("ovphysx")
    assert unavailable.value.status_code == 503
    with pytest.raises(HTTPException) as tune_unavailable:
        await _call_create_tune(engine="ovphysx")
    assert tune_unavailable.value.status_code == 503
    with pytest.raises(HTTPException) as refine_unavailable:
        await _call_create_refine(engine="ovphysx")
    assert refine_unavailable.value.status_code == 503

    monkeypatch.setattr(tune_router, "ovphysx_runtime_available", lambda: True)
    tune_router._validate_ovphysx_runtime_for_request("ovphysx")


def test_predict_download_and_find_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HTTPException, match="Invalid S3 URI"):
        predict_router._download_s3_to_session("bad", tmp_path)
    with pytest.raises(HTTPException, match="Invalid USD file type"):
        predict_router._download_s3_to_session("s3://bucket/", tmp_path)
    with pytest.raises(HTTPException, match="Invalid USD file type"):
        predict_router._download_s3_to_session("s3://bucket/file.txt", tmp_path)

    monkeypatch.setattr(predict_router.config, "s3_allowed_buckets", "bucket")
    monkeypatch.setattr(
        predict_router, "_preflight_s3_object_size", lambda *_args: None
    )

    def write_file(_uri: str, path: Path) -> None:
        path.write_bytes(b"usd")

    monkeypatch.setattr(predict_router, "download_file_from_s3", write_file)
    assert predict_router._download_s3_to_session(
        "s3://bucket/scene.usda", tmp_path
    ).exists()
    assert predict_router._find_input_usd(tmp_path).suffix == ".usda"

    for exc, status in (
        (FileNotFoundError(), 404),
        (PermissionError(), 403),
        (RuntimeError("network"), 502),
    ):

        def raise_exc(_uri: str, _path: Path, exc=exc) -> None:
            raise exc

        monkeypatch.setattr(predict_router, "download_file_from_s3", raise_exc)
        with pytest.raises(HTTPException) as err:
            predict_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
        assert err.value.status_code == status

    monkeypatch.setattr(predict_router.config, "max_upload_size_mb", 0)
    monkeypatch.setattr(predict_router, "download_file_from_s3", write_file)
    with pytest.raises(HTTPException) as too_large:
        predict_router._download_s3_to_session("s3://bucket/scene.usda", tmp_path)
    assert too_large.value.status_code == 413


@pytest.mark.asyncio
async def test_predict_status_results_cancel_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _Manager(tmp_path)
    predict_router.set_session_manager(manager)
    registry = _Registry(running=True, cancel_result=True)
    monkeypatch.setattr(predict_router, "get_job_registry", lambda: registry)
    get_event_bus().cleanup_session("sid")

    manager.metadata["preview_images"] = ["a.png"]
    status = await predict_router.get_predict_status("sid")
    assert status.preview_images == ["/artifacts/sid/preview/a.png"]

    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(
            session_id="sid",
            step="predict",
            state=StepState.RUNNING,
            percent=25,
        )
    )
    status = await predict_router.get_predict_status("sid")
    assert status.status == "running"
    bus.cleanup_session("sid")

    manager.metadata["created_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    status = await predict_router.get_predict_status("sid")
    assert status.elapsed_seconds >= 0

    manager.metadata = None
    with pytest.raises(HTTPException):
        await predict_router.get_predict_status("sid")
    with pytest.raises(HTTPException):
        await predict_router.get_predict_results("sid")
    manager.metadata = {"status": "failed", "error": "bad", "completed_steps": []}
    assert (await predict_router.get_predict_results("sid")).status == "failed"
    manager.metadata = {"status": "running"}
    with pytest.raises(HTTPException) as running:
        await predict_router.get_predict_results("sid")
    assert running.value.status_code == 202
    manager.metadata = {
        "status": "completed",
        "results": {"predictions_count": 5},
        "predict_mode": "dataset_only",
        "predict_steps_run": ["predict"],
    }
    restore_sentinel = "sentinel-predict-store-secret"

    async def fail_dataset_restore(_session_id: str, *, prefix: str = "") -> int:
        raise RuntimeError(restore_sentinel)

    manager.sync_from_store = fail_dataset_restore
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=predict_router.__name__):
        assert (await predict_router.get_predict_results("sid")).predictions_count == 5
    assert "predict_dataset_restore_failed" in caplog.text
    assert restore_sentinel not in caplog.text

    manager.metadata = None
    with pytest.raises(HTTPException):
        await predict_router.cancel_predict("sid")
    manager.metadata = {"status": "running", "config": {"predict_route": False}}
    with pytest.raises(HTTPException) as wrong_kind:
        await predict_router.cancel_predict("sid")
    assert wrong_kind.value.status_code == 409
    manager.metadata = {"status": "completed", "config": {"predict_route": True}}
    with pytest.raises(HTTPException, match="Cannot cancel"):
        await predict_router.cancel_predict("sid")
    manager.metadata = {"status": "running", "config": {"predict_route": True}}
    assert (await predict_router.cancel_predict("sid"))["status"] == "cancelling"

    bus = get_event_bus()
    bus.cleanup_session("missing")
    manager.exists = False
    with pytest.raises(HTTPException):
        await predict_router.stream_predict_events("missing")
    manager.exists = True
    manager.metadata = {"status": "running"}
    with pytest.raises(HTTPException) as unavailable:
        await predict_router.stream_predict_events("missing")
    assert unavailable.value.status_code == 503
    manager.metadata = {"status": "completed"}
    assert await predict_router.stream_predict_events("missing")


@pytest.mark.asyncio
async def test_tune_refine_status_results_cancel_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _Manager(tmp_path)
    tune_router.set_session_manager(manager)
    refine_router.set_session_manager(manager)

    manager.metadata = None
    with pytest.raises(HTTPException):
        await tune_router.get_tune_status("sid")
    with pytest.raises(HTTPException):
        await tune_router.get_tune_results("sid")
    manager.metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "results": {"best_score": float("inf"), "best_params": {}, "n_trials": 0},
        "config": {"max_trials": 2, "kind": "tune"},
    }
    assert (await tune_router.get_tune_status("sid")).best_score is None
    assert tune_router._coerce_finite_score("bad") is None

    manager.metadata["created_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    bus = get_event_bus()
    bus.cleanup_session("sid")
    await bus.emit(
        ProgressEvent(
            session_id="sid",
            step="tune",
            state=StepState.RUNNING,
            current=1,
            total=2,
        )
    )
    status = await tune_router.get_tune_status("sid")
    assert status.n_trials == 1
    assert status.max_trials == 2
    bus.cleanup_session("sid")

    assert (await tune_router.get_tune_results("sid")).best_score is None
    manager.metadata["status"] = "cancelled"
    assert (await tune_router.get_tune_results("sid")).status == "cancelled"
    manager.metadata["status"] = "failed"
    manager.metadata["error"] = "judge backend unavailable"
    assert (
        await tune_router.get_tune_status("sid")
    ).error_message == "judge backend unavailable"
    assert (await tune_router.get_tune_results("sid")).status == "failed"
    manager.metadata["results"] = {}
    assert (await tune_router.get_tune_results("sid")).status == "failed"
    manager.metadata["status"] = "running"
    with pytest.raises(HTTPException) as still_running:
        await tune_router.get_tune_results("sid")
    assert still_running.value.status_code == 202

    manager.metadata = {"status": "running", "config": {"kind": "pipeline"}}
    with pytest.raises(HTTPException) as wrong:
        await tune_router.cancel_tune("sid")
    assert wrong.value.status_code == 409
    manager.metadata = {"status": "completed", "kind": "tune", "config": {}}
    with pytest.raises(HTTPException, match="Cannot cancel"):
        await tune_router.cancel_tune("sid")
    manager.metadata = {"status": "running", "kind": "tune", "config": {}}
    assert (await tune_router.cancel_tune("sid"))["status"] == "cancelling"

    bus.cleanup_session("tune-cross")
    manager.exists = True
    manager.metadata = {"status": "running"}
    with pytest.raises(HTTPException) as tune_unavailable:
        await tune_router.stream_tune_events("tune-cross")
    assert tune_unavailable.value.status_code == 503

    with pytest.raises(HTTPException, match="Unknown artifact"):
        await tune_router.download_tune_artifact("sid", "bad.txt")
    tune_file = manager.get_session_dir("sid") / "tune" / "report.md"
    tune_file.parent.mkdir(parents=True)
    tune_file.write_text("report", encoding="utf-8")
    assert await tune_router.download_tune_artifact("sid", "report.md")
    tune_usd = tune_file.parent / "tuned_physics.usd"
    tune_usd.write_text("#usda 1.0\n", encoding="utf-8")
    tune_legacy_response = await tune_router.download_tune_artifact(
        "sid", "tuned_physics.usda"
    )
    tune_held_path = tune_usd.with_name("held-tuned-physics.usd")
    tune_usd.rename(tune_held_path)
    tune_outside = tmp_path / "outside-tune-held.usd"
    tune_outside.write_text("sentinel-tune-outside", encoding="utf-8")
    tune_usd.symlink_to(tune_outside)
    assert await _response_body(tune_legacy_response) == b"#usda 1.0\n"
    assert tune_outside.read_text(encoding="utf-8") == "sentinel-tune-outside"
    tune_file.unlink()
    with pytest.raises(HTTPException, match="Artifact not available"):
        await tune_router.download_tune_artifact("sid", "report.md")
    original_list_store_keys = manager.list_store_keys

    async def tune_store_only(_session_id: str, prefix: str = "") -> list[str]:
        return ["tune/report.md"] if prefix == "tune/" else []

    monkeypatch.setattr(manager, "list_store_keys", tune_store_only)
    with pytest.raises(HTTPException, match="Artifact not available"):
        await tune_router.download_tune_artifact("sid", "report.md")
    monkeypatch.setattr(manager, "list_store_keys", original_list_store_keys)

    outside_tune_file = tmp_path / "outside-tune-report.md"
    outside_tune_file.write_text("outside", encoding="utf-8")
    tune_file.symlink_to(outside_tune_file)
    with pytest.raises(HTTPException, match="Artifact not available"):
        await tune_router.download_tune_artifact("sid", "report.md")
    tune_file.unlink()

    valid_uuid = "00000000-0000-0000-0000-000000000000"
    manager.metadata = None
    with pytest.raises(HTTPException):
        await refine_router.get_refine_status(valid_uuid)
    with pytest.raises(HTTPException):
        await refine_router.get_refine_results(valid_uuid)
    with pytest.raises(HTTPException):
        await refine_router.cancel_refine(valid_uuid)
    with pytest.raises(HTTPException):
        await refine_router.stream_refine_events(valid_uuid)
    with pytest.raises(HTTPException):
        await refine_router.download_refine_artifact(valid_uuid, "final/report.md")
    manager.metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "kind": "refine",
        "config": {"max_trials": 2, "max_iterations": 3},
        "results": {
            "termination_reason": "approved",
            "iteration_count": 1,
            "final_iteration": 1,
            "final_judge_score": float("inf"),
            "iterations": [{"iteration": 1, "judge_score": 0.7, "best_score": 0.1}],
        },
    }
    assert (await refine_router.get_refine_status(valid_uuid)).judge_score == 0.7
    assert (
        await refine_router.get_refine_results(valid_uuid)
    ).final_judge_score is None
    manager.metadata["status"] = "failed"
    manager.metadata["error"] = "refine backend unavailable"
    assert (
        await refine_router.get_refine_status(valid_uuid)
    ).error_message == "refine backend unavailable"
    assert (await refine_router.get_refine_results(valid_uuid)).status == "failed"
    manager.metadata["results"] = {}
    assert (await refine_router.get_refine_results(valid_uuid)).status == "failed"
    manager.metadata["status"] = "running"
    with pytest.raises(HTTPException) as refine_running:
        await refine_router.get_refine_results(valid_uuid)
    assert refine_running.value.status_code == 202
    manager.metadata = {"status": "completed", "kind": "pipeline", "config": {}}
    with pytest.raises(HTTPException) as wrong_refine:
        await refine_router.cancel_refine(valid_uuid)
    assert wrong_refine.value.status_code == 409
    manager.metadata = {"status": "completed", "kind": "refine", "config": {}}
    with pytest.raises(HTTPException, match="Cannot cancel"):
        await refine_router.cancel_refine(valid_uuid)
    manager.metadata = {"status": "running", "kind": "refine", "config": {}}
    assert (await refine_router.cancel_refine(valid_uuid))["status"] == "cancelling"

    bus.cleanup_session(valid_uuid)
    manager.metadata = {"status": "running", "kind": "refine", "config": {}}
    with pytest.raises(HTTPException) as refine_unavailable:
        await refine_router.stream_refine_events(valid_uuid)
    assert refine_unavailable.value.status_code == 503
    manager.metadata = {"status": "completed", "kind": "refine", "config": {}}
    assert await refine_router.stream_refine_events(valid_uuid)

    with pytest.raises(HTTPException, match="Unknown artifact"):
        await refine_router.download_refine_artifact(valid_uuid, "bad.txt")
    refine_file = manager.get_session_dir(valid_uuid) / "refine" / "final" / "report.md"
    refine_file.parent.mkdir(parents=True)
    refine_file.write_text("report", encoding="utf-8")
    assert await refine_router.download_refine_artifact(valid_uuid, "final/report.md")
    refine_usd = refine_file.parent / "tuned_physics.usd"
    refine_usd.write_text("#usda 1.0\n", encoding="utf-8")
    refine_legacy_response = await refine_router.download_refine_artifact(
        valid_uuid, "final/tuned_physics.usda"
    )
    refine_held_path = refine_usd.with_name("held-tuned-physics.usd")
    refine_usd.rename(refine_held_path)
    refine_outside = tmp_path / "outside-refine-held.usd"
    refine_outside.write_text("sentinel-refine-outside", encoding="utf-8")
    refine_usd.symlink_to(refine_outside)
    assert await _response_body(refine_legacy_response) == b"#usda 1.0\n"
    assert refine_outside.read_text(encoding="utf-8") == "sentinel-refine-outside"
    camera_video = refine_file.parent / "render" / "Camera_A__render.mp4"
    camera_video.parent.mkdir()
    camera_video.write_bytes(b"video")
    with pytest.raises(HTTPException, match="Unknown artifact"):
        await refine_router.download_refine_artifact(
            valid_uuid, "final/render/Camera_A__render.mp4"
        )
    refine_file.unlink()
    with pytest.raises(HTTPException, match="Artifact not available"):
        await refine_router.download_refine_artifact(valid_uuid, "final/report.md")
    original_list_store_keys = manager.list_store_keys

    async def refine_store_only(_session_id: str, prefix: str = "") -> list[str]:
        return ["refine/final/report.md"] if prefix == "refine/" else []

    monkeypatch.setattr(manager, "list_store_keys", refine_store_only)
    with pytest.raises(HTTPException, match="Artifact not available"):
        await refine_router.download_refine_artifact(valid_uuid, "final/report.md")
    monkeypatch.setattr(manager, "list_store_keys", original_list_store_keys)

    outside_refine_file = tmp_path / "outside-refine-report.md"
    outside_refine_file.write_text("outside", encoding="utf-8")
    refine_file.symlink_to(outside_refine_file)
    with pytest.raises(HTTPException, match="Artifact not available"):
        await refine_router.download_refine_artifact(valid_uuid, "final/report.md")
