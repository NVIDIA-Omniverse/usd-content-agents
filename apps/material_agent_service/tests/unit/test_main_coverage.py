# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for material-agent-service main module helpers and lifespan."""

from __future__ import annotations

import asyncio
import logging
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from ...service import main as main_module
from ...service.runtime import registry as registry_module


def test_public_response_sanitizer_is_outermost() -> None:
    middleware = main_module.app.user_middleware[0]

    assert middleware.cls is PublicJsonResponseSanitizationMiddleware
    assert middleware.kwargs["session_roots"] == (
        main_module.config.session_storage_path,
    )


class _FakeConfig:
    service_name = "Material Agent Service"
    service_version = "1.2.3"
    description = "desc"
    session_storage_path = "/tmp/material-agent-sessions"
    session_ttl_hours = 1
    cleanup_enabled = False
    cleanup_interval_hours = 0.001
    cleanup_max_age_hours = 2.0
    has_required_api_keys = True
    image_gen_ready = True
    vlm_backend = "echo"
    vlm_model = "vlm"
    vlm_temperature = 0.0
    llm_backend = "echo"
    materials_library_path = "/materials.usd"


class _FakeSessionManager:
    instances: list[_FakeSessionManager] = []

    def __init__(
        self, *, storage_path: str, ttl_hours: int, store: object = None
    ) -> None:
        self.storage_path = storage_path
        self.ttl_hours = ttl_hours
        self.store = store
        self.cleanup_cache_calls = 0
        self.cleanup_expired_calls = 0
        _FakeSessionManager.instances.append(self)

    async def cleanup_stale_local_cache(self, *, max_age_hours: float) -> int:
        self.cleanup_cache_calls += 1
        return 1

    async def cleanup_expired_sessions(self) -> int:
        self.cleanup_expired_calls += 1
        return 2


class _FakeEventBus:
    def __init__(self) -> None:
        self.manager = None

    def set_session_manager(self, manager: object) -> None:
        self.manager = manager


class _FakeCleanupTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def __await__(self):
        async def wait():
            raise asyncio.CancelledError

        return wait().__await__()


@pytest.fixture
def main_harness(monkeypatch: pytest.MonkeyPatch):
    config = _FakeConfig()
    event_bus = _FakeEventBus()
    router_managers: list[object] = []
    _FakeSessionManager.instances.clear()

    monkeypatch.setattr(main_module, "config", config)
    monkeypatch.setattr(main_module, "SessionManager", _FakeSessionManager)
    monkeypatch.setattr(main_module, "get_event_bus", lambda: event_bus)
    for router in (
        main_module.pipeline_router,
        main_module.artifacts_router,
        main_module.assets_router,
        main_module.sessions_router,
    ):
        monkeypatch.setattr(
            router,
            "set_session_manager",
            lambda manager, _router=router: router_managers.append(manager),
        )

    return types.SimpleNamespace(
        config=config,
        event_bus=event_bus,
        router_managers=router_managers,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, 3), ("0", 0), ("5", 5), ("-1", 3), ("many", 3)],
)
@pytest.mark.asyncio
async def test_health_capacity_matches_enforced_registry(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: int,
) -> None:
    if env_value is None:
        monkeypatch.delenv("MA_MAX_ACTIVE_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("MA_MAX_ACTIVE_SESSIONS", env_value)
    monkeypatch.setattr(registry_module, "_job_registry", None)

    registry = registry_module.get_job_registry()
    health = await main_module.health_check()

    assert registry.max_concurrent == expected
    assert health["max_active_sessions"] == expected

    monkeypatch.setenv("MA_MAX_ACTIVE_SESSIONS", "7")
    assert (await main_module.health_check())["max_active_sessions"] == expected


@pytest.mark.unit
def test_load_aws_config_file_into_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = types.SimpleNamespace(
        infos=[],
        warnings=[],
        exceptions=[],
        info=lambda *args: log.infos.append(args),
        warning=lambda *args: log.warnings.append(args),
        exception=lambda *args: log.exceptions.append(args),
    )

    monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
    main_module._load_aws_config_file_into_env(log=log)

    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing.env"))
    main_module._load_aws_config_file_into_env(log=log)
    assert log.warnings

    config_file = tmp_path / "aws.env"
    config_file.write_text(
        "AWS_ACCESS_KEY_ID=from-file\nAWS_SECRET_ACCESS_KEY=secret\nregion=us-west-2\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "already-set")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    main_module._load_aws_config_file_into_env(log=log)

    assert main_module.os.environ["AWS_ACCESS_KEY_ID"] == "already-set"
    assert main_module.os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert main_module.os.environ["AWS_DEFAULT_REGION"] == "us-west-2"
    assert main_module.os.environ["AWS_REGION"] == "us-west-2"

    monkeypatch.setattr(
        main_module,
        "dotenv_values",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad env")),
    )
    main_module._load_aws_config_file_into_env(log=log)
    assert log.exceptions

    empty_config = tmp_path / "empty.env"
    empty_config.write_text("AWS_ACCESS_KEY_ID=\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty_config))
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr(
        main_module, "dotenv_values", lambda _path: {"AWS_ACCESS_KEY_ID": ""}
    )
    main_module._load_aws_config_file_into_env(log=log)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_periodic_cleanup_task_success_error_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _FakeSessionManager(storage_path="/tmp", ttl_hours=1)
    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            return None
        raise asyncio.CancelledError

    monkeypatch.setattr(main_module.asyncio, "sleep", fake_sleep)

    await main_module._periodic_cleanup_task(manager, 0.001, 3.0)
    assert manager.cleanup_cache_calls == 1
    assert manager.cleanup_expired_calls == 1

    class FailingManager(_FakeSessionManager):
        async def cleanup_stale_local_cache(self, *, max_age_hours: float) -> int:
            raise RuntimeError("cleanup-storage-secret-727")

    failing_manager = FailingManager(storage_path="/tmp", ttl_hours=1)
    sleep_calls = 0
    with caplog.at_level(logging.ERROR):
        await main_module._periodic_cleanup_task(failing_manager, 0.001, 3.0)
    assert sleep_calls == 2
    assert "cleanup-storage-secret-727" not in caplog.text
    assert "code=periodic_session_cleanup_failed" in caplog.text
    assert "phase=rollback" in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_local_store_cleanup_disabled(
    monkeypatch: pytest.MonkeyPatch,
    main_harness,
) -> None:
    telemetry_config = types.SimpleNamespace(
        enabled=False,
        service_name="svc",
        exporters=[],
    )
    monkeypatch.setattr(main_module, "TelemetryConfig", lambda: telemetry_config)
    monkeypatch.setattr(main_module, "initialize_telemetry", lambda _config: None)
    shutdowns: list[bool] = []
    monkeypatch.setattr(
        main_module, "shutdown_telemetry", lambda: shutdowns.append(True)
    )
    monkeypatch.setattr(
        main_module,
        "StorageConfig",
        lambda: types.SimpleNamespace(kind="local", local_root="/tmp/local"),
    )
    monkeypatch.setattr(main_module, "OTEL_INSTRUMENTATION_AVAILABLE", False)
    monkeypatch.setattr(
        main_module,
        "get_base_url",
        lambda *_args: (_ for _ in ()).throw(ValueError("missing")),
    )

    async with main_module.lifespan(main_module.app):
        assert _FakeSessionManager.instances[-1].store is None
        assert main_harness.event_bus.manager is _FakeSessionManager.instances[-1]
        assert len(main_harness.router_managers) == 4

    assert shutdowns == [True]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_warns_when_enabled_telemetry_fails(
    monkeypatch: pytest.MonkeyPatch,
    main_harness,
) -> None:
    monkeypatch.setattr(
        main_module,
        "TelemetryConfig",
        lambda: types.SimpleNamespace(enabled=True, service_name="svc", exporters=[]),
    )
    monkeypatch.setattr(main_module, "initialize_telemetry", lambda _config: None)
    monkeypatch.setattr(main_module, "shutdown_telemetry", lambda: None)
    monkeypatch.setattr(
        main_module,
        "StorageConfig",
        lambda: types.SimpleNamespace(kind="local", local_root="/tmp/local"),
    )
    monkeypatch.setattr(main_module, "OTEL_INSTRUMENTATION_AVAILABLE", False)
    monkeypatch.setattr(
        main_module,
        "get_base_url",
        lambda *_args: (_ for _ in ()).throw(ValueError("missing")),
    )

    async with main_module.lifespan(main_module.app):
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_s3_store_cleanup_and_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
    main_harness,
) -> None:
    main_harness.config.cleanup_enabled = True
    main_harness.config.has_required_api_keys = False
    cleanup_task = _FakeCleanupTask()
    created_tasks: list[object] = []

    def fake_create_task(coro):
        coro.close()
        created_tasks.append(coro)
        return cleanup_task

    class _RequestsInstrumentor:
        def instrument(self) -> None:
            created_tasks.append("instrumented")

    storage_cfg = types.SimpleNamespace(
        kind="s3",
        s3_bucket="bucket",
        s3_prefix="prefix",
        s3_endpoint_url="http://minio",
    )
    monkeypatch.setattr(
        main_module,
        "TelemetryConfig",
        lambda: types.SimpleNamespace(
            enabled=True, service_name="svc", exporters=["otlp"]
        ),
    )
    monkeypatch.setattr(main_module, "initialize_telemetry", lambda _config: object())
    monkeypatch.setattr(main_module, "shutdown_telemetry", lambda: None)
    monkeypatch.setattr(main_module, "StorageConfig", lambda: storage_cfg)
    monkeypatch.setattr(
        main_module.S3SessionStore,
        "from_config",
        lambda cfg: ("s3-store", cfg),
    )
    monkeypatch.setattr(main_module.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(main_module, "OTEL_INSTRUMENTATION_AVAILABLE", True)
    monkeypatch.setattr(
        main_module,
        "RequestsInstrumentor",
        _RequestsInstrumentor,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "get_base_url",
        lambda _default, endpoint_env, function_env: (
            "https://render.invocation.api.nvcf.nvidia.com"
            if endpoint_env == "RENDER_ENDPOINT"
            else "https://optimizer.invocation.api.nvcf.nvidia.com"
        ),
    )

    async with main_module.lifespan(main_module.app):
        assert _FakeSessionManager.instances[-1].store == ("s3-store", storage_cfg)
        assert "instrumented" in created_tasks

    assert cleanup_task.cancelled is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_main_endpoints_and_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ENABLE_COSMOS_VLM", "true")
    monkeypatch.setenv("MA_MAX_ACTIVE_SESSIONS", "4")
    monkeypatch.setattr(registry_module, "_job_registry", None)

    health = await main_module.health_check()
    assert health["status"] == "healthy"
    assert health["max_active_sessions"] == 4

    assert any(
        model["value"] == "nim/nvidia/cosmos-reason2-8b"
        for model in (await main_module.get_vlm_models())["models"]
    )
    assert "pipeline" in (await main_module.root_api_info())["api"]
    invalid_response = await main_module._invalid_session_id_handler(
        types.SimpleNamespace(),
        main_module.InvalidSessionIdError("bad/session"),
    )
    assert invalid_response.status_code == 400
    contended_response = await main_module._session_metadata_contention_handler(
        types.SimpleNamespace(),
        main_module.SessionMetadataContentionError("internal session details"),
    )
    assert contended_response.status_code == 503
    assert contended_response.headers["retry-after"] == "1"
    assert b"internal session details" not in contended_response.body

    fake_root = tmp_path / "app"
    service_dir = fake_root / "service"
    index_file = fake_root / "web" / "dist" / "index.html"
    index_file.parent.mkdir(parents=True)
    index_file.write_text("<html></html>")
    manual_file = fake_root / "manual.html"
    third_party_licenses_file = fake_root / "3rd_party_licenses.html"
    license_body_file = fake_root / "license_body.html"
    for artifact in (manual_file, third_party_licenses_file, license_body_file):
        artifact.write_text("<html></html>")
    service_dir.mkdir(parents=True)

    class _FakePath:
        def __init__(self, value: object) -> None:
            if value == main_module.__file__:
                self._path = service_dir / "main.py"
            else:
                self._path = Path(value)

        @property
        def parent(self):
            return _FakePath(self._path.parent)

        def __truediv__(self, other: str):
            return _FakePath(self._path / other)

        def exists(self) -> bool:
            return self._path.exists()

        def __fspath__(self) -> str:
            return str(self._path)

    monkeypatch.setattr(main_module, "Path", _FakePath)
    assert isinstance(await main_module.serve_index(), FileResponse)
    assert isinstance(await main_module.serve_manual(), FileResponse)
    assert isinstance(await main_module.serve_third_party_licenses(), FileResponse)
    assert isinstance(await main_module.serve_license_body(), FileResponse)
    for artifact in (
        index_file,
        manual_file,
        third_party_licenses_file,
        license_body_file,
    ):
        artifact.unlink()
    assert (await main_module.serve_index())[
        "service"
    ] == main_module.config.service_name
    assert await main_module.serve_manual() == {"error": "User manual not found"}
    assert await main_module.serve_third_party_licenses() == {
        "error": "Third-party licenses file not found"
    }
    with pytest.raises(HTTPException):
        await main_module.serve_license_body()

    run_calls: list[dict[str, Any]] = []
    fake_uvicorn = types.SimpleNamespace(
        run=lambda *args, **kwargs: run_calls.append(kwargs)
    )
    monkeypatch.setitem(main_module.sys.modules, "uvicorn", fake_uvicorn)
    main_module.main()
    assert run_calls == [
        {"host": "0.0.0.0", "port": 8000, "reload": False, "log_level": "info"}
    ]
