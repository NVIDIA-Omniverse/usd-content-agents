# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from ...service import main as service_main
from ...service.runtime import registry as registry_module
from ...service.session.manager import InvalidSessionIdError
from ...service.storage.local_store import LocalSessionStore


def test_public_response_sanitizer_is_outermost() -> None:
    middleware = service_main.app.user_middleware[0]

    assert middleware.cls is PublicJsonResponseSanitizationMiddleware
    assert middleware.kwargs["session_roots"] == (
        service_main.config.session_storage_path,
    )


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, 1), ("0", 0), ("5", 5), ("-1", 1), ("bad", 1)],
)
@pytest.mark.asyncio
async def test_health_capacity_matches_enforced_registry(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: int,
) -> None:
    if env_value is None:
        monkeypatch.delenv("PA_MAX_ACTIVE_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("PA_MAX_ACTIVE_SESSIONS", env_value)
    monkeypatch.setattr(registry_module, "_job_registry", None)

    registry = registry_module.get_job_registry()
    health = await service_main.health_check()

    assert registry.max_concurrent == expected
    assert health["max_active_sessions"] == expected

    monkeypatch.setenv("PA_MAX_ACTIVE_SESSIONS", "7")
    assert (await service_main.health_check())["max_active_sessions"] == expected


def test_is_tuning_extra_available_uses_optimizer_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.tuning import optimizers

    def available() -> bool:
        return True

    def unavailable() -> bool:
        return False

    monkeypatch.setattr(optimizers, "is_botorch_available", available)
    assert service_main._is_tuning_extra_available() is True

    monkeypatch.setattr(optimizers, "is_botorch_available", unavailable)
    assert service_main._is_tuning_extra_available() is False


def test_is_tuning_extra_available_handles_missing_optimizer_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "physics_agent.tuning.optimizers":
            raise ImportError("missing tuning optimizer module")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert service_main._is_tuning_extra_available() is False


def test_is_ovphysx_runtime_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_main, "ovphysx_runtime_available", lambda: True)
    assert service_main._is_ovphysx_runtime_available() is True


def test_load_aws_config_file_into_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("aws-test")
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
    service_main._load_aws_config_file_into_env(log=log)

    missing = tmp_path / "missing.env"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(missing))
    service_main._load_aws_config_file_into_env(log=log)
    assert "file does not exist" in caplog.text

    config_file = tmp_path / "aws.env"
    config_file.write_text(
        "aws_access_key_id=AKIA\naws_secret_access_key=SECRET\nregion=us-west-2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    service_main._load_aws_config_file_into_env(log=log)
    assert service_main.os.environ["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert service_main.os.environ["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    assert service_main.os.environ["AWS_DEFAULT_REGION"] == "us-west-2"
    assert service_main.os.environ["AWS_REGION"] == "us-west-2"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "EXISTING")
    service_main._load_aws_config_file_into_env(log=log)
    assert service_main.os.environ["AWS_ACCESS_KEY_ID"] == "EXISTING"

    empty_config = tmp_path / "empty.env"
    empty_config.write_text("region=\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty_config))
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    service_main._load_aws_config_file_into_env(log=log)
    assert "AWS_DEFAULT_REGION" not in service_main.os.environ


def test_load_aws_config_file_handles_read_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "aws.env"
    config_file.write_text("AWS_ACCESS_KEY_ID=AKIA\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))

    def fail_dotenv(_path):
        raise RuntimeError("bad file")

    monkeypatch.setattr(service_main, "dotenv_values", fail_dotenv, raising=False)
    # The helper imports dotenv_values inside the function, so patch the module too.
    import dotenv

    monkeypatch.setattr(dotenv, "dotenv_values", fail_dotenv)
    service_main._load_aws_config_file_into_env(log=logging.getLogger("aws-fail"))


class _S3ishStore(LocalSessionStore):
    @property
    def kind(self) -> str:
        return "s3"


class _FakeConfig:
    service_name = "Physics Agent Service"
    service_version = "test"
    vlm_backend = "mock"
    vlm_model = "mock-model"
    vlm_temperature = 0.0
    session_ttl_hours = 1
    cleanup_interval_hours = 1.0
    cleanup_max_age_hours = 24.0
    cleanup_enabled = False
    storage_s3_bucket = "bucket"
    storage_s3_prefix = "prefix"

    def __init__(self, storage_path: str, *, api_keys: bool, s3: bool = False) -> None:
        self.session_storage_path = storage_path
        self.has_required_api_keys = api_keys
        self._s3 = s3

    def build_session_store(self):
        if self._s3:
            return _S3ishStore(self.session_storage_path)
        return LocalSessionStore(self.session_storage_path)


@pytest.mark.asyncio
async def test_periodic_cleanup_task_runs_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CleanupManager:
        stale_max_age = None
        cleaned_cache = 0
        expired_sessions = 0

        async def cleanup_stale_local_cache(self, max_age_hours: float) -> int:
            self.stale_max_age = max_age_hours
            self.cleaned_cache += 1
            return 1

        async def cleanup_expired_sessions(self) -> int:
            self.expired_sessions += 1
            return 2

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise service_main.asyncio.CancelledError()

    monkeypatch.setattr(service_main.asyncio, "sleep", fake_sleep)
    manager = CleanupManager()

    await service_main._periodic_cleanup_task(
        manager=manager,
        interval_hours=0.25,
        max_age_hours=6.0,
    )

    assert manager.stale_max_age == 6.0
    assert manager.cleaned_cache == 1
    assert manager.expired_sessions == 1


@pytest.mark.asyncio
async def test_periodic_cleanup_task_logs_and_continues_after_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingCleanupManager:
        calls = 0

        async def cleanup_stale_local_cache(self, max_age_hours: float) -> int:
            self.calls += 1
            raise RuntimeError(f"cleanup-storage-secret-727 {max_age_hours}")

        async def cleanup_expired_sessions(self) -> int:
            raise AssertionError("expired cleanup should not run after cache error")

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise service_main.asyncio.CancelledError()

    monkeypatch.setattr(service_main.asyncio, "sleep", fake_sleep)
    manager = FailingCleanupManager()

    with caplog.at_level(logging.ERROR):
        await service_main._periodic_cleanup_task(
            manager=manager,
            interval_hours=0.25,
            max_age_hours=6.0,
        )

    assert manager.calls == 1
    assert "cleanup-storage-secret-727" not in caplog.text
    assert "code=periodic_session_cleanup_failed" in caplog.text
    assert "phase=rollback" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_initializes_routers_remote_and_s3(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setattr(
        service_main,
        "config",
        _FakeConfig(str(tmp_path / "sessions"), api_keys=False, s3=True),
    )
    async with service_main.lifespan(FastAPI()):
        assert service_main.pipeline_router.get_session_manager().storage_path == (
            tmp_path / "sessions"
        )


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_cleanup_task(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_config = _FakeConfig(str(tmp_path / "cleanup"), api_keys=True)
    fake_config.cleanup_enabled = True
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setattr(service_main, "config", fake_config)

    started = service_main.asyncio.Event()

    async def fake_periodic_cleanup_task(**_kwargs) -> None:
        started.set()
        await service_main.asyncio.sleep(3600)

    monkeypatch.setattr(
        service_main,
        "_periodic_cleanup_task",
        fake_periodic_cleanup_task,
    )

    with caplog.at_level(logging.INFO):
        async with service_main.lifespan(FastAPI()):
            await service_main.asyncio.wait_for(started.wait(), timeout=1)

    assert "Background cleanup enabled" in caplog.text
    assert "Cleanup task stopped" in caplog.text


@pytest.mark.asyncio
async def test_lifespan_warp_success_and_import_fallbacks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_RENDER_BACKEND", "warp")
    monkeypatch.setattr(
        service_main,
        "config",
        _FakeConfig(str(tmp_path / "warp"), api_keys=True),
    )

    warp = ModuleType("warp")
    warp.__version__ = "1.0"
    warp.init = lambda: None
    warp.get_cuda_device_count = lambda: 2
    monkeypatch.setitem(sys.modules, "warp", warp)
    for name in (
        "newton",
        "newton._src",
        "newton._src.sensors",
        "newton._src.sensors.warp_raytrace",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["newton._src.sensors.warp_raytrace"].RenderContext = object

    async with service_main.lifespan(FastAPI()):
        manager = service_main.pipeline_router.get_session_manager()
        assert manager.storage_path == tmp_path / "warp"
        assert manager.store.kind == "local"

    # Remove the optional modules to cover the ImportError fallbacks.
    for name in (
        "warp",
        "newton",
        "newton._src",
        "newton._src.sensors",
        "newton._src.sensors.warp_raytrace",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    async with service_main.lifespan(FastAPI()):
        manager = service_main.pipeline_router.get_session_manager()
        assert manager.storage_path == tmp_path / "warp"
        assert manager.store.kind == "local"


@pytest.mark.asyncio
async def test_lifespan_warp_init_warning(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PA_RENDER_BACKEND", "warp")
    monkeypatch.setattr(
        service_main,
        "config",
        _FakeConfig(str(tmp_path / "warp-fail"), api_keys=True),
    )
    warp = ModuleType("warp")

    def fail_init() -> None:
        raise RuntimeError("warp bad")

    warp.init = fail_init
    monkeypatch.setitem(sys.modules, "warp", warp)
    async with service_main.lifespan(FastAPI()):
        manager = service_main.pipeline_router.get_session_manager()
        assert manager.storage_path == tmp_path / "warp-fail"
        assert manager.store.kind == "local"


@pytest.mark.asyncio
async def test_lifespan_warp_import_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("PA_RENDER_BACKEND", "warp")
    monkeypatch.setattr(
        service_main,
        "config",
        _FakeConfig(str(tmp_path / "warp-missing"), api_keys=True),
    )
    original_import = builtins.__import__

    def import_without_warp(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "warp":
            raise ImportError("warp missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_warp)
    with caplog.at_level(logging.INFO):
        async with service_main.lifespan(FastAPI()):
            manager = service_main.pipeline_router.get_session_manager()
            assert manager.storage_path == tmp_path / "warp-missing"
            assert manager.store.kind == "local"
    assert "warp-lang not installed" in caplog.text


@pytest.mark.asyncio
async def test_handlers_and_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PA_MAX_ACTIVE_SESSIONS", "3")
    monkeypatch.setattr(registry_module, "_job_registry", None)
    monkeypatch.setattr(service_main, "_is_tuning_extra_available", lambda: True)
    monkeypatch.setattr(service_main, "_is_ovphysx_runtime_available", lambda: True)
    health = await service_main.health_check()
    assert health["status"] == "healthy"
    assert health["max_active_sessions"] == 3
    assert health["tuning_extra_available"] is True
    assert health["ovphysx_runtime_available"] is True

    api = await service_main.root_api_info()
    assert api["api"]["refine"]["create"] == "POST /refine"
    assert await service_main.root() == api

    response = await service_main._invalid_session_id_handler(
        SimpleNamespace(), InvalidSessionIdError("bad id")
    )
    assert response.status_code == 400

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    service_main.main()
    assert calls


def test_openapi_identifies_serving_nvcf_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_schema = service_main.app.openapi_schema
    monkeypatch.setenv("NVCF_FUNCTION_VERSION_ID", "version-under-test")
    service_main.app.openapi_schema = None

    try:
        schema = service_main.app.openapi()
        assert schema["info"]["x-nvcf-function-version-id"] == "version-under-test"
        assert schema["paths"]["/refine"]["post"]
    finally:
        service_main.app.openapi_schema = original_schema
