# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
import logging
import sys
from types import ModuleType, SimpleNamespace

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


def test_get_max_active_sessions_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JA_MAX_ACTIVE_SESSIONS", raising=False)
    assert registry_module.resolve_max_active_sessions() == 1
    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "0")
    assert registry_module.resolve_max_active_sessions() == 0
    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "5")
    assert registry_module.resolve_max_active_sessions() == 5
    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "-1")
    assert registry_module.resolve_max_active_sessions() == 1
    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "bad")
    assert registry_module.resolve_max_active_sessions() == 1


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
    service_name = "Joint Agent Service"
    service_version = "test"
    vlm_backend = "mock"
    vlm_model = "mock-model"
    vlm_temperature = 0.0
    session_ttl_hours = 1
    run_claim_lease_seconds = 300.0
    run_claim_heartbeat_seconds = 60.0
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
async def test_lifespan_initializes_routers_remote_and_s3(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
async def test_lifespan_warp_success_and_import_fallbacks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    real_import = builtins.__import__

    def import_without_warp(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "warp":
            raise ImportError("missing warp")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_warp)

    async with service_main.lifespan(FastAPI()):
        manager = service_main.pipeline_router.get_session_manager()
        assert manager.storage_path == tmp_path / "warp"
        assert manager.store.kind == "local"


@pytest.mark.asyncio
async def test_lifespan_warp_init_warning(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
async def test_handlers_and_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "3")
    monkeypatch.setattr(registry_module, "_job_registry", None)
    health = await service_main.health_check()
    assert health["status"] == "healthy"
    assert health["max_active_sessions"] == 3

    api = await service_main.root_api_info()
    assert api["api"]["pipeline"]["create"] == "POST /pipeline"
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
