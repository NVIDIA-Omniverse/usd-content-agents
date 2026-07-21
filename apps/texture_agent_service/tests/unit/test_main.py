# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from world_understanding.utils.public_response import (
    PublicJsonResponseSanitizationMiddleware,
)

from ...service import main
from ...service.config import ServiceConfig
from ...service.runtime import registry as registry_module


def test_public_response_sanitizer_is_outermost() -> None:
    middleware = main.app.user_middleware[0]

    assert middleware.cls is PublicJsonResponseSanitizationMiddleware
    assert middleware.kwargs["session_roots"] == (main.config.session_storage_path,)


class _Store:
    kind = "s3"


class _SessionManager:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.cleaned = False

    def cleanup_expired_sessions(self) -> list[str]:
        self.cleaned = True
        return ["expired"]


class _Task:
    def __init__(self, coro: Any) -> None:
        self.coro = coro
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
        self.coro.close()


class _Bus:
    def __init__(self) -> None:
        self.cleaned_sessions: list[str] = []

    async def cleanup_session(self, session_id: str) -> None:
        self.cleaned_sessions.append(session_id)

    async def cleanup_orphaned_sessions(self) -> list[str]:
        return ["orphan"]


@pytest.fixture(autouse=True)
def _reset_job_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep health tests isolated from the process-wide registry singleton."""
    monkeypatch.setattr(registry_module, "_job_registry", None)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, 4), ("0", 0), ("5", 5), ("-1", 4), ("bad", 4)],
)
@pytest.mark.asyncio
async def test_health_capacity_matches_enforced_registry(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: int,
) -> None:
    if env_value is None:
        monkeypatch.delenv("TA_MAX_ACTIVE_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("TA_MAX_ACTIVE_SESSIONS", env_value)

    registry = registry_module.get_job_registry()
    health = await main.health_check()

    assert registry.max_concurrent == expected
    assert health["max_active_sessions"] == expected

    monkeypatch.setenv("TA_MAX_ACTIVE_SESSIONS", "7")
    assert (await main.health_check())["max_active_sessions"] == expected


@pytest.mark.parametrize("env_value", ["-1", "bad"])
def test_capacity_env_is_owned_by_registry_not_service_config(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("TA_MAX_ACTIVE_SESSIONS", env_value)
    assert not hasattr(ServiceConfig(), "max_active_sessions")


async def test_health_root_and_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "set")
    monkeypatch.setattr(main.config, "image_gen_backend", "nim")
    monkeypatch.setattr(main.config, "llm_backend", "mock")
    monkeypatch.setattr(main.config, "nvidia_api_key", "set")

    health = await main.health_check()
    assert health["active_backend_key_configured"] is True
    assert health["llm_ready"] is True
    assert (await main.root())["api"]["pipeline"]["create"] == "POST /pipeline"

    calls: list[dict[str, Any]] = []

    class _Uvicorn:
        @staticmethod
        def run(*args: Any, **kwargs: Any) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _Uvicorn)
    main.main()
    assert calls[0]["args"] == ("service.main:app",)
    assert calls[0]["kwargs"]["port"] == 8001


async def test_lifespan_initializes_shared_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus = _Bus()
    tasks: list[_Task] = []
    managers: list[_SessionManager] = []

    monkeypatch.setattr(main.config, "nvidia_api_key", None)
    monkeypatch.setattr(main.config, "llm_backend", "openai")
    monkeypatch.setattr(main.config, "llm_api_key", None)
    monkeypatch.setattr(main.config, "llm_api_key_env", "MISSING_TEST_OPENAI_KEY")
    monkeypatch.setattr(
        type(main.config),
        "has_required_api_keys",
        property(lambda _self: False),
    )
    monkeypatch.delenv("MISSING_TEST_OPENAI_KEY", raising=False)
    monkeypatch.setattr(main.config, "session_storage_path", str(tmp_path))
    monkeypatch.setattr(main.config, "session_ttl_hours", 1)
    monkeypatch.setattr(main.config, "storage_s3_bucket", "bucket")
    monkeypatch.setattr(main.config, "storage_s3_prefix", "prefix")
    monkeypatch.setattr(
        type(main.config), "build_session_store", lambda _self: _Store()
    )
    monkeypatch.setattr(
        main,
        "SessionManager",
        lambda **kw: managers.append(_SessionManager(**kw)) or managers[-1],
    )
    monkeypatch.setattr(
        main.pipeline_router,
        "set_session_manager",
        lambda mgr: setattr(main.pipeline_router, "_test_mgr", mgr),
    )
    monkeypatch.setattr(
        main.artifacts_router,
        "set_session_manager",
        lambda mgr: setattr(main.artifacts_router, "_test_mgr", mgr),
    )
    monkeypatch.setattr(
        main.sessions_router,
        "set_session_manager",
        lambda mgr: setattr(main.sessions_router, "_test_mgr", mgr),
    )
    monkeypatch.setattr(
        "apps.texture_agent_service.service.runtime.bus.init_event_bus",
        lambda mgr: setattr(bus, "manager", mgr),
    )
    monkeypatch.setattr(
        "apps.texture_agent_service.service.runtime.bus.get_event_bus",
        lambda: bus,
    )
    monkeypatch.setattr(
        main.asyncio,
        "create_task",
        lambda coro: tasks.append(_Task(coro)) or tasks[-1],
    )

    async with main.lifespan(FastAPI()):
        assert managers[0].kwargs["ttl_hours"] == 1
        assert main.pipeline_router._test_mgr is managers[0]

    assert tasks[0].cancelled is True
