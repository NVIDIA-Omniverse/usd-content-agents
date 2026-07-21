# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joint_agent
import pytest
from fastapi import FastAPI

from ...service import __version__ as service_package_version
from ...service import main
from ...service.runtime import registry as registry_module


@pytest.fixture(autouse=True)
def _clear_joint_rigger_capability_cache():
    getattr(main._joint_rigger_capabilities, "cache_clear", lambda: None)()
    yield
    getattr(main._joint_rigger_capabilities, "cache_clear", lambda: None)()


class _Store:
    kind = "s3"


class _SessionManager:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


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
        monkeypatch.delenv("JA_MAX_ACTIVE_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", env_value)
    monkeypatch.setattr(registry_module, "_job_registry", None)

    registry = registry_module.get_job_registry()
    health = await main.health_check()

    assert registry.max_concurrent == expected
    assert health["max_active_sessions"] == expected

    monkeypatch.setenv("JA_MAX_ACTIVE_SESSIONS", "7")
    assert (await main.health_check())["max_active_sessions"] == expected


def test_load_aws_config_file_into_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = logging.getLogger("joint-main-test")
    monkeypatch.delenv("AWS_CONFIG_FILE", raising=False)
    main._load_aws_config_file_into_env(log=log)

    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing.env"))
    main._load_aws_config_file_into_env(log=log)

    bad = tmp_path / "bad.env"
    bad.mkdir()
    monkeypatch.setenv("AWS_CONFIG_FILE", str(bad))
    main._load_aws_config_file_into_env(log=log)

    cfg = tmp_path / "aws.env"
    cfg.write_text(
        "aws_access_key_id=key\naws_secret_access_key=secret\nregion=us-west-2\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg))
    main._load_aws_config_file_into_env(log=log)
    assert main.os.environ["AWS_ACCESS_KEY_ID"] == "key"
    assert main.os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert main.os.environ["AWS_DEFAULT_REGION"] == "us-west-2"
    assert main.os.environ["AWS_REGION"] == "us-west-2"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "existing")
    main._load_aws_config_file_into_env(log=log)
    assert main.os.environ["AWS_ACCESS_KEY_ID"] == "existing"


async def test_health_root_exception_handler_and_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main.config, "nvidia_api_key", "configured")
    monkeypatch.setattr(
        main,
        "_joint_rigger_capabilities",
        lambda: {
            "owned_core_available": True,
            "usd_joint_rigger_available": True,
            "has_handoff_create_joints": True,
        },
    )
    health = await main.health_check()
    assert health["api_keys_configured"] is True
    assert health["capabilities"]["joint_rigger"]["usd_joint_rigger_available"] is True
    assert health["capabilities"]["joint_rigger"]["owned_core_available"] is True
    assert (await main.root())["api"]["artifacts"]["predictions"] == (
        "GET /artifacts/{session_id}/predictions"
    )
    assert (await main.root())["api"]["artifacts"]["joint_rigger_output"] == (
        "GET /artifacts/{session_id}/joint-rigger-output"
    )
    assert (await main.root())["api"]["artifacts"][
        "joint_rigger_output_filename"
    ] == "rigged.usdz"

    response = await main._invalid_session_id_handler(
        object(),
        main.InvalidSessionIdError("bad session"),
    )
    assert response.status_code == 400
    assert b"bad session" in response.body

    calls: list[dict[str, Any]] = []

    class _Uvicorn:
        @staticmethod
        def run(*args: Any, **kwargs: Any) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", _Uvicorn)
    main.main()
    assert calls[0]["args"] == ("service.main:app",)
    assert calls[0]["kwargs"]["port"] == 8000


def test_joint_rigger_capabilities_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def _raise_import_error(_name: str) -> object:
        calls["count"] += 1
        raise ImportError("missing test package")

    monkeypatch.setattr(main.importlib, "import_module", _raise_import_error)

    capabilities = main._joint_rigger_capabilities()
    cached_capabilities = main._joint_rigger_capabilities()

    assert capabilities == {
        "owned_core_available": False,
        "owned_core_import_error_type": "ImportError",
        "usd_joint_rigger_available": False,
        "import_error_type": "ImportError",
    }
    assert cached_capabilities == capabilities
    assert calls["count"] == 2


def test_joint_rigger_capabilities_reports_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        apply_joint_rigger=lambda: None,
        create_joints=lambda: None,
    )
    monkeypatch.setattr(main.importlib, "import_module", lambda _name: module)

    capabilities = main._joint_rigger_capabilities()

    assert capabilities == {
        "owned_core_available": True,
        "usd_joint_rigger_available": True,
        "has_apply_joint_rigger": True,
        "has_handoff_create_joints": True,
    }


def test_installed_module_service_and_runtime_versions_align() -> None:
    joint_distribution_version = distribution_version("joint-agent")
    service_distribution_version = distribution_version("joint-agent-service")

    assert joint_agent.__version__ == joint_distribution_version
    assert service_package_version == service_distribution_version
    assert main.config.service_version == service_distribution_version
    assert service_distribution_version == joint_distribution_version


async def test_lifespan_initializes_shared_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managers: list[_SessionManager] = []
    capability_checks: list[bool] = []

    monkeypatch.setattr(main.config, "nvidia_api_key", None)
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
    monkeypatch.setattr(main, "_load_aws_config_file_into_env", lambda *, log: None)
    monkeypatch.setattr(
        main,
        "_joint_rigger_capabilities",
        lambda: capability_checks.append(True) or {"usd_joint_rigger_available": False},
    )

    async with main.lifespan(FastAPI()):
        assert managers[0].kwargs["ttl_hours"] == 1
        assert managers[0].kwargs["run_claim_lease_seconds"] == 300.0
        assert managers[0].kwargs["run_claim_heartbeat_seconds"] == 60.0
        assert main.artifacts_router._test_mgr is managers[0]
        assert capability_checks == [True]
