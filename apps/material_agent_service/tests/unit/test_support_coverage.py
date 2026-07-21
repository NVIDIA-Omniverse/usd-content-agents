# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for small support modules in material-agent-service."""

from __future__ import annotations

import builtins
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from ...service import config as config_module
from ...service import json_utils, utils
from ...service.config import ServiceConfig
from ...service.runtime import bus as bus_module
from ...service.runtime import registry as registry_module
from ...service.runtime.bus import EventBus, get_event_bus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.runtime.registry import JobRegistry, get_job_registry
from ...service.session import storage as session_storage
from ...service.storage.base import SessionStore
from ...service.storage.local_store import LocalSessionStore


class _Color(Enum):
    RED = "red"


@dataclass
class _Payload:
    path: Path
    created: date


class _ModelDumpPayload:
    def model_dump(self):
        return {"nested": {_Color.RED, "blue"}}


@pytest.mark.unit
def test_json_safe_handles_remaining_supported_types(tmp_path: Path) -> None:
    payload = {
        _Color.RED: _Payload(tmp_path / "asset.usd", date(2026, 1, 2)),
        "created_at": datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        "values": (_ModelDumpPayload(), {3, 1, 2}),
    }

    result = json_utils.to_json_safe(payload)

    assert result["red"] == {
        "path": str(tmp_path / "asset.usd"),
        "created": "2026-01-02",
    }
    assert result["created_at"] == "2026-01-02T03:04:00+00:00"
    assert set(result["values"][0]["nested"]) == {"blue", "red"}
    assert result["values"][1] == [1, 2, 3]
    assert result == {
        "red": {
            "path": str(tmp_path / "asset.usd"),
            "created": "2026-01-02",
        },
        "created_at": "2026-01-02T03:04:00+00:00",
        "values": [{"nested": result["values"][0]["nested"]}, [1, 2, 3]],
    }
    assert json_utils.to_json_safe(object()).startswith("<object object")


@pytest.mark.unit
def test_utils_version_paths_and_access_log_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "_version.txt").write_text("9.8.7\n")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    assert utils.get_version() == "9.8.7"

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(
        utils,
        "version",
        lambda _name: (_ for _ in ()).throw(utils.PackageNotFoundError),
    )
    assert utils.get_version() == "0.0.1-dev"

    log_filter = utils.AccessLogFilter()
    assert log_filter.filter(logging.LogRecord("x", 20, "", 1, "GET /ready", (), None))
    assert not log_filter.filter(
        logging.LogRecord("x", 20, "", 1, "GET /health", (), None)
    )
    assert not log_filter.filter(
        logging.LogRecord("x", 20, "", 1, "GET /metrics", (), None)
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_store_protocol_default_methods_are_covered() -> None:
    store = object()

    assert await SessionStore.list_sessions(store) is None
    assert SessionStore.invalidate_sessions_cache(store) is None
    assert await SessionStore.get_json_versioned(store, "sid", "session.json") is None
    assert (
        await SessionStore.replace_json_if_version(
            store,
            "sid",
            "session.json",
            {},
            None,
        )
        is None
    )
    assert await SessionStore.get_json_batch(store, [], "session.json") is None
    assert await SessionStore.sync_to_local(store, "sid", "/tmp/sid") is None
    assert await SessionStore.sync_from_local(store, "sid", "/tmp/sid") is None
    assert await SessionStore.cleanup_stale_local_sessions(store, "/tmp") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_store_remaining_branches(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "sessions"))

    await store.delete_session("missing")
    assert await store.list_sessions() == []
    store.invalidate_sessions_cache()
    assert await store.list_keys("missing") == []
    assert await store.get_event_log("missing") == []

    await store.append_event(
        "sid", {"step": _Color.RED, "path": tmp_path / "input.usd"}
    )
    assert await store.get_event_log("sid") == [
        {"step": "red", "path": str(tmp_path / "input.usd")}
    ]
    await store.put_bytes("sid", "events.jsonl", b'{"step": "partial"')
    assert await store.get_event_log("sid") == []

    await store.put_bytes("sid", "input/a.txt", b"a")
    await store.put_bytes("sid", "other.txt", b"b")
    assert await store.list_keys("sid", prefix="input/") == ["input/a.txt"]
    # Simulate a legacy pre-policy artifact directly. New writes into this
    # reserved namespace fail explicitly, while read/list/sync surfaces must
    # remain blind to historical files that are already present.
    legacy_temp = store._session_dir("sid") / "input/nested/.pipeline_temp"
    legacy_temp.mkdir(parents=True)
    (legacy_temp / "config.yaml").write_bytes(b"api_key: sentinel")
    await store.delete_file("sid", "other.txt")
    assert not await store.exists("sid", "other.txt")
    await store.put_json("sid", "one.json", {"value": 1})
    assert await store.get_json_batch(["sid", "missing"], "one.json") == [
        {"value": 1},
        None,
    ]
    assert (
        await store.sync_to_local(
            "sid",
            str(store._session_dir("sid")),
        )
        == 0
    )

    local_dir = tmp_path / "local-copy"
    assert await store.sync_to_local("sid", str(local_dir), prefix="input/") == 1
    assert (local_dir / "input" / "a.txt").read_bytes() == b"a"
    assert not (
        local_dir / "input" / "nested" / ".pipeline_temp" / "config.yaml"
    ).exists()
    assert await store.sync_to_local("sid", str(local_dir), prefix="input/") == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_sync_rejects_symlink_alias_to_pipeline_temp(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "store"))
    secret_path = store._session_dir("sid") / ".pipeline_temp" / "config.yaml"
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("api_key: sentinel", encoding="utf-8")
    alias = store._session_dir("sid") / "input" / "export.yaml"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(secret_path)

    destination = tmp_path / "local-copy"
    with pytest.raises(
        RuntimeError,
        match="symlinked session artifact",
    ):
        await store.sync_to_local("sid", str(destination), prefix="input/")

    assert not (destination / "input" / "export.yaml").exists()


@pytest.mark.unit
def test_count_jsonl_lines_logs_after_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "events.jsonl"
    file_path.write_text('{"id": 1}\n')

    def fail_open(*_args, **_kwargs):
        raise OSError("locked")

    monkeypatch.setattr(builtins, "open", fail_open)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    assert session_storage.count_jsonl_lines(file_path, retries=2) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_registry_cancel_and_global_registry_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = JobRegistry(max_concurrent=1)
    assert await registry.cancel("missing-session") is False

    done_task = SimpleNamespace(done=lambda: True)
    registry._tasks["done-session"] = done_task
    assert await registry.cancel("done-session") is False

    monkeypatch.setenv("MA_MAX_ACTIVE_SESSIONS", "5")
    monkeypatch.setattr(registry_module, "_job_registry", None)
    assert get_job_registry().max_concurrent == 5
    assert get_job_registry() is registry_module._job_registry


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_bus_remaining_state_and_persistence_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bus = EventBus()
    manager = SimpleNamespace()
    bus.set_session_manager(manager)
    assert bus._get_session_manager() is manager

    async def no_state(_event):
        return None

    monkeypatch.setattr(bus, "_apply_event_to_state", no_state)
    event = ProgressEvent(
        session_id="missing-state",
        step="render",
        state=StepState.RUNNING,
        percent=50,
    )
    await bus.emit(event)
    assert event.overall_percent == 0

    no_manager_bus = EventBus()
    monkeypatch.setattr(no_manager_bus, "_get_session_manager", lambda: None)
    await no_manager_bus._persist_status("sid", "completed")
    await no_manager_bus._save_event_to_log(event)

    class FakeManager:
        def __init__(self) -> None:
            self.updated: list[tuple[str, dict]] = []

        async def session_exists(self, session_id: str) -> bool:
            return session_id == "sid"

        async def update_session(self, session_id: str, data: dict) -> None:
            self.updated.append((session_id, data))

        def get_session_dir(self, session_id: str) -> Path:
            path = tmp_path / session_id
            path.mkdir()
            return path

    fake_manager = FakeManager()
    bus = EventBus()
    bus.set_session_manager(fake_manager)
    await bus._apply_event_to_state(
        ProgressEvent(
            session_id="sid",
            step="render",
            state=StepState.CANCELLED,
            percent=0,
        )
    )
    assert fake_manager.updated == [("sid", {"status": "cancelled"})]

    await bus._save_event_to_log(
        ProgressEvent(
            session_id="sid",
            step="render",
            state=StepState.RUNNING,
            percent=1,
        )
    )
    assert (tmp_path / "sid" / "event_log.jsonl").exists()

    bus.get_queue("sid")
    assert bus.get_snapshot("sid") is not None
    bus.cleanup_session("sid")
    assert bus.get_snapshot("sid") is None

    monkeypatch.setattr(bus_module, "_event_bus", None)
    assert get_event_bus() is bus_module._event_bus


@pytest.mark.unit
def test_config_helper_credential_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_module._is_local_render_endpoint(None) is False
    assert (
        config_module._backend_has_credentials(
            "",
            nvidia_api_key=None,
        )
        is True
    )
    assert (
        config_module._backend_has_credentials(
            "mock",
            nvidia_api_key=None,
        )
        is True
    )
    from world_understanding.functions.models.backends import registry

    plugin_backend = "test-material-service-vlm-plugin"
    monkeypatch.setitem(registry._vlm_backends, plugin_backend, lambda **_kwargs: None)
    monkeypatch.setitem(registry._vlm_backend_requires_api_key, plugin_backend, True)
    monkeypatch.setitem(registry._vlm_backend_capabilities, plugin_backend, frozenset())
    assert (
        config_module._backend_has_credentials(
            plugin_backend,
            nvidia_api_key=None,
            api_key="plugin-test",
        )
        is True
    )
    assert (
        config_module._image_gen_backend_has_credentials(
            "",
            nvidia_api_key=None,
        )
        is True
    )
    no_auth_image_backend = "test-material-service-no-auth-image-plugin"
    monkeypatch.setitem(
        registry._image_gen_backends,
        no_auth_image_backend,
        lambda **_kwargs: None,
    )
    monkeypatch.setitem(
        registry._image_gen_backend_requires_api_key,
        no_auth_image_backend,
        False,
    )
    assert (
        config_module._image_gen_backend_has_credentials(
            no_auth_image_backend,
            nvidia_api_key=None,
        )
        is True
    )

    monkeypatch.setattr(
        config_module,
        "get_env_api_key_for_backend",
        lambda _backend, key: key,
    )
    assert (
        config_module._backend_has_credentials(
            "openai",
            nvidia_api_key=None,
            api_key="openai-test",
        )
        is True
    )
    assert (
        config_module._backend_has_credentials(
            "anthropic",
            nvidia_api_key=None,
            api_key="anthropic-test",
        )
        is True
    )
    assert (
        config_module._backend_has_credentials(
            "custom",
            nvidia_api_key=None,
            api_key="plugin-test",
        )
        is True
    )
    assert (
        config_module._image_gen_backend_has_credentials(
            "custom-image-provider",
            api_key="plugin-test",
            nvidia_api_key=None,
        )
        is True
    )

    monkeypatch.setattr(
        config_module,
        "is_nvidia_provider_base_url",
        lambda _base_url: True,
    )
    monkeypatch.setattr(
        config_module,
        "get_nim_api_key_for_base_url",
        lambda _base_url, key: key,
    )
    assert (
        config_module._image_gen_backend_has_credentials(
            "nim",
            api_key=None,
            nvidia_api_key="nvapi-test",
            base_url="https://integrate.api.nvidia.com/v1",
        )
        is True
    )


@pytest.mark.unit
def test_config_library_discovery_and_legacy_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_root = tmp_path / "app"
    service_dir = app_root / "service"
    materials_root = app_root / "materials"
    service_dir.mkdir(parents=True)
    materials_root.mkdir()
    fake_config_file = service_dir / "config.py"
    fake_config_file.write_text("# fake")

    (materials_root / "plain-file").write_text("not a directory")
    (materials_root / "no-yaml").mkdir()

    invalid = materials_root / "invalid"
    invalid.mkdir()
    (invalid / "materials.yaml").write_text("- not-a-dict\n")

    incomplete = materials_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "materials.yaml").write_text("materials:\n  entries: []\n")

    good = materials_root / "wood-library"
    good.mkdir()
    (good / "wood.usd").write_text("#usda 1.0\n")
    (good / "materials.yaml").write_text(
        "materials:\n"
        "  library_path: wood.usd\n"
        "  entries:\n"
        "    - name: Oak\n"
        "      description: Wood\n"
        "      binding: /Oak\n"
        "      icon: icons/oak.png\n"
    )

    def fake_path(value):
        if value == config_module.__file__:
            return fake_config_file
        return Path(value)

    monkeypatch.setattr(config_module, "Path", fake_path)

    config = ServiceConfig.__new__(ServiceConfig)
    libraries = config._discover_libraries()

    assert list(libraries) == ["wood-library"]
    assert libraries["wood-library"].name == "wood library"
    assert libraries["wood-library"].icons == {"Oak": "icons/oak.png"}

    missing_root = tmp_path / "missing-app" / "service" / "config.py"
    monkeypatch.setattr(
        config_module,
        "Path",
        lambda value: missing_root if value == config_module.__file__ else Path(value),
    )
    assert config._discover_libraries() == {}
    assert ServiceConfig._load_description() == "Material Agent Service"


@pytest.mark.unit
def test_config_legacy_material_fallback_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ServiceConfig, "_discover_libraries", lambda _self: {})
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    config = ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
        materials_config_path="missing.yaml",
        default_library_id="missing-default",
    )
    assert config.materials == []
    assert config.get_library("missing") is None
    assert config.resolve_material_library("missing") is None

    config.material_icons = {}
    yaml_path = tmp_path / "materials.yaml"
    yaml_path.write_text(
        "materials:\n"
        "  entries:\n"
        "    - name: Copper\n"
        "      description: metal\n"
        "      binding: /Copper\n"
        "      icon: icons/copper.png\n"
    )
    materials = config._load_materials_from_yaml(yaml_path)

    assert materials[0]["name"] == "Copper"
    assert config.material_icons == {"Copper": "icons/copper.png"}
