# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-instance session storage coverage for Texture Agent Service."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import artifacts_router, pipeline_router, sessions_router
from ...service.runtime import bus as bus_module
from ...service.runtime.bus import init_event_bus
from ...service.session.manager import SessionManager
from ...service.storage import WORKER_RESERVATION_KEY, LocalSessionStore
from ...service.workers import executor

MINIMAL_USD = b'#usda 1.0\ndef Xform "Root" {}\n'


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(pipeline_router.router)
    app.include_router(sessions_router.router)
    app.include_router(artifacts_router.router)
    return TestClient(app)


def _make_upload_files() -> dict[str, tuple[str, bytes, str]]:
    return {"usd_file": ("cube.usda", MINIMAL_USD, "application/octet-stream")}


def _switch_to(manager: SessionManager) -> None:
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    artifacts_router.set_session_manager(manager)
    bus_module._event_bus = None
    init_event_bus(manager)


def _make_shared_pods(tmp_path: Path) -> tuple[SessionManager, SessionManager]:
    shared_store = LocalSessionStore(str(tmp_path / "shared_store"))
    pod_a = SessionManager(tmp_path / "pod_a_local", ttl_hours=1, store=shared_store)
    pod_b = SessionManager(tmp_path / "pod_b_local", ttl_hours=1, store=shared_store)
    return pod_a, pod_b


def test_generated_prompt_cache_survives_worker_replacement_for_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement worker can rebuild units from generated prompt specs."""
    from texture_agent.functions.material_discovery import MaterialInfo
    from texture_agent.tasks.generate_prompts import (
        GeneratePromptsTask as CachedGeneratePromptsTask,
    )
    from texture_agent.workflows import factory as workflow_factory
    from world_understanding.functions.models import chat_models

    # This regression isolates prompt-cache durability; plan gating is covered
    # independently by the planning API suite.
    monkeypatch.setattr(
        executor,
        "_requires_executable_texture_plan",
        lambda *args: False,
    )

    pod_a, pod_b = _make_shared_pods(tmp_path)
    session_id = "prompt-cache-worker-replacement"
    session_dir = pod_a.create_session(session_id)
    input_dir = session_dir / "input"
    (input_dir / "scene.usda").write_bytes(MINIMAL_USD)
    config = {
        "project": {"session_id": session_id},
        "input": {"usd_path": "scene.usda"},
        "material_textures": {},
        "auto_prompt": {"enabled": True},
    }
    (input_dir / "config.yaml").write_text(
        pipeline_router.yaml.safe_dump(config),
        encoding="utf-8",
    )
    # Upload already made the source asset durable before the first worker ran.
    pod_a.sync_to_store(session_id, "input/scene.usda")

    generated_spec = {
        "Paint": {
            "prompt": "generated matte blue paint",
            "opacity": 1.0,
        }
    }

    class GeneratePromptsTask:
        name = "GeneratePrompts"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            prompts_path = (
                Path(context["working_dir"]) / "prompts" / "material_prompts.json"
            )
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_path.write_text(json.dumps(generated_spec), encoding="utf-8")
            context["material_textures"] = generated_spec
            return context

    _switch_to(pod_a)
    asyncio.run(
        executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict=config,
            session_manager=pod_a,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=(
                lambda context, skip=None, only=None: [GeneratePromptsTask()]
            ),
        )
    )

    prompt_key = "cache/prompts/material_prompts.json"
    assert pod_a.get_session_metadata(session_id)["status"] == "completed"
    assert not (pod_b.get_session_dir(session_id) / prompt_key).exists()

    class DiscoverMaterialsTask:
        name = "DiscoverMaterials"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            context["discovered_materials"] = [
                MaterialInfo(
                    prim_path="/World/Looks/Paint",
                    name="Paint",
                    bound_prim_paths=["/World/Mesh"],
                )
            ]
            return context

    def _replacement_worker_factory(
        context: dict[str, Any],
        skip: list[str] | None = None,
        only: list[str] | None = None,
    ) -> list[Any]:
        return [DiscoverMaterialsTask(), CachedGeneratePromptsTask()]

    def _forbid_prompt_backend(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("cached regeneration must not contact the prompt backend")

    class _ImmediateRegistry:
        async def register(
            self,
            sid: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            assert sid == session_id
            try:
                await coro
            finally:
                if on_finished is not None:
                    on_finished()

    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        _forbid_prompt_backend,
    )
    monkeypatch.setattr(
        workflow_factory,
        "create_texture_pipeline_workflow",
        _replacement_worker_factory,
    )
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _ImmediateRegistry(),
    )

    _switch_to(pod_b)
    response = _build_app().post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert pod_a.store.exists(session_id, prompt_key)
    assert pod_b.get_session_metadata(session_id)["status"] == "completed"
    assert (
        json.loads(
            (pod_b.get_session_dir(session_id) / prompt_key).read_text(encoding="utf-8")
        )
        == generated_spec
    )


def test_generated_prompt_cache_survives_later_step_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt output is durable before a later pipeline step can fail."""
    monkeypatch.setattr(
        executor,
        "_requires_executable_texture_plan",
        lambda *args: False,
    )

    pod_a, pod_b = _make_shared_pods(tmp_path)
    session_id = "prompt-cache-later-failure"
    session_dir = pod_a.create_session(session_id)
    generated_spec = {
        "Paint": {
            "prompt": "generated matte blue paint",
            "opacity": 1.0,
        }
    }
    finalization_attempted = False

    async def _record_finalization() -> None:
        nonlocal finalization_attempted
        finalization_attempted = True

    class GeneratePromptsTask:
        name = "GeneratePrompts"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            prompts_path = (
                Path(context["working_dir"]) / "prompts" / "material_prompts.json"
            )
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_path.write_text(json.dumps(generated_spec), encoding="utf-8")
            return context

    class LaterFailingTask:
        name = "LaterFailingTask"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("later step failed")

    _switch_to(pod_a)
    with pytest.raises(RuntimeError, match="later step failed"):
        asyncio.run(
            executor._execute_pipeline_inner(
                session_id=session_id,
                config_dict={
                    "project": {"session_id": session_id},
                    "input": {"usd_path": "scene.usda"},
                },
                session_manager=pod_a,
                event_bus=bus_module.get_event_bus(),
                session_dir=session_dir,
                only_steps=None,
                skip_steps=None,
                create_texture_pipeline_workflow=(
                    lambda context, skip=None, only=None: [
                        GeneratePromptsTask(),
                        LaterFailingTask(),
                    ]
                ),
                on_artifacts_synced=_record_finalization,
            )
        )

    prompt_key = "cache/prompts/material_prompts.json"
    assert pod_a.get_session_metadata(session_id)["status"] == "failed"
    assert finalization_attempted is False
    assert pod_a.store.exists(session_id, prompt_key)
    assert not (pod_b.get_session_dir(session_id) / prompt_key).exists()

    assert pod_b.sync_from_store(session_id, "cache/prompts/") == 1
    assert (
        json.loads(
            (pod_b.get_session_dir(session_id) / prompt_key).read_text(encoding="utf-8")
        )
        == generated_spec
    )


def test_generated_prompt_cache_survives_between_step_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cooperative cancellation after prompt generation keeps its checkpoint."""
    monkeypatch.setattr(
        executor,
        "_requires_executable_texture_plan",
        lambda *args: False,
    )

    pod_a, pod_b = _make_shared_pods(tmp_path)
    session_id = "prompt-cache-between-step-cancel"
    session_dir = pod_a.create_session(session_id)
    generated_spec = {
        "Paint": {
            "prompt": "generated matte blue paint",
            "opacity": 1.0,
        }
    }
    later_task_ran = False
    finalization_attempted = False

    async def _record_finalization() -> None:
        nonlocal finalization_attempted
        finalization_attempted = True

    class GeneratePromptsTask:
        name = "GeneratePrompts"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            prompts_path = (
                Path(context["working_dir"]) / "prompts" / "material_prompts.json"
            )
            prompts_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_path.write_text(json.dumps(generated_spec), encoding="utf-8")
            pod_a.request_cancellation(session_id)
            return context

    class LaterTask:
        name = "LaterTask"

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            nonlocal later_task_ran
            later_task_ran = True
            return context

    _switch_to(pod_a)
    asyncio.run(
        executor._execute_pipeline_inner(
            session_id=session_id,
            config_dict={
                "project": {"session_id": session_id},
                "input": {"usd_path": "scene.usda"},
            },
            session_manager=pod_a,
            event_bus=bus_module.get_event_bus(),
            session_dir=session_dir,
            only_steps=None,
            skip_steps=None,
            create_texture_pipeline_workflow=(
                lambda context, skip=None, only=None: [
                    GeneratePromptsTask(),
                    LaterTask(),
                ]
            ),
            on_artifacts_synced=_record_finalization,
        )
    )

    prompt_key = "cache/prompts/material_prompts.json"
    assert pod_a.get_session_metadata(session_id)["status"] == "cancelled"
    assert later_task_ran is False
    assert finalization_attempted is False
    assert pod_a.store.exists(session_id, prompt_key)
    assert not (pod_b.get_session_dir(session_id) / prompt_key).exists()

    assert pod_b.sync_from_store(session_id, "cache/prompts/") == 1
    assert (
        json.loads(
            (pod_b.get_session_dir(session_id) / prompt_key).read_text(encoding="utf-8")
        )
        == generated_spec
    )


def test_uploaded_session_status_is_visible_across_instances(tmp_path: Path) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]

    _switch_to(pod_b)
    status = client.get(f"/pipeline/{session_id}/status")

    assert status.status_code == 200, status.text
    assert status.json()["session_id"] == session_id
    assert status.json()["status"] == "ready"


def test_shared_session_list_is_consistent_across_instances(tmp_path: Path) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    _switch_to(pod_a)
    first = client.post("/pipeline/upload-usd", files=_make_upload_files()).json()[
        "session_id"
    ]
    _switch_to(pod_b)
    second = client.post("/pipeline/upload-usd", files=_make_upload_files()).json()[
        "session_id"
    ]

    _switch_to(pod_a)
    sessions_a = client.get("/sessions").json()["sessions"]
    _switch_to(pod_b)
    sessions_b = client.get("/sessions").json()["sessions"]

    assert {item["session_id"] for item in sessions_a} == {first, second}
    assert {item["session_id"] for item in sessions_b} == {first, second}


def test_existing_uploaded_session_can_queue_on_another_instance_without_hydration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    class StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: StubRegistry())

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]

    assert not (pod_b.get_session_dir(session_id) / "input" / "scene.usda").exists()

    _switch_to(pod_b)
    response = client.post("/pipeline", data={"session_id": session_id})

    assert response.status_code == 202, response.text
    assert not (pod_b.get_session_dir(session_id) / "input" / "scene.usda").exists()
    assert (pod_b.get_session_dir(session_id) / "input" / "config.yaml").exists()


def test_pipeline_registers_queued_reservation_heartbeat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)
    shared_store = pod_a.store
    heartbeat_seen = False

    class StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_queued_heartbeat: Any = None,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            nonlocal heartbeat_seen
            assert on_queued_heartbeat is not None
            marker = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
            assert marker is not None
            marker["updated_at"] = "2026-01-01T00:00:00+00:00"
            shared_store.put_json(session_id, WORKER_RESERVATION_KEY, marker)

            result = on_queued_heartbeat()
            if inspect.isawaitable(result):
                await result

            updated = shared_store.get_json(session_id, WORKER_RESERVATION_KEY)
            assert updated is not None
            assert updated["owner_token"] == marker["owner_token"]
            assert updated["updated_at"] != "2026-01-01T00:00:00+00:00"
            heartbeat_seen = True

            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: StubRegistry())

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]

    _switch_to(pod_b)
    response = client.post("/pipeline", data={"session_id": session_id})

    assert response.status_code == 202, response.text
    assert heartbeat_seen is True


def test_remote_running_session_cannot_start_on_another_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    class StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("remote-running session should not be registered")

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: StubRegistry())

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]

    pod_b.sync_from_store(session_id)
    assert (pod_b.get_session_dir(session_id) / "session.json").is_file()
    pod_a.update_session(session_id, {"status": "running"})

    _switch_to(pod_b)
    response = client.post("/pipeline", data={"session_id": session_id})

    assert response.status_code == 409, response.text
    assert "already running" in response.json()["detail"]


def test_shared_worker_reservation_blocks_start_on_another_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    class StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("reserved session should not be registered")

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: StubRegistry())

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]

    worker_lock = pod_a.acquire_worker_lock(session_id, timeout=0)
    try:
        _switch_to(pod_b)
        response = client.post("/pipeline", data={"session_id": session_id})

        assert response.status_code == 409, response.text
        assert "already running" in response.json()["detail"]
    finally:
        pod_a.release_worker_lock(worker_lock, session_id)


def test_invalid_direct_upload_does_not_leave_shared_session(tmp_path: Path) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    _switch_to(pod_a)
    response = client.post(
        "/pipeline",
        files={"usd_file": ("cube.txt", b"not usd", "text/plain")},
    )

    assert response.status_code == 400
    assert pod_a.list_sessions() == []

    _switch_to(pod_b)
    assert client.get("/sessions").json()["sessions"] == []


def test_shared_artifacts_download_from_any_instance(tmp_path: Path) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    session_id = response.json()["session_id"]
    session_dir = pod_a.get_session_dir(session_id)
    materials_path = session_dir / "cache" / "discovery" / "materials.json"
    materials_path.parent.mkdir(parents=True, exist_ok=True)
    materials_path.write_text(json.dumps([{"name": "Steel"}]), encoding="utf-8")
    texture_path = session_dir / "cache" / "textures" / "steel.png"
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(b"png")
    pod_a.sync_to_store(session_id, "cache/discovery/")
    pod_a.sync_to_store(session_id, "cache/textures/")

    _switch_to(pod_b)
    materials = client.get(f"/artifacts/{session_id}/materials")
    textures = client.get(f"/artifacts/{session_id}/textures")

    assert materials.status_code == 200, materials.text
    assert materials.json() == [{"name": "Steel"}]
    assert textures.status_code == 200, textures.text
    assert textures.headers["content-type"] == "application/zip"


def test_sse_returns_503_when_running_on_another_instance(tmp_path: Path) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    session_id = response.json()["session_id"]
    pod_a.update_session(session_id, {"status": "running"})

    _switch_to(pod_b)
    response = client.get(f"/pipeline/{session_id}/events")

    assert response.status_code == 503
    assert "different instance" in response.json()["detail"]


def test_sse_returns_503_when_stale_local_metadata_but_remote_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _build_app()
    pod_a, pod_b = _make_shared_pods(tmp_path)

    class StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: StubRegistry())

    _switch_to(pod_a)
    response = client.post("/pipeline/upload-usd", files=_make_upload_files())
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    assert (pod_a.get_session_dir(session_id) / "session.json").is_file()

    worker_lock = pod_b.acquire_worker_lock(session_id, timeout=0)
    try:
        pod_b.update_session(session_id, {"status": "running"})

        _switch_to(pod_a)
        response = client.get(f"/pipeline/{session_id}/events")

        assert response.status_code == 503
        assert "different instance" in response.json()["detail"]
    finally:
        pod_b.release_worker_lock(worker_lock, session_id)
