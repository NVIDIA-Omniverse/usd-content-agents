# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WP2 service contract tests for plan-only and bounded planning fields."""

from __future__ import annotations

import asyncio
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from texture_agent.functions import cached_apply
from texture_agent.planning import (
    TexturePlan,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanExecution,
    TexturePlanLimits,
    TexturePlanRequest,
    TexturePlanSource,
    TexturePlanUnit,
)

from ...service.routers import pipeline_router
from ...service.session.manager import SessionManager
from ...service.storage import LocalSessionStore


def _plan() -> TexturePlan:
    request = TexturePlanRequest(
        source=TexturePlanSource(
            source_asset="session://sid/input/scene.usd",
        ),
        backend="service",
        backend_default_cap=16,
    )
    return TexturePlan(
        generated_at=datetime(2026, 6, 29, tzinfo=UTC),
        request=request,
        limits=TexturePlanLimits.from_request(request),
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=0,
            renderable_prim_count=0,
            renderable_subset_count=0,
            effective_bound_material_count=0,
            selected_material_count=0,
            selected_unit_count=0,
            skipped_item_count=0,
            planned_generation_job_count=0,
        ),
        decision=TexturePlanDecision(state="ready", execution_allowed=True),
    )


def _plan_with_unit() -> TexturePlan:
    request = TexturePlanRequest(
        source=TexturePlanSource(
            source_asset="session://sid/input/scene.usd",
        ),
        backend="service",
        backend_default_cap=16,
    )
    unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/Mesh",),
        display_name="Paint",
        selection_reason_code="effectively_bound",
        selection_reason="Used by renderable geometry.",
        detail_policy="surface_only",
    )
    return TexturePlan(
        generated_at=datetime(2026, 6, 29, tzinfo=UTC),
        request=request,
        limits=TexturePlanLimits.from_request(request),
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=1,
            renderable_prim_count=1,
            renderable_subset_count=0,
            effective_bound_material_count=1,
            selected_material_count=1,
            selected_unit_count=1,
            skipped_item_count=0,
            planned_generation_job_count=1,
        ),
        selected_units=(unit,),
        decision=TexturePlanDecision(state="ready", execution_allowed=True),
    )


def test_default_config_records_plan_only_and_operator_override() -> None:
    config = pipeline_router.build_default_pipeline_config(
        session_id="sid",
        usd_path="/private/sessions/sid/input/scene.usd",
        working_dir="/private/sessions/sid/cache",
        plan_only=True,
        planning_discovery_mode="explicit",
        planning_unit_mode="per_prim",
        explicit_material_paths=["/World/Looks/Paint"],
        explicit_prim_paths=["/World/Mesh"],
        operator_override_cap=64,
    )

    assert config["planning"] == {
        "source_asset": "session://sid/input/scene.usd",
        "discovery_mode": "explicit",
        "unit_mode": "per_prim",
        "explicit_material_paths": ["/World/Looks/Paint"],
        "explicit_prim_paths": ["/World/Mesh"],
        "backend_default_cap": 32,
        "operator_override_cap": 64,
        "plan_only": True,
    }


def test_default_config_derives_cap_from_engine_and_uv_policy() -> None:
    simple_remote = pipeline_router.build_default_pipeline_config(
        session_id="sid",
        usd_path="/private/sessions/sid/input/scene.usd",
        working_dir="/private/sessions/sid/cache",
        texture_backend="simple_image_gen",
        texture_endpoint="http://simple-texture.test",
    )
    force_projection = pipeline_router.build_default_pipeline_config(
        session_id="sid",
        usd_path="/private/sessions/sid/input/scene.usd",
        working_dir="/private/sessions/sid/cache",
        texture_backend="simple_image_gen",
        texture_endpoint="http://simple-texture.test",
        uv_policy="force_projection",
    )

    assert simple_remote["texture"]["backend"] == "service"
    assert simple_remote["texture"]["engine"] == "simple_image_gen"
    assert simple_remote["planning"]["backend_default_cap"] == 32
    assert force_projection["planning"]["backend_default_cap"] == 16


def test_plan_status_summary_exposes_counts_limits_and_url() -> None:
    summary = pipeline_router._texture_plan_status("sid", _plan())

    assert summary is not None
    assert summary.schema_version == "texture-agent-plan.v1"
    assert summary.decision_state == "ready"
    assert summary.execution_allowed is True
    assert summary.counts["selected_unit_count"] == 0
    assert summary.limits["backend_default_cap"] == 16
    assert summary.limits["hard_cap"] == 64
    assert summary.plan_url == "/pipeline/sid/plan"


def test_pipeline_openapi_documents_planning_fields() -> None:
    app = FastAPI()
    app.include_router(pipeline_router.router)
    schema = app.openapi()
    body_ref = schema["paths"]["/pipeline"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]["$ref"]
    body_name = body_ref.rsplit("/", 1)[-1]
    fields = schema["components"]["schemas"][body_name]["properties"]

    assert {
        "plan_only",
        "discovery_mode",
        "unit_mode",
        "explicit_material_paths_json",
        "explicit_prim_paths_json",
        "operator_override_cap",
    }.issubset(fields)
    assert "/pipeline/{session_id}/plan" in schema["paths"]
    assert "404" in schema["paths"]["/pipeline/{session_id}/plan"]["get"]["responses"]
    assert (
        "texture_plan"
        in schema["components"]["schemas"]["PipelineStatus"]["properties"]
    )


def test_get_plan_returns_validated_artifact(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_dir = manager.create_session("sid")
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(_plan().model_dump_json(), encoding="utf-8")
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).get("/pipeline/sid/plan")

    assert response.status_code == 200
    assert response.json()["schema_version"] == "texture-agent-plan.v1"
    assert response.json()["request"]["source"]["source_asset"].startswith("session://")


def test_get_plan_distinguishes_missing_session_and_missing_plan(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session("sid")
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    assert client.get("/pipeline/missing/plan").status_code == 404
    response = client.get("/pipeline/sid/plan")

    assert response.status_code == 404
    assert "not available yet" in response.json()["detail"]


def test_load_texture_plan_handles_shared_store_missing_and_invalid_local(
    tmp_path: Path,
) -> None:
    import asyncio

    class _MissingSharedManager:
        def uses_shared_store(self) -> bool:
            return True

        def sync_from_store(self, session_id: str, prefix: str) -> None:
            raise FileNotFoundError(prefix)

    assert (
        asyncio.run(
            pipeline_router._load_texture_plan(_MissingSharedManager(), "sid")  # type: ignore[arg-type]
        )
        is None
    )

    manager = SessionManager(tmp_path, ttl_hours=2)
    session_dir = manager.create_session("bad-plan")
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("{bad json", encoding="utf-8")

    assert asyncio.run(pipeline_router._load_texture_plan(manager, "bad-plan")) is None


def test_service_rejects_override_above_hard_cap_before_session_lookup(
    tmp_path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    response = client.post(
        "/pipeline",
        data={"session_id": "missing", "operator_override_cap": "65"},
    )

    assert response.status_code == 422
    assert "hard maximum of 64" in response.json()["detail"]
    assert "narrow" in response.json()["detail"].lower()


def test_service_rejects_redundant_override_before_session_lookup(
    tmp_path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    response = client.post(
        "/pipeline",
        data={"session_id": "missing", "operator_override_cap": "32"},
    )

    assert response.status_code == 422
    assert "greater than the effective backend default cap" in response.json()["detail"]


def test_service_rejects_explicit_discovery_without_scope_before_session_lookup(
    tmp_path,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        "/pipeline",
        data={"session_id": "missing", "discovery_mode": "explicit"},
    )

    assert response.status_code == 422
    assert "Explicit discovery requires" in response.json()["detail"]


def test_plan_only_request_schedules_only_discovery_and_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_id = "plan-only-session"
    session_dir = manager.create_session(session_id)
    manager.update_session(session_id, {"status": "ready"})
    scene = session_dir / "input" / "scene.usd"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _Bus:
        def clear_session_state(self, sid: str) -> None:
            captured["cleared"] = sid

        async def seed_pending_session(self, sid: str) -> None:
            captured["seeded"] = sid

    class _Registry:
        def is_running(self, sid: str) -> bool:
            return False

        async def register(
            self,
            sid: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    def _execute(**kwargs: Any) -> Any:
        captured.update(kwargs)

        async def _noop() -> None:
            return None

        return _noop()

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _Bus())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())
    monkeypatch.setattr(pipeline_router, "execute_pipeline_async", _execute)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        "/pipeline",
        data={"session_id": session_id, "plan_only": "true"},
    )

    assert response.status_code == 202, response.text
    assert captured["only_steps"] == ["discover_materials", "plan_textures"]
    assert captured["config_dict"]["planning"]["plan_only"] is True
    assert captured["config_dict"]["planning"]["source_asset"] == (
        f"session://{session_id}/input/scene.usd"
    )
    assert response.json()["plan_url"] == f"/pipeline/{session_id}/plan"


def _seed_regenerate_session(
    tmp_path: Path,
    *,
    session_id: str = "regen-session",
    include_plan: bool = False,
    include_prompts: bool = False,
) -> tuple[SessionManager, str]:
    manager = SessionManager(tmp_path, ttl_hours=2)
    session_dir = manager.create_session(session_id)
    manager.update_session(session_id, {"status": "completed"})
    config = {
        "project": {"session_id": session_id},
        "input": {"usd_path": "scene.usd"},
        "material_textures": {"Paint": {"prompt": "operator prompt"}},
        "steps": {
            "discover_materials": {"enabled": True},
            "plan_textures": {"enabled": True},
            "generate_prompts": {"enabled": True},
            "generate_textures": {"enabled": True},
        },
    }
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    (input_dir / "config.yaml").write_text(
        pipeline_router.yaml.safe_dump(config),
        encoding="utf-8",
    )
    if include_plan:
        plan_path = session_dir / "cache" / "texture_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(_plan_with_unit().model_dump_json(), encoding="utf-8")
    if include_prompts:
        prompt_path = session_dir / "cache" / "prompts" / "material_prompts.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            json.dumps(
                {
                    "Paint": {"prompt": "cached prompt"},
                    "Copper": {"prompt": "cached copper"},
                }
            ),
            encoding="utf-8",
        )
    return manager, session_id


def _write_apply_cache_key_mode(session_dir: Path, key_mode: str) -> Path:
    marker_path = session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": (pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_SCHEMA),
                "key_mode": key_mode,
            }
        ),
        encoding="utf-8",
    )
    return marker_path


def _seed_flat_plan_unit_textures(
    session_dir: Path,
    plan: TexturePlan,
    *,
    channels: tuple[str, ...] = ("albedo", "normal", "orm"),
) -> None:
    from PIL import Image

    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    for unit in plan.selected_units:
        for channel in channels:
            with Image.new("RGB", (2, 2), color=(64, 128, 192)) as image:
                image.save(textures_dir / f"{unit.unit_id}_{channel}.png")


def _install_regenerate_stubs(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    class _Bus:
        def clear_session_state(self, sid: str) -> None:
            captured["cleared"] = sid

        async def seed_pending_session(self, sid: str) -> None:
            captured["seeded"] = sid

    class _Registry:
        async def register(
            self,
            sid: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            captured["registered"] = sid
            coro.close()
            if on_finished is not None:
                on_finished()

    def _execute(**kwargs: Any) -> Any:
        captured.update(kwargs)

        async def _noop() -> None:
            return None

        return _noop()

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _Bus())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())
    monkeypatch.setattr(pipeline_router, "execute_pipeline_async", _execute)


def test_apply_cache_local_io_is_offloaded_from_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    _seed_flat_plan_unit_textures(session_dir, plan)
    plan_path = session_dir / "cache" / "texture_plan.json"
    calls: list[str] = []

    async def _record_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(getattr(func, "__name__", type(func).__name__))
        return func(*args, **kwargs)

    monkeypatch.setattr(pipeline_router.asyncio, "to_thread", _record_to_thread)

    assert asyncio.run(
        pipeline_router._has_complete_plan_unit_texture_cache(
            manager,
            session_id,
            session_dir,
            plan_path,
        )
    )
    assert calls == ["read_bytes", "_all_local_textures_are_valid"]

    calls.clear()
    asyncio.run(
        pipeline_router._persist_apply_cache_key_mode(
            manager,
            session_id,
            session_dir,
            pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
        )
    )
    assert calls == ["_write_local_marker", "sync_to_store"]


def test_apply_cache_marker_sync_failure_removes_local_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    session_dir = manager.get_session_dir(session_id)
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )

    def _fail_sync(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("shared marker upload failed")

    monkeypatch.setattr(manager, "sync_to_store", _fail_sync)

    with pytest.raises(RuntimeError, match="shared marker upload failed"):
        asyncio.run(
            pipeline_router._persist_apply_cache_key_mode(
                manager,
                session_id,
                session_dir,
                pipeline_router._APPLY_CACHE_KEY_MODE_PLAN,
            )
        )

    assert not marker_path.exists()


def test_post_sync_promotion_rejects_incomplete_plan_cache(
    tmp_path: Path,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    session_dir = manager.get_session_dir(session_id)
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    _seed_flat_plan_unit_textures(
        session_dir,
        _plan_with_unit(),
        channels=("albedo", "normal"),
    )

    with pytest.raises(RuntimeError, match="complete durable plan-unit texture cache"):
        asyncio.run(
            pipeline_router._promote_apply_cache_key_mode_after_artifact_sync(
                manager,
                session_id,
                session_dir,
                session_dir / "cache" / "texture_plan.json",
            )
        )

    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )


def test_plan_cache_validation_rejects_decompression_bomb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    _seed_flat_plan_unit_textures(session_dir, plan)
    monkeypatch.setattr(cached_apply.Image, "MAX_IMAGE_PIXELS", 1)

    assert not asyncio.run(
        pipeline_router._has_complete_plan_unit_texture_cache(
            manager,
            session_id,
            session_dir,
            session_dir / "cache" / "texture_plan.json",
        )
    )


def test_plan_cache_validation_rejects_decompression_bomb_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    _seed_flat_plan_unit_textures(session_dir, plan)
    monkeypatch.setattr(cached_apply.Image, "MAX_IMAGE_PIXELS", 3)

    with pytest.warns(cached_apply.Image.DecompressionBombWarning):
        assert not asyncio.run(
            pipeline_router._has_complete_plan_unit_texture_cache(
                manager,
                session_id,
                session_dir,
                session_dir / "cache" / "texture_plan.json",
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "error",
            cached_apply.Image.DecompressionBombWarning,
        )
        assert not asyncio.run(
            pipeline_router._has_complete_plan_unit_texture_cache(
                manager,
                session_id,
                session_dir,
                session_dir / "cache" / "texture_plan.json",
            )
        )


def test_targeted_regenerate_requires_existing_plan(tmp_path: Path) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={
            "steps": ["generate_textures"],
            "texture_unit_ids": ["tu_0123456789abcdefabcd"],
        },
    )

    assert response.status_code == 409
    assert "texture_plan.json" in response.json()["detail"]


@pytest.mark.parametrize(
    ("requested_steps", "expected_steps"),
    [
        (
            ["apply_textures"],
            [
                "prepare_uvs",
                "discover_materials",
                "plan_textures",
                "generate_prompts",
                "apply_textures",
            ],
        ),
        (
            ["apply_textures", "prepare_uvs"],
            [
                "prepare_uvs",
                "discover_materials",
                "plan_textures",
                "generate_prompts",
                "apply_textures",
            ],
        ),
    ],
)
def test_apply_textures_regenerate_prepends_and_deduplicates_prepare_uvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_steps: list[str],
    expected_steps: list[str],
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": requested_steps},
    )

    assert response.status_code == 202, response.text
    assert captured["only_steps"] == expected_steps
    planning = captured["config_dict"]["planning"]
    assert planning["resume_apply_textures"] is True
    assert planning["apply_texture_plan_unit_ids"] is False


def test_apply_textures_regenerate_respects_disabled_implicit_prepare_uvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
    stored_config = pipeline_router.yaml.safe_load(config_path.read_text())
    stored_config.setdefault("steps", {})["prepare_uvs"] = {"enabled": False}
    config_path.write_text(
        pipeline_router.yaml.safe_dump(stored_config),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert captured["only_steps"] == [
        "discover_materials",
        "plan_textures",
        "generate_prompts",
        "apply_textures",
    ]


def test_apply_textures_regenerate_rejects_explicit_disabled_prepare_uvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
    stored_config = pipeline_router.yaml.safe_load(config_path.read_text())
    stored_config.setdefault("steps", {})["prepare_uvs"] = {"enabled": False}
    config_path.write_text(
        pipeline_router.yaml.safe_dump(stored_config),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures", "prepare_uvs"]},
    )

    assert response.status_code == 422, response.text
    assert "prepare_uvs" in response.json()["detail"]
    assert "registered" not in captured


def test_apply_textures_regenerate_rejects_malformed_cache_key_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    session_dir = manager.get_session_dir(session_id)
    marker_path = session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("{not-json", encoding="utf-8")
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == (
        "Cached apply metadata is malformed or unsupported: "
        "cache/apply_cache_key_mode.json. Restore a valid marker from durable "
        "session state before retrying."
    )
    assert "registered" not in captured
    assert manager.get_session_metadata(session_id)["status"] == "completed"


def test_legacy_apply_cache_marker_survives_compatibility_plan_across_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    sync_prefixes: list[str] = []
    original_sync_to_store = manager.sync_to_store

    def _record_sync_to_store(sid: str, prefix: str = "") -> int:
        sync_prefixes.append(prefix)
        return original_sync_to_store(sid, prefix)

    monkeypatch.setattr(manager, "sync_to_store", _record_sync_to_store)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    first = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert first.status_code == 202, first.text
    first_planning = captured["config_dict"]["planning"]
    assert first_planning["apply_texture_plan_unit_ids"] is False
    assert "plan_textures" in captured["only_steps"]
    marker_key = pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    marker_path = session_dir / marker_key
    assert json.loads(marker_path.read_text(encoding="utf-8")) == {
        "schema_version": pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_SCHEMA,
        "key_mode": pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    }

    # Emulate the compatibility PlanTexturesTask completed by the first run.
    # The second request must still use display-derived cache filenames.
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.write_text(_plan_with_unit().model_dump_json(), encoding="utf-8")
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    second = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert second.status_code == 202, second.text
    second_planning = captured["config_dict"]["planning"]
    assert second_planning["apply_texture_plan_unit_ids"] is False
    assert "plan_textures" not in captured["only_steps"]
    assert sync_prefixes.count(marker_key) == 2


def test_markerless_legacy_generate_then_apply_retains_display_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generate-only plan must not relabel an existing legacy apply cache."""
    from PIL import Image

    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    for channel in ("albedo", "normal", "orm"):
        with Image.new("RGB", (2, 2), color=(64, 128, 192)) as image:
            image.save(textures_dir / f"Paint_{channel}.png")

    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    generated = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures"]},
    )

    assert generated.status_code == 202, generated.text
    assert "plan_textures" in captured["only_steps"]
    marker_path = session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )

    # Emulate the plan created by the first request. Its legacy blended maps
    # remain display-keyed because blend/apply did not run.
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(_plan_with_unit().model_dump_json(), encoding="utf-8")
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    applied = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert applied.status_code == 202, applied.text
    planning = captured["config_dict"]["planning"]
    assert planning["apply_texture_plan_unit_ids"] is False
    assert "plan_textures" not in captured["only_steps"]
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )


def test_markerless_modern_apply_requires_complete_plan_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    _seed_flat_plan_unit_textures(
        session_dir,
        _plan_with_unit(),
        channels=("albedo", "normal"),
    )
    # Stale display-key files must not turn an indeterminate modern cache into
    # an affirmative legacy-mode decision.
    from PIL import Image

    textures_dir = session_dir / "cache" / "textures"
    for channel in ("albedo", "normal", "orm"):
        with Image.new("RGB", (2, 2), color=(192, 64, 128)) as image:
            image.save(textures_dir / f"Paint_{channel}.png")

    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 409, response.text
    assert "cannot be determined safely" in response.json()["detail"]
    assert "registered" not in captured
    assert not (session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY).exists()


def test_markerless_modern_apply_adopts_complete_plan_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    _seed_flat_plan_unit_textures(session_dir, _plan_with_unit())
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True
    marker_path = session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )


def test_markerless_modern_apply_fails_closed_on_store_listing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    _seed_flat_plan_unit_textures(session_dir, _plan_with_unit())
    monkeypatch.setattr(manager, "uses_shared_store", lambda: True)

    def _raise_store_error(*args: Any, **kwargs: Any) -> list[str]:
        raise RuntimeError("shared store unavailable")

    monkeypatch.setattr(manager, "list_store_keys", _raise_store_error)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 409, response.text
    assert "cannot be determined safely" in response.json()["detail"]
    assert "registered" not in captured
    assert not (session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY).exists()


def test_legacy_apply_only_hydrates_hard_cap_without_authorizing_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 64-unit legacy cache applies, but the plan cannot authorize generation."""
    from texture_agent.functions.material_discovery import MaterialInfo
    from texture_agent.tasks.apply_textures import ApplyTexturesTask
    from texture_agent.tasks.discover_materials import DiscoverMaterialsTask
    from texture_agent.tasks.execute_texture_plan import ExecuteTexturePlanTask

    from ...service.runtime.bus import init_event_bus
    from ...service.workers import executor as texture_executor

    manager, session_id = _seed_regenerate_session(tmp_path)
    session_dir = manager.get_session_dir(session_id)
    config_path = session_dir / "input" / "config.yaml"
    stored_config = pipeline_router.yaml.safe_load(config_path.read_text())
    stored_config.setdefault("steps", {})["prepare_uvs"] = {"enabled": False}
    config_path.write_text(
        pipeline_router.yaml.safe_dump(stored_config),
        encoding="utf-8",
    )
    cached_prompts = {
        f"Paint{index:02d}": {
            "prompt": f"cached painted surface {index}",
            "opacity": 1.0,
        }
        for index in range(64)
    }
    prompt_path = session_dir / "cache" / "prompts" / "material_prompts.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(json.dumps(cached_prompts), encoding="utf-8")
    materials = [
        MaterialInfo(
            prim_path=f"/World/Looks/Paint{index:02d}",
            name=f"Paint{index:02d}",
            bound_prim_paths=[f"/World/Mesh{index:02d}"],
        )
        for index in range(64)
    ]
    observed_unit_counts: list[int] = []
    observed_unit_limits: list[int] = []
    generation_attempted = False

    def _discover(
        self: DiscoverMaterialsTask,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        context["discovered_materials"] = materials
        return context

    def _apply(
        self: ApplyTexturesTask,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        observed_unit_counts.append(len(context.get("prim_texture_units") or []))
        observed_unit_limits.append(context["texture_config"]["max_texture_units"])
        output_path = Path(context["working_dir"]) / "output" / "compat.usda"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        context["output_usd_paths"] = [str(output_path)]
        return context

    def _generate(
        self: ExecuteTexturePlanTask,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        nonlocal generation_attempted
        generation_attempted = True
        return context

    monkeypatch.setattr(DiscoverMaterialsTask, "run", _discover)
    monkeypatch.setattr(ApplyTexturesTask, "run", _apply)
    monkeypatch.setattr(ExecuteTexturePlanTask, "run", _generate)
    monkeypatch.setattr(texture_executor, "_package_usdz", lambda *args: None)
    init_event_bus(manager).clear_session_state(session_id)

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
            except Exception:
                # The production registry runs in the background. Keep this
                # synchronous test equivalent while asserting metadata below.
                pass
            finally:
                if on_finished is not None:
                    on_finished()

    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: _ImmediateRegistry(),
    )
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    response = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert manager.get_session_metadata(session_id)["status"] == "completed"
    assert observed_unit_counts == [64]
    assert observed_unit_limits == [64]
    stored_config = pipeline_router.yaml.safe_load(config_path.read_text())
    assert "planning" not in stored_config

    plan = TexturePlan.model_validate_json(
        (session_dir / "cache" / "texture_plan.json").read_text(encoding="utf-8")
    )
    assert plan.counts.selected_unit_count == 64
    assert plan.limits.hard_cap == 64
    assert plan.limits.operator_override_cap is None
    assert plan.decision.execution_allowed is False

    # The compatibility flag is ephemeral. A later generation request loads
    # the rejected plan and fails at the normal executable-plan gate before
    # the image-generation task can run.
    manager.update_session(session_id, {"status": "completed"})
    init_event_bus(manager).clear_session_state(session_id)
    generation = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures"]},
    )

    assert generation.status_code == 202, generation.text
    assert manager.get_session_metadata(session_id)["status"] == "failed"
    assert generation_attempted is False


def test_legacy_apply_cache_promotes_only_after_complete_plan_unit_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    _seed_flat_plan_unit_textures(session_dir, plan)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    promoted = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert promoted.status_code == 202, promoted.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )

    # Promotion is monotonic. Losing a stable artifact must fail closed in the
    # executor, not silently fall back to any stale display-key cache.
    unit_id = plan.selected_units[0].unit_id
    (session_dir / "cache" / "textures" / f"{unit_id}_orm.png").unlink()
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    after_cache_loss = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert after_cache_loss.status_code == 202, after_cache_loss.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )

    # The plan-mode decision is durable independently of the plan artifact.
    # If the plan is lost, regeneration must rebuild it while preserving
    # stable IDs instead of downgrading to any stale display-key cache.
    (session_dir / "cache" / "texture_plan.json").unlink()
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    after_plan_loss = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert after_plan_loss.status_code == 202, after_plan_loss.text
    assert "plan_textures" in captured["only_steps"]
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )


def test_legacy_apply_cache_promotion_uses_durable_shared_store_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    _seed_flat_plan_unit_textures(session_dir, plan)
    durable_keys: list[str] = []
    monkeypatch.setattr(manager, "uses_shared_store", lambda: True)
    monkeypatch.setattr(
        manager,
        "list_store_keys",
        lambda sid, prefix="": list(durable_keys),
    )

    def _unexpected_remote_reopen(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hydrated cache validation must not reopen remote files")

    monkeypatch.setattr(
        manager,
        "open_store_stream",
        _unexpected_remote_reopen,
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    stale_local_only = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert stale_local_only.status_code == 202, stale_local_only.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )

    durable_keys.extend(
        f"cache/textures/{unit.unit_id}_{channel}.png"
        for unit in plan.selected_units
        for channel in ("albedo", "normal", "orm")
    )
    unit_id = plan.selected_units[0].unit_id
    corrupt_key = f"cache/textures/{unit_id}_normal.png"
    # Model a corrupt object in the snapshot that regenerate_pipeline just
    # hydrated. A stale/local-only key still cannot authorize promotion because
    # durable membership is checked independently above.
    (session_dir / corrupt_key).write_bytes(b"")
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    durable_corrupt = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert durable_corrupt.status_code == 202, durable_corrupt.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )

    _seed_flat_plan_unit_textures(session_dir, plan, channels=("normal",))
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    durable_complete = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert durable_complete.status_code == 202, durable_complete.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )


def test_plan_apply_cache_revalidates_durable_shared_store_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    plan = _plan_with_unit()
    _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN,
    )
    _seed_flat_plan_unit_textures(session_dir, plan)
    durable_keys = [
        f"cache/textures/{unit.unit_id}_{channel}.png"
        for unit in plan.selected_units
        for channel in ("albedo", "normal", "orm")
    ]
    monkeypatch.setattr(manager, "uses_shared_store", lambda: True)
    monkeypatch.setattr(
        manager,
        "list_store_keys",
        lambda sid, prefix="": list(durable_keys),
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    complete = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert complete.status_code == 202, complete.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is True

    # Model remote loss after the worker hydrated this cache. The local file
    # remains valid, but it must no longer authorize plan-mode cached apply.
    lost_key = durable_keys.pop()
    assert (session_dir / lost_key).is_file()
    manager.update_session(session_id, {"status": "completed"})
    captured.clear()

    after_remote_loss = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert after_remote_loss.status_code == 409, after_remote_loss.text
    assert "durable session storage" in after_remote_loss.json()["detail"]
    assert "registered" not in captured


@pytest.mark.parametrize(
    "channels",
    [
        (),
        ("albedo", "normal"),
    ],
    ids=["failed-no-stable-output", "partial-stable-output"],
)
def test_legacy_apply_cache_retains_display_keys_for_incomplete_plan_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels: tuple[str, ...],
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    _seed_flat_plan_unit_textures(
        session_dir,
        _plan_with_unit(),
        channels=channels,
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )


@pytest.mark.parametrize(
    "corrupt_payload",
    [b"", b"not a png"],
    ids=["zero-byte", "invalid-png"],
)
def test_legacy_apply_cache_retains_display_keys_for_corrupt_plan_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_payload: bytes,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    session_dir = manager.get_session_dir(session_id)
    marker_path = _write_apply_cache_key_mode(
        session_dir,
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    plan = _plan_with_unit()
    _seed_flat_plan_unit_textures(session_dir, plan)
    unit_id = plan.selected_units[0].unit_id
    (session_dir / "cache" / "textures" / f"{unit_id}_orm.png").write_bytes(
        corrupt_payload
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    assert captured["config_dict"]["planning"]["apply_texture_plan_unit_ids"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY
    )


def test_apply_textures_regenerate_executes_from_seeded_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply-only regeneration rebuilds units, UVs, and a portable graph."""
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from ...service.runtime.bus import init_event_bus

    manager = SessionManager(tmp_path / "sessions", ttl_hours=2)
    session_id = "apply-regenerate-seeded-cache"
    session_dir = manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"

    stage = Usd.Stage.CreateNew(str(input_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Paint/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.3, 0.4)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    # Keep a second effectively-bound material outside the immutable plan. If
    # apply regeneration reaches GeneratePromptsTask without early resume/plan
    # scope, auto-prompting would try to contact an LLM for this material.
    copper_mesh = UsdGeom.Mesh.Define(stage, "/World/CopperMesh")
    copper_mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(2.0, 0.0, 0.0),
            Gf.Vec3f(3.0, 0.0, 0.0),
            Gf.Vec3f(3.0, 1.0, 0.0),
            Gf.Vec3f(2.0, 1.0, 0.0),
        ]
    )
    copper_mesh.GetFaceVertexCountsAttr().Set([4])
    copper_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    copper = UsdShade.Material.Define(stage, "/World/Looks/Copper")
    copper_surface = UsdShade.Shader.Define(stage, "/World/Looks/Copper/Surface")
    copper_surface.CreateIdAttr("UsdPreviewSurface")
    copper_surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.7, 0.3, 0.1)
    )
    copper.CreateSurfaceOutput().ConnectToSource(
        copper_surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(copper_mesh.GetPrim()).Bind(copper)
    stage.GetRootLayer().Save()

    config = {
        "project": {"session_id": session_id},
        "input": {"usd_path": str(input_path)},
        "texture": {
            "uv_policy": "generate_missing",
            "uv_projection": "planar",
        },
        "material_textures": {
            "Paint": {"prompt": "matte blue painted label", "opacity": 1.0}
        },
        "auto_prompt": {"enabled": True},
        "steps": {
            "render_previews": {"enabled": False},
            "render": {"enabled": False},
        },
    }
    (session_dir / "input" / "config.yaml").write_text(
        pipeline_router.yaml.safe_dump(config),
        encoding="utf-8",
    )

    plan = _plan_with_unit()
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    prompt_path = session_dir / "cache" / "prompts" / "material_prompts.json"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        json.dumps(config["material_textures"]),
        encoding="utf-8",
    )

    unit_id = plan.selected_units[0].unit_id
    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    for channel, color in {
        "albedo": (25, 90, 210),
        "normal": (128, 128, 255),
        "orm": (255, 120, 0),
    }.items():
        Image.new("RGB", (8, 8), color).save(textures_dir / f"{unit_id}_{channel}.png")

    manager.update_session(session_id, {"status": "completed"})
    init_event_bus(manager).clear_session_state(session_id)

    from texture_agent.tasks.generate_prompts import GeneratePromptsTask
    from world_understanding.functions.models import chat_models

    prompt_resume_values: list[bool] = []
    original_generate_prompts_run = GeneratePromptsTask.run

    def _record_generate_prompts_resume(
        self: GeneratePromptsTask,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        prompt_resume_values.append(bool(context.get("resume")))
        return original_generate_prompts_run(self, context, object_store)

    def _forbid_prompt_backend(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("apply-only regeneration must not contact the prompt backend")

    monkeypatch.setattr(GeneratePromptsTask, "run", _record_generate_prompts_resume)
    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        _forbid_prompt_backend,
    )

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
        pipeline_router,
        "get_job_registry",
        lambda: _ImmediateRegistry(),
    )
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert response.status_code == 202, response.text
    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "completed"
    assert prompt_resume_values == [True]
    output_usd = session_dir / "cache" / "output" / "textured_output.usd"
    output_usdz = session_dir / "cache" / "output" / "textured_output.usdz"
    assert output_usd.is_file()
    assert output_usdz.is_file()

    downloaded_stage = Usd.Stage.Open(str(output_usdz))
    assert downloaded_stage is not None
    downloaded_mesh = downloaded_stage.GetPrimAtPath("/World/Mesh")
    st = UsdGeom.PrimvarsAPI(downloaded_mesh).GetPrimvar("st")
    assert st.HasAuthoredValue()
    assert st.GetInterpolation() == "faceVarying"
    assert len(st.ComputeFlattened()) == 4

    bound_material, _ = UsdShade.MaterialBindingAPI(
        downloaded_mesh
    ).ComputeBoundMaterial()
    surface_source = bound_material.GetSurfaceOutput().GetConnectedSource()
    assert surface_source is not None
    downloaded_surface = UsdShade.Shader(surface_source[0].GetPrim())
    texture_source = downloaded_surface.GetInput("diffuseColor").GetConnectedSource()
    assert texture_source is not None
    texture = UsdShade.Shader(texture_source[0].GetPrim())
    assert texture.GetIdAttr().Get() == "UsdUVTexture"
    assert Path(texture.GetInput("file").Get().path).name == f"{unit_id}_albedo.png"
    st_source = texture.GetInput("st").GetConnectedSource()
    assert st_source is not None
    assert UsdShade.Shader(st_source[0].GetPrim()).GetIdAttr().Get() == (
        "UsdPrimvarReader_float2"
    )


def test_legacy_generate_blend_apply_executes_with_plan_unit_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-plan combined request keeps its freshly generated stable maps."""
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from texture_agent.functions.texture_generation import GeneratedTextures
    from texture_agent.tasks.generate_textures import GenerateTexturesTask

    from ...service.runtime.bus import init_event_bus

    shared_store = LocalSessionStore(str(tmp_path / "shared-store"))
    manager = SessionManager(
        tmp_path / "pod-a",
        ttl_hours=2,
        store=shared_store,
    )
    replacement_manager = SessionManager(
        tmp_path / "pod-b",
        ttl_hours=2,
        store=shared_store,
    )
    session_id = "legacy-generate-blend-apply"
    session_dir = manager.create_session(session_id)
    input_path = session_dir / "input" / "scene.usda"

    stage = Usd.Stage.CreateNew(str(input_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    surface = UsdShade.Shader.Define(stage, "/World/Looks/Paint/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.3, 0.4)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    config = {
        "project": {"session_id": session_id},
        "input": {"usd_path": str(input_path)},
        "texture": {
            "backend": "simple_image_gen",
            "size": 8,
            "uv_policy": "generate_missing",
            "uv_projection": "planar",
        },
        "material_textures": {
            "Paint": {"prompt": "matte blue painted label", "opacity": 1.0}
        },
        "auto_prompt": {"enabled": False},
        "steps": {
            "blend_textures": {"enabled": True, "output_size": 8},
            "render_previews": {"enabled": False},
            "render": {"enabled": False},
        },
    }
    (session_dir / "input" / "config.yaml").write_text(
        pipeline_router.yaml.safe_dump(config),
        encoding="utf-8",
    )
    # Model stale pre-plan blended maps that remain available under the old
    # display-derived key. A later cache loss must never fall back to these.
    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    for channel in ("albedo", "normal", "orm"):
        Image.new("RGB", (8, 8), (240, 10, 10)).save(
            textures_dir / f"Paint_{channel}.png"
        )
    manager.update_session(session_id, {"status": "completed"})
    init_event_bus(manager).clear_session_state(session_id)

    generated_unit_keys: list[str] = []

    def _generate_local_maps(
        self: GenerateTexturesTask,
        context: dict[str, Any],
        object_store: Any = None,
    ) -> dict[str, Any]:
        unit = context["prim_texture_units"][0]
        generated_unit_keys.append(unit.key)
        generated_dir = Path(context["working_dir"]) / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for channel, color in {
            "albedo": (25, 90, 210),
            "normal": (128, 128, 255),
            "orm": (255, 120, 0),
        }.items():
            path = generated_dir / f"{unit.key}_{channel}.png"
            Image.new("RGB", (8, 8), color).save(path)
            paths[channel] = str(path)
        context["generated_textures"] = {
            unit.key: GeneratedTextures(**paths),
        }
        context["generate_textures_errors"] = []
        context["generate_textures_failed_count"] = 0
        context["generate_textures_attempted_count"] = 1
        return context

    monkeypatch.setattr(GenerateTexturesTask, "run", _generate_local_maps)

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
        pipeline_router,
        "get_job_registry",
        lambda: _ImmediateRegistry(),
    )
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    client = TestClient(app)

    response = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures", "blend_textures", "apply_textures"]},
    )

    assert response.status_code == 202, response.text
    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "completed"
    plan = TexturePlan.model_validate_json(
        (session_dir / "cache" / "texture_plan.json").read_text(encoding="utf-8")
    )
    unit_id = plan.selected_units[0].unit_id
    assert generated_unit_keys == [unit_id]
    for channel in ("albedo", "normal", "orm"):
        assert (
            session_dir / "cache" / "textures" / f"{unit_id}_{channel}.png"
        ).is_file()

    marker_path = session_dir / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )
    assert (
        replacement_manager.sync_from_store(
            session_id,
            pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY,
        )
        == 1
    )
    replacement_marker = (
        replacement_manager.get_session_dir(session_id)
        / pipeline_router._APPLY_CACHE_KEY_MODE_MARKER_KEY
    )
    assert json.loads(replacement_marker.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )
    output_usdz = session_dir / "cache" / "output" / "textured_output.usdz"
    assert output_usdz.is_file()
    downloaded_stage = Usd.Stage.Open(str(output_usdz))
    assert downloaded_stage is not None
    downloaded_mesh = downloaded_stage.GetPrimAtPath("/World/Mesh")
    bound_material, _ = UsdShade.MaterialBindingAPI(
        downloaded_mesh
    ).ComputeBoundMaterial()
    surface_source = bound_material.GetSurfaceOutput().GetConnectedSource()
    assert surface_source is not None
    downloaded_surface = UsdShade.Shader(surface_source[0].GetPrim())
    texture_source = downloaded_surface.GetInput("diffuseColor").GetConnectedSource()
    assert texture_source is not None
    texture = UsdShade.Shader(texture_source[0].GetPrim())
    assert texture.GetIdAttr().Get() == "UsdUVTexture"
    assert Path(texture.GetInput("file").Get().path).name == f"{unit_id}_albedo.png"

    # Remove one durable plan-keyed artifact while leaving both this worker's
    # hydrated copy and the complete stale legacy triplet intact. Plan-mode
    # apply must reject the durable loss instead of consuming either cache.
    lost_key = f"cache/textures/{unit_id}_orm.png"
    assert (session_dir / lost_key).is_file()
    assert shared_store.exists(session_id, "cache/textures/Paint_orm.png")
    shared_store.delete_key(session_id, lost_key)

    after_plan_cache_loss = client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["apply_textures"]},
    )

    assert after_plan_cache_loss.status_code == 409, after_plan_cache_loss.text
    assert "durable session storage" in after_plan_cache_loss.json()["detail"]
    assert (session_dir / lost_key).is_file()
    assert json.loads(marker_path.read_text(encoding="utf-8"))["key_mode"] == (
        pipeline_router._APPLY_CACHE_KEY_MODE_PLAN
    )


def test_targeted_regenerate_rejects_unit_ids_outside_plan(tmp_path: Path) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={
            "steps": ["generate_textures"],
            "texture_unit_ids": ["tu_0123456789abcdefabcd"],
        },
    )

    assert response.status_code == 422
    assert "outside the approved texture plan" in response.json()["detail"]


def test_targeted_regenerate_merges_cached_prompts_and_exact_unit_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=True,
        include_prompts=True,
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)
    unit_id = _plan_with_unit().selected_units[0].unit_id

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures"], "texture_unit_ids": [unit_id]},
    )

    assert response.status_code == 202, response.text
    assert response.json()["plan_url"] == f"/pipeline/{session_id}/plan"
    assert captured["only_steps"] == [
        "discover_materials",
        "generate_prompts",
        "generate_textures",
    ]
    planning = captured["config_dict"]["planning"]
    assert planning["resume_execution"] is True
    assert planning["regenerate_unit_ids"] == [unit_id]
    assert captured["config_dict"]["material_textures"] == {
        "Paint": {"prompt": "operator prompt"},
        "Copper": {"prompt": "cached copper"},
    }


@pytest.mark.parametrize(
    ("include_plan", "marker_mode"),
    [
        (False, None),
        (True, pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY),
        (True, None),
    ],
    ids=["pre-plan", "existing-plan-legacy", "existing-plan-markerless"],
)
def test_full_generate_blend_schedules_post_sync_cache_mode_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_plan: bool,
    marker_mode: str | None,
) -> None:
    manager, session_id = _seed_regenerate_session(
        tmp_path,
        include_plan=include_plan,
    )
    if marker_mode is not None:
        _write_apply_cache_key_mode(manager.get_session_dir(session_id), marker_mode)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures", "blend_textures"]},
    )

    assert response.status_code == 202, response.text
    assert callable(captured["on_artifacts_synced"])


def test_targeted_generate_blend_does_not_schedule_cache_mode_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path, include_plan=True)
    _write_apply_cache_key_mode(
        manager.get_session_dir(session_id),
        pipeline_router._APPLY_CACHE_KEY_MODE_LEGACY,
    )
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={
            "steps": ["generate_textures", "blend_textures"],
            "texture_unit_ids": [_plan_with_unit().selected_units[0].unit_id],
        },
    )

    assert response.status_code == 202, response.text
    assert captured["on_artifacts_synced"] is None


def test_legacy_regenerate_without_plan_rebuilds_plan_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id = _seed_regenerate_session(tmp_path)
    captured: dict[str, Any] = {}
    _install_regenerate_stubs(monkeypatch, captured)
    pipeline_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(pipeline_router.router)

    response = TestClient(app).post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["generate_textures"]},
    )

    assert response.status_code == 202, response.text
    assert captured["only_steps"] == [
        "discover_materials",
        "plan_textures",
        "generate_prompts",
        "generate_textures",
    ]
