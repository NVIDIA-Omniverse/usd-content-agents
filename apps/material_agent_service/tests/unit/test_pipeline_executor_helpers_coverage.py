# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for pipeline-router and executor helper branches."""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from ...service import config as config_module
from ...service.artifact_lineage import initial_artifact_validity
from ...service.events import listener as listener_module
from ...service.events.listener import FastAPIEventListener
from ...service.models.requests import PipelineStep, RegenerateRequest
from ...service.routers import pipeline_router
from ...service.runtime.bus import EventBus
from ...service.runtime.events import ProgressEvent, StepState
from ...service.session.manager import SessionManager
from ...service.storage.local_store import LocalSessionStore
from ...service.workers import executor


def _expect_http(
    status_code: int, exc_info: pytest.ExceptionInfo[HTTPException]
) -> None:
    assert exc_info.value.status_code == status_code


def _upload(filename: str, data: bytes = b"#usda 1.0\n") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


async def _response_body(response: Any) -> bytes:
    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await response(
        {"type": "http", "method": "GET", "headers": [], "extensions": {}},
        receive,
        send,
    )
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message.get("type") == "http.response.body"
    )


def _direct_pipeline_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for name, parameter in inspect.signature(
        pipeline_router.create_pipeline
    ).parameters.items():
        default = parameter.default
        if hasattr(default, "default"):
            value = default.default
        else:
            value = default
        if isinstance(value, list):
            value = list(value)
        kwargs[name] = value
    kwargs.update(overrides)
    return kwargs


def _minimal_pipeline_config(
    *,
    project_name: str,
    input_usd_path: str,
    output_usd_path: str,
    enabled_steps: list[str],
    working_dir: str,
    materials_entries: list[dict] | None = None,
    **_: object,
) -> dict[str, object]:
    steps: dict[str, dict[str, object]] = {}
    for step in enabled_steps:
        if step == "build_dataset_usd":
            steps[step] = {
                "renderer": {},
                "num_workers": 1,
                "max_concurrent_requests": 1,
            }
        elif step == "build_dataset_prepare_dataset":
            steps[step] = {"prompts": {}}
        else:
            steps[step] = {}
    return {
        "project": {"name": project_name},
        "input": {"usd_path": input_usd_path},
        "output": {"usd_path": output_usd_path},
        "materials": {"entries": materials_entries or []},
        "steps": steps,
        "working_dir": working_dir,
    }


class _ClosingRegistry:
    def __init__(self) -> None:
        self.registered: list[tuple[str, object]] = []

    async def register(
        self,
        session_id: str,
        coro: object,
        *,
        before_start=None,
    ) -> None:
        self.registered.append((session_id, coro))
        if before_start is not None:
            await before_start()
        close = getattr(coro, "close", None)
        if close:
            close()

    def is_running(self, session_id: str) -> bool:
        return False

    async def cancel(self, session_id: str) -> bool:
        return False


class _NoopTask:
    def cancel(self) -> None:
        return None


def _install_direct_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> _ClosingRegistry:
    import material_agent.api as material_api

    registry = _ClosingRegistry()
    monkeypatch.setattr(
        material_api, "build_unified_pipeline_config", _minimal_pipeline_config
    )
    monkeypatch.setattr(
        pipeline_router,
        "get_stage_info_from_path",
        lambda path: {"prim_count": 0},
    )
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    async def noop_execute(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(pipeline_router, "execute_pipeline_async", noop_execute)
    monkeypatch.setattr(pipeline_router, "execute_scene_pipeline_async", noop_execute)
    return registry


def _create_task_and_close(coro: object) -> _NoopTask:
    close = getattr(coro, "close", None)
    if close:
        close()
    return _NoopTask()


def test_pipeline_model_routing_helper_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "vlm_temperature", 0.2)
    monkeypatch.setattr(pipeline_router.config, "vlm_max_tokens", 128)
    monkeypatch.setattr(pipeline_router.config, "llm_temperature", 0.1)
    monkeypatch.setattr(pipeline_router.config, "llm_max_tokens", 64)
    monkeypatch.setattr(
        pipeline_router.config,
        "vlm_backend_options",
        {"provider_option": "test-value"},
    )

    config = {
        "steps": {
            "build_dataset_prepare_dataset": {
                "prompts": {"vlm_system": "<reasoning>think</reasoning>"}
            },
            "predict": {"llm": {"api_key": "stale"}},
        }
    }
    routing = pipeline_router._ModelRouting(
        vlm_backend="nim",
        vlm_model="nvidia/cosmos-reason2-8b",
        vlm_nim_base_url="http://vlm.local/v1",
        llm_backend="openai",
        llm_model="gpt",
        llm_nim_base_url=None,
        llm_uses_vlm_sidecar=False,
        llm_base_url="https://llm.example/v1",
        llm_api_key="llm-key",
    )

    pipeline_router._configure_predict_model_routing(config, routing)
    predict = config["steps"]["predict"]
    assert predict["vlm"]["base_url"] == "http://vlm.local/v1"
    assert predict["vlm"]["reasoning_budget"] == 16384
    assert (
        config["steps"]["build_dataset_prepare_dataset"]["prompts"]["vlm_system"]
        == "<thinking>think</thinking>"
    )
    assert predict["llm"]["base_url"] == "https://llm.example/v1"
    assert predict["llm"]["api_key"] == "llm-key"

    api_key_routing = routing._replace(
        vlm_backend="openai",
        vlm_nim_base_url=None,
        vlm_base_url=None,
        vlm_api_key="vlm-key",
        llm_base_url=None,
        llm_api_key=None,
        llm_api_key_env="LLM_KEY",
    )
    api_key_config = {"steps": {"predict": {"llm": {}}}}
    pipeline_router._configure_predict_model_routing(api_key_config, api_key_routing)
    assert api_key_config["steps"]["predict"]["vlm"]["api_key"] == "vlm-key"
    assert api_key_config["steps"]["predict"]["llm"]["api_key_env"] == "${LLM_KEY}"

    no_predict = {"steps": {}}
    pipeline_router._configure_predict_model_routing(no_predict, routing)
    assert no_predict == {"steps": {}}

    assert pipeline_router._build_service_vlm_config(routing)["base_url"] == (
        "http://vlm.local/v1"
    )
    llm_config = pipeline_router._build_service_llm_config(routing, temperature=0.5)
    assert llm_config["temperature"] == 0.5
    assert llm_config["api_key"] == "llm-key"
    nim_llm = pipeline_router._build_service_llm_config(
        routing._replace(llm_nim_base_url="http://llm.local/v1")
    )
    assert nim_llm["backend"] == "nim"
    env_llm = pipeline_router._build_service_llm_config(api_key_routing)
    assert env_llm["api_key_env"] == "${LLM_KEY}"

    scene_config = {"scene": [], "steps": {}}
    result = pipeline_router._configure_scene_model_routing(scene_config, routing)
    assert result["analyze"]["llm"]["base_url"] == "https://llm.example/v1"
    scene_config = {"scene": {"analyze": []}, "steps": {}}
    result = pipeline_router._configure_scene_model_routing(scene_config, routing)
    assert isinstance(result["analyze"], dict)

    material_config: dict = {}
    pipeline_router._configure_generate_material_library_step(
        material_config,
        routing,
        material_generation_guidance="  orange plastic  ",
        material_generation_texture_size=256,
    )
    generated = material_config["steps"]["generate_material_library"]
    assert generated["material_guidance"] == "orange plastic"
    assert generated["texture_generation"]["texture_size"] == 256


def test_pipeline_small_parsers_and_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pipeline_router._coerce_positive_int("bad", 4) == 4
    assert pipeline_router._coerce_positive_int(None, 4) == 4
    assert pipeline_router._coerce_positive_int("-5", 4) == 1
    assert pipeline_router._parse_positive_int_form("count", None, 3) == 3
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._parse_positive_int_form("count", 0, 3)
    _expect_http(400, exc_info)

    assert (
        pipeline_router._build_cluster_complexity_thresholds(
            low=None, medium=None, high=None
        )
        is None
    )
    thresholds = pipeline_router._build_cluster_complexity_thresholds(
        low=0.91, medium=None, high=0.81
    )
    assert thresholds["low"][2] == 0.91
    assert thresholds["high"][2] == 0.81
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._build_cluster_complexity_thresholds(
            low=1.5, medium=None, high=None
        )
    _expect_http(400, exc_info)

    monkeypatch.setattr(
        pipeline_router.config,
        "cluster_embedding_model",
        "nvidia/custom-model",
    )
    assert pipeline_router._cluster_model_for_backend("nim", "explicit") == "explicit"
    assert pipeline_router._cluster_model_for_backend("nim", None) == (
        "nvidia/custom-model"
    )
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_model", "")
    assert pipeline_router._cluster_model_for_backend("openai", None)

    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_backend", "   ")
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_base_url", None)
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_api_key", None)
    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_api_key_env", None)
    built = pipeline_router._build_cluster_prims_step_config(
        cluster_min_prims=2,
        cluster_embedding_backend=" ",
        cluster_embedding_model="model",
        cluster_embedding_base_url=None,
        cluster_embedding_max_workers=1,
        cluster_embedding_batch_size=1,
        cluster_max_size=1,
        cluster_similarity_threshold_low=None,
        cluster_similarity_threshold_medium=None,
        cluster_similarity_threshold_high=None,
        cluster_report="false",
    )
    assert built["min_prims_to_activate"] == 2

    monkeypatch.setattr(pipeline_router.config, "cluster_embedding_backend", "nim")
    monkeypatch.setattr(
        pipeline_router.config,
        "cluster_embedding_base_url",
        "http://remote.internal/v1",
    )
    monkeypatch.setattr(pipeline_router, "is_local_base_url", lambda url: False)
    monkeypatch.setattr(
        pipeline_router, "is_nvidia_provider_base_url", lambda url: False
    )
    with pytest.raises(HTTPException):
        pipeline_router._build_cluster_prims_step_config(
            cluster_min_prims=None,
            cluster_embedding_backend=None,
            cluster_embedding_model=None,
            cluster_embedding_base_url=None,
            cluster_embedding_max_workers=None,
            cluster_embedding_batch_size=None,
            cluster_max_size=None,
            cluster_similarity_threshold_low=None,
            cluster_similarity_threshold_medium=None,
            cluster_similarity_threshold_high=None,
            cluster_report="true",
        )

    assert pipeline_router._normalize_optional_url("  http://x  ") == "http://x"
    assert pipeline_router._normalize_optional_url("   ") is None
    monkeypatch.setattr(
        pipeline_router.config, "cluster_embedding_base_url", "http://configured/v1"
    )
    assert pipeline_router._resolve_cluster_embedding_base_url(None) == (
        "http://configured/v1"
    )
    assert (
        pipeline_router._resolve_cluster_embedding_base_url("http://configured/v1/")
        == "http://configured/v1"
    )
    monkeypatch.setattr(
        pipeline_router, "is_nvidia_provider_base_url", lambda url: True
    )
    assert pipeline_router._resolve_cluster_embedding_base_url("https://hosted") == (
        "https://hosted"
    )
    monkeypatch.setattr(
        pipeline_router, "is_nvidia_provider_base_url", lambda url: False
    )
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._resolve_cluster_embedding_base_url("http://untrusted/v1")
    _expect_http(400, exc_info)

    assert pipeline_router._parse_bool_form("YES")
    assert not pipeline_router._parse_bool_form(None)
    assert pipeline_router._parse_csv_form(" a, ,b ") == ["a", "b"]
    assert pipeline_router._parse_json_object_form("", "filters") is None
    assert pipeline_router._parse_json_object_form('{"a": 1}', "filters") == {"a": 1}
    for value in ("[1]", "{"):
        with pytest.raises(HTTPException) as exc_info:
            pipeline_router._parse_json_object_form(value, "filters")
        _expect_http(400, exc_info)

    assert pipeline_router._parse_iso_datetime("2026-01-01T00:00:00Z").tzinfo
    assert pipeline_router._parse_iso_datetime("2026-01-01T00:00:00").tzinfo
    assert pipeline_router._terminal_metadata_ready(
        {"status": "cancelled", "cancelled_at": "2026-01-01T00:00:00Z"}
    )
    assert not pipeline_router._terminal_metadata_ready({"status": "cancelled"})
    assert not pipeline_router._terminal_metadata_ready({"status": "pending"})
    legacy_metadata = {
        "status": "completed",
        "completed_at": "2026-01-01T00:00:00Z",
        "results": {},
        "config": {"coverage_policy": "invalid"},
    }
    assert (
        pipeline_router.normalize_legacy_completed_coverage(
            legacy_metadata,
            pipeline_active=True,
        )
        is legacy_metadata
    )
    normalized_legacy = pipeline_router.normalize_legacy_completed_coverage(
        legacy_metadata,
        pipeline_active=False,
    )
    assert normalized_legacy["coverage"]["policy"] == "allow_partial"
    assert normalized_legacy["coverage"]["readiness_grade"] == "not_evaluated"
    coverage_metadata = {**legacy_metadata, "coverage": None}
    assert (
        pipeline_router.normalize_legacy_completed_coverage(
            coverage_metadata,
            pipeline_active=False,
        )
        is coverage_metadata
    )
    assert pipeline_router._current_step_with_fresh_elapsed(None) is None
    assert pipeline_router._current_step_with_fresh_elapsed({"started_at": 1}) == {
        "started_at": 1
    }
    assert (
        pipeline_router._current_step_with_fresh_elapsed(
            {"started_at": "not-a-date", "elapsed_seconds": 3}
        )["elapsed_seconds"]
        == 3
    )


@pytest.mark.asyncio
async def test_status_hides_provisional_terminal_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    snapshot = {
        "session_id": "sid",
        "status": "completed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "current_step": None,
        "completed_steps": [],
        "overall_progress": {
            "current_step": 3,
            "total_steps": 3,
            "percent": 100,
        },
        "preview_images": [],
    }
    snapshot_value: dict[str, dict | None] = {"value": snapshot}

    class _Bus:
        async def get_fenced_snapshot(self, session_id: str) -> dict | None:
            return snapshot_value["value"]

    class _Manager:
        async def get_session_metadata(self, session_id: str) -> dict:
            return {
                "status": "completed",
                "created_at": timestamp,
                "updated_at": timestamp,
                "results": {},
                "current_step": None,
                "completed_steps": [],
                "overall_progress": {
                    "current_step": 3,
                    "total_steps": 3,
                    "percent": 100,
                },
            }

    class _Registry:
        def is_running(self, session_id: str) -> bool:
            return True

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _Bus())
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: _Manager())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())

    status = await pipeline_router.get_pipeline_status("sid")

    assert status.status == "running"
    assert status.overall_progress.percent == 100

    snapshot_value["value"] = None
    disk_only_status = await pipeline_router.get_pipeline_status("sid")
    assert disk_only_status.status == "running"


def _completed_step(name: str, timestamp: str) -> dict[str, object]:
    return {
        "name": name,
        "display_name": name.title(),
        "started_at": timestamp,
        "completed_at": timestamp,
        "duration_seconds": 0,
        "stats": {},
    }


@pytest.mark.asyncio
async def test_status_prefers_authoritative_terminal_disk_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    stale_snapshot = {
        "session_id": "sid",
        "status": "completed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "current_step": {
            "name": "apply",
            "display_name": "Apply",
            "started_at": timestamp,
            "progress": {
                "current": 0,
                "total": 1,
                "percent": 0,
                "message": "stale",
            },
            "elapsed_seconds": 0,
        },
        "completed_steps": [_completed_step("predict", timestamp)],
        "overall_progress": {
            "current_step": 1,
            "total_steps": 2,
            "percent": 50,
        },
        "preview_images": ["stale.png"],
    }
    disk_metadata = {
        "status": "completed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "completed_at": timestamp,
        "results": {},
        "coverage": None,
        "current_step": None,
        "completed_steps": [
            _completed_step("predict", timestamp),
            _completed_step("apply", timestamp),
        ],
        "overall_progress": {
            "current_step": 2,
            "total_steps": 2,
            "percent": 100,
        },
        "preview_images": ["final.png"],
    }

    class _Bus:
        async def get_fenced_snapshot(self, session_id: str) -> dict:
            return stale_snapshot

    class _Manager:
        async def get_session_metadata(self, session_id: str) -> dict:
            return disk_metadata

    class _Registry:
        def is_running(self, session_id: str) -> bool:
            return False

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _Bus())
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: _Manager())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())

    status = await pipeline_router.get_pipeline_status("sid")

    assert status.current_step is None
    assert [step.name for step in status.completed_steps] == ["predict", "apply"]
    assert status.overall_progress.current_step == 2
    assert status.overall_progress.percent == 100
    assert status.preview_images == [
        "/assets/sid/preview/final.png",
        "/assets/sid/preview/stale.png",
    ]


@pytest.mark.asyncio
async def test_status_uses_snapshot_progress_when_terminal_disk_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    snapshot = {
        "session_id": "sid",
        "status": "completed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "current_step": None,
        "completed_steps": [_completed_step("predict", timestamp)],
        "overall_progress": {
            "current_step": 1,
            "total_steps": 1,
            "percent": 100,
        },
        "preview_images": ["snapshot.png"],
    }
    disk_metadata = {
        "status": "completed",
        "created_at": timestamp,
        "updated_at": timestamp,
        "completed_at": timestamp,
        "results": {},
        "coverage": None,
    }

    class _Bus:
        async def get_fenced_snapshot(self, session_id: str) -> dict:
            return snapshot

    class _Manager:
        async def get_session_metadata(self, session_id: str) -> dict:
            return disk_metadata

    class _Registry:
        def is_running(self, session_id: str) -> bool:
            return False

    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _Bus())
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: _Manager())
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _Registry())

    status = await pipeline_router.get_pipeline_status("sid")

    assert [step.name for step in status.completed_steps] == ["predict"]
    assert status.overall_progress.percent == 100
    assert status.preview_images == ["/assets/sid/preview/snapshot.png"]


def test_terminal_status_merge_prefers_advanced_large_scene_snapshot() -> None:
    timestamp = "2026-01-01T00:00:00+00:00"
    merged = pipeline_router._merge_terminal_status_metadata(
        {
            "status": "completed",
            "completed_steps": [
                _completed_step("scene_analyze", timestamp),
                _completed_step("scene_collect", timestamp),
            ],
            "overall_progress": {
                "current_step": 2,
                "total_steps": 9,
                "percent": 100,
            },
            "preview_images": [],
        },
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": timestamp,
            "completed_steps": [_completed_step("scene_analyze", timestamp)],
            "overall_progress": {
                "current_step": 0,
                "total_steps": 3,
                "percent": 0,
            },
            "preview_images": [],
        },
    )

    assert [step["name"] for step in merged["completed_steps"]] == [
        "scene_analyze",
        "scene_collect",
    ]
    assert merged["overall_progress"] == {
        "current_step": 2,
        "total_steps": 9,
        "percent": 100,
    }


def test_pipeline_step_injection_and_session_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert pipeline_router._inject_cluster_step(
        ["build_dataset_prepare_dataset", "predict"],
        enable_prim_clustering=True,
    ) == ["build_dataset_prepare_dataset", "cluster_prims", "predict"]
    assert pipeline_router._inject_cluster_step(
        ["cluster_prims", "predict"],
        enable_prim_clustering=True,
        require_prepare_step=False,
    ) == ["cluster_prims", "predict"]
    assert pipeline_router._inject_cluster_step(
        ["benchmark"],
        enable_prim_clustering=True,
        require_prepare_step=False,
    ) == ["cluster_prims", "benchmark"]
    assert pipeline_router._inject_cluster_step(
        ["predict"], enable_prim_clustering=False
    ) == ["predict"]
    with pytest.raises(HTTPException):
        pipeline_router._inject_cluster_step(["apply"], enable_prim_clustering=True)
    with pytest.raises(HTTPException):
        pipeline_router._inject_cluster_step(["predict"], enable_prim_clustering=True)

    no_apply = {"steps": {}}
    pipeline_router._configure_apply_step(
        no_apply, layer_only=False, request_context="test"
    )
    with pytest.raises(HTTPException):
        pipeline_router._configure_apply_step(
            no_apply, layer_only=True, request_context="test"
        )
    apply_config = {"steps": {"apply": {}}}
    pipeline_router._configure_apply_step(
        apply_config, layer_only=True, request_context="test"
    )
    assert apply_config["steps"]["apply"]["flatten_output"] is False

    config = pipeline_router._cluster_session_config_from_step_config(
        enabled=True,
        step_config={
            "min_prims_to_activate": 10,
            "embedding_service": "nim",
            "embedding_model": "embed",
            "base_url": "http://embed",
            "max_workers": 2,
            "batch_size": 8,
            "max_cluster_size": 12,
            "complexity_thresholds": {"low": [0, 0, 0.9], "medium": [0], "high": []},
            "report": False,
        },
    )
    assert config["cluster_similarity_threshold_low"] == 0.9
    assert config["cluster_similarity_threshold_medium"] is None
    assert config["cluster_report"] is False
    assert (
        pipeline_router._cluster_session_config_from_step_config(
            enabled=False, step_config=None
        )["enable_prim_clustering"]
        is False
    )

    monkeypatch.setattr(
        pipeline_router.config, "default_user_email", " fallback@nvidia.com "
    )
    assert pipeline_router._normalize_user_email(" user@nvidia.com ") == (
        "user@nvidia.com"
    )
    assert pipeline_router._normalize_user_email("") == "fallback@nvidia.com"
    monkeypatch.setattr(pipeline_router.config, "default_user_email", "")
    assert pipeline_router._normalize_user_email(None) == "anonymous@nvidia.com"

    assert pipeline_router._insert_step_before(
        ["predict"], "prepare", before_candidates=("predict",)
    ) == ["prepare", "predict"]
    assert pipeline_router._insert_step_before(
        ["prepare"], "prepare", before_candidates=("predict",)
    ) == ["prepare"]
    assert (
        pipeline_router._effective_scene_predict_workers({"steps": {}}, 2, None) is None
    )
    monkeypatch.setattr(pipeline_router.config, "max_scene_vlm_concurrency", 4)
    scene_config = {"steps": {"predict": {"max_workers": 8}}}
    assert pipeline_router._effective_scene_predict_workers(scene_config, 2, None) == 2
    with pytest.raises(HTTPException):
        pipeline_router._effective_scene_predict_workers(
            {"steps": {"predict": {}}}, 3, 2
        )


@pytest.mark.asyncio
async def test_pipeline_file_and_cache_helpers(tmp_path: Path) -> None:
    assert pipeline_router._find_input_usd(tmp_path) is None
    (tmp_path / "input").mkdir()
    scene = tmp_path / "input" / "scene.usda"
    scene.write_text("#usda 1.0\n")
    assert pipeline_router._find_input_usd(tmp_path) == scene

    target = pipeline_router._safe_zip_member_target("nested/file.usda", tmp_path)
    assert target == (tmp_path / "nested" / "file.usda").resolve()
    for name in ("", "../x", "/abs/x", "C:/abs/x", "bad\x00name"):
        with pytest.raises(HTTPException):
            pipeline_router._safe_zip_member_target(name, tmp_path)

    assert pipeline_router._session_files(tmp_path / "missing", "*.png") == []
    file_a = tmp_path / "refs" / "reference_0001.png"
    file_a.parent.mkdir()
    file_a.write_bytes(b"a")
    (tmp_path / "refs" / "reference_dir.png").mkdir()
    assert pipeline_router._session_files(tmp_path / "refs", "reference_*") == [file_a]

    (tmp_path / "refs" / "descriptions.json").write_text('["front"]')
    assert pipeline_router._load_reference_descriptions(tmp_path / "refs") == ["front"]
    (tmp_path / "refs" / "descriptions.json").write_text("{")
    assert pipeline_router._load_reference_descriptions(tmp_path / "refs") == []
    (tmp_path / "refs" / "descriptions.json").write_text("{}")
    assert pipeline_router._load_reference_descriptions(tmp_path / "refs") == []

    manager = SessionManager(tmp_path / "sessions")
    sid = str(uuid4())
    session_dir = await manager.create_session(sid)
    input_render = session_dir / "input" / "input_render.png"
    input_render.write_bytes(b"png")
    assert await pipeline_router._ensure_input_render_local(manager, sid, session_dir)
    input_render.unlink()
    await manager.store.put_bytes(sid, "input/input_render.png", b"png")
    assert await pipeline_router._ensure_input_render_local(manager, sid, session_dir)

    existing = await pipeline_router._restore_existing_session_files(
        manager, sid, session_dir, "input/reference_images", "reference_*"
    )
    assert existing == []
    ref = session_dir / "input" / "reference_images" / "reference_0000.png"
    ref.parent.mkdir(parents=True)
    ref.write_bytes(b"png")
    assert await pipeline_router._restore_existing_session_files(
        manager, sid, session_dir, "input/reference_images", "reference_*"
    ) == [str(ref)]

    upload = UploadFile(filename="raw.bin", file=io.BytesIO(b"abcdef"))
    copied = await pipeline_router._stream_copy(
        upload, tmp_path / "copy" / "raw.bin", 2
    )
    assert copied == 6
    assert (tmp_path / "copy" / "raw.bin").read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_pipeline_additional_helper_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        pipeline_router._cluster_session_config_from_step_config(
            enabled=True,
            step_config={"complexity_thresholds": "bad", "report": {"enabled": False}},
        )["cluster_similarity_threshold_low"]
        is None
    )
    pipeline_router._apply_build_dataset_render_worker_limit({"steps": {}}, None)
    pipeline_router._apply_large_scene_render_batch_limit({"steps": {}})
    assert pipeline_router._get_generated_reference_entry(None, "ref") is None
    assert (
        pipeline_router._get_generated_reference_entry(
            {"generated_reference_images": [{"id": "other"}]}, "ref"
        )
        is None
    )

    class _HydratingManager:
        async def sync_from_store(self, session_id: str, prefix: str) -> int:
            input_render.write_bytes(b"png")
            return 1

        async def read_from_store(self, session_id: str, key: str) -> bytes | None:
            return None

    class _MissingManager:
        async def sync_from_store(self, session_id: str, prefix: str) -> int:
            return 0

        async def read_from_store(self, session_id: str, key: str) -> bytes | None:
            return None

    session_dir = tmp_path / "session"
    input_render = session_dir / "input" / "input_render.png"
    input_render.parent.mkdir(parents=True)
    assert (
        await pipeline_router._ensure_input_render_local(
            _HydratingManager(), "sid", session_dir
        )
        == input_render
    )
    input_render.unlink()
    assert (
        await pipeline_router._ensure_input_render_local(
            _MissingManager(), "sid", session_dir
        )
        is None
    )

    cache_dir = tmp_path / "cache-session"
    assert not pipeline_router._ensure_cached_generated_material_library_state(
        cache_dir, session_id="sid"
    )
    manifest = cache_dir / "cache" / "generated_material_library" / "materials.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("materials:\n  library_path: lib.usda\n  entries: []\n")
    assert pipeline_router._load_cached_generated_material_library(cache_dir) is None
    (manifest.parent / "lib.usda").write_text("#usda 1.0\n")
    manifest.write_text(
        "materials:\n  library_path: lib.usda\n  entries:\n    - name: A\n"
    )
    (cache_dir / "cache" / ".pipeline_state.json").write_text("[]")
    assert pipeline_router._ensure_cached_generated_material_library_state(
        cache_dir, session_id="sid"
    )

    zip_path = tmp_path / "bad.zip"
    zip_path.write_bytes(b"not-a-zip")
    with pytest.raises(HTTPException) as exc_info:
        pipeline_router._extract_and_validate_materials_zip(zip_path, tmp_path / "out")
    _expect_http(400, exc_info)

    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/materials.yaml", "boom")
    extract_dir = tmp_path / "extract"
    (extract_dir / "nested").mkdir(parents=True)
    (extract_dir / "nested" / "keep.txt").write_text("keep")

    def fail_copy(*args: object, **kwargs: object) -> int:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(pipeline_router, "copy_stream_limited", fail_copy)
    with zipfile.ZipFile(archive_path, "r") as zf:
        with pytest.raises(RuntimeError):
            pipeline_router._safe_extract_materials_zip(zf, extract_dir)
    assert (extract_dir / "nested" / "keep.txt").exists()

    clean_extract_dir = tmp_path / "clean-extract"
    with zipfile.ZipFile(archive_path, "r") as zf:
        with pytest.raises(RuntimeError):
            pipeline_router._safe_extract_materials_zip(zf, clean_extract_dir)
    assert not (clean_extract_dir / "nested").exists()

    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    manager = SessionManager(local_root, store=LocalSessionStore(str(remote_root)))
    sid = str(uuid4())
    await manager.create_session(sid)
    await manager.store.put_bytes(
        sid,
        "materials/materials.yaml",
        b"library_path: library.usda\nentries:\n  - name: A\n",
    )
    await manager.store.put_bytes(sid, "materials/library.usda", b"#usda 1.0\n")
    restored = await pipeline_router._restore_existing_session_materials(
        manager, sid, manager.get_session_dir(sid)
    )
    assert restored is not None
    assert restored[1] == [{"name": "A"}]

    files_sid = str(uuid4())
    files_dir = await manager.create_session(files_sid)
    await manager.store.put_bytes(
        files_sid, "input/reference_pdfs/reference_0000.pdf", b"pdf"
    )
    restored_files = await pipeline_router._restore_existing_session_files(
        manager,
        files_sid,
        files_dir,
        "input/reference_pdfs",
        "reference_*.pdf",
    )
    assert len(restored_files) == 1


def test_pipeline_material_manifest_and_generated_cache_helpers(tmp_path: Path) -> None:
    base = tmp_path / "materials"
    base.mkdir()
    (base / "library.usda").write_text("#usda 1.0\n")

    library_path, entries = pipeline_router._validate_materials_yaml_content(
        {"materials": {"library_path": "library.usda", "entries": [{"name": "A"}]}},
        base,
    )
    assert library_path.endswith("library.usda")
    assert entries == [{"name": "A"}]

    invalid_values = [
        [],
        {"materials": []},
        {"materials": {"entries": [{"name": "A"}]}},
        {"materials": {"library_path": "library.usda", "entries": []}},
        {"materials": {"library_path": "library.usda", "entries": ["bad"]}},
        {"materials": {"library_path": "../library.usda", "entries": [{"name": "A"}]}},
        {"materials": {"library_path": "missing.usda", "entries": [{"name": "A"}]}},
    ]
    for value in invalid_values:
        with pytest.raises(HTTPException):
            pipeline_router._validate_materials_yaml_content(value, base)

    session_dir = tmp_path / "session"
    assert pipeline_router._load_cached_generated_material_library(session_dir) is None
    manifest = session_dir / "cache" / "generated_material_library" / "materials.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("- item")
    assert pipeline_router._load_cached_generated_material_library(session_dir) is None
    manifest.write_text("materials: []")
    assert pipeline_router._load_cached_generated_material_library(session_dir) is None
    manifest.write_text(
        "materials:\n  library_path: missing.usda\n  entries:\n    - name: A\n"
    )
    assert pipeline_router._load_cached_generated_material_library(session_dir) is None
    (manifest.parent / "missing.usda").write_text("#usda 1.0\n")
    cached = pipeline_router._load_cached_generated_material_library(session_dir)
    assert cached is not None
    assert cached["generated_material_entries"] == [{"name": "A"}]

    state_path = session_dir / "cache" / ".pipeline_state.json"
    state_path.write_text('{"step_outputs": []}')
    assert pipeline_router._ensure_cached_generated_material_library_state(
        session_dir, session_id="sid"
    )
    state = json.loads(state_path.read_text())
    assert state["step_outputs"]["generate_material_library"][
        "generated_material_entries"
    ] == [{"name": "A"}]


@pytest.mark.asyncio
async def test_regeneration_planning_fallback_and_hydration_edges(
    tmp_path: Path,
) -> None:
    class _PlanningManager:
        def __init__(self, kind: str) -> None:
            self.store = SimpleNamespace(kind=kind)

        async def get_session_metadata(self, _session_id: str) -> dict:
            return {}

        @staticmethod
        def resolve_published_artifact_key(
            _metadata: dict,
            key: str,
            *,
            legacy_key: str | None = None,
        ) -> str:
            return key

        async def read_from_store(
            self,
            _session_id: str,
            _key: str,
        ) -> None:
            return None

    session_dir = tmp_path / "planning"
    state_path = session_dir / "cache" / ".pipeline_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text('{"completed_steps": ["predict"]}')
    local_manager = _PlanningManager("local")
    assert await pipeline_router._read_regeneration_checkpoint(
        local_manager,  # type: ignore[arg-type]
        "sid",
        session_dir,
    ) == {"completed_steps": ["predict"]}
    state_path.write_text("{")
    assert (
        await pipeline_router._read_regeneration_checkpoint(
            local_manager,  # type: ignore[arg-type]
            "sid",
            session_dir,
        )
        == {}
    )

    local_input = session_dir / "input" / "scene.usda"
    local_input.parent.mkdir(parents=True)
    local_input.write_bytes(b"#usda 1.0\n")
    assert (
        await pipeline_router._read_regeneration_plan_key(
            local_manager,  # type: ignore[arg-type]
            "sid",
            session_dir,
            "input/scene.usda",
        )
        == b"#usda 1.0\n"
    )
    assert (
        await pipeline_router._read_regeneration_plan_key(
            _PlanningManager("remote"),  # type: ignore[arg-type]
            "sid",
            session_dir,
            "input/missing.usda",
        )
        is None
    )

    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    session_id = str(uuid4())
    hydrated_dir = await manager.create_session(session_id)
    source_objects = {
        "cache/dataset/usd/prims.jsonl": b"prims",
        "cache/clusters/cluster_map.jsonl": b"clusters",
        "cache/predictions/predictions.jsonl": b"predictions",
        "output/scene_with_materials.usd": b"output",
    }
    for key, data in source_objects.items():
        await store.put_bytes(session_id, key, data)
    await pipeline_router._hydrate_regeneration_inputs(
        manager,
        session_id,
        hydrated_dir,
        [
            "build_dataset_prepare_dataset",
            "expand_cluster_predictions",
            "render",
        ],
        optimize_usd_enabled=False,
    )
    for key, data in source_objects.items():
        assert (hydrated_dir / key).read_bytes() == data


@pytest.mark.asyncio
async def test_regeneration_bundle_ignores_bad_descriptions_and_missing_cache_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await store.put_bytes(session_id, "input/scene.usda", b"#usda 1.0\n")
    await store.put_bytes(
        session_id,
        "input/reference_images/descriptions.json",
        b"\xff",
    )
    generated_manifest_key = "cache/generated_material_library/materials.yaml"
    await store.put_bytes(session_id, generated_manifest_key, b"placeholder")
    original_read = manager.read_from_store

    async def missing_generated_manifest(
        requested_session_id: str,
        key: str,
    ) -> bytes | None:
        if key == generated_manifest_key:
            return None
        return await original_read(requested_session_id, key)

    monkeypatch.setattr(manager, "read_from_store", missing_generated_manifest)
    bundle = await pipeline_router._plan_regeneration_input_bundle(
        manager,
        session_id,
        session_dir,
        {"config": {"enable_material_generation": True}},
    )
    assert bundle.reference_descriptions == ()
    assert not bundle.generated_library_cache_available


def test_regeneration_material_manifest_and_extract_tree_edges(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    available = {"materials/library.usda"}
    invalid_manifests = [
        b"- list",
        b"materials: []",
        b"materials:\n  library_path: library.usda\n  entries: []\n",
        b"materials:\n  entries:\n    - name: A\n",
        b"materials:\n  library_path: ../library.usda\n  entries:\n    - name: A\n",
        b"materials:\n  library_path: missing.usda\n  entries:\n    - name: A\n",
    ]
    for manifest in invalid_manifests:
        with pytest.raises(HTTPException):
            pipeline_router._plan_material_manifest(
                manifest,
                "materials/materials.yaml",
                available,
                session_dir,
            )

    library, entries = pipeline_router._plan_material_manifest(
        b"materials:\n  library_path: nested/library.usda\n  entries:\n    - name: A\n",
        "materials/materials.yaml",
        available,
        session_dir,
    )
    assert library.endswith("materials/library.usda")
    assert entries == [{"name": "A"}]
    absolute_library, _ = pipeline_router._plan_material_manifest(
        b"materials:\n  library_path: /library.usda\n  entries:\n    - name: A\n",
        "materials/materials.yaml",
        available,
        session_dir,
    )
    assert absolute_library.endswith("materials/library.usda")

    materials_dir = tmp_path / "extracted"
    materials_dir.mkdir()
    assert pipeline_router._load_extracted_materials_tree(materials_dir) is None
    nested = materials_dir / "nested"
    nested.mkdir()
    (nested / "library.usda").write_text("#usda 1.0\n")
    manifest_path = nested / "materials.yaml"
    manifest_path.write_text("library_path: library.usda\nentries:\n  - name: A\n")
    loaded = pipeline_router._load_extracted_materials_tree(materials_dir)
    assert loaded is not None and loaded[1] == [{"name": "A"}]
    manifest_path.write_text("materials: [")
    with pytest.raises(HTTPException):
        pipeline_router._load_extracted_materials_tree(materials_dir)

    preserve = materials_dir / "materials.zip"
    preserve.write_bytes(b"zip")
    stale_dir = materials_dir / "stale"
    stale_dir.mkdir()
    (stale_dir / "file").write_text("stale")
    stale_file = materials_dir / "stale.txt"
    stale_file.write_text("stale")
    pipeline_router._clean_materials_extract_dir(materials_dir, preserve)
    assert preserve.exists()
    assert not stale_dir.exists() and not stale_file.exists()

    generated_session = tmp_path / "generated"
    generated_manifest = (
        generated_session / "cache" / "generated_material_library" / "materials.yaml"
    )
    generated_manifest.parent.mkdir(parents=True)
    generated_manifest.write_text(
        "materials:\n  library_path: library.usda\n  entries:\n    - name: A\n"
    )
    (generated_manifest.parent / "library.usda").write_text("#usda 1.0\n")
    assert pipeline_router._ensure_cached_generated_material_library_state(
        generated_session,
        session_id="sid",
    )
    assert (generated_session / "cache" / ".pipeline_state.json").exists()


@pytest.mark.asyncio
async def test_regeneration_step_evidence_and_artifact_dependency_edges(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    cluster_map = session_dir / "cache" / "clusters" / "cluster_map.jsonl"
    cluster_reps = session_dir / "cache" / "clusters" / "dataset_representatives.jsonl"
    rendered = session_dir / "output" / "scene_with_materials_flat.usd"
    for path in (cluster_map, cluster_reps, rendered):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data")
    (session_dir / "cache" / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "completed_steps": ["cluster_prims", "render"],
                "step_outputs": {
                    "cluster_prims": {"cluster_map_path": str(cluster_map)},
                    "render": {"flattened_usd_path": str(rendered)},
                },
            }
        )
    )
    validity = {
        **initial_artifact_validity(),
        "cluster_map": True,
        "cluster_representatives": True,
        "rendered_output_usd": True,
    }
    evidence = await pipeline_router._derive_regeneration_step_evidence(
        manager,
        session_id,
        session_dir,
        validity,
    )
    assert {"cluster_prims", "render"} <= evidence

    with pytest.raises(HTTPException, match="requires current 'raw_predictions'"):
        pipeline_router._validate_regeneration_dependency_closure(
            ["apply"],
            {"apply"},
            optimize_usd_enabled=False,
            metadata={
                "artifact_validity": initial_artifact_validity(),
                "_regeneration_step_evidence": {"predict"},
            },
        )


def test_regeneration_checkpoint_invalidation_is_downstream_only(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    state_path = session_dir / "cache" / ".pipeline_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "completed_steps": [
                    "generate_material_library",
                    "build_dataset_usd",
                    "predict",
                    "restore_usd",
                    "apply",
                ],
                "failed_steps": ["build_dataset_usd", "apply"],
                "step_errors": {
                    "build_dataset_usd": "old upstream warning",
                    "apply": "old apply failure",
                },
                "step_outputs": {
                    "generate_material_library": {"entries": ["cached"]},
                    "build_dataset_usd": {"num_prims": 2},
                    "predict": {"predictions_path": "raw.jsonl"},
                    "restore_usd": {"restored_predictions_path": "restored.jsonl"},
                    "apply": {"output_usd_path": "old.usd"},
                },
                "current_step": "apply",
            }
        ),
        encoding="utf-8",
    )

    invalidated = pipeline_router._invalidate_regeneration_pipeline_state(
        session_dir,
        ["apply", "predict"],
    )

    assert {"predict", "restore_usd", "apply"} <= invalidated
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed_steps"] == [
        "generate_material_library",
        "build_dataset_usd",
    ]
    assert state["failed_steps"] == ["build_dataset_usd"]
    assert state["step_errors"] == {"build_dataset_usd": "old upstream warning"}
    assert set(state["step_outputs"]) == {
        "generate_material_library",
        "build_dataset_usd",
    }
    assert state["current_step"] is None


def test_regeneration_checkpoint_invalidation_handles_missing_and_bad_state(
    tmp_path: Path,
) -> None:
    missing_session = tmp_path / "missing"
    assert not pipeline_router._invalidate_regeneration_pipeline_state(
        missing_session,
        ["not-a-step"],
    )
    invalidated = pipeline_router._invalidate_regeneration_pipeline_state(
        missing_session,
        ["predict"],
    )
    assert "restore_usd" in invalidated

    for index, payload in enumerate(("{", "[]")):
        session_dir = tmp_path / f"bad-{index}"
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(payload, encoding="utf-8")

        pipeline_router._invalidate_regeneration_pipeline_state(
            session_dir,
            ["predict"],
        )

        assert json.loads(state_path.read_text(encoding="utf-8")) == {
            "completed_steps": [],
            "step_outputs": {},
            "step_errors": {},
            "failed_steps": [],
            "current_step": None,
        }

    bad_collections_dir = tmp_path / "bad-collections"
    bad_collections_path = bad_collections_dir / "cache" / ".pipeline_state.json"
    bad_collections_path.parent.mkdir(parents=True)
    bad_collections_path.write_text(
        json.dumps(
            {
                "completed_steps": "predict",
                "step_outputs": [],
                "step_errors": [],
                "failed_steps": "apply",
            }
        ),
        encoding="utf-8",
    )
    pipeline_router._invalidate_regeneration_pipeline_state(
        bad_collections_dir,
        ["predict"],
    )
    assert json.loads(bad_collections_path.read_text(encoding="utf-8")) == {
        "completed_steps": [],
        "step_outputs": {},
        "step_errors": {},
        "failed_steps": [],
        "current_step": None,
    }


@pytest.mark.asyncio
@pytest.mark.real_executor
async def test_pipeline_restore_materials_and_preview_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    sid = str(uuid4())
    session_dir = await manager.create_session(sid)
    pipeline_router.set_session_manager(manager)

    assert (
        await pipeline_router._restore_existing_session_materials(
            manager, sid, session_dir
        )
        is None
    )

    materials_dir = session_dir / "materials"
    materials_dir.mkdir(exist_ok=True)
    (materials_dir / "library.usda").write_text("#usda 1.0\n")
    (materials_dir / "materials.yaml").write_text(
        "library_path: library.usda\nentries:\n  - name: A\n"
    )
    restored = await pipeline_router._restore_existing_session_materials(
        manager, sid, session_dir
    )
    assert restored[1] == [{"name": "A"}]

    empty_sid = str(uuid4())
    empty_dir = await manager.create_session(empty_sid)
    await pipeline_router._render_input_preview(empty_sid, empty_dir)
    metadata = await manager.get_session_metadata(empty_sid)
    assert metadata["preview_render_status"] == "failed"

    render_sid = str(uuid4())
    render_dir = await manager.create_session(render_sid)
    (render_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    generated = tmp_path / "rendered.png"
    preview_contexts: list[dict[str, Any]] = []

    class _Workflow:
        def run(self, context: dict[str, Any]) -> dict:
            preview_contexts.append(context)
            preview_config = context["config_dict"]
            marker = Path(preview_config["output_dir"]).parent / (
                ".input_render_config.yaml"
            )
            assert not marker.exists()
            generated.write_bytes(b"png")
            return {"rendered_preview_paths": [str(generated)]}

    module = types.ModuleType("material_agent.workflows")
    module.create_render_preview_workflow_from_config = lambda: _Workflow()
    monkeypatch.setitem(sys.modules, "material_agent.workflows", module)
    await pipeline_router._render_input_preview(render_sid, render_dir)
    metadata = await manager.get_session_metadata(render_sid)
    assert metadata["preview_render_status"] == "ready"
    assert (render_dir / "input" / "input_render.png").exists()
    assert "config_dict" in preview_contexts[-1]
    assert "config_path" not in preview_contexts[-1]

    original_sid = str(uuid4())
    original_dir = await manager.create_session(original_sid)
    copied_scene = original_dir / "input" / "scene.usda"
    copied_scene.write_text("#usda 1.0\n")
    original = tmp_path / "original.usda"
    original.write_text("#usda 1.0\n")
    await pipeline_router._render_input_preview(
        original_sid, original_dir, original_usd_path=original
    )
    assert (original_dir / "input" / "input_render.png").exists()

    failed_sid = str(uuid4())
    failed_dir = await manager.create_session(failed_sid)
    (failed_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    sentinel = "sentinel-preview-backend-secret"

    class _FailingWorkflow:
        def run(self, _context: dict[str, Any]) -> dict:
            raise RuntimeError(sentinel)

    module.create_render_preview_workflow_from_config = lambda: _FailingWorkflow()
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        await pipeline_router._render_input_preview(failed_sid, failed_dir)
    failed_metadata = await manager.get_session_metadata(failed_sid)
    assert failed_metadata["preview_render_status"] == "failed"
    assert failed_metadata["preview_render_error"] == "Input preview render failed"
    assert "pipeline_input_preview_failed" in caplog.text
    assert "phase=pipeline_execution" in caplog.text
    assert sentinel not in json.dumps(failed_metadata)
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_upload_and_open_usd_endpoint_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    pipeline_router.set_session_manager(manager)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.upload_usd_immediate(_upload("scene.txt", b"x"))
    _expect_http(400, exc_info)

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.upload_usd_immediate(_upload("scene.usda", b"x"))
    _expect_http(413, exc_info)

    local_sentinel = "material-local-publication-sentinel-727"

    async def broken_stream(upload: UploadFile, dest: Path, chunk_size: int = 1) -> int:
        raise RuntimeError(local_sentinel)

    with caplog.at_level(logging.ERROR), monkeypatch.context() as m:
        m.setattr(pipeline_router, "_stream_copy", broken_stream)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.upload_usd_immediate(_upload("scene.usda", b"x"))
        _expect_http(500, exc_info)
    assert exc_info.value.detail == "Failed to upload USD"
    assert local_sentinel not in caplog.text
    assert "code=pipeline_usd_ingest_failed" in caplog.text
    assert "phase=local_publication" in caplog.text

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 100)
    caplog.clear()
    mirror_sentinel = "material-sync-upload-sentinel-727"

    async def fail_put(*args: object, **kwargs: object) -> None:
        raise RuntimeError(mirror_sentinel)

    monkeypatch.setattr(manager, "put_file_to_store", fail_put)
    with caplog.at_level(logging.ERROR):
        uploaded = await pipeline_router.upload_usd_immediate(
            _upload("scene.usda", b"x")
        )
    assert uploaded.status == "ready"
    assert mirror_sentinel not in caplog.text
    assert "code=pipeline_usd_sync_failed" in caplog.text
    assert "phase=sync_upload" in caplog.text

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.open_usd_local(file_path="relative.usda")
    _expect_http(400, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.open_usd_local(file_path=str(tmp_path / "missing.usda"))
    _expect_http(400, exc_info)
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("x")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.open_usd_local(file_path=str(bad_file))
    _expect_http(400, exc_info)

    too_large = tmp_path / "too_large.usda"
    too_large.write_bytes(b"x")
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.open_usd_local(file_path=str(too_large))
    _expect_http(413, exc_info)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tiny_scene = source_dir / "asset.usda"
    tiny_scene.write_text("")
    (source_dir / "payload.bin").write_bytes(b"x" * 10)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0.000001)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.open_usd_local(file_path=str(tiny_scene))
    _expect_http(413, exc_info)

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 100)
    opened = await pipeline_router.open_usd_local(file_path=str(tiny_scene))
    assert opened.status == "ready"


@pytest.mark.asyncio
async def test_generated_reference_endpoint_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    pipeline_router.set_session_manager(manager)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(str(uuid4()), "prompt")
    _expect_http(404, exc_info)

    sid = str(uuid4())
    session_dir = await manager.create_session(sid)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(sid, "prompt")
    _expect_http(409, exc_info)

    await manager.update_session(
        sid, {"status": "ready", "preview_render_status": "ready"}
    )
    monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "custom")
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", None)
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key_env", None)
    monkeypatch.setattr(pipeline_router.config, "image_gen_base_url", None)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(sid, "prompt")
    _expect_http(503, exc_info)

    monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
    monkeypatch.setattr(
        pipeline_router.config, "image_gen_base_url", "https://api.openai.com/v1"
    )
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", "key")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(sid, "prompt")
    _expect_http(400, exc_info)

    (session_dir / "input" / "input_render.png").write_bytes(b"png")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(sid, "   ")
    _expect_http(400, exc_info)

    monkeypatch.setenv("IMAGE_GEN_TEST_KEY", "key")
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", None)
    monkeypatch.setattr(
        pipeline_router.config, "image_gen_api_key_env", "IMAGE_GEN_TEST_KEY"
    )

    class _NoImageWorkflow:
        def run(self, context: dict[str, str]) -> dict[str, list[str]]:
            return {"generated_reference_image_paths": []}

    module = types.ModuleType("material_agent.workflows")
    module.create_generate_reference_image_workflow_from_config = lambda: (
        _NoImageWorkflow()
    )
    monkeypatch.setitem(sys.modules, "material_agent.workflows", module)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.generate_reference_image(sid, "prompt")
    _expect_http(500, exc_info)

    generation_sentinel = "image-generation-provider-secret-727"

    class _FailingWorkflow:
        def run(self, context: dict[str, str]) -> dict[str, list[str]]:
            raise RuntimeError(generation_sentinel)

    module.create_generate_reference_image_workflow_from_config = lambda: (
        _FailingWorkflow()
    )
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.generate_reference_image(sid, "prompt")
    _expect_http(500, exc_info)
    assert generation_sentinel not in str(exc_info.value.detail)
    assert generation_sentinel not in caplog.text
    assert "code=pipeline_reference_image_generation_failed" in caplog.text

    class _ImageWorkflow:
        def run(self, context: dict[str, Any]) -> dict[str, list[str]]:
            data = context["config_dict"]
            output_path = Path(data["output_dir"]) / "generated_ref_0.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"png")
            return {"generated_reference_image_paths": [str(output_path)]}

    module.create_generate_reference_image_workflow_from_config = lambda: (
        _ImageWorkflow()
    )
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key_env", None)
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", "key")
    original_get_metadata = manager.get_session_metadata
    calls = {"count": 0}

    async def stale_metadata(session_id: str) -> dict | None:
        calls["count"] += 1
        if calls["count"] >= 2:
            return {"status": "running"}
        return await original_get_metadata(session_id)

    with monkeypatch.context() as m:
        m.setattr(manager, "get_session_metadata", stale_metadata)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.generate_reference_image(sid, "prompt")
        _expect_http(409, exc_info)

    original_put_file = manager.put_file_to_store

    async def fail_put(*args: object, **kwargs: object) -> None:
        raise RuntimeError("mirror failed")

    monkeypatch.setattr(manager, "put_file_to_store", fail_put)
    response = await pipeline_router.generate_reference_image(sid, "prompt")
    assert response["status"] == "ok"

    monkeypatch.setattr(manager, "put_file_to_store", original_put_file)

    async def reject_reference(*_args: object, **_kwargs: object) -> bool:
        return False

    with monkeypatch.context() as m:
        m.setattr(manager, "add_generated_reference_image", reject_reference)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.generate_reference_image(sid, "prompt")
        _expect_http(404, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.delete_generated_reference_image(str(uuid4()), "ref")
    _expect_http(404, exc_info)
    await manager.update_session(sid, {"status": "running"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.delete_generated_reference_image(sid, "ref")
    _expect_http(409, exc_info)

    await manager.update_session(sid, {"status": "ready"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.delete_generated_reference_image(sid, "missing")
    _expect_http(404, exc_info)
    await manager.add_generated_reference_image(
        sid,
        {
            "id": "ref",
            "key": "input/generated_references/ref/generated_ref_0.png",
            "prompt": "prompt",
        },
    )
    ref_path = session_dir / "input/generated_references/ref/generated_ref_0.png"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_bytes(b"png")
    (ref_path.parent / "keep.txt").write_text("keep")

    async def fail_delete(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("delete failed")

    monkeypatch.setattr(manager.store, "delete_file", fail_delete)
    deleted = await pipeline_router.delete_generated_reference_image(sid, "ref")
    assert deleted == {"status": "deleted", "reference_id": "ref"}


@pytest.mark.asyncio
async def test_create_pipeline_direct_endpoint_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(
        tmp_path / "local",
        store=LocalSessionStore(str(tmp_path / "remote")),
    )
    pipeline_router.set_session_manager(manager)
    pipeline_router.get_event_bus().set_session_manager(manager)
    registry = _install_direct_pipeline_stubs(monkeypatch)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 100)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(**_direct_pipeline_kwargs())
    _expect_http(400, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                usd_file=_upload("scene.txt", b"x"),
                optimize_usd="false",
            )
        )
    _expect_http(400, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=str(uuid4()),
                optimize_usd="false",
            )
        )
    _expect_http(404, exc_info)

    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(usd_file=_upload("scene.usda", b"x"))
        )
    _expect_http(413, exc_info)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 100)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                usd_file=_upload("scene.usda", b"x"),
                large_scene="true",
                enable_material_generation="true",
            )
        )
    _expect_http(400, exc_info)

    with pytest.raises(HTTPException, match="large-scene prim-level") as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                usd_file=_upload("scene.usda", b"x"),
                large_scene="true",
                coverage_policy="strict",
            )
        )
    _expect_http(400, exc_info)

    sid = str(uuid4())
    await manager.create_session(sid)
    await manager.update_session(sid, {"config": [], "status": "ready"})
    await manager.store.put_bytes(sid, "input/scene.usda", b"#usda 1.0\n")
    response = await pipeline_router.create_pipeline(
        **_direct_pipeline_kwargs(
            session_id=sid,
            camera_views="",
            optimize_usd="false",
            reference_images=[_upload("ref.jpg", b"jpeg")],
            reference_pdfs=[_upload("skip.txt", b"x"), _upload("ref.pdf", b"%PDF")],
            reference_descriptions="{",
            pdf_descriptions="{",
            pdf_first_page=2,
            pdf_last_page=3,
            steps="build_dataset_usd,build_dataset_prepare_dataset,predict,apply,render",
            user_email=" user@nvidia.com ",
        )
    )
    assert response.status == "pending"
    assert registry.registered[-1][0] == sid
    metadata = await manager.get_session_metadata(sid)
    assert (
        metadata["config"]["camera_views"] == pipeline_router.DEFAULT_CAMERA_DIRECTIONS
    )

    rich_sid = str(uuid4())
    rich_dir = await manager.create_session(rich_sid)
    await manager.update_session(rich_sid, {"status": "ready"})
    (rich_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_stage_info_from_path",
            lambda path: {
                "prim_count": pipeline_router.DEFAULT_USD_PRIM_WARNING_THRESHOLD + 1
            },
        )
        rich = await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=rich_sid,
                optimize_usd="false",
                material_library="unknown-library",
                reference_images=[_upload("ref.png", b"png")],
                reference_pdfs=[_upload("ref.pdf", b"%PDF")],
                reference_descriptions='["front"]',
                pdf_descriptions='["manual"]',
                steps="build_dataset_usd,build_dataset_prepare_dataset,predict,apply,render",
                vlm_model="nim/custom-vlm",
            )
        )
    assert rich.status == "pending"

    non_list_sid = str(uuid4())
    non_list_dir = await manager.create_session(non_list_sid)
    await manager.update_session(non_list_sid, {"status": "ready"})
    (non_list_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    non_list = await pipeline_router.create_pipeline(
        **_direct_pipeline_kwargs(
            session_id=non_list_sid,
            optimize_usd="false",
            reference_descriptions="{}",
            pdf_descriptions="{}",
            steps="predict",
        )
    )
    assert non_list.status == "pending"

    restore_pdf_sid = str(uuid4())
    restore_pdf_dir = await manager.create_session(restore_pdf_sid)
    await manager.update_session(restore_pdf_sid, {"status": "ready"})
    (restore_pdf_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    await manager.store.put_bytes(
        restore_pdf_sid, "input/reference_pdfs/reference_0000.pdf", b"pdf"
    )
    restored_pdf = await pipeline_router.create_pipeline(
        **_direct_pipeline_kwargs(
            session_id=restore_pdf_sid,
            optimize_usd="false",
            steps="build_dataset_prepare_dataset,predict",
        )
    )
    assert restored_pdf.status == "pending"

    pdf_prepare_sid = str(uuid4())
    pdf_prepare_dir = await manager.create_session(pdf_prepare_sid)
    await manager.update_session(pdf_prepare_sid, {"status": "ready"})
    (pdf_prepare_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    pdf_prepare = await pipeline_router.create_pipeline(
        **_direct_pipeline_kwargs(
            session_id=pdf_prepare_sid,
            optimize_usd="false",
            reference_pdfs=[_upload("ref.pdf", b"%PDF")],
            steps="predict",
        )
    )
    assert pdf_prepare.status == "pending"

    material_zip_sid = str(uuid4())
    material_zip_dir = await manager.create_session(material_zip_sid)
    await manager.update_session(material_zip_sid, {"status": "ready"})
    (material_zip_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 0)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=material_zip_sid,
                optimize_usd="false",
                materials_zip=_upload("materials.zip", b"x"),
            )
        )
    _expect_http(413, exc_info)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 100)

    import material_agent.api as material_api

    def no_optimize_builder(**kwargs: object) -> dict[str, object]:
        config = _minimal_pipeline_config(**kwargs)
        config["steps"].pop("optimize_usd", None)
        return config

    optimize_sid = str(uuid4())
    optimize_dir = await manager.create_session(optimize_sid)
    await manager.update_session(optimize_sid, {"status": "ready"})
    (optimize_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    with monkeypatch.context() as m:
        m.setattr(material_api, "build_unified_pipeline_config", no_optimize_builder)
        optimized = await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=optimize_sid,
                optimize_usd="true",
                enable_deinstance="true",
                enable_split="false",
                enable_deduplicate="false",
                steps="build_dataset_usd,predict,apply",
            )
        )
    assert optimized.status == "pending"

    no_input = str(uuid4())
    await manager.create_session(no_input)
    await manager.update_session(no_input, {"status": "ready"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(session_id=no_input, optimize_usd="false")
        )
    _expect_http(400, exc_info)

    generated_sid = str(uuid4())
    generated_dir = await manager.create_session(generated_sid)
    await manager.update_session(generated_sid, {"status": "ready"})
    (generated_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    await manager.update_session(
        generated_sid,
        {
            "generated_reference_images": [
                {"id": "ref", "key": "input/generated_references/ref.png"}
            ]
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=generated_sid,
                optimize_usd="false",
                generated_reference_id="missing",
            )
        )
    _expect_http(400, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=generated_sid,
                optimize_usd="false",
                generated_reference_id="ref",
            )
        )
    _expect_http(400, exc_info)

    await manager.update_session(
        generated_sid,
        {"generated_reference_images": [{"id": "no-key"}]},
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=generated_sid,
                optimize_usd="false",
                generated_reference_id="no-key",
            )
        )
    _expect_http(400, exc_info)

    monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
    monkeypatch.setattr(pipeline_router.config, "image_gen_api_key", "key")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=generated_sid,
                optimize_usd="false",
                enable_material_generation="true",
            )
        )
    _expect_http(400, exc_info)


@pytest.mark.asyncio
async def test_oversized_historical_descriptions_fail_before_session_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(
        tmp_path / "local",
        store=LocalSessionStore(str(tmp_path / "remote")),
    )
    pipeline_router.set_session_manager(manager)
    pipeline_router.get_event_bus().set_session_manager(manager)
    registry = _install_direct_pipeline_stubs(monkeypatch)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "ready"})
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    descriptions = session_dir / "input" / "reference_images" / "descriptions.json"
    descriptions.parent.mkdir(parents=True)
    descriptions.write_bytes(
        b"[" + b"x" * pipeline_router._MAX_HISTORICAL_DESCRIPTIONS_BYTES + b"]"
    )
    before = json.loads(json.dumps(await manager.get_session_metadata(session_id)))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=session_id,
                optimize_usd="false",
            )
        )

    _expect_http(409, exc_info)
    assert await manager.get_session_metadata(session_id) == before
    assert registry.registered == []
    assert pipeline_router.get_event_bus().get_snapshot(session_id) is None


@pytest.mark.asyncio
async def test_historical_descriptions_reject_forged_credentials_value_free(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    descriptions = session_dir / "input" / "reference_images" / "descriptions.json"
    descriptions.parent.mkdir(parents=True)
    sentinel = "historical-description-secret-727"
    descriptions.write_text(
        json.dumps([{"api_key": sentinel}]),
        encoding="utf-8",
    )
    before = json.loads(json.dumps(await manager.get_session_metadata(session_id)))

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._preflight_historical_reference_descriptions(
            manager,
            session_id,
            session_dir,
        )

    _expect_http(409, exc_info)
    assert sentinel not in str(exc_info.value)
    assert await manager.get_session_metadata(session_id) == before


@pytest.mark.asyncio
async def test_historical_descriptions_hold_open_file_across_reserved_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    descriptions = session_dir / "input" / "reference_images" / "descriptions.json"
    descriptions.parent.mkdir(parents=True)
    descriptions.write_text(json.dumps(["safe-description"]), encoding="utf-8")
    secret = session_dir / "cache" / ".pipeline_temp" / "descriptions.json"
    secret.parent.mkdir(parents=True)
    secret.write_text(json.dumps(["sentinel-description-secret"]), encoding="utf-8")
    original_open = manager.open_local_artifact

    async def open_then_swap(open_session_id: str, path: str | Path):
        artifact = await original_open(open_session_id, path)
        if artifact is not None and descriptions.exists():
            descriptions.rename(descriptions.with_name("descriptions.held.json"))
            descriptions.symlink_to(secret)
        return artifact

    monkeypatch.setattr(manager, "open_local_artifact", open_then_swap)

    loaded = await pipeline_router._preflight_historical_reference_descriptions(
        manager,
        session_id,
        session_dir,
    )

    assert loaded == ["safe-description"]
    assert secret.read_text(encoding="utf-8") == json.dumps(
        ["sentinel-description-secret"]
    )


def test_historical_material_snapshot_uses_held_session_root_during_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside-snapshot-sentinel", encoding="utf-8")
    detached = tmp_path / "session-held"
    original_write = pipeline_router.write_bytes_to_confined
    swapped = False

    def swap_then_write(*args: object, **kwargs: object) -> bool:
        nonlocal swapped
        if not swapped:
            swapped = True
            session_dir.rename(detached)
            session_dir.symlink_to(outside, target_is_directory=True)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(pipeline_router, "write_bytes_to_confined", swap_then_write)

    pipeline_router._write_historical_material_snapshot(
        session_dir,
        "materials/materials.yaml",
        b"safe-material-snapshot",
    )

    assert (detached / "materials" / "materials.yaml").read_bytes() == (
        b"safe-material-snapshot"
    )
    assert sentinel.read_text(encoding="utf-8") == "outside-snapshot-sentinel"


def test_clean_materials_extract_dir_uses_held_root_during_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_dir = tmp_path / "materials"
    extract_dir.mkdir()
    preserve = extract_dir / "materials.zip"
    preserve.write_bytes(b"zip")
    (extract_dir / "stale.txt").write_text("stale", encoding="utf-8")
    stale_dir = extract_dir / "stale"
    stale_dir.mkdir()
    (stale_dir / "nested.txt").write_text("stale", encoding="utf-8")
    outside = tmp_path / "outside-materials"
    outside.mkdir()
    outside_sentinel = outside / "sentinel.txt"
    outside_sentinel.write_text("outside-cleanup-sentinel", encoding="utf-8")
    detached = tmp_path / "materials-held"
    original_scandir = pipeline_router.os.scandir
    swapped = False

    def swap_then_scan(path: object):
        nonlocal swapped
        if not swapped:
            swapped = True
            extract_dir.rename(detached)
            extract_dir.symlink_to(outside, target_is_directory=True)
        return original_scandir(path)

    monkeypatch.setattr(pipeline_router.os, "scandir", swap_then_scan)

    pipeline_router._clean_materials_extract_dir(extract_dir, preserve)

    assert (detached / "materials.zip").read_bytes() == b"zip"
    assert not (detached / "stale.txt").exists()
    assert not (detached / "stale").exists()
    assert outside_sentinel.read_text(encoding="utf-8") == ("outside-cleanup-sentinel")


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_format", ["yaml", "zip"])
async def test_poisoned_historical_materials_fail_before_session_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    historical_format: str,
) -> None:
    manager = SessionManager(tmp_path / historical_format)
    pipeline_router.set_session_manager(manager)
    pipeline_router.get_event_bus().set_session_manager(manager)
    registry = _install_direct_pipeline_stubs(monkeypatch)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "ready"})
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )
    materials_dir = session_dir / "materials"
    materials_dir.mkdir(exist_ok=True)
    sentinel = f"sentinel-historical-{historical_format}-secret"
    manifest = (
        f"library_path: library.usda\napi_key: {sentinel}\nentries:\n  - name: Unsafe\n"
    )
    if historical_format == "yaml":
        (materials_dir / "materials.yaml").write_text(manifest, encoding="utf-8")
        (materials_dir / "library.usda").write_text("#usda 1.0\n", encoding="utf-8")
    else:
        with zipfile.ZipFile(materials_dir / "materials.zip", "w") as archive:
            archive.writestr("materials.yaml", manifest)
            archive.writestr("library.usda", "#usda 1.0\n")
    before_metadata = json.loads(
        json.dumps(await manager.get_session_metadata(session_id))
    )
    before_dirs = {
        path.relative_to(session_dir).as_posix()
        for path in session_dir.rglob("*")
        if path.is_dir()
    }
    before_files = {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in session_dir.rglob("*")
        if path.is_file()
    }

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.create_pipeline(
                **_direct_pipeline_kwargs(
                    session_id=session_id,
                    optimize_usd="false",
                )
            )

    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"
    assert await manager.get_session_metadata(session_id) == before_metadata
    assert {
        path.relative_to(session_dir).as_posix()
        for path in session_dir.rglob("*")
        if path.is_dir()
    } == before_dirs
    assert {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in session_dir.rglob("*")
        if path.is_file()
    } == before_files
    assert registry.registered == []
    assert sentinel not in str(exc_info.value.detail)
    assert sentinel not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hostile_key",
    [
        "materials/../escape/materials.yaml",
        r"materials\..\escape\materials.yaml",
        "/materials/escape/materials.yaml",
        "materials/C:/escape/materials.yaml",
        "materials//escape/materials.yaml",
        "materials/./materials.yaml",
        "materials/.pipeline_temp/materials.yaml",
    ],
)
async def test_hostile_historical_material_key_fails_before_session_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_key: str,
) -> None:
    manager = SessionManager(tmp_path / "hostile-key")
    pipeline_router.set_session_manager(manager)
    pipeline_router.get_event_bus().set_session_manager(manager)
    registry = _install_direct_pipeline_stubs(monkeypatch)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "ready"})
    (session_dir / "input" / "scene.usda").write_text(
        "#usda 1.0\n",
        encoding="utf-8",
    )

    async def hostile_listing(*args: object, **kwargs: object) -> list[str]:
        return [hostile_key]

    read_attempted = False

    async def fail_if_read(*args: object, **kwargs: object) -> object:
        nonlocal read_attempted
        read_attempted = True
        raise AssertionError("hostile keys must be rejected before reads")

    monkeypatch.setattr(manager.store, "list_keys", hostile_listing)
    monkeypatch.setattr(manager.store, "open_read", fail_if_read)
    before_metadata = json.loads(
        json.dumps(await manager.get_session_metadata(session_id))
    )
    before_files = {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in session_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=session_id,
                optimize_usd="false",
            )
        )

    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"
    assert not read_attempted
    assert await manager.get_session_metadata(session_id) == before_metadata
    assert {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in session_dir.rglob("*")
        if path.is_file()
    } == before_files
    assert pipeline_router.get_event_bus().get_snapshot(session_id) is None
    assert registry.registered == []


def test_historical_material_manifest_parse_failures_are_value_free(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "historical-material-yaml-sentinel-727"
    malformed = f"materials:\n  description: {sentinel}\n  entries: [\n".encode()
    session_dir = tmp_path / "session"

    with caplog.at_level(logging.ERROR), pytest.raises(HTTPException) as exc_info:
        pipeline_router._plan_material_manifest(
            malformed,
            "materials/materials.yaml",
            {"materials/materials.yaml", "materials/materials.usda"},
            session_dir,
        )

    assert exc_info.value.detail == "Invalid saved materials.yaml"
    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text
    assert "code=material_manifest_parse_failed" in caplog.text
    assert "phase=persistence_verification" in caplog.text

    caplog.clear()
    cached_manifest = (
        session_dir / "cache" / "generated_material_library" / "materials.yaml"
    )
    cached_manifest.parent.mkdir(parents=True)
    cached_manifest.write_bytes(malformed)
    with caplog.at_level(logging.ERROR):
        assert (
            pipeline_router._load_cached_generated_material_library(session_dir) is None
        )
    assert sentinel not in caplog.text
    assert "code=generated_material_manifest_parse_failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["terminal", "active-same-pod", "active-cross-pod"])
async def test_existing_pipeline_start_rejects_non_ready_session_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    shared_store = tmp_path / "shared"
    owner = SessionManager(
        tmp_path / "pod-a",
        store=LocalSessionStore(str(shared_store)),
    )
    other = SessionManager(
        tmp_path / "pod-b",
        store=LocalSessionStore(str(shared_store)),
    )
    session_id = str(uuid4())
    await owner.create_session(session_id)
    await owner.update_session(
        session_id,
        {"status": "completed", "config": {"sentinel": "unchanged"}},
        sync_files=False,
    )
    if mode != "terminal":
        planned = await owner.get_session_metadata_versioned(session_id)
        assert planned.version is not None
        await owner.claim_regeneration(
            session_id,
            expected_version=planned.version,
            lease_seconds=60,
        )

    target = other if mode == "active-cross-pod" else owner
    before = await target.get_session_metadata(session_id)
    pipeline_router.set_session_manager(target)
    _install_direct_pipeline_stubs(monkeypatch)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.create_pipeline(
            **_direct_pipeline_kwargs(
                session_id=session_id,
                optimize_usd="false",
            )
        )
    _expect_http(409, exc_info)
    assert await target.get_session_metadata(session_id) == before


@pytest.mark.asyncio
async def test_results_events_regenerate_and_session_icon_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    pipeline_router.set_session_manager(manager)
    event_bus = pipeline_router.get_event_bus()
    event_bus.set_session_manager(manager)
    registry = _install_direct_pipeline_stubs(monkeypatch)
    monkeypatch.setattr(pipeline_router.config, "nvidia_api_key", "test-key")

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_pipeline_results(str(uuid4()))
    _expect_http(404, exc_info)

    sid = str(uuid4())
    session_dir = await manager.create_session(sid)
    await manager.update_session(sid, {"status": "pending"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_pipeline_results(sid)
    _expect_http(202, exc_info)
    await manager.update_session(
        sid,
        {
            "status": "failed",
            "error": "bad",
            "failed_step": "predict",
            "failed_at": "2026-01-01T00:00:00+00:00",
            "completed_steps": [{"name": "build_dataset_usd"}],
        },
    )
    failed = await pipeline_router.get_pipeline_results(sid)
    assert failed.error_message == "bad"
    assert failed.download_urls == {}

    (session_dir / "output" / "scene_with_materials.usd").write_text("#usda 1.0\n")
    (session_dir / "cache" / "predictions" / "predictions.jsonl").write_text("{}\n")
    (session_dir / "cache" / "dataset" / "dataset.jsonl").write_text("{}\n")
    (session_dir / "cache" / "clusters").mkdir(parents=True, exist_ok=True)
    (session_dir / "cache" / "clusters" / "cluster_map.jsonl").write_text("{}\n")
    await manager.update_session(
        sid,
        {
            "status": "failed",
            "failed_step": "coverage_validation",
            "results": {"prims_processed": 1, "cluster_prims_ran": True},
        },
    )
    failed_with_artifacts = await pipeline_router.get_pipeline_results(sid)
    assert failed_with_artifacts.download_urls == {
        "output_usd": f"/artifacts/{sid}/output",
        "predictions": f"/artifacts/{sid}/predictions",
        "report": f"/artifacts/{sid}/report",
        "cluster_map": f"/artifacts/{sid}/cluster-map",
    }
    await manager.update_session(
        sid,
        {
            "status": "completed",
            "results": {"prims_processed": 1, "cluster_prims_ran": True},
            "coverage": None,
            "duration_seconds": 1,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    completed = await pipeline_router.get_pipeline_results(sid)
    assert "output_usd" in completed.download_urls
    assert "cluster_map" in completed.download_urls

    scene_sid = str(uuid4())
    await manager.create_session(scene_sid)
    await manager.update_session(
        scene_sid,
        {
            "status": "completed",
            "pipeline_type": "large_scene",
            "results": {"prims_processed": 1},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "scene": {
                "validation_report_path": "scene/validation_report.json",
                "scene_predictions_path": "scene/predictions.jsonl",
            },
        },
    )
    scene_results = await pipeline_router.get_pipeline_results(scene_sid)
    assert "scene_predictions" in scene_results.download_urls

    failed_scene_sid = str(uuid4())
    await manager.create_session(failed_scene_sid)
    await manager.update_session(
        failed_scene_sid,
        {
            "status": "failed",
            "pipeline_type": "large_scene",
            "error": "scene validation failed",
            "failed_step": "scene_validation",
            "failed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    failed_scene_results = await pipeline_router.get_pipeline_results(failed_scene_sid)
    assert failed_scene_results.download_urls == {}

    empty_completed_sid = str(uuid4())
    await manager.create_session(empty_completed_sid)
    await manager.update_session(
        empty_completed_sid,
        {
            "status": "completed",
            "results": {"prims_processed": 1},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    empty_completed = await pipeline_router.get_pipeline_results(empty_completed_sid)
    assert empty_completed.download_urls == {}

    class _VanishingResultsManager:
        class _Store:
            kind = "local"

        def __init__(self, session_dir: Path) -> None:
            self.session_dir = session_dir
            self.calls = 0
            self.store = self._Store()

        @staticmethod
        def resolve_published_artifact_key(
            metadata: dict,
            logical_name: str,
            *,
            legacy_key: str | None = None,
        ) -> str | None:
            return legacy_key

        @staticmethod
        def resolve_prediction_report_key(
            metadata: dict,
            *,
            legacy_key: str | None = None,
        ) -> str | None:
            return legacy_key

        async def sync_session_to_store(self, session_id: str) -> None:
            return None

        async def get_session_metadata(self, session_id: str) -> dict | None:
            self.calls += 1
            if self.calls == 1:
                return {"status": "completed", "results": {}}
            return None

        def get_session_dir(self, session_id: str) -> Path:
            return self.session_dir

    async def no_sleep(delay: float) -> None:
        return None

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _VanishingResultsManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.get_pipeline_results("sid")
        _expect_http(404, exc_info)

    class _DelayedResultsManager(_VanishingResultsManager):
        async def get_session_metadata(self, session_id: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"status": "completed", "results": {}}
            return {
                "status": "completed",
                "results": {"prims_processed": 2},
                "coverage": None,
                "completed_at": "2026-01-01T00:00:00+00:00",
            }

        async def exists_in_store(self, session_id: str, key: str) -> bool:
            return False

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _DelayedResultsManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        delayed = await pipeline_router.get_pipeline_results("sid")
        assert delayed.stats["prims_processed"] == 2

    class _FinalizingResultsManager(_DelayedResultsManager):
        async def get_session_metadata(self, session_id: str) -> dict:
            return {"status": "completed", "results": {}}

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _FinalizingResultsManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.get_pipeline_results("sid")
        _expect_http(202, exc_info)
        assert "finalizing" in str(exc_info.value.detail)

    class _FailedAfterProvisionalCompletionManager(_DelayedResultsManager):
        async def get_session_metadata(self, session_id: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"status": "completed", "results": {}}
            return {
                "status": "failed",
                "error": "strict coverage failed",
                "failed_step": "coverage_validation",
                "failed_at": "2026-01-01T00:00:00+00:00",
                "results": {"prims_processed": 1},
                "partial_results": {
                    "stats": {"prims_processed": 1},
                    "coverage": None,
                },
                "coverage": None,
            }

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _FailedAfterProvisionalCompletionManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        failed = await pipeline_router.get_pipeline_results("sid")
        assert failed.status == "failed"
        assert failed.failed_step == "coverage_validation"

    class _DelayedFailureManager(_DelayedResultsManager):
        async def get_session_metadata(self, session_id: str) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"status": "failed"}
            return {
                "status": "failed",
                "error": "prediction failed",
                "failed_step": "predict",
                "failed_at": "2026-01-01T00:00:00+00:00",
            }

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _DelayedFailureManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        failed = await pipeline_router.get_pipeline_results("sid")
        assert failed.status == "failed"
        assert failed.error_message == "prediction failed"

    class _FinalizingFailureManager(_DelayedResultsManager):
        async def get_session_metadata(self, session_id: str) -> dict:
            return {"status": "failed"}

    with monkeypatch.context() as m:
        m.setattr(
            pipeline_router,
            "get_session_manager",
            lambda: _FinalizingFailureManager(tmp_path),
        )
        m.setattr(pipeline_router.asyncio, "sleep", no_sleep)
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.get_pipeline_results("sid")
        _expect_http(202, exc_info)
        assert "failure diagnostics" in str(exc_info.value.detail)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.stream_progress_events(str(uuid4()))
    _expect_http(404, exc_info)
    running_sid = str(uuid4())
    await manager.create_session(running_sid)
    await manager.update_session(running_sid, {"status": "running"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.stream_progress_events(running_sid)
    _expect_http(503, exc_info)

    stream_sid = str(uuid4())
    await manager.create_session(stream_sid)
    await event_bus.seed_pending_session(stream_sid)
    response = await pipeline_router.stream_progress_events(stream_sid)
    await event_bus.emit(
        ProgressEvent(
            session_id=stream_sid,
            step="pipeline",
            state=StepState.FAILED,
            percent=0,
            message="done",
            extra={"pipeline_failed": True},
        )
    )
    first = await anext(response.body_iterator)
    second = await anext(response.body_iterator)
    assert first["event"] == "progress"
    assert second["event"] == "done"
    with pytest.raises(StopAsyncIteration):
        await anext(response.body_iterator)

    class _StreamBus:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        def get_snapshot(self, session_id: str) -> dict:
            return {"status": "running"}

        async def get_fenced_snapshot(self, session_id: str) -> dict:
            return self.get_snapshot(session_id)

        async def event_is_current(self, event: ProgressEvent) -> bool:
            return True

        async def queued_event_is_current(self, event: ProgressEvent) -> bool:
            return True

        def get_queue(self, session_id: str) -> asyncio.Queue[ProgressEvent]:
            return self.queue

    provisional_bus = _StreamBus()
    await provisional_bus.queue.put(
        ProgressEvent(
            session_id="stream-provisional",
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            overall_percent=100,
            extra={"pipeline_completed": True},
        )
    )
    await provisional_bus.queue.put(
        ProgressEvent(
            session_id="stream-provisional",
            step="coverage_validation",
            state=StepState.FAILED,
            percent=100,
            overall_percent=100,
        )
    )
    await provisional_bus.queue.put(
        ProgressEvent(
            session_id="stream-provisional",
            step="coverage_validation",
            state=StepState.FAILED,
            percent=100,
            overall_percent=100,
            extra={"pipeline_failed": True, "coverage": {}},
        )
    )
    with monkeypatch.context() as m:
        m.setattr(pipeline_router, "get_event_bus", lambda: provisional_bus)
        provisional_stream = await pipeline_router.stream_progress_events(
            "stream-provisional"
        )
        assert (await anext(provisional_stream.body_iterator))["event"] == "progress"
        assert (await anext(provisional_stream.body_iterator))["event"] == "progress"
        assert (await anext(provisional_stream.body_iterator))["event"] == "progress"
        assert (await anext(provisional_stream.body_iterator))["event"] == "done"

    completed_bus = _StreamBus()
    await completed_bus.queue.put(
        ProgressEvent(
            session_id="stream-completed",
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            overall_percent=100,
            extra={"pipeline_completed": True, "coverage": {}},
        )
    )
    with monkeypatch.context() as m:
        m.setattr(pipeline_router, "get_event_bus", lambda: completed_bus)
        completed_stream = await pipeline_router.stream_progress_events(
            "stream-completed"
        )
        assert (await anext(completed_stream.body_iterator))["event"] == "progress"
        assert (await anext(completed_stream.body_iterator))["event"] == "done"
        with pytest.raises(StopAsyncIteration):
            await anext(completed_stream.body_iterator)

    terminal_cancel_bus = _StreamBus()
    await terminal_cancel_bus.queue.put(
        ProgressEvent(
            session_id="stream-terminal-cancel",
            step="predict",
            state=StepState.CANCELLED,
        )
    )
    with monkeypatch.context() as m:
        m.setattr(pipeline_router, "get_event_bus", lambda: terminal_cancel_bus)
        terminal_cancel_stream = await pipeline_router.stream_progress_events(
            "stream-terminal-cancel"
        )
        assert (await anext(terminal_cancel_stream.body_iterator))[
            "event"
        ] == "progress"
        assert (await anext(terminal_cancel_stream.body_iterator))["event"] == "done"

    timeout_bus = _StreamBus()

    async def timeout_wait_for(awaitable: object, timeout: float) -> object:
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise TimeoutError

    with monkeypatch.context() as m:
        m.setattr(pipeline_router, "get_event_bus", lambda: timeout_bus)
        m.setattr(pipeline_router.asyncio, "wait_for", timeout_wait_for)
        ping_stream = await pipeline_router.stream_progress_events("stream-timeout")
        assert (await anext(ping_stream.body_iterator)) == {
            "event": "ping",
            "data": "keepalive",
        }

    async def cancelled_wait_for(awaitable: object, timeout: float) -> object:
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise asyncio.CancelledError

    with monkeypatch.context() as m:
        m.setattr(pipeline_router, "get_event_bus", lambda: _StreamBus())
        m.setattr(pipeline_router.asyncio, "wait_for", cancelled_wait_for)
        cancelled_stream = await pipeline_router.stream_progress_events("stream-cancel")
        with pytest.raises(asyncio.CancelledError):
            await anext(cancelled_stream.body_iterator)

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            str(uuid4()),
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    _expect_http(404, exc_info)
    regen_sid = str(uuid4())
    regen_dir = await manager.create_session(regen_sid)
    await manager.update_session(regen_sid, {"status": "running"})
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            regen_sid,
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    _expect_http(400, exc_info)
    await manager.update_session(
        regen_sid,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "config": {
                "camera_views": ["+x"],
                "enable_prim_clustering": True,
                "enable_material_generation": True,
                "render_num_workers": 1,
            },
            "user_email": "user@nvidia.com",
        },
    )
    (regen_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    (regen_dir / "input" / "reference_images" / "reference_0000.png").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (regen_dir / "input" / "reference_images" / "reference_0000.png").write_bytes(
        b"png"
    )
    (regen_dir / "input" / "reference_pdfs" / "reference_0000.pdf").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (regen_dir / "input" / "reference_pdfs" / "reference_0000.pdf").write_bytes(b"pdf")
    regen_dataset = regen_dir / "cache" / "dataset" / "dataset.jsonl"
    regen_dataset.parent.mkdir(parents=True, exist_ok=True)
    regen_dataset.write_text('{"id": "/Root"}\n')
    regen_usd_dataset = regen_dir / "cache" / "dataset" / "usd"
    regen_usd_dataset.mkdir(parents=True, exist_ok=True)
    (regen_usd_dataset / "prims.jsonl").write_text('{"id": "/Root"}\n')
    (regen_dir / "cache" / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "completed_steps": [
                    "build_dataset_usd",
                    "build_dataset_prepare_dataset",
                ],
                "step_outputs": {
                    "build_dataset_usd": {
                        "output_dir": str(regen_usd_dataset),
                    },
                    "build_dataset_prepare_dataset": {
                        "dataset_jsonl_path": str(regen_dataset),
                    },
                },
            }
        )
    )
    regenerated = await pipeline_router.regenerate_pipeline(
        regen_sid,
        RegenerateRequest(
            steps=[PipelineStep.PREDICT, PipelineStep.APPLY],
            user_prompt="new prompt",
            layer_only=True,
        ),
    )
    assert regenerated.status == "pending"
    assert registry.registered[-1][0] == regen_sid

    no_input_regen_sid = str(uuid4())
    await manager.create_session(no_input_regen_sid)
    await manager.update_session(
        no_input_regen_sid,
        {
            "status": "completed",
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.regenerate_pipeline(
            no_input_regen_sid,
            RegenerateRequest(steps=[PipelineStep.PREDICT]),
        )
    _expect_http(400, exc_info)

    zip_regen_sid = str(uuid4())
    zip_regen_dir = await manager.create_session(zip_regen_sid)
    (zip_regen_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    materials_zip = zip_regen_dir / "materials" / "materials.zip"
    with zipfile.ZipFile(materials_zip, "w") as zf:
        zf.writestr("materials.yaml", "library_path: lib.usda\nentries:\n  - name: A\n")
        zf.writestr("lib.usda", "#usda 1.0\n")
    await manager.update_session(
        zip_regen_sid,
        {
            "status": "completed",
            "config": {},
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    zip_regenerated = await pipeline_router.regenerate_pipeline(
        zip_regen_sid,
        RegenerateRequest(steps=[PipelineStep.BUILD_DATASET]),
    )
    assert zip_regenerated.status == "pending"

    yaml_regen_sid = str(uuid4())
    yaml_regen_dir = await manager.create_session(yaml_regen_sid)
    (yaml_regen_dir / "input" / "scene.usda").write_text("#usda 1.0\n")
    (yaml_regen_dir / "materials" / "materials.yaml").write_text(
        "library_path: lib.usda\nentries:\n  - name: B\n"
    )
    (yaml_regen_dir / "materials" / "lib.usda").write_text("#usda 1.0\n")
    await manager.update_session(
        yaml_regen_sid,
        {
            "status": "completed",
            "config": {},
            "results": {},
            "coverage": None,
            "completed_at": "2026-01-01T00:00:00+00:00",
        },
    )
    yaml_regenerated = await pipeline_router.regenerate_pipeline(
        yaml_regen_sid,
        RegenerateRequest(steps=[PipelineStep.BUILD_DATASET]),
    )
    assert yaml_regenerated.status == "pending"

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_event_log(str(uuid4()))
    _expect_http(404, exc_info)
    assert await pipeline_router.get_event_log(sid) == {"events": []}
    (session_dir / "event_log.jsonl").write_text('{"event": 1}\n\n{"event": 2}\n')
    event_log = await pipeline_router.get_event_log(sid)
    assert event_log["total"] == 2
    event_sentinel = "sentinel-material-event-secret"
    (session_dir / "event_log.jsonl").write_text(
        f"{{{event_sentinel}\n", encoding="utf-8"
    )
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.get_event_log(sid)
    _expect_http(500, exc_info)
    assert exc_info.value.detail == "Failed to load event log"
    assert "event_log_local_read_failed" in caplog.text
    assert event_sentinel not in str(exc_info.value.detail)
    assert event_sentinel not in caplog.text

    icon_sid = str(uuid4())
    icon_dir = await manager.create_session(icon_sid)
    pipeline_router.shutil.rmtree(icon_dir / "materials")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(404, exc_info)
    (icon_dir / "materials").mkdir()
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(404, exc_info)
    subdir = icon_dir / "materials" / "pack"
    subdir.mkdir(parents=True)
    (subdir / "library.usda").write_text("#usda 1.0\n")
    (subdir / "materials.yaml").write_text(
        "materials:\n  library_path: library.usda\n  entries:\n"
        "    - name: Aluminum\n      icon: thumbs/a.png\n"
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(404, exc_info)
    (subdir / "materials.yaml").write_text(
        "materials:\n  library_path: library.usda\n  entries:\n"
        "    - bad-entry\n    - name: Aluminum\n      icon: thumbs/a.png\n"
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Missing")
    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"
    (subdir / "materials.yaml").write_text(
        "materials:\n  library_path: library.usda\n  entries:\n"
        "    - name: Aluminum\n      icon: thumbs/a.png\n"
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Missing")
    _expect_http(404, exc_info)
    assert exc_info.value.detail == "Icon not found for material"
    (subdir / "thumbs").mkdir()
    (subdir / "thumbs" / "a.png").write_bytes(b"png")
    icon_response = await pipeline_router.get_session_material_icon(
        icon_sid, "Aluminum"
    )
    assert icon_response.media_type == "image/png"
    icon_path = subdir / "thumbs" / "a.png"
    icon_path.rename(icon_path.with_name("a.held.png"))
    icon_secret = icon_dir / "cache" / ".pipeline_temp" / "icon.png"
    icon_secret.parent.mkdir(parents=True)
    icon_secret.write_bytes(b"sentinel-material-icon-secret")
    icon_path.symlink_to(icon_secret)
    icon_body = await _response_body(icon_response)
    assert icon_body == b"png"
    assert b"sentinel-material-icon-secret" not in icon_body
    (icon_dir / "materials" / "secret.png").write_bytes(b"png")
    (subdir / "materials.yaml").write_text(
        "materials:\n  library_path: library.usda\n  entries:\n"
        "    - name: Secret\n      icon: ../secret.png\n"
    )
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Secret")
    _expect_http(403, exc_info)
    manifest_sentinel = "sentinel-material-manifest-secret"
    (subdir / "materials.yaml").write_text(
        "materials:\n  library_path: library.usda\n  api_key: "
        f"{manifest_sentinel}\n  entries:\n"
        "    - name: Aluminum\n      icon: thumbs/a.png\n"
    )
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=pipeline_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"
    assert "material_manifest_security_failed" in caplog.text
    assert manifest_sentinel not in str(exc_info.value.detail)
    assert manifest_sentinel not in caplog.text

    (subdir / "materials.yaml").write_text("[")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"

    external_pack = tmp_path / "external-material-pack"
    external_pack.mkdir()
    (external_pack / "library.usda").write_text("#usda 1.0\n")
    (external_pack / "materials.yaml").write_text(
        "library_path: library.usda\nentries:\n  - name: Aluminum\n    icon: icon.png\n"
    )
    (external_pack / "icon.png").write_bytes(b"external")
    pipeline_router.shutil.rmtree(subdir)
    subdir.symlink_to(external_pack, target_is_directory=True)
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(400, exc_info)
    assert exc_info.value.detail == "Invalid saved materials.yaml"

    subdir.unlink()
    subdir.mkdir()
    (subdir / "library.usda").write_text("#usda 1.0\n")
    (subdir / "materials.yaml").write_text(
        "library_path: library.usda\nentries:\n  - name: Aluminum\n    icon: icon.png\n"
    )
    (subdir / "icon.png").symlink_to(external_pack / "icon.png")
    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router.get_session_material_icon(icon_sid, "Aluminum")
    _expect_http(404, exc_info)
    assert exc_info.value.detail == "Icon file not found"


@pytest.mark.asyncio
async def test_executor_wrapper_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path)
    sid = str(uuid4())
    await manager.create_session(sid)
    event_bus = EventBus()
    event_bus.set_session_manager(manager)
    await event_bus.seed_pending_session(sid)
    monkeypatch.setattr(executor, "get_event_bus", lambda: event_bus)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: event_bus)
    monkeypatch.setattr(listener_module, "get_event_bus", lambda: event_bus)

    async def cancel_inner(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(executor, "_execute_pipeline_inner", cancel_inner)
    with pytest.raises(RuntimeError):
        await executor.execute_pipeline_async(sid, {}, manager)
    failed_metadata = await manager.get_session_metadata(sid)
    assert failed_metadata["status"] == "failed"
    assert failed_metadata["error"] == "material_pipeline_failed"
    assert failed_metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "material_pipeline_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "boom" not in json.dumps(failed_metadata)
    pipeline_failure = await event_bus.get_queue(sid).get()
    assert pipeline_failure.step == "pipeline"
    assert pipeline_failure.message == "material_pipeline_failed"
    assert pipeline_failure.extra == {"pipeline_failed": True}
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=sid,
            step="late_standard_progress",
            state=StepState.RUNNING,
        ),
        regeneration_claim=None,
    )
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=sid,
            step="late_standard_failure",
            state=StepState.FAILED,
        ),
        regeneration_claim=None,
    )
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=sid,
            step="pipeline",
            state=StepState.COMPLETED,
            extra={"pipeline_completed": True, "coverage": {}},
        ),
        regeneration_claim=None,
    )
    assert event_bus.get_queue(sid).empty()
    standard_snapshot = event_bus.get_snapshot(sid)
    assert standard_snapshot is not None
    assert standard_snapshot["status"] == "failed"

    async def cancelled_inner(*args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError()

    import asyncio

    monkeypatch.setattr(executor, "_execute_pipeline_inner", cancelled_inner)
    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(sid, {}, manager)
    assert (await manager.get_session_metadata(sid))["status"] == "cancelled"
    cancelled_snapshot = event_bus.get_snapshot(sid)
    assert cancelled_snapshot is not None
    assert cancelled_snapshot["status"] == "cancelled"
    cancelled_stream = await pipeline_router.stream_progress_events(sid)
    cancelled_progress = await anext(cancelled_stream.body_iterator)
    assert cancelled_progress["event"] == "progress"
    assert json.loads(cancelled_progress["data"])["state"] == "cancelled"
    cancelled_done = await anext(cancelled_stream.body_iterator)
    assert cancelled_done["event"] == "done"
    with pytest.raises(StopAsyncIteration):
        await anext(cancelled_stream.body_iterator)

    durable_sid = str(uuid4())
    await manager.create_session(durable_sid)
    durable_started = asyncio.Event()

    async def block_until_durable_cancel(*args: object, **kwargs: object) -> None:
        durable_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        executor,
        "_execute_pipeline_inner",
        block_until_durable_cancel,
    )
    durable_worker = asyncio.create_task(
        executor.execute_pipeline_async(durable_sid, {}, manager)
    )
    await durable_started.wait()
    await manager.request_cancellation(durable_sid)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(durable_worker, timeout=2)
    assert (await manager.get_session_metadata(durable_sid))["status"] == "cancelled"

    scene_sid = str(uuid4())
    await manager.create_session(scene_sid)
    await event_bus.seed_pending_session(scene_sid)
    monkeypatch.setattr(executor, "_execute_scene_pipeline_inner", cancel_inner)
    with pytest.raises(RuntimeError):
        await executor.execute_scene_pipeline_async(scene_sid, {}, manager)
    scene_metadata = await manager.get_session_metadata(scene_sid)
    assert scene_metadata["status"] == "failed"
    assert scene_metadata["error"] == "material_scene_pipeline_failed"
    assert scene_metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "material_scene_pipeline_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "boom" not in json.dumps(scene_metadata)
    assert "terminal_events_quiesced" not in scene_metadata
    scene_failure = await event_bus.get_queue(scene_sid).get()
    assert scene_failure.step == "scene_pipeline"
    assert scene_failure.message == "material_scene_pipeline_failed"
    assert scene_failure.extra == {"pipeline_failed": True}
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=scene_sid,
            step="late_scene_progress",
            state=StepState.RUNNING,
        ),
        regeneration_claim=None,
    )
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=scene_sid,
            step="late_scene_failure",
            state=StepState.FAILED,
        ),
        regeneration_claim=None,
    )
    assert event_bus.get_queue(scene_sid).empty()
    scene_snapshot = event_bus.get_snapshot(scene_sid)
    assert scene_snapshot is not None
    assert scene_snapshot["status"] == "failed"

    monkeypatch.setattr(executor, "_execute_scene_pipeline_inner", cancelled_inner)
    with pytest.raises(asyncio.CancelledError):
        await executor.execute_scene_pipeline_async(scene_sid, {}, manager)
    assert (await manager.get_session_metadata(scene_sid))["status"] == "cancelled"
    scene_cancelled = await event_bus.get_queue(scene_sid).get()
    assert scene_cancelled.step == "scene_pipeline"
    assert scene_cancelled.state == StepState.CANCELLED
    assert scene_cancelled.extra == {"pipeline_cancelled": True}

    regeneration_sid = str(uuid4())
    await manager.create_session(regeneration_sid)
    await manager.update_session(regeneration_sid, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(regeneration_sid)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        regeneration_sid,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    await event_bus.seed_pending_session(
        regeneration_sid,
        regeneration_claim=claim,
    )
    stale_listener = FastAPIEventListener(
        regeneration_sid,
        loop=asyncio.get_running_loop(),
        regeneration_claim=claim,
    )
    monkeypatch.setattr(executor, "_execute_pipeline_inner", cancelled_inner)
    with pytest.raises(asyncio.CancelledError):
        await executor.execute_pipeline_async(
            regeneration_sid,
            {},
            manager,
            regeneration_claim=claim,
        )
    regeneration_metadata = await manager.get_session_metadata(regeneration_sid)
    assert regeneration_metadata["status"] == "cancelled"
    regeneration_cancelled = await event_bus.get_queue(regeneration_sid).get()
    assert regeneration_cancelled.step == "pipeline"
    assert regeneration_cancelled.state == StepState.CANCELLED
    assert regeneration_cancelled.extra == {"pipeline_cancelled": True}

    rebound_plan = await manager.get_session_metadata_versioned(regeneration_sid)
    assert rebound_plan.version is not None
    rebound_claim = await manager.claim_regeneration(
        regeneration_sid,
        expected_version=rebound_plan.version,
        updates={"status": "running"},
    )
    await executor._emit_persisted_pipeline_cancellation(
        regeneration_sid,
        step="pipeline",
        regeneration_claim=claim,
    )
    assert event_bus.get_queue(regeneration_sid).empty()
    await event_bus.seed_pending_session(
        regeneration_sid,
        regeneration_claim=rebound_claim,
    )
    stale_listener.event(
        "step.progress",
        {
            "step_name": "apply",
            "current": 1,
            "total": 2,
            "percent": 50,
        },
    )
    stale_listener.event(
        "step.failed",
        {"step_name": "apply", "error": "late generation-1 failure"},
    )
    stale_listener.event("step.cancelled", {"step_name": "apply"})
    stale_listener.event("task.cancelled", {"task_name": "ApplyTask"})
    stale_listener.event("workflow.cancelled", {"step_name": "apply"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not await event_bus.emit_for_owner(
        ProgressEvent(
            session_id=regeneration_sid,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True, "coverage": {}},
        ),
        regeneration_claim=claim,
    )
    await executor._emit_persisted_pipeline_cancellation(
        regeneration_sid,
        step="pipeline",
        regeneration_claim=claim,
    )
    await executor._emit_persisted_pipeline_cancellation(
        regeneration_sid,
        step="pipeline",
        regeneration_claim=None,
    )
    rebound_metadata = await manager.get_session_metadata(regeneration_sid)
    assert rebound_metadata is not None
    assert rebound_metadata["status"] == "running"
    assert rebound_metadata["regeneration_claim"]["active"] is True
    assert rebound_metadata["regeneration_claim"]["token"] == rebound_claim.token
    rebound_snapshot = event_bus.get_snapshot(regeneration_sid)
    assert rebound_snapshot is not None
    assert rebound_snapshot["status"] == "pending"
    assert event_bus.get_queue(regeneration_sid).empty()

    current_event = ProgressEvent(
        session_id=regeneration_sid,
        step="apply",
        state=StepState.RUNNING,
        percent=1,
    )
    assert await event_bus.emit_for_owner(
        current_event,
        regeneration_claim=rebound_claim,
    )
    assert await event_bus.get_queue(regeneration_sid).get() is current_event
    current_snapshot = event_bus.get_snapshot(regeneration_sid)
    assert current_snapshot is not None
    assert current_snapshot["status"] == "running"
    assert current_snapshot["current_step"]["name"] == "apply"

    strict_sid = str(uuid4())
    await manager.create_session(strict_sid)
    with pytest.raises(ValueError, match="requires the single-asset pipeline"):
        await executor.execute_scene_pipeline_async(
            strict_sid,
            {},
            manager,
            coverage_policy="strict",
        )
    strict_metadata = await manager.get_session_metadata(strict_sid)
    assert strict_metadata["status"] == "failed"
    assert strict_metadata["error"] == "material_scene_pipeline_failed"


class _Span:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _Telemetry:
    def __init__(self, inner: object) -> None:
        self.inner = inner

    def get_step_timings(self) -> list[dict[str, object]]:
        return [
            {
                "name": "predict",
                "status": "completed",
                "started_at_ns": 1,
                "completed_at_ns": 1_000_000_001,
            }
        ]


@pytest.mark.asyncio
async def test_executor_pipeline_inner_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "material-service-result-credential-713"
    caplog.set_level(logging.DEBUG, logger=executor.__name__)
    manager = SessionManager(tmp_path)
    sid = str(uuid4())
    session_dir = await manager.create_session(sid)
    await manager.update_session(
        sid,
        {
            "asset": {
                "filename": "scene.usd",
                "file_size_bytes": 5,
                "file_extension": ".usd",
            }
        },
    )
    span = _Span()
    monkeypatch.setattr(executor, "get_current_span", lambda: span)
    monkeypatch.setattr(executor, "TelemetryEventListener", _Telemetry)
    monkeypatch.setattr(executor, "get_event_bus", lambda: _FailingBus())
    await manager.update_session(
        sid,
        {
            "completed_steps": [{"name": "cached", "display_name": "Cached"}],
            "artifact_validity": initial_artifact_validity(),
        },
    )

    async def arun_success(pipeline_input: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["predict"],
            step_results={
                "predict": {
                    "predictions_count": 2,
                    "diagnostics": {"api_key": sentinel},
                }
            },
            raw_result={
                "dataset_info": {"num_entries": 2},
                "config_dict": {"api_key": sentinel},
            },
        )

    monkeypatch.setattr(executor, "arun_pipeline", arun_success)
    await executor._execute_pipeline_inner(
        sid,
        {
            "materials": {"entries": [{"name": "A", "icon": "a.png"}]},
            "steps": {"predict": {"vlm": {"model": "model"}}},
        },
        manager,
        user_email="user@nvidia.com",
    )
    metadata = await manager.get_session_metadata(sid)
    assert metadata["status"] == "completed"
    assert metadata["results"]["predictions_made"] == 2
    assert span.attributes["maa.pipeline.status"] == "completed"
    assert metadata["completed_steps"][0]["name"] == "cached"
    assert metadata["artifact_validity"]["restored_predictions"] is False
    assert sentinel not in caplog.text
    assert sentinel not in json.dumps(metadata, default=str)

    restored_path = session_dir / "cache" / "restored" / "restored_predictions.jsonl"
    restored_path.parent.mkdir(parents=True, exist_ok=True)

    async def arun_restore(pipeline_input: object) -> SimpleNamespace:
        restored_path.write_text('{"id": "/Root", "material": "Steel"}\n')
        return SimpleNamespace(
            success=True,
            error=None,
            completed_steps=["restore_usd"],
            step_results={
                "restore_usd": {"restored_predictions_path": str(restored_path)}
            },
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", arun_restore)
    await executor._execute_pipeline_inner(
        sid,
        {"steps": {"restore_usd": {}}},
        manager,
    )
    metadata = await manager.get_session_metadata(sid)
    assert metadata["artifact_validity"]["restored_predictions"] is True

    monkeypatch.setattr(
        executor,
        "build_material_coverage",
        lambda *_args, **_kwargs: {
            "policy": "strict",
            "readiness_grade": "partial",
            "usable_prediction_count": 1,
            "fallback_count": 0,
            "target_count": 2,
            "bound_count": 1,
        },
    )
    monkeypatch.setattr(executor, "arun_pipeline", arun_success)
    await executor._execute_pipeline_inner(
        sid,
        {"steps": {"predict": {"vlm": {"model": "model"}}}},
        manager,
        coverage_policy="strict",
    )
    metadata = await manager.get_session_metadata(sid)
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "coverage_validation"
    assert span.attributes["maa.pipeline.status"] == "failed"

    async def arun_failure(pipeline_input: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            error="durable-result-error-sentinel-727",
            completed_steps=[],
            step_results={},
            raw_result={},
        )

    monkeypatch.setattr(executor, "arun_pipeline", arun_failure)
    with pytest.raises(RuntimeError, match="material_pipeline_result_failed"):
        await executor._execute_pipeline_inner(sid, {"steps": {}}, manager)

    regeneration_sid = str(uuid4())
    await manager.create_session(regeneration_sid)
    await manager.update_session(regeneration_sid, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(regeneration_sid)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        regeneration_sid,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    regeneration_bus = EventBus()
    regeneration_bus.set_session_manager(manager)
    await regeneration_bus.seed_pending_session(
        regeneration_sid,
        regeneration_claim=claim,
    )
    monkeypatch.setattr(executor, "get_event_bus", lambda: regeneration_bus)
    monkeypatch.setattr(listener_module, "get_event_bus", lambda: regeneration_bus)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: regeneration_bus)
    monkeypatch.setattr(pipeline_router, "get_session_manager", lambda: manager)
    original_finalize = manager.finalize_regeneration_claim
    finalize_calls = 0

    async def count_finalize(*args: object, **kwargs: object) -> bool:
        nonlocal finalize_calls
        finalize_calls += 1
        return await original_finalize(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(manager, "finalize_regeneration_claim", count_finalize)
    await executor.execute_pipeline_async(
        regeneration_sid,
        {"steps": {}},
        manager,
        regeneration_claim=claim,
    )
    assert finalize_calls == 1
    regeneration_metadata = await manager.get_session_metadata(regeneration_sid)
    assert regeneration_metadata["status"] == "failed"
    assert regeneration_metadata["error"] == "material_pipeline_result_failed"
    assert regeneration_metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "material_pipeline_result_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "durable-result-error-sentinel-727" not in json.dumps(regeneration_metadata)
    failure_stream = await pipeline_router.stream_progress_events(regeneration_sid)
    failure_progress = await anext(failure_stream.body_iterator)
    assert failure_progress["event"] == "progress"
    failure_data = json.loads(failure_progress["data"])
    assert failure_data["state"] == "failed"
    assert failure_data["message"] == "material_pipeline_result_failed"
    assert failure_data["extra"] == {"pipeline_failed": True}
    assert "durable-result-error-sentinel-727" not in json.dumps(failure_data)
    failure_done = await anext(failure_stream.body_iterator)
    assert failure_done["event"] == "done"
    with pytest.raises(StopAsyncIteration):
        await anext(failure_stream.body_iterator)


class _FailingBus:
    def get_snapshot(self, session_id: str) -> dict:
        return {"status": "running"}

    async def emit_for_owner(
        self,
        event: object,
        *,
        regeneration_claim: object | None,
    ) -> None:
        raise RuntimeError("emit failed")


def _scene_result(
    tmp_path: Path,
    *,
    success: bool = True,
    validation_failure: bool = False,
    error: str = "scene failed",
):
    output = tmp_path / "scene_with_materials.usd"
    output.write_text("#usda 1.0\n")
    flat = tmp_path / "composed_scene_flat.usd"
    flat.write_text("#usda 1.0\n")
    manifest = tmp_path / "manifest_scene.json"
    manifest.write_text("{}")
    render = tmp_path / "render_scene.png"
    render.write_bytes(b"png")
    return SimpleNamespace(
        success=success,
        error=error if not success else None,
        validation_passed=False if validation_failure else True,
        validation_report={"errors": ["bad"]} if validation_failure else {},
        raw_result={"sub_assets": 1, "payload_groups": 0},
        completed_assets=1,
        completed_payloads=0,
        failed_assets=0,
        failed_payloads=0,
        working_dir=str(tmp_path / "scene"),
        manifest_path=str(manifest),
        output_usd_path=str(output),
        rendered_images=[str(render)],
        warnings=[],
    )


@pytest.mark.asyncio
async def test_executor_scene_pipeline_inner_success_validation_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    sid = str(uuid4())
    await manager.create_session(sid)
    await manager.update_session(
        sid,
        {
            "asset": {
                "filename": "scene.usd",
                "file_size_bytes": 5,
                "file_extension": ".usd",
            }
        },
    )
    span = _Span()
    monkeypatch.setattr(executor, "get_current_span", lambda: span)
    monkeypatch.setattr(executor, "TelemetryEventListener", _Telemetry)
    monkeypatch.setattr(executor, "get_event_bus", lambda: _FailingBus())

    async def scene_success(scene_input: object) -> SimpleNamespace:
        return _scene_result(tmp_path, success=True)

    monkeypatch.setattr(executor, "arun_scene_pipeline", scene_success)
    await executor._execute_scene_pipeline_inner(
        sid,
        {
            "materials": {"entries": [{"name": "A", "icon": "a.png"}]},
            "steps": {"predict": {"vlm": {"model": "model"}}},
            "project": [],
        },
        manager,
        user_email="user@nvidia.com",
        scene_options={"assets": ["A"], "max_workers": 1},
    )
    assert (await manager.get_session_metadata(sid))["status"] == "completed"

    scene_bus = EventBus()
    scene_bus.set_session_manager(manager)
    await scene_bus.seed_pending_session(sid)
    monkeypatch.setattr(executor, "get_event_bus", lambda: scene_bus)

    async def validation_failure(scene_input: object) -> SimpleNamespace:
        return _scene_result(
            tmp_path,
            success=False,
            validation_failure=True,
            error="scene-validation-error-sentinel-727",
        )

    monkeypatch.setattr(executor, "arun_scene_pipeline", validation_failure)
    await executor._execute_scene_pipeline_inner(sid, {"steps": {}}, manager)
    metadata = await manager.get_session_metadata(sid)
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "scene_validate"
    assert metadata["error"] == "material_scene_validation_failed"
    assert metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "material_scene_validation_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "scene-validation-error-sentinel-727" not in json.dumps(metadata)
    validation_event = await scene_bus.get_queue(sid).get()
    assert validation_event.message == "material_scene_validation_failed"
    assert (
        "scene-validation-error-sentinel-727" not in validation_event.model_dump_json()
    )

    scene_failure_sentinel = "scene-pipeline-result-secret-727"

    async def scene_failure(scene_input: object) -> SimpleNamespace:
        return _scene_result(
            tmp_path,
            success=False,
            error=scene_failure_sentinel,
        )

    monkeypatch.setattr(executor, "arun_scene_pipeline", scene_failure)
    with pytest.raises(
        RuntimeError,
        match="^material_scene_pipeline_result_failed$",
    ) as exc_info:
        await executor._execute_scene_pipeline_inner(sid, {"steps": {}}, manager)
    assert scene_failure_sentinel not in str(exc_info.value)

    async def scene_cancel(scene_input: object) -> SimpleNamespace:
        assert not scene_input.cancel_checker()
        (manager.get_session_dir(sid) / ".cancel").write_text("cancel")
        assert scene_input.cancel_checker()
        assert scene_input.cancel_checker()
        raise asyncio.CancelledError

    monkeypatch.setattr(executor, "arun_scene_pipeline", scene_cancel)
    with pytest.raises(asyncio.CancelledError):
        await executor._execute_scene_pipeline_inner(sid, {"steps": {}}, manager)


@pytest.mark.asyncio
async def test_executor_scene_cancellation_polling_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    sid = str(uuid4())
    await manager.create_session(sid)
    original_sleep = asyncio.sleep
    calls = {"count": 0}

    async def flaky_is_cancelled(session_id: str) -> bool:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary store failure")
        return True

    async def fast_sleep(delay: float) -> None:
        await original_sleep(0)

    async def scene_success(scene_input: object) -> SimpleNamespace:
        for _ in range(3):
            await original_sleep(0)
        return _scene_result(tmp_path, success=True)

    monkeypatch.setattr(manager, "is_cancelled", flaky_is_cancelled)
    monkeypatch.setattr(executor.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(executor, "arun_scene_pipeline", scene_success)
    await executor._execute_scene_pipeline_inner(
        sid,
        {"steps": {}, "project": []},
        manager,
        scene_options={"max_workers": 1},
    )
    assert calls["count"] >= 2

    file_cancel_sid = str(uuid4())
    await manager.create_session(file_cancel_sid)
    (manager.get_session_dir(file_cancel_sid) / ".cancel").write_text("cancel")
    calls["count"] = 0
    await executor._execute_scene_pipeline_inner(
        file_cancel_sid,
        {"steps": {}, "project": []},
        manager,
        scene_options={"max_workers": 1},
    )


def test_executor_scene_stats_and_prediction_index_helpers(tmp_path: Path) -> None:
    work = tmp_path / "asset"
    renders = work / "dataset" / "usd" / "renders"
    renders.mkdir(parents=True)
    (renders / "a.png").write_bytes(b"png")
    pred = tmp_path / "predictions.jsonl"
    pred.write_text('{"material": "A"}\nnot-json\n\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sub_assets": [
                    {
                        "id": "sub-1",
                        "name": "Sub",
                        "prim_path": "/World/Sub",
                        "status": "failed",
                        "working_dir": str(work),
                        "predictions_path": str(pred),
                    }
                ],
                "payload_groups": [
                    {
                        "id": "payload-1",
                        "group_name": "Payload",
                        "payload_file": "payload.usd",
                        "status": "failed",
                        "predictions_path": str(pred),
                    }
                ],
            }
        )
    )
    result = SimpleNamespace(
        raw_result={"sub_assets": 1, "payload_groups": 1},
        completed_assets=2,
        completed_payloads=1,
        failed_assets=1,
        failed_payloads=1,
        manifest_path=str(manifest),
        rendered_images=["render.png"],
        validation_report={"errors": ["e"], "warnings": ["w"]},
        validation_passed=False,
        warnings=["warn"],
    )

    stats = executor._extract_scene_stats(result)
    assert stats["scene_failed_items"][0]["source_type"] == "sub_asset"
    assert stats["scene_asset_image_count"] == 1
    assert stats["scene_validation_errors"] == 1

    assert executor._extract_failed_scene_items("") == []
    assert executor._extract_failed_scene_items(str(tmp_path / "missing.json")) == []
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text("{")
    assert executor._extract_failed_scene_items(str(bad_manifest)) == []
    list_manifest = tmp_path / "list.json"
    list_manifest.write_text("[]")
    assert executor._extract_failed_scene_items(str(list_manifest)) == []
    assert executor._extract_scene_asset_image_count("") == 0
    assert (
        executor._extract_scene_asset_image_count(str(tmp_path / "missing.json")) == 0
    )
    assert executor._extract_scene_asset_image_count(str(bad_manifest)) == 0
    assert executor._extract_scene_asset_image_count(str(list_manifest)) == 0
    assert executor._extract_scene_item_image_count({}) == 0
    assert (
        executor._extract_scene_item_image_count(
            {"working_dir": str(tmp_path / "empty-work")}
        )
        == 0
    )

    session_dir = tmp_path / "session"
    assert (
        executor._write_scene_validation_report(
            session_dir, SimpleNamespace(validation_report=None)
        )
        is None
    )
    report_path = executor._write_scene_validation_report(session_dir, result)
    assert report_path is not None
    index_path = executor._write_scene_predictions_index(session_dir, result)
    assert index_path is not None
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert len(records) == 2
    assert (
        executor._write_scene_predictions_index(
            session_dir, SimpleNamespace(manifest_path="", validation_report=None)
        )
        is None
    )
    assert (
        executor._write_scene_predictions_index(
            session_dir, SimpleNamespace(manifest_path=str(tmp_path / "missing.json"))
        )
        is None
    )
    assert (
        executor._write_scene_predictions_index(
            session_dir, SimpleNamespace(manifest_path=str(list_manifest))
        )
        is None
    )
    assert (
        executor._write_scene_prediction_records(
            io.StringIO(),
            {"predictions_path": str(tmp_path / "missing.json")},
            source_type="sub_asset",
        )
        == 0
    )

    state_work = tmp_path / "state-work"
    state_work.mkdir()
    (state_work / ".pipeline_state.json").write_text(
        '{"step_outputs": {"build_dataset_usd": {"num_images": 7}}}'
    )
    assert (
        executor._extract_scene_item_image_count({"working_dir": str(state_work)}) == 7
    )


@pytest.mark.asyncio
async def test_artifact_promotion_accepts_completed_step_canonical_files(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    (session_dir / "cache" / "predictions").mkdir(parents=True, exist_ok=True)
    (session_dir / "cache" / "predictions" / "predictions.jsonl").write_text("{}\n")
    (session_dir / "output").mkdir(exist_ok=True)
    (session_dir / "output" / "scene_with_materials.usd").write_text("#usda 1.0\n")
    (session_dir / "output" / "scene_with_materials_flat.usd").write_text(
        "#usda 1.0\n# stale-flat-a\n"
    )
    (session_dir / "output" / "composed_scene_flat.usd").write_text(
        "#usda 1.0\n# stale-flat-b\n"
    )
    (session_dir / "cache" / "preview").mkdir(exist_ok=True)
    (session_dir / "cache" / "preview" / "current.png").write_bytes(b"png")
    (session_dir / "cache" / "dataset" / "usd").mkdir(parents=True, exist_ok=True)
    (session_dir / "cache" / "dataset" / "usd" / "prims.jsonl").write_text("stale\n")
    (session_dir / "cache" / ".pipeline_state.json").write_text("stale-state")
    baseline = executor._capture_promotable_file_signatures(session_dir)
    artifact_map: dict[str, str] = {}

    stale_promoted = await executor._promote_current_run_artifacts(
        manager,
        session_id,
        session_dir,
        ["build_dataset_usd", "predict", "apply"],
        {
            "build_dataset_usd": {"num_images": 1},
            "predict": {"predictions_count": 1},
            "apply": {"materials_applied": {"Steel": ["/Root"]}},
        },
        artifact_map=artifact_map,
        baseline_signatures=baseline,
    )
    assert stale_promoted == set()
    assert artifact_map == {}

    explicit_stale_map: dict[str, str] = {}
    explicit_stale_promoted = await executor._promote_current_run_artifacts(
        manager,
        session_id,
        session_dir,
        ["predict", "render"],
        {
            "predict": {
                "predictions_path": str(
                    session_dir / "cache" / "predictions" / "predictions.jsonl"
                )
            },
            "render": {
                "flattened_usd_path": str(
                    session_dir / "output" / "scene_with_materials_flat.usd"
                )
            },
        },
        artifact_map=explicit_stale_map,
        baseline_signatures=baseline,
    )
    assert explicit_stale_promoted == set()
    assert explicit_stale_map == {}

    (session_dir / "cache" / "predictions" / "predictions.jsonl").write_text(
        '{"current": true}\n'
    )
    (session_dir / "output" / "scene_with_materials.usd").write_text(
        "#usda 1.0\n# current\n"
    )
    (session_dir / "output" / "scene_with_materials_flat.usd").write_text(
        "#usda 1.0\n# current-flat-a\n"
    )
    (session_dir / "cache" / "preview" / "current.png").write_bytes(b"current-png")
    (session_dir / "cache" / "dataset" / "usd" / "prims.jsonl").write_text("current\n")
    (session_dir / "cache" / ".pipeline_state.json").write_text("current-state")
    promoted = await executor._promote_current_run_artifacts(
        manager,
        session_id,
        session_dir,
        ["build_dataset_usd", "predict", "apply", "render"],
        {
            "build_dataset_usd": {"num_images": 1},
            "predict": {"predictions_count": 1},
            "apply": {"materials_applied": {"Steel": ["/Root"]}},
            "render": {
                "flattened_usd_path": str(
                    session_dir / "output" / "scene_with_materials_flat.usd"
                )
            },
        },
        artifact_map=artifact_map,
        baseline_signatures=baseline,
    )
    assert {
        "raw_predictions",
        "applied_output_usd",
        "rendered_output_usd",
        "previews",
    } <= promoted
    assert "output/scene_with_materials_flat.usd" in artifact_map
    assert "output/composed_scene_flat.usd" not in artifact_map
    assert "cache/dataset/usd/prims.jsonl" in artifact_map
    assert "cache/.pipeline_state.json" in artifact_map


@pytest.mark.asyncio
async def test_claim_monitor_does_not_cancel_after_normal_finalize(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
    )
    owner = asyncio.create_task(asyncio.sleep(60))
    try:
        assert await manager.finalize_regeneration_claim(
            session_id,
            claim,
            updates={"status": "completed", "results": {}, "coverage": None},
        )
        await asyncio.wait_for(
            executor._monitor_regeneration_claim(
                manager,
                session_id,
                claim,
                owner,
            ),
            timeout=1,
        )
        assert not owner.cancelled()
    finally:
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["renew", "read"])
async def test_claim_monitor_fails_closed_on_store_errors(
    tmp_path: Path,
    failure_stage: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    await manager.create_session(session_id)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
        updates={"status": "running"},
    )

    class _FailingClaimManager:
        async def is_regeneration_cancel_requested(self, *_args: object) -> bool:
            return False

        async def renew_regeneration_claim(self, *_args: object, **_kwargs: object):
            if failure_stage == "renew":
                raise RuntimeError("claim-store-secret-727")
            return False

        async def get_session_metadata(self, *_args: object):
            raise RuntimeError("claim-store-secret-727")

    owner = asyncio.create_task(asyncio.sleep(60))
    with caplog.at_level(logging.ERROR):
        await executor._monitor_regeneration_claim(
            _FailingClaimManager(),  # type: ignore[arg-type]
            session_id,
            claim,
            owner,
        )
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert owner.cancelled()
    assert "claim-store-secret-727" not in caplog.text
    assert "code=regeneration_claim_monitor_failed" in caplog.text


@pytest.mark.asyncio
async def test_storage_failure_diagnostics_do_not_log_backend_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "signed-url-secret-storage-failure-727"
    artifact = tmp_path / "artifact.usda"
    artifact.write_text("#usda 1.0\n", encoding="utf-8")

    class FailingMirrorManager:
        async def put_file_to_store(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(sentinel)

    with caplog.at_level(logging.ERROR):
        await executor._mirror_scene_artifact(
            FailingMirrorManager(),  # type: ignore[arg-type]
            str(uuid4()),
            str(artifact),
            "output/artifact.usda",
        )

    assert sentinel not in caplog.text
    assert "code=scene_artifact_sync_failed" in caplog.text
    assert "phase=sync_upload" in caplog.text

    class FailingStatusManager:
        async def session_exists(self, _session_id: str) -> bool:
            raise RuntimeError(sentinel)

    caplog.clear()
    bus = EventBus()
    bus.set_session_manager(FailingStatusManager())  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR):
        await bus._persist_status(str(uuid4()), "failed")

    assert sentinel not in caplog.text
    assert "code=event_bus_status_persistence_failed" in caplog.text
    assert "phase=local_publication" in caplog.text

    class FailingServiceConfig:
        def __init__(self) -> None:
            raise RuntimeError(sentinel)

    caplog.clear()
    monkeypatch.setattr(config_module, "ServiceConfig", FailingServiceConfig)
    fallback_bus = EventBus()
    with caplog.at_level(logging.ERROR):
        assert fallback_bus._get_session_manager() is None
    assert sentinel not in caplog.text
    assert "code=event_bus_session_manager_unavailable" in caplog.text

    class FailingEventLogManager:
        async def session_exists(self, _session_id: str) -> bool:
            return True

        def get_session_dir(self, _session_id: str) -> Path:
            return tmp_path / "missing-parent" / "session"

    caplog.clear()
    event_log_bus = EventBus()
    event_log_bus.set_session_manager(  # type: ignore[arg-type]
        FailingEventLogManager()
    )
    with caplog.at_level(logging.ERROR):
        await event_log_bus._save_event_to_log(
            ProgressEvent(
                session_id=str(uuid4()),
                step="pipeline",
                state=StepState.RUNNING,
                percent=1,
            )
        )
    assert "missing-parent" not in caplog.text
    assert "code=event_log_persistence_failed" in caplog.text


@pytest.mark.asyncio
async def test_executor_mirroring_and_stats_file_fallbacks(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    sid = str(uuid4())
    await manager.create_session(sid)
    output = tmp_path / "scene_with_materials.usd"
    output.write_text("#usda 1.0\n")
    flat = tmp_path / "composed_scene_flat.usd"
    flat.write_text("#usda 1.0\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    render = tmp_path / "render.png"
    render.write_bytes(b"png")
    result = SimpleNamespace(
        output_usd_path=str(output),
        manifest_path=str(manifest),
        rendered_images=[str(render)],
    )
    await executor._mirror_scene_outputs(
        manager,
        sid,
        result,
        validation_report_path=manifest,
        scene_predictions_path=manifest,
    )
    assert await manager.store.exists(sid, "output/scene_with_materials.usd")
    await executor._mirror_scene_artifact(manager, sid, "", "missing")
    await executor._mirror_scene_artifact(
        manager, sid, str(tmp_path / "missing"), "missing"
    )

    session_dir = tmp_path / "stats"
    (session_dir / "cache" / "dataset").mkdir(parents=True)
    (session_dir / "cache" / "dataset" / "dataset.jsonl").write_text("{}\n\n{}\n")
    (session_dir / "cache" / "dataset" / "a.png").write_bytes(b"png")
    (session_dir / "cache" / "predictions").mkdir(parents=True)
    (session_dir / "cache" / "predictions" / "predictions.jsonl").write_text("{}\n")
    (session_dir / "cache" / "clusters").mkdir(parents=True)
    (session_dir / "cache" / "clusters" / "cluster_map.jsonl").write_text(
        '{"cluster_id": 1, "is_representative": true, "cluster_size": 2}\n'
        '{"cluster_id": 1, "is_representative": false, "cluster_size": 2}\n'
    )
    stats = executor._count_stats_from_files(
        session_dir,
        {
            "prims_processed": 0,
            "images_generated": 0,
            "predictions_made": 0,
            "cluster_prims_ran": True,
            "cluster_count": 0,
        },
    )
    assert stats["prims_processed"] == 2
    assert stats["predictions_made"] == 1
    assert stats["cluster_count"] == 1

    result = SimpleNamespace(
        step_results={
            "benchmark": {"predictions_count": 4},
            "apply": {"materials_applied": {"A": ["/World/A"]}},
        },
        raw_result={
            "pipeline_results": {
                "cluster_prims": {
                    "cluster_prims_ran": True,
                    "cluster_total_prims": 10,
                    "cluster_count": 5,
                },
                "optimize_usd": {
                    "original_prim_count": 0,
                    "optimization_metadata": {"original_prim_count": 12},
                },
                "build_dataset_prepare_dataset": {"num_entries": 6},
            }
        },
    )
    extracted = executor._extract_stats_from_result(result, None)
    assert extracted["original_prim_count"] == 12
    assert extracted["predictions_made"] == 4
    assert extracted["materials_applied"] == 1
    assert extracted["prims_processed"] == 6

    top_level_opt = executor._extract_stats_from_result(
        SimpleNamespace(
            step_results={},
            raw_result={
                "optimization_metadata": {"original_prim_count": 8},
                "dataset_info": {"num_entries": 2},
            },
        ),
        None,
    )
    assert top_level_opt["original_prim_count"] == 8
    legacy_usd = executor._extract_stats_from_result(
        SimpleNamespace(
            step_results={},
            raw_result={"build_dataset_usd_result": {"num_prims": 3, "num_images": 4}},
        ),
        None,
    )
    assert legacy_usd["images_generated"] == 4
    legacy_prepare = executor._extract_stats_from_result(
        SimpleNamespace(
            step_results={},
            raw_result={"build_dataset_prepare_dataset_result": {"num_entries": 5}},
        ),
        None,
    )
    assert legacy_prepare["prims_processed"] == 5
    stats = {"prims_processed": 0}
    assert executor._count_stats_from_files(None, stats) is stats

    attrs = executor._cluster_telemetry_attributes(
        {}, {"cluster_prims_ran": True}, step_name="cluster_prims"
    )
    assert "maa.clustering.enabled" not in attrs


def test_executor_step_span_and_merge_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged = executor._merge_completed_steps_from_result(
        "not-list",
        SimpleNamespace(
            completed_steps=["cluster_prims"],
            step_results={"cluster_prims": {"cluster_prims_ran": True}},
        ),
        {"cluster_prims": 2.5},
    )
    assert merged[0]["name"] == "cluster_prims"

    raw_outputs = {
        "restore_stats": {
            "restored_prim_sources": {"/A": "/Optimized/A"},
            "uncovered_originals": ["/B"],
        },
        "assignment_stats": {
            "bound_prim_ids": ["/A"],
            "unbound_prim_ids": ["/B", "/C"],
        },
    }
    bounded = executor._merge_completed_steps_from_result(
        [],
        SimpleNamespace(
            completed_steps=["restore_usd", "apply"],
            step_results={
                "restore_usd": raw_outputs,
                "apply": raw_outputs,
            },
        ),
        {},
    )
    for step in bounded:
        outputs = step["stats"]["outputs"]
        assert outputs["restore_stats"] == {
            "restored_prim_source_count": 1,
            "uncovered_original_count": 1,
        }
        assert outputs["assignment_stats"] == {
            "bound_prim_count": 1,
            "unbound_prim_count": 2,
        }
    assert "restored_prim_sources" in raw_outputs["restore_stats"]
    assert "bound_prim_ids" in raw_outputs["assignment_stats"]
    executor._emit_step_spans("sid")

    class _StepSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    spans: list[_StepSpan] = []

    class _Tracer:
        def start_as_current_span(self, name: str) -> object:
            span = _StepSpan()
            spans.append(span)

            class _Context:
                def __enter__(self) -> _StepSpan:
                    return span

                def __exit__(self, *args: object) -> None:
                    return None

            return _Context()

    import world_understanding.telemetry as telemetry

    monkeypatch.setattr(telemetry, "get_tracer", lambda name: _Tracer())
    executor._emit_step_spans(
        "sid",
        step_timings=[
            {
                "name": "cluster_prims",
                "status": "failed",
                "started_at_ns": 1,
                "completed_at_ns": 1_000_000_001,
                "error": "bad",
            }
        ],
        step_results={"cluster_prims": {"cluster_prims_ran": True, "cluster_count": 2}},
    )
    assert spans[-1].attributes[executor.MAAttributes.PIPELINE_STEP_ERROR] == "bad"
    assert any(key.endswith("cluster_count") for key in spans[-1].attributes)
