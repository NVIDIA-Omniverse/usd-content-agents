# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError
from texture_agent.config.unified_config import config_to_context

from ...service import config as config_module
from ...service.config import ServiceConfig, _llm_backend_has_credentials
from ...service.routers import pipeline_router as pipeline_router_module
from ...service.routers.pipeline_router import (
    _decode_json_form_field,
    _normalize_uri_list,
    _require_projection_endpoint,
    _save_reference_image_upload,
    build_default_pipeline_config,
)


def test_service_config_reads_prefixed_or_unprefixed_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TA_NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "fallback-key")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.nvidia_api_key == "fallback-key"
    assert config.session_storage_path == str(sessions)
    assert config.description == "desc"


def test_service_config_falls_back_to_local_sessions_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_NVIDIA_API_KEY", "prefixed-key")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    config = ServiceConfig(session_storage_path=str(tmp_path / "missing"))

    assert config.nvidia_api_key == "prefixed-key"
    assert config.session_storage_path.endswith("apps/texture_agent_service/sessions")
    assert config.description == "desc"


def test_service_config_reads_cancel_drain_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_CANCEL_DRAIN_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.cancel_drain_timeout_seconds == 2.5


def test_service_config_reads_projection_backend_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_TEXTURE_ENDPOINT", "http://texture-gen-step1x:8000")
    monkeypatch.setenv("TA_BACKEND_ENGINE", "step1x")
    monkeypatch.setenv("TA_SIMPLE_TEXTURE_ENDPOINT", "http://texture-gen-simple:8000")
    monkeypatch.setenv("TA_TEXTURE_JOB_TIMEOUT_SEC", "7200")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.texture_endpoint == "http://texture-gen-step1x:8000"
    assert config.backend_engine == "step1x"
    assert config.simple_texture_endpoint == "http://texture-gen-simple:8000"
    assert config.texture_job_timeout_sec == 7200


def test_service_config_reads_auto_prompt_material_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_AUTO_PROMPT_MAX_GENERATED_MATERIALS", "12")
    monkeypatch.setenv("TA_MAX_TEXTURE_UNITS", "13")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.auto_prompt_max_generated_materials == 12
    assert config.max_texture_units == 13


def test_service_config_enforces_texture_plan_v1_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_TEXTURE_PLAN_DEFAULT_CAP", "32")
    monkeypatch.setenv("TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP", "16")
    monkeypatch.setenv("TA_TEXTURE_PLAN_HARD_CAP", "64")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.texture_plan_default_cap == 32
    assert config.texture_plan_uv_aware_default_cap == 16
    assert config.texture_plan_hard_cap == 64

    monkeypatch.setenv("TA_TEXTURE_PLAN_HARD_CAP", "65")
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        ServiceConfig(session_storage_path=str(sessions))


def test_service_config_reads_render_sidecar_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TA_RENDER_ENABLED", "true")
    monkeypatch.setenv("TA_RENDER_IMAGE_WIDTH", "640")
    monkeypatch.setenv("TA_RENDER_IMAGE_HEIGHT", "480")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.render_enabled is True
    assert config.render_image_width == 640
    assert config.render_image_height == 480


@pytest.mark.parametrize("value", ["0", "-1"])
def test_service_config_rejects_non_positive_render_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    monkeypatch.setenv("TA_RENDER_TIMEOUT_SEC", value)
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()

    with pytest.raises(ValidationError, match="greater than 0"):
        ServiceConfig(session_storage_path=str(sessions))


def test_service_config_reads_uv_sidecar_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_UV_POLICY", "force_projection")
    monkeypatch.setenv("TA_UV_SCOPE", "target_prims")
    monkeypatch.setenv("TA_UV_BACKEND", "python")
    monkeypatch.setenv("TA_UV_PROJECTION", "box")
    monkeypatch.setenv("TA_UV_OVERWRITE_EXISTING", "true")
    monkeypatch.setenv("TA_UV_REBAKE_SOURCE_ALBEDO", "true")
    monkeypatch.setenv("TA_UV_REBAKE_SIZE", "2048")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    config = ServiceConfig(session_storage_path=str(sessions))

    assert config.uv_policy == "force_projection"
    assert config.uv_scope == "target_prims"
    assert config.uv_backend == "python"
    assert config.uv_projection == "box"
    assert config.uv_overwrite_existing is True
    assert config.uv_rebake_source_albedo is True
    assert config.uv_rebake_size == 2048


def test_service_config_passes_s3_connection_pool_size(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(
        session_storage_path=str(sessions),
        storage_kind="s3",
        storage_s3_bucket="bucket",
        storage_s3_max_pool_connections=123,
    )
    store = config.build_session_store()

    assert store._max_pool_connections == 123


def test_service_config_uses_wu_s3_fallbacks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TA_STORAGE_S3_BUCKET", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_REGION", raising=False)
    monkeypatch.delenv("TA_STORAGE_S3_PROFILE", raising=False)
    monkeypatch.setenv("WU_S3_BUCKET", "wu-bucket")
    monkeypatch.setenv("WU_S3_REGION", "us-east-2")
    monkeypatch.setenv("WU_S3_PROFILE", "wu-profile")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(
        session_storage_path=str(sessions),
        storage_kind="s3",
    )
    store = config.build_session_store()

    assert config.storage_s3_bucket == "wu-bucket"
    assert config.storage_s3_region == "us-east-2"
    assert config.storage_s3_profile == "wu-profile"
    assert store.bucket == "wu-bucket"
    assert store._region == "us-east-2"
    assert store._profile == "wu-profile"


def test_service_config_builds_local_store(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    config = ServiceConfig(
        session_storage_path=str(sessions),
        storage_kind="local",
    )

    assert config.build_session_store().kind == "local"


def test_load_description_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingPath:
        def __init__(self, *_args) -> None:
            pass

        @property
        def parent(self) -> MissingPath:
            return self

        def __truediv__(self, _name: str) -> MissingPath:
            return self

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(config_module, "Path", MissingPath)

    assert ServiceConfig._load_description() == "Texture Agent REST API Service"


def test_llm_backend_credential_detection_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _llm_backend_has_credentials("", nvidia_api_key=None) is True
    assert _llm_backend_has_credentials("echo", nvidia_api_key=None) is True
    assert _llm_backend_has_credentials("unknown", nvidia_api_key=None) is False
    assert (
        _llm_backend_has_credentials(
            "unknown", nvidia_api_key=None, api_key="plugin-key"
        )
        is True
    )

    monkeypatch.setattr(
        config_module,
        "get_nim_api_key_for_base_url",
        lambda _base_url, explicit_key: explicit_key,
    )
    monkeypatch.setattr(
        config_module,
        "is_nvidia_provider_base_url",
        lambda _base_url: True,
    )
    assert _llm_backend_has_credentials(
        "nim",
        nvidia_api_key="nvidia-key",
        base_url="https://integrate.api.nvidia.com/v1",
    )

    monkeypatch.setattr(
        config_module,
        "get_env_api_key_for_backend",
        lambda _backend, explicit_key: explicit_key,
    )
    assert _llm_backend_has_credentials(
        "anthropic",
        nvidia_api_key=None,
        api_key="anthropic-key",
    )


def test_load_description_reads_repo_readme() -> None:
    description = ServiceConfig._load_description()

    assert "Texture Agent" in description


def test_default_pipeline_config_fails_on_any_texture_generation_error(
    tmp_path: Path,
) -> None:
    """Service-created runs must not silently complete with dropped materials."""
    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Steel": {"prompt": "brushed steel"}},
    )

    context = config_to_context(config)

    assert context["texture_config"]["failure_threshold"] == 0.0


def test_default_pipeline_config_preserves_llm_api_key_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "llm_backend", "openai")
    monkeypatch.setattr(pipeline_router_module.config, "llm_model", "my-custom-llm")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "llm_base_url",
        "https://api.openai-compatible.example/v1",
    )
    monkeypatch.setattr(pipeline_router_module.config, "llm_api_key", None)
    monkeypatch.setattr(
        pipeline_router_module.config,
        "llm_api_key_env",
        "OPENAI_API_KEY",
    )

    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Steel": {"prompt": "brushed steel"}},
    )

    llm_config = config["auto_prompt"]["llm"]
    assert llm_config["backend"] == "openai"
    assert llm_config["base_url"] == "https://api.openai-compatible.example/v1"
    assert llm_config["api_key_env"] == "OPENAI_API_KEY"


def test_default_pipeline_config_sets_auto_prompt_material_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        pipeline_router_module.config,
        "auto_prompt_max_generated_materials",
        7,
    )

    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
    )

    assert config["auto_prompt"]["max_generated_materials"] == 7


def test_default_pipeline_config_sets_texture_unit_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "max_texture_units", 9)

    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
    )

    assert config["texture"]["max_texture_units"] == 9


def test_service_config_reports_llm_readiness_for_openai_key_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "endpoint-openai-key")
    monkeypatch.setattr(
        ServiceConfig, "_load_description", staticmethod(lambda: "desc")
    )

    ready_config = ServiceConfig(
        llm_backend="openai",
        llm_base_url="https://api.openai-compatible.example/v1",
        llm_api_key_env="CUSTOM_OPENAI_API_KEY",
        session_storage_path=str(tmp_path / "sessions-ready"),
    )
    assert ready_config.llm_ready is True
    assert ready_config.has_required_api_keys is True

    monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
    missing_config = ServiceConfig(
        llm_backend="openai",
        llm_base_url="https://api.openai-compatible.example/v1",
        llm_api_key_env="CUSTOM_OPENAI_API_KEY",
        session_storage_path=str(tmp_path / "sessions-missing"),
    )
    assert missing_config.llm_ready is False
    assert missing_config.has_required_api_keys is False


def test_default_pipeline_config_preserves_image_gen_api_key_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "image_gen_backend", "openai")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "image_gen_model",
        "black-forest-labs/flux.2-klein-4b",
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "image_gen_base_url",
        "https://api.openai-compatible.example/v1",
    )
    monkeypatch.setattr(pipeline_router_module.config, "image_gen_api_key", None)
    monkeypatch.setattr(
        pipeline_router_module.config,
        "image_gen_api_key_env",
        "IMAGE_GEN_API_KEY",
    )

    config = build_default_pipeline_config(
        session_id="session-1",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Steel": {"prompt": "brushed steel"}},
    )

    image_gen_config = config["texture"]["image_gen"]
    assert image_gen_config["backend"] == "openai"
    assert image_gen_config["base_url"] == "https://api.openai-compatible.example/v1"
    assert image_gen_config["api_key_env"] == "IMAGE_GEN_API_KEY"


def test_default_pipeline_config_preserves_projection_backend_overrides(
    tmp_path: Path,
) -> None:
    config = build_default_pipeline_config(
        session_id="session-projection",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Aluminum_Matte": {"prompt": "matte aluminum"}},
        texture_backend="service",
        texture_endpoint="http://projection-backend",
        backend_engine="fake_projection",
        backend_custom_parameters={"variant": "success_full_pbr"},
        detail_policy="surface_only",
        reference_image_uris=["file:///ref.png"],
        turntable_video_uri="file:///turntable.mp4",
        multiview_image_uris=["file:///view0.png"],
        seed=11631,
        strength=0.8,
        strict_scope=True,
    )

    texture_config = config_to_context(config)["texture_config"]

    assert texture_config["backend"] == "service"
    assert texture_config["endpoint"] == "http://projection-backend"
    assert texture_config["engine"] == "fake_projection"
    assert texture_config["custom_parameters"] == {"variant": "success_full_pbr"}
    assert texture_config["detail_policy"] == "surface_only"
    assert texture_config["reference_image_uris"] == ["file:///ref.png"]
    assert texture_config["turntable_video_uri"] == "file:///turntable.mp4"
    assert texture_config["multiview_image_uris"] == ["file:///view0.png"]
    assert texture_config["seed"] == 11631
    assert texture_config["strength"] == 0.8
    assert texture_config["strict_scope"] is True
    assert texture_config["job_timeout_sec"] == 3600


def test_default_pipeline_config_uses_configured_projection_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "texture_backend", "SERVICE")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "texture_endpoint",
        "http://texture-gen-step1x:8000",
    )
    monkeypatch.setattr(pipeline_router_module.config, "backend_engine", "step1x")

    config = build_default_pipeline_config(
        session_id="session-step1x",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Paint": {"prompt": "chipped paint"}},
    )

    texture_config = config_to_context(config)["texture_config"]

    assert texture_config["backend"] == "service"
    assert texture_config["endpoint"] == "http://texture-gen-step1x:8000"
    assert texture_config["engine"] == "step1x"
    assert texture_config["job_timeout_sec"] == 3600


def test_default_pipeline_config_routes_configured_simple_alias_to_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        pipeline_router_module.config, "texture_backend", "SIMPLE-IMAGE-GEN"
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "texture_endpoint",
        "http://texture-gen-step1x:8000",
    )
    monkeypatch.setattr(pipeline_router_module.config, "backend_engine", "step1x")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_texture_endpoint",
        "http://texture-gen-simple:8000",
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_backend_engine",
        "simple_image_gen",
    )

    config = build_default_pipeline_config(
        session_id="session-simple-configured",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Paint": {"prompt": "worn paint"}},
    )

    texture_config = config_to_context(config)["texture_config"]

    assert texture_config["backend"] == "service"
    assert texture_config["endpoint"] == "http://texture-gen-simple:8000"
    assert texture_config["engine"] == "simple_image_gen"


def test_default_pipeline_config_routes_simple_backend_to_simple_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "texture_backend", "service")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "texture_endpoint",
        "http://texture-gen-step1x:8000",
    )
    monkeypatch.setattr(pipeline_router_module.config, "backend_engine", "step1x")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_texture_endpoint",
        "http://texture-gen-simple:8000",
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_backend_engine",
        "simple_image_gen",
    )
    monkeypatch.setattr(pipeline_router_module.config, "texture_workers", 1)
    monkeypatch.setattr(pipeline_router_module.config, "simple_texture_workers", 4)
    monkeypatch.setattr(
        pipeline_router_module.config,
        "texture_job_timeout_sec",
        7200,
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_texture_job_timeout_sec",
        3600,
    )
    monkeypatch.setattr(pipeline_router_module.config, "uv_scope", "target_prims")
    monkeypatch.setattr(pipeline_router_module.config, "simple_uv_scope", "stage")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "uv_rebake_source_albedo",
        True,
    )
    monkeypatch.setattr(
        pipeline_router_module.config,
        "simple_uv_rebake_source_albedo",
        False,
    )
    monkeypatch.setattr(pipeline_router_module.config, "uv_rebake_size", 2048)
    monkeypatch.setattr(pipeline_router_module.config, "simple_uv_rebake_size", None)

    config = build_default_pipeline_config(
        session_id="session-simple",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Paint": {"prompt": "worn paint"}},
        texture_backend="simple_image_gen",
    )

    texture_config = config_to_context(config)["texture_config"]

    assert texture_config["backend"] == "service"
    assert texture_config["endpoint"] == "http://texture-gen-simple:8000"
    assert texture_config["engine"] == "simple_image_gen"
    assert texture_config["workers"] == 4
    assert texture_config["job_timeout_sec"] == 3600
    assert texture_config["uv_scope"] == "stage"
    assert texture_config["uv_rebake_source_albedo"] is False
    assert "uv_rebake_size" not in texture_config
    assert config["steps"]["generate_textures"]["max_workers"] == 4


def test_default_pipeline_config_uses_configured_render_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "render_enabled", True)
    monkeypatch.setattr(pipeline_router_module.config, "render_image_width", 640)
    monkeypatch.setattr(pipeline_router_module.config, "render_image_height", 480)

    config = build_default_pipeline_config(
        session_id="session-render",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Paint": {"prompt": "chipped paint"}},
    )

    render_config = config["steps"]["render"]

    assert render_config["enabled"] is True
    assert render_config["backend"] == "remote"
    assert render_config["image_width"] == 640
    assert render_config["image_height"] == 480
    assert "timeout_sec" not in render_config


def test_default_pipeline_config_accepts_render_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "render_timeout_sec", 300)

    default_config = build_default_pipeline_config(
        session_id="session-render-timeout-default",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work-default"),
    )
    assert default_config["steps"]["render"]["timeout_sec"] == 300

    config = build_default_pipeline_config(
        session_id="session-render-timeout",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        render_timeout_sec=900,
    )

    assert config["steps"]["render"]["timeout_sec"] == 900


def test_default_pipeline_config_uses_configured_uv_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "uv_policy", "force_projection")
    monkeypatch.setattr(pipeline_router_module.config, "uv_scope", "target_prims")
    monkeypatch.setattr(pipeline_router_module.config, "uv_backend", "python")
    monkeypatch.setattr(pipeline_router_module.config, "uv_projection", "box")
    monkeypatch.setattr(pipeline_router_module.config, "uv_overwrite_existing", True)
    monkeypatch.setattr(pipeline_router_module.config, "uv_rebake_source_albedo", True)
    monkeypatch.setattr(pipeline_router_module.config, "uv_rebake_size", 2048)
    monkeypatch.setattr(
        pipeline_router_module.config, "uv_normalize_out_of_range", False
    )

    config = build_default_pipeline_config(
        session_id="session-uv",
        usd_path=str(tmp_path / "asset.usd"),
        working_dir=str(tmp_path / "work"),
        material_textures={"Paint": {"prompt": "chipped paint"}},
    )

    texture_config = config["texture"]

    assert texture_config["uv_policy"] == "force_projection"
    assert texture_config["uv_scope"] == "target_prims"
    assert texture_config["uv_backend"] == "python"
    assert texture_config["uv_projection"] == "box"
    assert texture_config["uv_mode"] == "box"
    assert texture_config["uv_overwrite_existing"] is True
    assert texture_config["uv_rebake_source_albedo"] is True
    assert texture_config["uv_rebake_size"] == 2048
    assert texture_config["uv_normalize_out_of_range"] is False


def test_projection_json_form_fields_validate_shape() -> None:
    custom = _decode_json_form_field(
        '{"variant":"success_full_pbr"}',
        field_name="backend_custom_parameters_json",
        expected_type=dict,
    )
    refs = _normalize_uri_list(
        _decode_json_form_field(
            '[" file:///ref.png ", ""]',
            field_name="reference_image_uris_json",
            expected_type=list,
        ),
        field_name="reference_image_uris_json",
    )

    assert custom == {"variant": "success_full_pbr"}
    assert refs == ["file:///ref.png"]

    with pytest.raises(HTTPException) as exc_info:
        _decode_json_form_field(
            '["not-a-dict"]',
            field_name="backend_custom_parameters_json",
            expected_type=dict,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["loc"] == [
        "form",
        "backend_custom_parameters_json",
    ]


def test_service_projection_backend_requires_endpoint() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _require_projection_endpoint(texture_backend="service", texture_endpoint=None)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["loc"] == ["form", "texture_endpoint"]


def test_service_projection_backend_accepts_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "texture_backend", "service")
    monkeypatch.setattr(
        pipeline_router_module.config,
        "texture_endpoint",
        "http://texture-gen-step1x:8000",
    )

    _require_projection_endpoint(texture_backend=None, texture_endpoint=None)


def test_service_projection_backend_normalizes_blank_form_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router_module.config, "texture_backend", "service")
    monkeypatch.setattr(pipeline_router_module.config, "texture_endpoint", "")

    with pytest.raises(HTTPException) as exc_info:
        _require_projection_endpoint(texture_backend="   ", texture_endpoint=None)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail[0]["loc"] == ["form", "texture_endpoint"]


@pytest.mark.asyncio
async def test_reference_image_upload_is_saved_as_file_uri(tmp_path: Path) -> None:
    upload = UploadFile(filename="reference.png", file=BytesIO(b"image bytes"))

    uri = await _save_reference_image_upload(upload, tmp_path / "session")

    assert uri is not None
    assert uri.startswith("file://")
    saved_path = Path(uri.removeprefix("file://"))
    assert saved_path.read_bytes() == b"image bytes"


def test_openapi_exposes_projection_backend_request_fields() -> None:
    openapi_path = Path(__file__).resolve().parents[2] / "openapi.yaml"
    payload = yaml.safe_load(openapi_path.read_text(encoding="utf-8"))
    schemas = payload["components"]["schemas"]
    body = schemas["Body_create_pipeline_pipeline_post"]["properties"]
    material_override = schemas["MaterialTextureOverride"]["properties"]

    assert "reference_image_file" in body
    assert "backend_custom_parameters_json" in body
    assert "reference_image_uris_json" in body
    assert "multiview_image_uris_json" in body
    assert "seed" in body
    assert "strength" in body
    assert "strict_scope" in body
    assert "uv_policy" in body
    assert "uv_scope" in body
    assert "uv_rebake_source_albedo" in body
    assert "uv_rebake_size" in body
    assert "material_path" in material_override
    assert "prim_paths" in material_override
    assert "reference_image_uris" in material_override
    assert "turntable_video_uri" in material_override
    assert "multiview_image_uris" in material_override
    for path in ("/pipeline/upload-usd", "/pipeline"):
        response = payload["paths"][path]["post"]["responses"]["403"]
        assert "bucket allowlist" in response["description"]
