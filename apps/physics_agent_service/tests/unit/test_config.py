# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Physics Agent Service configuration semantics."""

from pathlib import Path
from types import SimpleNamespace

from ...service import config as config_module
from ...service.config import ServiceConfig
from ...service.storage.local_store import LocalSessionStore


def test_has_required_api_keys_accepts_public_nim_credentials_with_sidecar_renderer(
    monkeypatch, tmp_path: Path
):
    """Public NIM + local sidecar rendering should not require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_accepts_physics_local_nim_sidecar(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("PA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")

    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_uses_effective_vlm_backend_for_nim_override(
    monkeypatch, tmp_path: Path
):
    """PA_VLM_NIM_BASE_URL makes readiness validate the effective NIM backend."""
    monkeypatch.setenv("PA_VLM_BACKEND", "openai")
    monkeypatch.setenv("PA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.vlm_backend == "openai"
    assert config.has_required_api_keys is True


def test_has_required_api_keys_accepts_openai_custom_base_url_with_key_env(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "endpoint-openai-key")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("PA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="openai",
        vlm_base_url="https://api.openai-compatible.example/v1",
        vlm_api_key_env="CUSTOM_OPENAI_API_KEY",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_ignores_endpoint_key_for_vlm_nim_override(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "endpoint-openai-key")
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("PA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="openai",
        vlm_base_url="https://api.openai-compatible.example/v1",
        vlm_api_key_env="CUSTOM_OPENAI_API_KEY",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_prefers_configured_api_key_env(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("MISSING_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("PA_VLM_NIM_BASE_URL", raising=False)

    config = ServiceConfig(
        vlm_backend="openai",
        vlm_base_url="https://api.openai-compatible.example/v1",
        vlm_api_key="inline-openai-key",
        vlm_api_key_env="MISSING_OPENAI_API_KEY",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_accepts_explicit_gemini_key(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("PA_VLM_NIM_BASE_URL", raising=False)

    config = ServiceConfig(
        vlm_backend="gemini",
        vlm_api_key="inline-gemini-key",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_requires_ngc_for_authenticated_remote_renderer(
    monkeypatch, tmp_path: Path
):
    """Authenticated remote renderer endpoints still require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("RENDER_ENDPOINT", "https://ai.api.nvidia.com/v1/render")
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_requires_ngc_for_nvcf_render_function(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("PA_RENDER_BACKEND", "remote")
    monkeypatch.setenv("NVCF_RENDER_FUNCTION_ID", "render-function-id")
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="mock",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_config_helper_branches(monkeypatch, tmp_path: Path):
    assert config_module._is_local_render_endpoint(None) is False
    assert config_module._is_local_render_endpoint("http://localhost:8080") is True

    assert config_module._backend_has_credentials(
        "",
        nvidia_api_key=None,
    )
    assert config_module._backend_has_credentials(
        "mock",
        nvidia_api_key=None,
    )
    assert config_module._backend_has_credentials(
        "openai",
        nvidia_api_key=None,
        api_key="openai-test",
    )
    assert config_module._backend_has_credentials(
        "anthropic",
        nvidia_api_key=None,
        api_key="anthropic-key",
    )
    assert not config_module._backend_has_credentials(
        "unknown-backend",
        nvidia_api_key=None,
    )
    assert config_module._backend_has_credentials(
        "unknown-backend",
        nvidia_api_key=None,
        api_key="plugin-key",
    )
    from world_understanding.functions.models.backends import registry

    plugin_backend = "test-physics-service-vlm-plugin"
    monkeypatch.setitem(registry._vlm_backends, plugin_backend, lambda **_kwargs: None)
    monkeypatch.setitem(registry._vlm_backend_requires_api_key, plugin_backend, True)
    monkeypatch.setitem(registry._vlm_backend_capabilities, plugin_backend, frozenset())
    assert config_module._backend_has_credentials(
        plugin_backend,
        nvidia_api_key=None,
        api_key="plugin-key",
    )

    real_exists = Path.exists

    def readme_missing(self: Path) -> bool:
        if self.name == "README.md":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", readme_missing)
    assert ServiceConfig._load_description() == "Physics Agent REST API Service"

    config = ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
        storage_kind="local",
    )
    assert isinstance(config.build_session_store(), LocalSessionStore)

    from ...service import storage as storage_module

    monkeypatch.setattr(
        storage_module.S3SessionStore,
        "from_config",
        classmethod(
            lambda _cls, storage_cfg: SimpleNamespace(kind="s3", cfg=storage_cfg)
        ),
    )
    config = ServiceConfig(
        session_storage_path=str(tmp_path / "sessions"),
        storage_kind="s3",
        storage_s3_bucket="bucket",
        storage_s3_prefix="prefix",
    )
    store = config.build_session_store()
    assert store.kind == "s3"
    assert store.cfg.s3_bucket == "bucket"
