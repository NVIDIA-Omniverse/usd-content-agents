# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for public model backend factory modules."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.functions.models import backends, nim_timeout
from world_understanding.functions.models import image_generation_models as image_models
from world_understanding.functions.models import text_embedding_models as text_models
from world_understanding.functions.models import vision_language_models as vlm_models
from world_understanding.functions.models.backends import registry
from world_understanding.functions.models.backends.public import anthropic, gemini, nim
from world_understanding.functions.models.backends.public import (
    openai as openai_backend,
)
from world_understanding.utils import credentials


class CapturingModel:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_backend_package_description_is_provider_neutral() -> None:
    description = backends.__doc__ or ""

    assert "entry-point contract" in description
    assert "world_" + "understanding_internal" not in description
    assert registry.list_loaded_backend_plugins() == tuple(
        sorted(registry._loaded_backend_plugins)
    )


def test_anthropic_chat_and_vlm_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_anthropic",
        SimpleNamespace(ChatAnthropic=CapturingModel),
    )
    monkeypatch.setattr(vlm_models, "AnthropicVLM", CapturingModel)

    with pytest.raises(ValueError, match="Anthropic backend"):
        anthropic.create_anthropic_chat()

    chat = anthropic.create_anthropic_chat(
        api_key="anthropic-key",
        temperature=0.2,
        top_p=0.9,
        max_tokens=123,
        streaming=True,
        api_version="drop-me",
        custom="kept",
    )
    assert chat.kwargs == {
        "model_name": "claude-opus-4-6",
        "api_key": "anthropic-key",
        "streaming": True,
        "timeout": 120.0,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 123,
        "custom": "kept",
    }

    with pytest.raises(ValueError, match="Anthropic backend"):
        anthropic.create_anthropic_vlm()

    vlm = anthropic.create_anthropic_vlm(api_key="anthropic-key", model="claude")
    assert vlm.kwargs == {"api_key": "anthropic-key", "model": "claude"}


def test_openai_chat_and_vlm_factory_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=CapturingModel),
    )
    monkeypatch.setattr(vlm_models, "OpenAIVLM", CapturingModel)

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    assert openai_backend._resolve_base_url() == "https://api.openai.com/v1"

    chat = openai_backend.create_openai_chat(
        api_key="openai-key",
        base_url="https://api.openai.com/v1",
        temperature=0.1,
        top_p=0.4,
        max_tokens=99,
    )
    assert chat.kwargs["temperature"] == 0.1
    assert chat.kwargs["top_p"] == 0.4
    assert chat.kwargs["max_tokens"] == 99
    assert chat.kwargs["base_url"] == "https://api.openai.com/v1"

    vlm = openai_backend.create_openai_vlm(api_key="openai-key")
    assert vlm.kwargs["base_url"] == "https://api.openai.com/v1"
    assert vlm.kwargs["api_key"] == "openai-key"

    monkeypatch.setattr(image_models, "OpenAIImageGenerationModel", CapturingModel)
    image_model = openai_backend.create_openai_image_gen(
        api_key="openai-key", model="image-model"
    )
    assert image_model.kwargs == {"api_key": "openai-key", "model": "image-model"}


def test_nim_chat_factory_builds_model_and_applies_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=CapturingModel),
    )
    applied_timeouts: list[tuple[CapturingModel, float | None, str]] = []
    monkeypatch.setattr(
        nim,
        "_apply_nim_chat_timeout",
        lambda model, timeout, label: applied_timeouts.append((model, timeout, label)),
    )

    chat = nim.create_nim_chat(
        api_key="nim-key",
        temperature=0.2,
        top_p=0.6,
        max_tokens=111,
        streaming=True,
        timeout=9,
        api_version="drop-me",
        custom="kept",
    )

    assert chat.kwargs == {
        "model": "qwen/qwen3.5-397b-a17b",
        "nvidia_api_key": "nim-key",
        "streaming": True,
        "temperature": 0.2,
        "top_p": 0.6,
        "max_tokens": 111,
        "custom": "kept",
    }
    assert applied_timeouts == [(chat, 9, "create_nim_chat")]


def test_nim_timeout_none_is_noop() -> None:
    model = CapturingModel()

    nim_timeout._apply_nim_chat_timeout(model, None, label="test")

    assert not hasattr(model, "timeout")


def test_nim_vlm_and_image_generation_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vlm_models, "NvidiaNIMVLM", CapturingModel)
    monkeypatch.setattr(image_models, "NIMImageGenerationModel", CapturingModel)

    vlm = nim.create_nim_vlm(api_key="nim-key", model="vlm-model")
    assert vlm.kwargs == {"api_key": "nim-key", "model": "vlm-model"}

    image_model = nim.create_nim_image_gen(api_key="nim-key", model="image-model")
    assert image_model.kwargs == {"api_key": "nim-key", "model": "image-model"}


def test_gemini_chat_optional_sampling_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=CapturingModel),
    )

    chat = gemini.create_gemini_chat(
        api_key="gemini-key",
        temperature=0.3,
        top_p=0.8,
        max_tokens=321,
        api_version="drop-me",
        custom="kept",
    )

    assert chat.kwargs == {
        "model": "gemini-3-pro-preview",
        "google_api_key": "gemini-key",
        "streaming": False,
        "timeout": 120.0,
        "temperature": 0.3,
        "top_p": 0.8,
        "max_tokens": 321,
        "custom": "kept",
    }

    monkeypatch.setattr(vlm_models, "GeminiVLM", CapturingModel)
    monkeypatch.setattr(image_models, "GeminiImageGenerationModel", CapturingModel)

    vlm = gemini.create_gemini_vlm(api_key="gemini-key", model="vlm-model")
    assert vlm.kwargs == {"api_key": "gemini-key", "model": "vlm-model"}

    image_model = gemini.create_gemini_image_gen(model="image-model")
    assert image_model.kwargs == {"model": "image-model"}


def test_backend_registry_error_and_list_edges() -> None:
    assert "anthropic" in registry.list_chat_backends()
    assert "anthropic" in registry.list_vlm_backends()
    assert registry.list_image_gen_backends()

    assert registry.get_chat_factory("anthropic") is anthropic.create_anthropic_chat
    assert registry.get_vlm_factory("anthropic") is anthropic.create_anthropic_vlm
    assert registry.get_image_gen_factory("gemini") is gemini.create_gemini_image_gen

    with pytest.raises(ValueError, match="Unknown chat backend"):
        registry.get_chat_factory("missing-chat")
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        registry.get_vlm_factory("missing-vlm")
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        registry.get_image_gen_factory("missing-image")


def test_backend_registry_metadata_helpers_reject_unknown_backends() -> None:
    assert registry.vlm_backend_requires_api_key("openai") is True
    assert registry.vlm_backend_supports("openai", "reasoning_effort") is True

    with pytest.raises(ValueError, match="Unknown chat backend"):
        registry.chat_backend_requires_api_key("missing-chat-metadata")
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        registry.vlm_backend_requires_api_key("missing-vlm-metadata")
    with pytest.raises(ValueError, match="Unknown VLM backend"):
        registry.vlm_backend_supports("missing-vlm-capabilities", "reasoning_effort")
    with pytest.raises(ValueError, match="Unknown image generation backend"):
        registry.image_gen_backend_requires_api_key("missing-image-metadata")


def test_backend_registry_lists_registered_text_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_name = "test-text-embedding-list"

    monkeypatch.setitem(
        registry._text_embedding_backends,
        backend_name,
        CapturingModel,
    )

    assert backend_name in registry.list_text_embedding_backends()


def test_backend_plugins_load_once_without_package_name_coupling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeEntryPoint:
        name = "test-provider"
        value = "provider_package:register"

        @staticmethod
        def load():
            return lambda: calls.append("registered")

    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda **kwargs: [FakeEntryPoint()]
        if kwargs == {"group": "world_understanding.model_backends"}
        else [],
    )
    monkeypatch.setattr(registry, "_loaded_backend_plugins", set())

    assert registry.load_backend_plugins() == (
        "test-provider:provider_package:register",
    )
    assert registry.load_backend_plugins() == (
        "test-provider:provider_package:register",
    )
    assert calls == ["registered"]


def test_optional_provider_can_register_credential_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credentials, "API_KEY_ENV_VAR_MAP", {})
    credentials.register_api_key_env_vars("test-provider", "TEST_PROVIDER_KEY")
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret")

    assert credentials.get_env_api_key_for_backend("test-provider") == "secret"


def test_text_embedding_plugin_preserves_provider_default_model() -> None:
    calls: list[dict[str, Any]] = []

    def create_text_embedding(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return CapturingModel(**kwargs)

    registry.register_text_embedding_backend(
        "test-text-embedding-default", create_text_embedding
    )

    text_models.create_text_embedding_model(
        "test-text-embedding-default", api_key="key"
    )
    text_models.create_text_embedding_model(
        "test-text-embedding-default", api_key="key", model="explicit"
    )

    assert calls == [
        {"api_key": "key"},
        {"api_key": "key", "model": "explicit"},
    ]
