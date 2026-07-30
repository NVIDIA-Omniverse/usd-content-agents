# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for vision-language model construction."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.functions.models import vision_language_models as vlm_module
from world_understanding.functions.models.token_limits import (
    clamp_model_output_tokens,
    model_output_token_cap,
    normalize_openai_token_kwargs,
    openai_token_parameter,
)
from world_understanding.functions.models.vision_language_models import (
    NvidiaNIMVLM,
    create_vlm,
)
from world_understanding.telemetry import GenAIAttributes


def test_create_gemini_vlm_accepts_gemini_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test direct Gemini VLM construction accepts GEMINI_API_KEY."""
    captured: dict[str, object] = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )

    import world_understanding.functions.models.backends  # noqa: F401

    vlm = create_vlm("gemini")

    assert vlm.backend_name == "gemini"
    assert captured["google_api_key"] == "gemini-key"


def test_create_gemini_vlm_replaces_placeholder_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct Gemini VLM construction should not pass placeholders to LangChain."""
    captured: dict[str, object] = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key")
    monkeypatch.setitem(
        sys.modules,
        "langchain_google_genai",
        SimpleNamespace(ChatGoogleGenerativeAI=FakeChatGoogleGenerativeAI),
    )

    import world_understanding.functions.models.backends  # noqa: F401

    vlm = create_vlm("gemini", api_key="YOUR_GOOGLE_API_KEY")

    assert vlm.backend_name == "gemini"
    assert captured["google_api_key"] == "real-gemini-key"


def test_create_openai_vlm_rejects_explicit_key_with_env_redirected_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` redirects the OpenAI SDK; an explicit hosted
    ``OPENAI_API_KEY`` passed directly to the VLM factory must not follow
    that redirect to a non-provider endpoint without an explicit
    ``base_url`` pairing."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        create_vlm("openai", api_key="sk-real-openai-key")


def test_nvidia_nim_vlm_omits_constructor_timeout_and_sets_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NIM timeout must not be serialized as a chat-completion body field."""
    captured: dict[str, object] = {}
    sync_client = SimpleNamespace(timeout=None)
    async_client = SimpleNamespace(timeout=None)

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.max_tokens = 1024
            self._client = sync_client
            self._async_client = async_client

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )

    vlm = NvidiaNIMVLM(
        api_key="test-key",
        model="test-model",
        timeout=42,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    assert captured == {
        "model": "test-model",
        "nvidia_api_key": "test-key",
        "base_url": "https://integrate.api.nvidia.com/v1",
    }
    assert vlm.chat_model.max_tokens is None
    assert sync_client.timeout == 42.0
    assert async_client.timeout == 42.0


def test_nvidia_nim_vlm_warns_when_timeout_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ChatNVIDIA client attrs should surface as a warning."""
    captured: dict[str, object] = {}

    class FakeChatNVIDIA:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.max_tokens = 1024

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )

    with caplog.at_level("WARNING"):
        vlm = NvidiaNIMVLM(
            api_key="test-key",
            model="test-model",
            timeout=42,
        )

    assert captured == {
        "model": "test-model",
        "nvidia_api_key": "test-key",
    }
    assert vlm.chat_model.max_tokens is None
    assert "NvidiaNIMVLM could not apply timeout=42.0" in caplog.text


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-20241120",
        "openai/openai/gpt-4o-mini-2024-07-18",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "azure/openai/gpt-4.1-2025-04-14",
    ],
)
def test_openai_output_cap_resolves_supported_model_aliases(model_name: str) -> None:
    assert model_output_token_cap(model_name) == 16_384
    assert clamp_model_output_tokens(model_name, 24_576) == 16_384
    assert clamp_model_output_tokens(model_name, 8_000) == 8_000


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-5.4",
        "claude-opus-4-6",
        "google/gemma-4-31b-it",
        "unknown-model",
        "gpt-4.10",
        "custom-gpt-4o-deployment",
        "azure/prod-gpt-5-20250807",
    ],
)
def test_openai_output_cap_does_not_guess_unknown_models(model_name: str) -> None:
    assert model_output_token_cap(model_name) is None
    assert clamp_model_output_tokens(model_name, 24_576) == 24_576


def test_openai_token_normalization_uses_one_correct_parameter() -> None:
    assert normalize_openai_token_kwargs("gpt-4o", 24_576, {}) == {"max_tokens": 16_384}
    assert normalize_openai_token_kwargs("gpt-4.1", 8_000, {}) == {"max_tokens": 8_000}
    assert normalize_openai_token_kwargs(
        "azure/openai/gpt-4.1-mini",
        None,
        {"max_completion_tokens": 24_576, "extra": "kept", "drop": None},
    ) == {"max_tokens": 16_384, "extra": "kept"}
    assert normalize_openai_token_kwargs("openai/openai/gpt-5.4", 24_576, {}) == {
        "max_completion_tokens": 24_576
    }
    assert normalize_openai_token_kwargs(
        "gpt-4o",
        24_576,
        {"max_tokens": 8_000, "max_completion_tokens": 12_000},
    ) == {"max_tokens": 8_000}
    assert normalize_openai_token_kwargs(
        "gpt-5.4",
        24_576,
        {"max_tokens": 8_000, "max_completion_tokens": 12_000},
    ) == {"max_completion_tokens": 12_000}
    assert normalize_openai_token_kwargs(
        "gpt-4o",
        8_000,
        {"max_tokens": 12_000, "max_completion_tokens": 14_000},
        prefer_max_tokens_argument=True,
    ) == {"max_tokens": 8_000}
    assert openai_token_parameter("openai/openai/gpt-5.4") == "max_completion_tokens"
    assert (
        openai_token_parameter("azure/prod-gpt-5-20250807") == "max_completion_tokens"
    )
    assert normalize_openai_token_kwargs(
        "azure/prod-gpt-5-20250807",
        None,
        {"max_completion_tokens": 24_576},
    ) == {"max_completion_tokens": 24_576}
    assert openai_token_parameter("not-gpt-50") == "max_tokens"


class _CapturingLangChainChat:
    def __init__(self, **constructor_kwargs: Any) -> None:
        self.constructor_kwargs = constructor_kwargs
        self.calls: list[dict[str, Any]] = []

    def invoke(self, messages: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        requested = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        model = self.constructor_kwargs.get("model")
        cap = model_output_token_cap(model if isinstance(model, str) else None)
        if isinstance(requested, int) and cap is not None and requested > cap:
            raise AssertionError("provider would reject this output-token request")
        return SimpleNamespace(content="ok", usage_metadata=None)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> SimpleNamespace:
        return self.invoke(messages, **kwargs)


def test_public_openai_factory_caps_constructor_and_request_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_CapturingLangChainChat] = []
    trace_attributes: dict[str, Any] = {}

    class CapturingSpan:
        def set_attribute(self, key: str, value: Any) -> None:
            trace_attributes[key] = value

    class FakeChatOpenAI(_CapturingLangChainChat):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    monkeypatch.setattr(vlm_module, "get_current_span", lambda: CapturingSpan())
    import world_understanding.functions.models.backends  # noqa: F401

    constructor_vlm = create_vlm(
        "openai",
        api_key="test-key",
        model="gpt-4o",
        max_tokens=24_576,
    )
    assert created[-1].constructor_kwargs["max_tokens"] == 16_384
    assert "max_completion_tokens" not in created[-1].constructor_kwargs

    constructor_vlm.generate("describe", max_tokens=24_576)
    assert created[-1].calls[-1]["max_tokens"] == 16_384
    assert trace_attributes[GenAIAttributes.REQUEST_MAX_TOKENS] == 16_384

    gpt5_vlm = create_vlm(
        "openai",
        api_key="test-key",
        model="gpt-5.4",
        max_tokens=24_576,
    )
    assert created[-1].constructor_kwargs["max_completion_tokens"] == 24_576
    assert "max_tokens" not in created[-1].constructor_kwargs

    gpt5_vlm.generate("describe", max_tokens=24_576)
    assert created[-1].calls[-1]["max_completion_tokens"] == 24_576

    create_vlm(
        "openai",
        api_key="test-key",
        model="gpt-5.4",
        max_completion_tokens=12_000,
    )
    assert created[-1].constructor_kwargs["max_completion_tokens"] == 12_000
    assert "max_tokens" not in created[-1].constructor_kwargs


def test_shipped_default_reaches_public_openai_request_with_safe_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_CapturingLangChainChat] = []

    class FakeChatOpenAI(_CapturingLangChainChat):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    from material_agent.api.defaults import DEFAULT_VLM_MAX_TOKENS

    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.agentic.domain_tasks.model_provisioning import (
        ModelProvisioningTask,
    )

    assert DEFAULT_VLM_MAX_TOKENS == 24_576

    context: dict[str, Any] = {
        "config": {
            "vlm": {
                "backend": "openai",
                "model": "gpt-4o-mini",
                "api_key": "test-key",
                "max_tokens": DEFAULT_VLM_MAX_TOKENS,
            }
        }
    }
    ModelProvisioningTask().run(context)

    assert context["vlm_invoke_kwargs"]["max_tokens"] == 24_576
    context["vlm"].generate("describe", **context["vlm_invoke_kwargs"])
    assert created[-1].calls[-1]["max_tokens"] == 16_384
