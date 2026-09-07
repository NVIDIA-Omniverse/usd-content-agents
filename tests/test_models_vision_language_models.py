# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for vision-language model construction."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from requests.exceptions import Timeout as RequestsTimeout

from world_understanding.functions.models import vision_language_models as vlm_module
from world_understanding.functions.models.backends import registry as backend_registry
from world_understanding.functions.models.token_limits import (
    backend_supports_reasoning_effort,
    clamp_model_output_tokens,
    ensure_model_output_token_budget,
    model_output_token_cap,
    model_output_token_floor,
    model_reasoning_effort_default,
    model_uses_openai_responses_api,
    normalize_openai_token_kwargs,
    openai_token_parameter,
    resolve_reasoning_effort_for_backend,
)
from world_understanding.functions.models.vision_language_models import (
    NonRetryableVLMTimeoutError,
    NvidiaNIMVLM,
    create_vlm,
)
from world_understanding.telemetry import GenAIAttributes
from world_understanding.utils.model_timeout import (
    NON_RETRYABLE_VLM_TIMEOUT_MESSAGE,
)


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
    assert vlm.has_bounded_request_timeout is True


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
    assert vlm.has_bounded_request_timeout is False
    assert "NvidiaNIMVLM could not apply timeout=42.0" in caplog.text


def test_nvidia_nim_vlm_uses_kimi_k3_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChatNVIDIA:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.max_tokens = 1024
            self._client = SimpleNamespace(timeout=None)
            self._async_client = SimpleNamespace(timeout=None)

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )

    NvidiaNIMVLM(api_key="test-key")

    assert captured["model"] == "moonshotai/kimi-k3"


@pytest.mark.parametrize("error_type", [TimeoutError, RequestsTimeout])
@pytest.mark.parametrize(
    "invocation",
    ["generate", "generate_with_image_caption_pairs"],
)
def test_nvidia_nim_vlm_translates_retry_unsafe_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
    invocation: str,
) -> None:
    """Both NIM invoke paths must expose a fixed, non-retryable timeout."""

    class FakeChatNVIDIA:
        def __init__(self, **_kwargs: Any) -> None:
            self.max_tokens = 1024
            self._client = SimpleNamespace(timeout=None)
            self._async_client = SimpleNamespace(timeout=None)

        def invoke(self, _messages: Any, **_kwargs: Any) -> None:
            raise error_type("provider-secret timeout detail")

    monkeypatch.setitem(
        sys.modules,
        "langchain_nvidia_ai_endpoints",
        SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
    )
    vlm = NvidiaNIMVLM(api_key="test-key", timeout=13)

    with pytest.raises(NonRetryableVLMTimeoutError) as exc_info:
        if invocation == "generate":
            vlm.generate("prompt", images=None)
        else:
            vlm.generate_with_image_caption_pairs([], "prompt")

    assert str(exc_info.value) == NON_RETRYABLE_VLM_TIMEOUT_MESSAGE
    assert "provider-secret" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


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


def test_sol_output_budget_leaves_room_for_xhigh_reasoning() -> None:
    assert model_output_token_floor("openai/openai/gpt-5.6-sol") == 16_384
    assert ensure_model_output_token_budget("gpt-5.6-sol", 256) == 16_384
    assert ensure_model_output_token_budget("gpt-5.6-sol", 24_576) == 24_576
    assert ensure_model_output_token_budget("gpt-5.6-sol", None) is None
    assert normalize_openai_token_kwargs("gpt-5.6-sol", 512, {}) == {
        "max_completion_tokens": 16_384
    }


def test_kimi_k3_output_budget_leaves_room_for_reasoning() -> None:
    assert model_output_token_floor("moonshotai/kimi-k3") == 16_384
    assert ensure_model_output_token_budget("kimi-k3", 512) == 16_384
    assert ensure_model_output_token_budget("kimi-k3", 24_576) == 24_576
    assert ensure_model_output_token_budget("other-nim-model", 512) == 512


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


def test_model_reasoning_effort_default_uses_sol_ceiling() -> None:
    assert (
        model_reasoning_effort_default("openai/openai/gpt-5.6-sol", fallback="high")
        == "xhigh"
    )
    assert model_reasoning_effort_default("gpt-5.5", fallback="high") == "high"
    assert model_reasoning_effort_default("moonshotai/kimi-k3") == "max"
    assert model_uses_openai_responses_api("openai/openai/gpt-5.6-sol")
    assert not model_uses_openai_responses_api("gpt-5.5")


def test_reasoning_effort_respects_backend_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_only_backend = "reasoning-chat-only"
    monkeypatch.setitem(
        backend_registry._chat_backends,
        chat_only_backend,
        lambda **_kwargs: None,
    )
    monkeypatch.setitem(
        backend_registry._chat_backend_capabilities,
        chat_only_backend,
        frozenset({"reasoning_effort"}),
    )

    assert not backend_supports_reasoning_effort(chat_only_backend)
    assert backend_supports_reasoning_effort(chat_only_backend, interface="chat")
    assert backend_supports_reasoning_effort("openai")
    assert backend_supports_reasoning_effort("openai", interface="chat")
    assert not backend_supports_reasoning_effort("nim")
    assert backend_supports_reasoning_effort(
        "nim",
        model_name="moonshotai/kimi-k3",
    )
    assert not backend_supports_reasoning_effort(
        "nim",
        model_name="google/gemma-4-31b-it",
    )
    assert backend_supports_reasoning_effort(
        "nim",
        model_name="moonshotai/kimi-k3",
        interface="chat",
    )
    assert not backend_supports_reasoning_effort("missing-backend")
    assert not backend_supports_reasoning_effort(None)
    with pytest.raises(ValueError, match="Unknown model backend interface"):
        backend_supports_reasoning_effort(
            "openai",
            interface="audio",  # type: ignore[arg-type]
        )
    assert (
        resolve_reasoning_effort_for_backend(
            "openai",
            "openai/openai/gpt-5.6-sol",
        )
        == "xhigh"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "openai",
            "custom-model",
            explicit="medium",
            interface="chat",
        )
        == "medium"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "openai",
            "custom-model",
            fallback="high",
        )
        == "high"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "nim",
            "moonshotai/kimi-k3",
        )
        == "max"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "nim",
            "moonshotai/kimi-k3",
            interface="chat",
        )
        == "max"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "nim",
            "moonshotai/kimi-k3",
            explicit="low",
        )
        == "low"
    )
    assert (
        resolve_reasoning_effort_for_backend(
            "nim",
            "openai/openai/gpt-5.6-sol",
            explicit="xhigh",
        )
        is None
    )
