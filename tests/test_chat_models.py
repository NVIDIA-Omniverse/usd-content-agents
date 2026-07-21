# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for chat model implementations."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from world_understanding.functions.models.chat_models import (
    _DEFAULT_NIM_MODEL,
    EchoChatModel,
    create_chat_model,
    create_echo_chat_model,
    create_nim_chat_model,
)


class TestEchoChatModel:
    """Test cases for EchoChatModel."""

    def test_echo_model_creation(self):
        """Test basic echo model creation."""
        model = create_echo_chat_model()
        assert isinstance(model, EchoChatModel)
        assert model.prefix == "Echo: "

    def test_echo_model_custom_prefix(self):
        """Test echo model with custom prefix."""
        model = create_echo_chat_model(prefix="Reply: ")
        assert model.prefix == "Reply: "

    def test_echo_generates_response(self):
        """Test echo model generates expected response."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Hello")]
        result = model._generate(messages)

        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Echo: Hello"

    def test_echo_with_system_message(self):
        """Test echo model handles system + human messages."""
        model = create_echo_chat_model()
        messages = [
            SystemMessage(content="You are a test bot."),
            HumanMessage(content="Test message"),
        ]
        result = model._generate(messages)

        # Should echo the last message (human message)
        assert result.generations[0].message.content == "Echo: Test message"

    def test_echo_stores_request(self):
        """Test echo model stores last request for testing."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Test")]
        model._generate(messages, stop=["STOP"], extra_param="value")

        assert model.last_request is not None
        assert model.last_request["messages"] == messages
        assert model.last_request["stop"] == ["STOP"]
        assert model.last_request["extra_param"] == "value"

    def test_echo_with_dict_message(self):
        """Test echo model handles dict-format messages."""
        model = create_echo_chat_model()
        messages = [{"content": "Dict message"}]
        result = model._generate(messages)

        assert result.generations[0].message.content == "Echo: Dict message"

    def test_echo_empty_messages(self):
        """Test echo model handles empty message list."""
        model = create_echo_chat_model()
        result = model._generate([])

        assert result.generations[0].message.content == "Echo: "

    def test_echo_returns_ai_message(self):
        """Test echo model returns AIMessage."""
        model = create_echo_chat_model()
        messages = [HumanMessage(content="Test")]
        result = model._generate(messages)

        assert isinstance(result.generations[0].message, AIMessage)

    @pytest.mark.asyncio
    async def test_echo_async_generate_delegates_to_sync_generate(self):
        """Test async echo generation."""
        model = create_echo_chat_model()
        result = await model._agenerate([HumanMessage(content="Async")])

        assert result.generations[0].message.content == "Echo: Async"

    def test_echo_llm_type(self):
        """Test echo model returns correct LLM type."""
        model = create_echo_chat_model()
        assert model._llm_type == "echo"


class TestNIMChatModel:
    """Test cases for NIM chat model creation."""

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_default_params(self, mock_nvidia):
        """Test creating NIM model with defaults.

        ``streaming=False`` and ``timeout`` are not declared ctor fields on
        the installed langchain_nvidia_ai_endpoints version; passing them
        routes into ``model_kwargs``, which gets serialized as request-body
        fields. Strict NIM serving (e.g. Nemotron Nano 8B) rejects unknown
        body fields with "400 extra_forbidden". The factory therefore only
        forwards ``streaming`` when the caller explicitly requests it and
        never forwards ``timeout``. Timeout is applied to the underlying
        HTTP clients after construction when those clients are available.
        """
        mock_instance = Mock()
        mock_nvidia.return_value = mock_instance

        result = create_nim_chat_model(api_key="test_key")

        assert result == mock_instance
        mock_nvidia.assert_called_once_with(
            model=_DEFAULT_NIM_MODEL,
            nvidia_api_key="test_key",
        )

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_custom_params(self, mock_nvidia):
        """Test creating NIM model with custom parameters.

        See ``test_create_nim_default_params`` for why ``timeout`` is not
        forwarded. ``streaming=True`` *is* forwarded because the caller
        asked for it explicitly.
        """
        mock_instance = Mock()
        mock_nvidia.return_value = mock_instance

        result = create_nim_chat_model(
            api_key="test_key",
            model="custom-model",
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
            streaming=True,
            custom_param="value",
        )

        assert result == mock_instance
        mock_nvidia.assert_called_once_with(
            model="custom-model",
            nvidia_api_key="test_key",
            streaming=True,
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
            custom_param="value",
        )

    def test_create_nim_applies_timeout_to_underlying_clients(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Timeout must stay out of ctor kwargs and land on the HTTP clients."""
        captured: dict[str, object] = {}
        sync_client = SimpleNamespace(timeout=None)
        async_client = SimpleNamespace(timeout=None)

        class FakeChatNVIDIA:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self._client = sync_client
                self._async_client = async_client

        monkeypatch.setitem(
            sys.modules,
            "langchain_nvidia_ai_endpoints",
            SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
        )

        result = create_nim_chat_model(
            api_key="test_key",
            model="custom-model",
            timeout=42,
            custom_param="value",
        )

        assert isinstance(result, FakeChatNVIDIA)
        assert captured == {
            "model": "custom-model",
            "nvidia_api_key": "test_key",
            "custom_param": "value",
        }
        assert sync_client.timeout == 42.0
        assert async_client.timeout == 42.0

    def test_create_nim_applies_default_timeout_to_underlying_clients(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The chat path should get the same default timeout as the VLM path."""
        captured: dict[str, object] = {}
        sync_client = SimpleNamespace(timeout=None)
        async_client = SimpleNamespace(timeout=None)

        class FakeChatNVIDIA:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self._client = sync_client
                self._async_client = async_client

        monkeypatch.setitem(
            sys.modules,
            "langchain_nvidia_ai_endpoints",
            SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
        )

        result = create_nim_chat_model(
            api_key="test_key",
            model="custom-model",
        )

        assert isinstance(result, FakeChatNVIDIA)
        assert captured == {
            "model": "custom-model",
            "nvidia_api_key": "test_key",
        }
        assert sync_client.timeout == 120.0
        assert async_client.timeout == 120.0

    @pytest.mark.parametrize(
        ("has_sync_client", "has_async_client", "expect_warning"),
        [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_create_nim_timeout_handles_missing_client_attrs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        has_sync_client: bool,
        has_async_client: bool,
        expect_warning: bool,
    ) -> None:
        """Missing ChatNVIDIA client attrs should be observable only if none work."""
        sync_client = SimpleNamespace(timeout=None)
        async_client = SimpleNamespace(timeout=None)

        class FakeChatNVIDIA:
            def __init__(self, **kwargs):
                if has_sync_client:
                    self._client = sync_client
                if has_async_client:
                    self._async_client = async_client

        monkeypatch.setitem(
            sys.modules,
            "langchain_nvidia_ai_endpoints",
            SimpleNamespace(ChatNVIDIA=FakeChatNVIDIA),
        )

        with caplog.at_level("WARNING"):
            result = create_nim_chat_model(
                api_key="test_key",
                model="custom-model",
                timeout=42,
            )

        assert isinstance(result, FakeChatNVIDIA)
        assert sync_client.timeout == (42.0 if has_sync_client else None)
        assert async_client.timeout == (42.0 if has_async_client else None)
        warning_text = "create_nim_chat could not apply timeout=42.0"
        if expect_warning:
            assert warning_text in caplog.text
        else:
            assert warning_text not in caplog.text


# Tests for private Azure OpenAI backends live under tests/internal/.


class TestCreateChatModel:
    """Test cases for the unified create_chat_model function."""

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_backend(self, mock_nvidia):
        """Test creating NIM backend through unified function."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        result = create_chat_model(
            backend="nim", api_key="test_key", model="test-model"
        )

        assert result == mock_model

    def test_create_nim_without_key(self, monkeypatch):
        """Test NIM backend requires API key."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(backend="nim")

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_remote_base_url_requires_real_key(
        self, mock_nvidia, monkeypatch
    ):
        """Remote/custom NIM endpoints should not get an implicit placeholder."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(
                backend="nim",
                base_url="https://nim.example.com/v1",
            )
        mock_nvidia.assert_not_called()

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_custom_remote_base_url_requires_endpoint_key(
        self, mock_nvidia, monkeypatch
    ):
        """Custom NIM endpoints must not receive the hosted NVIDIA_API_KEY."""
        monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
        monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(
                backend="nim",
                base_url="https://nim.example.com/v1",
            )
        mock_nvidia.assert_not_called()

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_custom_remote_base_url_accepts_explicit_endpoint_key(
        self, mock_nvidia, monkeypatch
    ):
        """Custom NIM endpoints may use an explicit endpoint-scoped key."""
        monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        result = create_chat_model(
            backend="nim",
            api_key="endpoint-nim-key",
            base_url="https://nim.example.com/v1",
        )

        assert result == mock_model
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["nvidia_api_key"] == "endpoint-nim-key"
        assert call_kwargs["base_url"] == "https://nim.example.com/v1"

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_remote_base_url_ignores_ma_nim_key(
        self, mock_nvidia, monkeypatch
    ):
        """Hosted NIM endpoints must not receive the local sidecar key."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("MA_NIM_API_KEY", "local-sidecar-key")

        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(
                backend="nim",
                base_url="https://inference-api.nvidia.com/v1",
            )
        mock_nvidia.assert_not_called()

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_local_base_url_requires_explicit_placeholder(
        self, mock_nvidia, monkeypatch
    ):
        """Local NIM no-auth mode requires the documented opt-in value."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        result = create_chat_model(
            backend="nim",
            base_url="http://llm-nim:8000/v1",
        )

        assert result == mock_model
        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["nvidia_api_key"] == "not-used"
        assert call_kwargs["base_url"] == "http://llm-nim:8000/v1"

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_nim_local_base_url_ignores_global_nvidia_key(
        self, mock_nvidia, monkeypatch
    ):
        """Local NIM endpoints must not receive the hosted NVIDIA_API_KEY."""
        monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
        monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key is required"):
            create_chat_model(
                backend="nim",
                base_url="http://llm-nim:8000/v1",
            )
        mock_nvidia.assert_not_called()

    def test_create_echo_backend(self):
        """Test creating echo backend through unified function."""
        result = create_chat_model(backend="echo", prefix="Test: ")

        assert isinstance(result, EchoChatModel)
        assert result.prefix == "Test: "

    def test_create_echo_ignores_api_key(self):
        """Test echo backend ignores API key parameter."""
        model = create_chat_model(
            backend="echo",
            api_key="ignored",
            temperature=0.1,  # Also ignored
        )

        assert isinstance(model, EchoChatModel)
        assert model.prefix == "Echo: "  # Default prefix

    def test_create_gemini_accepts_gemini_api_key_alias(self, monkeypatch):
        """Test Gemini chat backend accepts GEMINI_API_KEY."""
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

        result = create_chat_model(backend="gemini")

        assert isinstance(result, FakeChatGoogleGenerativeAI)
        assert captured["google_api_key"] == "gemini-key"

    def test_create_unknown_backend(self):
        """Test error for unknown backend."""
        with pytest.raises(ValueError, match="Unknown chat backend: unknown"):
            create_chat_model(backend="unknown")

    def test_create_openai_rejects_explicit_key_with_env_redirected_base_url(
        self, monkeypatch
    ):
        """``OPENAI_BASE_URL`` redirects the OpenAI SDK; an explicit hosted
        ``OPENAI_API_KEY`` passed directly to the factory must not follow
        the redirect to a non-provider endpoint the caller didn't pair it
        with via ``base_url``."""
        monkeypatch.setenv(
            "OPENAI_BASE_URL", "https://api.openai-compatible.example/v1"
        )
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            create_chat_model(backend="openai", api_key="sk-real-openai-key")

    def test_create_openai_rejects_explicit_key_with_legacy_env_base_url(
        self, monkeypatch
    ):
        """Legacy ``OPENAI_API_BASE`` env var triggers the same protection."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv(
            "OPENAI_API_BASE", "https://api.openai-compatible.example/v1"
        )

        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            create_chat_model(backend="openai", api_key="sk-real-openai-key")

    @patch("langchain_openai.ChatOpenAI")
    def test_create_openai_accepts_explicit_key_paired_with_base_url(
        self, mock_chat_openai, monkeypatch
    ):
        """Explicit ``api_key`` paired with explicit ``base_url`` is the
        documented way to point at a custom OpenAI-compatible endpoint, and
        must continue to work even when env vars also point elsewhere."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example/v1")
        mock_model = Mock()
        mock_chat_openai.return_value = mock_model

        result = create_chat_model(
            backend="openai",
            api_key="sk-paired-endpoint-key",
            base_url="https://api.openai-compatible.example/v1",
        )

        assert result is mock_model
        call_kwargs = mock_chat_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-paired-endpoint-key"
        assert call_kwargs["base_url"] == "https://api.openai-compatible.example/v1"

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_with_default_model(self, mock_nvidia):
        """Test using default model when not specified."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        create_chat_model(backend="nim", api_key="key")

        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["model"] == _DEFAULT_NIM_MODEL

    @patch("langchain_nvidia_ai_endpoints.ChatNVIDIA")
    def test_create_with_streaming(self, mock_nvidia):
        """Test enabling streaming mode."""
        mock_model = Mock()
        mock_nvidia.return_value = mock_model

        create_chat_model(backend="nim", api_key="key", streaming=True)

        mock_nvidia.assert_called_once()
        call_kwargs = mock_nvidia.call_args[1]
        assert call_kwargs["streaming"] is True
