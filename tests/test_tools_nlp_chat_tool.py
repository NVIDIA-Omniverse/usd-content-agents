# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for chat tool."""

import os
from unittest.mock import patch

import pytest

from world_understanding.tools.nlp.chat_tool import (
    ChatInput,
    ChatOutput,
    _display_chat_response,
    chat_tool,
)


@pytest.fixture(autouse=True)
def _default_nim_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a deterministic key for tests using the default NIM backend."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")


@pytest.fixture(autouse=True)
def _stub_chat_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid live backend model initialization in chat tool unit tests."""
    monkeypatch.setattr(
        "world_understanding.tools.nlp.chat_tool.create_chat_model",
        lambda **_kwargs: object(),
    )


class TestChatInput:
    """Tests for ChatInput model."""

    def test_valid_input(self):
        """Test creating valid ChatInput."""
        input_obj = ChatInput(
            prompt="Hello, how are you?",
            backend="nim",
            model="llama",
            temperature=0.7,
            max_tokens=100,
        )

        assert input_obj.prompt == "Hello, how are you?"
        assert input_obj.backend == "nim"
        assert input_obj.model == "llama"
        assert input_obj.temperature == 0.7
        assert input_obj.max_tokens == 100

    def test_default_values(self):
        """Test default values for optional fields."""
        input_obj = ChatInput(prompt="Test prompt")

        assert input_obj.backend == "nim"  # default
        assert input_obj.model is None  # default (backend provides its own default)
        assert input_obj.temperature == 0.7  # default
        assert input_obj.max_tokens == 1024  # default

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        ChatInput(prompt="test", temperature=0.0)
        ChatInput(prompt="test", temperature=1.0)
        ChatInput(prompt="test", temperature=0.5)

        # Invalid temperatures
        with pytest.raises(ValueError):
            ChatInput(prompt="test", temperature=-0.1)

        with pytest.raises(ValueError):
            ChatInput(prompt="test", temperature=2.1)  # max is 2.0

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        ChatInput(prompt="test", max_tokens=1)
        ChatInput(prompt="test", max_tokens=8192)  # max is 8192

        # Invalid max_tokens
        with pytest.raises(ValueError):
            ChatInput(prompt="test", max_tokens=0)

        with pytest.raises(ValueError):
            ChatInput(prompt="test", max_tokens=8193)  # exceeds max


class TestChatOutput:
    """Tests for ChatOutput model."""

    def test_valid_output(self):
        """Test creating valid ChatOutput."""
        output = ChatOutput(
            response="Hello! I can help you with that.",
            backend_used="nim",
            model_used="llama-2-7b",
        )

        assert output.response == "Hello! I can help you with that."
        assert output.backend_used == "nim"
        assert output.model_used == "llama-2-7b"

    def test_optional_fields(self):
        """Test optional fields in output."""
        output = ChatOutput(
            response="Test response",
            backend_used="echo",
        )

        assert output.response == "Test response"
        assert output.backend_used == "echo"
        assert output.model_used is None  # model_used is optional


class RecordingConsole:
    """Minimal console double that records print calls."""

    def __init__(self) -> None:
        self.calls = []

    def print(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def test_display_chat_response_includes_optional_model() -> None:
    console = RecordingConsole()

    _display_chat_response(
        {
            "backend_used": "openai",
            "model_used": "gpt-5.4",
            "response": "Hello from the model.",
        },
        console,
        indent="  ",
    )

    rendered = "\n".join(str(call[0][0]) for call in console.calls)
    assert "Chat Response" in rendered
    assert "Backend: openai" in rendered
    assert "Model: gpt-5.4" in rendered
    assert "Hello from the model." in rendered


class TestChatTool:
    """Tests for chat_tool function."""

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_basic_chat(self, mock_generate):
        """Test basic chat functionality."""
        # Mock the response
        mock_generate.return_value = {
            "response": "Hello! How can I help you?",
            "model": "gpt-4",
            "token_count": 8,
        }

        inputs = ChatInput(prompt="Hello!")
        output = chat_tool(inputs)

        assert isinstance(output, ChatOutput)
        assert output.response == "Hello! How can I help you?"
        assert output.backend_used == "nim"
        assert output.model_used is None  # backend provides its own default

        # Verify the function was called with correct parameters
        mock_generate.assert_called_once()
        # Verify the function was called with correct parameters
        call_args = mock_generate.call_args[1]
        assert call_args["prompt"] == "Hello!"

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_custom_backend(self, mock_generate):
        """Test chat with custom backend."""
        mock_generate.return_value = {
            "response": "OpenAI response",
            "model": "gpt-4",
        }

        inputs = ChatInput(
            prompt="Test prompt",
            backend="openai",
            api_key="test-key",
            model="gpt-4",
        )
        output = chat_tool(inputs)

        assert output.response == "OpenAI response"
        assert output.backend_used == "openai"
        assert output.model_used == "gpt-4"

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_temperature_and_max_tokens(self, mock_generate):
        """Test chat with custom temperature and max_tokens."""
        mock_generate.return_value = {"response": "Short response"}

        inputs = ChatInput(
            prompt="Be creative",
            temperature=0.9,
            max_tokens=50,
        )
        chat_tool(inputs)

        mock_generate.call_args[1]
        # Note: These parameters might not be passed directly in the current implementation
        # This test verifies the tool accepts them without error

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_empty_response(self, mock_generate):
        """Test handling of empty response."""
        mock_generate.return_value = {"response": ""}

        inputs = ChatInput(prompt="Test")
        output = chat_tool(inputs)

        assert output.response == ""

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_long_prompt(self, mock_generate):
        """Test chat with a long prompt."""
        long_prompt = "This is a very long prompt. " * 100
        mock_generate.return_value = {"response": "Response to long prompt"}

        inputs = ChatInput(prompt=long_prompt)
        output = chat_tool(inputs)

        assert output.response == "Response to long prompt"
        call_args = mock_generate.call_args[1]
        assert call_args["prompt"] == long_prompt

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_error_handling(self, mock_generate):
        """Test error handling in chat tool."""
        mock_generate.side_effect = Exception("API Error")

        inputs = ChatInput(prompt="Test")
        output = chat_tool(inputs)

        # The tool catches exceptions and returns error message
        assert "Error generating response" in output.response
        assert "API Error" in output.response

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_echo_backend(self, mock_generate):
        """Test chat with echo backend."""
        mock_generate.return_value = {
            "response": "Echo: Test prompt",
            "model": "echo",
        }

        inputs = ChatInput(
            prompt="Test prompt",
            backend="echo",
        )
        output = chat_tool(inputs)

        assert output.response == "Echo: Test prompt"
        assert output.backend_used == "echo"
        assert output.model_used is None  # backend provides its own default

    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_response_format(self, mock_generate):
        """Test that chat response has correct format."""
        mock_generate.return_value = {
            "response": "Formatted response",
            "model": "test-model",
            "token_count": 3,
        }

        inputs = ChatInput(prompt="Format test")
        output = chat_tool(inputs)

        # Verify output is properly structured
        assert hasattr(output, "response")
        assert hasattr(output, "backend_used")
        assert hasattr(output, "model_used")
        assert isinstance(output.response, str)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_env_api_key(self, mock_generate):
        """Test that chat tool can use environment variables."""
        mock_generate.return_value = {"response": "Response with env key"}

        inputs = ChatInput(
            prompt="Test with env",
            backend="openai",
        )
        output = chat_tool(inputs)

        assert output.response == "Response with env key"
        assert output.backend_used == "openai"

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-real-openai-key",
            "OPENAI_BASE_URL": "https://api.openai-compatible.example/v1",
        },
        clear=True,
    )
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    @patch("world_understanding.tools.nlp.chat_tool.create_chat_model")
    def test_chat_openai_rejects_hosted_key_with_env_redirected_base_url(
        self, mock_create, mock_generate
    ):
        """``OPENAI_BASE_URL`` redirects the OpenAI SDK to a custom endpoint;
        the chat tool must not silently forward the hosted ``OPENAI_API_KEY``
        to that endpoint, even though the tool API has no ``base_url`` field.
        """
        mock_generate.return_value = {"response": "ok"}
        inputs = ChatInput(prompt="Hi", backend="openai")

        chat_tool(inputs)

        assert mock_create.call_count == 1
        call_kwargs = mock_create.call_args.kwargs
        # The hosted ``OPENAI_API_KEY`` must not flow through to the OpenAI
        # factory when the env-resolved base URL is a non-openai.com host.
        assert call_kwargs["api_key"] != "sk-real-openai-key"
        assert call_kwargs["api_key"] is None

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-real-openai-key",
            "OPENAI_API_BASE": "https://api.openai-compatible.example/v1",
        },
        clear=True,
    )
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    @patch("world_understanding.tools.nlp.chat_tool.create_chat_model")
    def test_chat_openai_rejects_hosted_key_with_legacy_env_base_url(
        self, mock_create, mock_generate
    ):
        """Legacy ``OPENAI_API_BASE`` env var triggers the same protection."""
        mock_generate.return_value = {"response": "ok"}
        inputs = ChatInput(prompt="Hi", backend="openai")

        chat_tool(inputs)

        assert mock_create.call_args.kwargs["api_key"] is None

    @patch.dict(os.environ, {}, clear=True)
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    @patch("world_understanding.tools.nlp.chat_tool.create_chat_model")
    def test_chat_openai_explicit_api_key_paired_with_base_url_is_honored(
        self, mock_create, mock_generate
    ):
        """A caller pointing at a custom OpenAI-compatible endpoint must be
        able to pair an explicit endpoint-scoped ``api_key`` with the
        ``base_url`` via the tool input. Without the ``base_url`` field on
        the tool, the explicit key would be rejected because the resolver
        cannot validate the pairing."""
        mock_generate.return_value = {"response": "ok"}
        inputs = ChatInput(
            prompt="Hi",
            backend="openai",
            api_key="sk-explicit-endpoint-key",
            base_url="https://api.openai-compatible.example/v1",
        )

        chat_tool(inputs)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-explicit-endpoint-key"
        assert call_kwargs["base_url"] == "https://api.openai-compatible.example/v1"

    @patch("world_understanding.tools.nlp.chat_tool.create_chat_model")
    @patch("world_understanding.tools.nlp.chat_tool.generate_chat_response")
    def test_chat_with_gemini_api_key_alias(
        self, mock_generate, mock_create_chat_model, monkeypatch
    ):
        """Test that chat tool accepts GEMINI_API_KEY without GOOGLE_API_KEY."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        mock_chat_model = object()
        mock_create_chat_model.return_value = mock_chat_model
        mock_generate.return_value = {"response": "Gemini response"}

        inputs = ChatInput(prompt="Test with Gemini", backend="gemini")
        output = chat_tool(inputs)

        assert output.response == "Gemini response"
        assert output.backend_used == "gemini"
        mock_create_chat_model.assert_called_once()
        call_kwargs = mock_create_chat_model.call_args[1]
        assert call_kwargs["api_key"] == "gemini-key"
