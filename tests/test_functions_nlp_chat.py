# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the chat generation function."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from world_understanding.functions.nlp.chat import (
    agenerate_chat_response,
    generate_chat_response,
)


class TestGenerateChatResponse:
    """Tests for generate_chat_response function."""

    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock chat model."""
        mock_model = Mock()
        return mock_model

    def test_successful_chat_response(self, mock_chat_model):
        """Test successful chat response generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "This is a test response"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model,
            prompt="Hello, how are you?",
            system_prompt="You are a helpful assistant.",
        )

        assert "response" in result
        assert result["response"] == "This is a test response"
        assert "error" not in result

        # Verify the model was called correctly
        mock_chat_model.invoke.assert_called_once()
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert len(call_args) == 2  # System and Human messages

    def test_default_system_prompt(self, mock_chat_model):
        """Test using default system prompt."""
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt="Test prompt"
        )

        assert result["response"] == "Test response"

        # Check that default system prompt was used
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content == "You are a helpful AI assistant."

    def test_custom_system_prompt(self, mock_chat_model):
        """Test with custom system prompt."""
        mock_response = Mock()
        mock_response.content = "Custom response"
        mock_chat_model.invoke.return_value = mock_response

        custom_prompt = "You are a Python expert."
        result = generate_chat_response(
            chat_model=mock_chat_model,
            prompt="Explain decorators",
            system_prompt=custom_prompt,
        )

        assert result["response"] == "Custom response"

        # Verify custom system prompt was used
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content == custom_prompt

    @patch("world_understanding.functions.nlp.chat.SystemMessage")
    @patch("world_understanding.functions.nlp.chat.HumanMessage")
    def test_message_construction(
        self, mock_human_msg, mock_system_msg, mock_chat_model
    ):
        """Test proper message construction."""
        # Mock the message classes
        mock_system_instance = Mock()
        mock_human_instance = Mock()
        mock_system_msg.return_value = mock_system_instance
        mock_human_msg.return_value = mock_human_instance

        mock_response = Mock()
        mock_response.content = "Response"
        mock_chat_model.invoke.return_value = mock_response

        generate_chat_response(
            chat_model=mock_chat_model,
            prompt="User question",
            system_prompt="System instructions",
        )

        # Verify messages were constructed correctly
        mock_system_msg.assert_called_once_with(content="System instructions")
        mock_human_msg.assert_called_once_with(content="User question")

        # Verify they were passed to the model in correct order
        mock_chat_model.invoke.assert_called_once_with(
            [mock_system_instance, mock_human_instance]
        )

    def test_import_error_handling(self, mock_chat_model):
        """Test handling of import errors (no longer applicable since imports are at module level)."""
        # This test is no longer relevant since we removed the try-except import block
        # The import now happens at module level, so ImportError would prevent the module from loading
        # We'll test that the function works normally instead
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")
        assert "response" in result
        assert result["response"] == "Test response"

    def test_model_invocation_error(self, mock_chat_model):
        """Test handling of model invocation errors."""
        # Make the model raise an exception
        mock_chat_model.invoke.side_effect = Exception("API Error")

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt="Test prompt"
        )

        assert "error" in result
        assert "Failed to generate response" in result["error"]
        assert "API Error" in result["error"]
        assert "response" not in result

    def test_empty_prompt(self, mock_chat_model):
        """Test with empty prompt."""
        mock_response = Mock()
        mock_response.content = "Response to empty prompt"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="")

        assert result["response"] == "Response to empty prompt"

    def test_very_long_prompt(self, mock_chat_model):
        """Test with very long prompt."""
        long_prompt = "Test " * 1000  # 5000 characters

        mock_response = Mock()
        mock_response.content = "Response to long prompt"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt=long_prompt)

        assert result["response"] == "Response to long prompt"

        # Verify the full prompt was passed
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[1].content == long_prompt

    def test_unicode_handling(self, mock_chat_model):
        """Test handling of unicode characters."""
        unicode_prompt = "Hello 世界! How are you? 🌍"
        unicode_response = "你好！I'm doing great! 😊"

        mock_response = Mock()
        mock_response.content = unicode_response
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt=unicode_prompt
        )

        assert result["response"] == unicode_response

    def test_none_chat_model(self):
        """Test with None chat model."""
        result = generate_chat_response(chat_model=None, prompt="Test")

        assert "error" in result
        assert "Failed to generate response" in result["error"]

    def test_attribute_error_on_response(self, mock_chat_model):
        """Test handling when response doesn't have content attribute."""
        # Return something without .content attribute
        mock_chat_model.invoke.return_value = "Plain string response"

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert "error" in result
        assert "Failed to generate response" in result["error"]

    def test_multiline_prompt_and_response(self, mock_chat_model):
        """Test with multiline prompts and responses."""
        multiline_prompt = """This is a multiline prompt.
        It has multiple lines.
        And some indentation."""

        multiline_response = """This is a multiline response.
        It also has multiple lines.
        With preserved formatting."""

        mock_response = Mock()
        mock_response.content = multiline_response
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt=multiline_prompt
        )

        assert result["response"] == multiline_response

    def test_special_characters_in_prompt(self, mock_chat_model):
        """Test handling of special characters."""
        special_prompt = "What about <html> tags & special chars?"

        mock_response = Mock()
        mock_response.content = "Response with <tags> & chars"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt=special_prompt
        )

        assert result["response"] == "Response with <tags> & chars"

    def test_system_message_import_error(self, mock_chat_model):
        """Test when imports fail (no longer applicable since imports are at module level)."""
        # This test is no longer relevant since we removed the try-except import block
        # The import now happens at module level, so ImportError would prevent the module from loading
        # We'll test that the function works normally instead
        mock_response = Mock()
        mock_response.content = "System message test"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")
        assert "response" in result
        assert result["response"] == "System message test"

    def test_model_timeout_simulation(self, mock_chat_model):
        """Test handling of timeout-like errors."""
        mock_chat_model.invoke.side_effect = TimeoutError("Request timed out")

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt="Test prompt"
        )

        assert "error" in result
        assert "Failed to generate response" in result["error"]
        assert "Request timed out" in result["error"]

    def test_empty_response_content(self, mock_chat_model):
        """Test handling of empty response content."""
        mock_response = Mock()
        mock_response.content = ""
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(
            chat_model=mock_chat_model, prompt="Give me nothing"
        )

        assert "response" in result
        assert result["response"] == ""
        assert "error" not in result

    def test_whitespace_only_response(self, mock_chat_model):
        """Test handling of whitespace-only responses."""
        mock_response = Mock()
        mock_response.content = "   \n\t   "
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert result["response"] == "   \n\t   "

    def test_response_type_validation(self, mock_chat_model):
        """Test that response is always a dictionary."""
        mock_response = Mock()
        mock_response.content = "Test"
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())

    def test_concurrent_calls_simulation(self, mock_chat_model):
        """Test that function handles concurrent calls properly."""
        responses = ["Response 1", "Response 2", "Response 3"]
        mock_responses = []

        for resp in responses:
            mock_resp = Mock()
            mock_resp.content = resp
            mock_responses.append(mock_resp)

        mock_chat_model.invoke.side_effect = mock_responses

        results = []
        for i in range(3):
            result = generate_chat_response(
                chat_model=mock_chat_model, prompt=f"Prompt {i}"
            )
            results.append(result)

        # Verify each call got its own response
        for i, result in enumerate(results):
            assert result["response"] == responses[i]

    def test_list_content_response_extracts_text_parts(self, mock_chat_model):
        mock_response = Mock()
        mock_response.content = [
            {"type": "thinking", "text": "hidden"},
            {"type": "text", "text": "visible 1"},
            {"type": "text", "text": "visible 2"},
        ]
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert result["response"] == "visible 1\nvisible 2"

    def test_list_content_without_text_parts_falls_back_to_string(
        self, mock_chat_model
    ):
        mock_response = Mock()
        mock_response.content = [{"type": "thinking", "text": "hidden"}]
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert result["response"] == "[{'type': 'thinking', 'text': 'hidden'}]"

    def test_non_string_response_content_falls_back_to_string(self, mock_chat_model):
        mock_response = Mock()
        mock_response.content = {"structured": "value"}
        mock_chat_model.invoke.return_value = mock_response

        result = generate_chat_response(chat_model=mock_chat_model, prompt="Test")

        assert result["response"] == "{'structured': 'value'}"


@pytest.mark.asyncio
async def test_agenerate_chat_response_success_and_error() -> None:
    chat_model = Mock()
    response = Mock()
    response.content = [{"type": "text", "text": "async hello"}]
    chat_model.ainvoke = AsyncMock(return_value=response)

    result = await agenerate_chat_response(chat_model, "hi", "system")

    assert result == {"response": "async hello"}
    chat_model.ainvoke.assert_called_once()

    failing_model = Mock()
    failing_model.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    result = await agenerate_chat_response(failing_model, "hi")
    assert result["error"] == "Failed to generate response: boom"
