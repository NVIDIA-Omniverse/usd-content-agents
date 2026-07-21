# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SimpleAgent."""

import pytest

from world_understanding.agentic.agents.simple import SimpleAgent
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool
from world_understanding.utils.object_store import InMemoryObjectStore


# Define test tools
class SimpleToolInput(ToolInput):
    value: str


class SimpleToolOutput(ToolOutput):
    result: str


@pytest.fixture
def test_tools():
    """Register test tools for simple agent tests."""
    from world_understanding.tools.base import _TOOL_REGISTRY

    # Clear any existing test tools
    test_tool_names = ["test_color_analyzer", "test_chat", "test_processor"]
    for name in test_tool_names:
        if name in _TOOL_REGISTRY:
            del _TOOL_REGISTRY[name]

    # Register test tools
    @register_tool(
        name="test_color_analyzer",
        version="1.0.0",
        description="Analyze colors in an image",
        input_model=SimpleToolInput,
        output_model=SimpleToolOutput,
        tags=["cv", "color"],
    )
    def color_analyzer_tool(inputs: SimpleToolInput) -> SimpleToolOutput:
        return SimpleToolOutput(result=f"Colors analyzed for {inputs.value}")

    @register_tool(
        name="test_chat",
        version="1.0.0",
        description="Chat with an AI model",
        input_model=SimpleToolInput,
        output_model=SimpleToolOutput,
        tags=["nlp", "chat"],
    )
    def chat_tool(inputs: SimpleToolInput) -> SimpleToolOutput:
        return SimpleToolOutput(result=f"Chat response to: {inputs.value}")

    @register_tool(
        name="test_processor",
        version="1.0.0",
        description="Process data",
        input_model=SimpleToolInput,
        output_model=SimpleToolOutput,
        tags=["processing"],
    )
    def processor_tool(inputs: SimpleToolInput) -> SimpleToolOutput:
        return SimpleToolOutput(result=f"Processed: {inputs.value}")

    return _TOOL_REGISTRY


@pytest.fixture
def simple_agent(test_tools):
    """Create a simple agent with test tools."""
    return SimpleAgent(
        tools=test_tools,
        name="test_simple_agent",
        description="Test simple agent",
    )


@pytest.fixture
def object_store():
    """Create an in-memory object store."""
    return InMemoryObjectStore()


def test_simple_agent_initialization(simple_agent):
    """Test simple agent initialization."""
    assert simple_agent.name == "test_simple_agent"
    assert simple_agent.description == "Test simple agent"
    assert simple_agent.tools is not None
    assert len(simple_agent.tools) >= 3  # At least our test tools


def test_simple_agent_can_handle(simple_agent):
    """Test simple agent can_handle method."""
    # SimpleAgent should handle tool execution tasks
    assert simple_agent.can_handle("test_color_analyzer") is True
    assert simple_agent.can_handle("execute: test_chat") is True
    assert simple_agent.can_handle("run test_processor") is True
    # Should also handle general tasks
    assert simple_agent.can_handle("analyze this") is True


def test_run_single_tool_success(simple_agent, object_store):
    """Test running a single tool successfully."""
    # Run agent with tool name as task
    # SimpleAgent expects params in {tool_name}_params key
    context = {"test_color_analyzer_params": {"value": "test.jpg"}}
    result = simple_agent.run(
        task="test_color_analyzer",
        context=context,
        object_store=object_store,
    )

    # Verify results - SimpleAgent returns context with {tool_name}_success and {tool_name}_output
    assert result["test_color_analyzer_success"] is True
    assert "test_color_analyzer_output" in result
    assert (
        result["test_color_analyzer_output"]["result"] == "Colors analyzed for test.jpg"
    )

    # Check object store
    assert object_store.exists("test_color_analyzer_output")


def test_run_with_task_params(simple_agent, object_store):
    """Test running with task:params_key format."""
    # Put params in a custom key
    context = {"my_params": {"value": "Hello"}}
    result = simple_agent.run(
        task="test_chat:my_params",
        context=context,
        object_store=object_store,
    )

    # Verify results
    assert result["test_chat_success"] is True
    assert result["test_chat_output"]["result"] == "Chat response to: Hello"


def test_run_tool_not_found(simple_agent, object_store):
    """Test running with non-existent tool."""
    result = simple_agent.run(
        task="non_existent_tool",
        context={},
        object_store=object_store,
    )

    # Should have error in context
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_run_without_context_uses_empty_context(simple_agent):
    """A missing context should be initialized before tool lookup."""
    result = simple_agent.run(task="non_existent_tool")

    assert result["error"] == "Tool 'non_existent_tool' not found"


def test_run_with_invalid_inputs(simple_agent, object_store):
    """Test running with invalid inputs."""
    # Run without required input
    context = {"test_color_analyzer_params": {}}  # Missing 'value' field
    result = simple_agent.run(
        task="test_color_analyzer",
        context=context,
        object_store=object_store,
    )

    # Should fail with error
    assert result["test_color_analyzer_success"] is False
    assert "test_color_analyzer_error" in result


def test_run_stores_results(simple_agent, object_store):
    """Test that results are stored in object store."""
    # Run multiple tools
    context1 = {"test_color_analyzer_params": {"value": "image1.jpg"}}
    simple_agent.run(
        task="test_color_analyzer",
        context=context1,
        object_store=object_store,
    )

    context2 = {"test_chat_params": {"value": "Hello"}}
    simple_agent.run(
        task="test_chat",
        context=context2,
        object_store=object_store,
    )

    # Check both results are stored
    assert object_store.exists("test_color_analyzer_output")
    assert object_store.exists("test_chat_output")

    # Verify stored values
    stored1 = object_store.get("test_color_analyzer_output")
    assert stored1["result"] == "Colors analyzed for image1.jpg"

    stored2 = object_store.get("test_chat_output")
    assert stored2["result"] == "Chat response to: Hello"


def test_run_with_context_merging(simple_agent, object_store):
    """Test that context is properly merged with inputs."""
    # Initial context with params and extra data
    initial_context = {
        "test_color_analyzer_params": {"value": "test.jpg"},
        "extra_data": "extra_value",
    }

    result = simple_agent.run(
        task="test_color_analyzer",
        context=initial_context,
        object_store=object_store,
    )

    # Should succeed with merged context
    assert result["test_color_analyzer_success"] is True
    assert (
        result["test_color_analyzer_output"]["result"] == "Colors analyzed for test.jpg"
    )
    # Extra data should still be in context
    assert result["extra_data"] == "extra_value"


def test_run_updates_context(simple_agent, object_store):
    """Test that context is updated with results."""
    context = {"test_processor_params": {"value": "test"}}

    result = simple_agent.run(
        task="test_processor",
        context=context,
        object_store=object_store,
    )

    # Result should contain the tool output
    assert "test_processor_output" in result
    assert result["test_processor_output"]["result"] == "Processed: test"

    # Context should have success flag
    assert result["test_processor_success"] is True


def test_error_handling(simple_agent, object_store):
    """Test error handling in various scenarios."""
    # Test with invalid input type (should cause validation error)
    context = {"test_color_analyzer_params": {"value": 123}}  # Should be string

    result = simple_agent.run(
        task="test_color_analyzer",
        context=context,
        object_store=object_store,
    )

    # Should have error
    assert result["test_color_analyzer_success"] is False
    assert "test_color_analyzer_error" in result


def test_multiple_sequential_runs(simple_agent, object_store):
    """Test running multiple tools in sequence."""
    # Run first tool
    context1 = {"test_color_analyzer_params": {"value": "image.jpg"}}
    result1 = simple_agent.run(
        task="test_color_analyzer",
        context=context1,
        object_store=object_store,
    )

    # Use output from first tool as input to second
    chat_input = result1["test_color_analyzer_output"]["result"]
    context2 = {"test_chat_params": {"value": chat_input}}
    result2 = simple_agent.run(
        task="test_chat",
        context=context2,
        object_store=object_store,
    )

    # Use output from second tool as input to third
    processor_input = result2["test_chat_output"]["result"]
    context3 = {"test_processor_params": {"value": processor_input}}
    result3 = simple_agent.run(
        task="test_processor",
        context=context3,
        object_store=object_store,
    )

    # All should succeed
    assert result1["test_color_analyzer_success"] is True
    assert result2["test_chat_success"] is True
    assert result3["test_processor_success"] is True

    # Final result should be processed chat response
    expected = "Processed: Chat response to: Colors analyzed for image.jpg"
    assert result3["test_processor_output"]["result"] == expected


def test_default_params_key(simple_agent, object_store):
    """Test that default params key is {tool_name}_params."""
    # When no colon in task, it should look for {tool_name}_params
    context = {
        "test_processor_params": {"value": "data"},
        "other_params": {"value": "other"},
    }

    result = simple_agent.run(
        task="test_processor",
        context=context,
        object_store=object_store,
    )

    # Should use test_processor_params, not other_params
    assert result["test_processor_success"] is True
    assert result["test_processor_output"]["result"] == "Processed: data"


def test_custom_params_key(simple_agent, object_store):
    """Test using custom params key with colon syntax."""
    context = {
        "custom_key": {"value": "custom_data"},
        "test_processor_params": {"value": "default_data"},
    }

    result = simple_agent.run(
        task="test_processor:custom_key",
        context=context,
        object_store=object_store,
    )

    # Should use custom_key, not default test_processor_params
    assert result["test_processor_success"] is True
    assert result["test_processor_output"]["result"] == "Processed: custom_data"
