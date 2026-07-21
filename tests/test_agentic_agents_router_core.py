# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage tests for router agent edge paths."""

import asyncio
from types import SimpleNamespace

import world_understanding.agentic.agents.router as router_module
from world_understanding.agentic.agents.router import RouterAgent
from world_understanding.tools.base import ToolInput, ToolOutput
from world_understanding.utils.object_store import InMemoryObjectStore


class _RouterInput(ToolInput):
    value: str = "default"
    shared: str | None = None


class _BadSchemaInput(ToolInput):
    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        raise RuntimeError("schema unavailable")


class _RouterOutput(ToolOutput):
    value: str


class _DumpLike:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _DictLike:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


class _AsyncTool:
    def __init__(self, description="tool", tags=None, input_model=_RouterInput):
        self.spec = SimpleNamespace(
            description=description,
            tags=tags or ["route"],
            input_model=input_model,
        )
        self.calls = []

    async def arun(self, inputs):
        self.calls.append(inputs)
        value = getattr(inputs, "value", "raw")
        return _RouterOutput(value=value)


class _SpecLessTool:
    def __init__(self):
        self.calls = []

    async def arun(self, inputs):
        self.calls.append(inputs)
        return {"plain": True}


def _router(tools=None, chat_model_config=None):
    return RouterAgent(
        tools=tools or {},
        chat_model_config=chat_model_config
        if chat_model_config is not None
        else {"service": "fake", "api_key": "key", "model_name": "model"},
    )


def test_analyze_task_with_llm_filters_invalid_tools_and_handles_failures(
    monkeypatch, caplog
):
    agent = _router()
    available_tools = [
        {"name": "color", "description": "Find colors", "tags": ["vision", "color"]},
        {"name": "plain", "description": "Plain tool"},
    ]

    monkeypatch.setattr(router_module, "create_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: {
            "response": 'prefix {"tools": ["color", "missing"], "reasoning": "fits"} suffix'
        },
    )

    assert agent.analyze_task_with_llm("inspect color", available_tools) == ["color"]
    assert "non-existent tool" in caplog.text

    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: {"response": "no json here"},
    )
    assert agent.analyze_task_with_llm("inspect color", available_tools) == []

    monkeypatch.setattr(
        router_module,
        "create_chat_model",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model failed")),
    )
    assert agent.analyze_task_with_llm("inspect color", available_tools) == []


def test_analyze_and_generate_inputs_parses_json_and_uses_schema_fallback(
    monkeypatch,
):
    agent = _router()
    good_spec = SimpleNamespace(input_model=_RouterInput)

    monkeypatch.setattr(router_module, "create_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: {"response": 'prefix {"value": "generated"} suffix'},
    )

    assert agent.analyze_and_generate_inputs("route it", "tool", good_spec) == {
        "value": "generated"
    }

    bad_spec = SimpleNamespace(input_model=_BadSchemaInput)
    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: {"response": "still not json"},
    )
    assert agent.analyze_and_generate_inputs("route it", "tool", bad_spec) == {}

    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )
    assert agent.analyze_and_generate_inputs("route it", "tool", good_spec) == {}


def test_select_tools_with_llm_skips_tools_without_specs_and_handles_empty(caplog):
    tool = _AsyncTool(description="Useful routing tool", tags=["useful"])
    agent = _router(tools={"useful": tool, "no_spec": _SpecLessTool()})
    seen_available_tools = []

    def fake_analyze(task, available_tools):
        seen_available_tools.extend(available_tools)
        return ["useful"]

    agent.analyze_task_with_llm = fake_analyze
    assert agent.select_tools("do useful work") == ["useful"]
    assert seen_available_tools == [
        {"name": "useful", "description": "Useful routing tool", "tags": ["useful"]}
    ]

    agent.analyze_task_with_llm = lambda task, available_tools: []
    assert agent.select_tools("do useful work") == []
    assert "returned no tools" in caplog.text


def test_execute_tool_error_is_reported_and_success_can_store_result():
    broken = _AsyncTool()

    async def fail(inputs):
        raise RuntimeError("boom")

    broken.arun = fail
    agent = _router(tools={"broken": broken, "ok": _AsyncTool()})

    assert agent.execute_tool("broken", {"value": "x"}) == {
        "success": False,
        "error": "boom",
        "tool": "broken",
    }

    store = InMemoryObjectStore()
    result = agent.execute_tool("ok", _RouterInput(value="stored"), store)
    assert result["success"] is True
    assert store.exists("ok_result")


def test_generate_answer_from_results_covers_all_serialization_paths(monkeypatch):
    agent = _router(chat_model_config={})
    assert (
        agent.generate_answer_from_results(
            "task", [{"success": False, "tool": "bad", "error": "nope"}]
        )
        == "Failed to complete the task due to errors."
    )

    agent = _router()
    captured_prompts = []

    monkeypatch.setattr(router_module, "create_chat_model", lambda **kwargs: object())

    def fake_generate(**kwargs):
        captured_prompts.append(kwargs["prompt"])
        return {"response": "summarized"}

    monkeypatch.setattr(router_module, "generate_chat_response", fake_generate)

    assert (
        agent.generate_answer_from_results(
            "summarize",
            [
                {"success": True, "tool": "dump", "result": _DumpLike({"a": 1})},
                {"success": True, "tool": "dict", "result": _DictLike({"b": 2})},
                {"success": False, "tool": "fail", "error": "not today"},
            ],
        )
        == "summarized"
    )
    assert '"a": 1' in captured_prompts[-1]
    assert '"b": 2' in captured_prompts[-1]
    assert "not today" in captured_prompts[-1]

    monkeypatch.setattr(
        router_module,
        "generate_chat_response",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm down")),
    )
    assert (
        agent.generate_answer_from_results(
            "summarize", [{"success": True, "tool": "dump", "result": {"a": 1}}]
        )
        == "Task completed using dump. (LLM summary unavailable)"
    )
    assert (
        agent.generate_answer_from_results(
            "summarize", [{"success": False, "tool": "fail", "error": "bad"}]
        )
        == "Failed to complete the task due to errors"
    )


def test_arun_skips_missing_tools_and_accepts_generated_input_shapes():
    tools = {
        "dump": _AsyncTool(),
        "dict": _AsyncTool(),
        "other": _AsyncTool(),
        "plain": _SpecLessTool(),
    }
    agent = _router(tools=tools)
    agent.select_tools = lambda task: ["missing", "dump", "dict", "other", "plain"]
    generated_inputs = iter(
        [
            _DumpLike({"value": "from-dump"}),
            {"value": "from-dict"},
            object(),
        ]
    )
    agent.analyze_and_generate_inputs = lambda task, tool_name, spec: next(
        generated_inputs
    )
    agent.generate_answer_from_results = lambda task, results: "done"

    result = asyncio.run(agent.arun("route everything", {"shared": "ctx"}))

    assert result["success"] is True
    assert result["selected_tools"] == ["missing", "dump", "dict", "other", "plain"]
    assert [item["tool"] for item in result["tool_results"]] == [
        "dump",
        "dict",
        "other",
        "plain",
    ]
    assert tools["dump"].calls[0].value == "from-dump"
    assert tools["dict"].calls[0].value == "from-dict"
    assert tools["other"].calls[0].value == "default"
    assert tools["plain"].calls[0]["shared"] == "ctx"


def test_arun_uses_context_for_tools_without_chat_config():
    tool = _AsyncTool(input_model=None)
    agent = _router(tools={"plain_input": tool}, chat_model_config={})
    agent.select_tools = lambda task: ["plain_input"]
    agent.generate_answer_from_results = lambda task, results: "done"

    result = asyncio.run(agent.arun("route context", {"value": "raw"}))

    assert result["success"] is True
    assert tool.calls[0] == {"value": "raw"}
