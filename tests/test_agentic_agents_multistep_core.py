# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.agentic.agents import multistep


class _Input:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Output:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self.data)


class _Tool:
    def __init__(self, data: dict[str, Any] | Exception) -> None:
        self.data = data
        self.spec = SimpleNamespace(input_model=_Input)
        self.inputs: list[_Input] = []

    def run(self, input_obj: _Input) -> _Output:
        self.inputs.append(input_obj)
        if isinstance(self.data, Exception):
            raise self.data
        return _Output(self.data)

    async def arun(self, input_obj: _Input) -> _Output:
        self.inputs.append(input_obj)
        if isinstance(self.data, Exception):
            raise self.data
        return _Output(self.data)


class _Store:
    def __init__(self) -> None:
        self.values = {"token": "store-value"}
        self.set_calls: list[tuple[str, Any]] = []

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.set_calls.append((key, value))
        self.values[key] = value


def test_init_uses_global_tool_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = {"tool": _Tool({"ok": True})}
    monkeypatch.setattr(multistep, "get_tool_registry", lambda: tools)

    agent = multistep.MultiStepAgent(pipeline=[{"tool": "tool"}])

    assert agent.name == "MultiStepAgent"
    assert agent.description == "Multi-step pipeline execution agent"
    assert agent.tools is tools
    assert agent.pipeline == [{"tool": "tool"}]


def test_run_routes_workflow_and_single_step(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = multistep.MultiStepAgent(tools={})
    monkeypatch.setattr(
        agent,
        "execute_workflow",
        lambda context, object_store=None: {"workflow": context},
    )
    monkeypatch.setattr(
        agent,
        "execute_single_step",
        lambda task, context, object_store=None: {"single": task},
    )

    assert agent.run("execute_workflow") == {"workflow": {}}
    assert agent.run("tool") == {"single": "tool"}


def test_execute_workflow_success_resolves_references_and_confidence() -> None:
    tool_a = _Tool({"value": {"nested": 4}, "confidence": {"a": 0.8}})
    tool_b = _Tool({"result": "done", "confidence": 0.6})
    store = _Store()
    agent = multistep.MultiStepAgent(tools={"a": tool_a, "b": tool_b})
    context = {
        "source": {"id": 7},
        "pipeline": [
            {
                "name": "first",
                "tool": "a",
                "params": {
                    "literal": "x",
                    "from_context": "${context.source.id}",
                    "from_store": "${store.token}",
                    "unknown_ref": "${unknown.value}",
                },
            },
            {
                "name": "second",
                "tool": "b",
                "params": {
                    "from_output": "${output.first.value.nested}",
                    "missing_output": "${output.nope.value}",
                },
            },
        ],
    }

    result = agent.execute_workflow(context, store)

    assert result["completed"] is True
    assert result["avg_confidence"] == 0.7
    assert result["confidence_scores"] == {"a": 0.8, "second": 0.6}
    assert result["pipeline_outputs"]["first"]["value"]["nested"] == 4
    assert tool_a.inputs[0].kwargs == {
        "literal": "x",
        "from_context": 7,
        "from_store": "store-value",
        "unknown_ref": "${unknown.value}",
    }
    assert tool_b.inputs[0].kwargs == {"from_output": 4, "missing_output": None}
    assert store.set_calls == [
        ("step_first_output", {"value": {"nested": 4}, "confidence": {"a": 0.8}}),
        ("step_second_output", {"result": "done", "confidence": 0.6}),
    ]


def test_execute_workflow_errors() -> None:
    agent = multistep.MultiStepAgent(tools={"bad": _Tool(RuntimeError("boom"))})

    assert agent.execute_workflow({}) == {
        "error": "No pipeline defined",
        "completed": False,
    }
    missing = agent.execute_workflow({"pipeline": [{"name": "x", "tool": "missing"}]})
    assert missing == {
        "pipeline": [{"name": "x", "tool": "missing"}],
        "error": "Tool 'missing' not found",
        "completed": False,
    }

    failed = agent.execute_workflow({"pipeline": [{"name": "explode", "tool": "bad"}]})
    assert failed["completed"] is False
    assert failed["error"] == "Step 'explode' failed: boom"


def test_execute_single_step_success_missing_and_failure() -> None:
    store = _Store()
    ok_tool = _Tool({"answer": 42})
    bad_tool = _Tool(RuntimeError("bad input"))
    agent = multistep.MultiStepAgent(tools={"ok": ok_tool, "bad": bad_tool})

    context = agent.execute_single_step("missing", {})
    assert context["error"] == "Tool 'missing' not found"

    context = agent.execute_single_step("ok", {"ok_params": {"x": 1}}, store)
    assert context["ok_success"] is True
    assert context["ok_output"] == {"answer": 42}
    assert store.set_calls[-1] == ("ok_output", {"answer": 42})

    context = agent.execute_single_step("bad", {"bad_params": {"x": 1}})
    assert context["bad_success"] is False
    assert context["bad_error"] == "bad input"


def test_resolve_params_and_get_nested_without_store() -> None:
    agent = multistep.MultiStepAgent(tools={})

    assert agent._resolve_params(
        {
            "plain": 1,
            "bad_context": "${context.missing.value}",
            "store_without_store": "${store.token}",
        },
        {"present": {"value": 2}},
        {},
        None,
    ) == {
        "plain": 1,
        "bad_context": None,
        "store_without_store": "${store.token}",
    }
    assert agent._get_nested({"a": {"b": 3}}, ["a", "b"]) == 3
    assert agent._get_nested({"a": None}, ["a", "b"]) is None


def test_async_routes_and_workflow_success() -> None:
    async def _run() -> None:
        tool = _Tool({"value": 2, "confidence": 0.5})
        dict_conf_tool = _Tool({"value": 3, "confidence": {"dict_step": 0.9}})
        store = _Store()
        agent = multistep.MultiStepAgent(
            tools={"async_tool": tool, "dict_tool": dict_conf_tool},
            pipeline=[
                {"name": "step", "tool": "async_tool"},
                {"name": "dict", "tool": "dict_tool"},
            ],
        )

        workflow = await agent.arun("execute_workflow", object_store=store)
        assert workflow["completed"] is True
        assert workflow["avg_confidence"] == 0.7
        assert workflow["confidence_scores"] == {"step": 0.5, "dict_step": 0.9}
        assert store.set_calls[:2] == [
            ("step_step_output", {"value": 2, "confidence": 0.5}),
            ("step_dict_output", {"value": 3, "confidence": {"dict_step": 0.9}}),
        ]

        single = await agent.arun(
            "async_tool",
            {"async_tool_params": {"x": 1}},
            object_store=store,
        )
        assert single["async_tool_success"] is True
        assert single["async_tool_output"] == {"value": 2, "confidence": 0.5}

    asyncio.run(_run())


def test_async_workflow_and_single_step_errors() -> None:
    async def _run() -> None:
        agent = multistep.MultiStepAgent(
            tools={"bad": _Tool(RuntimeError("async boom"))}
        )

        assert await agent.aexecute_workflow({}) == {
            "error": "No pipeline defined",
            "completed": False,
        }
        missing = await agent.aexecute_workflow({"pipeline": [{"tool": "missing"}]})
        assert missing["error"] == "Tool 'missing' not found"
        failed = await agent.aexecute_workflow(
            {"pipeline": [{"name": "bad_step", "tool": "bad"}]}
        )
        assert failed["completed"] is False
        assert failed["error"] == "Step 'bad_step' failed: async boom"

        missing_single = await agent.aexecute_single_step("missing", {})
        assert missing_single["error"] == "Tool 'missing' not found"
        failed_single = await agent.aexecute_single_step(
            "bad", {"bad_params": {"x": 1}}
        )
        assert failed_single["bad_success"] is False
        assert failed_single["bad_error"] == "async boom"

    asyncio.run(_run())
