# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for NAT NLP registration without requiring the NAT package."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from typing import Any

import pytest


class FakeFunctionBaseConfig:
    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = name


class FakeFunctionInfo:
    def __init__(self, fn: Any, description: str) -> None:
        self.fn = fn
        self.description = description

    @classmethod
    def from_fn(cls, fn: Any, description: str) -> FakeFunctionInfo:
        return cls(fn, description)


def _install_fake_nat(monkeypatch: pytest.MonkeyPatch) -> None:
    nat = types.ModuleType("nat")
    builder_pkg = types.ModuleType("nat.builder")
    builder_mod = types.ModuleType("nat.builder.builder")
    builder_mod.Builder = type("Builder", (), {})
    function_info_mod = types.ModuleType("nat.builder.function_info")
    function_info_mod.FunctionInfo = FakeFunctionInfo
    cli_pkg = types.ModuleType("nat.cli")
    register_workflow_mod = types.ModuleType("nat.cli.register_workflow")
    register_workflow_mod.register_function = lambda **_kwargs: (lambda fn: fn)
    data_models_pkg = types.ModuleType("nat.data_models")
    function_mod = types.ModuleType("nat.data_models.function")
    function_mod.FunctionBaseConfig = FakeFunctionBaseConfig

    for name, module in {
        "nat": nat,
        "nat.builder": builder_pkg,
        "nat.builder.builder": builder_mod,
        "nat.builder.function_info": function_info_mod,
        "nat.cli": cli_pkg,
        "nat.cli.register_workflow": register_workflow_mod,
        "nat.data_models": data_models_pkg,
        "nat.data_models.function": function_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


async def _registered_chat_fn(module: Any) -> Any:
    generator = module.chat(module.ChatToolConfig(), object())
    return await anext(generator)


def test_nat_register_nlp_chat_success_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_nat(monkeypatch)
    sys.modules.pop("world_understanding.nat.register_nlp", None)
    module = importlib.import_module("world_understanding.nat.register_nlp")
    module = importlib.reload(module)

    assert module.ChatToolConfig.name == "chat"

    import world_understanding.functions.models.chat_models as chat_models
    import world_understanding.functions.nlp.chat as nlp_chat

    monkeypatch.setattr(
        chat_models, "create_chat_model", lambda **_kwargs: "chat-model"
    )
    monkeypatch.setattr(
        nlp_chat,
        "generate_chat_response",
        lambda **_kwargs: {"response": "hello"},
    )
    info = asyncio.run(_registered_chat_fn(module))
    assert "Generate text responses" in info.description
    payload = json.loads(
        asyncio.run(
            info.fn(
                "hi",
                backend="echo",
                model="demo",
                system_prompt="system",
            )
        )
    )
    assert payload == {"response": "hello", "backend": "echo", "model": "demo"}

    monkeypatch.setattr(
        nlp_chat,
        "generate_chat_response",
        lambda **_kwargs: {"error": "bad response"},
    )
    info = asyncio.run(_registered_chat_fn(module))
    assert asyncio.run(info.fn("hi")).startswith("Error: bad response")

    monkeypatch.setattr(
        chat_models,
        "create_chat_model",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad backend")),
    )
    info = asyncio.run(_registered_chat_fn(module))
    assert asyncio.run(info.fn("hi")).startswith("Invalid backend or configuration")

    monkeypatch.setattr(
        chat_models,
        "create_chat_model",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    info = asyncio.run(_registered_chat_fn(module))
    assert asyncio.run(info.fn("hi")).startswith("Failed to generate response")
