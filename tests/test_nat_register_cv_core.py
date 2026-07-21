# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for NAT CV registration without requiring the NAT package."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


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
    modules = {
        "nat": types.ModuleType("nat"),
        "nat.builder": types.ModuleType("nat.builder"),
        "nat.builder.builder": types.ModuleType("nat.builder.builder"),
        "nat.builder.function_info": types.ModuleType("nat.builder.function_info"),
        "nat.cli": types.ModuleType("nat.cli"),
        "nat.cli.register_workflow": types.ModuleType("nat.cli.register_workflow"),
        "nat.data_models": types.ModuleType("nat.data_models"),
        "nat.data_models.function": types.ModuleType("nat.data_models.function"),
    }
    modules["nat.builder.builder"].Builder = type("Builder", (), {})
    modules["nat.builder.function_info"].FunctionInfo = FakeFunctionInfo
    modules["nat.cli.register_workflow"].register_function = lambda **_kwargs: (
        lambda fn: fn
    )
    modules["nat.data_models.function"].FunctionBaseConfig = FakeFunctionBaseConfig
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


async def _function_info(module: Any, function_name: str, config_name: str) -> Any:
    generator = getattr(module, function_name)(getattr(module, config_name)(), object())
    return await anext(generator)


def _load_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_fake_nat(monkeypatch)
    sys.modules.pop("world_understanding.nat.register_cv", None)
    module = importlib.import_module("world_understanding.nat.register_cv")
    return importlib.reload(module)


def test_nat_register_cv_dominant_colors_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module(monkeypatch)
    assert module.DominantColorsToolConfig.name == "get_dominant_colors"

    import world_understanding.functions.cv.get_dominant_colors as colors_module

    monkeypatch.setattr(
        colors_module,
        "get_dominant_colors",
        lambda **_kwargs: {
            "dominant_colors": [
                {"hex": "#ff0000", "rgb": [255, 0, 0], "percentage": 0.25}
            ],
            "average_brightness": 10.0,
            "color_diversity": 0.5,
            "n_clusters": 1,
        },
    )
    info = asyncio.run(
        _function_info(module, "get_dominant_colors", "DominantColorsToolConfig")
    )
    payload = json.loads(asyncio.run(info.fn("image.png", n_colors=1)))
    assert payload["dominant_colors"][0]["percentage"] == "25.0%"
    assert payload["average_brightness"] == "10.0/255"

    for exc, prefix in [
        (FileNotFoundError("missing"), "Error:"),
        (ValueError("bad"), "Invalid parameter:"),
        (RuntimeError("boom"), "Failed to analyze image:"),
    ]:
        monkeypatch.setattr(
            colors_module,
            "get_dominant_colors",
            lambda _exc=exc, **_kwargs: (_ for _ in ()).throw(_exc),
        )
        info = asyncio.run(
            _function_info(module, "get_dominant_colors", "DominantColorsToolConfig")
        )
        assert asyncio.run(info.fn("image.png")).startswith(prefix)


def test_nat_register_cv_find_similar_color_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(monkeypatch)
    assert module.FindSimilarColorToolConfig.name == "find_similar_color"

    import world_understanding.functions.cv.find_similar_color as similar_module

    monkeypatch.setattr(
        similar_module,
        "find_similar_color",
        lambda **_kwargs: {
            "contains_color": True,
            "matching_percentage": 12.345,
            "pixel_count": 4,
            "total_pixels": 10,
            "target_color_rgb": [1, 2, 3],
            "target_color_hex": "#010203",
            "closest_colors": [{"hex": "#000000", "rgb": [0, 0, 0], "distance": 1.25}],
        },
    )
    info = asyncio.run(
        _function_info(module, "find_similar_color", "FindSimilarColorToolConfig")
    )
    payload = json.loads(asyncio.run(info.fn("image.png", 1, 2, 3, tolerance=7)))
    assert payload["matching_percentage"] == "12.35%"
    assert payload["closest_colors"][0]["distance"] == "1.2"

    for exc, prefix in [
        (ValueError("bad"), "Invalid parameter:"),
        (FileNotFoundError("missing"), "Error:"),
        (RuntimeError("boom"), "Failed to match color:"),
    ]:
        monkeypatch.setattr(
            similar_module,
            "find_similar_color",
            lambda _exc=exc, **_kwargs: (_ for _ in ()).throw(_exc),
        )
        info = asyncio.run(
            _function_info(module, "find_similar_color", "FindSimilarColorToolConfig")
        )
        assert asyncio.run(info.fn("image.png", 1, 2, 3)).startswith(prefix)


def test_nat_register_cv_vlm_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(monkeypatch)
    assert module.VLMToolConfig.name == "vlm"

    image_path = tmp_path / "gray.png"
    Image.new("L", (1, 1), 128).save(image_path)

    import world_understanding.functions.cv.vlm as vlm_module
    import world_understanding.functions.models.vision_language_models as models_module

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    info = asyncio.run(_function_info(module, "vlm", "VLMToolConfig"))
    assert asyncio.run(info.fn(str(image_path), "describe")).startswith(
        "Error: NVIDIA_API_KEY"
    )

    create_calls: list[dict[str, Any]] = []
    monkeypatch.setenv("NVIDIA_API_KEY", "key")
    monkeypatch.setattr(
        models_module,
        "create_vlm",
        lambda **kwargs: create_calls.append(kwargs) or "vlm-model",
    )
    monkeypatch.setattr(
        vlm_module,
        "generate_vlm_response",
        lambda **_kwargs: {"response": "caption"},
    )
    info = asyncio.run(_function_info(module, "vlm", "VLMToolConfig"))
    assert asyncio.run(info.fn(str(image_path), "describe")) == "caption"
    assert create_calls[-1]["model"] == "qwen/qwen3.5-397b-a17b"

    monkeypatch.setattr(
        models_module,
        "create_vlm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("factory")),
    )
    info = asyncio.run(_function_info(module, "vlm", "VLMToolConfig"))
    assert asyncio.run(info.fn(str(image_path), "describe")).startswith(
        "Error creating VLM:"
    )

    monkeypatch.setattr(models_module, "create_vlm", lambda **_kwargs: "vlm-model")
    monkeypatch.setattr(
        vlm_module,
        "generate_vlm_response",
        lambda **_kwargs: {"error": "bad generation"},
    )
    info = asyncio.run(_function_info(module, "vlm", "VLMToolConfig"))
    assert asyncio.run(info.fn(str(image_path), "describe")).startswith("Error:")

    info = asyncio.run(_function_info(module, "vlm", "VLMToolConfig"))
    assert asyncio.run(info.fn(str(tmp_path / "missing.png"), "describe")).startswith(
        "Error: Image file not found:"
    )

    bad_file = tmp_path / "not-image.txt"
    bad_file.write_text("not an image", encoding="utf-8")
    assert asyncio.run(info.fn(str(bad_file), "describe")).startswith(
        "Failed to analyze image:"
    )
