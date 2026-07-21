# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

import texture_agent.functions.prompt_generation as prompt_generation
from texture_agent.functions.material_discovery import MaterialInfo


def _make_material(name: str, **overrides) -> MaterialInfo:
    defaults = {
        "prim_path": f"/Looks/{name}",
        "name": name,
        "base_color": (0.2, 0.3, 0.4),
        "base_metalness": 0.7,
        "specular_roughness": 0.2,
    }
    defaults.update(overrides)
    return MaterialInfo(**defaults)


def test_format_materials_list_renders_unknown_values() -> None:
    text = prompt_generation._format_materials_list(
        [
            _make_material(
                "Steel_Panel",
                base_metalness=None,
                specular_roughness=None,
            )
        ]
    )

    assert "Steel_Panel" in text
    assert "Base Color (RGB linear): (0.200, 0.300, 0.400)" in text
    assert "Metalness: unknown" in text
    assert "Roughness: unknown" in text


def test_generate_texture_prompts_returns_empty_for_no_materials() -> None:
    assert prompt_generation.generate_texture_prompts([], llm=object()) == {}


def test_generate_texture_prompts_uses_fallback_when_llm_raises() -> None:
    class BrokenLLM:
        def invoke(self, _messages):
            raise RuntimeError("boom")

    result = prompt_generation.generate_texture_prompts(
        materials=[_make_material("Paint_Blue")],
        llm=BrokenLLM(),
        user_prompt="weathered",
        default_opacity=0.72,
    )

    # Fallback template leads with the user's aesthetic direction and
    # applies it to the material name (updated from the older
    # "realistic <name> surface texture, <user_prompt>" wording).
    assert result == {
        "Paint_Blue": {
            "prompt": "weathered, applied to paint blue",
            "opacity": 0.72,
        }
    }


def test_fallback_prompt_without_user_prompt() -> None:
    assert (
        prompt_generation._fallback_prompt_for_material(_make_material("Brushed_Steel"))
        == "realistic brushed steel surface texture"
    )


def test_generate_texture_prompts_normalizes_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda _content, expected_keys: {
            "materials": {
                "Steel_Panel": {"prompt": "brushed steel", "opacity": 1.5},
                "Plastic_Handle": {"opacity": 0.1},
            }
        },
    )

    class FakeLLM:
        def invoke(self, _messages):
            return SimpleNamespace(content="{json}")

    result = prompt_generation.generate_texture_prompts(
        materials=[
            _make_material("Steel_Panel"),
            _make_material("Plastic_Handle"),
        ],
        llm=FakeLLM(),
        user_prompt="factory fresh",
        default_opacity=0.8,
    )

    assert result["Steel_Panel"] == {"prompt": "brushed steel", "opacity": 1.0}
    assert result["Plastic_Handle"]["prompt"] == (
        "factory fresh, applied to plastic handle"
    )
    assert result["Plastic_Handle"]["opacity"] == 0.3


def test_generate_texture_prompts_uses_fallback_when_parse_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda _content, expected_keys: None,
    )

    class FakeLLM:
        def invoke(self, _messages):
            return SimpleNamespace(content="not-json")

    result = prompt_generation.generate_texture_prompts(
        materials=[_make_material("Raw_Aluminum")],
        llm=FakeLLM(),
        user_prompt="clean",
        default_opacity=0.66,
    )

    assert result["Raw_Aluminum"]["prompt"] == ("clean, applied to raw aluminum")
    assert result["Raw_Aluminum"]["opacity"] == 0.66
