# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for generated material library task wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import material_agent.tasks.generate_material_library as generate_module
from material_agent.material_library_generation import MaterialGenerationPlan
from material_agent.tasks.generate_material_library import GenerateMaterialLibraryTask
from material_agent.tasks.generate_material_library_config import (
    GenerateMaterialLibraryConfigTask,
)


def _recipe_dict() -> dict:
    return {
        "id": "blue",
        "name": "Blue Plastic",
        "description": "blue plastic",
        "appearance_prompt": "blue plastic seamless",
    }


class _FakeVLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate_with_image_caption_pairs(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_generate_material_library_config_resolves_all_options(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "generate.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": "out",
                "usd_path": "../asset.usd",
                "material_generation_plan_path": "plans/plan.yaml",
                "material_generation_plan": {"materials": [_recipe_dict()]},
                "texture_generation": {"texture_size": 8},
                "material_authoring": {"prototype_min_score": 0.2},
                "prototype_materials_path": "prototypes/materials.yaml",
                "prototype_materials_data": {"library_path": "prototypes/lib.usda"},
                "write_material_generation_plan": False,
                "include_generation_metadata": False,
                "reference_images": "refs/front.png",
                "rendered_preview_paths": ["/abs/preview.png"],
                "composition_images": ["composition.png"],
                "identification": {"asset_type": "tool"},
                "material_guidance": ["prefer metal"],
                "vlm": {"backend": "mock"},
            }
        ),
        encoding="utf-8",
    )

    result = GenerateMaterialLibraryConfigTask().run({"config_path": str(config_path)})

    assert result["output_dir"] == str((config_path.parent / "out").resolve())
    assert result["input_usd_path"] == str((tmp_path / "asset.usd").resolve())
    assert result["material_generation_plan_path"] == str(
        (config_path.parent / "plans/plan.yaml").resolve()
    )
    assert result["material_generation_plan_base_dir"] == str(config_path.parent)
    assert result["prototype_materials_path"] == str(
        (config_path.parent / "prototypes/materials.yaml").resolve()
    )
    assert result["prototype_materials_data"]["library_path"] == str(
        (config_path.parent / "prototypes/lib.usda").resolve()
    )
    assert result["reference_images"] == [
        str((config_path.parent / "refs/front.png").resolve())
    ]
    assert result["rendered_preview_paths"] == ["/abs/preview.png"]
    assert result["composition_images"] == [
        str((config_path.parent / "composition.png").resolve())
    ]
    assert result["material_guidance"] == ["prefer metal"]
    assert result["vlm_config"] == {"backend": "mock"}
    assert result["write_material_generation_plan"] is False


def test_generate_material_library_config_defaults_and_errors(tmp_path: Path) -> None:
    task = GenerateMaterialLibraryConfigTask()
    with pytest.raises(ValueError, match="config_path is required"):
        task.run({})
    with pytest.raises(FileNotFoundError):
        task.run({"config_path": str(tmp_path / "missing.yaml")})

    config_path = tmp_path / "generate.yaml"
    config_path.write_text(
        yaml.safe_dump({"planning_guidance": "keep it simple"}),
        encoding="utf-8",
    )
    result = task.run({"config_path": str(config_path)})
    assert result["output_dir"] == str(tmp_path / "generated_material_library")
    assert result["planning_guidance"] == "keep it simple"

    bad_paths = tmp_path / "bad-paths.yaml"
    bad_paths.write_text(
        yaml.safe_dump({"reference_images": {"not": "a list"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_images must be a list"):
        task.run({"config_path": str(bad_paths)})

    inline_plan = tmp_path / "inline-plan.yaml"
    inline_plan.write_text(
        yaml.safe_dump({"material_generation_plan": {"materials": [_recipe_dict()]}}),
        encoding="utf-8",
    )
    assert task.run({"config_path": str(inline_plan)})["material_generation_plan"] == {
        "materials": [_recipe_dict()]
    }


def test_generate_material_library_run_updates_context(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_build_generated_material_library(plan, output_dir, **kwargs):
        captured["plan"] = plan
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs
        manifest = tmp_path / "materials.yaml"
        library = tmp_path / "material_library.usda"
        manifest.write_text("entries: []\n", encoding="utf-8")
        library.write_text("#usda 1.0\n", encoding="utf-8")
        return SimpleNamespace(
            material_library_path=library,
            materials_manifest_path=manifest,
            generation_plan_path=None,
            materials_data={
                "library_path": str(library),
                "entries": [{"name": "Blue", "binding": "/World/Looks/Blue"}],
            },
        )

    monkeypatch.setattr(
        generate_module,
        "build_generated_material_library",
        fake_build_generated_material_library,
    )
    monkeypatch.setattr(
        generate_module,
        "validate_generated_material_library",
        lambda path: SimpleNamespace(
            ok=True,
            errors=(),
            warnings=("warn",),
            metadata={"entry_count": 1},
        ),
    )

    existing_plan = tmp_path / "existing-plan.yaml"
    existing_plan.write_text(
        yaml.safe_dump({"materials": [_recipe_dict()]}),
        encoding="utf-8",
    )
    context = {
        "output_dir": tmp_path / "package",
        "material_generation_plan": {"materials": [_recipe_dict()]},
        "material_authoring": {
            "use_default_prototypes": False,
            "prototype_min_score": 0.25,
        },
        "prototype_materials_data": {"library_path": "ignored"},
        "prototype_materials_path": "ignored.yaml",
        "material_profile": "display_color",
        "write_material_generation_plan": False,
        "include_generation_metadata": False,
        "material_generation_plan_path": str(existing_plan),
    }
    result = GenerateMaterialLibraryTask().run(context)

    assert result["material_library_path"].endswith("material_library.usda")
    assert result["material_generation_plan_path"] == str(existing_plan)
    assert result["generated_material_entries"] == [
        {"name": "Blue", "binding": "/World/Looks/Blue"}
    ]
    kwargs = captured["kwargs"]
    assert kwargs["prototype_materials_data"] is None
    assert kwargs["prototype_materials_path"] is None
    assert kwargs["prototype_min_score"] == 0.25
    assert kwargs["include_generation_metadata"] is False


def test_generate_material_library_run_rejects_failed_validation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        generate_module,
        "build_generated_material_library",
        lambda *args, **kwargs: SimpleNamespace(
            materials_manifest_path=tmp_path / "materials.yaml"
        ),
    )
    monkeypatch.setattr(
        generate_module,
        "validate_generated_material_library",
        lambda path: SimpleNamespace(ok=False, errors=("bad texture",)),
    )

    with pytest.raises(ValueError, match="bad texture"):
        GenerateMaterialLibraryTask().run(
            {"material_generation_plan": {"materials": [_recipe_dict()]}}
        )


def test_generate_material_library_loads_plan_files_and_vlm_paths(
    monkeypatch, tmp_path: Path
) -> None:
    task = GenerateMaterialLibraryTask()
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump({"materials": [_recipe_dict()]}),
        encoding="utf-8",
    )

    loaded = task._load_or_create_plan(
        {"material_generation_plan_path": str(plan_path)},
        listener=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    assert loaded.materials[0].name == "Blue Plastic"
    with pytest.raises(FileNotFoundError):
        task._load_or_create_plan(
            {"material_generation_plan_path": str(tmp_path / "missing.yaml")},
            listener=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
    with pytest.raises(ValueError, match="requires material_generation_plan"):
        task._create_plan_with_vlm({}, listener=SimpleNamespace(info=lambda *a: None))

    created: dict[str, object] = {}

    def fake_create_vlm(backend: str, **kwargs):
        created["backend"] = backend
        created["kwargs"] = kwargs
        return _FakeVLM(json.dumps({"materials": [_recipe_dict()]}))

    monkeypatch.setattr(generate_module, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(
        generate_module,
        "get_api_key_for_model_config",
        lambda backend, config, label: "api-key",
    )
    plan = task._create_plan_with_vlm(
        {
            "reference_images": ["front.png", "front.png", "side.png"],
            "composition_images": ["comp.png"],
            "rendered_preview_paths": ["render.png"],
            "vlm_config": {
                "backend": "mock",
                "model": "planner",
                "base_url": "http://local",
                "timeout": 30,
            },
            "material_guidance": ["prefer satin finish", "avoid chipped paint"],
        },
        listener=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    assert plan.materials[0].material_id == "blue"
    assert created["backend"] == "mock"
    assert created["kwargs"] == {
        "model": "planner",
        "base_url": "http://local",
        "timeout": 30,
        "api_key": "api-key",
    }
    prompt = task._build_planning_prompt(
        {"material_guidance": ["prefer satin finish", "avoid chipped paint"]}
    )
    assert "- prefer satin finish" in prompt
    assert "- avoid chipped paint" in prompt

    fallback_task = GenerateMaterialLibraryTask()
    monkeypatch.setattr(
        fallback_task,
        "_create_plan_with_vlm",
        lambda context, listener: MaterialGenerationPlan.from_dict(
            {"materials": [_recipe_dict()]}
        ),
    )
    assert (
        fallback_task._load_or_create_plan(
            {"reference_images": ["front.png"]},
            listener=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
        .materials[0]
        .name
        == "Blue Plastic"
    )

    monkeypatch.setattr(
        generate_module,
        "extract_json_from_llm_response",
        lambda *args, **kwargs: [_recipe_dict()],
    )
    list_plan = task._create_plan_with_vlm(
        {"vlm": _FakeVLM("ignored"), "reference_images": ["front.png"]},
        listener=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    assert list_plan.materials[0].name == "Blue Plastic"

    monkeypatch.setattr(
        generate_module,
        "extract_json_from_llm_response",
        lambda *args, **kwargs: "bad",
    )
    with pytest.raises(ValueError, match="did not return a JSON object"):
        task._create_plan_with_vlm(
            {"vlm": _FakeVLM("ignored"), "reference_images": ["front.png"]},
            listener=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
