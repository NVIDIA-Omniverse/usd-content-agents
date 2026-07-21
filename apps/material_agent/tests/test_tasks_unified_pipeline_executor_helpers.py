# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for UnifiedPipelineExecutorTask helper and runtime behavior."""

from __future__ import annotations

import asyncio
import enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from world_understanding.utils.artifacts import remove_legacy_pipeline_temp
from world_understanding.utils.credentials import InlineSecretError

from material_agent.materials import FALLBACK_MATERIAL_NAME
from material_agent.tasks.prepare_dataset import (
    render_system_prompt_from_prepare_config,
)
from material_agent.tasks.unified_pipeline_executor import (
    UnifiedPipelineExecutorTask,
    _auto_wire_reference_generation_inputs,
    _build_child_config_dict,
    _build_runtime_pipeline_context,
    _dedupe_paths,
    _make_yaml_safe,
    _pipeline_input_config,
    _pipeline_predict_vlm_config,
    _raise_if_cancelled,
)


class _Mode(enum.Enum):
    FAST = "fast"


class _NonCopyableRuntimeClient:
    def __deepcopy__(self, _memo: dict[int, Any]) -> _NonCopyableRuntimeClient:
        raise AssertionError("runtime clients must not be deep-copied")


_WORKFLOW_FACTORY_BY_STEP = {
    "validate_input": "create_validate_input_workflow_from_config",
    "optimize_usd": "create_optimize_usd_workflow_from_config",
    "render_preview": "create_render_preview_workflow_from_config",
    "identify_asset": "create_identify_asset_workflow_from_config",
    "generate_reference_image": "create_generate_reference_image_workflow_from_config",
    "generate_material_library": "create_generate_material_library_workflow_from_config",
    "build_dataset_usd": "create_usd_data_preparation_workflow_from_config",
    "build_dataset_pdf_vectorstore": "create_pdf_vectorstore_workflow_from_config",
    "build_dataset_prepare_dataset": "create_prepare_dataset_workflow_from_config",
    "cluster_prims": "create_cluster_prims_workflow_from_config",
    "predict": "create_prediction_workflow_from_config",
    "expand_cluster_predictions": "create_expand_cluster_predictions_workflow_from_config",
    "benchmark": "create_benchmark_workflow_from_config",
    "validate_predictions": "create_validate_predictions_workflow_from_config",
    "harmonize_predictions": "create_harmonize_predictions_workflow_from_config",
    "evaluate": "create_evaluation_workflow_from_config",
    "apply": "create_apply_workflow_from_config",
    "refine": "create_iterative_apply_workflow_from_config",
    "restore_usd": "create_restore_usd_workflow_from_config",
    "validate_output": "create_validate_output_workflow_from_config",
    "render": "create_render_workflow_from_config",
}


_DEFAULT_WORKFLOW_RESULTS = {
    "validate_input": {"validation_result": {"issues": []}, "validation_success": True},
    "optimize_usd": {
        "optimized_usd_path": "optimized.usdc",
        "optimization_success": True,
        "original_usd_path": "input.usda",
        "original_prim_count": 10,
        "optimization_metadata": {"prim_map": {}},
    },
    "render_preview": {
        "output_dir": "preview",
        "rendered_preview_paths": ["preview.png"],
    },
    "identify_asset": {
        "identification": {"asset": "widget"},
        "identification_path": "identify.json",
        "image_gen_prompt": "make references",
    },
    "generate_reference_image": {
        "output_dir": "refs",
        "generated_reference_image_paths": ["generated.png"],
    },
    "generate_material_library": {
        "output_dir": "generated_materials",
        "generated_material_library_path": "generated.usda",
        "generated_material_entries": [],
        "generated_materials_data": {"entries": []},
    },
    "build_dataset_usd": {"output_dir": "usd_dataset", "num_prims": 4, "num_images": 8},
    "build_dataset_pdf_vectorstore": {"output_dir": "vectorstore"},
    "build_dataset_prepare_dataset": {
        "dataset_path": "dataset",
        "dataset_jsonl_path": "dataset.jsonl",
        "vlm_prompt_path": "prompt.txt",
        "num_entries": 2,
    },
    "cluster_prims": {
        "cluster_map_path": "cluster_map.jsonl",
        "dataset_representatives_path": "representatives.jsonl",
        "cluster_prims_ran": True,
    },
    "predict": {"predictions_path": "predictions.jsonl", "predictions_count": 3},
    "expand_cluster_predictions": {"predictions_path": "expanded.jsonl"},
    "benchmark": {"predictions_path": "benchmark.jsonl", "predictions_count": 3},
    "validate_predictions": {
        "predictions_path": "validated.jsonl",
        "validation_stats": {"valid": 3},
    },
    "harmonize_predictions": {
        "predictions_path": "harmonized.jsonl",
        "harmonized_count": 3,
        "remap": {},
    },
    "evaluate": {
        "evaluation_path": "evaluation.json",
        "html_report_path": "report.html",
        "metrics": {},
    },
    "apply": {"output_usd_path": "applied.usda", "materials_applied": {}},
    "refine": {"final_output_path": "refined.usda"},
    "restore_usd": {
        "restored_usd_path": "restored.usda",
        "restored_predictions_path": "restored.jsonl",
        "restore_success": True,
    },
    "validate_output": {
        "validation_result": {"issues": []},
        "validation_success": True,
    },
    "render": {
        "rendered_image_paths": ["render.png"],
        "rendered_image_path": "render.png",
    },
}


class _FakeWorkflow:
    def __init__(self, step_name: str, result, captured: dict[str, list[dict]]):
        self.step_name = step_name
        self.result = result
        self.captured = captured

    def run(self, step_context):
        self.captured.setdefault(self.step_name, []).append(dict(step_context))
        return self.result

    async def arun(self, step_context):
        self.captured.setdefault(self.step_name, []).append(dict(step_context))
        return self.result


def _patch_fake_workflows(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, list[dict]],
    result_overrides: dict[str, object] | None = None,
) -> None:
    import material_agent.workflows as workflows

    results = {**_DEFAULT_WORKFLOW_RESULTS, **(result_overrides or {})}
    for step_name, factory_name in _WORKFLOW_FACTORY_BY_STEP.items():
        monkeypatch.setattr(
            workflows,
            factory_name,
            lambda step_name=step_name: _FakeWorkflow(
                step_name,
                results.get(step_name, {"ok": True}),
                captured,
            ),
        )


def test_make_yaml_safe_normalizes_nested_non_primitives() -> None:
    safe = _make_yaml_safe(
        {
            Path("root"): {
                "path": Path("/tmp/example"),
                "mode": _Mode.FAST,
                "items": (1, Path("child")),
                "flags": {"b", "a"},
            }
        }
    )

    assert safe["root"]["path"] == "/tmp/example"
    assert safe["root"]["mode"] == "fast"
    assert safe["root"]["items"] == [1, "child"]
    assert safe["root"]["flags"] == ["a", "b"]


def test_make_yaml_safe_canonicalizes_mixed_sets_after_normalization() -> None:
    safe = _make_yaml_safe(
        {
            "mixed": {
                Path("b"),
                "a",
                2,
                ("nested", 1),
                frozenset({"z", 3}),
            }
        }
    )

    assert safe == {
        "mixed": [2, "a", "b", ["nested", 1], [3, "z"]],
    }


@pytest.mark.parametrize(
    "unsupported",
    [object(), float("nan"), float("inf"), b"binary", date(2026, 7, 15)],
)
def test_make_yaml_safe_rejects_unsupported_values_value_free(
    unsupported: object,
) -> None:
    sentinel = "unsupported-config-sentinel-727"

    with pytest.raises(
        TypeError,
        match="^Unsupported YAML-equivalent configuration value$",
    ) as exc_info:
        _make_yaml_safe({"ordered": ["before", unsupported, sentinel]})

    assert sentinel not in str(exc_info.value)
    assert repr(unsupported) not in str(exc_info.value)


@pytest.mark.parametrize(
    "unsupported_mapping",
    [{1: "numeric", "1": "string"}, {("nested", 1): "value"}],
)
def test_make_yaml_safe_rejects_ambiguous_mapping_keys_value_free(
    unsupported_mapping: dict[object, str],
) -> None:
    with pytest.raises(
        TypeError,
        match="^Unsupported YAML-equivalent configuration value$",
    ) as exc_info:
        _make_yaml_safe(unsupported_mapping)

    assert repr(unsupported_mapping) not in str(exc_info.value)


def test_make_yaml_safe_rejects_recursive_containers_value_free() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(
        TypeError,
        match="^Unsupported YAML-equivalent configuration value$",
    ):
        _make_yaml_safe({"recursive": recursive})


def test_executor_top_level_helper_branches(tmp_path: Path) -> None:
    event_listener = MagicMock()
    listener = MagicMock()
    with pytest.raises(asyncio.CancelledError):
        _raise_if_cancelled(
            {
                "cancel_checker": lambda: True,
                "event_listener": event_listener,
                "current_step": "predict",
            },
            listener,
        )
    event_listener.event.assert_called_once_with(
        "step.cancelled",
        {"step_name": "predict", "message": "Pipeline cancellation requested"},
    )
    listener.event.assert_not_called()

    assert _dedupe_paths(["a", "", "a", Path("ignored"), 1, "b"]) == ["a", "b"]
    assert _pipeline_input_config({"pipeline_config": "bad"}) == {}
    assert _pipeline_predict_vlm_config(
        {
            "step_configs": {"predict": "bad"},
            "pipeline_config": {"steps": {"predict": {"vlm": {"backend": "fallback"}}}},
        }
    ) == {"backend": "fallback"}
    assert (
        _pipeline_predict_vlm_config({"pipeline_config": {"steps": {"predict": {}}}})
        == {}
    )

    state = {
        "step_outputs": {
            "render_preview": {
                "rendered_preview_paths": ["preview.png"],
                "composition_images": ["composition.png"],
            },
            "identify_asset": {
                "identification": {"asset_type": "tool"},
                "image_gen_prompt": "polished metal tool",
            },
            "generate_reference_image": {
                "generated_reference_image_paths": ["generated.png"]
            },
        }
    }
    context = {
        "working_dir": str(tmp_path),
        "pipeline_config": {
            "input": {"usd_path": "asset.usd", "reference_images": ["asset_ref.png"]}
        },
        "step_configs": {"predict": {"vlm": {"backend": "nim"}}},
    }
    identify_config: dict[str, object] = {}
    _auto_wire_reference_generation_inputs(
        step_name="identify_asset",
        step_config=identify_config,
        context=context,
        pipeline_state=state,
    )
    assert identify_config["rendered_preview_paths"] == ["preview.png"]
    assert identify_config["composition_images"] == ["composition.png"]
    assert identify_config["reference_images"] == ["asset_ref.png"]
    assert identify_config["usd_path"] == "asset.usd"
    assert identify_config["output_dir"] == str(tmp_path / "identify_asset")
    assert identify_config["vlm_config"] == {"backend": "nim"}

    reference_context = {
        "pipeline_config": {"input": {"reference_images": ["scene.png"]}}
    }
    reference_config: dict[str, object] = {}
    _auto_wire_reference_generation_inputs(
        step_name="generate_reference_image",
        step_config=reference_config,
        context=reference_context,
        pipeline_state=state,
    )
    assert reference_config["rendered_preview_paths"] == ["preview.png"]
    assert reference_config["identification"] == {"asset_type": "tool"}
    assert reference_config["image_gen_prompt"] == "polished metal tool"
    assert reference_config["reference_images"] == ["scene.png"]

    prepare_config: dict[str, object] = {"reference_images": "not-a-list"}
    _auto_wire_reference_generation_inputs(
        step_name="build_dataset_prepare_dataset",
        step_config=prepare_config,
        context=reference_context,
        pipeline_state=state,
    )
    assert prepare_config["reference_images"] == ["generated.png", "scene.png"]

    non_list_reference_config: dict[str, object] = {}
    _auto_wire_reference_generation_inputs(
        step_name="generate_reference_image",
        step_config=non_list_reference_config,
        context={"pipeline_config": {"input": {"reference_images": "bad"}}},
        pipeline_state={"step_outputs": {}},
    )
    assert "reference_images" not in non_list_reference_config

    executor = UnifiedPipelineExecutorTask()
    assert executor._get_step_list_key() == "steps_to_run"
    assert executor._get_required_context_keys() == ["steps_to_run", "step_configs"]
    assert executor._get_state_file({"working_dir": tmp_path}) == (
        tmp_path / ".pipeline_state.json"
    )


def test_executor_material_activation_and_prediction_helpers() -> None:
    executor = UnifiedPipelineExecutorTask()

    context = {"materials_data": {"library_path": "/default.usda", "entries": []}}
    prepare_config = {"materials_list": ["Default Steel"]}
    original_system_prompt = render_system_prompt_from_prepare_config(prepare_config)
    step_configs = {
        "build_dataset_prepare_dataset": prepare_config,
        "predict": {"system_prompt": original_system_prompt},
        "benchmark": {"system_prompt": "Explicit trusted benchmark prompt"},
        "validate_predictions": {},
        "harmonize_predictions": {},
        "apply": {},
        "refine": {},
    }
    executor._activate_generated_material_library({}, context, step_configs)
    executor._activate_generated_material_library(
        {"generated_materials_data": {"entries": []}},
        context,
        step_configs,
    )
    executor._activate_generated_material_library(
        {
            "generated_materials_data": {
                "library_path": "/generated.usda",
                "entries": [
                    {
                        "name": "Generated Blue",
                        "description": "blue generated material",
                        "binding": "/World/Looks/Generated_Blue",
                    }
                ],
            }
        },
        context,
        step_configs,
    )

    assert context["default_materials_data"]["library_path"] == "/default.usda"
    assert context["materials_data"]["library_path"] == "/generated.usda"
    refreshed_prompt = step_configs["predict"]["system_prompt"]
    assert "Generated Blue" in refreshed_prompt
    assert FALLBACK_MATERIAL_NAME in refreshed_prompt
    assert "__USE_DEFAULT_LIBRARY__" in refreshed_prompt
    assert "Default Steel" not in refreshed_prompt
    assert (
        step_configs["benchmark"]["system_prompt"]
        == "Explicit trusted benchmark prompt"
    )
    assert (
        step_configs["refine"]["apply"]["materials_mapping"]["Generated Blue"]
        == "/World/Looks/Generated_Blue"
    )
    assert executor._with_default_fallback_entry(
        [{"name": "__USE_DEFAULT_LIBRARY__"}]
    ) == [{"name": "__USE_DEFAULT_LIBRARY__"}]
    assert executor._material_slug("  ***  ") == "Material"

    assert executor._selected_prediction_material({"materials": "Steel"}) == "Steel"
    assert executor._selected_prediction_material({"material": "Copper"}) == "Copper"
    assert (
        executor._selected_prediction_material({"predicted_material": "Rubber"})
        == "Rubber"
    )
    assert executor._selected_prediction_material({}) is None
    prediction: dict[str, object] = {"materials": "old"}
    executor._set_selected_prediction_material(prediction, "Plastic")
    assert prediction["materials"] == {"material": "Plastic"}
    assert executor._prediction_reason_text({"reasoning": " because "}) == "because"
    assert executor._truncate_text("short", max_chars=20) == "short"
    assert executor._truncate_text("word " * 20, max_chars=15).endswith("...")
    assert executor._material_salient_colors(None) == set()
    assert executor._text_mentions_color("off-white shell", "ivory") is True


def test_executor_fallback_reference_and_choice_helpers(
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    for path in (image_a, image_b, image_c):
        path.write_bytes(b"png")
    state_file = tmp_path / ".pipeline_state.json"
    state_file.write_text(
        json.dumps(
            {
                "step_outputs": {
                    "generate_material_library": {"reference_images": [str(image_c), 7]}
                }
            }
        ),
        encoding="utf-8",
    )

    paths = executor._fallback_reference_image_paths(
        {
            "working_dir": str(tmp_path),
            "generated_reference_image_paths": str(image_a),
            "reference_images": [str(image_a), str(image_b), 42],
            "pipeline_state": {
                "step_outputs": {
                    "generate_reference_image": {
                        "generated_reference_image_paths": [str(image_b)]
                    }
                }
            },
        },
        {"build_dataset_prepare_dataset": {"reference_images": [Path(image_c)]}},
    )
    assert paths == [str(image_a), str(image_b), str(image_c)]

    valid_names = {"Black Plastic", "Blue Plastic"}
    assert executor._coerce_default_material_choice("Black Plastic", valid_names) == (
        "Black Plastic",
        "",
    )
    assert executor._coerce_default_material_choice(
        {"selected_material": "Blue Plastic", "reason": "blue clip"},
        valid_names,
    ) == ("Blue Plastic", "blue clip")
    assert executor._coerce_default_material_choice(
        {"material": "Orange Plastic", "evidence": "orange"},
        valid_names,
    ) == (None, "orange")

    entries = [
        {"name": "Red Plastic", "description": "red shell"},
        {"name": "Blue Plastic", "description": "blue shell"},
    ]
    selected, rejected = executor._guarded_default_material_choice(
        {
            "id": "/World/BlueClip",
            "materials": {"original_response": "blue visible clip"},
        },
        "Red Plastic",
        entries,
        "blue clip",
        listener=MagicMock(),
    )
    assert (selected, rejected) == ("Blue Plastic", "Red Plastic")
    assert executor._guarded_default_material_choice(
        {"id": "/World/Part"},
        "Generic Plastic",
        [{"name": "Generic Plastic", "description": "neutral part"}],
        "",
        listener=MagicMock(),
    ) == ("Generic Plastic", None)
    with (
        patch.object(
            executor,
            "_default_choice_has_color_evidence",
            return_value=False,
        ),
        patch.object(executor, "_heuristic_default_material", return_value="Heuristic"),
    ):
        assert executor._guarded_default_material_choice(
            {"id": "/World/Part"},
            "Selected",
            [{"name": ""}, {"name": "Selected"}],
            "",
            listener=MagicMock(),
        ) == ("Selected", None)
        assert executor._guarded_default_material_choice(
            {"id": "/World/Part"},
            None,
            [{"name": "Heuristic"}],
            "",
            listener=MagicMock(),
        ) == ("Heuristic", None)

    assert executor._fallback_llm_config(
        {"predict": {"llm": {"max_completion_tokens": 99}}}
    ) == {"max_completion_tokens": 99, "max_tokens": 99}
    assert executor._fallback_llm_config({"predict": {"llm": {}}}) is None
    assert executor._fallback_vlm_config(
        {"predict": {"vlm": {"max_completion_tokens": 77}}}
    ) == {"max_completion_tokens": 77, "max_tokens": 77}
    assert executor._fallback_vlm_config({"predict": {"vlm": {}}}) is None


def test_executor_fallback_human_message_content_with_images(tmp_path: Path) -> None:
    from PIL import Image

    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    image_path = tmp_path / "ref.png"
    Image.new("RGB", (2, 2), color="blue").save(image_path)
    bad_image = tmp_path / "bad.png"
    bad_image.write_text("not an image", encoding="utf-8")

    content = executor._fallback_human_message_content(
        "Choose a default material.",
        [str(image_path), str(bad_image)],
        listener,
    )

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["text"].startswith("Reference image 0")
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    listener.warning.assert_called()


def test_executor_llm_default_material_choices_vlm_and_llm_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    reference_image = tmp_path / "ref.png"
    reference_image.write_bytes(b"png")
    predictions = [
        {"id": "/World/A", "materials": {"original_response": "blue cap"}},
        {"id": "/World/B", "materials": {"original_response": "plain insert"}},
    ]
    default_entries = [
        {"name": "No Description"},
        {"name": "Blue Plastic", "description": "blue shell"},
    ]

    assert (
        executor._llm_default_material_choices(
            predictions,
            [],
            default_entries,
            llm_config={"backend": "mock"},
            vlm_config=None,
            reference_image_paths=None,
            listener=listener,
        )
        == {}
    )

    class FakeVLM:
        def generate(self, **kwargs):
            assert kwargs["images"] == [str(reference_image)]
            assert "blue shell" not in kwargs["prompt"]
            assert "blue shell" not in kwargs["system_prompt"]
            assert "untrusted JSON data" in kwargs["prompt"]
            return "vlm response"

    monkeypatch.setattr(
        "world_understanding.agentic.domain_tasks.model_provisioning.ModelProvisioningTask.create_vlm",
        lambda self, config: FakeVLM(),
    )
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda text: {
            "bad": {"material": "Blue Plastic"},
            "99": {"material": "Blue Plastic"},
            "0": {"material": "Missing Material"},
            "1": {"selected_material": "Blue Plastic", "evidence": "blue cap"},
        },
    )

    choices = executor._llm_default_material_choices(
        predictions,
        [0, 1],
        default_entries,
        llm_config=None,
        vlm_config={"backend": "mock", "temperature": 0.0, "max_tokens": 32},
        reference_image_paths=[str(reference_image)],
        listener=listener,
    )
    assert choices == {1: {"material": "Blue Plastic", "visual_evidence": "blue cap"}}

    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config: None,
    )
    assert (
        executor._llm_default_material_choices(
            predictions,
            [0],
            default_entries,
            llm_config={"backend": "mock"},
            vlm_config=None,
            reference_image_paths=[],
            listener=listener,
        )
        == {}
    )

    class FakeLLM:
        def invoke(self, messages):
            assert len(messages) == 2
            return SimpleNamespace(content="not json")

    monkeypatch.setattr(
        "world_understanding.functions.models.chat_models.create_chat_model_from_config",
        lambda config: FakeLLM(),
    )
    monkeypatch.setattr(
        "world_understanding.utils.llm_parsing.extract_json_from_llm_response",
        lambda text: ["not", "a", "mapping"],
    )
    assert (
        executor._llm_default_material_choices(
            predictions,
            [0],
            [{}, {"name": "Blue Plastic", "description": "blue shell"}],
            llm_config={"backend": "mock"},
            vlm_config=None,
            reference_image_paths=[],
            listener=listener,
        )
        == {}
    )


def test_executor_copy_and_build_combined_material_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pxr import Sdf

    executor = UnifiedPipelineExecutorTask()
    source_path = tmp_path / "source.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    Sdf.CreatePrimInLayer(source_layer, Sdf.Path("/World/Looks/Steel"))
    source_layer.Save()

    import material_agent.tasks.apply_materials_to_usd as apply_materials_module

    remapped = []
    cleared = []
    monkeypatch.setattr(
        apply_materials_module,
        "remap_asset_paths_in_prim",
        lambda *args: remapped.append(args),
    )
    monkeypatch.setattr(
        apply_materials_module,
        "clear_color_space_on_empty_asset_inputs",
        lambda *args: cleared.append(args),
    )

    output_path = tmp_path / "combined" / "materials.usda"
    output_path.parent.mkdir()
    output_path.write_text("stale", encoding="utf-8")
    executor._copy_materials_into_combined_library(
        output_path,
        [
            (
                "Steel",
                str(source_path),
                "/World/Looks/Steel",
                "/World/Looks/Generated_Steel",
            )
        ],
        listener=MagicMock(),
    )
    copied_layer = Sdf.Layer.FindOrOpen(str(output_path))
    assert copied_layer.GetPrimAtPath("/World/Looks/Generated_Steel")
    assert remapped
    assert cleared

    with pytest.raises(RuntimeError, match="Failed to open material library"):
        executor._copy_materials_into_combined_library(
            tmp_path / "missing-output.usda",
            [
                (
                    "Missing",
                    str(tmp_path / "missing.usda"),
                    "/World/Looks/M",
                    "/World/Looks/M",
                )
            ],
            listener=MagicMock(),
        )
    with pytest.raises(RuntimeError, match="Material prim not found"):
        executor._copy_materials_into_combined_library(
            tmp_path / "missing-prim.usda",
            [("Missing", str(source_path), "/World/Looks/Missing", "/World/Looks/M")],
            listener=MagicMock(),
        )

    copied: dict[str, object] = {}
    monkeypatch.setattr(
        executor,
        "_copy_materials_into_combined_library",
        lambda output, sources, listener: copied.update(
            {"output": output, "sources": sources}
        ),
    )
    context = {
        "working_dir": str(tmp_path),
        "generated_materials_data": {
            "library_path": str(source_path),
            "entries": [{"name": "Steel", "binding": "/World/Looks/Steel"}],
        },
        "default_materials_data": {"library_path": "", "entries": []},
    }
    step_configs = {"apply": {}, "refine": {}}
    result = executor._build_combined_material_library_for_apply(
        context,
        step_configs,
        {FALLBACK_MATERIAL_NAME, "Steel", "Missing"},
        listener=MagicMock(),
    )
    assert result is not None
    assert copied["sources"] == [
        (
            "Steel",
            str(source_path),
            "/World/Looks/Steel",
            "/World/Looks/Steel",
        )
    ]
    assert step_configs["refine"]["apply"]["materials_mapping"]["Steel"] == (
        "/World/Looks/Steel"
    )
    assert FALLBACK_MATERIAL_NAME in step_configs["apply"]["materials_mapping"]

    assert (
        executor._build_combined_material_library_for_apply(
            {},
            {},
            {"Steel"},
            listener=MagicMock(),
        )
        is None
    )
    assert (
        executor._build_combined_material_library_for_apply(
            context,
            {},
            set(),
            listener=MagicMock(),
        )
        is None
    )
    assert (
        executor._build_combined_material_library_for_apply(
            {
                "generated_materials_data": {
                    "library_path": str(source_path),
                    "entries": [],
                },
                "default_materials_data": {
                    "library_path": str(source_path),
                    "entries": [],
                },
            },
            {},
            {"Missing"},
            listener=MagicMock(),
        )
        is None
    )


def test_executor_combined_library_root_preserves_created_texture_paths(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask

    work_dir = tmp_path / "work"
    package_dir = work_dir / "created_materials" / "packages" / "paint"
    texture_dir = package_dir / "textures" / "paint"
    texture_dir.mkdir(parents=True)
    (texture_dir / "albedo.png").write_bytes(b"png")

    source_path = package_dir / "material_library.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    UsdShade.Material.Define(source_stage, "/World/Looks/Paint")
    texture_shader = UsdShade.Shader.Define(
        source_stage,
        "/World/Looks/Paint/AlbedoTexture",
    )
    texture_shader.CreateIdAttr("UsdUVTexture")
    texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/paint/albedo.png")
    )
    source_stage.GetRootLayer().Save()

    combined_path = work_dir / "combined_material_library.usda"
    UnifiedPipelineExecutorTask()._copy_materials_into_combined_library(
        combined_path,
        [("Paint", str(source_path), "/World/Looks/Paint", "/World/Looks/Paint")],
        listener=MagicMock(),
    )

    combined_stage = Usd.Stage.Open(str(combined_path))
    combined_attr = combined_stage.GetAttributeAtPath(
        "/World/Looks/Paint/AlbedoTexture.inputs:file"
    )
    assert combined_attr.Get().path == (
        "created_materials/packages/paint/textures/paint/albedo.png"
    )

    output_dir = work_dir / "output"
    output_dir.mkdir()
    output_path = output_dir / "output.usda"
    output_stage = Usd.Stage.CreateNew(str(output_path))
    task = ApplyMaterialsToUSDTask()
    task.listener = MagicMock()
    output_stage, copied = task._copy_library_materials(
        output_stage,
        str(combined_path),
        output_path,
        {"Paint": "/World/Looks/Paint"},
    )

    assert copied == {"Paint": "/World/Looks/Paint"}
    output_attr = output_stage.GetAttributeAtPath(
        "/World/Looks/Paint/AlbedoTexture.inputs:file"
    )
    assert output_attr.Get().path == (
        "../created_materials/packages/paint/textures/paint/albedo.png"
    )
    task.listener.warning.assert_not_called()


def test_executor_resolve_fallbacks_and_hydration_early_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()

    executor._resolve_generated_material_fallbacks_for_apply({}, {}, listener)
    executor._resolve_generated_material_fallbacks_for_apply(
        {"default_materials_data": {}, "generated_materials_data": {}},
        {"apply": {}},
        listener,
    )
    executor._resolve_generated_material_fallbacks_for_apply(
        {"default_materials_data": {}, "generated_materials_data": {}},
        {"apply": {"predictions_path": str(tmp_path / "missing.jsonl")}},
        listener,
    )

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n"
        + json.dumps(
            {"id": "/World/A", "materials": {"material": "__USE_DEFAULT_LIBRARY__"}}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        executor,
        "_llm_default_material_choices",
        lambda *args, **kwargs: {0: "Default Plastic"},
    )
    monkeypatch.setattr(
        executor,
        "_build_combined_material_library_for_apply",
        lambda *args, **kwargs: None,
    )
    executor._resolve_generated_material_fallbacks_for_apply(
        {
            "default_materials_data": {
                "entries": [
                    {"name": "Default Plastic", "description": "neutral plastic"}
                ]
            },
            "generated_materials_data": {"entries": []},
        },
        {"apply": {"predictions_path": str(predictions_path)}},
        listener,
    )
    rewritten = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert rewritten["materials"]["material"] == "Default Plastic"
    assert rewritten["materials"]["fallback_source"] == "__USE_DEFAULT_LIBRARY__"

    executor._hydrate_simready_materials_for_apply({}, {}, listener)
    executor._hydrate_simready_materials_for_apply({"materials_data": {}}, {}, listener)
    executor._hydrate_simready_materials_for_apply(
        {"materials_data": {"simready": {}, "entries": []}},
        {"apply": {}},
        listener,
    )

    no_selected_simready = tmp_path / "simready_no_selected.jsonl"
    no_selected_simready.write_text(
        "\n"
        + json.dumps(["not", "a", "prediction"])
        + "\n"
        + json.dumps({"id": "/A", "materials": {"material": "Missing SimReady"}})
        + "\n",
        encoding="utf-8",
    )
    executor._hydrate_simready_materials_for_apply(
        {
            "working_dir": str(tmp_path),
            "materials_data": {
                "simready": {},
                "entries": [
                    {
                        "name": "Plastic Test",
                        "binding": "/World/Looks/Plastic_Test",
                        "simready_source_path": "Materials/Plastic.usda",
                    }
                ],
            },
        },
        {"apply": {"predictions_path": str(no_selected_simready)}},
        listener,
    )

    import material_agent.simready as simready_catalog

    class FakeCatalogError(Exception):
        pass

    monkeypatch.setattr(simready_catalog, "SimReadyCatalogError", FakeCatalogError)
    monkeypatch.setattr(simready_catalog, "load_manifest", lambda path=None: {})
    monkeypatch.setattr(
        "material_agent.simready.hydration.hydrate_simready_library",
        lambda **kwargs: (_ for _ in ()).throw(FakeCatalogError("offline")),
    )

    simready_predictions = tmp_path / "simready.jsonl"
    simready_predictions.write_text(
        "{bad-json}\n"
        + json.dumps({"id": "/A", "materials": {"material": "Plastic Test"}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Failed to hydrate SimReady materials"):
        executor._hydrate_simready_materials_for_apply(
            {
                "working_dir": str(tmp_path),
                "materials_data": {
                    "simready": {},
                    "entries": [
                        {
                            "name": "Plastic Test",
                            "binding": "/World/Looks/Plastic_Test",
                            "simready_source_path": "Materials/Plastic.usda",
                        }
                    ],
                },
            },
            {"apply": {"predictions_path": str(simready_predictions)}},
            listener,
        )


def test_generated_fallback_rewrite_rejects_inline_credentials_before_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    predictions_path = tmp_path / "predictions.jsonl"
    original = (
        json.dumps(
            {
                "id": "/World/A",
                "materials": {"material": "__USE_DEFAULT_LIBRARY__"},
            }
        )
        + "\n"
    )
    predictions_path.write_text(original, encoding="utf-8")
    sentinel = "fallback-rewrite-secret-713"
    monkeypatch.setattr(
        executor,
        "_llm_default_material_choices",
        lambda *args, **kwargs: {
            0: {
                "material": "Default Plastic",
                "visual_evidence": (
                    "https://assets.example.test/evidence.png?"
                    f"X-Amz-Signature={sentinel}"
                ),
            }
        },
    )

    with pytest.raises(
        InlineSecretError,
        match="fallback-resolved predictions artifact contains inline credential",
    ):
        executor._resolve_generated_material_fallbacks_for_apply(
            {
                "default_materials_data": {
                    "entries": [
                        {
                            "name": "Default Plastic",
                            "description": "neutral plastic",
                        }
                    ]
                },
                "generated_materials_data": {"entries": []},
            },
            {"apply": {"predictions_path": str(predictions_path)}},
            listener,
        )

    assert predictions_path.read_text(encoding="utf-8") == original
    assert sentinel not in predictions_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        (
            {"id": "/World/PCB", "materials": {"reason": "circuit board"}},
            "Gray Plastic",
        ),
        ({"id": "/World/Foot", "materials": {"reason": "rubber foot"}}, "Black Rubber"),
        (
            {"id": "/World/Window", "materials": {"reason": "clear glass"}},
            "Clear Glass",
        ),
        ({"id": "/World/Screw", "materials": {"reason": "metal pin"}}, "Steel Metal"),
        ({"id": "/World/Knob", "materials": {"reason": "black knob"}}, "Black Plastic"),
        ({"id": "/World/Panel", "materials": {"reason": "gray panel"}}, "Gray Plastic"),
        (
            {"id": "/World/Case", "materials": {"reason": "white casing"}},
            "White Plastic",
        ),
        (
            {"id": "/World/Unknown", "materials": {"reason": "plain body"}},
            "Gray Plastic",
        ),
    ],
)
def test_executor_default_material_heuristic_branches(
    prediction: dict[str, object],
    expected: str,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    entries = [
        {"name": "Black Plastic", "description": "black dark plastic"},
        {"name": "Black Rubber", "description": "black matte rubber silicone"},
        {"name": "Clear Glass", "description": "clear transparent glass acrylic"},
        {"name": "Steel Metal", "description": "steel metal screw"},
        {"name": "Gray Plastic", "description": "gray grey dark plastic"},
        {"name": "White Plastic", "description": "white ivory cream plastic"},
    ]

    assert executor._heuristic_default_material(prediction, entries) == expected
    assert (
        executor._best_default_entry_by_tokens(
            [
                {"name": ""},
                {"bad": "entry"},
                {"name": "Orange", "description": "orange"},
            ],
            ("orange",),
            avoid_tokens=("orange",),
        )
        is None
    )
    assert (
        executor._heuristic_default_material(
            {"id": "/World/plain"},
            [{"name": "Fallback Only", "description": "unmatched material"}],
        )
        == "Fallback Only"
    )
    assert json.loads(
        executor._format_materials_for_prompt([{"name": "Fallback Only"}])
    ) == {"material_names": ["Fallback Only"]}
    with pytest.raises(ValueError, match="no usable entries"):
        executor._heuristic_default_material({"id": "/World/Part"}, [{"name": ""}])


def test_generated_material_fallbacks_are_resolved_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions = [
        {
            "id": "/World/part_a",
            "materials": {"material": "Generated Plastic"},
        },
        {
            "id": "/World/part_b",
            "materials": {
                "material": "__USE_DEFAULT_LIBRARY__",
                "original_response": "A small black internal clip.",
            },
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    ref_path = tmp_path / "reference_image_0.png"
    ref_path.write_text("placeholder", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_llm_choices(
        predictions,
        fallback_indices,
        default_entries,
        llm_config,
        vlm_config,
        reference_image_paths,
        listener,
    ):
        captured["reference_image_paths"] = reference_image_paths
        return {1: {"material": "Plastic Black", "visual_evidence": "black clip"}}

    monkeypatch.setattr(
        executor,
        "_llm_default_material_choices",
        fake_llm_choices,
    )

    copied: dict[str, object] = {}

    def fake_copy(output_path, material_sources, listener):
        copied["output_path"] = output_path
        copied["material_sources"] = material_sources
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(executor, "_copy_materials_into_combined_library", fake_copy)

    context = {
        "working_dir": str(tmp_path),
        "generated_materials_data": {
            "library_path": str(tmp_path / "generated.usda"),
            "entries": [
                {
                    "name": "Generated Plastic",
                    "description": "Generated asset-specific plastic",
                    "binding": "/World/Looks/Generated_Plastic",
                }
            ],
        },
        "default_materials_data": {
            "library_path": str(tmp_path / "default.usda"),
            "entries": [
                {
                    "name": "Plastic Black",
                    "description": "Default black plastic",
                    "binding": "/World/Looks/Plastic_Black",
                }
            ],
        },
    }
    step_configs = {
        "predict": {"llm": {"backend": "mock"}},
        "build_dataset_prepare_dataset": {"reference_images": [str(ref_path)]},
        "apply": {"predictions_path": str(predictions_path)},
    }

    executor._resolve_generated_material_fallbacks_for_apply(
        context,
        step_configs,
        listener=MagicMock(),
    )

    rewritten = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rewritten[1]["materials"]["material"] == "Plastic Black"
    assert rewritten[1]["materials"]["fallback_source"] == "__USE_DEFAULT_LIBRARY__"
    assert context["unknown_material_predictions"] == 0
    assert captured["reference_image_paths"] == [str(ref_path)]
    mapping = step_configs["apply"]["materials_mapping"]
    assert (
        Path(mapping["material_library_path"]).name == "combined_material_library.usda"
    )
    assert mapping["Generated Plastic"] == "/World/Looks/Generated_Plastic"
    assert mapping["Plastic Black"] == "/World/Looks/Plastic_Black"
    assert FALLBACK_MATERIAL_NAME in mapping
    assert copied["material_sources"] == [
        (
            "Generated Plastic",
            str(tmp_path / "generated.usda"),
            "/World/Looks/Generated_Plastic",
            "/World/Looks/Generated_Plastic",
        ),
        (
            "Plastic Black",
            str(tmp_path / "default.usda"),
            "/World/Looks/Plastic_Black",
            "/World/Looks/Plastic_Black",
        ),
    ]


def test_simready_materials_are_hydrated_before_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "{not-json}\n"
        + json.dumps(
            {
                "id": "/World/part",
                "materials": {"material": "Plastic Test"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "predictions": [
                    {
                        "id": "/World/nested",
                        "materials": {"material": "Metal Test"},
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    import material_agent.simready as simready_catalog
    import material_agent.simready.hydration as simready_hydration

    monkeypatch.setattr(
        simready_catalog,
        "load_manifest",
        lambda path=None: {"release_tag": "v-test"},
    )
    captured: dict[str, object] = {}

    def fake_hydrate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            library_path=tmp_path / "hydrated.usda",
            entries=[
                {
                    "name": "Plastic Test",
                    "description": "Synthetic SimReady plastic",
                    "binding": "/World/Looks/Plastic_Test",
                },
                {
                    "name": "Metal Test",
                    "description": "Synthetic SimReady metal",
                    "binding": "/World/Looks/Metal_Test",
                },
            ],
            report={"categories": ["Metal", "Plastic"]},
        )

    monkeypatch.setattr(
        simready_hydration,
        "hydrate_simready_library",
        fake_hydrate,
    )

    context = {
        "working_dir": str(tmp_path),
        "materials_data": {
            "library_path": "",
            "simready": {
                "library_id": "simready-light",
                "manifest_path": None,
                "cache_dir": str(tmp_path / "simready-cache"),
            },
            "entries": [
                {
                    "name": "Plastic Test",
                    "description": "Synthetic SimReady plastic",
                    "binding": "/World/Looks/Plastic_Test",
                    "simready_category": "Plastic",
                    "simready_source_path": "Materials/Plastic/Plastic_Test.usda",
                },
                {
                    "name": "Metal Test",
                    "description": "Not predicted",
                    "binding": "/World/Looks/Metal_Test",
                    "simready_category": "Metal",
                    "simready_source_path": "Materials/Metal/Metal_Test.usda",
                },
            ],
        },
    }
    step_configs = {"apply": {"predictions_path": str(predictions_path)}}
    listener = MagicMock()

    executor._hydrate_simready_materials_for_apply(
        context,
        step_configs,
        listener=listener,
    )

    listener.warning.assert_called_once()
    assert captured["material_names"] == {"Metal Test", "Plastic Test"}
    assert captured["cache_dir"] == str(tmp_path / "simready-cache")
    mapping = step_configs["apply"]["materials_mapping"]
    assert mapping["material_library_path"] == str(tmp_path / "hydrated.usda")
    assert mapping["Plastic Test"] == "/World/Looks/Plastic_Test"
    assert mapping["Metal Test"] == "/World/Looks/Metal_Test"
    assert FALLBACK_MATERIAL_NAME in mapping
    assert context["simready_hydration_report"] == {"categories": ["Metal", "Plastic"]}


def test_simready_materials_are_hydrated_from_explicit_apply_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()

    import material_agent.simready as simready_catalog
    import material_agent.simready.hydration as simready_hydration

    monkeypatch.setattr(
        simready_catalog,
        "load_manifest",
        lambda path=None: {"release_tag": "v-test"},
    )
    captured: dict[str, object] = {}

    def fake_hydrate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            library_path=tmp_path / "hydrated.usda",
            entries=[
                {
                    "name": "Plastic Test",
                    "description": "Synthetic SimReady plastic",
                    "binding": "/World/Looks/Plastic_Test",
                }
            ],
            report={"categories": ["Plastic"]},
        )

    monkeypatch.setattr(
        simready_hydration,
        "hydrate_simready_library",
        fake_hydrate,
    )

    context = {
        "working_dir": str(tmp_path),
        "materials_data": {
            "library_path": "",
            "simready": {
                "library_id": "simready-light",
                "manifest_path": None,
                "cache_dir": str(tmp_path / "simready-cache"),
            },
            "entries": [
                {
                    "name": " Plastic Test ",
                    "description": "Synthetic SimReady plastic",
                    "binding": "/World/Looks/Plastic_Test",
                    "simready_category": "Plastic",
                    "simready_source_path": "Materials/Plastic/Plastic_Test.usda",
                }
            ],
        },
    }
    step_configs = {
        "apply": {
            "materials_mapping": {
                "material_library_path": "",
                "Plastic Test": "/World/Looks/Plastic_Test",
            }
        }
    }

    executor._hydrate_simready_materials_for_apply(
        context,
        step_configs,
        listener=MagicMock(),
    )

    assert captured["material_names"] == {" Plastic Test "}
    mapping = step_configs["apply"]["materials_mapping"]
    assert mapping["material_library_path"] == str(tmp_path / "hydrated.usda")
    assert mapping["Plastic Test"] == "/World/Looks/Plastic_Test"
    assert context["simready_hydration_report"] == {"categories": ["Plastic"]}


def test_generated_material_fallback_refs_are_loaded_from_pipeline_state(
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    ref_dir = tmp_path / "generated_refs"
    ref_dir.mkdir()
    ref_path = ref_dir / "generated_ref_0.png"
    ref_path.write_text("placeholder", encoding="utf-8")
    (tmp_path / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "step_outputs": {
                    "generate_reference_image": {
                        "generated_reference_image_paths": [str(ref_path)]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    refs = executor._fallback_reference_image_paths(
        {"working_dir": str(tmp_path)},
        {},
    )

    assert refs == [str(ref_path)]


def test_generated_material_fallback_rejects_unsupported_salient_color(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions = [
        {
            "id": "/World/front_panel_board",
            "materials": {
                "material": "__USE_DEFAULT_LIBRARY__",
                "original_response": (
                    "The highlighted part is a thin rectangular internal "
                    "electronic component or PCB behind the front panel. "
                    "It is not visible in the reference images."
                ),
            },
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        executor,
        "_llm_default_material_choices",
        lambda predictions,
        fallback_indices,
        default_entries,
        llm_config,
        vlm_config,
        reference_image_paths,
        listener: {
            0: {
                "material": "Plastic Green",
                "visual_evidence": "internal electronic board",
            }
        },
    )

    monkeypatch.setattr(
        executor,
        "_copy_materials_into_combined_library",
        lambda output_path, material_sources, listener: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.write_text("#usda 1.0\n", encoding="utf-8"),
        ),
    )

    context = {
        "working_dir": str(tmp_path),
        "generated_materials_data": {
            "library_path": str(tmp_path / "generated.usda"),
            "entries": [
                {
                    "name": "Generated Plastic",
                    "description": "Generated asset-specific plastic",
                    "binding": "/World/Looks/Generated_Plastic",
                }
            ],
        },
        "default_materials_data": {
            "library_path": str(tmp_path / "default.usda"),
            "entries": [
                {
                    "name": "Plastic Green",
                    "description": "Dark green glossy plastic",
                    "binding": "/World/Looks/Plastic_Green",
                },
                {
                    "name": "Plastic Black",
                    "description": "Black plastic",
                    "binding": "/World/Looks/Plastic_Black",
                },
            ],
        },
    }
    step_configs = {
        "predict": {"llm": {"backend": "mock"}},
        "apply": {"predictions_path": str(predictions_path)},
    }

    executor._resolve_generated_material_fallbacks_for_apply(
        context,
        step_configs,
        listener=MagicMock(),
    )

    rewritten = json.loads(predictions_path.read_text(encoding="utf-8").strip())
    assert rewritten["materials"]["material"] == "Plastic Black"
    assert rewritten["materials"]["fallback_source"] == "__USE_DEFAULT_LIBRARY__"
    assert rewritten["materials"]["fallback_rejected_material"] == "Plastic Green"


def test_generated_material_fallback_allows_salient_color_with_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions = [
        {
            "id": "/World/green_button",
            "materials": {
                "material": "__USE_DEFAULT_LIBRARY__",
                "original_response": "The highlighted part is a visible green button.",
            },
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(prediction) + "\n" for prediction in predictions),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        executor,
        "_llm_default_material_choices",
        lambda predictions,
        fallback_indices,
        default_entries,
        llm_config,
        vlm_config,
        reference_image_paths,
        listener: {
            0: {
                "material": "Plastic Green",
                "visual_evidence": "visible green button in the reference image",
            }
        },
    )

    monkeypatch.setattr(
        executor,
        "_copy_materials_into_combined_library",
        lambda output_path, material_sources, listener: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.write_text("#usda 1.0\n", encoding="utf-8"),
        ),
    )

    context = {
        "working_dir": str(tmp_path),
        "generated_materials_data": {
            "library_path": str(tmp_path / "generated.usda"),
            "entries": [],
        },
        "default_materials_data": {
            "library_path": str(tmp_path / "default.usda"),
            "entries": [
                {
                    "name": "Plastic Green",
                    "description": "Dark green glossy plastic",
                    "binding": "/World/Looks/Plastic_Green",
                },
                {
                    "name": "Plastic Black",
                    "description": "Black plastic",
                    "binding": "/World/Looks/Plastic_Black",
                },
            ],
        },
    }
    step_configs = {
        "predict": {"llm": {"backend": "mock"}},
        "apply": {"predictions_path": str(predictions_path)},
    }

    executor._resolve_generated_material_fallbacks_for_apply(
        context,
        step_configs,
        listener=MagicMock(),
    )

    rewritten = json.loads(predictions_path.read_text(encoding="utf-8").strip())
    assert rewritten["materials"]["material"] == "Plastic Green"
    assert "fallback_rejected_material" not in rewritten["materials"]


def test_apply_resolves_generated_fallbacks_after_restore_autowiring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    stale_predictions = tmp_path / "stale_predictions.jsonl"
    restored_predictions = tmp_path / "restored_predictions.jsonl"
    stale_predictions.write_text("", encoding="utf-8")
    restored_predictions.write_text("", encoding="utf-8")
    input_usd = tmp_path / "input.usda"
    output_usd = tmp_path / "output.usda"
    input_usd.write_text("#usda 1.0\n", encoding="utf-8")

    captured: dict[str, str] = {}

    def capture_resolver(context, step_configs, listener):
        captured["predictions_path"] = step_configs["apply"]["predictions_path"]

    monkeypatch.setattr(
        executor,
        "_resolve_generated_material_fallbacks_for_apply",
        capture_resolver,
    )

    class FakeWorkflow:
        def run(self, step_context):
            return {
                "output_usd_path": str(output_usd),
                "materials_applied": {},
            }

    import material_agent.workflows as workflows

    monkeypatch.setattr(
        workflows,
        "create_apply_workflow_from_config",
        lambda: FakeWorkflow(),
    )

    step_config = {
        "input_usd_path": str(input_usd),
        "output_usd_path": str(output_usd),
        "predictions_path": str(stale_predictions),
        "materials_mapping": {},
    }
    context = {
        "working_dir": str(tmp_path),
        "step_configs": {"apply": step_config},
    }
    pipeline_state = {
        "step_outputs": {
            "optimize_usd": {"original_usd_path": str(input_usd)},
            "restore_usd": {"restored_predictions_path": str(restored_predictions)},
        }
    }

    outputs = executor._execute_step(
        "apply",
        step_config,
        context,
        object_store=None,
        pipeline_state=pipeline_state,
    )

    assert captured["predictions_path"] == str(restored_predictions)
    assert outputs["output_usd_path"] == str(output_usd)


def test_execute_step_sync_autowires_dispatch_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    captured: dict[str, list[dict]] = {}
    _patch_fake_workflows(monkeypatch, captured)
    resolver_calls: list[dict] = []
    hydrate_calls: list[dict] = []
    monkeypatch.setattr(
        executor,
        "_resolve_generated_material_fallbacks_for_apply",
        lambda _context, step_configs, _listener: resolver_calls.append(step_configs),
    )
    monkeypatch.setattr(
        executor,
        "_hydrate_simready_materials_for_apply",
        lambda _context, step_configs, _listener: hydrate_calls.append(step_configs),
    )

    base_context = {
        "working_dir": str(tmp_path),
        "event_listener": MagicMock(),
        "original_prim_count": 0,
        "num_prims": 0,
        "num_images": 0,
    }

    def run_step(
        step_name: str,
        step_config: dict[str, object] | None = None,
        step_outputs: dict[str, object] | None = None,
        *,
        extra_context: dict[str, object] | None = None,
        state_extra: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        config = step_config or {}
        context = {**base_context, **(extra_context or {})}
        pipeline_state = {"step_outputs": step_outputs or {}, **(state_extra or {})}
        outputs = executor._execute_step(
            step_name,
            config,
            context,
            object_store=None,
            pipeline_state=pipeline_state,
        )
        return config, outputs

    fixed = {"validate_input": {"validation_fixed_usd_path": "fixed.usda"}}
    config, _outputs = run_step("optimize_usd", {}, fixed)
    assert config["input_usd_path"] == "fixed.usda"
    config, _outputs = run_step("render_preview", {}, fixed)
    assert config["usd_path"] == "fixed.usda"

    config, _outputs = run_step(
        "generate_material_library",
        {},
        {"optimize_usd": {"optimized_usd_path": "optimized.usdc"}},
    )
    assert config["input_usd_path"] == "optimized.usdc"

    config, _outputs = run_step(
        "apply",
        {},
        {},
        extra_context={"step_configs": "not-a-dict"},
        state_extra={"optimize_usd_skipped_original_input": "original.usda"},
    )
    assert config["input_usd_path"] == "original.usda"
    assert resolver_calls[-1]["apply"] is config
    assert hydrate_calls[-1]["apply"] is config

    config, _outputs = run_step(
        "refine",
        {},
        {
            "optimize_usd": {"original_usd_path": "original.usda"},
            "restore_usd": {"restored_predictions_path": "restored.jsonl"},
        },
        extra_context={
            "pipeline_config": {"input": {"reference_images": ["reference.png"]}}
        },
    )
    assert config["input_usd_path"] == "original.usda"
    assert config["predictions_path"] == "restored.jsonl"
    assert config["judge"]["reference_images"] == ["reference.png"]

    config, _outputs = run_step(
        "identify_asset",
        {"vlm": {"backend": "mock"}},
        {"render_preview": {"rendered_preview_paths": ["preview.png"]}},
        extra_context={
            "path_resolver": SimpleNamespace(
                reference_images=[tmp_path / "scene_ref.png"]
            )
        },
    )
    assert config["composition_images"] == ["preview.png"]
    assert config["reference_images"] == [str(tmp_path / "scene_ref.png")]
    assert captured["identify_asset"][-1]["vlm_config"] == {"backend": "mock"}

    config, _outputs = run_step(
        "generate_reference_image",
        {},
        {
            "render_preview": {"rendered_preview_paths": ["preview.png"]},
            "identify_asset": {
                "identification": {"asset": "widget"},
                "image_gen_prompt": "make it",
            },
        },
    )
    assert config["rendered_preview_paths"] == ["preview.png"]
    assert config["identification"] == {"asset": "widget"}
    assert config["image_gen_prompt"] == "make it"

    config, _outputs = run_step(
        "build_dataset_prepare_dataset",
        {"reference_images": ["existing.png"]},
        {
            "generate_reference_image": {
                "generated_reference_image_paths": ["existing.png", "new.png"]
            }
        },
    )
    assert config["reference_images"] == ["existing.png", "new.png"]

    config, _outputs = run_step(
        "cluster_prims",
        {},
        {"build_dataset_prepare_dataset": {"dataset_jsonl_path": "prepared.jsonl"}},
    )
    assert config["dataset_path"] == "prepared.jsonl"
    assert config["working_dir"] == str(tmp_path)

    config, _outputs = run_step("cluster_prims", {}, {})
    assert config["dataset_path"].endswith("dataset.jsonl")
    assert config["working_dir"] == str(tmp_path)

    config, _outputs = run_step(
        "predict",
        {},
        {
            "cluster_prims": {
                "cluster_prims_ran": True,
                "dataset_representatives_path": "representatives.jsonl",
            }
        },
    )
    assert config["dataset"] == "representatives.jsonl"
    assert captured["predict"][-1]["original_prim_count"] == 0
    assert captured["predict"][-1]["num_prims"] == 0
    assert captured["predict"][-1]["num_images"] == 0

    config, _outputs = run_step("expand_cluster_predictions", {}, {})
    assert config["cluster_prims_ran"] is False
    assert config["predictions_path"].endswith("predictions.jsonl")
    assert config["cluster_map_path"].endswith("cluster_map.jsonl")

    config, _outputs = run_step(
        "expand_cluster_predictions",
        {},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
    )
    assert config["predictions_path"] == "benchmark.jsonl"

    config, _outputs = run_step(
        "expand_cluster_predictions",
        {},
        {"cluster_prims": {"cluster_prims_ran": True}},
    )
    assert config["predictions_path"].endswith("predictions.jsonl")
    assert config["cluster_map_path"].endswith("cluster_map.jsonl")

    config, _outputs = run_step("evaluate", {}, {})
    assert config["predictions_path"].endswith("predictions.jsonl")
    assert config["dataset_path"].endswith("dataset.jsonl")
    assert config["output_dir"].endswith("evaluation")

    for source_step in ("validate_predictions", "predict", "benchmark"):
        config, _outputs = run_step(
            "evaluate",
            {},
            {source_step: {"predictions_path": f"{source_step}.jsonl"}},
        )
        assert config["predictions_path"] == f"{source_step}.jsonl"

    config, _outputs = run_step(
        "render",
        {},
        {"refine": {"output_usd_path": "refined.usda"}},
    )
    assert config["input_usd_path"] == "refined.usda"

    config, _outputs = run_step(
        "harmonize_predictions",
        {},
        {
            "benchmark": {"predictions_path": "benchmark.jsonl"},
            "optimize_usd": {"optimized_usd_path": "optimized.usdc"},
        },
    )
    assert config["predictions_path"] == "benchmark.jsonl"
    assert config["optimized_usd_path"] == "optimized.usdc"

    config, _outputs = run_step("harmonize_predictions", {}, {})
    assert config["predictions_path"].endswith("predictions.jsonl")

    config, _outputs = run_step(
        "validate_predictions",
        {},
        {"harmonize_predictions": {"predictions_path": "harmonized.jsonl"}},
    )
    assert config["predictions_path"] == "harmonized.jsonl"

    for step_outputs in (
        {"predict": {"predictions_path": "predictions.jsonl"}},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
        {},
    ):
        config, _outputs = run_step("validate_predictions", {}, step_outputs)
        assert config["predictions_path"].endswith(".jsonl")

    config, _outputs = run_step(
        "validate_predictions",
        {"predictions_path": "provided.jsonl"},
        {"harmonize_predictions": {"predictions_path": "harmonized.jsonl"}},
    )
    assert config["predictions_path"] == "provided.jsonl"

    config, _outputs = run_step(
        "restore_usd",
        {},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
        extra_context={
            "path_resolver": SimpleNamespace(input_usd=tmp_path / "original.usda")
        },
    )
    assert config["original_usd_path"] == str(tmp_path / "original.usda")
    assert config["predictions_path"] == "benchmark.jsonl"
    assert config["output_predictions_path"].endswith("restored_predictions.jsonl")

    config, _outputs = run_step(
        "restore_usd",
        {},
        {"validate_predictions": {"predictions_path": "validated.jsonl"}},
        state_extra={"optimization_metadata": {"restore": True}},
    )
    assert config["predictions_path"] == "validated.jsonl"
    assert config["optimization_metadata"] == {"restore": True}

    config, _outputs = run_step(
        "restore_usd",
        {},
        {"predict": {"predictions_path": "predictions.jsonl"}},
    )
    assert config["predictions_path"] == "predictions.jsonl"

    config, _outputs = run_step(
        "validate_output",
        {},
        {
            "apply": {"output_usd_path": "applied.usda"},
            "validate_input": {"validation_fixed_usd_path": "fixed.usda"},
        },
        extra_context={
            "path_resolver": SimpleNamespace(resolve_path=lambda path: tmp_path / path),
            "config": {"input": {"usd_path": "raw.usda"}},
        },
    )
    assert config["input_usd_path"] == "applied.usda"
    assert config["original_usd_path"] == "fixed.usda"

    config, _outputs = run_step(
        "validate_output",
        {},
        {
            "refine": {"final_output_path": "final.usda"},
            "optimize_usd": {"original_usd_path": "original.usda"},
            "validate_input": {"validation_result": {"issues": [1, 2]}},
        },
    )
    assert config["input_usd_path"] == "final.usda"
    assert config["original_usd_path"] == "original.usda"
    assert config["baseline_validation"] == {"issues": [1, 2]}

    with pytest.raises(ValueError, match="Unknown step"):
        run_step("unknown_step")


@pytest.mark.asyncio
async def test_execute_step_async_autowires_dispatch_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    captured: dict[str, list[dict]] = {}
    _patch_fake_workflows(monkeypatch, captured)
    monkeypatch.setattr(
        executor,
        "_resolve_generated_material_fallbacks_for_apply",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        executor,
        "_hydrate_simready_materials_for_apply",
        lambda *_args, **_kwargs: None,
    )
    metadata_dir = tmp_path / "optimized"
    metadata_dir.mkdir()
    (metadata_dir / "optimized_input.metadata.json").write_text(
        json.dumps({"restorable": True}),
        encoding="utf-8",
    )

    base_context = {
        "working_dir": str(tmp_path),
        "event_listener": MagicMock(),
        "original_prim_count": 0,
        "num_prims": 0,
        "num_images": 0,
    }

    async def run_step(
        step_name: str,
        step_config: dict[str, object] | None = None,
        step_outputs: dict[str, object] | None = None,
        *,
        extra_context: dict[str, object] | None = None,
        state_extra: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        config = step_config or {}
        context = {**base_context, **(extra_context or {})}
        pipeline_state = {"step_outputs": step_outputs or {}, **(state_extra or {})}
        outputs = await executor._aexecute_step(
            step_name,
            config,
            context,
            object_store=None,
            pipeline_state=pipeline_state,
        )
        return config, outputs, pipeline_state

    fixed = {"validate_input": {"validation_fixed_usd_path": "fixed.usda"}}
    config, _outputs, _state = await run_step("optimize_usd", {}, fixed)
    assert config["input_usd_path"] == "fixed.usda"
    config, _outputs, _state = await run_step("render_preview", {}, fixed)
    assert config["usd_path"] == "fixed.usda"

    config, _outputs, _state = await run_step(
        "generate_material_library",
        {},
        {"optimize_usd": {"optimized_usd_path": "optimized.usdc"}},
    )
    assert config["input_usd_path"] == "optimized.usdc"

    config, _outputs, _state = await run_step(
        "apply",
        {},
        {
            "optimize_usd": {"original_usd_path": "original.usda"},
            "restore_usd": {"restored_predictions_path": "restored.jsonl"},
        },
        extra_context={"step_configs": "bad"},
        state_extra={"optimize_usd_skipped_original_input": "skipped.usda"},
    )
    assert config["input_usd_path"] == "original.usda"
    assert config["predictions_path"] == "restored.jsonl"

    config, _outputs, _state = await run_step(
        "refine",
        {},
        {},
        extra_context={
            "pipeline_config": {"input": {"reference_images": ["reference.png"]}}
        },
        state_extra={"optimize_usd_skipped_original_input": "skipped.usda"},
    )
    assert config["input_usd_path"] == "skipped.usda"
    assert config["judge"]["reference_images"] == ["reference.png"]

    config, _outputs, _state = await run_step(
        "identify_asset",
        {"vlm": {"backend": "mock"}},
        {"render_preview": {"rendered_preview_paths": ["preview.png"]}},
        extra_context={
            "path_resolver": SimpleNamespace(
                reference_images=[tmp_path / "scene_ref.png"]
            )
        },
    )
    assert config["composition_images"] == ["preview.png"]
    assert captured["identify_asset"][-1]["vlm_config"] == {"backend": "mock"}

    config, _outputs, _state = await run_step(
        "generate_reference_image",
        {},
        {
            "render_preview": {"rendered_preview_paths": ["preview.png"]},
            "identify_asset": {
                "identification": {"asset": "widget"},
                "image_gen_prompt": "make it",
            },
        },
    )
    assert config["image_gen_prompt"] == "make it"

    config, _outputs, _state = await run_step(
        "build_dataset_prepare_dataset",
        {"reference_images": ["existing.png"]},
        {
            "generate_reference_image": {
                "generated_reference_image_paths": ["existing.png", "new.png"]
            }
        },
    )
    assert config["reference_images"] == ["existing.png", "new.png"]

    config, _outputs, _state = await run_step(
        "cluster_prims",
        {},
        {"build_dataset_prepare_dataset": {"dataset_jsonl_path": "prepared.jsonl"}},
    )
    assert config["dataset_path"] == "prepared.jsonl"

    config, _outputs, _state = await run_step("cluster_prims", {}, {})
    assert config["dataset_path"].endswith("dataset.jsonl")

    config, _outputs, _state = await run_step(
        "benchmark",
        {},
        {
            "cluster_prims": {
                "cluster_prims_ran": True,
                "dataset_representatives_path": "representatives.jsonl",
            }
        },
    )
    assert config["dataset"] == "representatives.jsonl"
    assert captured["benchmark"][-1]["original_prim_count"] == 0

    config, _outputs, _state = await run_step(
        "predict",
        {},
        {
            "cluster_prims": {
                "cluster_prims_ran": True,
                "dataset_representatives_path": "representatives.jsonl",
            }
        },
    )
    assert config["dataset"] == "representatives.jsonl"

    config, _outputs, _state = await run_step(
        "expand_cluster_predictions",
        {},
        {"cluster_prims": {"cluster_prims_ran": False}},
    )
    assert config["cluster_prims_ran"] is False

    config, _outputs, _state = await run_step(
        "expand_cluster_predictions",
        {},
        {"cluster_prims": {"cluster_prims_ran": True}},
    )
    assert config["predictions_path"].endswith("predictions.jsonl")
    assert config["cluster_map_path"].endswith("cluster_map.jsonl")

    config, _outputs, _state = await run_step(
        "expand_cluster_predictions",
        {},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
    )
    assert config["predictions_path"] == "benchmark.jsonl"

    config, _outputs, _state = await run_step(
        "evaluate",
        {"report": {"image_max_size": 512, "image_format": "webp"}},
        {},
    )
    assert config["output_dir"].endswith("evaluation")
    assert captured["evaluate"][-1]["report_image_max_size"] == 512
    assert captured["evaluate"][-1]["report_image_format"] == "webp"

    for source_step in (
        "harmonize_predictions",
        "validate_predictions",
        "predict",
        "benchmark",
    ):
        config, _outputs, _state = await run_step(
            "evaluate",
            {},
            {source_step: {"predictions_path": f"{source_step}.jsonl"}},
        )
        assert config["predictions_path"] == f"{source_step}.jsonl"

    config, _outputs, _state = await run_step(
        "evaluate",
        {},
        {
            "build_dataset_prepare_dataset": {
                "dataset_jsonl_path": "prepared.jsonl",
                "vlm_prompt_path": "prompt.txt",
            }
        },
    )
    assert config["dataset_path"] == "prepared.jsonl"
    assert config["system_prompt_file"] == "prompt.txt"

    config, _outputs, _state = await run_step(
        "render",
        {},
        {"apply": {"output_usd_path": "applied.usda"}},
    )
    assert config["input_usd_path"] == "applied.usda"

    config, _outputs, _state = await run_step(
        "render",
        {},
        {"refine": {"output_usd_path": "refined.usda"}},
    )
    assert config["input_usd_path"] == "refined.usda"

    config, _outputs, _state = await run_step(
        "harmonize_predictions",
        {},
        {"predict": {"predictions_path": "predictions.jsonl"}},
    )
    assert config["predictions_path"] == "predictions.jsonl"

    config, _outputs, _state = await run_step("harmonize_predictions", {}, {})
    assert config["predictions_path"].endswith("predictions.jsonl")

    config, _outputs, _state = await run_step(
        "validate_predictions",
        {},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
    )
    assert config["predictions_path"] == "benchmark.jsonl"

    config, _outputs, _state = await run_step(
        "validate_predictions",
        {},
        {"predict": {"predictions_path": "predictions.jsonl"}},
    )
    assert config["predictions_path"] == "predictions.jsonl"

    config, _outputs, _state = await run_step("validate_predictions", {}, {})
    assert config["predictions_path"].endswith("predictions.jsonl")

    config, _outputs, _state = await run_step(
        "restore_usd",
        {},
        {"predict": {"predictions_path": "predictions.jsonl"}},
        extra_context={
            "path_resolver": SimpleNamespace(input_usd=tmp_path / "original.usda")
        },
    )
    assert config["optimization_metadata"] == {"restorable": True}

    config, _outputs, _state = await run_step(
        "restore_usd",
        {},
        {"harmonize_predictions": {"predictions_path": "harmonized.jsonl"}},
    )
    assert config["predictions_path"] == "harmonized.jsonl"

    config, _outputs, _state = await run_step(
        "restore_usd",
        {},
        {"validate_predictions": {"predictions_path": "validated.jsonl"}},
    )
    assert config["predictions_path"] == "validated.jsonl"

    config, _outputs, _state = await run_step(
        "restore_usd",
        {},
        {"benchmark": {"predictions_path": "benchmark.jsonl"}},
    )
    assert config["predictions_path"] == "benchmark.jsonl"

    no_metadata_dir = tmp_path / "no_metadata"
    config, _outputs, _state = await run_step(
        "restore_usd",
        {},
        {},
        extra_context={"working_dir": str(no_metadata_dir)},
    )
    assert config["predictions_path"].endswith("predictions.jsonl")

    config, _outputs, _state = await run_step(
        "validate_output",
        {},
        {
            "apply": {"output_usd_path": "applied.usda"},
            "validate_input": {"validation_fixed_usd_path": "fixed.usda"},
        },
    )
    assert config["original_usd_path"] == "fixed.usda"

    config, _outputs, _state = await run_step(
        "validate_output",
        {},
        {
            "refine": {"final_output_path": "final.usda"},
            "optimize_usd": {"original_usd_path": "original.usda"},
            "validate_input": {"validation_result": {"issues": [1]}},
        },
    )
    assert config["input_usd_path"] == "final.usda"
    assert config["original_usd_path"] == "original.usda"
    assert config["baseline_validation"] == {"issues": [1]}

    config, _outputs, _state = await run_step(
        "validate_output",
        {},
        {},
        extra_context={
            "path_resolver": SimpleNamespace(resolve_path=lambda path: tmp_path / path),
            "config": {"input": {"usd_path": "raw.usda"}},
        },
    )
    assert config["original_usd_path"] == str(tmp_path / "raw.usda")

    apply_config: dict[str, object] = {"existing": True}
    config, _outputs, _state = await run_step(
        "apply",
        apply_config,
        {},
        extra_context={"step_configs": {"apply": apply_config}},
    )
    assert config is apply_config

    config, _outputs, state = await run_step(
        "optimize_usd",
        {},
        {},
    )
    assert state["optimization_metadata"] == {"prim_map": {}}

    with pytest.raises(ValueError, match="Unknown step"):
        await run_step("unknown_step")

    import material_agent.workflows as workflows

    monkeypatch.setattr(
        workflows,
        "create_prediction_workflow_from_config",
        lambda: _FakeWorkflow(
            "predict",
            {"error": "bad", "failed_task": "PredictTask"},
            captured,
        ),
    )
    with pytest.raises(RuntimeError, match="PredictTask"):
        await run_step("predict", {}, {})


def test_default_fallback_heuristic_ignores_generated_material_list() -> None:
    executor = UnifiedPipelineExecutorTask()
    prediction = {
        "id": "/World/front_panel_board",
        "materials": {
            "original_response": (
                "The highlighted part is an internal electronic component.\n\n"
                "Looking at the available materials:\n"
                "- Dark Grey Matte Rubber is for the chamber seal.\n"
                "- Brushed Silver Aluminum is for the rotor.\n\n"
                "Since this part is a PCB-like component, use the default library."
            )
        },
    }
    default_entries = [
        {"name": "Rubber Black Matte"},
        {"name": "Plastic Black"},
    ]

    assert (
        executor._heuristic_default_material(prediction, default_entries)
        == "Plastic Black"
    )


def test_pipeline_predict_vlm_config_ignores_non_mapping_steps() -> None:
    context = {
        "step_configs": "not-a-mapping",
        "pipeline_config": {"steps": ["not", "a", "mapping"]},
    }

    assert _pipeline_predict_vlm_config(context) == {}


def test_build_child_config_dict_is_yaml_equivalent_and_isolated() -> None:
    source: dict[str, Any] = {
        "renderer": {
            "backend": "remote",
            "_unified_config": object(),
            "_rendering_modes_config": object(),
        },
        "path": Path("/tmp/output.usd"),
        "mode": _Mode.FAST,
        "flags": {"b", "a"},
        "vlm": {"api_key": "x", "nested": [{"token": "abc"}]},
    }

    loaded = _build_child_config_dict(source)
    loaded["vlm"]["nested"][0]["token"] = "mutated"

    assert loaded["renderer"] == {"backend": "remote"}
    assert loaded["path"] == "/tmp/output.usd"
    assert loaded["mode"] == "fast"
    assert loaded["flags"] == ["a", "b"]
    assert loaded["vlm"]["api_key"] == "x"
    assert source["vlm"]["nested"][0]["token"] == "abc"


def test_runtime_context_clones_containers_and_preserves_runtime_leaves() -> None:
    client = _NonCopyableRuntimeClient()
    shared: dict[str, Any] = {"client": client}
    recursive: list[Any] = []
    predict_config: dict[str, Any] = {
        "renderer": {"_runtime": shared},
        "runtime_alias": shared,
        "recursive": recursive,
        "frozen": frozenset({"a", "b"}),
        "nested": [{"enabled": True}],
    }
    recursive.append(predict_config)
    source_step_configs: dict[str, dict[str, Any]] = {
        "predict": predict_config,
    }
    source_context: dict[str, Any] = {"step_configs": source_step_configs}

    runtime_context = _build_runtime_pipeline_context(source_context)
    runtime_step_configs = runtime_context["step_configs"]
    runtime_step_configs["predict"]["nested"][0]["enabled"] = False

    assert runtime_context is not source_context
    assert runtime_step_configs is not source_step_configs
    assert runtime_step_configs["predict"] is not source_step_configs["predict"]
    assert source_step_configs["predict"]["nested"] == [{"enabled": True}]
    assert runtime_step_configs["predict"]["renderer"]["_runtime"]["client"] is client
    assert (
        runtime_step_configs["predict"]["renderer"]["_runtime"]
        is runtime_step_configs["predict"]["runtime_alias"]
    )
    assert (
        runtime_step_configs["predict"]["recursive"][0]
        is runtime_step_configs["predict"]
    )
    assert runtime_step_configs["predict"]["frozen"] == frozenset({"a", "b"})
    assert source_context["step_configs"] is source_step_configs


def test_remove_legacy_pipeline_temp_does_not_follow_symlink(tmp_path: Path) -> None:
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_secret = outside_dir / "secret.yaml"
    outside_secret.write_text("api_key: ZXCV713SECRETQWER", encoding="utf-8")
    (working_dir / ".pipeline_temp").symlink_to(outside_dir, target_is_directory=True)

    remove_legacy_pipeline_temp(working_dir)

    assert not (working_dir / ".pipeline_temp").exists()
    assert outside_secret.read_text(encoding="utf-8") == "api_key: ZXCV713SECRETQWER"


@pytest.mark.parametrize(
    ("step_name", "result", "expected"),
    [
        (
            "build_dataset_prepare_dataset",
            {
                "dataset_path": "dataset",
                "dataset_jsonl_path": "dataset.jsonl",
                "vlm_prompt_path": "prompt.txt",
                "num_entries": 7,
            },
            {
                "dataset_path": "dataset",
                "dataset_jsonl_path": "dataset.jsonl",
                "vlm_prompt_path": "prompt.txt",
                "num_entries": 7,
            },
        ),
        (
            "build_dataset_pdf_vectorstore",
            {"output_dir": "vec"},
            {"vectorstore_dir": "vec"},
        ),
        (
            "build_dataset_usd",
            {"output_dir": "usd_data", "num_prims": 4, "num_images": 9},
            {
                "output_dir": "usd_data",
                "usd_dataset_dir": "usd_data",
                "num_prims": 4,
                "num_images": 9,
            },
        ),
        (
            "cluster_prims",
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": True,
                "cluster_summary_path": "clusters/cluster_summary.json",
                "cluster_report_path": "clusters/cluster_report.html",
                "cluster_total_prims": 117,
                "cluster_count": 88,
                "cluster_representative_count": 88,
                "cluster_reduction_percent": 24.786,
                "cluster_multi_member_count": 13,
                "cluster_singleton_count": 75,
                "cluster_max_size": 25,
                "cluster_capped_count": 0,
            },
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": True,
                "cluster_summary_path": "clusters/cluster_summary.json",
                "cluster_report_path": "clusters/cluster_report.html",
                "cluster_total_prims": 117,
                "cluster_count": 88,
                "cluster_representative_count": 88,
                "cluster_reduction_percent": 24.786,
                "cluster_multi_member_count": 13,
                "cluster_singleton_count": 75,
                "cluster_max_size": 25,
                "cluster_capped_count": 0,
            },
        ),
        (
            "cluster_prims",
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
            },
            {
                "cluster_map_path": "clusters/map.jsonl",
                "dataset_representatives_path": "dataset/reps.jsonl",
                "cluster_prims_ran": False,
                "cluster_summary_path": None,
                "cluster_report_path": None,
                "cluster_total_prims": 0,
                "cluster_count": 0,
                "cluster_representative_count": 0,
                "cluster_reduction_percent": 0.0,
                "cluster_multi_member_count": 0,
                "cluster_singleton_count": 0,
                "cluster_max_size": None,
                "cluster_capped_count": 0,
            },
        ),
        (
            "predict",
            {"predictions_path": "preds.jsonl", "predictions_count": 5},
            {"predictions_path": "preds.jsonl", "predictions_count": 5},
        ),
        (
            "expand_cluster_predictions",
            {"predictions_path": "expanded.jsonl"},
            {"predictions_path": "expanded.jsonl"},
        ),
        (
            "harmonize_predictions",
            {
                "predictions_path": "harmonized.jsonl",
                "harmonized_count": 3,
                "remap": {"a": "b"},
            },
            {
                "predictions_path": "harmonized.jsonl",
                "harmonized_count": 3,
                "remap": {"a": "b"},
            },
        ),
        (
            "evaluate",
            {
                "evaluation_path": "evaluation.json",
                "html_report_path": "report.html",
                "metrics": {"acc": 1.0},
            },
            {
                "evaluation_path": "evaluation.json",
                "html_report_path": "report.html",
                "metrics": {"acc": 1.0},
            },
        ),
        (
            "optimize_usd",
            {
                "optimized_usd_path": "optimized.usdc",
                "optimization_success": True,
                "original_usd_path": "original.usd",
                "original_prim_count": 10,
                "optimization_metadata": {"map": {}},
            },
            {
                "optimized_usd_path": "optimized.usdc",
                "optimization_success": True,
                "original_usd_path": "original.usd",
                "original_prim_count": 10,
                "optimization_metadata": {"map": {}},
            },
        ),
        (
            "apply",
            {"output_usd_path": "applied.usd", "materials_applied": 4},
            {"output_usd_path": "applied.usd", "materials_applied": 4},
        ),
        (
            "apply",
            {
                "output_usd_path": "covered.usd",
                "materials_applied": 4,
                "assignment_stats": {
                    "bound_prim_ids": ["/World/Part"],
                    "unbound_prim_ids": [],
                },
            },
            {
                "output_usd_path": "covered.usd",
                "materials_applied": 4,
                "assignment_stats": {
                    "bound_prim_ids": ["/World/Part"],
                    "unbound_prim_ids": [],
                },
            },
        ),
        (
            "apply",
            {
                "output_usd_path": "profiled.usd",
                "materials_applied": 2,
                "material_profile_result": {"resolved_profile": "openpbr_materialx"},
                "resolved_material_profile": "openpbr_materialx",
                "material_profile_warnings": [],
                "material_profile_errors": [],
            },
            {
                "output_usd_path": "profiled.usd",
                "materials_applied": 2,
                "material_profile_result": {"resolved_profile": "openpbr_materialx"},
                "resolved_material_profile": "openpbr_materialx",
                "material_profile_warnings": [],
                "material_profile_errors": [],
            },
        ),
        (
            "validate_input",
            {
                "validation_result": {"issues": []},
                "validation_summary": "ok",
                "validation_is_valid": True,
                "validation_fixed_usd_path": None,
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
            {
                "validation_result": {"issues": []},
                "validation_summary": "ok",
                "validation_is_valid": True,
                "validation_fixed_usd_path": None,
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
        ),
        (
            "refine",
            {"final_output_path": "refined.usd"},
            {
                "output_usd_path": "refined.usd",
                "final_output_path": "refined.usd",
            },
        ),
        (
            "render",
            {
                "rendered_image_paths": ["a.png"],
                "rendered_image_path": "a.png",
                "flattened_usd_path": "flat.usd",
            },
            {
                "rendered_image_paths": ["a.png"],
                "rendered_image_path": "a.png",
                "flattened_usd_path": "flat.usd",
            },
        ),
        (
            "validate_output",
            {
                "validation_result": {"ok": True},
                "validation_summary": "done",
                "validation_is_valid": True,
                "validation_regression": False,
                "validation_new_issues": [],
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
            {
                "validation_result": {"ok": True},
                "validation_summary": "done",
                "validation_is_valid": True,
                "validation_regression": False,
                "validation_new_issues": [],
                "validation_skipped": None,
                "validation_error": None,
                "validation_success": True,
            },
        ),
        (
            "restore_usd",
            {
                "restored_usd_path": "restored.usd",
                "restored_predictions_path": "restored.jsonl",
                "restore_success": True,
                "predictions_count": 12,
                "restore_stats": {
                    "restored_prim_sources": {"/Original": "/Optimized"},
                    "expected_target_count": 1,
                    "mapping_complete": True,
                },
            },
            {
                "restored_usd_path": "restored.usd",
                "restored_predictions_path": "restored.jsonl",
                "restore_success": True,
                "predictions_count": 12,
                "restore_stats": {
                    "restored_prim_sources": {"/Original": "/Optimized"},
                    "expected_target_count": 1,
                    "mapping_complete": True,
                },
            },
        ),
    ],
)
def test_extract_step_outputs_maps_expected_keys(
    step_name: str,
    result: dict[str, object],
    expected: dict[str, object],
) -> None:
    executor = UnifiedPipelineExecutorTask()

    assert executor._extract_step_outputs(step_name, result) == expected


def test_run_cleans_outputs_and_updates_context(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    (working_dir / "stale.txt").write_text("old", encoding="utf-8")

    output_usd = tmp_path / "output" / "scene.usd"
    output_usd.parent.mkdir()
    output_usd.write_text("usd", encoding="utf-8")
    (output_usd.parent / "scene_flat.usd").write_text("flat", encoding="utf-8")
    renders_dir = output_usd.parent / "renders"
    renders_dir.mkdir()
    (renders_dir / "preview.png").write_text("x", encoding="utf-8")

    listener = MagicMock()
    context = {
        "working_dir": str(working_dir),
        "working_dir_base": str(tmp_path),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"enabled": True},
            "build_dataset_usd": {"enabled": True},
        },
        "clean": True,
        "session_id": "session-1",
        "project_name": "project-1",
        "path_resolver": SimpleNamespace(output_usd=output_usd),
    }
    pipeline_state = {
        "session_id": "session-1",
        "project_name": "project-1",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    def step_outputs(step_name, step_config, ctx, object_store, state):
        if step_name == "optimize_usd":
            return {
                "optimized_usd_path": str(tmp_path / "optimized.usdc"),
                "original_usd_path": str(tmp_path / "input.usd"),
                "original_prim_count": 12,
            }
        return {
            "output_dir": str(tmp_path / "dataset"),
            "num_prims": 8,
            "num_images": 16,
        }

    executor._execute_step = MagicMock(side_effect=step_outputs)

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 12
    assert result["num_prims"] == 8
    assert result["num_images"] == 16
    assert not (working_dir / "stale.txt").exists()
    assert not output_usd.exists()
    assert not (output_usd.parent / "scene_flat.usd").exists()
    assert not renders_dir.exists()
    assert result["pipeline_results"]["optimize_usd"]["original_prim_count"] == 12
    assert result["pipeline_results"]["build_dataset_usd"]["num_prims"] == 8
    assert context["pipeline_results"] is result["pipeline_results"]
    assert context["pipeline_state"] == "completed"


def test_run_generated_activation_is_local_when_step_configs_are_reused(
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    shared_step_configs: dict[str, dict[str, Any]] = {
        "predict": {"nested": {"owner": "source"}},
        "build_dataset_prepare_dataset": {"materials_list": ["Source"]},
        "validate_predictions": {"material_names": ["Source"]},
        "harmonize_predictions": {"material_names": ["Source"]},
        "apply": {"materials_mapping": {"Source": "/World/Looks/Source"}},
        "refine": {"apply": {"materials_mapping": {"Source": "source"}}},
    }

    def load_state(
        _working_dir: str,
        session_id: str,
        _project_name: str | None,
        _resume: bool,
    ) -> dict[str, Any]:
        material_name = f"Generated {session_id}"
        return {
            "session_id": session_id,
            "project_name": None,
            "completed_steps": [],
            "failed_steps": [],
            "step_errors": {},
            "step_outputs": {
                "generate_material_library": {
                    "generated_materials_data": {
                        "library_path": f"/{session_id}.usda",
                        "entries": [
                            {
                                "name": material_name,
                                "binding": f"/World/Looks/{session_id}",
                            }
                        ],
                    }
                }
            },
            "current_step": None,
        }

    overlap = threading.Barrier(2)

    def execute_step(
        _step_name: str,
        step_config: dict[str, Any],
        runtime_context: dict[str, Any],
        _object_store: Any,
        _pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        assert step_config is runtime_context["step_configs"]["predict"]
        overlap.wait(timeout=5)
        step_config["nested"]["owner"] = runtime_context["session_id"]
        return {}

    executor._execute_step = MagicMock(side_effect=execute_step)
    source_contexts = [
        {
            "working_dir": str(tmp_path / session_id),
            "steps_to_run": ["predict"],
            "step_configs": shared_step_configs,
            "session_id": session_id,
        }
        for session_id in ("first", "second")
    ]

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            side_effect=load_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=MagicMock(),
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(executor.run, source_contexts))

    assert shared_step_configs["predict"]["nested"] == {"owner": "source"}
    assert shared_step_configs["build_dataset_prepare_dataset"] == {
        "materials_list": ["Source"]
    }
    assert shared_step_configs["apply"] == {
        "materials_mapping": {"Source": "/World/Looks/Source"}
    }
    assert all(
        source["step_configs"] is shared_step_configs for source in source_contexts
    )
    assert results[0]["step_configs"]["predict"]["nested"] == {"owner": "first"}
    assert results[1]["step_configs"]["predict"]["nested"] == {"owner": "second"}
    assert "Generated first" in results[0]["step_configs"]["apply"]["materials_mapping"]
    assert (
        "Generated second" in results[1]["step_configs"]["apply"]["materials_mapping"]
    )
    assert (
        "Generated second"
        not in results[0]["step_configs"]["apply"]["materials_mapping"]
    )


@pytest.mark.asyncio
async def test_arun_created_activation_is_local_during_overlapping_runs(
    tmp_path: Path,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    shared_step_configs: dict[str, dict[str, Any]] = {
        "predict": {"nested": {"owner": "source"}},
        "apply": {},
        "refine": {},
    }

    def load_state(
        _working_dir: str,
        session_id: str,
        _project_name: str | None,
        _resume: bool,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "project_name": None,
            "completed_steps": [],
            "failed_steps": [],
            "step_errors": {},
            "step_outputs": {
                "create_materials": {
                    "created_materials_data": {
                        "library_path": f"/{session_id}.usda",
                        "entries": [
                            {
                                "name": f"Created {session_id}",
                                "binding": f"/World/Looks/{session_id}",
                            }
                        ],
                    }
                }
            },
            "current_step": None,
        }

    both_started = asyncio.Event()
    started = 0

    async def execute_step(
        _step_name: str,
        step_config: dict[str, Any],
        runtime_context: dict[str, Any],
        _object_store: Any,
        _pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        step_config["nested"]["owner"] = runtime_context["session_id"]
        return {}

    executor._aexecute_step = AsyncMock(side_effect=execute_step)
    source_contexts = [
        {
            "working_dir": str(tmp_path / session_id),
            "steps_to_run": ["predict"],
            "step_configs": shared_step_configs,
            "session_id": session_id,
        }
        for session_id in ("first", "second")
    ]

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            side_effect=load_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=MagicMock(),
        ),
    ):
        results = await asyncio.gather(
            *(executor.arun(source_context) for source_context in source_contexts)
        )

    assert shared_step_configs == {
        "predict": {"nested": {"owner": "source"}},
        "apply": {},
        "refine": {},
    }
    assert all(
        source["step_configs"] is shared_step_configs for source in source_contexts
    )
    assert results[0]["step_configs"]["predict"]["nested"] == {"owner": "first"}
    assert results[1]["step_configs"]["predict"]["nested"] == {"owner": "second"}
    assert "Created first" in results[0]["step_configs"]["apply"]["materials_mapping"]
    assert "Created second" in results[1]["step_configs"]["apply"]["materials_mapping"]


def test_run_resume_activation_and_in_step_cancel_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    listener = MagicMock()

    resume_executor = UnifiedPipelineExecutorTask()
    resume_executor._execute_step = MagicMock()
    resume_dir = tmp_path / "resume"
    legacy_temp = resume_dir / ".pipeline_temp"
    legacy_temp.mkdir(parents=True)
    (legacy_temp / "predict_config_legacy.yaml").write_text(
        "api_key: ZXCV713SECRETQWER", encoding="utf-8"
    )
    resume_state = {
        "session_id": "resume",
        "project_name": "project",
        "completed_steps": ["predict"],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=resume_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = resume_executor.run(
            {
                "working_dir": str(resume_dir),
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {"enabled": True}},
                "resume": True,
            }
        )
    resume_executor._execute_step.assert_not_called()
    assert result["pipeline_state"] == "completed"
    assert not legacy_temp.exists()
    assert "Removed retained legacy .pipeline_temp" in caplog.text
    assert "ZXCV713SECRETQWER" not in caplog.text
    activation_executor = UnifiedPipelineExecutorTask()
    activation_calls: list[dict | None] = []

    def monkey_activate(outputs, _context, _configs):
        return activation_calls.append(outputs)

    activation_executor._activate_generated_material_library = MagicMock(
        side_effect=monkey_activate
    )
    activation_executor._execute_step = MagicMock(
        return_value={"generated_materials_data": {"entries": []}}
    )
    activation_state = {
        "session_id": "activation",
        "project_name": "project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=activation_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        activation_executor.run(
            {
                "working_dir": str(tmp_path / "activation"),
                "steps_to_run": ["generate_material_library"],
                "step_configs": {"generate_material_library": {"enabled": True}},
            }
        )
    assert activation_calls[-1] == {"generated_materials_data": {"entries": []}}

    cancel_executor = UnifiedPipelineExecutorTask()
    cancel_executor._execute_step = MagicMock(side_effect=asyncio.CancelledError())
    cancel_state = {
        "session_id": "cancel",
        "project_name": "project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    cancel_dir = tmp_path / "cancel"
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=cancel_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        cancel_executor.run(
            {
                "working_dir": str(cancel_dir),
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {"enabled": True}},
            }
        )
    saved = json.loads((cancel_dir / ".pipeline_state.json").read_text())
    assert saved["current_step"] is None


@pytest.mark.asyncio
async def test_arun_resume_removes_legacy_pipeline_temp(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    executor._aexecute_step = AsyncMock()
    working_dir = tmp_path / "async-resume"
    legacy_temp = working_dir / ".pipeline_temp"
    legacy_temp.mkdir(parents=True)
    (legacy_temp / "predict_config_legacy.yaml").write_text(
        "api_key: ZXCV713SECRETQWER", encoding="utf-8"
    )
    pipeline_state = {
        "session_id": "async-resume",
        "project_name": "project",
        "completed_steps": ["predict"],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=MagicMock(),
        ),
    ):
        result = await executor.arun(
            {
                "working_dir": str(working_dir),
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {"enabled": True}},
                "resume": True,
            }
        )
    assert "Removed retained legacy .pipeline_temp" in caplog.text
    assert "ZXCV713SECRETQWER" not in caplog.text

    executor._aexecute_step.assert_not_awaited()
    assert result["pipeline_state"] == "completed"
    assert not legacy_temp.exists()


def test_run_skips_failed_optimize_usd_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    working_dir = tmp_path / "work"
    listener = MagicMock()
    event_listener = MagicMock()
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"input_usd_path": str(tmp_path / "input.usd")},
            "build_dataset_usd": {"usd_path": str(tmp_path / "scene.usd")},
        },
        "session_id": "session-2",
        "project_name": "project-2",
        "event_listener": event_listener,
    }
    pipeline_state = {
        "session_id": "session-2",
        "project_name": "project-2",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }

    secret = "material-optimizer-failure-api-key-713"

    def execute_step(step_name, step_config, ctx, object_store, state):
        if step_name == "optimize_usd":
            raise RuntimeError(f"provider failed with api_key={secret}")
        return {"output_dir": "dataset", "num_prims": 5, "num_images": 9}

    executor._execute_step = MagicMock(side_effect=execute_step)

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    state_file = working_dir / ".pipeline_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8"))

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 5
    assert result["pipeline_results"]["build_dataset_usd"]["num_images"] == 9
    assert saved["optimize_usd_skipped_original_input"] == str(tmp_path / "input.usd")
    assert saved["completed_steps"] == ["build_dataset_usd"]
    event_listener.event.assert_any_call(
        "step.skipped",
        {
            "step_name": "optimize_usd",
            "reason": ("optimize_usd failed: RuntimeError during step execution"),
        },
    )
    observable = caplog.text + repr(event_listener.mock_calls) + json.dumps(saved)
    assert secret not in observable


def test_run_records_step_error_for_failed_step(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-error",
        "project_name": "project-error",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    secret = "material-provider-failure-api-key-713"
    executor._execute_step = MagicMock(
        side_effect=RuntimeError(f"provider failed with api_key={secret}")
    )
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "session-error",
        "project_name": "project-error",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(
            RuntimeError,
            match=(
                "Pipeline failed at step 'predict': RuntimeError during step execution"
            ),
        ) as exc_info,
    ):
        executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert saved["failed_steps"] == ["predict"]
    assert saved["step_errors"] == {"predict": "RuntimeError during step execution"}
    assert saved["current_step"] is None
    listener.event.assert_called_with(
        "step.failed",
        {
            "step_name": "predict",
            "error": "RuntimeError during step execution",
        },
    )
    observable = caplog.text + str(exc_info.value) + repr(listener.mock_calls)
    observable += json.dumps(saved)
    assert secret not in observable
    assert exc_info.value.__cause__ is None
    failure_records = [
        record
        for record in caplog.records
        if "RuntimeError during step execution" in record.getMessage()
    ]
    assert failure_records
    assert all(record.exc_info is None for record in failure_records)


def test_run_clears_prior_step_error_on_success(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-retry",
        "project_name": "project-retry",
        "completed_steps": [],
        "failed_steps": ["predict"],
        "step_errors": {"predict": "old prediction failure"},
        "step_outputs": {},
        "current_step": None,
    }
    executor._execute_step = MagicMock(return_value={"predictions_path": "preds.jsonl"})
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "session-retry",
        "project_name": "project-retry",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert result["pipeline_state"] == "completed"
    assert saved["completed_steps"] == ["predict"]
    assert saved["failed_steps"] == []
    assert saved["step_errors"] == {}
    assert saved["step_outputs"]["predict"] == {"predictions_path": "preds.jsonl"}


def test_run_cancel_checker_stops_before_next_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    completed: list[str] = []
    pipeline_state = {
        "session_id": "cancel-sync",
        "project_name": "cancel-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }

    def execute_step(step_name, *_args, **_kwargs):
        completed.append(step_name)
        return {"step": step_name}

    executor._execute_step = MagicMock(side_effect=execute_step)
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["build_dataset_usd", "predict"],
        "step_configs": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
        },
        "cancel_checker": lambda: bool(completed),
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        executor.run(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert completed == ["build_dataset_usd"]
    assert saved["completed_steps"] == ["build_dataset_usd"]
    assert saved["current_step"] is None
    listener.event.assert_any_call(
        "step.cancelled",
        {
            "step_name": "predict",
            "message": "Pipeline cancellation requested",
        },
    )


@pytest.mark.asyncio
async def test_arun_skips_restore_usd_without_optimize(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    pipeline_state = {
        "session_id": None,
        "project_name": None,
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }
    executor._aexecute_step = AsyncMock()
    context = {
        "working_dir": str(tmp_path / "work"),
        "steps_to_run": ["restore_usd"],
        "step_configs": {"restore_usd": {"enabled": True}},
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await executor.arun(context)

    executor._aexecute_step.assert_not_awaited()
    assert result["pipeline_state"] == "completed"
    assert result["pipeline_results"] == {}


@pytest.mark.asyncio
async def test_arun_activation_success_and_in_step_cancel_paths(tmp_path: Path) -> None:
    listener = MagicMock()

    activation_executor = UnifiedPipelineExecutorTask()
    activation_calls: list[dict | None] = []
    activation_executor._activate_generated_material_library = MagicMock(
        side_effect=lambda outputs, _context, _configs: activation_calls.append(outputs)
    )
    activation_executor._aexecute_step = AsyncMock(
        return_value={"generated_materials_data": {"entries": []}}
    )
    activation_state = {
        "session_id": "async-activation",
        "project_name": "project",
        "completed_steps": [],
        "failed_steps": ["generate_material_library"],
        "step_errors": {"generate_material_library": "old"},
        "step_outputs": {},
        "current_step": None,
    }
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=activation_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await activation_executor.arun(
            {
                "working_dir": str(tmp_path / "async-activation"),
                "steps_to_run": ["generate_material_library"],
                "step_configs": {"generate_material_library": {"enabled": True}},
            }
        )
    assert result["pipeline_state"] == "completed"
    assert activation_calls[-1] == {"generated_materials_data": {"entries": []}}
    assert activation_state["failed_steps"] == []
    assert activation_state["step_errors"] == {}

    optimize_executor = UnifiedPipelineExecutorTask()
    optimize_executor._aexecute_step = AsyncMock(
        side_effect=[
            {"original_prim_count": 11},
            {"num_prims": 7, "num_images": 5},
        ]
    )
    optimize_state = {
        "session_id": "async-success",
        "project_name": "project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=optimize_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await optimize_executor.arun(
            {
                "working_dir": str(tmp_path / "async-success"),
                "steps_to_run": ["optimize_usd", "build_dataset_usd"],
                "step_configs": {
                    "optimize_usd": {"enabled": True},
                    "build_dataset_usd": {"enabled": True},
                },
            }
        )
    assert result["original_prim_count"] == 11
    assert result["num_prims"] == 7
    assert result["num_images"] == 5

    cancel_executor = UnifiedPipelineExecutorTask()
    cancel_executor._aexecute_step = AsyncMock(side_effect=asyncio.CancelledError())
    cancel_state = {
        "session_id": "async-cancel",
        "project_name": "project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    cancel_dir = tmp_path / "async-cancel"
    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=cancel_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await cancel_executor.arun(
            {
                "working_dir": str(cancel_dir),
                "steps_to_run": ["predict"],
                "step_configs": {"predict": {"enabled": True}},
            }
        )
    saved = json.loads((cancel_dir / ".pipeline_state.json").read_text())
    assert saved["current_step"] is None


@pytest.mark.asyncio
async def test_arun_cancel_checker_stops_before_next_step(tmp_path: Path) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    completed: list[str] = []
    pipeline_state = {
        "session_id": "cancel-async",
        "project_name": "cancel-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }

    async def execute_step(step_name, *_args, **_kwargs):
        completed.append(step_name)
        return {"step": step_name}

    executor._aexecute_step = AsyncMock(side_effect=execute_step)
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["build_dataset_usd", "predict"],
        "step_configs": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
        },
        "cancel_checker": lambda: bool(completed),
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await executor.arun(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert completed == ["build_dataset_usd"]
    assert saved["completed_steps"] == ["build_dataset_usd"]
    assert saved["current_step"] is None
    listener.event.assert_any_call(
        "step.cancelled",
        {
            "step_name": "predict",
            "message": "Pipeline cancellation requested",
        },
    )


@pytest.mark.asyncio
async def test_arun_skips_failed_optimize_usd_and_continues(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    event_listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "session-3",
        "project_name": "project-3",
        "completed_steps": [],
        "failed_steps": [],
        "step_outputs": {},
        "current_step": None,
    }
    secret = "material-async-optimizer-failure-api-key-713"
    executor._aexecute_step = AsyncMock(
        side_effect=[
            RuntimeError(f"provider failed with api_key={secret}"),
            {"output_dir": "dataset", "num_prims": 4, "num_images": 6},
        ]
    )
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["optimize_usd", "build_dataset_usd"],
        "step_configs": {
            "optimize_usd": {"input_usd_path": str(tmp_path / "input.usd")},
            "build_dataset_usd": {"usd_path": str(tmp_path / "scene.usd")},
        },
        "session_id": "session-3",
        "project_name": "project-3",
        "event_listener": event_listener,
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
    ):
        result = await executor.arun(context)

    state_file = working_dir / ".pipeline_state.json"
    saved = json.loads(state_file.read_text(encoding="utf-8"))

    assert result["pipeline_state"] == "completed"
    assert result["original_prim_count"] == 4
    assert result["pipeline_results"]["build_dataset_usd"]["num_images"] == 6
    assert saved["optimize_usd_skipped_original_input"] == str(tmp_path / "input.usd")
    assert saved["completed_steps"] == ["build_dataset_usd"]
    event_listener.event.assert_any_call(
        "step.skipped",
        {
            "step_name": "optimize_usd",
            "reason": ("optimize_usd failed: RuntimeError during step execution"),
        },
    )
    observable = caplog.text + repr(event_listener.mock_calls) + json.dumps(saved)
    assert secret not in observable


@pytest.mark.asyncio
async def test_arun_records_step_error_for_failed_step(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = UnifiedPipelineExecutorTask()
    listener = MagicMock()
    working_dir = tmp_path / "work"
    pipeline_state = {
        "session_id": "async-error",
        "project_name": "async-project",
        "completed_steps": [],
        "failed_steps": [],
        "step_errors": {},
        "step_outputs": {},
        "current_step": None,
    }
    secret = "material-async-provider-failure-api-key-713"
    executor._aexecute_step = AsyncMock(
        side_effect=RuntimeError(f"provider failed with api_key={secret}")
    )
    context = {
        "working_dir": str(working_dir),
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {"enabled": True}},
        "session_id": "async-error",
        "project_name": "async-project",
    }

    with (
        patch(
            "material_agent.tasks.unified_pipeline_executor._load_pipeline_state",
            return_value=pipeline_state,
        ),
        patch(
            "material_agent.tasks.unified_pipeline_executor.get_listener",
            return_value=listener,
        ),
        pytest.raises(
            RuntimeError,
            match=(
                "Pipeline failed at step 'predict': RuntimeError during step execution"
            ),
        ) as exc_info,
    ):
        await executor.arun(context)

    saved = json.loads((working_dir / ".pipeline_state.json").read_text())
    assert saved["failed_steps"] == ["predict"]
    assert saved["step_errors"] == {"predict": "RuntimeError during step execution"}
    assert saved["current_step"] is None
    listener.event.assert_called_with(
        "step.failed",
        {
            "step_name": "predict",
            "error": "RuntimeError during step execution",
        },
    )
    observable = caplog.text + str(exc_info.value) + repr(listener.mock_calls)
    observable += json.dumps(saved)
    assert secret not in observable
    assert exc_info.value.__cause__ is None
    failure_records = [
        record
        for record in caplog.records
        if "RuntimeError during step execution" in record.getMessage()
    ]
    assert failure_records
    assert all(record.exc_info is None for record in failure_records)
