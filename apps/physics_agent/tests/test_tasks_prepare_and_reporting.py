# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from physics_agent.api.defaults import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT
from physics_agent.tasks.prepare_dataset import (
    PrepareDatasetTask,
    _merged_vlm_image_prompts,
)
from physics_agent.tasks.reporting import GeneratePredictionReportTask


class MemoryObjectStore:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = dict(data or {})

    def exists(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


def test_prepare_dataset_task_builds_v02_dataset_with_context_and_failures(
    tmp_path: Path,
):
    usd_dir = tmp_path / "usd"
    model_dir = usd_dir / "."
    model_dir.mkdir(parents=True)

    renders_dir = model_dir / "renders"
    renders_dir.mkdir()
    (renders_dir / "composition.png").write_bytes(b"composition")
    (renders_dir / "prim_only.png").write_bytes(b"prim")

    (model_dir / "dataset.json").write_text(
        json.dumps({"statistics": {"total_prims": 1}}),
        encoding="utf-8",
    )
    prim_record = {
        "prim_path": "/World/Body",
        "world_bbox_meters": {"size": [1.0, 2.0, 3.0]},
        "relative_metrics": {"relative_size": [0.1, 0.2, 0.3]},
        "renders": [
            {
                "path": "renders/composition.png",
                "view": "+x+y+z",
                "camera": "cam-a",
                "render_mode": "composition",
            },
            {
                "path": "renders/prim_only.png",
                "view": "-x-y-z",
                "camera": "cam-b",
                "render_mode": "prim_only",
            },
        ],
    }
    (model_dir / "prims.jsonl").write_text(
        json.dumps(prim_record) + "\n",
        encoding="utf-8",
    )

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    reference_2 = tmp_path / "reference_2.png"
    reference_2.write_bytes(b"reference 2")
    structure_assignments = tmp_path / "assignments.json"
    structure_assignments.write_text(
        json.dumps(
            {
                "assignments": {
                    "/World/Body": {"component_name": "chassis"},
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_path = tmp_path / "prepared"
    result = PrepareDatasetTask().run(
        {
            "usd_dir": str(usd_dir),
            "dataset_path": str(dataset_path),
            "models": [".", "missing-model"],
            "reference_images": [str(reference), str(reference_2)],
            "config": {
                "include_prim_path_context": True,
                "include_geometric_context": True,
                "structure_assignments_path": str(structure_assignments),
                "render_mode_filter": ["composition", "prim_only"],
                "prompts": {
                    "system": "system prompt",
                    "user": "Describe the component.",
                    "vlm_image_prompts": [
                        {"composition": "Composition view."},
                        {"prim_only": "Prim-only view."},
                        {"reference_images": ["Reference A.", "Reference B."]},
                    ],
                },
            },
        }
    )

    assert result["failed_models"] == ["missing-model"]
    assert result["dataset_jsonl_path"].endswith("dataset.jsonl")
    assert result["dataset_config_path"].endswith("dataset.json")
    assert len(result["dataset_entries"]) == 1

    entry = result["dataset_entries"][0]
    assert entry["id"] == "/World/Body"
    assert "The prim path of this 3D asset is: /World/Body" in entry["user_prompt"]
    assert "Bounding box volume" in entry["user_prompt"]
    assert "chassis" in entry["user_prompt"]
    assert entry["metadata"]["world_bbox_meters"] == {"size": [1.0, 2.0, 3.0]}
    assert entry["metadata"]["relative_metrics"] == {"relative_size": [0.1, 0.2, 0.3]}

    media_images = entry["media"]["images"]
    assert media_images[0]["type"] == "reference"
    assert media_images[0]["metadata"]["vlm_prompt"] == "Reference A."
    assert media_images[1]["type"] == "reference"
    assert media_images[1]["metadata"]["vlm_prompt"] == "Reference B."
    assert media_images[2]["metadata"]["render_mode"] == "composition"
    assert "Camera Position" in media_images[2]["metadata"]["vlm_prompt"]
    assert "Composition view." in media_images[2]["metadata"]["vlm_prompt"]

    dataset_config = json.loads((dataset_path / "dataset.json").read_text())
    assert dataset_config["metadata"]["num_entries"] == 1
    assert dataset_config["inference"]["prompts"][0]["system_prompt"] == "system prompt"


def test_prepare_dataset_task_uses_physics_schema_prompt_by_default(
    tmp_path: Path,
) -> None:
    usd_dir = tmp_path / "usd"
    model_dir = usd_dir / "."
    model_dir.mkdir(parents=True)
    renders_dir = model_dir / "renders"
    renders_dir.mkdir()
    (renders_dir / "composition.png").write_bytes(b"composition")
    (model_dir / "dataset.json").write_text(
        json.dumps({"statistics": {"total_prims": 1}}),
        encoding="utf-8",
    )
    (model_dir / "prims.jsonl").write_text(
        json.dumps(
            {
                "prim_path": "/World/Wheel",
                "renders": [
                    {
                        "path": "renders/composition.png",
                        "view": "+x",
                        "camera": "cam-a",
                        "render_mode": "composition",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_path = tmp_path / "prepared"
    result = PrepareDatasetTask().run(
        {
            "usd_dir": str(usd_dir),
            "dataset_path": str(dataset_path),
            "models": ["."],
            "config": {},
        }
    )

    entry = result["dataset_entries"][0]
    assert entry["user_prompt"].startswith(DEFAULT_USER_PROMPT.splitlines()[0])
    dataset_config = json.loads((dataset_path / "dataset.json").read_text())
    assert (
        dataset_config["inference"]["prompts"][0]["system_prompt"]
        == DEFAULT_SYSTEM_PROMPT
    )


def test_merged_vlm_image_prompts_rejects_invalid_list_item() -> None:
    with pytest.raises(ValueError, match="entries must be mappings"):
        _merged_vlm_image_prompts([{"composition": "Composition view."}, "bad"])


def test_merged_vlm_image_prompts_rejects_non_string_render_prompt() -> None:
    with pytest.raises(ValueError, match="composition must be a string"):
        _merged_vlm_image_prompts({"composition": {"bad": "value"}})


def test_merged_vlm_image_prompts_accepts_reference_prompt_list() -> None:
    prompts = _merged_vlm_image_prompts(
        {"reference_images": ["Reference A.", "Reference B."]}
    )

    assert prompts["reference_images"] == ["Reference A.", "Reference B."]


def test_generate_prediction_report_task_creates_html_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "component_type": "panel",
                    "component_name": "hood",
                    "material": "metal",
                    "confidence": "high",
                    "physical_properties": {
                        "density": 7.8,
                        "static_friction": 0.3,
                        "dynamic_friction": 0.2,
                        "restitution": 0.1,
                    },
                    "original_response": "raw output",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "user_prompt": "Describe this asset.",
                "media": {
                    "images": [{"path": "render.png", "metadata": {"view": "+x"}}]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_images_html",
        lambda *args, **kwargs: "<div class='images'>stub</div>",
    )
    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_system_prompt_section",
        lambda prompt: f"<section>{prompt}</section>",
    )
    monkeypatch.setattr(
        "physics_agent.tasks.reporting.validate_image_options",
        lambda image_format, image_quality, image_max_size: (
            image_format or "png",
            image_quality or 85,
            image_max_size or 128,
        ),
    )

    object_store = MemoryObjectStore(
        {
            "dataset": [
                {
                    "id": "prim-1",
                    "text": "fallback prompt",
                    "images": ["render.png"],
                    "image_metadata": [{"view": "+x"}],
                }
            ]
        }
    )
    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_path),
            "predictions_count": 1,
            "failed_count": 0,
            "token_stats": {
                "total_tokens": 42,
                "total_input_tokens": 30,
                "total_output_tokens": 12,
                "invocation_count": 1,
                "by_model": {
                    "demo-model": {
                        "count": 1,
                        "input_tokens": 30,
                        "output_tokens": 12,
                        "total_tokens": 42,
                    }
                },
            },
            "actual_system_prompt_used": "system prompt",
            "report_image_max_size": 96,
            "report_image_format": "jpeg",
            "report_image_quality": 77,
        },
        object_store,
    )

    report_path = Path(result["report_path"])
    html = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Physics Agent - Prediction Report" in html
    assert "hood" in html
    assert "metal" in html
    assert "demo-model" in html
    assert "<div class='images'>stub</div>" in html
    assert "<section>system prompt</section>" in html


def test_generate_prediction_report_task_unwraps_nested_output_key_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "classification": {
                        "component_type": "panel",
                        "component_name": "hood",
                        "material": "metal",
                        "confidence": "high",
                        "physical_properties": {"density": 7.8},
                    },
                    "original_response": "raw nested output",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_images_html",
        lambda *args, **kwargs: "<div class='images'>stub</div>",
    )

    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "predictions_count": 1,
            "failed_count": 0,
            "output_key": "classification",
        }
    )

    html = Path(result["report_path"]).read_text(encoding="utf-8")

    assert "hood" in html
    assert "metal" in html
    assert "rho: 7.8" in html
    assert "raw nested output" in html


def test_generate_prediction_report_escapes_physical_property_values(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    script = '<img src=x onerror="alert(1)">'
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "component_type": "panel",
                    "physical_properties": {
                        "estimated_mass_kg": script,
                        "density": script,
                        "static_friction": script,
                        "dynamic_friction": script,
                        "restitution": script,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "predictions_count": 1,
            "failed_count": 0,
        }
    )

    html = Path(result["report_path"]).read_text(encoding="utf-8")
    assert script not in html
    assert html.count("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;") == 5


def test_generate_prediction_report_task_builds_warnings_from_nested_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "classification": {
                        "component_type": "link",
                        "component_name": "oversized robot link",
                        "material": "metal",
                        "confidence": "high",
                        "physical_properties": {
                            "density": 2700,
                            "estimated_mass_kg": 25000,
                            "static_friction": 0.5,
                            "dynamic_friction": 0.4,
                            "restitution": 0.3,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "user_prompt": "Describe this robot part.",
                "metadata": {"world_bbox_meters": {"size": [8.0, 0.8, 0.8]}},
                "media": {"images": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_images_html",
        lambda *args, **kwargs: "<div class='images'>stub</div>",
    )

    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_path),
            "predictions_count": 1,
            "failed_count": 0,
            "output_key": "classification",
        }
    )

    html = Path(result["report_path"]).read_text(encoding="utf-8")

    assert "Mass/Scale Warnings" in html
    assert "mass_scale_suspicious" in html
    assert "m: 25000" in html


def test_generate_prediction_report_task_surfaces_mass_scale_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "classification": {
                    "component_type": "link",
                    "component_name": "oversized robot link",
                    "material": "metal",
                    "confidence": "high",
                    "physical_properties": {
                        "density": 2700,
                        "estimated_mass_kg": 25000,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                        "restitution": 0.3,
                    },
                },
                "quality_warnings": [
                    {
                        "code": "custom_warning",
                        "severity": "warning bad",
                        "message": "custom warning",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "prim-1",
                "user_prompt": "Describe this robot part.",
                "metadata": {"world_bbox_meters": {"size": [8.0, 0.8, 0.8]}},
                "media": {"images": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "physics_agent.tasks.reporting.format_images_html",
        lambda *args, **kwargs: "<div class='images'>stub</div>",
    )

    result = GeneratePredictionReportTask().run(
        {
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_path),
            "predictions_count": 1,
            "failed_count": 0,
        }
    )

    html = Path(result["report_path"]).read_text(encoding="utf-8")

    assert "Mass/Scale Warnings" in html
    assert "mass_scale_suspicious" in html
    assert 'class="qa-warningbad"' in html
    assert 'class="qa-warning bad"' not in html
    assert "m: 25000" in html
