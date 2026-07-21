# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for HTML reporting tasks."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from world_understanding.utils.object_store import InMemoryObjectStore

from material_agent.tasks.reporting import (
    GenerateEvaluationReportTask,
    GeneratePredictionReportTask,
)


def _write_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60)
    Image.new(mode, (8, 6), color).save(path)


def test_base_report_image_processing_and_formatting_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = GeneratePredictionReportTask()
    rgba = tmp_path / "rgba.png"
    _write_image(rgba, mode="RGBA")
    paletted = tmp_path / "paletted.png"
    Image.new("P", (4, 4)).save(paletted)

    assert base64.b64decode(task._process_and_encode_image(rgba)).startswith(b"\x89PNG")
    assert base64.b64decode(
        task._process_and_encode_image(paletted, image_format="jpeg")
    ).startswith(b"\xff\xd8")

    jpeg_data = base64.b64decode(
        task._process_and_encode_image(
            rgba,
            image_max_size=4,
            image_format="jpeg",
            image_quality=70,
        )
    )
    assert jpeg_data.startswith(b"\xff\xd8")

    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"not an image")
    assert base64.b64decode(task._process_and_encode_image(raw)) == b"not an image"

    with pytest.raises(FileNotFoundError):
        task._process_and_encode_image(tmp_path / "missing.bin")

    assert task._format_images([], None) == "No images"
    html = task._format_images(
        ["rgba.png", "missing.png"],
        str(tmp_path / "dataset.jsonl"),
        [{"vlm_prompt": "Prompt <with> html"}],
        image_format="jpeg",
    )
    assert "Prompt &lt;with&gt; html" in html
    assert "missing.png (not found)" in html
    absolute_html = task._format_images([str(rgba)], None)
    assert "image-thumbnail" in absolute_html

    def fail_processing(*args, **kwargs):
        raise RuntimeError("bad image")

    monkeypatch.setattr(task, "_process_and_encode_image", fail_processing)
    listener = MagicMock()
    error_html = task._format_images(
        ["rgba.png"], str(tmp_path / "dataset.jsonl"), listener=listener
    )
    assert "rgba.png (error)" in error_html
    listener.debug.assert_called()
    assert task._format_token_stats_html({}) == ""


def test_prediction_report_run_uses_object_store_and_writes_html(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "images" / "part.png"
    _write_image(image_path)
    dataset = [
        {
            "id": "/A",
            "user_prompt": "Classify <A>",
            "media": {
                "images": [
                    {
                        "path": "images/part.png",
                        "metadata": {"vlm_prompt": "front view"},
                    }
                ]
            },
        },
        {
            "id": "/B",
            "text": "Fallback text",
            "images": ["images/part.png"],
            "image_metadata": [{"vlm_prompt": "legacy view"}],
        },
    ]
    store = InMemoryObjectStore()
    store.set(
        "predictions",
        [
            {
                "id": "/A",
                "vlm_response": {
                    "material": "Steel",
                    "original_response": "steel response",
                },
            },
            {"id": "/B", "vlm_response": "Rubber response"},
        ],
    )
    store.set("failed_predictions", [{"id": "/C", "error": "model failed"}])
    store.set("dataset", dataset)

    context = {
        "output_dir": tmp_path / "reports",
        "dataset_path": str(tmp_path / "dataset.jsonl"),
        "report_image_format": "jpeg",
        "report_image_quality": 60,
        "report_image_max_size": 4,
        "actual_system_prompt_used": "system prompt",
        "token_stats": {
            "invocation_count": 2,
            "total_tokens": 30,
            "total_input_tokens": 10,
            "total_output_tokens": 20,
            "by_model": {
                "model-a": {
                    "count": 2,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                }
            },
        },
        "original_prim_count": 3,
        "num_prims": 2,
        "num_images": 1,
    }

    result = GeneratePredictionReportTask().run(context, store)

    report_path = Path(result["html_report_path"])
    html = report_path.read_text(encoding="utf-8")
    assert "Material Agent Prediction Report" in html
    assert "Steel" in html
    assert "model failed" in html
    assert "model-a" in html


def test_prediction_report_handles_generation_errors(tmp_path: Path) -> None:
    task = GeneratePredictionReportTask()
    listener = MagicMock()

    with patch.object(task, "_create_html_content", side_effect=RuntimeError("boom")):
        assert (
            task._generate_report(
                predictions=[],
                failed=[],
                dataset=[],
                output_dir=tmp_path,
                context={},
                listener=listener,
            )
            is None
        )
    listener.error.assert_called()

    with patch.object(task, "_generate_report", side_effect=RuntimeError("boom")):
        result = task.run({"output_dir": tmp_path})
    assert "html_report_path" not in result


def test_evaluation_report_run_loads_files_and_writes_html(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "part.png"
    _write_image(image_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "/A",
                "text": "Evaluate A",
                "images": ["images/part.png"],
                "image_metadata": [{"vlm_prompt": "eval view"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "/A",
                "images": ["images/part.png"],
                "materials": {
                    "material": "Steel",
                    "original_response": "steel response",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    context = {
        "output_dir": tmp_path / "reports",
        "dataset_path": str(dataset_path),
        "predictions_path": str(predictions_path),
        "evaluations": [
            {
                "id": "/A",
                "predicted_material": "N/A",
                "ground_truth": "Steel",
                "score": 5,
                "exact_match": True,
                "explanation": "good",
            },
            {
                "id": "/B",
                "predicted_material": "Rubber",
                "ground_truth": "Plastic",
                "score": 1,
                "exact_match": False,
                "explanation": "bad",
            },
        ],
        "metrics": {
            "functional_correctness_score": 4.5,
            "success_rate": 50.0,
            "exact_match_rate": 50.0,
            "total_cases": 2,
            "valid_cases": 2,
            "exact_matches": 1,
        },
        "config": {"system_prompt": "judge prompt"},
    }

    result = GenerateEvaluationReportTask().run(context)

    report_path = Path(result["evaluation_html_report_path"])
    html = report_path.read_text(encoding="utf-8")
    assert "Material Agent Evaluation Report" in html
    assert "4.50" in html
    assert "✓ Match" in html
    assert "Rubber" in html


def test_evaluation_report_uses_object_store_and_error_paths(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    store.set(
        "evaluations",
        [
            {
                "id": "/A",
                "ground_truth": "Steel",
                "score": 3,
                "exact_match": False,
            }
        ],
    )
    store.set("metrics", {"fcs": 3.0, "total_cases": 1})
    store.set("predictions", [{"id": "/A", "text": "Prompt", "materials": "Steel"}])

    context = {
        "output_dir": tmp_path / "reports",
        "predictions_path": str(tmp_path / "missing_predictions.jsonl"),
        "config_path": str(tmp_path / "config.yaml"),
    }
    result = GenerateEvaluationReportTask().run(context, store)
    assert Path(result["evaluation_html_report_path"]).exists()

    task = GenerateEvaluationReportTask()
    listener = MagicMock()
    assert (
        task._generate_report(
            evaluations=[],
            metrics={},
            output_dir=tmp_path,
            predictions=[{}],
            context={},
            listener=listener,
        )
        is None
    )
    listener.error.assert_called()

    with patch.object(task, "_generate_report", side_effect=RuntimeError("boom")):
        result = task.run({"output_dir": tmp_path})
    assert "evaluation_html_report_path" not in result
