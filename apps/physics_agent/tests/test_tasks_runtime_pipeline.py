# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from world_understanding.agentic.usd_tasks.optimizer_models import UsdFormat
from world_understanding.utils.token_tracking import TokenUsage

from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.inference import VLMInferenceTask
from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask
from physics_agent.workflows import (
    create_identify_asset_workflow_from_config,
    create_optimize_usd_workflow_from_config,
    create_prediction_workflow_from_config,
    create_prepare_dataset_workflow_from_config,
    create_restore_usd_workflow_from_config,
    create_unified_pipeline_workflow,
    create_usd_data_preparation_workflow_from_config,
)


class RecordingListener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.warnings: list[str] = []

    def event(self, name: str, payload: dict[str, object]) -> None:
        self.events.append((name, payload))

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass


class MemoryObjectStore:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = dict(data or {})

    def exists(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._data.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._data[key] = value


class FakeVLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_workflow_factories_build_expected_task_sequences():
    assert [task.name for task in create_prediction_workflow_from_config().tasks] == [
        "PredictConfig",
        "ModelProvisioning",
        "DatasetLoading",
        "VLMInference",
        "GeneratePredictionReport",
        "SavePredictions",
    ]
    assert [
        task.name for task in create_prepare_dataset_workflow_from_config().tasks
    ] == [
        "PrepareDatasetConfig",
        "PrepareDataset",
    ]
    assert [
        task.name for task in create_identify_asset_workflow_from_config().tasks
    ] == [
        "IdentifyAssetConfig",
        "RenderScenePreview",
        "ModelProvisioning",
        "IdentifyAsset",
    ]
    assert [
        task.__class__.__name__
        for task in create_optimize_usd_workflow_from_config().tasks
    ] == ["OptimizeUSDConfigTask", "OptimizeUSDTask"]
    assert [
        task.__class__.__name__
        for task in create_restore_usd_workflow_from_config().tasks
    ] == ["RestoreUSDConfigTask", "RestoreUSDTask"]
    assert (
        create_usd_data_preparation_workflow_from_config().name
        == "USD → Dataset Preparation"
    )
    assert [task.name for task in create_unified_pipeline_workflow().tasks] == [
        "UnifiedConfigLoading",
        "UnifiedPipelineExecutor",
    ]


def test_identify_asset_task_parses_and_saves_results(tmp_path: Path):
    output_dir = tmp_path / "identify"
    vlm = FakeVLM(
        """```json
        {"asset_type":"vehicle","asset_subtype":"forklift","confidence":"high"}
        ```"""
    )
    images = [str(tmp_path / f"img-{i}.png") for i in range(7)]

    result = IdentifyAssetTask().run(
        {
            "vlm": vlm,
            "composition_images": images,
            "identify_system_prompt": "identify it",
            "output_dir": str(output_dir),
        }
    )

    saved = json.loads((output_dir / "identification.json").read_text(encoding="utf-8"))
    assert result["identification"]["asset_type"] == "vehicle"
    assert saved["asset_subtype"] == "forklift"
    assert len(vlm.calls[0]["images"]) == 6

    no_images = IdentifyAssetTask().run({"vlm": vlm, "output_dir": str(output_dir)})
    assert no_images["identification"]["asset_type"] == "unknown"


def test_vlm_inference_task_streams_predictions_and_supports_resume(
    tmp_path: Path, monkeypatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_entries = [
        {"id": "prim-1", "images": ["a.png"]},
        {"id": "prim-2", "image_path": "b.png"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(entry) for entry in dataset_entries) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "prim-1", "classification": {"label": "old"}}) + "\n",
        encoding="utf-8",
    )
    listener = RecordingListener()

    captured: dict[str, object] = {}

    def fake_batch_classify_assets(**kwargs):
        captured["processed_ids"] = kwargs["processed_ids"]
        captured["max_workers"] = kwargs["max_workers"]
        kwargs["token_tracker"].add_usage(
            TokenUsage(
                input_tokens=80,
                output_tokens=20,
                total_tokens=100,
                model_name="qwen-test",
                invocation_type="vlm",
            )
        )
        kwargs["on_progress"]("prim-2", "ok")
        kwargs["on_prediction"](
            "prim-2", {"classification": {"label": "metal"}, "confidence": "high"}
        )
        kwargs["on_result"](
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            dataset_entries[1],
        )
        return [
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            {"id": "prim-3", "status": "error", "error": "failed"},
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    object_store = MemoryObjectStore({"dataset": dataset_entries})
    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
            "vlm_config": {"max_retries": 5},
            "max_workers": 8,
            "event_listener": listener,
        },
        object_store,
    )

    saved_lines = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    assert result["predictions_count"] == 2
    assert result["failed_count"] == 1
    assert result["inference_complete"] is True
    assert result["token_stats"]["all_usages"][0]["model_name"] == "qwen-test"
    assert json.dumps(result["token_stats"])
    assert captured["processed_ids"] == {"prim-1"}
    assert captured["max_workers"] == 8
    assert saved_lines[-1]["classification"] == {"label": "metal"}
    assert any(event[0] == "prediction.completed" for event in listener.events)


def test_vlm_inference_task_quarantines_malformed_resume_lines(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [
        {"id": "prim-1", "images": ["a.png"]},
        {"id": "prim-2", "image_path": "b.png"},
        {"id": "prim-stale", "image_path": "c.png"},
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    valid_prediction = {"id": "prim-1", "classification": {"label": "old"}}
    later_tail_prediction = {"id": "prim-stale", "classification": {"label": "old"}}
    malformed_tail = '{"id": "partial"\n'
    predictions_path.write_text(
        json.dumps(valid_prediction)
        + "\n"
        + malformed_tail
        + json.dumps(later_tail_prediction)
        + "\n",
        encoding="utf-8",
    )
    diagnostics_path = tmp_path / "predictions.diagnostics.jsonl"
    captured: dict[str, object] = {}

    def fake_batch_classify_assets(**kwargs):
        captured["processed_ids"] = kwargs["processed_ids"]
        kwargs["on_result"](
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            dataset_entries[1],
        )
        return [
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}}
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    saved_lines = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    assert result["predictions_count"] == 3
    assert captured["processed_ids"] == {"prim-1", "prim-stale"}
    assert saved_lines[0] == valid_prediction
    assert saved_lines[1] == later_tail_prediction
    assert saved_lines[2]["id"] == "prim-2"
    assert saved_lines[2]["classification"] == {"label": "metal"}
    assert saved_lines[2]["image_path"] == "b.png"
    assert diagnostics_path.read_text(encoding="utf-8") == malformed_tail


def test_vlm_inference_task_normalizes_resume_predictions_missing_newline(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [
        {"id": "prim-1", "images": ["a.png"]},
        {"id": "prim-2", "image_path": "b.png"},
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    valid_prediction = {"id": "prim-1", "classification": {"label": "old"}}
    predictions_path.write_text(json.dumps(valid_prediction), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_batch_classify_assets(**kwargs):
        captured["processed_ids"] = kwargs["processed_ids"]
        kwargs["on_result"](
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}},
            dataset_entries[1],
        )
        return [
            {"id": "prim-2", "status": "success", "vlm_response": {"label": "metal"}}
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    saved_lines = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    assert result["predictions_count"] == 2
    assert captured["processed_ids"] == {"prim-1"}
    assert saved_lines[0] == valid_prediction
    assert saved_lines[1]["id"] == "prim-2"
    assert saved_lines[1]["classification"] == {"label": "metal"}
    assert predictions_path.read_text(encoding="utf-8").startswith(
        json.dumps(valid_prediction) + "\n"
    )


def test_vlm_inference_task_appends_stream_diagnostics_to_resume_quarantine(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [
        {"id": "prim-1", "images": ["a.png"]},
        {"id": "prim-2", "image_path": "b.png"},
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    diagnostics_path = tmp_path / "predictions.diagnostics.jsonl"
    valid_prediction = {"id": "prim-1", "classification": {"label": "old"}}
    resume_malformed_line = "resume debug line\n"
    stream_malformed_line = "stream debug line\n"
    predictions_path.write_text(
        json.dumps(valid_prediction) + "\n" + resume_malformed_line,
        encoding="utf-8",
    )

    def fake_batch_classify_assets(**kwargs):
        with open(predictions_path, "a", encoding="utf-8") as f:
            f.write(stream_malformed_line)
        return [{"id": "prim-2", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="malformed JSON records"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "image_base_dir": str(tmp_path),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "resume": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert predictions_path.read_text(encoding="utf-8") == (
        json.dumps(valid_prediction) + "\n" + stream_malformed_line
    )
    assert diagnostics_path.read_text(encoding="utf-8") == (
        resume_malformed_line + stream_malformed_line
    )


def test_vlm_inference_task_serializes_concurrent_streaming_writes(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": f"prim-{i}", "image_path": f"{i}.png"} for i in range(40)]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        def emit(entry: dict[str, str]) -> dict[str, object]:
            result = {
                "id": entry["id"],
                "status": "success",
                "vlm_response": {"label": entry["id"]},
            }
            kwargs["on_result"](result, entry)
            return result

        with ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(emit, dataset_entries))

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "output_key": "classification",
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    saved_lines = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]
    assert result["predictions_count"] == len(dataset_entries)
    assert {line["id"] for line in saved_lines} == {
        entry["id"] for entry in dataset_entries
    }
    assert all(line["classification"]["label"] == line["id"] for line in saved_lines)


def test_vlm_inference_task_fails_closed_when_all_predictions_fail(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        kwargs["on_error"]("prim-1", "hosted VLM unavailable")
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="zero successful predictions"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert not predictions_path.exists()


def test_vlm_inference_task_fails_closed_when_non_streaming_predictions_fail(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="zero successful predictions"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": False,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert not predictions_path.exists()


def test_vlm_inference_task_fails_closed_for_empty_dataset_by_default(
    tmp_path: Path, monkeypatch
):
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    called = False

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert called is False
    assert not predictions_path.exists()
    assert not predictions_path.parent.exists()


def test_vlm_inference_task_preserves_stale_predictions_when_resuming_empty_dataset(
    tmp_path: Path,
):
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    predictions_path.write_text('{"id": "stale"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "resume": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert predictions_path.read_text(encoding="utf-8") == '{"id": "stale"}\n'


def test_vlm_inference_task_validates_allow_empty_predictions_type(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="allow_empty_predictions"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "allow_empty_predictions": "yes",
            },
            MemoryObjectStore({"dataset": [{"id": "prim-1", "images": ["a.png"]}]}),
        )


def test_vlm_inference_task_preserves_explicit_predictions_for_empty_dataset(
    tmp_path: Path, monkeypatch
):
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    existing_predictions = '{"id": "stale"}\n'
    predictions_path.write_text(existing_predictions, encoding="utf-8")
    called = False

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert called is False
    assert predictions_path.read_text(encoding="utf-8") == existing_predictions


def test_vlm_inference_task_preserves_explicit_predictions_for_non_streaming_empty_dataset(
    tmp_path: Path,
):
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    existing_predictions = '{"id": "stale"}\n'
    predictions_path.write_text(existing_predictions, encoding="utf-8")

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": False,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert predictions_path.read_text(encoding="utf-8") == existing_predictions


def test_vlm_inference_task_clears_stale_predictions_from_output_dir(
    tmp_path: Path,
):
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    predictions_path.write_text('{"id": "stale"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "output_dir": str(predictions_path.parent),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert not predictions_path.exists()


def test_vlm_inference_task_clears_stale_predictions_from_dataset_path(
    tmp_path: Path,
):
    dataset_path = tmp_path / "dataset.jsonl"
    predictions_path = tmp_path / "output" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    predictions_path.write_text('{"id": "stale"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="zero dataset entries"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(dataset_path),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert not predictions_path.exists()


def test_vlm_inference_task_allows_empty_predictions_when_opted_in(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "output_key": "classification",
            "allow_empty_predictions": True,
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    assert result["predictions_count"] == 0
    assert result["failed_count"] == 1
    assert predictions_path.exists()
    assert predictions_path.read_text(encoding="utf-8") == ""


def test_vlm_inference_task_refuses_to_overwrite_explicit_non_streaming_empty_predictions(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"
    existing_predictions = '{"id": "stale"}\n'
    predictions_path.write_text(existing_predictions, encoding="utf-8")

    def fake_batch_classify_assets(**kwargs):
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": False,
                "output_key": "classification",
                "allow_empty_predictions": True,
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert predictions_path.read_text(encoding="utf-8") == existing_predictions


def test_vlm_inference_task_defers_non_streaming_empty_file_creation(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": False,
            "output_key": "classification",
            "allow_empty_predictions": True,
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    assert result["predictions_count"] == 0
    assert result["failed_count"] == 1
    assert not predictions_path.exists()


def test_vlm_inference_task_rejects_stream_diagnostics_for_empty_opt_in(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"
    diagnostics_path = tmp_path / "predictions.diagnostics.jsonl"

    def fake_batch_classify_assets(**kwargs):
        predictions_path.write_text("debug: hosted VLM unavailable\n", encoding="utf-8")
        return [{"id": "prim-1", "status": "error", "error": "hosted VLM unavailable"}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="malformed JSON records"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
                "allow_empty_predictions": True,
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert predictions_path.read_text(encoding="utf-8") == (
        "debug: hosted VLM unavailable\n"
    )
    assert diagnostics_path.read_text(encoding="utf-8") == (
        "debug: hosted VLM unavailable\n"
    )


def test_vlm_inference_task_rejects_stream_diagnostics_with_success(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"
    diagnostics_path = tmp_path / "predictions.diagnostics.jsonl"
    prediction = {"id": "prim-1", "classification": {"label": "dynamic"}}

    def fake_batch_classify_assets(**kwargs):
        predictions_path.write_text(
            json.dumps(prediction) + "\ndebug: hosted VLM latency spike\n",
            encoding="utf-8",
        )
        return [{"id": "prim-1", "status": "success", "classification": {}}]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="malformed JSON records"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
            },
            MemoryObjectStore({"dataset": dataset_entries}),
        )

    assert predictions_path.read_text(encoding="utf-8") == (
        json.dumps(prediction) + "\ndebug: hosted VLM latency spike\n"
    )
    assert diagnostics_path.read_text(encoding="utf-8") == (
        "debug: hosted VLM latency spike\n"
    )


def test_vlm_inference_task_allows_empty_dataset_when_opted_in(
    tmp_path: Path, monkeypatch
):
    called = False
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "output_key": "classification",
            "allow_empty_predictions": True,
        },
        MemoryObjectStore({"dataset": []}),
    )

    assert called is True
    assert result["predictions_count"] == 0
    assert result["failed_count"] == 0
    assert predictions_path.exists()
    assert predictions_path.read_text(encoding="utf-8") == ""


def test_vlm_inference_task_refuses_to_overwrite_explicit_non_streaming_empty_dataset(
    tmp_path: Path, monkeypatch
):
    called = False
    predictions_path = tmp_path / "predictions.jsonl"
    existing_predictions = '{"id": "stale"}\n'
    predictions_path.write_text(existing_predictions, encoding="utf-8")

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": False,
                "output_key": "classification",
                "allow_empty_predictions": True,
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert called is True
    assert predictions_path.read_text(encoding="utf-8") == existing_predictions


def test_vlm_inference_task_refuses_to_overwrite_explicit_streaming_empty_dataset(
    tmp_path: Path, monkeypatch
):
    called = False
    predictions_path = tmp_path / "predictions.jsonl"
    existing_predictions = '{"id": "stale"}\n'
    predictions_path.write_text(existing_predictions, encoding="utf-8")

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        VLMInferenceTask(vlm=FakeVLM()).run(
            {
                "dataset_path": str(tmp_path / "dataset.jsonl"),
                "predictions_path": str(predictions_path),
                "stream_predictions": True,
                "output_key": "classification",
                "allow_empty_predictions": True,
            },
            MemoryObjectStore({"dataset": []}),
        )

    assert called is False
    assert predictions_path.read_text(encoding="utf-8") == existing_predictions


def test_vlm_inference_task_preserves_explicit_resume_predictions_for_empty_dataset(
    tmp_path: Path, monkeypatch
):
    called = False
    predictions_path = tmp_path / "predictions.jsonl"
    existing_prediction = {"id": "prim-1", "classification": {"label": "old"}}
    predictions_path.write_text(
        json.dumps(existing_prediction) + "\n", encoding="utf-8"
    )

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
            "allow_empty_predictions": True,
        },
        MemoryObjectStore({"dataset": []}),
    )

    assert called is True
    assert result["predictions_count"] == 1
    assert result["failed_count"] == 0
    assert predictions_path.read_text(encoding="utf-8") == (
        json.dumps(existing_prediction) + "\n"
    )


def test_vlm_inference_task_preserves_implicit_resume_predictions_for_empty_dataset(
    tmp_path: Path, monkeypatch
):
    called = False
    predictions_path = tmp_path / "out" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    existing_prediction = {"id": "prim-1", "classification": {"label": "old"}}
    predictions_path.write_text(
        json.dumps(existing_prediction) + "\n", encoding="utf-8"
    )

    def fake_batch_classify_assets(**kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    result = VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "output_dir": str(predictions_path.parent),
            "stream_predictions": True,
            "resume": True,
            "output_key": "classification",
            "allow_empty_predictions": True,
        },
        MemoryObjectStore({"dataset": []}),
    )

    assert called is True
    assert result["predictions_count"] == 1
    assert result["failed_count"] == 0
    assert predictions_path.read_text(encoding="utf-8") == (
        json.dumps(existing_prediction) + "\n"
    )


def test_vlm_inference_task_streams_mass_scale_quality_warnings(
    tmp_path: Path, monkeypatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_entries = [
        {
            "id": "prim-1",
            "images": ["a.png"],
            "metadata": {"world_bbox_meters": {"size": [8.0, 0.8, 0.8]}},
        }
    ]
    dataset_path.write_text(
        json.dumps(dataset_entries[0]) + "\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.jsonl"
    listener = RecordingListener()
    oversized_prediction = {
        "component_type": "link",
        "physical_properties": {
            "density": 2700,
            "estimated_mass_kg": 25000,
        },
    }

    def fake_batch_classify_assets(**kwargs):
        kwargs["on_result"](
            {
                "id": "prim-1",
                "status": "success",
                "vlm_response": oversized_prediction,
            },
            dataset_entries[0],
        )
        return [
            {
                "id": "prim-1",
                "status": "success",
                "vlm_response": oversized_prediction,
            }
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    object_store = MemoryObjectStore({"dataset": dataset_entries})
    VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(dataset_path),
            "image_base_dir": str(tmp_path),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "output_key": "classification",
            "event_listener": listener,
        },
        object_store,
    )

    saved = json.loads(predictions_path.read_text(encoding="utf-8"))
    stored_predictions = object_store.get("predictions")

    assert saved["quality_warnings"][0]["code"] == "mass_scale_suspicious"
    assert stored_predictions[0]["quality_warnings"][0]["code"] == (
        "mass_scale_suspicious"
    )
    assert listener.warnings


def test_vlm_inference_task_streams_unwrapped_output_key_payload(
    tmp_path: Path, monkeypatch
):
    dataset_entries = [{"id": "prim-1", "images": ["a.png"]}]
    predictions_path = tmp_path / "predictions.jsonl"

    def fake_batch_classify_assets(**kwargs):
        nested_prediction = {
            "classification": {
                "component_type": "optical",
                "physical_properties": {"density": 2500},
            }
        }
        kwargs["on_result"](
            {
                "id": "prim-1",
                "status": "success",
                "vlm_response": nested_prediction,
            },
            dataset_entries[0],
        )
        return [
            {
                "id": "prim-1",
                "status": "success",
                "vlm_response": nested_prediction,
            }
        ]

    monkeypatch.setattr(
        "physics_agent.tasks.inference.batch_classify_assets",
        fake_batch_classify_assets,
    )

    VLMInferenceTask(vlm=FakeVLM()).run(
        {
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "predictions_path": str(predictions_path),
            "stream_predictions": True,
            "output_key": "classification",
        },
        MemoryObjectStore({"dataset": dataset_entries}),
    )

    saved = json.loads(predictions_path.read_text(encoding="utf-8"))

    assert saved["classification"] == {
        "component_type": "optical",
        "physical_properties": {"density": 2500},
    }


def test_unified_pipeline_executor_autowires_steps_and_collects_outputs(
    tmp_path: Path, monkeypatch
):
    executor = UnifiedPipelineExecutorTask()
    listener = RecordingListener()
    recorded_configs: dict[str, dict[str, object]] = {}
    optimize_output = tmp_path / "optimized.usdc"
    original_usd = tmp_path / "original.usd"
    original_usd.write_text("#usda 1.0\n", encoding="utf-8")

    step_results = {
        "optimize_usd": {
            "optimized_usd_path": str(optimize_output),
            "optimization_metadata": {
                "optimization_config": {
                    "scene_optimizer_settings": {"output_format": UsdFormat.USDC}
                }
            },
            "optimization_success": True,
            "original_usd_path": str(original_usd),
        },
        "build_dataset_usd": {"output_dir": str(tmp_path / "dataset-usd")},
        "identify_asset": {
            "identification": {"asset_type": "vehicle", "asset_subtype": "forklift"},
            "identification_path": str(tmp_path / "identification.json"),
        },
        "build_dataset_prepare_dataset": {
            "dataset_path": str(tmp_path / "dataset"),
            "dataset_jsonl_path": str(tmp_path / "dataset" / "dataset.jsonl"),
        },
        "predict": {
            "predictions_path": str(tmp_path / "predictions.jsonl"),
            "predictions_count": 1,
            "output_key": "classification",
        },
        "restore_usd": {
            "restored_predictions_path": str(tmp_path / "restored_predictions.jsonl"),
            "restore_success": True,
            "predictions_count": 1,
            "restore_stats": {"restored": 1},
        },
        "apply_physics": {
            "output_usd_path": str(tmp_path / "physics" / "original_physics.usda")
        },
    }

    class FakeWorkflow:
        def __init__(self, step_name: str) -> None:
            self.step_name = step_name

        def run(self, context: dict[str, object]) -> dict[str, object]:
            config_dict = context["config_dict"]
            assert isinstance(config_dict, dict)
            recorded_configs[self.step_name] = deepcopy(config_dict)
            return step_results[self.step_name]

    import physics_agent.workflows as workflows_module

    monkeypatch.setattr(
        workflows_module,
        "create_optimize_usd_workflow_from_config",
        lambda: FakeWorkflow("optimize_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_usd_data_preparation_workflow_from_config",
        lambda: FakeWorkflow("build_dataset_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_identify_asset_workflow_from_config",
        lambda: FakeWorkflow("identify_asset"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_prepare_dataset_workflow_from_config",
        lambda: FakeWorkflow("build_dataset_prepare_dataset"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_prediction_workflow_from_config",
        lambda: FakeWorkflow("predict"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_restore_usd_workflow_from_config",
        lambda: FakeWorkflow("restore_usd"),
    )
    monkeypatch.setattr(
        workflows_module,
        "create_apply_physics_workflow_from_config",
        lambda: FakeWorkflow("apply_physics"),
    )

    context = {
        "steps_to_run": [
            "optimize_usd",
            "build_dataset_usd",
            "identify_asset",
            "build_dataset_prepare_dataset",
            "predict",
            "restore_usd",
            "apply_physics",
        ],
        "step_configs": {
            "optimize_usd": {"renderer": {"backend": "remote", "_internal": "omit"}},
            "build_dataset_usd": {"usd_path": str(original_usd)},
            "identify_asset": {"usd_path": str(original_usd)},
            "build_dataset_prepare_dataset": {
                "prompts": {"system": "Classify the part."}
            },
            "predict": {"report": {"image_format": "jpeg", "image_quality": 72}},
            "restore_usd": {},
            "apply_physics": {"usd_path": str(original_usd)},
        },
        "working_dir": str(tmp_path / "run"),
        "session_id": "session-1",
        "project_name": "physics-demo",
        "event_listener": listener,
    }

    result = executor.run(context)

    assert result["pipeline_state"] == "completed"
    assert result["pipeline_results"]["predict"]["predictions_count"] == 1
    assert recorded_configs["build_dataset_usd"]["usd_path"] == str(optimize_output)
    assert recorded_configs["identify_asset"]["usd_path"] == str(optimize_output)
    assert recorded_configs["build_dataset_prepare_dataset"]["prompts"][
        "system"
    ].startswith("This is a vehicle (forklift). ")
    assert recorded_configs["predict"]["report"]["image_format"] == "jpeg"
    assert recorded_configs["restore_usd"]["original_usd_path"] == str(original_usd)
    assert recorded_configs["restore_usd"]["predictions_path"] == str(
        tmp_path / "predictions.jsonl"
    )
    assert (
        recorded_configs["restore_usd"]["optimization_metadata"]["optimization_config"][
            "scene_optimizer_settings"
        ]["output_format"]
        == "usdc"
    )
    assert recorded_configs["restore_usd"]["output_predictions_path"].endswith(
        "restored_predictions.jsonl"
    )
    assert recorded_configs["apply_physics"]["usd_path"] == str(optimize_output)
    assert recorded_configs["apply_physics"]["predictions_path"] == str(
        tmp_path / "predictions.jsonl"
    )
    assert any(event[0] == "pipeline.completed" for event in listener.events)


def test_unified_pipeline_executor_checkpoint_serializes_token_usage(
    tmp_path: Path, monkeypatch
):
    executor = UnifiedPipelineExecutorTask()
    token_usage = TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        model_name="qwen-test",
        invocation_type="vlm",
    )

    class FakeWorkflow:
        def run(self, context: dict[str, object]) -> dict[str, object]:
            return {
                "predictions_path": str(tmp_path / "predictions.jsonl"),
                "predictions_count": 1,
                "failed_count": 0,
                "token_stats": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 20,
                    "total_tokens": 120,
                    "invocation_count": 1,
                    "by_model": {
                        "qwen-test": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "count": 1,
                        }
                    },
                    "by_type": {
                        "vlm": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "count": 1,
                        }
                    },
                    "all_usages": [token_usage.to_dict()],
                },
                "output_key": "classification",
            }

    import physics_agent.workflows as workflows_module

    monkeypatch.setattr(
        workflows_module,
        "create_prediction_workflow_from_config",
        lambda: FakeWorkflow(),
    )

    context = {
        "steps_to_run": ["predict"],
        "step_configs": {"predict": {}},
        "working_dir": str(tmp_path / "run"),
        "session_id": "session-1",
        "project_name": "physics-demo",
    }

    result = executor.run(context)
    state = json.loads(
        (tmp_path / "run" / ".pipeline_state.json").read_text(encoding="utf-8")
    )

    assert result["pipeline_state"] == "completed"
    saved_usage = state["step_outputs"]["predict"]["token_stats"]["all_usages"][0]
    assert saved_usage["model_name"] == "qwen-test"
