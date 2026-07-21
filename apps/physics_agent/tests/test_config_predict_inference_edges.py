# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for predict config and VLM inference tasks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import physics_agent.tasks.inference as inference_mod
from physics_agent.tasks.config_predict import PredictConfigTask
from physics_agent.tasks.inference import VLMInferenceTask


class _Store:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def exists(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


class _VLM:
    pass


def _write_dataset(path: Path, count: int = 1) -> list[dict[str, Any]]:
    rows = [
        {"id": f"asset-{idx}", "image_path": f"img-{idx}.png"} for idx in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_predict_config_dataset_and_working_dir_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    out_dir = tmp_path / "predict-out"

    context = PredictConfigTask().run(
        {
            "config_dict": {"vlm": {}},
            "dataset_override": str(dataset),
            "output_dir_override": str(out_dir),
        }
    )
    assert context["dataset_path"] == str(dataset)
    assert context["output_dir"] == str(out_dir)

    working_dir = tmp_path / "work"
    fallback_dataset = working_dir / "dataset" / "dataset.jsonl"
    _write_dataset(fallback_dataset)
    context = PredictConfigTask().run(
        {
            "config_dict": {
                "project": {"working_dir": str(working_dir)},
                "vlm": {},
            }
        }
    )
    assert context["dataset_path"] == str(fallback_dataset)

    task = PredictConfigTask()
    config_path = tmp_path / "configs" / "predict.yaml"
    config_path.parent.mkdir()
    assert (
        task._derive_working_dir(
            {"project": {"working_dir": "relative-work"}},
            config_path,
        )
        == (config_path.parent / "relative-work").resolve()
    )
    assert task._derive_working_dir({}, None) is None

    import physics_agent.config.path_resolver as path_resolver_mod

    monkeypatch.setattr(
        path_resolver_mod,
        "ProjectPathResolver",
        lambda *_args, **_kwargs: SimpleNamespace(working_dir=tmp_path / "resolved"),
    )
    assert task._derive_working_dir({"project": {}}, config_path) == (
        tmp_path / "resolved"
    )

    monkeypatch.setattr(
        path_resolver_mod,
        "ProjectPathResolver",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad config")),
    )
    assert task._derive_working_dir({"project": {}}, config_path) is None


def test_vlm_inference_file_dataset_implicit_output_and_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    rows = _write_dataset(dataset_path, count=10)
    implicit_predictions = dataset_path.parent / "output" / "predictions.jsonl"
    implicit_predictions.parent.mkdir()
    implicit_predictions.write_text('{"id":"stale"}\n', encoding="utf-8")

    def fake_batch(**kwargs: Any) -> list[dict[str, Any]]:
        results = []
        for entry in kwargs["entries"]:
            kwargs["on_progress"](entry["id"], "ok")
            result = {
                "status": "success",
                "id": entry["id"],
                "vlm_response": {"label": "ok"},
            }
            kwargs["on_prediction"](entry["id"], {"classification": {"label": "ok"}})
            kwargs["on_result"](result, entry)
            results.append(result)
        implicit_predictions.write_text(
            implicit_predictions.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return results

    monkeypatch.setattr(inference_mod, "batch_classify_assets", fake_batch)
    result = VLMInferenceTask(vlm=_VLM()).run(
        {
            "dataset_path": str(dataset_path),
            "stream_predictions": True,
        }
    )
    assert result["predictions_path"] == str(implicit_predictions)
    assert result["predictions_count"] == len(rows)
    assert "stale" not in implicit_predictions.read_text(encoding="utf-8")


def test_vlm_inference_on_result_skip_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_dataset(dataset_path)

    def fake_non_streaming(**kwargs: Any) -> list[dict[str, Any]]:
        entry = kwargs["entries"][0]
        kwargs["on_result"](
            {"status": "success", "id": entry["id"], "vlm_response": {}},
            entry,
        )
        return [{"status": "success", "id": entry["id"], "vlm_response": {}}]

    monkeypatch.setattr(inference_mod, "batch_classify_assets", fake_non_streaming)
    result = VLMInferenceTask(vlm=_VLM()).run(
        {
            "dataset_path": str(dataset_path),
            "predictions_path": str(tmp_path / "nonstream.jsonl"),
            "stream_predictions": False,
        }
    )
    assert result["predictions_count"] == 1

    def fake_error_result(**kwargs: Any) -> list[dict[str, Any]]:
        entry = kwargs["entries"][0]
        kwargs["on_result"]({"status": "error", "id": entry["id"]}, entry)
        return []

    monkeypatch.setattr(inference_mod, "batch_classify_assets", fake_error_result)
    result = VLMInferenceTask(vlm=_VLM()).run(
        {
            "dataset_path": str(dataset_path),
            "predictions_path": str(tmp_path / "empty.jsonl"),
            "stream_predictions": True,
            "allow_empty_predictions": True,
        }
    )
    assert result["predictions_count"] == 0


def test_vlm_inference_empty_resume_touches_prediction_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text("\n", encoding="utf-8")
    monkeypatch.setattr(
        inference_mod,
        "batch_classify_assets",
        lambda **_kwargs: [],
    )
    store = _Store({"dataset": [{"id": "asset"}]})
    result = VLMInferenceTask(vlm=_VLM()).run(
        {
            "output_dir": str(output_dir),
            "stream_predictions": True,
            "resume": True,
            "allow_empty_predictions": True,
        },
        object_store=store,
    )
    assert result["predictions_count"] == 0
    assert store.values["predictions"] == []
    assert predictions_path.exists()


def test_vlm_inference_validates_required_context() -> None:
    with pytest.raises(ValueError, match="VLM not provided"):
        VLMInferenceTask().run({"dataset_path": "unused.jsonl"})
    with pytest.raises(ValueError, match="dataset_path not found"):
        VLMInferenceTask(vlm=_VLM()).run({})
