# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional coverage for judge evaluation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from world_understanding.utils.object_store import InMemoryObjectStore

from material_agent.tasks.evaluation import EvaluationTask


class _FakeJudge:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_evaluation_run_enriches_from_dataset_and_stores_results(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    dataset_path = tmp_path / "dataset.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/A",
                "materials": {
                    "material": "Steel",
                    "original_response": "Steel because the part is metallic.",
                },
                "prompt": "N/A",
            },
            {"id": "/B", "materials": {"material": "Rubber"}},
            {
                "id": "/C",
                "materials": "Plastic",
                "ground_truth": "Plastic",
            },
            {"id": "/D", "materials": {"material": "Glass"}},
        ],
    )
    _write_jsonl(
        dataset_path,
        [
            {"id": "/A", "ground_truth": "Steel", "text": "Classify A"},
            {"id": "/B", "text": "No ground truth"},
        ],
    )
    judge = _FakeJudge(
        'prefix {"score": 5, "explanation": "exact"}',
        '{"score": 1, "explanation": "missing ground truth"}',
        "score: 3 plausible",
    )
    store = InMemoryObjectStore()

    result = EvaluationTask().run(
        {
            "llm_judge": judge,
            "llm_judge_config": {"temperature": 0.2, "max_tokens": 64},
            "predictions_path": str(predictions_path),
            "dataset_path": str(dataset_path),
            "output_dir": tmp_path / "eval",
        },
        store,
    )

    assert result["evaluation_complete"] is True
    assert Path(result["evaluation_path"]).exists()
    assert store.exists("evaluations")
    evaluations = store.get("evaluations")
    assert [evaluation["id"] for evaluation in evaluations] == ["/A", "/B", "/C"]
    assert evaluations[0]["exact_match"] is True
    assert evaluations[1]["ground_truth"] == ""
    assert evaluations[2]["score"] == 3
    assert judge.calls[0]["kwargs"] == {"temperature": 0.2, "max_tokens": 64}


def test_evaluation_run_uses_constructor_defaults_and_prediction_directory(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    missing_dataset = tmp_path / "missing_dataset.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/A",
                "materials": {"material": "Steel"},
                "ground_truth": "Steel",
            }
        ],
    )
    judge = _FakeJudge('{"score": 4, "explanation": "close"}')

    result = EvaluationTask(
        llm_judge=judge,
        dataset_path=missing_dataset,
        temperature=0.0,
        max_tokens=32,
        success_threshold=3.5,
    ).run({"predictions_path": str(predictions_path)})

    assert Path(result["evaluation_path"]).parent == predictions_path.parent
    assert result["metrics"]["successful_cases"] == 1
    assert judge.calls[0]["kwargs"] == {"temperature": 0.0, "max_tokens": 32}


def test_evaluation_requires_judge(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, [])

    with pytest.raises(ValueError, match="llm_judge not provided"):
        EvaluationTask().run({"predictions_path": str(predictions_path)})


def test_evaluate_single_handles_unparseable_and_failed_judge() -> None:
    listener = MagicMock()
    task = EvaluationTask()

    unparseable = task._evaluate_single(
        {
            "id": "/A",
            "materials": {"material": "Steel"},
            "ground_truth": "Steel",
        },
        _FakeJudge("no score here"),
        temperature=None,
        max_tokens=None,
        listener=listener,
    )
    assert unparseable["score"] == 0
    assert unparseable["exact_match"] is True

    failed = task._evaluate_single(
        {"id": "/B", "ground_truth": "Rubber"},
        _FakeJudge(RuntimeError("judge down")),
        temperature=None,
        max_tokens=None,
        listener=listener,
    )
    assert failed["score"] == 0
    assert failed["predicted_material"] == ""
    assert "judge down" in failed["explanation"]
