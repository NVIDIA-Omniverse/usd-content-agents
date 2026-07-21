# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from material_agent.benchmark import BaseBenchmark, VLMBenchmark, create_benchmark


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[object, dict[str, object]]] = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return FakeResponse(self._responses.pop(0))


def test_vlm_benchmark_run_inference_and_evaluate_with_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "ground_truth": "steel"}),
                json.dumps({"id": "b", "ground_truth": "plastic"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "material_agent.benchmark.console.print", lambda *args, **kwargs: None
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").write_text("stale\n", encoding="utf-8")

    def fake_batch_assign_materials(**kwargs):
        kwargs["on_progress"]("a", "steel")
        kwargs["on_error"]("b", "bad input")
        return [
            {"id": "a", "status": "success", "vlm_response": "steel"},
            {"id": "b", "status": "error", "error": "bad input"},
        ]

    monkeypatch.setattr(
        "material_agent.benchmark.batch_assign_materials",
        fake_batch_assign_materials,
    )

    judge = FakeJudge(
        [
            '{"score": 5, "explanation": "exact match"}',
            "score: 3 explanation: acceptable alternative",
        ]
    )
    benchmark = VLMBenchmark(
        vlm=object(),
        llm_judge=judge,
        vlm_temperature=0.1,
        vlm_max_tokens=256,
        llm_judge_temperature=0.2,
        llm_judge_max_tokens=128,
        system_prompt="Assign materials.",
    )

    predictions_path = benchmark.run_inference(dataset_path, output_dir)
    saved_predictions = [
        json.loads(line) for line in predictions_path.read_text().splitlines()
    ]

    assert predictions_path.exists()
    assert saved_predictions == [
        {"id": "a", "vlm_response": "steel", "ground_truth": "steel"}
    ]

    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "a", "vlm_response": "steel", "ground_truth": "steel"}
                ),
                json.dumps(
                    {"id": "b", "vlm_response": "nylon", "ground_truth": "plastic"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = benchmark.evaluate_with_judge(predictions_path, tmp_path / "eval")

    assert metrics["functional_correctness_score"] == 4.0
    assert metrics["success_rate"] == 50.0
    assert metrics["score_distribution"] == {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}
    assert (tmp_path / "eval" / "evaluation_results.json").exists()
    assert judge.calls[0][1]["temperature"] == 0.2
    assert judge.calls[0][1]["max_tokens"] == 128


def test_base_benchmark_runs_inference_then_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "material_agent.benchmark.console.print", lambda *args, **kwargs: None
    )

    class FakeBenchmark(BaseBenchmark):
        def __init__(self) -> None:
            super().__init__(name="fake")
            self.calls: list[tuple[str, Path, Path | None]] = []

        def run_inference(
            self, dataset_path: Path, output_dir: Path | None = None
        ) -> Path:
            self.calls.append(("inference", dataset_path, output_dir))
            return output_dir / "predictions.jsonl" if output_dir else dataset_path

        def evaluate_with_judge(
            self, predictions_path: Path, output_dir: Path | None = None
        ) -> dict[str, object]:
            self.calls.append(("evaluate", predictions_path, output_dir))
            return {"predictions_path": str(predictions_path)}

    benchmark = FakeBenchmark()
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "results"

    metrics = benchmark.run_full_benchmark(dataset_path, output_dir)

    assert metrics == {"predictions_path": str(output_dir / "predictions.jsonl")}
    assert benchmark.calls == [
        ("inference", dataset_path, output_dir),
        ("evaluate", output_dir / "predictions.jsonl", output_dir),
    ]


def test_vlm_benchmark_default_output_dirs_and_no_valid_evaluations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"id": "a", "ground_truth": "steel"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "material_agent.benchmark.console.print", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "material_agent.benchmark.batch_assign_materials",
        lambda **kwargs: [{"id": "a", "status": "success", "vlm_response": "steel"}],
    )

    benchmark = VLMBenchmark(vlm=object(), llm_judge=FakeJudge([]))

    predictions_path = benchmark.run_inference(dataset_path)

    assert predictions_path == tmp_path / "predictions.jsonl"
    assert predictions_path.exists()

    predictions_path.write_text("", encoding="utf-8")

    assert benchmark.evaluate_with_judge(predictions_path) == {}


def test_vlm_benchmark_helpers_and_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(
        "material_agent.benchmark.console.print", lambda *args, **kwargs: None
    )
    benchmark = VLMBenchmark(vlm=object(), llm_judge=FakeJudge([]))

    metrics = benchmark._calculate_metrics(
        [5, 4, 2],
        [
            {"judge_score": 5},
            {"judge_score": 4},
            {"judge_score": 2},
        ],
    )
    assert metrics == {
        "functional_correctness_score": 3.67,
        "success_rate": 66.7,
        "total_cases": 3,
        "successful_cases": 2,
        "score_distribution": {1: 0, 2: 1, 3: 0, 4: 1, 5: 1},
        "failure_count": 1,
    }
    benchmark._display_metrics(metrics)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        created = create_benchmark("vlm", vlm=object(), llm_judge=FakeJudge([]))

    assert isinstance(created, VLMBenchmark)
    assert caught
    assert "deprecated" in str(caught[0].message).lower()

    with pytest.raises(ValueError, match="Unknown benchmark type"):
        create_benchmark("unknown", vlm=object(), llm_judge=FakeJudge([]))
