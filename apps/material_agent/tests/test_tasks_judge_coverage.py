# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional branch coverage for material-agent judge tasks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from material_agent.tasks import judge as judge_module
from material_agent.tasks.judge import JudgeTask


def test_prediction_analysis_handles_empty_predictions_and_dataset_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = JudgeTask()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(judge_module, "load_predictions", lambda path: [])

    score, critique = task._run_prediction_analysis(
        {"predictions_path": str(predictions_path)},
        {},
    )

    assert score == 1.0
    assert critique == ""

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        judge_module,
        "load_predictions",
        lambda path: [{"prim_path": "/World/A", "material": "Steel"}],
    )
    monkeypatch.setattr(
        judge_module,
        "load_prims_metadata",
        lambda path: [{"prim_path": "/World/A", "bbox_center": [0, 0, 0]}],
    )

    class FakeAnalyzer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def analyze(self):
            return SimpleNamespace(
                symmetry_pairs=[],
                symmetry_violations=[],
                consistency_violations=[],
                prim_feedback={},
                resolved_assignments={"/World/A": "Steel"},
                score=0.75,
                critique="consistent enough",
            )

    monkeypatch.setattr(judge_module, "PredictionAnalyzer", FakeAnalyzer)
    context = {
        "predictions_path": str(predictions_path),
        "dataset_path": str(dataset_path),
    }

    score, critique = task._run_prediction_analysis(
        context,
        {"symmetry_tolerance": 2.0},
    )

    assert score == 0.75
    assert critique == "consistent enough"
    assert context["resolved_assignments"] == {"/World/A": "Steel"}


def test_vlm_judge_validation_and_image_path_branches(tmp_path: Path) -> None:
    task = JudgeTask()
    ref_dir = tmp_path / "config"
    ref_dir.mkdir()
    ref = ref_dir / "ref.png"
    ref.touch()
    rendered = tmp_path / "rendered.png"
    rendered.touch()

    with pytest.raises(ValueError, match="VLM is required"):
        task._run_vlm_judge({}, {"reference_images": [str(ref)]}, 1)

    with pytest.raises(ValueError, match="Rendered images are required"):
        task._run_vlm_judge(
            {"vlm": object()},
            {"reference_images": [str(ref)]},
            1,
        )

    with pytest.raises(ValueError, match="Reference images are required"):
        task._run_vlm_judge(
            {"vlm": object(), "rendered_image_paths": [str(rendered)]},
            {},
            1,
        )

    class FakeVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_with_image_caption_pairs(self, **kwargs):
            self.calls.append(kwargs)
            return "Critique: Looks coherent.\nScore: 8\nDecision: APPROVE"

    vlm = FakeVLM()
    result = task._run_vlm_judge(
        {
            "vlm_judge": vlm,
            "vlm_judge_config": {"temperature": 0.2, "max_tokens": 32},
            "rendered_image_path": str(rendered),
            "config_path": str(ref_dir / "config.yaml"),
        },
        {"reference_images": ["missing.png", "ref.png"]},
        1,
    )

    assert result.decision == "approve"
    assert result.score == 0.8
    assert vlm.calls[0]["temperature"] == 0.2
    assert vlm.calls[0]["max_tokens"] == 32
    assert vlm.calls[0]["image_caption_pairs"] == [
        ("Reference Image 1:", str(ref_dir / "ref.png")),
        ("Rendered 3D Model (Current Result) - View 1:", str(rendered)),
    ]

    second_vlm = FakeVLM()
    result = task._run_vlm_judge(
        {
            "vlm": second_vlm,
            "rendered_image_paths": [
                str(tmp_path / "missing-render.png"),
                str(rendered),
            ],
            "vlm_config": {},
        },
        {"reference_images": [str(ref)]},
        1,
    )

    assert result.decision == "approve"
    assert second_vlm.calls[0]["temperature"] == 0.1
    assert second_vlm.calls[0]["max_tokens"] == 2048
    assert second_vlm.calls[0]["image_caption_pairs"] == [
        ("Reference Image 1:", str(ref)),
        ("Rendered 3D Model (Current Result) - View 1:", str(rendered)),
    ]


def test_parse_vlm_critique_defaults_missing_score_and_truncates_reasoning() -> None:
    long_critique = "Critique: " + ("x" * 260) + "\nDecision: APPROVE"

    decision, score, reasoning, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        long_critique,
        iteration_count=1,
        score_threshold=0.4,
    )

    assert decision == "approve"
    assert score == 0.5
    assert len(reasoning) == 200
    assert reasoning.endswith("...")
    assert decision_parsed is True
