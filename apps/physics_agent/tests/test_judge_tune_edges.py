# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for ``physics_agent.tasks.judge_tune``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tasks.judge_tune import (
    _best_trial_summary,
    _build_llm_prompt,
    _coerce_jsonable_number,
    _jsonable,
    _load_trajectory_jsonl,
    _parse_strict_json,
    _scenario_judge_config,
    _score_programmatic,
    _score_vlm,
    _summarise_reasoning,
)
from physics_agent.tuning.types import Scenario, TrialRecord, TunableParam
from physics_agent.tuning.visual_evidence import (
    DEFAULT_JUDGE_GENERATED_FRAMES,
    DEFAULT_JUDGE_REFERENCE_FRAMES,
    sample_evenly,
)


class _VLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.response


def _scenario(*, extra: dict[str, Any] | None = None) -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),
            TunableParam(name="restitution", min_value=0.0, max_value=1.0),
        ),
        target={"drop_height_m": 0.5},
        metric="settle_distance",
        extra=extra or {},
    )


def _trial(
    score: float = 0.1,
    *,
    failed: bool = False,
    backend_metrics: dict[str, Any] | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_index=0,
        params={"mass_scale": 1.0, "restitution": 0.5},
        score=score,
        backend_metrics=backend_metrics or {},
        duration_seconds=0.0,
        failed=failed,
    )


def test_programmatic_score_and_sampling_edges() -> None:
    score, critique = _score_programmatic(
        _scenario(),
        [],
        {"mass_scale": "heavy", "restitution": math.inf},  # type: ignore[dict-item]
    )
    assert score < 1.0
    assert "not numeric" in critique
    assert "not finite" in critique
    assert "no trials" in critique

    assert sample_evenly([1, 2, 3], 0) == []
    assert sample_evenly([1, 2, 3], 1) == [1]


def test_vlm_score_parsing_and_reasoning_edges() -> None:
    non_numeric = _VLM('{"score": "high", "reasoning": "bad"}')
    score, critique, unavailable = _score_vlm(
        scenario=_scenario(),
        history=[],
        best_params={},
        user_prompt=None,
        vlm_model=non_numeric,
        visual_evidence=None,
        programmatic_score=0.25,
        judge_max_tokens=None,
        judge_temperature=None,
        judge_reference_frames=DEFAULT_JUDGE_REFERENCE_FRAMES,
        judge_generated_frames=DEFAULT_JUDGE_GENERATED_FRAMES,
    )
    assert (score, unavailable) == (0.25, True)
    assert "non-numeric" in critique

    non_finite = _VLM('{"score": NaN, "reasoning": "bad"}')
    score, critique, unavailable = _score_vlm(
        scenario=_scenario(),
        history=[],
        best_params={},
        user_prompt=None,
        vlm_model=non_finite,
        visual_evidence=None,
        programmatic_score=0.25,
        judge_max_tokens=None,
        judge_temperature=None,
        judge_reference_frames=DEFAULT_JUDGE_REFERENCE_FRAMES,
        judge_generated_frames=DEFAULT_JUDGE_GENERATED_FRAMES,
    )
    assert (score, unavailable) == (0.25, True)
    assert "non-finite" in critique

    blank_reason = _VLM('{"score": 0.8, "reasoning": "   "}')
    score, critique, unavailable = _score_vlm(
        scenario=_scenario(),
        history=[],
        best_params={},
        user_prompt=None,
        vlm_model=blank_reason,
        visual_evidence=None,
        programmatic_score=0.25,
        judge_max_tokens=None,
        judge_temperature=None,
        judge_reference_frames=DEFAULT_JUDGE_REFERENCE_FRAMES,
        judge_generated_frames=DEFAULT_JUDGE_GENERATED_FRAMES,
    )
    assert (score, unavailable) == (0.8, False)
    assert critique == "(no VLM reasoning provided)"

    long_reason = _VLM(json.dumps({"score": 0.8, "reasoning": "x" * 600}))
    score, critique, unavailable = _score_vlm(
        scenario=_scenario(),
        history=[],
        best_params={},
        user_prompt=None,
        vlm_model=long_reason,
        visual_evidence=None,
        programmatic_score=0.25,
        judge_max_tokens=None,
        judge_temperature=None,
        judge_reference_frames=DEFAULT_JUDGE_REFERENCE_FRAMES,
        judge_generated_frames=DEFAULT_JUDGE_GENERATED_FRAMES,
    )
    assert (score, unavailable) == (0.8, False)
    assert len(critique) == 500
    assert critique.endswith("...")


def test_judge_config_json_and_reasoning_helpers() -> None:
    assert _scenario_judge_config(_scenario(extra={"judge": None})) == {}
    with pytest.raises(ValueError, match="must be a mapping"):
        _scenario_judge_config(_scenario(extra={"judge": "bad"}))

    assert _parse_strict_json("prefix {bad json} suffix") is None
    assert _coerce_jsonable_number(object()).__class__ is object
    assert _coerce_jsonable_number(float("nan")) == "NaN"
    assert _coerce_jsonable_number(float("inf")) == "Infinity"
    assert _jsonable("ready") == "ready"
    assert _jsonable({"x": object()})["x"].startswith("<object object")

    text = _summarise_reasoning(
        decision="continue",
        combined=0.1,
        prog_score=0.1,
        llm_score=0.1,
        prog_critique="p" * 600,
        llm_critique="l" * 600,
        llm_unavailable=False,
    )
    assert len(text) == 500
    assert text.endswith("...")


def test_trajectory_jsonl_and_best_summary_edges(tmp_path: Path) -> None:
    jsonl = tmp_path / "trajectory.jsonl"
    jsonl.write_text(
        "\n"
        "not json\n"
        + json.dumps({"t": "bad", "pose": [], "vel": []})
        + "\n"
        + json.dumps({"t": 0.0, "pose": [0.0] * 7, "vel": [0.0] * 6})
        + "\n",
        encoding="utf-8",
    )
    assert _load_trajectory_jsonl(str(jsonl)) == [(0.0, [0.0] * 7, [0.0] * 6)]

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert (
        _best_trial_summary(_trial(backend_metrics={"trajectory_jsonl": str(empty)}))
        is None
    )

    summary = _best_trial_summary(
        _trial(
            backend_metrics={
                "trajectory_jsonl": str(jsonl),
                "world_up": ["not", "numeric", "payload"],
                "rest_position": [0.0, 0.0, 1.0],
            }
        )
    )
    assert summary is not None
    assert summary["n_samples"] == 1


def test_prompt_helpers_optional_fields(tmp_path: Path) -> None:
    prompt = _build_llm_prompt(
        scenario=_scenario(),
        history=[_trial(failed=True)],
        best_params={"mass_scale": 1.0},
        user_prompt="goal",
        prior_refine_history=[{"iteration": 1}],
    )
    payload = json.loads(prompt[prompt.index("{") :])
    assert payload["prior_refine_history"] == [{"iteration": 1}]
    assert payload["user_prompt"] == "goal"
