# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.tasks.iterative_physics_refinement``.

The orchestrator is exercised with a stub ``run_tune`` so the loop logic
is testable without spinning OvPhysX. We verify:

* Approve at iteration 1 short-circuits the loop (no refine call).
* Continue causes scenario_refine to fire and the next iteration loads
  the refined YAML.
* Hitting max_iterations terminates with ``termination_reason="max_iterations"``.
* The on-disk layout (``iter_N/`` + ``final/`` + ``refine_summary.json``)
  matches the spec.
* Loop emits ``continue_iteration`` / ``judge_score`` / ``judge_reasoning``
  on the listener context (material-loop contract parity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from physics_agent.tasks.iterative_physics_refinement import (
    IterationRecord,
    IterativePhysicsRefinementTask,
    _compact_refine_history,
)
from physics_agent.tuning.types import (
    Scenario,
    TrialRecord,
    TunableParam,
    TuneInput,
    TuneOutput,
)


def _scenario(metric: str = "settle_distance") -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.95),
            TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),
        ),
        target={"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        metric=metric,
    )


def _scenario_yaml_dict(metric: str = "settle_distance") -> dict[str, Any]:
    return {
        "name": "drop_settle",
        "metric": metric,
        "target": {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        "parameters": [
            {"name": "restitution", "min": 0.4, "max": 0.95},
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
        ],
    }


def _freeform_scenario_yaml_dict() -> dict[str, Any]:
    return {
        "name": "freeform",
        "metric": "judge_score",
        "target": {
            "description": "realistic generated rollout",
            "duration_s": 2.0,
            "observations": ["the object behaves realistically"],
        },
        "parameters": [
            {"name": "restitution", "min": 0.0, "max": 1.0},
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
        ],
    }


def _trial(idx: int, score: float, **extra: Any) -> TrialRecord:
    bm = dict(extra)
    return TrialRecord(
        trial_index=idx,
        params={"restitution": 0.7, "mass_scale": 1.0},
        score=score,
        backend_metrics=bm,
        duration_seconds=0.0,
        failed=False,
    )


def _make_tune_output(
    output_dir: Path,
    *,
    history: list[TrialRecord],
    best_score: float = 0.05,
    best_params: dict[str, float] | None = None,
) -> TuneOutput:
    return TuneOutput(
        success=True,
        output_dir=output_dir,
        best_params=best_params or {"restitution": 0.7, "mass_scale": 1.0},
        best_score=best_score,
        n_trials=len(history),
        optimizer_used="random",
        engine_used="fake",
        history=history,
        artifacts={},
        cancelled=False,
        needs_refinement=False,
    )


def _json_payload_from_prompt(prompt: str) -> dict[str, Any]:
    return json.loads(prompt[prompt.index("{") :])


# ---------------------------------------------------------------------------
# Fake run_tune that records calls and returns canned outputs
# ---------------------------------------------------------------------------


class _FakeRunTune:
    def __init__(self, scripted_outputs: list[Any] | None = None) -> None:
        # Each scripted output may be either a TuneOutput or a callable
        # ``(params: TuneInput) -> TuneOutput``.
        self.scripted = list(scripted_outputs or [])
        self.calls: list[TuneInput] = []

    def __call__(self, params: TuneInput) -> TuneOutput:
        self.calls.append(params)
        # Materialize the scenario YAML the orchestrator wrote (so
        # we mimic the real runner reading from disk).
        if not self.scripted:
            history = [_trial(0, 0.05, settle_distance=0.05)]
            return _make_tune_output(
                Path(params.output_dir),
                history=history,
                best_score=0.05,
            )
        nxt = self.scripted.pop(0)
        if callable(nxt):
            return nxt(params)
        return nxt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _write_initial_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(_scenario_yaml_dict()), encoding="utf-8")
    return p


def _write_freeform_initial_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(_freeform_scenario_yaml_dict()), encoding="utf-8")
    return p


class _DefaultJudgeVLM:
    def generate_with_image_caption_pairs(self, **_kwargs: Any) -> str:
        return json.dumps(
            {
                "score": 1.0,
                "decision": "approve",
                "reasoning": "default test judge",
            }
        )


@pytest.fixture(autouse=True)
def _patch_default_judge_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.resolve_default_judge_vlm",
        lambda: _DefaultJudgeVLM(),
    )


def test_constructor_rejects_empty_user_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        IterativePhysicsRefinementTask(
            user_prompt="   ",
            initial_scenario=_scenario_yaml_dict(),
            physics_usd=tmp_path / "fake.usda",
            output_dir=tmp_path / "out",
            run_tune_callable=_FakeRunTune(),
        )


def test_constructor_rejects_zero_iterations(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        IterativePhysicsRefinementTask(
            user_prompt="bouncy",
            initial_scenario=_scenario_yaml_dict(),
            physics_usd=tmp_path / "fake.usda",
            output_dir=tmp_path / "out",
            max_iterations=0,
            run_tune_callable=_FakeRunTune(),
        )


def test_constructor_rejects_negative_history_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        IterativePhysicsRefinementTask(
            user_prompt="bouncy",
            initial_scenario=_scenario_yaml_dict(),
            physics_usd=tmp_path / "fake.usda",
            output_dir=tmp_path / "out",
            history_window=-1,
            run_tune_callable=_FakeRunTune(),
        )


def test_approve_at_first_iteration_short_circuits(tmp_path: Path) -> None:
    """Score >= threshold → judge returns approve → loop exits after 1 iter,
    refine is NOT called, final/ is populated."""
    initial = _write_initial_yaml(tmp_path)
    worse_recording = tmp_path / "worse-recording.usd"
    worse_recording.write_bytes(b"must not be selected")
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[
                    _trial(0, 0.0, settle_distance=0.0),  # perfect → approve
                    _trial(
                        1,
                        0.1,
                        settle_distance=0.1,
                        recording_usd=str(worse_recording),
                    ),
                ],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=3,
        score_threshold=0.7,
        chat_model=None,  # refine degrades; judge has no VLM
        run_tune_callable=fake,
    )
    result = task.run({})
    assert len(result.iterations) == 1
    assert result.termination_reason == "approved"
    assert result.iterations[0].judge_decision == "approve"
    # Only 1 tune call ever issued.
    assert len(fake.calls) == 1
    # final/ is populated
    final_dir = tmp_path / "out" / "final"
    assert final_dir.exists()
    assert (final_dir / "scenario.yaml").exists()
    assert (final_dir / "judge_result.json").exists()
    assert result.final_recording_usd is None
    assert result.final_recording_error == (
        "winning trial did not persist recording_usd"
    )


def test_continue_then_approve_runs_refine_between(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iteration 1 returns continue (judge would say so via low param plausibility);
    refine produces a new scenario; iteration 2 approves. Two tune calls fire and
    the iter_2/scenario.yaml differs from iter_1/scenario.yaml."""
    initial = _write_initial_yaml(tmp_path)

    fake_runner = _FakeRunTune()
    # Iteration 1: BOTH best params OUTSIDE bounds → programmatic_score
    # drops to 0.4 → combined < 0.7 → continue.
    fake_runner.scripted.append(
        _make_tune_output(
            tmp_path / "out" / "iter_1",
            history=[_trial(0, 0.5, settle_distance=0.5)],
            best_score=0.5,
            best_params={"restitution": 10.0, "mass_scale": 100.0},
        )
    )
    # Iteration 2: best params back in bounds + zero settle distance.
    fake_runner.scripted.append(
        _make_tune_output(
            tmp_path / "out" / "iter_2",
            history=[_trial(0, 0.0, settle_distance=0.0)],
            best_score=0.0,
            best_params={"restitution": 0.9, "mass_scale": 1.0},
        )
    )

    # Stub the refine LLM so it produces a valid refined scenario with
    # a different metric (proves cross-iteration scenario actually changed).
    refined_yaml_dict = _scenario_yaml_dict(metric="max_bounce_height")
    refine_prompts: list[dict[str, str | None]] = []

    def fake_refine_chat(
        chat_model: Any, prompt: str, *, system_prompt: str | None = None
    ) -> dict[str, Any]:
        refine_prompts.append({"prompt": prompt, "system_prompt": system_prompt})
        return {
            "response": json.dumps(
                {
                    "scenario": refined_yaml_dict,
                    "reasoning": "swap to max_bounce_height for bouncier eval",
                }
            )
        }

    class StubJudgeVLM:
        def __init__(self) -> None:
            self.scores = iter((0.0, 1.0))

        def generate_with_image_caption_pairs(self, **_kwargs: Any) -> str:
            score = next(self.scores)
            decision = "continue" if score < 0.7 else "approve"
            return json.dumps(
                {
                    "score": score,
                    "decision": decision,
                    "reasoning": "stubbed judge",
                }
            )

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake_refine_chat,
    )
    judge_vlm = StubJudgeVLM()

    task = IterativePhysicsRefinementTask(
        user_prompt="make it bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=3,
        score_threshold=0.7,
        chat_model=object(),  # any non-None triggers the refine LLM path
        vlm_model=judge_vlm,
        run_tune_callable=fake_runner,
    )
    result = task.run({})
    assert len(fake_runner.calls) == 2
    assert len(result.iterations) == 2
    assert result.iterations[0].judge_decision == "continue"
    assert result.iterations[1].judge_decision == "approve"
    assert result.termination_reason == "approved"

    iter_1_yaml = (tmp_path / "out" / "iter_1" / "scenario.yaml").read_text(
        encoding="utf-8"
    )
    iter_2_yaml = (tmp_path / "out" / "iter_2" / "scenario.yaml").read_text(
        encoding="utf-8"
    )
    assert iter_1_yaml != iter_2_yaml
    parsed_2 = yaml.safe_load(iter_2_yaml)
    assert parsed_2["metric"] == "max_bounce_height"
    assert refine_prompts
    assert "Active backend: ovphysx" in str(refine_prompts[0]["system_prompt"])
    assert "contact_ke" not in str(refine_prompts[0]["system_prompt"])

    # refine_result.json captured for iter_1.
    refine_path = tmp_path / "out" / "iter_1" / "refine_result.json"
    assert refine_path.exists()


def test_compact_prior_history_written_and_passed_to_judge_and_refiner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each iteration writes prior loop memory, and prompts receive only the
    compact recent rows plus the best-so-far row when it falls outside the
    recent window.
    """
    initial = _write_initial_yaml(tmp_path)
    out_dir = tmp_path / "out"
    fake_runner = _FakeRunTune(
        [
            _make_tune_output(
                out_dir / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            ),
            _make_tune_output(
                out_dir / "iter_2",
                history=[_trial(0, 0.2, settle_distance=0.2)],
                best_score=0.2,
                best_params={"restitution": 0.8, "mass_scale": 1.0},
            ),
            _make_tune_output(
                out_dir / "iter_3",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.95, "mass_scale": 1.0},
            ),
        ]
    )

    refine_prompts: list[str] = []

    def fake_refine_chat(
        chat_model: Any, prompt: str, *, system_prompt: str | None = None
    ) -> dict[str, Any]:
        refine_prompts.append(prompt)
        return {
            "response": json.dumps(
                {
                    "scenario": _scenario_yaml_dict(),
                    "reasoning": "keep scenario for test",
                }
            )
        }

    class StubJudgeVLM:
        def __init__(self) -> None:
            self.scores = iter((0.6, 0.1, 1.0))
            self.prompts: list[str] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.prompts.append(str(kwargs["final_prompt"]))
            score = next(self.scores)
            return json.dumps(
                {
                    "score": score,
                    "decision": "approve" if score >= 1.0 else "continue",
                    "reasoning": f"stubbed judge score {score}",
                }
            )

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake_refine_chat,
    )
    judge_vlm = StubJudgeVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="make it bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=out_dir,
        max_iterations=3,
        history_window=1,
        score_threshold=0.99,
        chat_model=object(),
        vlm_model=judge_vlm,
        run_tune_callable=fake_runner,
    )

    result = task.run({})

    assert result.termination_reason == "approved"
    assert len(result.iterations) == 3
    assert (
        json.loads((out_dir / "iter_1" / "prior_refine_history.json").read_text()) == []
    )

    iter_2_history = json.loads(
        (out_dir / "iter_2" / "prior_refine_history.json").read_text()
    )
    assert [row["iteration"] for row in iter_2_history] == [1]
    assert iter_2_history[0]["parameter_bounds"]["restitution"] == [0.4, 0.95]

    iter_3_history = json.loads(
        (out_dir / "iter_3" / "prior_refine_history.json").read_text()
    )
    assert [row["iteration"] for row in iter_3_history] == [1, 2]
    assert iter_3_history[0]["best_so_far"] is True
    assert iter_3_history[1]["best_so_far"] is False

    judge_iter_3_payload = json.loads(
        judge_vlm.prompts[2][judge_vlm.prompts[2].index("{") :]
    )
    assert [
        row["iteration"] for row in judge_iter_3_payload["prior_refine_history"]
    ] == [
        1,
        2,
    ]

    refine_iter_2_payload = json.loads(
        refine_prompts[1][refine_prompts[1].index("{") :]
    )
    assert [
        row["iteration"] for row in refine_iter_2_payload["prior_refine_history"]
    ] == [1]

    summary = json.loads((out_dir / "refine_summary.json").read_text(encoding="utf-8"))
    assert summary["compact_history"][0]["iteration"] == 3
    assert summary["compact_history"][0]["best_so_far"] is True


def test_history_window_zero_disables_prompt_history_but_keeps_summary_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _write_initial_yaml(tmp_path)
    out_dir = tmp_path / "out"
    fake_runner = _FakeRunTune(
        [
            _make_tune_output(
                out_dir / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            ),
            _make_tune_output(
                out_dir / "iter_2",
                history=[_trial(0, 0.2, settle_distance=0.2)],
                best_score=0.2,
                best_params={"restitution": 0.8, "mass_scale": 1.0},
            ),
            _make_tune_output(
                out_dir / "iter_3",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.95, "mass_scale": 1.0},
            ),
        ]
    )

    refine_prompts: list[str] = []

    def fake_refine_chat(
        chat_model: Any, prompt: str, *, system_prompt: str | None = None
    ) -> dict[str, Any]:
        refine_prompts.append(prompt)
        return {
            "response": json.dumps(
                {
                    "scenario": _scenario_yaml_dict(),
                    "reasoning": "keep scenario for test",
                }
            )
        }

    class StubJudgeVLM:
        def __init__(self) -> None:
            self.scores = iter((0.9, 0.5, 0.4))
            self.prompts: list[str] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.prompts.append(str(kwargs["final_prompt"]))
            score = next(self.scores)
            return json.dumps(
                {
                    "score": score,
                    "decision": "approve" if score >= 1.0 else "continue",
                    "reasoning": f"stubbed judge score {score}",
                }
            )

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake_refine_chat,
    )
    judge_vlm = StubJudgeVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="make it bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=out_dir,
        max_iterations=3,
        history_window=0,
        score_threshold=0.99,
        chat_model=object(),
        vlm_model=judge_vlm,
        run_tune_callable=fake_runner,
    )

    result = task.run({})

    assert result.termination_reason == "max_iterations"
    assert len(result.iterations) == 3
    for iteration in (1, 2, 3):
        prior_path = out_dir / f"iter_{iteration}" / "prior_refine_history.json"
        assert json.loads(prior_path.read_text(encoding="utf-8")) == []

    for prompt in judge_vlm.prompts:
        payload = _json_payload_from_prompt(prompt)
        assert payload["prior_refine_history"] == []
    for prompt in refine_prompts:
        payload = _json_payload_from_prompt(prompt)
        assert payload["prior_refine_history"] == []

    summary = json.loads((out_dir / "refine_summary.json").read_text(encoding="utf-8"))
    assert [row["iteration"] for row in summary["compact_history"]] == [1]
    assert summary["compact_history"][0]["judge_score"] == pytest.approx(
        result.iterations[0].judge_score
    )
    assert summary["compact_history"][0]["best_so_far"] is True


def test_compact_history_does_not_mark_best_when_judge_scores_are_non_finite(
    tmp_path: Path,
) -> None:
    scenario_path = _write_initial_yaml(tmp_path)

    def record(iteration: int, judge_score: float) -> IterationRecord:
        iter_dir = tmp_path / f"iter_{iteration}"
        iter_dir.mkdir()
        return IterationRecord(
            iteration=iteration,
            iteration_dir=iter_dir,
            scenario_yaml_path=scenario_path,
            tune_output_dir=iter_dir,
            best_params={"restitution": 0.7},
            best_score=0.0,
            n_trials=1,
            judge_decision="continue",
            judge_score=judge_score,
            judge_reasoning="degraded",
            judge_llm_unavailable=True,
            refine_llm_unavailable=True,
            refine_reasoning="",
            metric_name="settle_distance",
            metric_value=0.0,
        )

    rows = _compact_refine_history(
        [record(1, float("nan")), record(2, float("inf"))],
        history_window=2,
    )

    assert [row["iteration"] for row in rows] == [1, 2]
    assert [row["best_so_far"] for row in rows] == [False, False]


def test_compact_history_treats_zero_judge_score_as_finite(
    tmp_path: Path,
) -> None:
    scenario_path = _write_initial_yaml(tmp_path)

    def record(iteration: int, judge_score: float) -> IterationRecord:
        iter_dir = tmp_path / f"iter_{iteration}"
        iter_dir.mkdir()
        return IterationRecord(
            iteration=iteration,
            iteration_dir=iter_dir,
            scenario_yaml_path=scenario_path,
            tune_output_dir=iter_dir,
            best_params={"restitution": 0.7},
            best_score=0.0,
            n_trials=1,
            judge_decision="continue",
            judge_score=judge_score,
            judge_reasoning="zero score is still finite",
            judge_llm_unavailable=False,
            refine_llm_unavailable=False,
            refine_reasoning="",
            metric_name="settle_distance",
            metric_value=0.0,
        )

    rows = _compact_refine_history(
        [record(1, -0.01), record(2, 0.0)],
        history_window=2,
    )

    assert [row["iteration"] for row in rows] == [1, 2]
    assert [row["best_so_far"] for row in rows] == [False, True]


def test_max_iterations_terminates_loop(tmp_path: Path) -> None:
    """Every iteration returns continue → loop runs exactly max_iterations
    times and stops with reason=max_iterations. Refine without a chat
    model degrades, but the loop still completes."""
    initial = _write_initial_yaml(tmp_path)

    def always_continue(params: TuneInput) -> TuneOutput:
        # BOTH params out of bounds → param_plausibility=0 → combined=0.4 → continue.
        return _make_tune_output(
            Path(params.output_dir),
            history=[_trial(0, 0.5, settle_distance=0.5)],
            best_score=0.5,
            best_params={"restitution": 10.0, "mass_scale": 100.0},
        )

    fake = _FakeRunTune([always_continue, always_continue])
    task = IterativePhysicsRefinementTask(
        user_prompt="make it bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=2,
        score_threshold=0.7,
        chat_model=None,  # refine degrades silently
        run_tune_callable=fake,
    )
    result = task.run({})
    assert len(result.iterations) == 2
    assert result.termination_reason == "max_iterations"
    # Both iter dirs persisted.
    assert (tmp_path / "out" / "iter_1" / "scenario.yaml").exists()
    assert (tmp_path / "out" / "iter_2" / "scenario.yaml").exists()
    assert (tmp_path / "out" / "iter_1" / "judge_result.json").exists()
    # refine_summary at the loop-level
    summary = json.loads(
        (tmp_path / "out" / "refine_summary.json").read_text(encoding="utf-8")
    )
    assert summary["termination_reason"] == "max_iterations"
    assert len(summary["iterations"]) == 2


def test_metric_value_extracted_from_history(tmp_path: Path) -> None:
    """When the metric is ``max_bounce_height``, the orchestrator surfaces
    the raw bounce height (from backend_metrics) on the iteration record."""
    initial = tmp_path / "scenario.yaml"
    initial.write_text(
        yaml.safe_dump(_scenario_yaml_dict(metric="max_bounce_height")),
        encoding="utf-8",
    )

    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[
                    _trial(
                        0,
                        -1.4,  # negated bounce height = score
                        max_bounce_height=1.4,
                        settle_distance=0.05,
                    )
                ],
                best_score=-1.4,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="bouncy",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.0,  # always approve
        chat_model=None,
        run_tune_callable=fake,
    )
    result = task.run({})
    assert len(result.iterations) == 1
    rec = result.iterations[0]
    assert rec.metric_name == "max_bounce_height"
    assert rec.metric_value == pytest.approx(1.4, abs=1e-6)


def test_listener_context_keys(tmp_path: Path) -> None:
    """Loop must update the shared context dict with the same keys
    material's IterativeRefinementTask emits."""
    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    ctx: dict[str, Any] = {}
    task.run(ctx)
    # Keys mirror material's loop contract.
    assert "judge_score" in ctx
    assert "judge_reasoning" in ctx
    assert "iteration_count" in ctx
    assert "continue_iteration" in ctx
    # Approved iteration → no continue.
    assert ctx["continue_iteration"] is False


def test_error_summary_is_strict_json(tmp_path: Path) -> None:
    """refine_summary.json must parse with strict JSON (no Infinity / NaN
    barewords) even when the error path stores best_score=inf."""
    initial = _write_initial_yaml(tmp_path)

    sentinel = "physics-refine-api-key-sentinel"

    def raise_tune(_params: TuneInput) -> TuneOutput:
        raise RuntimeError(f"provider rejected api_key={sentinel}")

    fake = _FakeRunTune([raise_tune])
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=2,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    from world_understanding.agentic.events import CollectingEventListener

    listener = CollectingEventListener()
    result = task.run({"event_listener": listener})
    summary_path = tmp_path / "out" / "refine_summary.json"
    assert summary_path.exists()
    raw = summary_path.read_text(encoding="utf-8")
    # Strict-JSON parsers reject Infinity / -Infinity / NaN tokens.
    parsed = json.loads(raw)
    assert parsed["termination_reason"] == "error"
    assert parsed["iterations"][0]["best_score"] is None
    assert parsed["iterations"][0]["error"] == "Tune execution failed before judge."
    assert result.iterations[0].error == "Tune execution failed before judge."
    published = raw + repr(listener.logs) + repr(listener.events)
    assert sentinel not in published
    # The bareword "Infinity" must not appear anywhere in the file.
    assert "Infinity" not in raw
    assert "NaN" not in raw


def test_tune_failure_via_success_false_terminates_with_error(tmp_path: Path) -> None:
    """When the inner tune returns ``TuneOutput(success=False)`` without
    raising (every backend trial failed, or runtime cancelled), the loop
    must terminate with reason=error/cancelled rather than running the
    judge over a junk history and possibly approving."""
    initial = _write_initial_yaml(tmp_path)

    failed_output = TuneOutput(
        success=False,
        output_dir=tmp_path / "out" / "iter_1",
        best_params={},
        best_score=float("nan"),
        n_trials=0,
        optimizer_used="random",
        engine_used="fake",
        history=[],
        artifacts={},
        cancelled=False,
        needs_refinement=False,
        error="all 30 trials failed",
    )
    fake = _FakeRunTune([failed_output])
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=3,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    result = task.run({})
    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].judge_decision == "skipped"
    assert result.iterations[0].error == "all 30 trials failed"
    # final_dir not promoted on error
    assert result.final_dir is None


def test_tune_cancellation_terminates_with_cancelled(tmp_path: Path) -> None:
    """``TuneOutput(success=False, cancelled=True)`` → termination=cancelled."""
    initial = _write_initial_yaml(tmp_path)

    cancelled_output = TuneOutput(
        success=False,
        output_dir=tmp_path / "out" / "iter_1",
        best_params={},
        best_score=float("nan"),
        n_trials=2,
        optimizer_used="random",
        engine_used="fake",
        history=[_trial(0, 0.05), _trial(1, 0.04)],
        artifacts={},
        cancelled=True,
        needs_refinement=False,
        error=None,
    )
    fake = _FakeRunTune([cancelled_output])
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=3,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    result = task.run({})
    assert result.termination_reason == "cancelled"
    assert result.iterations[0].cancelled is True
    assert result.final_dir is None


def test_per_iteration_seed_is_offset_to_avoid_artifact_collisions(
    tmp_path: Path,
) -> None:
    """Each iteration's tune call must run with a seed offset so the
    drop_settle backend's ``.tune_scenes/trial_seed_<seed>/`` directories
    are disjoint across iterations. Otherwise iter_2's tune would
    overwrite iter_1's per-trial recording.usd / trajectory.jsonl
    files that iter_1's history.jsonl still references."""
    initial = _write_initial_yaml(tmp_path)

    seen_seeds: list[int] = []

    def recording_tune(params: TuneInput) -> TuneOutput:
        seen_seeds.append(int(params.seed))
        return _make_tune_output(
            Path(params.output_dir),
            history=[_trial(0, 0.5, settle_distance=0.5)],
            best_score=0.5,
            best_params={"restitution": 10.0, "mass_scale": 100.0},
        )

    fake = _FakeRunTune([recording_tune, recording_tune])
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=2,
        max_trials=4,
        seed=42,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    task.run({})
    assert len(seen_seeds) == 2
    # Iter 1 keeps the base seed; iter 2 is offset by max_trials.
    assert seen_seeds[0] == 42
    assert seen_seeds[1] == 42 + 4


def test_max_iterations_clears_continue_iteration_flag(tmp_path: Path) -> None:
    """When the iteration cap stops the loop while the judge said
    "continue", ctx["continue_iteration"] must report False so callers
    pinning on it don't think there's still a next iteration coming."""
    initial = _write_initial_yaml(tmp_path)

    def always_continue(params: TuneInput) -> TuneOutput:
        return _make_tune_output(
            Path(params.output_dir),
            history=[_trial(0, 0.5, settle_distance=0.5)],
            best_score=0.5,
            best_params={"restitution": 10.0, "mass_scale": 100.0},
        )

    fake = _FakeRunTune([always_continue, always_continue])
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=2,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    ctx: dict[str, Any] = {}
    result = task.run(ctx)
    assert result.termination_reason == "max_iterations"
    # Judge said continue on iter 2, but iteration cap is final → False.
    assert ctx["continue_iteration"] is False
    assert ctx["iteration_count"] == 2


def test_visual_judge_fail_closed_when_generated_render_missing(
    tmp_path: Path,
) -> None:
    """Reference media means a missing generated render is not silently
    accepted as programmatic-only approval."""
    initial = _write_initial_yaml(tmp_path)
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")

    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="match this reference motion",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=object(),
        reference_images=[reference],
        run_tune_callable=fake,
        render_winning_trial=False,
    )

    result = task.run({})
    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error


def test_visual_judge_uses_rendered_best_trial_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When reference media exists, rendered best-trial frames are fed into
    the VLM judge and recorded on judge_result.json."""

    class StubVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_with_image_caption_pairs(self, **kwargs):
            self.calls.append(kwargs)
            return (
                '{"score": 1.0, "decision": "approve", '
                '"reasoning": "reference and output align"}'
            )

    initial = _write_initial_yaml(tmp_path)
    scenario_payload = _scenario_yaml_dict()
    scenario_payload["target"]["duration_s"] = 3.0
    initial.write_text(yaml.safe_dump(scenario_payload), encoding="utf-8")
    Image = pytest.importorskip("PIL.Image")
    reference = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    recording = tmp_path / "recording.usda"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    frame = tmp_path / "frame_0001__t250.png"
    Image.new("RGB", (8, 8), "blue").save(frame)

    render_kwargs: dict[str, Any] = {}

    def fake_render(*_args: Any, **kwargs: Any) -> list[Path]:
        render_kwargs.update(kwargs)
        return [frame]

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_time_sampled_usd",
        fake_render,
    )

    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[
                    _trial(0, 0.0, settle_distance=0.0, recording_usda=str(recording))
                ],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    vlm = StubVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="match this reference motion",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=vlm,
        judge_max_tokens=888,
        judge_temperature=0.4,
        reference_images=[reference],
        run_tune_callable=fake,
        render_winning_trial=False,
    )

    result = task.run({})

    assert result.termination_reason == "approved"
    assert result.final_recording_usd is not None
    assert result.final_recording_usd == tmp_path / "out" / "final" / "recording.usd"
    assert result.final_recording_usd.is_file()
    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["extra"]["judge_modality"] == "vlm"
    assert payload["extra"]["visual_evidence_enabled"] is True
    assert payload["extra"]["reference_image_count"] == 1
    assert payload["extra"]["generated_image_count"] == 1
    evidence = payload["extra"]["visual_evidence"]
    assert evidence["comparison_image"] == str(
        tmp_path / "out" / "iter_1" / "comparison.png"
    )
    assert Path(evidence["comparison_image"]).exists()
    assert evidence["generated_images"][0]["caption"] == (
        "Generated Physics Output - Frame 1 (t=0.250s):"
    )
    assert vlm.calls[0]["max_tokens"] == 888
    assert vlm.calls[0]["temperature"] == 0.4
    assert vlm.calls[0]["image_caption_pairs"][1] == (
        "Generated Physics Output - Frame 1 (t=0.250s):",
        frame,
    )
    assert render_kwargs["max_duration_seconds"] == 3.0
    assert render_kwargs["make_mp4"] is False


def test_no_visual_evidence_keeps_render_but_judge_gets_no_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return (
                '{"score": 1.0, "decision": "approve", '
                '"reasoning": "text evidence is sufficient"}'
            )

    initial = _write_initial_yaml(tmp_path)
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"not inspected when visual evidence is disabled")
    recording = tmp_path / "recording.usda"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    frame = tmp_path / "frame_0001__t250.png"
    frame.write_bytes(b"rendered frame")
    render_calls: list[dict[str, Any]] = []

    def fake_render(*_args: Any, **kwargs: Any) -> list[Path]:
        render_calls.append(kwargs)
        return [frame]

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_time_sampled_usd",
        fake_render,
    )
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[
                    _trial(0, 0.0, settle_distance=0.0, recording_usda=str(recording))
                ],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    vlm = StubVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="judge without media",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=vlm,
        reference_images=[reference],
        run_tune_callable=fake,
        render_winning_trial=True,
        visual_evidence_enabled=False,
    )

    result = task.run({})

    assert result.termination_reason == "approved"
    assert len(render_calls) == 1
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_caption_pairs"] == []
    payload = json.loads(
        (tmp_path / "out" / "iter_1" / "judge_result.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["visual_evidence_enabled"] is False
    assert payload["extra"]["reference_image_count"] == 0
    assert payload["extra"]["generated_image_count"] == 0
    assert payload["extra"]["visual_evidence"] is None


def test_generated_visual_judge_renders_when_artifact_render_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return (
                '{"score": 1.0, "decision": "approve", '
                '"reasoning": "generated motion looks realistic"}'
            )

    initial = _write_freeform_initial_yaml(tmp_path)
    recording = tmp_path / "recording.usda"
    recording.write_text("#usda 1.0\n", encoding="utf-8")
    frame = tmp_path / "frame_0001__t250.png"
    frame.write_bytes(b"rendered frame")
    render_calls: list[dict[str, Any]] = []

    def fake_render(*_args: Any, **kwargs: Any) -> list[Path]:
        render_calls.append(kwargs)
        return [frame]

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_time_sampled_usd",
        fake_render,
    )
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[
                    _trial(0, 0.0, settle_distance=0.0, recording_usda=str(recording))
                ],
                best_score=0.0,
                best_params={"restitution": 0.1, "mass_scale": 1.0},
            )
        ]
    )
    vlm = StubVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="judge generated realism",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=vlm,
        run_tune_callable=fake,
        render_winning_trial=False,
        visual_evidence_enabled=True,
    )

    result = task.run({})

    assert result.termination_reason == "approved"
    assert len(render_calls) == 1
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_caption_pairs"] == [
        ("Generated Physics Output - Frame 1 (t=0.250s):", frame)
    ]
    payload = json.loads(
        (tmp_path / "out" / "iter_1" / "judge_result.json").read_text(encoding="utf-8")
    )
    assert payload["extra"]["reference_image_count"] == 0
    assert payload["extra"]["generated_image_count"] == 1
    evidence = payload["extra"]["visual_evidence"]
    assert evidence["reference_images"] == []
    assert evidence["generated_images"][0]["path"] == str(frame)


def test_generated_visual_judge_preserves_render_error_without_reference(
    tmp_path: Path,
) -> None:
    class StubVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return (
                '{"score": 1.0, "decision": "approve", '
                '"reasoning": "should not judge text-only"}'
            )

    initial = _write_freeform_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.1, "mass_scale": 1.0},
            )
        ]
    )
    vlm = StubVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="judge generated realism",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=vlm,
        run_tune_callable=fake,
        render_winning_trial=False,
        visual_evidence_enabled=True,
    )

    result = task.run({})

    assert result.termination_reason == "error"
    assert len(vlm.calls) == 0
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error
    payload = json.loads(
        (tmp_path / "out" / "iter_1" / "judge_result.json").read_text(encoding="utf-8")
    )
    assert payload["llm_unavailable"] is True
    evidence = payload["extra"]["visual_evidence"]
    assert evidence["reference_images"] == []
    assert evidence["generated_images"] == []
    assert evidence["generated_error"] is not None


def test_text_only_judge_still_runs_when_optional_render_missing(
    tmp_path: Path,
) -> None:
    class StubVLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            self.calls.append(kwargs)
            return (
                '{"score": 1.0, "decision": "approve", '
                '"reasoning": "text judge passed"}'
            )

    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    vlm = StubVLM()
    task = IterativePhysicsRefinementTask(
        user_prompt="judge this text-only result",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        vlm_model=vlm,
        run_tune_callable=fake,
    )

    result = task.run({})

    assert result.termination_reason == "approved"
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_caption_pairs"] == []
    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is False
    assert payload["extra"]["visual_evidence_enabled"] is True
    assert payload["extra"]["visual_evidence"] is None


def test_text_only_refine_fail_closed_when_judge_vlm_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_default_vlm() -> object:
        raise RuntimeError("missing key")

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.resolve_default_judge_vlm",
        fail_default_vlm,
    )
    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="judge this text-only result",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )

    result = task.run({})

    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error
    summary_path = tmp_path / "out" / "refine_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["termination_reason"] == "error"
    assert summary["iterations"][0]["error"] == result.iterations[0].error

    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is True
    assert payload["extra"]["visual_evidence"] is None


def test_judge_llm_timeout_synthesises_unavailable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung judge VLM call must time out via the wall-clock guard
    rather than blocking ``physics-agent refine`` indefinitely. The
    timeout produces an llm_unavailable JudgeResult so the loop can
    continue or terminate cleanly."""
    import time as _time

    initial = _write_initial_yaml(tmp_path)

    def hang_judge(*_args: Any, **_kwargs: Any) -> Any:
        _time.sleep(2.0)  # > timeout below
        raise AssertionError("judge should have been timed out")

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.run_tune_judge",
        hang_judge,
    )

    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
        llm_timeout_seconds=0.3,
    )
    result = task.run({})

    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error

    # The timeout produced an llm_unavailable JudgeResult before the loop
    # failed closed with an audit record rather than approving from
    # programmatic-only scores.
    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    assert judge_result_path.exists()
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is True
    assert "timeout" in payload["llm_critique"].lower()


def test_reference_evidence_prep_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time as _time

    def hang_prepare_reference_media(*_args: Any, **_kwargs: Any) -> Any:
        _time.sleep(2.0)
        raise AssertionError("reference evidence prep should have timed out")

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.prepare_reference_media",
        hang_prepare_reference_media,
    )

    def fail_if_rendered(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("reference prep timeout should not trigger rendering")

    monkeypatch.setattr(
        IterativePhysicsRefinementTask,
        "_render_best_trial_into_iter_dir",
        fail_if_rendered,
    )

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")
    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="match the reference",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        vlm_model=_DefaultJudgeVLM(),
        reference_images=[reference],
        run_tune_callable=fake,
        render_winning_trial=False,
        llm_timeout_seconds=0.2,
    )

    start = _time.monotonic()
    result = task.run({})

    assert _time.monotonic() - start < 1.5
    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error

    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is True
    evidence = payload["extra"]["visual_evidence"]
    assert evidence["reference_error"] == "VisualEvidencePreparationTimeout"


def test_winning_trial_render_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time as _time

    def hang_render(*_args: Any, **_kwargs: Any) -> Any:
        _time.sleep(2.0)
        raise AssertionError("winning-trial render should have timed out")

    monkeypatch.setattr(
        IterativePhysicsRefinementTask,
        "_render_best_trial_into_iter_dir",
        hang_render,
    )

    reference = tmp_path / "reference.png"
    reference.write_bytes(b"fake image bytes")
    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="match the reference",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        vlm_model=_DefaultJudgeVLM(),
        reference_images=[reference],
        run_tune_callable=fake,
        render_winning_trial=False,
        llm_timeout_seconds=0.2,
    )

    start = _time.monotonic()
    result = task.run({})

    assert _time.monotonic() - start < 1.5
    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error

    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is True
    evidence = payload["extra"]["visual_evidence"]
    assert evidence["generated_error"] == "VisualEvidenceRenderTimeout"


def test_default_judge_vlm_setup_timeout_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time as _time

    def hang_default_vlm() -> object:
        _time.sleep(2.0)
        return _DefaultJudgeVLM()

    monkeypatch.setattr(
        "physics_agent.tasks.iterative_physics_refinement.resolve_default_judge_vlm",
        hang_default_vlm,
    )
    initial = _write_initial_yaml(tmp_path)
    fake = _FakeRunTune(
        [
            _make_tune_output(
                tmp_path / "out" / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="judge this result",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=tmp_path / "out",
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
        llm_timeout_seconds=0.2,
    )

    start = _time.monotonic()
    result = task.run({})

    assert _time.monotonic() - start < 1.5
    assert result.termination_reason == "error"
    assert len(result.iterations) == 1
    assert result.iterations[0].error is not None
    assert "Judge VLM unavailable" in result.iterations[0].error

    judge_result_path = tmp_path / "out" / "iter_1" / "judge_result.json"
    payload = json.loads(judge_result_path.read_text(encoding="utf-8"))
    assert payload["llm_unavailable"] is True
    assert "VLM unavailable" in payload["llm_critique"]


def test_rerun_preserves_user_iter_named_dirs(tmp_path: Path) -> None:
    """The iter_*/ wipe must match iter_<digit> exactly, so user-authored
    folders like iter_notes/ or iter_backup/ are NOT deleted on rerun."""
    initial = _write_initial_yaml(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # User-authored siblings that LOOK iter_-ish but aren't generated.
    for name in ("iter_notes", "iter_backup", "iteration_old", "iter"):
        d = out_dir / name
        d.mkdir()
        (d / "user_data.txt").write_text("important", encoding="utf-8")

    # And a stale generated iter_2 from a previous run.
    (out_dir / "iter_2").mkdir()
    (out_dir / "iter_2" / "stale.json").write_text("{}", encoding="utf-8")

    fake = _FakeRunTune(
        [
            _make_tune_output(
                out_dir / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=out_dir,
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    task.run({})

    # Generated iter_2 wiped.
    assert not (out_dir / "iter_2").exists()
    # User-authored siblings preserved.
    for name in ("iter_notes", "iter_backup", "iteration_old", "iter"):
        assert (out_dir / name).exists(), f"{name} should not have been wiped"
        assert (out_dir / name / "user_data.txt").read_text(
            encoding="utf-8"
        ) == "important"


def test_rerun_into_same_output_dir_clears_stale_iter_dirs(tmp_path: Path) -> None:
    """Running into a populated output_dir wipes prior iter_N/ + final/.

    Previously: a 5-iter run followed by a 2-iter run would leave iter_3
    through iter_5 and the old final/ visible — making the on-disk state
    a confusing union of two runs.
    """
    initial = _write_initial_yaml(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Pretend a previous 5-iter run left these behind.
    for n in range(1, 6):
        d = out_dir / f"iter_{n}"
        d.mkdir()
        (d / "stale.json").write_text("{}", encoding="utf-8")
    (out_dir / "final").mkdir()
    (out_dir / "final" / "stale.json").write_text("{}", encoding="utf-8")
    # An unrelated file at the top level must NOT be touched.
    (out_dir / "user_note.txt").write_text("keep me", encoding="utf-8")

    fake = _FakeRunTune(
        [
            _make_tune_output(
                out_dir / "iter_1",
                history=[_trial(0, 0.0, settle_distance=0.0)],
                best_score=0.0,
                best_params={"restitution": 0.9, "mass_scale": 1.0},
            )
        ]
    )
    task = IterativePhysicsRefinementTask(
        user_prompt="goal",
        initial_scenario=initial,
        physics_usd=tmp_path / "fake.usda",
        output_dir=out_dir,
        max_iterations=1,
        score_threshold=0.7,
        chat_model=None,
        run_tune_callable=fake,
    )
    task.run({})

    # iter_3..iter_5 wiped (only iter_1 was produced this run)
    assert not (out_dir / "iter_3").exists()
    assert not (out_dir / "iter_4").exists()
    assert not (out_dir / "iter_5").exists()
    # iter_1 contains THIS run's artifacts (no stale.json)
    assert (out_dir / "iter_1" / "scenario.yaml").exists()
    assert not (out_dir / "iter_1" / "stale.json").exists()
    # final/ rebuilt without stale content
    assert (out_dir / "final" / "scenario.yaml").exists()
    assert not (out_dir / "final" / "stale.json").exists()
    # User-authored top-level files untouched
    assert (out_dir / "user_note.txt").read_text(encoding="utf-8") == "keep me"
