# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``physics_agent.tasks.scenario_refine.run_scenario_refine``.

Covers degraded paths (no chat_model, LLM error, parse failure,
invalid refined scenario) and the happy path with a stubbed chat
model. Goal-neutral assertions only — we do not bake "bouncy" or any
specific adjective into the test set.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from physics_agent.tasks.judge_tune import JudgeResult
from physics_agent.tasks.scenario_refine import (
    RefineResult,
    _scenario_to_dict,
    _summarise_history,
    run_scenario_refine,
)
from physics_agent.tuning.types import Scenario, TunableParam


def _scenario(metric: str = "settle_distance") -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        metric=metric,
    )


def _scenario_with_extra(extra: dict[str, Any]) -> Scenario:
    return Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        metric="settle_distance",
        extra=extra,
    )


def _judge(decision: str = "continue", score: float = 0.45) -> JudgeResult:
    return JudgeResult(
        decision=decision,  # type: ignore[arg-type]
        score=score,
        programmatic_score=score,
        llm_score=score,
        reasoning=f"{decision} (combined={score:.2f})",
        iterations=1,
        llm_unavailable=False,
        programmatic_critique="(programmatic critique placeholder)",
        llm_critique="(llm critique placeholder)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubChat:
    """Minimal stub chat model — patched generate_chat_response reads model_name."""

    model_name = "stub-test-model"


def _patch_chat(
    monkeypatch: pytest.MonkeyPatch,
    fake_response: dict[str, Any] | str,
) -> None:
    """Patch ``generate_chat_response`` so the LLM call returns a canned reply.

    Pass a dict to mimic ``{"response": <text>}`` shape, or pass ``"raise"``
    to make the call raise, or pass ``"error"`` to make it return
    ``{"error": "..."}``.
    """

    def fake(
        chat_model: Any,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if fake_response == "raise":
            raise RuntimeError("simulated provider failure")
        if fake_response == "error":
            return {"error": "simulated provider error"}
        if isinstance(fake_response, dict):
            return fake_response
        return {"response": fake_response}

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake,
    )


# ---------------------------------------------------------------------------
# Helper-function unit checks
# ---------------------------------------------------------------------------


def test_scenario_to_dict_round_trips_through_yaml() -> None:
    sc = _scenario()
    d = _scenario_to_dict(sc)
    yml = yaml.safe_dump(d, sort_keys=False)
    parsed = yaml.safe_load(yml)
    assert parsed["name"] == "drop_settle"
    assert parsed["metric"] == "settle_distance"
    assert {p["name"] for p in parsed["parameters"]} == {"restitution", "mass_scale"}


def test_summarise_history_picks_best_scores_and_trims() -> None:
    trials = [
        {"trial_index": i, "score": float(i), "params": {}, "failed": False}
        for i in range(20)
    ]
    top = _summarise_history(trials)
    assert len(top) == 5
    # Lowest scores first.
    assert [t["trial_index"] for t in top] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Degraded paths (chat_model None, error, malformed response)
# ---------------------------------------------------------------------------


def test_no_chat_model_returns_input_scenario() -> None:
    sc = _scenario()
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="make it bouncy",
        chat_model=None,
        iteration=1,
    )
    assert isinstance(result, RefineResult)
    assert result.llm_unavailable is True
    assert result.scenario.name == "drop_settle"
    assert result.scenario.metric == "settle_distance"
    # Dump round-trips:
    parsed_back = yaml.safe_load(result.refined_yaml)
    assert parsed_back["name"] == "drop_settle"


def test_llm_invoke_raises_returns_input_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    _patch_chat(monkeypatch, "raise")
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="anything",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True
    assert result.scenario is sc


def test_llm_returns_error_dict_returns_input_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    _patch_chat(monkeypatch, "error")
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="anything",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True


def test_llm_returns_garbage_string_returns_input_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    _patch_chat(monkeypatch, "Sure thing — here is some prose with no JSON")
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="anything",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True


def test_llm_omits_scenario_key_returns_input_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    _patch_chat(monkeypatch, json.dumps({"reasoning": "no scenario block"}))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="anything",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True


def test_llm_emits_invalid_scenario_returns_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM emits a scenario dict with an unsupported parameter name."""
    sc = _scenario()
    bad = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.5},
            "parameters": [{"name": "not_a_real_param", "min": 0.0, "max": 1.0}],
        },
        "reasoning": "broken parameter name",
    }
    _patch_chat(monkeypatch, json.dumps(bad))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="anything",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True
    # Echoes input.
    assert result.scenario is sc


def test_backend_allowlist_scopes_refine_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    captured: dict[str, str | None] = {}

    def fake(
        chat_model: Any,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return {
            "response": json.dumps(
                {"scenario": _scenario_to_dict(sc), "reasoning": "unchanged"}
            )
        }

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake,
    )

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="make it bouncy",
        chat_model=_StubChat(),
        backend_name="ovphysx",
        supported_param_keys=(
            "mass_scale",
            "static_friction",
            "dynamic_friction",
            "restitution",
        ),
    )

    assert result.llm_unavailable is False
    assert "Active backend: ovphysx" in str(captured["system_prompt"])
    assert "contact_ke" not in str(captured["system_prompt"])
    assert "allowed_tunable_parameters" in str(captured["prompt"])


def test_prior_refine_history_none_omits_legacy_prompt_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    captured_prompts: list[str] = []

    def fake(
        chat_model: Any,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        captured_prompts.append(prompt)
        return {
            "response": json.dumps(
                {"scenario": _scenario_to_dict(sc), "reasoning": "unchanged"}
            )
        }

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake,
    )

    legacy_result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="make it bouncy",
        chat_model=_StubChat(),
    )
    legacy_payload = json.loads(captured_prompts[-1][captured_prompts[-1].index("{") :])
    assert legacy_result.notes["prior_refine_history_size"] == 0
    assert "prior_refine_history" not in legacy_payload

    explicit_result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="make it bouncy",
        chat_model=_StubChat(),
        prior_refine_history=[],
    )
    explicit_payload = json.loads(
        captured_prompts[-1][captured_prompts[-1].index("{") :]
    )
    assert explicit_result.notes["prior_refine_history_size"] == 0
    assert explicit_payload["prior_refine_history"] == []


def test_backend_allowlist_rejects_refined_disallowed_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    bad = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.5},
            "parameters": [{"name": "contact_ke", "min": 100.0, "max": 100000.0}],
        },
        "reasoning": "tried a Newton-only contact knob",
    }
    _patch_chat(monkeypatch, json.dumps(bad))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="make it bouncy",
        chat_model=_StubChat(),
        backend_name="ovphysx",
        supported_param_keys=(
            "mass_scale",
            "static_friction",
            "dynamic_friction",
            "restitution",
        ),
    )

    assert result.llm_unavailable is True
    assert result.scenario is sc
    assert "backend-disallowed" in result.reasoning


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_llm_widens_bounds_and_swaps_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = _scenario(metric="settle_distance")
    refined = {
        "scenario": {
            "name": "drop_settle",
            "metric": "max_bounce_height",
            "target": {"drop_height_m": 0.6, "duration_s": 2.5, "gravity": -9.81},
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "biased restitution upward and lowered mass for higher rebound",
    }
    _patch_chat(monkeypatch, json.dumps(refined))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",  # generic; template is goal-neutral
        chat_model=_StubChat(),
        iteration=2,
    )
    assert result.llm_unavailable is False
    assert result.scenario.metric == "max_bounce_height"
    rest_param = next(p for p in result.scenario.params if p.name == "restitution")
    assert rest_param.min_value == pytest.approx(0.7)
    assert rest_param.max_value == pytest.approx(0.99)
    # Reasoning carried through (truncated).
    assert "rebound" in result.reasoning
    parsed_back = yaml.safe_load(result.refined_yaml)
    assert parsed_back["metric"] == "max_bounce_height"


def test_drop_settle_metric_preserved_when_llm_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM returns a refined scenario with no ``metric`` field. Without
    the preservation guard, ``parse_scenario`` would default the missing
    metric back to ``settle_distance``, silently switching a run that
    was optimising ``max_bounce_height`` away from the user's intent.
    The fix preserves the current metric when the LLM omits one."""
    sc = _scenario(metric="max_bounce_height")
    refined_no_metric = {
        "scenario": {
            "name": "drop_settle",
            # NO metric field here — LLM only changed bounds/target
            "target": {"drop_height_m": 0.6, "duration_s": 2.5, "gravity": -9.81},
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "tightened bounds, kept the metric",
    }
    _patch_chat(monkeypatch, json.dumps(refined_no_metric))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is False
    # Metric preserved — would have defaulted to settle_distance otherwise.
    assert result.scenario.metric == "max_bounce_height"


def test_top_level_judge_extra_preserved_when_llm_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario_with_extra({"judge": {"max_tokens": 1234, "temperature": 0.25}})
    refined_no_judge = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.6, "duration_s": 2.5, "gravity": -9.81},
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "tightened bounds",
    }
    _patch_chat(monkeypatch, json.dumps(refined_no_judge))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.llm_unavailable is False
    assert result.scenario.extra["judge"] == {
        "max_tokens": 1234,
        "temperature": 0.25,
    }
    parsed_back = yaml.safe_load(result.refined_yaml)
    assert parsed_back["judge"] == {"max_tokens": 1234, "temperature": 0.25}


def test_top_level_judge_extra_can_be_overridden_by_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario_with_extra({"judge": {"max_tokens": 1234}})
    refined_with_judge = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.6, "duration_s": 2.5, "gravity": -9.81},
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
            "judge": {"max_tokens": 4321},
        },
        "reasoning": "tightened bounds and judge budget",
    }
    _patch_chat(monkeypatch, json.dumps(refined_with_judge))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.llm_unavailable is False
    assert result.scenario.extra["judge"] == {"max_tokens": 4321}


def test_target_keys_omitted_by_llm_are_carried_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the refine LLM authors a refined ``target`` dict that omits
    a key the current scenario sets (e.g. ``camera_ground_bias_fraction``,
    ``sample_fps``, ``record_video``), the preservation helper carries it
    forward so the loop doesn't silently regress to that key's default
    behavior between iterations. This guards against the LLM never being
    told a key existed (system-prompt enumeration drift)."""
    sc = Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={
            "drop_height_m": 0.5,
            "duration_s": 2.0,
            "gravity": -9.81,
            "camera_ground_bias_fraction": 0.75,
            "sample_fps": 30,
            "record_video": "always",
        },
        metric="settle_distance",
    )
    refined_minimal = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {
                # The LLM only re-emits the keys it wants to change.
                "drop_height_m": 1.0,
                "duration_s": 3.0,
                "gravity": -9.81,
            },
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "tightened bounds; bumped drop height",
    }
    _patch_chat(monkeypatch, json.dumps(refined_minimal))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.llm_unavailable is False
    # Explicitly-refined keys win:
    assert result.scenario.target["drop_height_m"] == 1.0
    assert result.scenario.target["duration_s"] == 3.0
    # Omitted keys are carried forward verbatim:
    assert result.scenario.target["camera_ground_bias_fraction"] == 0.75
    assert result.scenario.target["sample_fps"] == 30
    assert result.scenario.target["record_video"] == "always"
    # The serialized YAML reflects the merged target.
    parsed_back = yaml.safe_load(result.refined_yaml)
    assert parsed_back["target"]["camera_ground_bias_fraction"] == 0.75


@pytest.mark.parametrize(
    "missing_kind", ["omit_entirely", "explicit_null", "empty_dict"]
)
def test_target_keys_preserved_when_refined_target_is_empty_or_missing(
    monkeypatch: pytest.MonkeyPatch, missing_kind: str
) -> None:
    """The LLM may emit a refined scenario that omits ``target``,
    sets it to ``null``, or returns an empty ``{}`` — all three are
    common refusal-style outputs. ``parse_scenario`` coerces all three
    to ``{}``, so without the preservation helper every current target
    key would silently disappear (the very regression the helper
    exists to prevent). Pin all three shapes."""
    sc = Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={
            "drop_height_m": 0.5,
            "duration_s": 2.0,
            "gravity": -9.81,
            "camera_ground_bias_fraction": 0.75,
        },
        metric="settle_distance",
    )
    refined_scenario: dict[str, Any] = {
        "name": "drop_settle",
        "metric": "settle_distance",
        "parameters": [
            {"name": "restitution", "min": 0.7, "max": 0.99},
            {"name": "mass_scale", "min": 0.5, "max": 1.0},
        ],
    }
    if missing_kind == "explicit_null":
        refined_scenario["target"] = None
    elif missing_kind == "empty_dict":
        refined_scenario["target"] = {}
    # omit_entirely: leave the key out

    _patch_chat(
        monkeypatch,
        json.dumps({"scenario": refined_scenario, "reasoning": "narrowed bounds"}),
    )

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.llm_unavailable is False
    # All current target keys survive regardless of which missing shape
    # the LLM emitted.
    for key, expected in sc.target.items():
        assert result.scenario.target[key] == expected, (
            f"key {key!r} dropped when LLM target was {missing_kind!r}"
        )


@pytest.mark.parametrize("wrong_shape", ["string", "list", "scalar_int", "scalar_bool"])
def test_target_wrong_shape_defers_to_parse_scenario(
    monkeypatch: pytest.MonkeyPatch, wrong_shape: str
) -> None:
    """When the LLM emits ``target`` with a non-dict, non-None shape
    (e.g. a string, list, or scalar), the preservation helper must
    NOT mutate it into a dict on the fly — instead it defers to
    ``parse_scenario``, which rejects with a clean error. The whole
    refine result then falls back to the input scenario via the
    ``llm_unavailable=True`` degraded path. This pins that "wrong
    shape" is handled by validation, not silently fixed by the
    helper. Round-4 CX flagged this surface as untested."""
    sc = Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        metric="settle_distance",
    )
    bad_target: Any = {
        "string": "drop_height_m=0.5",
        "list": [{"drop_height_m": 0.5}],
        "scalar_int": 42,
        "scalar_bool": True,
    }[wrong_shape]
    refined = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": bad_target,
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "wrong-shape target",
    }
    _patch_chat(monkeypatch, json.dumps(refined))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    # parse_scenario rejected the wrong-shape target → loop falls back to
    # the input scenario, marked llm_unavailable. The original target
    # survives intact.
    assert result.llm_unavailable is True
    assert result.scenario.target == sc.target


def test_target_key_explicit_overrides_win_over_preservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit refined value for a previously-set target key REPLACES
    the current value — preservation only fires on omission, not on
    intentional change."""
    sc = Scenario(
        name="drop_settle",
        params=(
            TunableParam(name="restitution", min_value=0.4, max_value=0.8),
            TunableParam(name="mass_scale", min_value=0.7, max_value=1.3),
        ),
        target={
            "drop_height_m": 0.5,
            "duration_s": 2.0,
            "gravity": -9.81,
            "camera_ground_bias_fraction": 0.75,
        },
        metric="settle_distance",
    )
    refined_with_explicit_bias = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {
                "drop_height_m": 0.5,
                "duration_s": 2.0,
                "gravity": -9.81,
                # LLM picked a different bias on purpose.
                "camera_ground_bias_fraction": 0.5,
            },
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "lowered camera bias",
    }
    _patch_chat(monkeypatch, json.dumps(refined_with_explicit_bias))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.scenario.target["camera_ground_bias_fraction"] == 0.5


def test_non_judge_extra_is_not_reinjected_when_llm_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario_with_extra(
        {
            "judge": {"max_tokens": 1234},
            "transient_note": {"drop_after_refine": True},
        }
    )
    refined_no_extras = {
        "scenario": {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.6, "duration_s": 2.5, "gravity": -9.81},
            "parameters": [
                {"name": "restitution", "min": 0.7, "max": 0.99},
                {"name": "mass_scale", "min": 0.5, "max": 1.0},
            ],
        },
        "reasoning": "tightened bounds",
    }
    _patch_chat(monkeypatch, json.dumps(refined_no_extras))

    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="bouncy",
        chat_model=_StubChat(),
    )

    assert result.llm_unavailable is False
    assert result.scenario.extra["judge"] == {"max_tokens": 1234}
    assert "transient_note" not in result.scenario.extra


def test_freeform_metric_swap_is_snapped_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """freeform.evaluate ignores scenario.metric — letting the LLM swap
    it would author misleading artifacts. Snap any metric change back to
    the input value while still accepting all other refinements.

    Round 13 (CodeRabbit thread #2 follow-on): ``parse_scenario`` now
    rejects freeform metrics other than ``judge_score`` so the snap-back
    target is the canonical freeform metric. Constructing the input
    Scenario via ``parse_scenario`` (instead of direct ``Scenario(...)``)
    makes the test track the public contract rather than the dataclass
    constructor.
    """
    from physics_agent.tuning.scenario import parse_scenario

    sc = parse_scenario(
        {
            "name": "freeform",
            "metric": "judge_score",
            "target": {"duration_s": 2.0},
            "parameters": [
                {"name": "restitution", "min": 0.4, "max": 0.95},
            ],
        }
    )
    bad = {
        "scenario": {
            "name": "freeform",
            "metric": "max_bounce_height",  # would be misleading
            "target": {"duration_s": 2.0},
            "parameters": [{"name": "restitution", "min": 0.7, "max": 0.99}],
        },
        "reasoning": "tightened restitution upward",
    }
    _patch_chat(monkeypatch, json.dumps(bad))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="goal",
        chat_model=_StubChat(),
    )
    # Metric snapped back to the input value; other fields kept.
    assert result.llm_unavailable is False
    assert result.scenario.metric == "judge_score"
    rest_param = next(p for p in result.scenario.params if p.name == "restitution")
    assert rest_param.min_value == pytest.approx(0.7)


def test_unsupported_drop_settle_metric_falls_back_to_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM emits a metric not in the drop_settle registry — refine must
    fall back to the input scenario rather than returning a "successful"
    refinement that the next tune iteration would only reject.
    """
    sc = _scenario(metric="settle_distance")
    bad = {
        "scenario": {
            "name": "drop_settle",
            "metric": "max_velocity",  # not in _METRICS
            "target": {"drop_height_m": 0.5},
            "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}],
        },
        "reasoning": "tried to swap to a metric that doesn't exist",
    }
    _patch_chat(monkeypatch, json.dumps(bad))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="goal",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is True
    # Falls back to the input scenario, NOT the refined one.
    assert result.scenario is sc
    assert "max_velocity" in result.reasoning


def test_llm_attempt_to_change_scenario_name_is_snapped_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    bad = {
        "scenario": {
            "name": "freeform",  # would normally break runtime — must be reverted
            "metric": "settle_distance",
            "target": {"drop_height_m": 0.5},
            "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}],
        },
        "reasoning": "tried to change kind",
    }
    _patch_chat(monkeypatch, json.dumps(bad))
    result = run_scenario_refine(
        current_scenario=sc,
        judge_result=_judge(),
        user_goal_text="goal",
        chat_model=_StubChat(),
    )
    assert result.llm_unavailable is False
    assert result.scenario.name == "drop_settle"


def test_history_top_k_passed_to_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompts: list[str] = []

    def fake(
        chat_model: Any,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        captured_prompts.append(prompt)
        return {
            "response": json.dumps(
                {
                    "scenario": _scenario_to_dict(_scenario()),
                    "reasoning": "echo",
                }
            )
        }

    monkeypatch.setattr(
        "physics_agent.tasks.scenario_refine.generate_chat_response",
        fake,
    )

    history = [
        {"trial_index": i, "score": float(i), "params": {}, "failed": False}
        for i in range(10)
    ]
    run_scenario_refine(
        current_scenario=_scenario(),
        judge_result=_judge(),
        user_goal_text="goal",
        history_summary=history,
        chat_model=_StubChat(),
    )
    assert len(captured_prompts) == 1
    body = captured_prompts[0]
    # Top trials (lowest scores) appear; trial_index 9 should be omitted.
    assert '"trial_index": 0' in body
    assert '"trial_index": 4' in body
    assert '"trial_index": 9' not in body
