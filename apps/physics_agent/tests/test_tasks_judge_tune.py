# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the VLM-as-judge in ``physics_agent.tasks.judge_tune``.

Covers programmatic-only path (vlm_model=None), VLM JSON parsing variations,
threshold/decision logic, and iteration passing. (No cache — caching was
removed; rerun-determinism comes from the daemon's seed contract.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from physics_agent.api.defaults import DEFAULT_JUDGE_MAX_TOKENS
from physics_agent.tasks.judge_tune import (
    JudgeResult,
    run_tune_judge,
)
from physics_agent.tuning.types import Scenario, TrialRecord, TunableParam
from physics_agent.tuning.visual_evidence import JudgeVisualEvidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubChat:
    """Compatibility stub patched with the VLM method in older text tests."""

    model_name = "stub-test-model"


class StubVLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate_with_image_caption_pairs(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class RaisingVLM:
    def generate_with_image_caption_pairs(self, **_kwargs):
        raise TimeoutError("provider deadline")


def _scenario(
    *,
    params: tuple[TunableParam, ...] | None = None,
    name: str = "drop_settle",
    metric: str = "settle_distance",
    target: dict | None = None,
) -> Scenario:
    if params is None:
        params = (TunableParam(name="mass_scale", min_value=0.5, max_value=2.0),)
    return Scenario(
        name=name,
        params=params,
        target=target if target is not None else {"drop_height_m": 0.5},
        metric=metric,
    )


def _trial(
    trial_index: int,
    score: float,
    *,
    failed: bool = False,
    duration: float = 0.0,
    backend_metrics: dict | None = None,
    params: dict | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_index=trial_index,
        params=params if params is not None else {"mass_scale": 1.0},
        score=score,
        backend_metrics=backend_metrics if backend_metrics is not None else {},
        duration_seconds=duration,
        failed=failed,
    )


def _history(*scores_failed: tuple[float, bool]) -> list[TrialRecord]:
    """Build a history from ``(score, failed)`` tuples."""
    return [
        _trial(i, score, failed=failed)
        for i, (score, failed) in enumerate(scores_failed)
    ]


def _patch_llm(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    def wrapped(self, **kwargs):
        result = fake(
            self,
            kwargs["final_prompt"],
            system_prompt=kwargs.get("system_prompt"),
        )
        if isinstance(result, dict) and "response" in result:
            return result["response"]
        return result

    monkeypatch.setattr(
        StubChat,
        "generate_with_image_caption_pairs",
        wrapped,
        raising=False,
    )


# ---------------------------------------------------------------------------
# Programmatic-only path (vlm_model=None)
# ---------------------------------------------------------------------------


def test_all_in_bounds_no_failed_trials_finite_score_approves() -> None:
    sc = _scenario()
    history = _history((0.1, False), (0.05, False), (0.02, False))
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=None,
        score_threshold=0.7,
    )
    assert result.programmatic_score == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)
    assert result.decision == "approve"
    assert result.llm_unavailable is True
    assert result.llm_score == pytest.approx(result.programmatic_score)


def test_param_out_of_bounds_drops_param_plausibility() -> None:
    sc = _scenario()
    history = _history((0.1, False))
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 10.0},  # outside [0.5, 2.0]
        chat_model=None,
        score_threshold=0.7,
    )
    # plausibility=0, failed=1, finite=1 → 0*0.6 + 1*0.3 + 1*0.1 = 0.4
    assert result.programmatic_score == pytest.approx(0.4)
    assert result.score == pytest.approx(0.4)
    assert result.decision == "continue"


def test_failed_trials_penalty() -> None:
    sc = _scenario()
    history = _history(
        (0.1, False),
        (0.2, False),
        (0.0, True),
        (0.0, True),
    )
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=None,
        score_threshold=0.7,
    )
    # plausibility=1, failed_penalty=0.5 (2/4 failed), finite=1
    # → 0.6 + 0.3*0.5 + 0.1 = 0.85
    assert result.programmatic_score == pytest.approx(0.85)
    assert result.decision == "approve"


def test_non_finite_best_score_drops_finite_check() -> None:
    sc = _scenario()
    # One successful trial whose score is inf — isolates the finite signal
    # (param plausibility=1.0, failed_penalty=1.0).
    history = [_trial(0, float("inf"), failed=False)]
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=None,
        score_threshold=0.7,
    )
    # 0.6 + 0.3 + 0.0 = 0.9
    assert result.programmatic_score == pytest.approx(0.9)
    assert result.decision == "approve"


def test_empty_history_defaults_finite_to_one() -> None:
    sc = _scenario()
    result = run_tune_judge(
        sc,
        [],
        {"mass_scale": 1.0},
        chat_model=None,
        score_threshold=0.7,
    )
    # plausibility=1, failed_penalty=1 (empty), finite=1 (default)
    assert result.programmatic_score == pytest.approx(1.0)
    assert result.decision == "approve"


def test_missing_param_in_best_params_drops_plausibility() -> None:
    sc = _scenario()
    result = run_tune_judge(
        sc,
        [],
        {},  # missing mass_scale
        chat_model=None,
        score_threshold=0.7,
    )
    # plausibility=0, failed_penalty=1, finite=1 → 0.4
    assert result.programmatic_score == pytest.approx(0.4)
    assert result.decision == "continue"


def test_backend_programmatic_failure_blocks_vlm_approval() -> None:
    """The outer judge must not approve when the backend evaluator already
    proved a hard freeform objective failed."""
    sc = _scenario(
        name="freeform",
        metric="judge_score",
        target={"observations": ["the object does not fall through the floor"]},
    )
    history = [
        _trial(
            0,
            0.5625,
            backend_metrics={
                "programmatic_score": 0.25,
                "programmatic_critique": (
                    "settled=fail; finite_position=pass; ground_clearance=fail"
                ),
            },
        )
    ]
    vlm = StubVLM('{"score": 0.95, "decision": "approve", "reasoning": "looks fine"}')

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        score_threshold=0.7,
    )

    assert result.programmatic_score == pytest.approx(0.25)
    assert result.llm_score == pytest.approx(0.95)
    assert result.score == pytest.approx(0.53)
    assert result.decision == "continue"
    assert "backend_programmatic=0.25" in result.programmatic_critique
    assert "ground_clearance=fail" in result.programmatic_critique


def test_backend_programmatic_hard_fail_cannot_approve_at_threshold() -> None:
    sc = _scenario(
        name="freeform",
        metric="judge_score",
        target={"observations": ["settle on the floor"]},
    )
    history = [
        _trial(
            0,
            0.5,
            backend_metrics={
                "programmatic_score": 0.5,
                "programmatic_critique": (
                    "settled=pass; finite_position=pass; ground_clearance=fail"
                ),
            },
        )
    ]
    vlm = StubVLM('{"score": 1.0, "decision": "approve", "reasoning": "looks fine"}')

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        score_threshold=0.7,
    )

    assert result.programmatic_score == pytest.approx(0.5)
    assert result.llm_score == pytest.approx(1.0)
    assert result.score < 0.7
    assert result.decision == "continue"
    assert "ground_clearance=fail" in result.programmatic_critique


def test_replayed_legacy_upright_component_has_no_effect() -> None:
    sc = _scenario(name="freeform", metric="judge_score")
    history = [
        _trial(
            0,
            0.4,
            backend_metrics={
                "programmatic_score": 0.6,
                "programmatic_critique": (
                    "upright=fail; settled=pass; finite_position=pass"
                ),
            },
        )
    ]
    vlm = StubVLM('{"score": 1.0, "decision": "approve", "reasoning": "looks fine"}')

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        score_threshold=0.7,
    )

    assert result.programmatic_score == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)
    assert result.decision == "approve"
    assert "upright" not in result.programmatic_critique


def test_legacy_upright_only_replay_preserves_non_component_notes() -> None:
    from physics_agent.tasks.judge_tune import _drop_legacy_upright_component

    score, critique = _drop_legacy_upright_component(
        0.2,
        "upright=fail; diagnostic note",
    )

    assert score == pytest.approx(1.0)
    assert critique == "diagnostic note"


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def test_llm_strict_json_response_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = _scenario()
    history = _history((0.1, False))

    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": (
                '{"score": 0.9, "decision": "approve", "reasoning": "looks good"}'
            )
        }

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is False
    assert result.llm_score == pytest.approx(0.9)
    # programmatic=1.0, llm=0.9 → combined = 0.6 + 0.36 = 0.96
    assert result.score == pytest.approx(0.96)
    assert result.decision == "approve"


def test_vlm_judge_tokens_and_temperature_forwarded_for_text_only() -> None:
    sc = _scenario()
    history = _history((0.1, False))

    class RecordingVLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_with_image_caption_pairs(self, **kwargs):
            self.calls.append(kwargs)
            return '{"score": 0.9, "decision": "approve", "reasoning": "looks good"}'

    vlm = RecordingVLM()

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        judge_max_tokens=1234,
        judge_temperature=0.25,
    )

    assert result.llm_unavailable is False
    assert vlm.calls[0]["image_caption_pairs"] == []
    assert vlm.calls[0]["temperature"] == 0.25
    assert vlm.calls[0]["max_tokens"] == 1234


def test_judge_config_can_come_from_scenario_extra_for_vlm() -> None:
    sc = _scenario()
    sc.extra["judge"] = {"max_tokens": 4321, "temperature": 0.15}

    class RecordingVLM:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate_with_image_caption_pairs(self, **kwargs):
            self.calls.append(kwargs)
            return '{"score": 0.8, "decision": "approve", "reasoning": "ok"}'

    vlm = RecordingVLM()

    run_tune_judge(sc, _history((0.1, False)), {"mass_scale": 1.0}, vlm_model=vlm)

    assert vlm.calls[0]["temperature"] == 0.15
    assert vlm.calls[0]["max_tokens"] == 4321


def test_vlm_reference_and_generated_frames_replace_text_llm(tmp_path: Path) -> None:
    sc = _scenario()
    history = _history((0.1, False))
    vlm = StubVLM(
        '{"score": 0.2, "decision": "continue", "reasoning": "motion mismatch"}'
    )
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(
            ("Reference Image 1: target", tmp_path / "ref.png"),
        ),
        generated_image_paths=(tmp_path / "frame_0001.png",),
    )

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        visual_evidence=evidence,
        score_threshold=0.7,
    )

    assert result.llm_unavailable is False
    assert result.llm_score == pytest.approx(0.2)
    assert result.score == pytest.approx(0.68)
    assert result.decision == "continue"
    assert result.extra["judge_modality"] == "vlm"
    assert result.extra["reference_image_count"] == 1
    assert result.extra["generated_image_count"] == 1
    assert result.extra["visual_evidence"] == {
        "reference_images": [
            {
                "caption": "Reference Image 1: target",
                "path": str(tmp_path / "ref.png"),
            }
        ],
        "generated_images": [
            {
                "caption": "Generated Physics Output - Frame 1:",
                "path": str(tmp_path / "frame_0001.png"),
            }
        ],
        "comparison_image": None,
        "reference_error": None,
        "generated_error": None,
        "comparison_error": None,
    }
    pairs = vlm.calls[0]["image_caption_pairs"]
    assert pairs[0] == ("Reference Image 1: target", tmp_path / "ref.png")
    assert pairs[1] == (
        "Generated Physics Output - Frame 1:",
        tmp_path / "frame_0001.png",
    )
    assert vlm.calls[0]["max_tokens"] == DEFAULT_JUDGE_MAX_TOKENS


def test_vlm_visual_media_is_sampled_before_invoke(tmp_path: Path) -> None:
    sc = _scenario()
    history = _history((0.1, False))
    vlm = StubVLM('{"score": 0.8, "decision": "approve", "reasoning": "ok"}')
    references = tuple(
        (f"Reference Image {idx}: target", tmp_path / f"ref_{idx:02d}.png")
        for idx in range(1, 21)
    )
    generated = tuple(tmp_path / f"frame_{idx:04d}.png" for idx in range(1, 61))
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=references,
        generated_image_paths=generated,
    )

    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        visual_evidence=evidence,
    )

    pairs = vlm.calls[0]["image_caption_pairs"]
    assert len(pairs) == 24
    assert pairs[0] == references[0]
    assert pairs[7] == references[-1]
    assert pairs[8] == ("Generated Physics Output - Frame 1:", generated[0])
    assert pairs[-1] == ("Generated Physics Output - Frame 60:", generated[-1])
    assert result.extra["reference_image_count"] == 20
    assert result.extra["generated_image_count"] == 60
    assert result.extra["judge_reference_frames"] == 8
    assert result.extra["judge_generated_frames"] == 16

    custom_vlm = StubVLM('{"score": 0.8, "decision": "approve", "reasoning": "ok"}')
    run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=custom_vlm,
        visual_evidence=evidence,
        judge_reference_frames=5,
        judge_generated_frames=7,
    )

    custom_pairs = custom_vlm.calls[0]["image_caption_pairs"]
    assert len(custom_pairs) == 12
    assert custom_pairs[0] == references[0]
    assert custom_pairs[4] == references[-1]
    assert custom_pairs[5] == ("Generated Physics Output - Frame 1:", generated[0])
    assert custom_pairs[-1] == ("Generated Physics Output - Frame 60:", generated[-1])


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("judge_reference_frames", {"judge_reference_frames": cast(int, 3.0)}),
        ("judge_generated_frames", {"judge_generated_frames": cast(int, "7")}),
    ],
)
def test_vlm_judge_validates_frame_limits_without_visual_evidence(
    field_name: str,
    kwargs: dict[str, int],
) -> None:
    sc = _scenario()
    history = _history((0.1, False))
    vlm = StubVLM('{"score": 0.8, "decision": "approve", "reasoning": "ok"}')

    with pytest.raises(ValueError, match=field_name):
        run_tune_judge(
            sc,
            history,
            {"mass_scale": 1.0},
            vlm_model=vlm,
            **kwargs,
        )

    assert vlm.calls == []


def test_vlm_judge_max_tokens_override(tmp_path: Path) -> None:
    sc = _scenario()
    history = _history((0.1, False))
    vlm = StubVLM('{"score": 0.8, "decision": "approve", "reasoning": "ok"}')
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(("Reference Image 1:", tmp_path / "ref.png"),),
        generated_image_paths=(tmp_path / "frame_0001.png",),
    )

    run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        visual_evidence=evidence,
        judge_max_tokens=1234,
    )

    assert vlm.calls[0]["max_tokens"] == 1234


def test_vlm_judge_temperature_override(tmp_path: Path) -> None:
    sc = _scenario()
    history = _history((0.1, False))
    vlm = StubVLM('{"score": 0.8, "decision": "approve", "reasoning": "ok"}')
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(("Reference Image 1:", tmp_path / "ref.png"),),
        generated_image_paths=(tmp_path / "frame_0001.png",),
    )

    run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        vlm_model=vlm,
        visual_evidence=evidence,
        judge_temperature=0.3,
    )

    assert vlm.calls[0]["temperature"] == 0.3


def test_vlm_reference_without_generated_frames_degrades(tmp_path: Path) -> None:
    sc = _scenario()
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(("Reference Image 1:", tmp_path / "ref.png"),),
    )

    result = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        vlm_model=StubVLM('{"score": 1, "reasoning": "unused"}'),
        visual_evidence=evidence,
    )

    assert result.llm_unavailable is True
    assert result.llm_score == pytest.approx(result.programmatic_score)
    assert "no generated render frames" in result.llm_critique
    assert result.extra["judge_modality"] == "vlm"


def test_vlm_invoke_exception_records_exception_type() -> None:
    result = run_tune_judge(
        _scenario(),
        _history((0.1, False)),
        {"mass_scale": 1.0},
        vlm_model=RaisingVLM(),
    )

    assert result.llm_unavailable is True
    assert "TimeoutError" in result.llm_critique
    assert "provider deadline" not in result.llm_critique


def test_llm_response_with_preamble_still_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()

    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": (
                "Here is the verdict: "
                '{"score": 0.5, "decision": "continue", "reasoning": "meh"}'
            )
        }

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is False
    assert result.llm_score == pytest.approx(0.5)


def test_llm_response_with_code_fence_still_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()

    fenced = '```json\n{"score": 0.8, "decision": "approve", "reasoning": "ok"}\n```'

    def fake_generate(model, prompt, system_prompt=None):
        return {"response": fenced}

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is False
    assert result.llm_score == pytest.approx(0.8)


def test_llm_bad_json_falls_back_to_programmatic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()
    history = _history((0.1, False))

    def fake_generate(model, prompt, system_prompt=None):
        return {"response": "not json at all"}

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is True
    assert result.llm_score == pytest.approx(result.programmatic_score)
    assert result.score == pytest.approx(result.programmatic_score)
    assert result.decision == "approve"  # programmatic=1.0 ≥ 0.7


def test_llm_error_response_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = _scenario()
    history = _history((0.1, False))

    def fake_generate(model, prompt, system_prompt=None):
        return {"error": "rate limit"}

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is True
    assert result.llm_score == pytest.approx(result.programmatic_score)


def test_llm_score_clamped_to_unit_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()

    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": ('{"score": 1.7, "decision": "approve", "reasoning": "n/a"}')
        }

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is False
    assert result.llm_score == pytest.approx(1.0)


def test_llm_decision_does_not_override_threshold_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()

    # Force programmatic low: param out of bounds → programmatic=0.4
    # llm_score=0.5 → combined = 0.6*0.4 + 0.4*0.5 = 0.44 < 0.7
    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": (
                '{"score": 0.5, "decision": "approve", "reasoning": "trust me"}'
            )
        }

    _patch_llm(monkeypatch, fake_generate)
    result = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 10.0},  # out of bounds
        chat_model=StubChat(),
        score_threshold=0.7,
    )
    assert result.llm_unavailable is False
    assert result.score < 0.7
    # The LLM said "approve" but the authoritative decision is threshold-based.
    assert result.decision == "continue"


# ---------------------------------------------------------------------------
# Threshold control
# ---------------------------------------------------------------------------


def test_threshold_zero_always_approves() -> None:
    sc = _scenario()
    # Force programmatic very low (out of bounds + missing finite signal).
    history = [_trial(0, float("nan"), failed=True)]
    result = run_tune_judge(
        sc,
        history,
        {"mass_scale": 10.0},
        chat_model=None,
        score_threshold=0.0,
    )
    # Even programmatic=0 still approves with threshold=0 (>= 0 is true).
    assert result.programmatic_score >= 0.0
    assert result.decision == "approve"


def test_threshold_one_only_approves_perfect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = _scenario()

    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": ('{"score": 1.0, "decision": "approve", "reasoning": "ok"}')
        }

    _patch_llm(monkeypatch, fake_generate)
    # programmatic=1.0, llm=1.0 → combined=1.0 → approve at threshold=1.0
    result_pass = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=1.0,
    )
    assert result_pass.score == pytest.approx(1.0)
    assert result_pass.decision == "approve"

    # Drop llm one notch → combined < 1.0 → continue.
    def fake_lower(model, prompt, system_prompt=None):
        return {
            "response": ('{"score": 0.99, "decision": "approve", "reasoning": "ok"}')
        }

    _patch_llm(monkeypatch, fake_lower)
    result_fail = run_tune_judge(
        sc,
        _history((0.1, False)),
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        score_threshold=1.0,
    )
    assert result_fail.score < 1.0
    assert result_fail.decision == "continue"


# ---------------------------------------------------------------------------
# iteration field
# ---------------------------------------------------------------------------


def test_iteration_field_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``iteration`` is propagated onto the result every call.

    Cache was removed, so each call recomputes; the field is purely
    metadata for the ``tune_results.json`` audit trail.
    """
    sc = _scenario()
    history = _history((0.1, False))

    def fake_generate(model, prompt, system_prompt=None):
        return {
            "response": ('{"score": 0.9, "decision": "approve", "reasoning": "ok"}')
        }

    _patch_llm(monkeypatch, fake_generate)

    first = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        iteration=3,
    )
    assert first.iterations == 3

    second = run_tune_judge(
        sc,
        history,
        {"mass_scale": 1.0},
        chat_model=StubChat(),
        iteration=5,
    )
    assert second.iterations == 5


# ---------------------------------------------------------------------------
# Trajectory enrichment — per-trial metrics whitelist + best_trial_summary
# ---------------------------------------------------------------------------


def _capture_prompt(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the compatibility VLM method to capture the judge prompt arg.

    Returns a dict that the test can read after run_tune_judge runs.
    The fake model response always approves so the call returns cleanly.
    """
    captured: dict = {}

    def fake(model, prompt, system_prompt=None):
        captured["prompt"] = prompt
        return {"response": '{"score": 0.9, "decision": "approve", "reasoning": "ok"}'}

    _patch_llm(monkeypatch, fake)
    return captured


def _payload_from_prompt(prompt: str) -> dict:
    """Extract the JSON payload from the judge prompt wrapper text."""
    start = prompt.index("{")
    return json.loads(prompt[start:])


def test_build_prompt_includes_per_trial_metrics_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-trial backend_metrics scalars (settle_distance, etc.) land in
    each history_summary entry; non-whitelisted keys (file paths, the
    trajectory object) are dropped."""
    import json

    sc = _scenario()
    history = [
        _trial(
            0,
            0.1,
            backend_metrics={
                "settle_distance": 0.000448,
                "max_bounce_height": -0.025,
                "first_bounce_height": 0.025,
                "final_position": [0.0, 0.498, 0.0],
                # These are NOT whitelisted and must NOT leak into the prompt.
                "trajectory_jsonl": "/tmp/x/trajectory.jsonl",
                "recording_usd": "/tmp/x/recording.usd",
                "recording_usda": "/tmp/x/recording.usd",
                "scene_usd": "/tmp/x/scene.usd",
                "trajectory": [(0.0, [0.0] * 7, [0.0] * 6)] * 60,
            },
        ),
    ]

    captured = _capture_prompt(monkeypatch)
    run_tune_judge(sc, history, {"mass_scale": 1.0}, chat_model=StubChat())

    payload = _payload_from_prompt(captured["prompt"])
    metrics = payload["history_summary"][0]["metrics"]

    assert metrics["settle_distance"] == pytest.approx(0.000448)
    assert metrics["max_bounce_height"] == pytest.approx(-0.025)
    assert metrics["first_bounce_height"] == pytest.approx(0.025)
    assert metrics["final_position"] == [0.0, pytest.approx(0.498), 0.0]
    assert "trajectory_jsonl" not in metrics
    assert "recording_usd" not in metrics
    assert "recording_usda" not in metrics
    assert "scene_usd" not in metrics
    assert "trajectory" not in metrics


def test_build_prompt_includes_best_trial_summary_when_jsonl_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the winning trial's backend_metrics carries a readable
    trajectory_jsonl path, the prompt gains a best_trial_summary block
    with the trajectory_summary fields (max_linear_speed, settle_time_s,
    fell_over, etc.)."""
    import json

    # Author a tiny trajectory.jsonl directly — three frames, monotonic
    # falling y, non-zero linear velocity peak.
    jsonl_path = tmp_path / "trajectory.jsonl"
    frames = [
        {
            "frame": 0,
            "t": 0.0,
            "pose": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "frame": 15,
            "t": 0.5,
            "pose": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0],
            "vel": [0.0, -2.5, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "frame": 30,
            "t": 1.0,
            "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "vel": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8"
    )

    sc = _scenario()
    history = [
        _trial(0, 0.5, backend_metrics={"trajectory_jsonl": str(jsonl_path)}),
        _trial(1, 0.1, backend_metrics={"trajectory_jsonl": str(jsonl_path)}),
        _trial(2, 0.3, backend_metrics={}),  # no jsonl — must not be picked
    ]

    captured = _capture_prompt(monkeypatch)
    run_tune_judge(sc, history, {"mass_scale": 1.0}, chat_model=StubChat())

    payload = _payload_from_prompt(captured["prompt"])

    assert "best_trial_summary" in payload, (
        "winning trial has trajectory_jsonl; best_trial_summary must be present"
    )
    summary = payload["best_trial_summary"]
    # trajectory_summary returns these keys (verified in trajectory.py:223).
    assert "max_linear_speed" in summary
    assert "settle_time_s" in summary
    assert "fell_over" in summary
    # Peak |v| in our synthetic trajectory is 2.5 (from frame 15's vel).
    assert summary["max_linear_speed"] == pytest.approx(2.5)


def test_explicit_world_up_in_backend_metrics_is_preferred_over_inferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When backend_metrics carries an explicit ``world_up`` from the
    scene builder, the judge uses it directly instead of inferring from
    ``rest_position``.

    Regression for the corner-origin-asset case: a body whose bbox-min
    already sits at the stage origin gets ``rest_position == [0, 0, 0]``,
    and ``infer_world_up`` falls back to legacy Y-up — wrong on a Z-up
    stage. The scene builder now stamps the actual axis as ``world_up``
    so the judge can route around the inference ambiguity.
    """
    import json

    jsonl_path = tmp_path / "trajectory.jsonl"
    # Z-up body that tips over its own +Z axis: the body's local up
    # rotates from +Z to +X via a 90° rotation around Y. Under Y-up
    # default, fell_over would (incorrectly) compare against world +Y
    # which the body's local +Y still aligns with — so fell_over=False.
    # Under explicit Z-up world_up, fell_over correctly fires.
    import math as _math

    half_pi = _math.pi / 2
    qy = _math.sin(half_pi / 2)
    qw = _math.cos(half_pi / 2)
    frames = [
        {
            "frame": 0,
            "t": 0.0,
            "pose": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],  # upright on Z
            "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        {
            "frame": 1,
            "t": 0.5,
            # 90° around Y → body's local +Z now points along world +X.
            "pose": [0.0, 0.0, 0.0, 0.0, qy, 0.0, qw],
            "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8"
    )

    sc = _scenario()
    history = [
        # Corner-origin shape: rest_position is all-zero (so infer_world_up
        # would fall back to Y-up), but the scene builder authored world_up
        # explicitly as Z.
        _trial(
            0,
            0.1,
            backend_metrics={
                "trajectory_jsonl": str(jsonl_path),
                "rest_position": [0.0, 0.0, 0.0],
                "world_up": [0.0, 0.0, 1.0],
            },
        ),
    ]

    captured = _capture_prompt(monkeypatch)
    run_tune_judge(sc, history, {"mass_scale": 1.0}, chat_model=StubChat())

    payload = _payload_from_prompt(captured["prompt"])
    summary = payload["best_trial_summary"]
    # Under the explicit Z-up world_up, this 90° pitch DOES count as
    # fell_over=True. Under the legacy inference fallback (Y-up) it
    # would be False.
    assert summary["fell_over"] is True


def test_build_prompt_omits_summary_when_jsonl_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no trial carries a trajectory_jsonl path, the prompt simply
    omits best_trial_summary — the judge must still produce a result
    with no errors."""
    import json

    sc = _scenario()
    history = _history((0.1, False), (0.2, False))  # no backend_metrics

    captured = _capture_prompt(monkeypatch)
    result = run_tune_judge(sc, history, {"mass_scale": 1.0}, chat_model=StubChat())

    payload = _payload_from_prompt(captured["prompt"])
    assert "best_trial_summary" not in payload
    # Judge still ran cleanly.
    assert isinstance(result, JudgeResult)
    assert not result.llm_unavailable


def test_build_prompt_omits_summary_when_jsonl_path_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bogus trajectory_jsonl path is treated like missing — degrade
    quietly so a single broken trial cannot abort the judge."""
    import json

    sc = _scenario()
    history = [
        _trial(
            0,
            0.1,
            backend_metrics={
                "trajectory_jsonl": str(tmp_path / "does_not_exist.jsonl")
            },
        ),
    ]

    captured = _capture_prompt(monkeypatch)
    result = run_tune_judge(sc, history, {"mass_scale": 1.0}, chat_model=StubChat())

    payload = _payload_from_prompt(captured["prompt"])
    assert "best_trial_summary" not in payload
    assert isinstance(result, JudgeResult)
