# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material-agent judge helpers."""

from __future__ import annotations

import pytest

from material_agent.tasks.judge import JudgeTask, _VlmJudgeResult


class _FixedVlmDecisionJudgeTask(JudgeTask):
    def __init__(self, image_decision: str, *, decision_parsed: bool = True) -> None:
        super().__init__()
        self._image_decision = image_decision
        self._decision_parsed = decision_parsed

    def _run_vlm_judge(
        self,
        context: dict,
        judge_config: dict,
        iteration_count: int,
    ) -> _VlmJudgeResult:
        return _VlmJudgeResult(
            score=0.9,
            critique="Score: 9\nRecommendation: keep current palette.",
            decision=self._image_decision,
            decision_parsed=self._decision_parsed,
        )


def test_judge_task_uses_shared_labeled_score_and_decision_parsing() -> None:
    decision, score, reasoning, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
**Critique:**
The material assignment is coherent but still needs a targeted handle fix.

**Score:** 6/10

**Decision:** CONTINUE
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.6
    assert "material assignment" in reasoning
    assert decision_parsed is True


def test_judge_task_score_threshold_still_overrides_approve_decision() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The model still has inconsistent materials.
Score: 5
Decision: APPROVE
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.5
    assert decision_parsed is True


def test_judge_task_handles_multiline_fractional_score() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is poor.
Score:
1/10
Decision: APPROVE
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.1
    assert decision_parsed is True


def test_judge_task_treats_unqualified_decimal_as_ten_point_score() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is poor.
Score: 0.5
Decision: APPROVE
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.05
    assert decision_parsed is True


def test_judge_task_handles_multiline_decision_value() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment needs more work.
Score: 8
Decision:
CONTINUE
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_handles_multiline_decision_preamble() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision:
Given the high visual score,
APPROVE
""",
        iteration_count=1,
    )

    assert decision == "approve"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_precise_decision_is_not_overridden_by_later_prose() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision: APPROVE
Recommendation: Continue monitoring minor texture details.
""",
        iteration_count=1,
    )

    assert decision == "approve"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_accepts_approve_with_no_issue_reasoning() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision: APPROVE. No issues remain.
""",
        iteration_count=1,
    )

    assert decision == "approve"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_decision_stops_before_improvement_suggestions() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision: APPROVE
Improvement Suggestions: Continue using the current coherent palette.
""",
        iteration_count=1,
    )

    assert decision == "approve"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_whitespace_decision_is_not_overridden_by_later_mentions() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision APPROVE
Historical note: a previous response said decision: continue.
""",
        iteration_count=1,
    )

    assert decision == "approve"
    assert score == 0.8
    assert decision_parsed is True


def test_judge_task_missing_decision_defaults_to_continue() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Recommendation: Keep the current palette.
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.8
    assert decision_parsed is False


@pytest.mark.parametrize(
    "decision_line",
    [
        "Decision: [APPROVE if score >= 7, or CONTINUE if targeted fixes are needed]",
        "Decision: APPROVE if score >= 7, CONTINUE if targeted fixes are needed",
    ],
)
def test_judge_task_option_list_decision_echo_defaults_to_continue(
    decision_line: str,
) -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        f"""
Critique: The material assignment is acceptable.
Score: 8
{decision_line}
""",
        iteration_count=1,
    )

    assert decision == "continue"
    assert score == 0.8
    assert decision_parsed is False


def test_judge_task_run_respects_fail_closed_vlm_decision() -> None:
    context = {
        "iteration_count": 1,
        "materials_applied": {},
        "assignment_stats": {"total_prims": 1},
        "judge_config": {
            "score_threshold": 0.7,
            "prediction_analysis": {"enabled": False},
        },
    }

    result = _FixedVlmDecisionJudgeTask("continue").run(context)

    assert result["judge_decision"] == "continue"
    assert result["continue_iteration"] is True
    assert result["judge_score"] == 0.9
    assert result["judge_image_decision"] == "continue"
    assert result["judge_image_decision_parsed"] is True


def test_judge_task_refreshes_evaluation_signals_each_run() -> None:
    context = {
        "iteration_count": 1,
        "materials_applied": {},
        "assignment_stats": {"total_prims": 1},
        "evaluation_signals": {"schema_version": "stale"},
        "judge_config": {
            "score_threshold": 0.7,
            "prediction_analysis": {"enabled": False},
            "reference_images": ["/tmp/reference.png"],
        },
        "previous_prim_feedback": {"/World/Mesh": "Use darker material."},
    }

    result = _FixedVlmDecisionJudgeTask("continue").run(context)

    assert result["evaluation_signals"]["schema_version"] == (
        "material-self-evaluation-signals/v1"
    )
    assert result["evaluation_signals"]["visual_evaluation"][
        "reference_image_paths"
    ] == ["/tmp/reference.png"]
    assert result["evaluation_signals"]["prediction_analysis"][
        "previous_prim_feedback"
    ] == {"/World/Mesh": "Use darker material."}


def test_judge_task_explicit_continue_vetoes_prediction_blend() -> None:
    context = {
        "iteration_count": 1,
        "materials_applied": {},
        "assignment_stats": {"total_prims": 1},
        "judge_config": {
            "score_threshold": 0.7,
            "prediction_analysis": {"enabled": True, "weight": 0.6},
        },
    }

    result = _FixedVlmDecisionJudgeTask("continue").run(context)

    assert result["judge_decision"] == "continue"
    assert result["continue_iteration"] is True
    assert result["judge_score"] == 0.96
    assert result["judge_image_decision"] == "continue"
    assert result["judge_image_decision_parsed"] is True


def test_judge_task_unparseable_image_decision_vetoes_prediction_blend() -> None:
    context = {
        "iteration_count": 1,
        "materials_applied": {},
        "assignment_stats": {"total_prims": 1},
        "judge_config": {
            "score_threshold": 0.7,
            "prediction_analysis": {"enabled": True, "weight": 0.6},
        },
    }

    result = _FixedVlmDecisionJudgeTask(
        "continue",
        decision_parsed=False,
    ).run(context)

    assert result["judge_decision"] == "continue"
    assert result["continue_iteration"] is True
    assert result["judge_score"] == 0.96
    assert result["judge_image_decision"] == "continue"
    assert result["judge_image_decision_parsed"] is False


def test_judge_task_approve_image_decision_uses_prediction_blend() -> None:
    context = {
        "iteration_count": 1,
        "materials_applied": {},
        "assignment_stats": {"total_prims": 1},
        "judge_config": {
            "score_threshold": 0.7,
            "prediction_analysis": {"enabled": True, "weight": 0.6},
        },
    }

    result = _FixedVlmDecisionJudgeTask("approve").run(context)

    assert result["judge_decision"] == "approve"
    assert result["continue_iteration"] is False
    assert result["judge_score"] == 0.96
    assert result["judge_image_decision"] == "approve"
    assert result["judge_image_decision_parsed"] is True


def test_judge_task_uses_configured_score_threshold_for_vlm_decision() -> None:
    decision, score, _, decision_parsed = JudgeTask()._parse_vlm_critique(
        {},
        """
Critique: The material assignment is acceptable.
Score: 8
Decision: APPROVE
""",
        iteration_count=1,
        score_threshold=0.85,
    )

    assert decision == "continue"
    assert score == 0.8
    assert decision_parsed is True
