# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for contract-neutral look_right visual judge planning."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_understanding.functions.cv.look_right import (
    VISUAL_BLOCKING_DEFECT,
    VISUAL_EVIDENCE_MISSING,
    VISUAL_JUDGE_UNAVAILABLE,
    VISUAL_LOW_CONFIDENCE,
    VISUAL_PROMPT_MISMATCH,
    VISUAL_PROMPT_MISSING,
    VISUAL_REFERENCE_MISMATCH,
    VISUAL_RENDER_PREFLIGHT_FAILED,
    VISUAL_RENDER_PREFLIGHT_UNAVAILABLE,
    LookRightIssue,
    LookRightJudgePlan,
    _safe_optional_attr,
    _token_usage_dict,
    build_look_right_judge_plan,
    invoke_look_right_judge,
    normalize_look_right_judgment,
    parse_look_right_judgment,
)


def _issue_codes(plan: LookRightJudgePlan) -> set[str]:
    return {issue.code for issue in plan.issues}


def _path(value: str) -> str:
    return str(Path(value))


class _RecordingVLM:
    backend_name = "fake"
    model_name = "fake-vlm"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_with_image_caption_pairs(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "Critique: ok\nScore: 9\nDecision: PASS\nIssue Codes: none"


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _RecordingLLM:
    backend_name = "fake"
    model_name = "fake-llm"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def invoke(self, messages: list[object], **kwargs: object) -> _FakeLLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return _FakeLLMResponse(self.response)


class _GeneratingLLM:
    backend_name = "fake-generate"
    model_name = "fake-generate-llm"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.response


def test_build_look_right_judge_plan_orders_multimodal_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate that the generated asset looks like a yellow toolbox.",
        current_image_paths=[
            Path("renders/front.png"),
            "renders/back.png",
        ],
        reference_image_paths=["refs/toolbox_thumb.png"],
        focused_image_paths={"/World/Handle": ["renders/focus_handle.png"]},
        focus_prim_paths=["/World/Handle"],
    )

    assert plan.ready_for_judge
    assert plan.focus_prim_paths == ("/World/Handle",)
    assert plan.image_caption_pairs == (
        ("Reference Image 1:", _path("refs/toolbox_thumb.png")),
        ("Current Asset Evidence - View 1:", _path("renders/front.png")),
        ("Current Asset Evidence - View 2:", _path("renders/back.png")),
        (
            "Focused Asset Evidence - /World/Handle - View 1:",
            _path("renders/focus_handle.png"),
        ),
    )
    assert "Do not invent or" in plan.final_prompt
    assert "Do not average away defects across views" in plan.final_prompt
    assert "Do not escalate NEEDS_REFINEMENT to FAIL only because" in (
        plan.final_prompt
    )
    assert "views revealed new defects or only confirmed defects" in plan.final_prompt
    assert "visual.blocking_defect" in plan.final_prompt
    assert "/World/Handle" in plan.final_prompt
    assert [image.source_kind for image in plan.evidence_images] == [
        "reference_image",
        "direct_image",
        "direct_image",
        "focused_render",
    ]


@pytest.mark.parametrize(
    ("current_paths", "expected_pairs"),
    [
        (
            ["renders/corner.png"],
            (("Current Asset Evidence - View 1:", _path("renders/corner.png")),),
        ),
        (
            ["renders/corner.png", "renders/front.png", "renders/right.png"],
            (
                ("Current Asset Evidence - View 1:", _path("renders/corner.png")),
                ("Current Asset Evidence - View 2:", _path("renders/front.png")),
                ("Current Asset Evidence - View 3:", _path("renders/right.png")),
            ),
        ),
    ],
)
def test_invoke_look_right_judge_passes_single_or_multiple_images_in_one_call(
    current_paths: list[str],
    expected_pairs: tuple[tuple[str, str], ...],
) -> None:
    plan = build_look_right_judge_plan(
        "Validate the rendered asset.",
        current_image_paths=current_paths,
    )
    vlm = _RecordingVLM()

    invocation = invoke_look_right_judge(plan, vlm)

    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_caption_pairs"] == list(expected_pairs)
    assert invocation.raw_response.startswith("Critique: ok")
    assert invocation.metadata["image_count"] == len(expected_pairs)


def test_invoke_look_right_judge_serializes_token_usage() -> None:
    class Usage:
        def to_dict(self) -> dict[str, int]:
            return {"input_tokens": 3, "output_tokens": 4}

    plan = build_look_right_judge_plan(
        "Validate the rendered asset.",
        current_image_paths=["renders/corner.png"],
    )
    vlm = _RecordingVLM()
    vlm.last_token_usage = Usage()

    invocation = invoke_look_right_judge(plan, vlm)

    assert invocation.to_dict()["token_usage"] == {
        "input_tokens": 3,
        "output_tokens": 4,
    }


def test_look_right_issue_to_dict() -> None:
    issue = LookRightIssue(
        code=VISUAL_LOW_CONFIDENCE,
        message="Evidence is limited.",
        severity="warning",
        subject="render.png",
        details={"view": "front"},
    )

    assert issue.to_dict() == {
        "code": VISUAL_LOW_CONFIDENCE,
        "message": "Evidence is limited.",
        "severity": "warning",
        "subject": "render.png",
        "details": {"view": "front"},
    }


def test_normalize_look_right_judgment_uses_llm_final_decision() -> None:
    llm = _RecordingLLM(
        """
{"decision": "needs_refinement", "score": 0.82,
 "issue_codes": ["visual.low_confidence"],
 "rationale": "The judge explicitly rejects PASS.",
 "evidence_notes": "Decision text says it should not pass."}
"""
    )

    result = normalize_look_right_judgment(
        """
Critique: The render still has unresolved issues.
Score: 9
Decision: I do not think this should PASS
Issue Codes: none
""",
        llm_judge=llm,
    )

    assert result.judgment.verdict == "needs_refinement"
    assert result.judgment.score == 0.82
    assert result.judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)
    assert "rejects PASS" in result.judgment.reasoning
    assert result.metadata["method"] == "llm"
    assert llm.calls[0]["kwargs"] == {"temperature": 0.0, "max_tokens": 512}


def test_normalize_look_right_judgment_parser_result_to_dict() -> None:
    result = normalize_look_right_judgment(
        """
Critique: The render matches the requested asset.
Score: 9
Decision: PASS
Issue Codes: none
"""
    )

    data = result.to_dict()
    assert data["judgment"]["verdict"] == "pass"
    assert data["metadata"] == {"method": "parser", "llm_invoked": False}


def test_normalize_look_right_judgment_is_stable_for_same_response() -> None:
    raw_response = """
Critique: The corner view and added views show the same acceptable result.
Score: 9
Decision: PASS
Issue Codes: none
"""
    llm_response = (
        '{"decision": "pass", "score": 0.9, "issue_codes": [], '
        '"rationale": "The judge clearly accepts the asset."}'
    )
    single_view_plan = build_look_right_judge_plan(
        "Validate the toolbox.",
        current_image_paths=["renders/corner.png"],
    )
    multi_view_plan = build_look_right_judge_plan(
        "Validate the toolbox.",
        current_image_paths=[
            "renders/corner.png",
            "renders/front.png",
            "renders/right.png",
        ],
    )

    single_result = normalize_look_right_judgment(
        raw_response,
        llm_judge=_RecordingLLM(llm_response),
        judge_plan=single_view_plan,
    )
    multi_result = normalize_look_right_judgment(
        raw_response,
        llm_judge=_RecordingLLM(llm_response),
        judge_plan=multi_view_plan,
    )

    assert single_result.judgment.to_dict() == multi_result.judgment.to_dict()


def test_normalize_look_right_judgment_preserves_parser_blocking_issue_floor() -> None:
    result = normalize_look_right_judgment(
        """
Critique: The asset mismatches the reference.
Score: 9
Decision: PASS
Issue Codes: visual.reference_mismatch
""",
        llm_judge=_RecordingLLM(
            '{"decision": "pass", "score": 0.9, "issue_codes": [], '
            '"rationale": "The normalizer missed the issue code."}'
        ),
    )

    assert result.judgment.verdict == "needs_refinement"
    assert result.judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)


def test_normalize_look_right_judgment_fails_closed_on_llm_error() -> None:
    result = normalize_look_right_judgment(
        """
Critique: The render mostly matches.
Score: 9
Decision: PASS
Issue Codes: none
""",
        llm_judge=_RecordingLLM(RuntimeError("api_key=secret-token")),
    )

    assert result.judgment.verdict == "warn"
    assert result.judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)
    assert result.metadata["method"] == "llm_failed"
    assert "secret-token" not in str(result.metadata)


def test_build_look_right_judge_plan_allows_prompt_only_references() -> None:
    plan = build_look_right_judge_plan(
        "Validate that the asset has the requested red finish.",
        current_image_paths=["renders/front.png"],
    )

    assert plan.ready_for_judge
    assert "No reference images were supplied" in plan.final_prompt
    assert plan.to_dict()["ready_for_judge"] is True


def test_build_look_right_judge_plan_reports_missing_current_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate that the asset looks like a wheeled scaffold.",
        reference_image_paths=["refs/scaffold.png"],
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_EVIDENCE_MISSING}


def test_build_look_right_judge_plan_reports_missing_prompt() -> None:
    plan = build_look_right_judge_plan(
        "",
        current_image_paths=["renders/front.png"],
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_PROMPT_MISSING}


def test_build_look_right_judge_plan_orders_render_and_video_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate the asset from rendered views and sampled rollout frames.",
        current_image_paths=["inputs/uploaded_reference_view.png"],
        render_image_paths=["renders/front.png", "renders/back.png"],
        sampled_video_frame_paths=["frames/frame_0001.png", "frames/frame_0010.png"],
        reference_image_paths=["refs/reference.png"],
    )

    assert plan.ready_for_judge
    assert plan.image_caption_pairs == (
        ("Reference Image 1:", _path("refs/reference.png")),
        (
            "Current Asset Evidence - View 1:",
            _path("inputs/uploaded_reference_view.png"),
        ),
        ("Current Render Output - View 1:", _path("renders/front.png")),
        ("Current Render Output - View 2:", _path("renders/back.png")),
        ("Sampled Video Frame - Frame 1:", _path("frames/frame_0001.png")),
        ("Sampled Video Frame - Frame 2:", _path("frames/frame_0010.png")),
    )
    assert [image.source_kind for image in plan.evidence_images] == [
        "reference_image",
        "direct_image",
        "render_output",
        "render_output",
        "sampled_video_frame",
        "sampled_video_frame",
    ]
    assert plan.to_dict()["evidence_images"][2]["source_kind"] == "render_output"


def test_build_look_right_judge_plan_accepts_sampled_frames_as_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate behavior from sampled rollout frames.",
        sampled_video_frame_paths=["frames/frame_0001.png"],
    )

    assert plan.ready_for_judge
    assert plan.image_caption_pairs == (
        ("Sampled Video Frame - Frame 1:", _path("frames/frame_0001.png")),
    )
    assert _issue_codes(plan) == set()


def test_build_look_right_judge_plan_blocks_when_vlm_unavailable() -> None:
    plan = build_look_right_judge_plan(
        "Validate the visual appearance.",
        current_image_paths=["renders/front.png"],
        vlm_available=False,
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_JUDGE_UNAVAILABLE}
    issue = plan.issues[0]
    assert issue.severity == "warning"
    assert issue.details == {"vlm_available": False}


def test_build_look_right_judge_plan_blocks_failed_render_preflight() -> None:
    plan = build_look_right_judge_plan(
        "Validate the rendered asset.",
        render_image_paths=["renders/front.png"],
        render_valid_result={
            "status": "fail",
            "issues": [
                {"code": "render.blank_image"},
                {"code": "ovrtx.render_artifact_detected"},
            ],
        },
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_FAILED}
    assert plan.issues[0].details["render_valid_issue_codes"] == [
        "render.blank_image",
        "ovrtx.render_artifact_detected",
    ]


def test_look_right_plan_blocks_failed_render_with_direct_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate a directly supplied image after generated render preflight failed.",
        current_image_paths=["inputs/photo.png"],
        render_valid_result={"status": "failed"},
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_FAILED}


@pytest.mark.parametrize("verdict", ["fail", "failed", "error"])
def test_build_look_right_judge_plan_blocks_failed_render_verdict_aliases(
    verdict: str,
) -> None:
    plan = build_look_right_judge_plan(
        "Validate rendered evidence.",
        render_image_paths=["renders/front.png"],
        render_valid_result={"verdict": verdict},
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_FAILED}
    assert plan.issues[0].details["render_valid_verdict"] == verdict


def test_build_look_right_judge_plan_warns_on_unavailable_render_preflight() -> None:
    plan = build_look_right_judge_plan(
        "Validate a directly supplied image even when generated render preflight skipped.",
        current_image_paths=["inputs/photo.png"],
        render_valid_result={
            "status": "skipped",
            "issues": [{"code": "render.evidence_missing"}],
        },
    )

    assert plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_UNAVAILABLE}
    assert plan.issues[0].severity == "warning"


def test_look_right_plan_blocks_unavailable_generated_render_evidence() -> None:
    plan = build_look_right_judge_plan(
        "Validate generated render evidence after render preflight skipped.",
        render_image_paths=["renders/front.png"],
        render_valid_result={"status": "skipped"},
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_UNAVAILABLE}
    assert plan.evidence_images[0].source_kind == "render_output"


@pytest.mark.parametrize("verdict", ["skipped", "warn", "warning"])
def test_build_look_right_judge_plan_warns_on_unavailable_render_verdict_aliases(
    verdict: str,
) -> None:
    plan = build_look_right_judge_plan(
        "Validate a directly supplied image when render evidence is unavailable.",
        current_image_paths=["inputs/photo.png"],
        render_valid_result={"verdict": verdict},
    )

    assert plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_RENDER_PREFLIGHT_UNAVAILABLE}
    assert plan.issues[0].details["render_valid_verdict"] == verdict


def test_build_look_right_judge_plan_allows_passing_render_preflight() -> None:
    plan = build_look_right_judge_plan(
        "Validate rendered evidence after render preflight passed.",
        render_image_paths=["renders/front.png"],
        render_valid_result={"status": "pass", "verdict": "pass"},
    )

    assert plan.ready_for_judge
    assert _issue_codes(plan) == set()


def test_build_look_right_judge_plan_ignores_unknown_render_preflight_result() -> None:
    plan = build_look_right_judge_plan(
        "Validate a directly supplied image with an unknown preflight payload.",
        current_image_paths=["inputs/photo.png"],
        render_valid_result={"status": "unknown"},
    )

    assert plan.ready_for_judge
    assert _issue_codes(plan) == set()


def test_look_right_plan_blocks_when_vlm_unavailable_and_evidence_missing() -> None:
    plan = build_look_right_judge_plan(
        "Validate visual appearance.",
        vlm_available=False,
    )

    assert not plan.ready_for_judge
    assert _issue_codes(plan) == {VISUAL_EVIDENCE_MISSING, VISUAL_JUDGE_UNAVAILABLE}


def test_build_look_right_judge_plan_ignores_malformed_render_issue_payloads() -> None:
    plan = build_look_right_judge_plan(
        "Validate a directly supplied image with malformed preflight issues.",
        current_image_paths=["inputs/photo.png"],
        render_valid_result={
            "status": "skipped",
            "issues": [
                {"code": "render.blank_image"},
                {"code": None},
                {"message": "missing code"},
                "not-an-issue",
            ],
        },
    )

    assert plan.ready_for_judge
    assert plan.issues[0].details["render_valid_issue_codes"] == ["render.blank_image"]


def test_parse_look_right_judgment_extracts_structured_result() -> None:
    judgment = parse_look_right_judgment(
        """
**Critique:**
The rendered asset mostly matches, but the handle is missing.

**Score:** 6/10

**Decision:** NEEDS_REFINEMENT

**Issue Codes:** visual.prompt_mismatch, visual.reference_mismatch
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.score == 0.6
    assert judgment.issue_codes == (
        VISUAL_PROMPT_MISMATCH,
        "visual.reference_mismatch",
    )
    assert "handle is missing" in judgment.critique


def test_parse_look_right_judgment_keeps_colon_lists_inside_sections() -> None:
    judgment = parse_look_right_judgment(
        """
Critique:
Observed defects:
Front-view: handle is too bright.
Rear-view Score: 3 is only a note, not a heading.
Top-view: lid inserts are missing.

Score: 6
Decision: NEEDS_REFINEMENT
Issue Codes: visual.reference_mismatch
Evidence Notes:
View details:
Corner-view: confirms the same defects.
"""
    )

    assert "Front-view: handle is too bright" in judgment.critique
    assert "Rear-view Score: 3" in judgment.critique
    assert "Top-view: lid inserts are missing" in judgment.critique
    assert "Corner-view: confirms" in judgment.evidence_notes


def test_parse_look_right_judgment_accepts_issue_and_evidence_heading_aliases() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render misses visible detail.
Score: 6
Decision: NEEDS_REFINEMENT
Issue Code: visual.reference_mismatch
Evidence Note: Alias headings should still populate structured evidence.
"""
    )

    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)
    assert "Alias headings" in judgment.evidence_notes


def test_parse_look_right_judgment_stops_sections_at_recommendation_heading() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Recommendation: Keep the current material details.
Score: 9
Decision: PASS
Issue Codes: none
"""
    )

    assert "matches the reference" in judgment.critique
    assert "Recommendation" not in judgment.critique


def test_parse_look_right_judgment_accepts_bold_heading_with_external_colon() -> None:
    judgment = parse_look_right_judgment(
        """
**Critique**: The current render matches the reference.
**Score**: 9
**Decision**: PASS
**Issue Codes**: none
"""
    )

    assert judgment.verdict == "pass"
    assert "matches the reference" in judgment.critique


def test_parse_look_right_judgment_accepts_numbered_section_labels() -> None:
    judgment = parse_look_right_judgment(
        """
1. Critique: The corner view reveals a missing handle.
2. Score: 8/10
3. Decision: PASS
4. Issue Codes:
   1. visual.blocking_defect
5. Evidence Notes: The defect is already visible in the corner view.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.score == 0.8
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)
    assert "missing handle" in judgment.critique
    assert "corner view" in judgment.evidence_notes


def test_parse_look_right_judgment_accepts_markdown_heading_sections() -> None:
    judgment = parse_look_right_judgment(
        """
### Critique: The corner view reveals a missing handle.
### Score: 8/10
### Decision: PASS
### Issue Codes:
   (1) visual.blocking_defect
### Evidence Notes: The defect is already visible in the corner view.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.score == 0.8
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)
    assert "missing handle" in judgment.critique
    assert "corner view" in judgment.evidence_notes


def test_parse_look_right_judgment_accepts_markdown_headers_without_colons() -> None:
    judgment = parse_look_right_judgment(
        """
### Critique
The corner view reveals a missing handle.
### Score
8/10
### Decision
PASS
### Issue Codes
   (1) visual.blocking_defect
### Evidence Notes
The defect is already visible in the corner view.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.score == 0.8
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)
    assert "missing handle" in judgment.critique
    assert "corner view" in judgment.evidence_notes


def test_parse_look_right_judgment_decision_uses_first_choice_token_only() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Score: 9
Decision: PASS
Because it did not fail the visual check.
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"


def test_parse_look_right_judgment_accepts_no_colon_decision() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Score: 9
Decision PASS
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.score == 0.9


def test_parse_look_right_judgment_accepts_chatty_decision_value() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render has a severe visual identity mismatch.
Score: 9
Decision: I conclude we should FAIL.
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_accepts_terminal_chatty_decision_value() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render has a severe visual identity mismatch.
Score: 9
Decision: Due to the visible issues, FAIL.
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_uses_terminal_rejected_choice() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render has a severe visual identity mismatch.
Score: 9
Decision: PASS is inappropriate, so FAIL.
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_uses_terminal_less_harsh_choice() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render is acceptable despite minor differences.
Score: 9
Decision: FAIL is too harsh, so PASS.
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_accepts_pass_despite_harsh_lighting() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Score: 9
Decision: PASS despite harsh lighting.
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_accepts_pass_with_no_issue_reasoning() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Score: 9
Decision: PASS because there are no blocking defects.
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_keeps_leading_choice_with_contrastive_reasoning() -> (
    None
):
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the reference.
Score: 9
Decision: PASS because it did not fail.
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_warns_on_high_score_missing_decision() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render mostly matches the reference.
Score: 9
Issue Codes: none
"""
    )

    assert judgment.verdict == "warn"
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_warns_on_option_list_decision_echo() -> None:
    for decision_line in (
        "Decision: [PASS, FAIL, NEEDS_REFINEMENT, or WARN]",
        "Decision: PASS, FAIL, NEEDS_REFINEMENT, or WARN",
    ):
        judgment = parse_look_right_judgment(
            f"""
Critique: The current render mostly matches the reference.
Score: 9
{decision_line}
Issue Codes: none
"""
        )

        assert judgment.verdict == "warn"
        assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_downgrades_low_score() -> None:
    judgment = parse_look_right_judgment(
        """
**Critique:** The asset identity is unclear.
**Score:** 4
**Decision:** PASS
**Issue Codes:** none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.4
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_fails_bare_one_score() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset identity is unclear.
Score: 1
Decision: PASS
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.1
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_fails_bare_one_decimal_score() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset identity is unclear.
Score: 1.0
Decision: PASS
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.1
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_fails_bare_fractional_ten_point_score() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset identity is unclear.
Score: 0.5
Decision: PASS
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.05
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_fails_low_score_needs_refinement() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset keeps the rough shape but misses required visual identity.
Score: 4
Decision: NEEDS_REFINEMENT
Issue Codes: visual.reference_mismatch
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.4
    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)


def test_parse_look_right_judgment_fails_low_score_warn() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset keeps the rough shape but misses required visual identity.
Score: 4
Decision: WARN
Issue Codes: none
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.4
    assert judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_parse_look_right_judgment_blocks_pass_with_mismatch_issue() -> None:
    judgment = parse_look_right_judgment(
        """
**Critique:** The corner view shows the body color mismatches the reference.

**Score:** 9

**Decision:** PASS

**Issue Codes:** visual.reference_mismatch

**Evidence Notes:** The corner view supports the defect finding.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.score == 0.9
    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)
    assert "corner view" in judgment.evidence_notes


def test_parse_look_right_judgment_blocks_pass_when_issue_code_blocks() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The response contradicts itself by listing a mismatch issue.
Score: 9
Decision: PASS
Issue Codes: visual.reference_mismatch
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)


def test_parse_look_right_judgment_blocks_pass_with_generic_blocking_issue() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
Issue Codes: visual.blocking_defect
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_blocks_chatty_issue_code_mentions() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
Issue Codes: I found visual.blocking_defect.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_prioritizes_blocking_over_low_confidence() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
Issue Codes: visual.low_confidence, visual.blocking_defect
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE, VISUAL_BLOCKING_DEFECT)


def test_parse_look_right_judgment_blocks_issue_codes_after_clause_negation() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
Issue Codes: No geometry issue, but visual.blocking_defect.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_keeps_contrastive_trailing_negation() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
Issue Codes: visual.blocking_defect, not visual.low_confidence.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_keeps_negated_defect_description() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The wheels do not match the reference.
Score: 9
Decision: PASS
Issue Codes: The wheels do not match the reference, indicating visual.reference_mismatch.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)


def test_parse_look_right_judgment_blocks_code_first_unresolved_issue() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The required attribute is still wrong.
Score: 8
Decision: PASS
Issue Codes: visual.reference_mismatch is not resolved.
Evidence Notes: front view shows the mismatch.
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_REFERENCE_MISMATCH,)


def test_parse_look_right_judgment_ignores_code_first_negated_issue() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render mostly matches the reference.
Score: 9
Decision: PASS
Issue Codes:
- visual.blocking_defect is not visible in this view.
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_ignores_common_negated_issue_phrases() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render mostly matches the reference.
Score: 9
Decision: PASS
Issue Codes: I don't see any visual.blocking_defect.
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


@pytest.mark.parametrize(
    "issue_codes_section",
    [
        "Issue Codes: `visual.blocking_defect`",
        "Issue Codes: [visual.blocking_defect]",
        "Issue Codes:\n- `visual.blocking_defect`",
        "Issue Codes:\n1. visual.blocking_defect",
        "Issue Codes:\nThe issue is visual.blocking_defect",
        "Issue Codes: visual.blocking_defect.",
    ],
)
def test_parse_look_right_judgment_blocks_markdown_issue_codes(
    issue_codes_section: str,
) -> None:
    judgment = parse_look_right_judgment(
        f"""
Critique: The visible mismatch remains unresolved.
Score: 9
Decision: PASS
{issue_codes_section}
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_fails_low_score_blocking_issue() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The generated asset identity is clearly wrong.
Score: 4
Decision: PASS
Issue Codes: visual.blocking_defect
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.score == 0.4
    assert judgment.issue_codes == (VISUAL_BLOCKING_DEFECT,)


def test_parse_look_right_judgment_filters_unknown_issue_codes() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The issue section includes trailing prose from an unknown heading.
Score: 8
Decision: PASS
Issue Codes: none
Recommendation: See visual.reference_mismatch and physics.behavior for background.
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.issue_codes == ()


def test_parse_look_right_judgment_stops_codes_before_unknown_heading() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The issue section has one real code and trailing prose.
Score: 8
Decision: PASS
Issue Codes:
- visual.low_confidence
Recommendation: Do not also capture visual.reference_mismatch from this prose.
"""
    )

    assert judgment.verdict == "warn"
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_stops_codes_before_generic_markdown_heading() -> (
    None
):
    judgment = parse_look_right_judgment(
        """
Critique: The issue section has one real code and trailing prose.
Score: 8
Decision: PASS
Issue Codes:
- visual.low_confidence
**Summary:**
Mention visual.reference_mismatch only as unrelated explanatory prose.
"""
    )

    assert judgment.verdict == "warn"
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_keeps_view_markdown_headers_in_critique() -> None:
    judgment = parse_look_right_judgment(
        """
Critique:
### Front view:
The handle is too bright.
### Right view:
The side panel remains aligned.
Score: 8
Decision: NEEDS_REFINEMENT
Issue Codes: visual.low_confidence
"""
    )

    assert "Front view" in judgment.critique
    assert "Right view" in judgment.critique
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_keeps_field_words_inside_critique() -> None:
    judgment = parse_look_right_judgment(
        """
Critique:
Score is generally high, but one handle still needs work.
Decision requires checking all visible sides before approval.
Score: 8
Decision: NEEDS_REFINEMENT
Issue Codes: visual.low_confidence
"""
    )

    assert "Score is generally high" in judgment.critique
    assert "Decision requires" in judgment.critique
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_stops_critique_at_whitespace_decision_boundary() -> (
    None
):
    judgment = parse_look_right_judgment(
        """
Critique:
The current render matches the reference.
**Decision** PASS
Score: 9
Issue Codes: none
"""
    )

    assert judgment.critique == "The current render matches the reference."
    assert judgment.verdict == "pass"


def test_parse_look_right_judgment_keeps_generic_headings_inside_critique() -> None:
    judgment = parse_look_right_judgment(
        """
Critique:
**Summary:**
The object matches the requested form.
**Analysis:**
All visible panels are aligned.
Score: 8
Decision: NEEDS_REFINEMENT
Issue Codes: visual.low_confidence
"""
    )

    assert "Summary" in judgment.critique
    assert "Analysis" in judgment.critique
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_warns_on_pass_with_low_confidence_issue() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset mostly appears correct, but only one blurry view is usable.
Score: 9
Decision: PASS
Issue Codes: visual.low_confidence
"""
    )

    assert judgment.verdict == "warn"
    assert judgment.score == 0.9
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_warns_on_missing_score() -> None:
    judgment = parse_look_right_judgment(
        """
**Critique:** The judge response did not include the required score.
**Decision:** PASS
**Issue Codes:** none
"""
    )

    assert judgment.verdict == "warn"
    assert judgment.score is None
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_accepts_plain_section_headers() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the requested asset.
Score: 8
Decision: PASS
Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.score == 0.8
    assert judgment.critique == "The current render matches the requested asset."


def test_parse_look_right_judgment_stops_critique_at_bulleted_headers() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render matches the requested asset.
- Score: 8
- Decision: PASS
- Issue Codes: none
"""
    )

    assert judgment.verdict == "pass"
    assert judgment.score == 0.8
    assert judgment.critique == "The current render matches the requested asset."


def test_parse_look_right_judgment_accepts_multiline_issue_codes() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The current render misses required visual details.
Score: 5
Decision: FAIL
Issue Codes:
- visual.prompt_mismatch
- visual.low_confidence

Evidence Notes:
The front render lacks the requested handle detail.
"""
    )

    assert judgment.verdict == "fail"
    assert judgment.issue_codes == (
        VISUAL_PROMPT_MISMATCH,
        VISUAL_LOW_CONFIDENCE,
    )


def test_parse_look_right_judgment_adds_low_confidence_for_uncoded_refinement() -> None:
    judgment = parse_look_right_judgment(
        """
Critique: The asset is close, but one material detail should be refined.
Score: 6
Decision: NEEDS_REFINEMENT
Issue Codes: none
"""
    )

    assert judgment.verdict == "needs_refinement"
    assert judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_parse_look_right_judgment_truncates_long_reasoning_summary() -> None:
    long_line = "Critique: " + ("The visible evidence is acceptable. " * 20)

    judgment = parse_look_right_judgment(
        f"""
{long_line}
Score: 9
Decision: PASS
Issue Codes: none
"""
    )

    assert len(judgment.reasoning) == 240
    assert judgment.reasoning.endswith("...")


def test_normalize_look_right_judgment_uses_generate_llm_and_string_codes() -> None:
    llm = _GeneratingLLM(
        """
{"decision": "continue", "score": 8,
 "issue_codes": "visual.low_confidence, unknown.code",
 "rationale": "The response asks for another refinement pass."}
"""
    )

    result = normalize_look_right_judgment(
        """
Critique: The render is close but needs one material adjustment.
Score: 8
Decision: NEEDS_REFINEMENT
Issue Codes: none
""",
        llm_judge=llm,
        temperature=None,
        max_tokens=None,
    )

    assert result.judgment.verdict == "needs_refinement"
    assert result.judgment.score == 0.8
    assert result.judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)
    assert result.backend_name == "fake-generate"
    assert result.model_name == "fake-generate-llm"
    assert llm.calls[0]["system_prompt"]
    assert "temperature" not in llm.calls[0]
    assert "max_tokens" not in llm.calls[0]


def test_normalize_look_right_judgment_adds_default_fail_code_from_final_judge() -> (
    None
):
    result = normalize_look_right_judgment(
        """
Critique: The asset is clearly wrong.
Score: 2
Decision: FAIL
Issue Codes: none
""",
        llm_judge=_RecordingLLM(
            '{"decision": "fail", "score": null, "issue_codes": null, '
            '"rationale": "The final judge confirms failure."}'
        ),
    )

    assert result.judgment.verdict == "fail"
    assert result.judgment.score == 0.2
    assert result.judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_normalize_look_right_judgment_adds_fail_code_when_parser_had_none() -> None:
    result = normalize_look_right_judgment(
        """
Critique: The parser fallback sees a clean pass.
Score: 9
Decision: PASS
Issue Codes: none
""",
        llm_judge=_RecordingLLM(
            '{"decision": "fail", "score": 0.9, "issue_codes": null, '
            '"rationale": "The final judge found an identity mismatch."}'
        ),
    )

    assert result.judgment.verdict == "fail"
    assert result.judgment.issue_codes == (VISUAL_PROMPT_MISMATCH,)


def test_normalize_look_right_judgment_adds_default_warn_code_from_final_judge() -> (
    None
):
    result = normalize_look_right_judgment(
        """
Critique: The evidence is not sufficient to decide.
Score: 8
Decision: WARN
Issue Codes: none
""",
        llm_judge=_RecordingLLM(
            '{"decision": "warn", "score": 0.8, "issue_codes": null, '
            '"rationale": "The evidence is limited."}'
        ),
    )

    assert result.judgment.verdict == "warn"
    assert result.judgment.issue_codes == (VISUAL_LOW_CONFIDENCE,)


def test_look_right_metadata_helpers_ignore_unsupported_values() -> None:
    class MappingUsage:
        def to_dict(self) -> dict[str, int]:
            return {"total_tokens": 7}

    class NonMappingUsage:
        def to_dict(self) -> list[str]:
            return ["not", "a", "mapping"]

    class ExplodingAttribute:
        @property
        def backend_name(self) -> str:
            raise RuntimeError("attribute unavailable")

    class NullAttribute:
        backend_name = None

    assert _token_usage_dict(MappingUsage()) == {"total_tokens": 7}
    assert _token_usage_dict(NonMappingUsage()) is None
    assert _token_usage_dict({"input_tokens": 3}) == {"input_tokens": 3}
    assert _token_usage_dict(42) is None
    assert _safe_optional_attr(ExplodingAttribute(), "backend_name") is None
    assert _safe_optional_attr(NullAttribute(), "backend_name") is None
    assert _safe_optional_attr(object(), "missing_name") is None
