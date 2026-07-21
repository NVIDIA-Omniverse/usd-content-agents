# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LLM parsing utilities."""

import logging
import math

import pytest

from world_understanding.utils import llm_parsing as lp
from world_understanding.utils.llm_parsing import (
    create_json_prompt_instructions,
    extract_json_from_llm_response,
    extract_labeled_choice,
    extract_labeled_codes,
    extract_labeled_score,
    extract_labeled_value,
    extract_material_from_json,
)


class TestExtractLabeledJudgeFields:
    """Test lightweight labeled-field parsing shared by VLM judges."""

    def test_extract_labeled_value_empty_response(self):
        assert extract_labeled_value("", "Decision") == ""

    def test_extract_labeled_score_normalizes_ten_point_scores(self):
        assert extract_labeled_score("**Score:** 8/10") == 0.8
        assert extract_labeled_score("Score: 1/10") == 0.1
        assert extract_labeled_score("Score: 1") == 0.1
        assert extract_labeled_score("Score: 1.0") == 0.1
        assert extract_labeled_score("Score: 0.75") == 0.075
        assert extract_labeled_score("Score 7") == 0.7

    def test_extract_labeled_score_can_parse_normalized_score_domain(self):
        assert extract_labeled_score("Score: 0.75", score_max=1.0) == 0.75

    def test_extract_labeled_score_handles_newline_after_label(self):
        response = """**Score:**
8/10
Decision: PASS
"""

        assert extract_labeled_score(response) == 0.8

    def test_extract_labeled_score_skips_leading_blank_multiline_value(self):
        response = """**Score:**

8/10
Decision: PASS
"""

        assert extract_labeled_score(response) == 0.8

    def test_extract_labeled_score_prefers_fraction_over_prose_numbers(self):
        response = """Score:
I found 3 visible defects.
The final score is 8/10.
Decision: NEEDS_REFINEMENT
"""

        assert extract_labeled_score(response) == 0.8

    def test_extract_labeled_score_prefers_leading_score_over_later_fraction(self):
        response = """Score:
8/10. Improvement potential is 3/10.
Decision: PASS
"""

        assert extract_labeled_score(response) == 0.8

    def test_extract_labeled_score_handles_out_of_format(self):
        assert extract_labeled_score("Score: 8 out of 10") == 0.8

    def test_extract_labeled_score_handles_fallback_score_forms(self):
        assert extract_labeled_score("Overall Score 8/10") == 0.8
        assert extract_labeled_score("Score: the rating is 8") == 0.8
        assert extract_labeled_score("Score: the result is 8 out of 10") == 0.8
        assert extract_labeled_score("Score: final value 8/10") == 0.8

    def test_score_normalization_edges(self):
        assert lp._normalize_score(math.inf, None, score_max=10.0) is None
        assert lp._normalize_score(1.0, 0.0, score_max=10.0) is None
        assert lp._normalize_score(8.0, None, score_max=10.0) == 0.8
        assert lp._normalize_score(1e308, 1e-308, score_max=10.0) is None

    def test_extract_labeled_score_ignores_unrelated_numbers(self):
        response = """Score:
I found 3 visible defects.
Decision: NEEDS_REFINEMENT
"""

        assert extract_labeled_score(response) is None

    def test_extract_labeled_score_ignores_compound_label_names(self):
        assert extract_labeled_score("SomeScore: 5") is None

    def test_extract_labeled_fields_accept_numbered_section_labels(self):
        response = """1. Critique: The visible corner view matches.
2. Score: 8/10
3. Decision: PASS
4. Issue Codes:
   1. visual.low_confidence
5. Evidence Notes: One usable view.
"""

        assert extract_labeled_score(response) == 0.8
        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence"},
        ) == ("visual.low_confidence",)

    def test_extract_labeled_fields_accept_markdown_and_parenthesized_labels(self):
        response = """### Score: 8/10
### Decision: PASS
### Issue Codes:
   (1) visual.low_confidence
"""

        assert extract_labeled_score(response) == 0.8
        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence"},
        ) == ("visual.low_confidence",)

    def test_extract_labeled_fields_accept_unpunctuated_markdown_headers(self):
        response = """### Score
8/10
### Decision
FAIL
### Issue Codes
- visual.blocking_defect
"""

        assert extract_labeled_score(response) == 0.8
        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "fail"
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    def test_extract_labeled_value_handles_markdown_label(self):
        response = "**Decision:** CONTINUE\nCritique: Needs work"

        assert extract_labeled_value(response, "Decision") == "CONTINUE"

    def test_extract_labeled_choice_accepts_bold_label_without_colon(self):
        response = """**Decision**
PASS
"""

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"

    def test_extract_labeled_choice_uses_first_choice_token_only(self):
        response = """Decision: PASS
Because the asset did not fail the visual check.
"""

        assert (
            extract_labeled_choice(
                response,
                "Decision",
                ("fail", "pass"),
            )
            == "pass"
        )

    def test_extract_labeled_choice_accepts_led_prose(self):
        response = "Decision: I conclude we should FAIL."

        assert (
            extract_labeled_choice(
                response,
                "Decision",
                ("fail", "pass"),
            )
            == "fail"
        )

    def test_extract_labeled_choice_accepts_terminal_explicit_choice(self):
        response = "Decision: Due to the visible issues, FAIL."

        assert (
            extract_labeled_choice(
                response,
                "Decision",
                ("fail", "pass"),
            )
            == "fail"
        )

    def test_extract_labeled_choice_skips_rejected_leading_choice(self):
        response = "Decision: PASS is inappropriate, so FAIL."

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "fail"

    def test_extract_labeled_choice_skips_leading_choice_marked_not_valid(self):
        response = "Decision: PASS is not valid, so FAIL."

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == ""
        assert lp._tokens_reject_leading_choice(
            ("pass", "is", "not", "valid"),
            ("pass",),
        )

    def test_extract_labeled_choice_skips_too_harsh_leading_choice(self):
        response = "Decision: FAIL is too harsh, so PASS."

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"

    def test_extract_labeled_choice_keeps_harsh_condition_reasoning(self):
        response = "Decision: PASS despite harsh lighting."

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"

    def test_extract_labeled_choice_accepts_affirmative_no_issue_reasoning(self):
        response = "Decision: PASS because there are no blocking defects."

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"

    def test_extract_labeled_choice_keeps_leading_choice_with_contrastive_reasoning(
        self,
    ):
        assert (
            extract_labeled_choice(
                "Decision: PASS because it did not fail.",
                "Decision",
                ("fail", "pass"),
            )
            == "pass"
        )
        assert (
            extract_labeled_choice(
                "Decision: FAIL (normally would PASS).",
                "Decision",
                ("fail", "pass"),
            )
            == "fail"
        )

    def test_extract_labeled_choice_accepts_choice_after_multiline_preamble(self):
        response = """Decision:
Given the high score,
PASS
"""

        assert extract_labeled_choice(response, "Decision", ("fail", "pass")) == "pass"

    def test_extract_labeled_choice_rejects_option_list_echo(self):
        for response in (
            "Decision: [PASS, FAIL, NEEDS_REFINEMENT, or WARN]",
            "Decision: PASS, FAIL, NEEDS_REFINEMENT, or WARN",
            "Decision: PASS if score >= 7, NEEDS_REFINEMENT if fixes are needed",
        ):
            assert (
                extract_labeled_choice(
                    response,
                    "Decision",
                    ("fail", "needs_refinement", "warn", "pass"),
                )
                == ""
            )

    def test_extract_labeled_choice_ignores_alias_to_unknown_choice(self):
        assert (
            extract_labeled_choice(
                "Decision: ok",
                "Decision",
                ("pass",),
                aliases={"ok": "unknown"},
            )
            == ""
        )
        assert (
            extract_labeled_choice(
                "Decision: I recommend ok",
                "Decision",
                ("pass",),
                aliases={"ok": "unknown"},
            )
            == ""
        )

    def test_choice_helpers_handle_empty_tokens_and_blank_value(self):
        assert lp._choice_tokens("") == ()
        assert lp._parse_choice_value(" \n ", ("pass", "fail")) == ""
        assert lp._tokens_contain_sequence(("pass",), ("needs", "refinement")) is False

    def test_extract_labeled_choice_accepts_whitespace_delimited_label(self):
        response = "Decision PASS"

        assert (
            extract_labeled_choice(
                response,
                "Decision",
                ("fail", "pass"),
            )
            == "pass"
        )

    @pytest.mark.parametrize(
        "response",
        [
            "Decision: PASSIVE",
            "Decision: The asset should not fail.",
            "Decision: The asset should not be classified as FAIL.",
        ],
    )
    def test_extract_labeled_choice_avoids_prefix_and_negated_matches(
        self, response: str
    ):
        assert (
            extract_labeled_choice(
                response,
                "Decision",
                ("fail", "pass"),
            )
            == ""
        )

    def test_extract_labeled_value_skips_leading_blank_multiline_value(self):
        response = """Issue Codes:

- visual.low_confidence
Decision: PASS
"""

        assert (
            extract_labeled_value(response, "Issue Codes", multiline=True)
            == "- visual.low_confidence"
        )

    def test_extract_labeled_value_keeps_colon_prose_inside_value(self):
        response = """Evidence Notes:
Top-view: confirms the same missing handle.
Corner-view: confirms the same defect.
Decision: NEEDS_REFINEMENT
"""

        assert (
            extract_labeled_value(response, "Evidence Notes", multiline=True)
            == "Top-view: confirms the same missing handle.\n"
            "Corner-view: confirms the same defect."
        )

    def test_extract_labeled_value_keeps_field_words_inside_multiline_prose(self):
        response = """Critique:
Score is generally fine, but the handle still needs work.
Decision requires checking all visible sides before approval.
Issue Codes: visual.low_confidence
"""

        assert (
            extract_labeled_value(response, "Critique", multiline=True)
            == "Score is generally fine, but the handle still needs work.\n"
            "Decision requires checking all visible sides before approval."
        )

    def test_extract_labeled_value_stops_at_whitespace_decision_boundary(self):
        response = """Critique:
The asset matches the reference.
**Decision** PASS
Score: 9
"""

        assert (
            extract_labeled_value(response, "Critique", multiline=True)
            == "The asset matches the reference."
        )

    def test_extract_labeled_value_keeps_generic_headings_inside_free_text(self):
        response = """Critique:
**Summary:**
The object matches the requested form.
**Analysis:**
All visible panels are aligned.
Score: 8
"""

        assert (
            extract_labeled_value(response, "Critique", multiline=True)
            == "**Summary:**\nThe object matches the requested form.\n"
            "**Analysis:**\nAll visible panels are aligned."
        )

    def test_extract_labeled_value_can_stop_at_markdown_generic_heading(self):
        response = """Critique:
The object matches the requested form.
### Summary
This should not be captured.
Score: 8
"""

        assert (
            extract_labeled_value(
                response,
                "Critique",
                multiline=True,
                stop_at_generic_headings=True,
            )
            == "The object matches the requested form."
        )
        assert lp._strip_bold_heading("**Summary") == "**Summary"

    def test_extract_labeled_codes_stops_at_next_heading(self):
        response = """Issue Codes:
- visual.low_confidence
Recommendation: Do not capture visual.reference_mismatch here.
"""

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.reference_mismatch"},
        ) == ("visual.low_confidence",)

    def test_extract_labeled_codes_stops_at_suggestion_heading(self):
        response = """Issue Codes:
- visual.low_confidence
Improvement Suggestions: Do not capture visual.reference_mismatch here.
"""

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.reference_mismatch"},
        ) == ("visual.low_confidence",)

    @pytest.mark.parametrize(
        "response",
        [
            "Issue Codes: `visual.blocking_defect`",
            "Issue Codes: [visual.blocking_defect]",
            "Issue Codes:\n- `visual.blocking_defect`",
            "Issue Codes:\n1. visual.blocking_defect",
            "Issue Codes:\nThe issue is visual.blocking_defect",
        ],
    )
    def test_extract_labeled_codes_accepts_explicit_code_formatting(
        self, response: str
    ):
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    @pytest.mark.parametrize(
        "response",
        [
            "Issue Codes: visual.blocking_defect.",
            "Issue Codes: visual.blocking_defect,",
            "Issue Codes: visual.blocking_defect;",
            "Issue Codes: `visual.blocking_defect`.",
        ],
    )
    def test_extract_labeled_codes_ignores_terminal_punctuation(self, response: str):
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    def test_extract_labeled_codes_keeps_blank_line_separated_items(self):
        response = """Issue Codes:
- visual.prompt_mismatch

- visual.reference_mismatch
Decision: NEEDS_REFINEMENT
"""

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.prompt_mismatch", "visual.reference_mismatch"},
        ) == ("visual.prompt_mismatch", "visual.reference_mismatch")

    def test_extract_labeled_codes_ignores_trailing_explanatory_mentions(self):
        response = """Issue Codes:
- visual.low_confidence
Note: this is not a visual.reference_mismatch.
Decision: WARN
"""

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.reference_mismatch"},
        ) == ("visual.low_confidence",)

    def test_extract_labeled_codes_stops_at_generic_markdown_heading(self):
        response = """Issue Codes:
- visual.low_confidence
**Summary:**
The summary mentions visual.reference_mismatch as a possible future category.
Decision: WARN
"""

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.reference_mismatch"},
        ) == ("visual.low_confidence",)

    def test_extract_labeled_codes_accepts_chatty_positive_mentions(self):
        response = "Issue Codes: I found visual.blocking_defect."

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    def test_extract_labeled_codes_scopes_negation_to_local_clause(self):
        response = "Issue Codes: not visual.low_confidence, but visual.blocking_defect."

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    def test_extract_labeled_codes_keeps_contrastive_trailing_negation(self):
        response = "Issue Codes: visual.blocking_defect, not visual.low_confidence."

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.low_confidence", "visual.blocking_defect"},
        ) == ("visual.blocking_defect",)

    def test_extract_labeled_codes_keeps_negated_defect_description(self):
        response = (
            "Issue Codes: The wheels do not match the reference, "
            "indicating visual.reference_mismatch."
        )

        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.reference_mismatch"},
        ) == ("visual.reference_mismatch",)

    def test_extract_labeled_codes_respects_code_first_negation(self):
        response = "Issue Codes:\n- visual.blocking_defect is not visible."

        assert (
            extract_labeled_codes(
                response,
                "Issue Codes",
                allowed_codes={"visual.blocking_defect"},
            )
            == ()
        )

    @pytest.mark.parametrize(
        "response",
        [
            "Issue Codes: visual.blocking_defect is not present.",
            "Issue Codes: visual.blocking_defect is no longer visible.",
            "Issue Codes: visual.blocking_defect cannot be seen.",
            "Issue Codes: visual.blocking_defect is not an issue.",
        ],
    )
    def test_extract_labeled_codes_ignores_code_first_absence_phrases(
        self, response: str
    ):
        assert (
            extract_labeled_codes(
                response,
                "Issue Codes",
                allowed_codes={"visual.blocking_defect"},
            )
            == ()
        )

    @pytest.mark.parametrize(
        "response",
        [
            "Issue Codes: visual.reference_mismatch is not resolved.",
            "Issue Codes: visual.reference_mismatch is not fixed.",
            "Issue Codes: visual.reference_mismatch should not be ignored.",
        ],
    )
    def test_extract_labeled_codes_preserves_code_first_unresolved_explanations(
        self, response: str
    ):
        assert extract_labeled_codes(
            response,
            "Issue Codes",
            allowed_codes={"visual.reference_mismatch"},
        ) == ("visual.reference_mismatch",)

    @pytest.mark.parametrize(
        "response",
        [
            "Issue Codes: I don't see any visual.blocking_defect.",
            "Issue Codes: It cannot be considered a visual.reference_mismatch.",
            "Issue Codes: The result can't be called visual.blocking_defect.",
        ],
    )
    def test_extract_labeled_codes_handles_common_negation_contractions(
        self,
        response: str,
    ):
        assert (
            extract_labeled_codes(
                response,
                "Issue Codes",
                allowed_codes={
                    "visual.blocking_defect",
                    "visual.reference_mismatch",
                },
            )
            == ()
        )

    def test_extract_labeled_codes_skips_lines_without_codes(self):
        assert extract_labeled_codes("Issue Codes:\n- none here", "Issue Codes") == ()
        assert extract_labeled_codes("Issue Codes: no issues", "Issue Codes") == ()


class TestExtractJsonFromLLMResponse:
    """Test cases for extract_json_from_llm_response function."""

    def test_extract_json_from_markdown_code_block(self):
        """Test extracting JSON from markdown code blocks."""
        response = """Here is the JSON response:
```json
{
    "name": "test",
    "value": 123,
    "active": true
}
```
Some additional text here."""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["active"] is True

    def test_extract_json_from_plain_code_block(self):
        """Test extracting JSON from code blocks without json specifier."""
        response = """Here is the response:
```
{
    "tools": ["tool1", "tool2"],
    "reasoning": "This is why"
}
```
"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["tools"] == ["tool1", "tool2"]
        assert result["reasoning"] == "This is why"

    def test_extract_json_skips_non_json_fence_before_later_object(self):
        """A prose fence should not prevent scanning the rest of the response."""
        response = """<answer>
```
This is a summary, not JSON.
```
</answer>

Final remap:
{"material": "steel", "confidence": "high"}
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_prefers_answer_block_over_outer_json(self):
        """Tagged answer content should win over earlier outer reasoning JSON."""
        response = """Before the answer: {"material": "plastic"}
<answer>
```json
{"material": "steel", "confidence": "high"}
```
</answer>
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_prefers_last_answer_block(self):
        """Repeated answer tags should use the final answer block."""
        response = """<answer>
```json
{"material": "prompt example"}
```
</answer>
<answer>
```json
{"material": "steel", "confidence": "high"}
```
</answer>
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_prefers_last_answer_block_with_attrs_and_case(self):
        """Answer tag parsing should match classification answer extraction."""
        response = """<answer>
```json
{"material": "prompt example"}
```
</answer>
<ANSWER reason="final">
```json
{"material": "steel", "confidence": "high"}
```
</ANSWER>
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_ignores_answering_pseudo_tag(self):
        """Malformed answer-like tags should not get answer-block priority."""
        response = """Before: {"material": "outer"}
<answering>
{"material": "pseudo tag"}
</answer>
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "outer"}

    def test_extract_json_suppresses_missing_key_debug_when_log_failures_false(
        self,
        caplog,
    ):
        """Quiet mode should suppress skipped-candidate parser logging."""
        response = '{"reasoning": "draft"}'

        with caplog.at_level(
            logging.DEBUG,
            logger="world_understanding.utils.llm_parsing",
        ):
            result = extract_json_from_llm_response(
                response,
                expected_keys=["material"],
                log_failures=False,
            )

        assert result is None
        assert "Skipping JSON object missing expected keys" not in caplog.text

    def test_extract_json_scans_later_code_fences(self):
        """A non-JSON fence should not hide a later JSON fence."""
        response = """<answer>
```
This is a summary, not JSON.
```
```json
{"material": "steel", "confidence": "high"}
```
</answer>
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_handles_unclosed_code_fence(self):
        response = """```json
{"material": "steel"}
"""

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel"}

    def test_parse_json_dict_candidate_rejects_non_dict(self, caplog):
        with caplog.at_level(
            logging.ERROR, logger="world_understanding.utils.llm_parsing"
        ):
            assert lp._parse_json_dict_candidate("[1, 2, 3]") is None

        assert "Expected dict" in caplog.text

    def test_iter_json_dicts_empty_response(self):
        assert list(lp.iter_json_dicts_from_llm_response("")) == []

    def test_extract_json_skips_invalid_brace_prose_before_later_json(self):
        """Invalid prose braces should not hide the later valid JSON object."""
        response = (
            'The rough format is {key: value}. Final: {"key": "value", "ok": true}'
        )

        result = extract_json_from_llm_response(response)
        assert result == {"key": "value", "ok": True}

    def test_extract_json_recovers_after_many_unmatched_braces(self):
        """Candidate scanning should not repeatedly walk the full suffix."""
        response = ("{" * 5000) + 'Final: {"key": "value", "ok": true}'

        result = extract_json_from_llm_response(response)
        assert result == {"key": "value", "ok": True}

    def test_extract_json_preserves_double_braces_inside_string_values(self):
        """Valid JSON template strings should not be mutated while parsing."""
        response = '{"pattern": "{{ variable }}", "ok": true}'

        result = extract_json_from_llm_response(response)
        assert result == {"pattern": "{{ variable }}", "ok": True}

    def test_extract_json_still_accepts_vlm_double_brace_object_wrapping(self):
        """Some VLMs wrap the entire object in doubled braces."""
        response = '{{"material": "steel", "confidence": "high"}}'

        result = extract_json_from_llm_response(response)
        assert result == {"material": "steel", "confidence": "high"}

    def test_extract_json_from_text_with_surrounding_content(self):
        """Test extracting JSON from text with surrounding content."""
        response = """Let me generate the JSON for you.
{"key": "value", "number": 42, "array": [1, 2, 3]}
That's the JSON you requested."""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["array"] == [1, 2, 3]

    def test_extract_plain_json(self):
        """Test extracting plain JSON without any surrounding text."""
        response = '{"status": "success", "count": 5}'

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["status"] == "success"
        assert result["count"] == 5

    def test_extract_nested_json(self):
        """Test extracting nested JSON structures."""
        response = """```json
{
    "user": {
        "name": "John",
        "age": 30,
        "addresses": [
            {"type": "home", "city": "NYC"},
            {"type": "work", "city": "SF"}
        ]
    },
    "active": true
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30
        assert len(result["user"]["addresses"]) == 2
        assert result["user"]["addresses"][0]["city"] == "NYC"

    def test_extract_json_with_comments_fails(self):
        """Test that JSON with comments (invalid JSON) returns None."""
        response = """```json
{
    "name": "test", // This is a comment
    "value": 123
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is None  # Should fail due to comments

    def test_extract_json_with_inline_comments_from_llm(self):
        """Test the exact format seen in the terminal output."""
        response = """Here is the JSON object containing the tool inputs:
```
{
  "image": {
    "path": null, // in-memory image
    "width": 640, // default width for a typical image
    "height": 480, // default height for a typical image
    "mime": "image/jpeg" // common MIME type for JPEG images
  },
  "target_color": [255, 105, 180], // pink color in RGB (0-255)
  "color_tolerance": 20, // relatively low tolerance to match exact pink color
  "min_percentage": 5.0 // require at least 5% of pixels to match the target color
}
```
Note that I've chosen sensible defaults for the image dimensions and MIME type."""

        result = extract_json_from_llm_response(response)
        assert result is None  # Should fail due to inline comments

    def test_empty_response(self):
        """Test handling of empty response."""
        result = extract_json_from_llm_response("")
        assert result is None

        result = extract_json_from_llm_response(None)
        assert result is None

    def test_no_json_in_response(self):
        """Test handling when no JSON is found."""
        response = "This is just plain text without any JSON content."

        result = extract_json_from_llm_response(response)
        assert result is None

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = """```json
{
    "name": "test"
    "missing": "comma"
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is None

    def test_expected_keys_validation(self):
        """Test validation of expected keys."""
        response = '{"name": "test", "value": 123}'

        # All expected keys present
        result = extract_json_from_llm_response(
            response, expected_keys=["name", "value"]
        )
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123

        # Missing expected key should reject this candidate.
        result = extract_json_from_llm_response(
            response, expected_keys=["name", "value", "missing"]
        )
        assert result is None

    def test_expected_keys_skip_reasoning_json_before_answer(self):
        """Schema hints should select the matching object, not the first dict."""
        response = """{"thought_process": "Need to inspect the scene first."}
Final answer:
{"name": "test", "value": 123}
"""

        result = extract_json_from_llm_response(
            response, expected_keys=["name", "value"]
        )

        assert result == {"name": "test", "value": 123}

    def test_multiple_json_objects(self):
        """Test extracting when multiple JSON objects are present."""
        response = """First object: {"a": 1}
Second object: {"b": 2}"""

        # Should extract the first complete JSON object
        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result == {"a": 1}

    def test_json_with_special_characters(self):
        """Test JSON with special characters and escape sequences."""
        response = """```json
{
    "message": "Hello\\nWorld",
    "path": "C:\\\\Users\\\\test",
    "unicode": "\\u00A9 2024"
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["message"] == "Hello\nWorld"
        assert result["path"] == "C:\\Users\\test"
        assert result["unicode"] == "© 2024"

    def test_outer_double_brace_unwrap_preserves_string_braces(self):
        """VLM wrapper cleanup must not mutate braces inside string values."""
        response = '{{"key": "value}}", "msg": "{{hello}}"}}'

        result = extract_json_from_llm_response(response)

        assert result == {"key": "value}}", "msg": "{{hello}}"}

    def test_json_with_different_types(self):
        """Test JSON with various data types."""
        response = """```json
{
    "string": "text",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, "two", 3.0, false, null],
    "empty_object": {},
    "empty_array": []
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["string"] == "text"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["null_value"] is None
        assert result["array"] == [1, "two", 3.0, False, None]
        assert result["empty_object"] == {}
        assert result["empty_array"] == []


class TestCreateJsonPromptInstructions:
    """Test cases for create_json_prompt_instructions function."""

    def test_returns_string(self):
        """Test that the function returns a string."""
        result = create_json_prompt_instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_key_instructions(self):
        """Test that the instructions contain key phrases."""
        result = create_json_prompt_instructions()
        assert "JSON" in result
        assert "ONLY" in result
        assert "valid" in result

    def test_consistent_output(self):
        """Test that the function returns consistent output."""
        result1 = create_json_prompt_instructions()
        result2 = create_json_prompt_instructions()
        assert result1 == result2


def test_extract_material_from_single_key_string_fallback() -> None:
    class ContainsOnly:
        def __iter__(self):
            return iter(())

        def __contains__(self, item: object) -> bool:
            return item == "material_name"

    assert (
        extract_material_from_json(
            {"material_name": "Brushed Steel"},
            ContainsOnly(),  # type: ignore[arg-type]
        )
        == "Brushed Steel"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
