# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for parsing LLM responses."""

import json
import logging
import math
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_OPTIONAL_LIST_PREFIX = r"(?:#{1,6}\s*)?(?:(?:[-*+]|\d+[.)]|\(\d+\))\s*)?"


def extract_labeled_value(
    response_text: str,
    labels: str | Sequence[str],
    *,
    multiline: bool = False,
    boundary_labels: Sequence[str] | None = None,
    stop_at_generic_headings: bool = False,
) -> str:
    """Extract a simple labeled value from an LLM response.

    This intentionally handles only the common judge-output shape used across
    agents: ``Label: value`` with optional markdown bold around the label.
    """
    if not response_text:
        return ""

    label_values = (labels,) if isinstance(labels, str) else tuple(labels)
    boundary_label_values = dedupe_strings(
        (
            *label_values,
            *(boundary_labels or _DEFAULT_MULTILINE_BOUNDARY_LABELS),
        )
    )
    lines = response_text.splitlines()
    for index, line in enumerate(lines):
        for label in label_values:
            match = _match_labeled_line(line, label)
            if not match:
                continue

            parts = [(match.group("value") or "").strip()]
            if multiline:
                for next_line in lines[index + 1 :]:
                    stripped = next_line.strip()
                    if _looks_like_labeled_heading(
                        stripped,
                        boundary_label_values,
                        include_generic=stop_at_generic_headings,
                    ):
                        break
                    if not stripped:
                        continue
                    parts.append(stripped)
            return "\n".join(part for part in parts if part).strip()
    return ""


def extract_labeled_score(
    response_text: str,
    labels: str | Sequence[str] = "Score",
    *,
    score_max: float = 10.0,
) -> float | None:
    """Extract a normalized 0-1 score from a labeled LLM judge response."""
    value = extract_labeled_value(response_text, labels, multiline=True)
    if value:
        parsed_value = _parse_score_value(value, score_max=score_max)
        if parsed_value is not None:
            return parsed_value

    label_values = (labels,) if isinstance(labels, str) else tuple(labels)
    label_pattern = "|".join(re.escape(label) for label in label_values)
    legacy_match = re.search(
        rf"(?<!\w)(?:\*\*)?(?:{label_pattern})(?:\*\*)?\s*:?\s*(?:\*\*)?\s*"
        r"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?",
        response_text,
        flags=re.IGNORECASE,
    )
    if legacy_match:
        explicit_max = float(legacy_match.group(2)) if legacy_match.group(2) else None
        if explicit_max is None:
            return _normalize_unqualified_score(
                legacy_match.group(1),
                score_max=score_max,
            )
        return _normalize_score(
            float(legacy_match.group(1)),
            explicit_max,
            score_max=score_max,
        )
    return None


def extract_labeled_choice(
    response_text: str,
    labels: str | Sequence[str],
    choices: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    boundary_labels: Sequence[str] | None = None,
) -> str:
    """Extract a structured choice token from a labeled LLM response field."""
    label_values = (labels,) if isinstance(labels, str) else tuple(labels)
    value = extract_labeled_value(
        response_text,
        label_values,
        multiline=True,
        boundary_labels=boundary_labels,
    )
    parsed = _parse_choice_value(value, choices, aliases=aliases)
    if parsed:
        return parsed
    return _extract_whitespace_labeled_choice(
        response_text,
        label_values,
        choices,
        aliases=aliases,
    )


def _match_labeled_line(line: str, label: str) -> re.Match[str] | None:
    return re.match(
        rf"^\s*{_OPTIONAL_LIST_PREFIX}(?:\*\*)?{re.escape(label)}(?:\*\*)?"
        r"(?:\s*:\s*(?:\*\*)?\s*(?P<value>.*)|\s*$)",
        line,
        flags=re.IGNORECASE,
    )


def _parse_score_value(value: str, *, score_max: float) -> float | None:
    """Parse a score value without treating unrelated prose numbers as scores."""
    leading_fraction_match = re.match(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?(\d+(?:\.\d+)?)"
        r"\s*/\s*(\d+(?:\.\d+)?)",
        value,
    )
    if leading_fraction_match:
        return _normalize_score(
            float(leading_fraction_match.group(1)),
            float(leading_fraction_match.group(2)),
            score_max=score_max,
        )

    leading_out_of_match = re.match(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?(\d+(?:\.\d+)?)"
        r"\s+out\s+of\s+(\d+(?:\.\d+)?)\b",
        value,
        flags=re.IGNORECASE,
    )
    if leading_out_of_match:
        return _normalize_score(
            float(leading_out_of_match.group(1)),
            float(leading_out_of_match.group(2)),
            score_max=score_max,
        )

    leading_match = re.match(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?(\d+(?:\.\d+)?)(?!\s*/)",
        value,
    )
    if leading_match:
        return _normalize_unqualified_score(
            leading_match.group(1),
            score_max=score_max,
        )

    score_word_matches = list(
        re.finditer(
            r"\b(?:score|rating)\b\s*(?:is|=|:)?\s*"
            r"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?",
            value,
            flags=re.IGNORECASE,
        )
    )
    if score_word_matches:
        match = score_word_matches[-1]
        explicit_max = float(match.group(2)) if match.group(2) else None
        if explicit_max is None:
            return _normalize_unqualified_score(
                match.group(1),
                score_max=score_max,
            )
        return _normalize_score(
            float(match.group(1)),
            explicit_max,
            score_max=score_max,
        )

    out_of_matches = list(
        re.finditer(
            r"\b(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\b",
            value,
            flags=re.IGNORECASE,
        )
    )
    if out_of_matches:
        match = out_of_matches[0]
        return _normalize_score(
            float(match.group(1)),
            float(match.group(2)),
            score_max=score_max,
        )

    fraction_matches = list(
        re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w.])",
            value,
        )
    )
    if fraction_matches:
        match = fraction_matches[0]
        return _normalize_score(
            float(match.group(1)),
            float(match.group(2)),
            score_max=score_max,
        )
    return None


def _parse_choice_value(
    value: str,
    choices: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    if not value:
        return ""

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return ""

    canonical = {_choice_key(choice): choice for choice in choices}
    alias_values = aliases or {}
    candidates: list[tuple[tuple[str, ...], str]] = [
        (_choice_tokens(alias), target) for alias, target in alias_values.items()
    ]
    candidates.extend((_choice_tokens(choice), choice) for choice in choices)

    for line in lines:
        parsed = _parse_choice_line(line, candidates, canonical)
        if parsed:
            return parsed
    return ""


def _parse_choice_line(
    line: str,
    candidates: Sequence[tuple[tuple[str, ...], str]],
    canonical: Mapping[str, str],
) -> str:
    stripped_line = _strip_choice_formatting(line)
    tokens = _choice_tokens(stripped_line)
    if _looks_like_choice_option_list(stripped_line, tokens, candidates):
        return ""
    for candidate_tokens, target in sorted(
        candidates,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        target_key = _choice_key(target)
        if target_key not in canonical or not candidate_tokens:
            continue
        if _tokens_start_with(
            tokens,
            candidate_tokens,
        ) and not _tokens_reject_leading_choice(tokens, candidate_tokens):
            return canonical[target_key]
    for candidate_tokens, target in sorted(
        candidates,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        target_key = _choice_key(target)
        if target_key not in canonical or not candidate_tokens:
            continue
        if _tokens_contain_led_choice(tokens, candidate_tokens):
            return canonical[target_key]
        if _tokens_end_with_unnegated_choice(tokens, candidate_tokens):
            return canonical[target_key]
    return ""


def _looks_like_choice_option_list(
    value: str,
    tokens: Sequence[str],
    candidates: Sequence[tuple[tuple[str, ...], str]],
) -> bool:
    matched_targets = {
        target
        for candidate_tokens, target in candidates
        if candidate_tokens and _tokens_contain_sequence(tokens, candidate_tokens)
    }
    if len(matched_targets) <= 1:
        return False
    if value.lstrip().startswith(("[", "(")):
        return True
    if "if" in tokens:
        return True

    choice_tokens = {
        token for candidate_tokens, _ in candidates for token in candidate_tokens
    }
    non_choice_tokens = [
        token for token in tokens if token not in choice_tokens and token != "or"
    ]
    return len(non_choice_tokens) <= 1


def _extract_whitespace_labeled_choice(
    response_text: str,
    labels: Sequence[str],
    choices: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    for line in response_text.splitlines():
        for label in labels:
            match = re.match(
                rf"^\s*(?:[-*+]|\d+[.)])?\s*(?:\*\*)?{re.escape(label)}"
                r"(?:\*\*)?\s+(?:\*\*)?\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            parsed = _parse_choice_value(
                match.group(1),
                choices,
                aliases=aliases,
            )
            if parsed:
                return parsed
    return ""


def _choice_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _choice_tokens(value: str) -> tuple[str, ...]:
    key = _choice_key(value)
    if not key:
        return ()
    return tuple(token for token in key.split("_") if token)


def _strip_choice_formatting(value: str) -> str:
    return re.sub(rf"^\s*{_OPTIONAL_LIST_PREFIX}", "", value).strip("`*_ \t")


def _tokens_start_with(
    tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    return len(tokens) >= len(candidate_tokens) and tuple(
        tokens[: len(candidate_tokens)]
    ) == tuple(candidate_tokens)


def _tokens_contain_sequence(
    tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    if len(tokens) < len(candidate_tokens):
        return False
    for index in range(0, len(tokens) - len(candidate_tokens) + 1):
        if tuple(tokens[index : index + len(candidate_tokens)]) == tuple(
            candidate_tokens
        ):
            return True
    return False


_CHOICE_LEADIN_TOKENS = {
    "be",
    "choose",
    "conclude",
    "conclusion",
    "decision",
    "final",
    "is",
    "must",
    "recommend",
    "recommendation",
    "result",
    "select",
    "selected",
    "should",
    "therefore",
    "verdict",
    "will",
    "would",
}
_CHOICE_NEGATION_TOKENS = {"cannot", "never", "no", "not", "without"}
_CHOICE_REJECTION_TOKENS = {
    "inappropriate",
    "incorrect",
    "invalid",
    "reject",
    "rejected",
    "wrong",
}
_CHOICE_REJECTION_TARGET_TOKENS = {
    "acceptable",
    "appropriate",
    "correct",
    "valid",
}
_CHOICE_REJECTION_PHRASES = {
    ("too", "harsh"),
    ("too", "severe"),
    ("too", "strict"),
}


def _tokens_reject_leading_choice(
    tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    rest = tokens[len(candidate_tokens) : len(candidate_tokens) + 8]
    for index, token in enumerate(rest):
        if token in _CHOICE_REJECTION_TOKENS:
            return True
        if any(
            tuple(rest[index : index + len(phrase)]) == phrase
            for phrase in _CHOICE_REJECTION_PHRASES
        ):
            return True
        if token in _CHOICE_NEGATION_TOKENS and any(
            target in _CHOICE_REJECTION_TARGET_TOKENS
            for target in rest[index + 1 : index + 4]
        ):
            return True
    return False


def _tokens_contain_led_choice(
    tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    for index in range(1, len(tokens) - len(candidate_tokens) + 1):
        if tuple(tokens[index : index + len(candidate_tokens)]) != tuple(
            candidate_tokens
        ):
            continue
        context = tokens[max(0, index - 6) : index]
        if any(token in _CHOICE_NEGATION_TOKENS for token in context):
            continue
        if any(token in _CHOICE_LEADIN_TOKENS for token in context):
            return True
    return False


def _tokens_end_with_unnegated_choice(
    tokens: Sequence[str],
    candidate_tokens: Sequence[str],
) -> bool:
    if len(tokens) <= len(candidate_tokens):
        return False
    start_index = len(tokens) - len(candidate_tokens)
    if tuple(tokens[start_index:]) != tuple(candidate_tokens):
        return False
    context = tokens[max(0, start_index - 6) : start_index]
    return not any(token in _CHOICE_NEGATION_TOKENS for token in context)


def _normalize_score(
    score_value: float,
    explicit_max: float | None,
    *,
    score_max: float,
) -> float | None:
    if not math.isfinite(score_value):
        return None
    if explicit_max is not None:
        if explicit_max <= 0 or not math.isfinite(explicit_max):
            return None
        score_value /= explicit_max
    elif score_value > 1.0 and score_max > 0:
        score_value /= score_max
    if not math.isfinite(score_value):
        return None
    return max(0.0, min(score_value, 1.0))


def _normalize_unqualified_score(score_text: str, *, score_max: float) -> float | None:
    score_value = float(score_text)
    if score_max > 1.0:
        return _normalize_score(score_value, score_max, score_max=score_max)
    return _normalize_score(score_value, None, score_max=score_max)


def extract_labeled_codes(
    response_text: str,
    labels: str | Sequence[str],
    *,
    allowed_codes: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Extract dotted issue/category codes from a labeled response field."""
    value = extract_labeled_value(
        response_text,
        labels,
        multiline=True,
        stop_at_generic_headings=True,
    )
    normalized = value.strip().lower()
    if not normalized or normalized in {"none", "n/a", "no issues", "no defects"}:
        return ()

    allowed = set(allowed_codes or ())
    codes: list[str] = []
    for line in normalized.splitlines():
        candidate = _strip_issue_code_line_prefix(line)
        matches = list(re.finditer(_ISSUE_CODE_PATTERN, candidate))
        if not matches:
            continue
        for match in matches:
            code = match.group(0)
            if _issue_code_match_is_negated(
                candidate,
                match.start(),
                match.end(),
            ):
                continue
            codes.append(code)
    if allowed:
        codes = [code for code in codes if code in allowed]
    return dedupe_strings(codes)


_ISSUE_CODE_PATTERN = r"[a-z][a-z0-9_]*(?:\.[a-z0-9_][a-z0-9_-]*)+"


def _strip_issue_code_line_prefix(line: str) -> str:
    candidate = re.sub(rf"^\s*{_OPTIONAL_LIST_PREFIX}", "", line).strip()
    return candidate.strip("`[](){}<> \t")


def _issue_code_match_is_negated(
    candidate: str,
    match_start: int,
    match_end: int,
) -> bool:
    context = candidate[max(0, match_start - 80) : match_start]
    local_context = re.split(
        r"\b(?:but|however|though|although|while)\b|[.;,]",
        context,
        flags=re.IGNORECASE,
    )[-1]
    before_negated = _issue_code_preceded_by_absence_phrase(local_context)
    if before_negated:
        return True
    trailing_context = candidate[match_end : match_end + 80]
    local_trailing_context = re.split(
        r"\b(?:but|however|though|although|while)\b|[.;]",
        trailing_context,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _issue_code_followed_by_absence_phrase(local_trailing_context)


def _issue_code_followed_by_absence_phrase(local_trailing_context: str) -> bool:
    prefix = r"^[\s`'\")\]]*"
    absent_state = (
        r"(?:visible|present|seen|detected|found|observed|identified|"
        r"apparent|evident)"
    )
    passive_absent_state = r"(?:seen|detected|found|observed|identified)"
    return bool(
        re.search(
            prefix
            + r"(?:(?:is|are|was|were)\s+)?"
            + r"(?:(?:not|no\s+longer)\s+(?:clearly\s+)?"
            + absent_state
            + r"|(?:not|no\s+longer)\s+(?:an?\s+)?"
            + r"(?:issue|defect|problem|applicable))\b",
            local_trailing_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            prefix
            + r"(?:isn't|aren't|wasn't|weren't)\s+(?:clearly\s+)?"
            + absent_state
            + r"\b",
            local_trailing_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            prefix
            + r"(?:doesn't|does\s+not|don't|do\s+not|cannot|can't|"
            + r"couldn't|could\s+not)\s+(?:appear|seem|look)\s+"
            + r"(?:to\s+be\s+)?(?:clearly\s+)?"
            + absent_state
            + r"\b",
            local_trailing_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            prefix
            + r"(?:cannot|can't|couldn't|could\s+not)\s+be\s+"
            + passive_absent_state
            + r"\b",
            local_trailing_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            prefix
            + r"(?:is|are|was|were|appears|seems|looks)\s+"
            + r"(?:absent|missing|resolved|fixed)\b",
            local_trailing_context,
            flags=re.IGNORECASE,
        )
    )


_DEFAULT_MULTILINE_BOUNDARY_LABELS = (
    "Blocking Defect",
    "Blocking Defects",
    "Critique",
    "Decision",
    "Evidence Note",
    "Evidence Notes",
    "Improvement Suggestion",
    "Improvement Suggestions",
    "Issue Code",
    "Issue Codes",
    "Recommendation",
    "Recommendations",
    "Score",
    "Suggestion",
    "Suggestions",
)
_GENERIC_MULTILINE_BOUNDARY_LABELS = (
    "Analysis",
    "Note",
    "Notes",
    "Observation",
    "Observations",
    "Reasoning",
    "Summary",
)


def _looks_like_labeled_heading(
    line: str,
    labels: Sequence[str],
    *,
    include_generic: bool = False,
) -> bool:
    for label in labels:
        if _match_labeled_line(line, label) or _match_labeled_boundary_line(
            line,
            label,
        ):
            return True
    if not include_generic:
        return False
    return _looks_like_generic_section_heading(line)


def _looks_like_generic_section_heading(line: str) -> bool:
    stripped = line.strip()
    candidate = _strip_markdown_heading_prefix(stripped)
    if candidate is not None:
        candidate = candidate.strip()
    elif stripped.startswith("**"):
        candidate = stripped
    else:
        candidate = _strip_optional_heading_list_prefix(line).strip()
    candidate = _strip_bold_heading(candidate)
    return any(
        _candidate_starts_labeled_heading(candidate, label)
        for label in _GENERIC_MULTILINE_BOUNDARY_LABELS
    )


def _strip_markdown_heading_prefix(stripped: str) -> str | None:
    marker_count = len(stripped) - len(stripped.lstrip("#"))
    if (
        1 <= marker_count <= 6
        and len(stripped) > marker_count
        and stripped[marker_count].isspace()
    ):
        return stripped[marker_count:]
    return None


def _strip_optional_heading_list_prefix(line: str) -> str:
    stripped = line.lstrip()
    if not stripped:
        return ""
    if stripped[0] in "-*+":
        return stripped[1:].lstrip() if _prefix_has_boundary(stripped, 1) else stripped
    digit_end = _consume_digits(stripped, 0)
    if digit_end > 0 and digit_end < len(stripped) and stripped[digit_end] in ".)":
        return (
            stripped[digit_end + 1 :].lstrip()
            if _prefix_has_boundary(stripped, digit_end + 1)
            else stripped
        )
    if stripped.startswith("("):
        closing_index = _consume_digits(stripped, 1)
        if (
            closing_index > 1
            and closing_index < len(stripped)
            and stripped[closing_index] == ")"
        ):
            return (
                stripped[closing_index + 1 :].lstrip()
                if _prefix_has_boundary(stripped, closing_index + 1)
                else stripped
            )
    return stripped


def _consume_digits(value: str, start: int) -> int:
    index = start
    while index < len(value) and value[index].isdigit():
        index += 1
    return index


def _prefix_has_boundary(value: str, prefix_end: int) -> bool:
    return prefix_end >= len(value) or value[prefix_end].isspace()


def _strip_bold_heading(candidate: str) -> str:
    if not candidate.startswith("**"):
        return candidate
    closing_index = candidate.find("**", 2)
    if closing_index == -1:
        return candidate
    return f"{candidate[2:closing_index]}{candidate[closing_index + 2 :]}".strip()


def _candidate_starts_labeled_heading(candidate: str, label: str) -> bool:
    if not candidate.casefold().startswith(label.casefold()):
        return False
    remainder = candidate[len(label) :]
    return not remainder or remainder.lstrip().startswith(":")


def _match_labeled_boundary_line(line: str, label: str) -> bool:
    label_key = label.lower()
    if label_key == "decision":
        value_pattern = r"approve|continue|fail|needs[_\s-]?refinement|pass|warn"
    elif label_key == "score":
        value_pattern = r"\d"
    else:
        return False
    return bool(
        re.match(
            rf"^\s*{_OPTIONAL_LIST_PREFIX}(?:\*\*)?{re.escape(label)}"
            rf"(?:\*\*)?\s+(?:\*\*)?(?:{value_pattern})\b",
            line,
            flags=re.IGNORECASE,
        )
    )


def _issue_code_preceded_by_absence_phrase(local_context: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\b)(?:not|no|without)\s+(?:any\s+)?(?:a\s+|an\s+)?$",
            local_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:don't|do\s+not|cannot|can't|couldn't|could\s+not|"
            r"shouldn't|should\s+not)\s+"
            r"(?:see|detect|find|observe|identify|consider|call|classify|"
            r"treat)(?:\s+(?:any|a|an|as|it|this|that))*\s*$",
            local_context,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:cannot|can't|couldn't|could\s+not|shouldn't|should\s+not)"
            r"\s+be\s+(?:considered|called|classified|treated)"
            r"(?:\s+as)?\s+(?:a\s+|an\s+)?$",
            local_context,
            flags=re.IGNORECASE,
        )
    )


def dedupe_strings(values: Sequence[str]) -> tuple[str, ...]:
    """Return strings in first-seen order with duplicates removed."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def extract_last_answer_block(text: str) -> str | None:
    """Extract content from the last exact ``<answer>`` block in text."""
    answer_matches = re.findall(
        r"<answer\b[^>]*>(.*?)</answer>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not answer_matches:
        return None
    return answer_matches[-1].strip()


def _strip_code_fence_language(content: str) -> str:
    stripped = content.strip()
    lower = stripped.lower()
    if lower.startswith("json") and (len(stripped) == 4 or stripped[4].isspace()):
        return stripped[4:].strip()
    return stripped


def _iter_code_fences(text: str) -> Iterator[str]:
    search_start = 0
    while True:
        fence_start = text.find("```", search_start)
        if fence_start == -1:
            return
        content_start = fence_start + 3
        fence_end = text.find("```", content_start)
        if fence_end == -1:
            return
        yield _strip_code_fence_language(text[content_start:fence_end])
        search_start = fence_end + 3


def _iter_json_objects(text: str) -> Iterator[str]:
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            spans.append((start, index + 1))

    for start, end in sorted(spans, key=lambda span: (span[0], span[0] - span[1])):
        yield text[start:end]


def _parse_json_dict_candidate(json_str: str) -> dict[str, Any] | None:
    attempts = [json_str]
    stripped = json_str.strip()
    if stripped.startswith("{{") and stripped.endswith("}}"):
        attempts.append(stripped[1:-1])

    for attempt in attempts:
        try:
            result = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict):
            logger.error(f"Expected dict but got {type(result)}")
            return None
        return result
    return None


def _missing_expected_keys(
    result: Mapping[str, Any],
    expected_keys: Collection[str] | None,
) -> list[str]:
    if not expected_keys:
        return []
    return [key for key in expected_keys if key not in result]


def _iter_json_candidates(candidate: str) -> Iterator[str]:
    for fenced in _iter_code_fences(candidate):
        yield from _iter_json_objects(fenced)
    yield from _iter_json_objects(candidate)


def iter_json_dicts_from_llm_response(response_text: str) -> Iterator[dict[str, Any]]:
    """Yield JSON dictionaries found in an LLM response."""
    if not response_text:
        return

    candidates = [response_text.strip()]
    answer_text = extract_last_answer_block(response_text)
    if answer_text:
        candidates.insert(0, answer_text)

    seen_json: set[str] = set()
    for candidate in candidates:
        for json_str in _iter_json_candidates(candidate):
            dedupe_key = json_str.strip()
            if dedupe_key in seen_json:
                continue
            seen_json.add(dedupe_key)

            result = _parse_json_dict_candidate(json_str)
            if result is not None:
                yield result


def iter_json_dicts_in_text_order(response_text: str) -> Iterator[dict[str, Any]]:
    """Yield JSON dictionaries in their original textual order.

    Unlike :func:`iter_json_dicts_from_llm_response`, this does not prioritize
    fenced or ``<answer>`` content. It is intended for contracts that explicitly
    select the last structurally valid object in a response.
    """
    if not response_text:
        return

    seen_json: set[str] = set()
    for json_str in _iter_json_objects(response_text):
        dedupe_key = json_str.strip()
        if dedupe_key in seen_json:
            continue
        seen_json.add(dedupe_key)
        result = _parse_json_dict_candidate(json_str)
        if result is not None:
            yield result


def extract_json_from_llm_response(
    response_text: str,
    expected_keys: Collection[str] | None = None,
    *,
    log_failures: bool = True,
) -> dict[str, Any] | None:
    """
    Extract JSON object from LLM response text.

    Handles various formats including:
    - JSON wrapped in markdown code blocks (```json or ```)
    - JSON with surrounding explanatory text
    - Plain JSON responses

    Args:
        response_text: The raw response text from the LLM
        expected_keys: Optional collection of keys required in the returned JSON.
            Candidates missing these keys are skipped so earlier reasoning
            dictionaries do not mask the final answer object.
        log_failures: Whether to log parser failures before returning ``None``.

    Returns:
        Parsed JSON as a dictionary, or None if parsing fails
    """
    if not response_text:
        if log_failures:
            logger.error("Empty response text provided")
        return None

    try:
        result = None
        last_missing_keys: list[str] = []
        for candidate_result in iter_json_dicts_from_llm_response(response_text):
            missing_keys = _missing_expected_keys(candidate_result, expected_keys)
            if missing_keys:
                last_missing_keys = missing_keys
                if log_failures:
                    logger.debug(
                        "Skipping JSON object missing expected keys: %s",
                        missing_keys,
                    )
                continue
            result = candidate_result
            logger.debug("Found JSON object in LLM response")
            break

        if result is not None:
            logger.debug("Found JSON object in response text")

        if result is None:
            if log_failures and last_missing_keys:
                logger.warning(f"JSON missing expected keys: {last_missing_keys}")
            if log_failures:
                truncated_response = f"{response_text[:200]}..."
                logger.error(
                    "No JSON found in LLM response: %s",
                    truncated_response,
                )
            return None

        return result

    except Exception as e:
        if log_failures:
            logger.error(f"Unexpected error parsing LLM response: {e}")
        return None


def extract_material_from_json(
    json_obj: dict[str, Any], possible_keys: list[str] | None = None
) -> str | None:
    """
    Recursively extract material value from JSON with flexible schema.

    Handles various JSON structures including:
    - Direct: {"material": "value"}
    - Nested: {{"material": "value"}}
    - Alternative keys: {"predicted_material": "value"}
    - Deep nesting: {"result": {"material": "value"}}

    Args:
        json_obj: The JSON object to extract from
        possible_keys: List of possible key names for material (default: common variations)

    Returns:
        The material string value, or None if not found

    Examples:
        >>> extract_material_from_json({"material": "Plastic Dark Blue"})
        "Plastic Dark Blue"
        >>> extract_material_from_json({{"material": "Plastic Dark Blue"}})
        "Plastic Dark Blue"
        >>> extract_material_from_json({"result": {"predicted_material": "Steel"}})
        "Steel"
    """
    if possible_keys is None:
        possible_keys = [
            "material",
            "predicted_material",
            "material_name",
            "name",
            "value",
            "result",
        ]

    # If json_obj is a string, return it directly
    if isinstance(json_obj, str):
        return json_obj

    # If not a dict, can't extract
    if not isinstance(json_obj, dict):
        logger.debug(f"Cannot extract material from non-dict: {type(json_obj)}")
        return None

    # Try direct key lookup first
    for key in possible_keys:
        if key in json_obj:
            value = json_obj[key]
            # If value is a string, we found it
            if isinstance(value, str):
                return value
            # If value is a dict, recurse into it
            elif isinstance(value, dict):
                result = extract_material_from_json(value, possible_keys)
                if result:
                    return result

    # If no direct match, check if there's a single nested dict
    # This handles cases like {{"material": "value"}}
    if len(json_obj) == 1:
        single_key = next(iter(json_obj))
        single_value = json_obj[single_key]

        # If the single value is a dict, recurse
        if isinstance(single_value, dict):
            result = extract_material_from_json(single_value, possible_keys)
            if result:
                return result

        # If it's a string and key matches material patterns
        if isinstance(single_value, str) and single_key in possible_keys:
            return single_value

    # Last resort: search all values recursively
    for value in json_obj.values():
        if isinstance(value, dict):
            result = extract_material_from_json(value, possible_keys)
            if result:
                return result

    logger.debug(f"No material found in JSON: {json_obj}")
    return None


def create_json_prompt_instructions() -> str:
    """
    Get standard instructions for prompting LLMs to return JSON.

    Returns:
        String with instructions to include in prompts
    """
    return (
        "Return ONLY a valid JSON object. Do not include any explanatory "
        "text, markdown formatting, or code blocks. Just the raw JSON."
    )
