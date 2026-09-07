# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal prompt-intent helpers for weathering-capable backends."""

from __future__ import annotations

import re
from typing import Literal

WeatheringEffect = Literal["rust", "dust", "dirt", "wear"]

_PRIMARY_EFFECT_PATTERNS: tuple[tuple[WeatheringEffect, re.Pattern[str]], ...] = (
    (
        "rust",
        re.compile(r"\b(?:rust(?:ed|ing|y)?|corrosion|corroded|oxidation|oxidized)\b"),
    ),
    ("dust", re.compile(r"\b(?:dust(?:ed|ing|y)?|powdery)\b")),
    ("dirt", re.compile(r"\b(?:dirt(?:y)?|grime|grimy|mud|muddy|soil(?:ed|ing)?)\b")),
)
_WEAR_PATTERN = re.compile(r"\b(?:wear|worn|abrasion)\b")
_NEGATED_PREFIX = re.compile(
    r"(?:\bno|\bwithout|\bfree\s+of|\bavoid(?:ing)?)"
    r"(?:[\s,;/]+(?!(?:but|except)\b)[\w-]+){0,3}[\s,;/]*$"
)


def _has_positive_match(text: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 24) : match.start()]
        if not _NEGATED_PREFIX.search(prefix):
            return True
    return False


def infer_weathering_effect(text_prompt: str | None) -> WeatheringEffect | None:
    """Infer one material-behavior class while leaving placement in the prompt.

    Rust, dust, and dirt are primary classes. Generic wear words can refine a
    primary request (for example, "rust around worn edges") without turning it
    into a multi-effect request. Multiple primary classes fail closed because
    their PBR correlation rules differ.
    """

    text = (text_prompt or "").strip().lower()
    if not text:
        return None
    primary = [
        effect
        for effect, pattern in _PRIMARY_EFFECT_PATTERNS
        if _has_positive_match(text, pattern)
    ]
    if len(primary) > 1:
        raise ValueError(
            "prompt requests multiple weathering classes; use one targeted "
            "material request per rust, dust, dirt, or wear behavior"
        )
    if primary:
        return primary[0]
    return "wear" if _has_positive_match(text, _WEAR_PATTERN) else None


def prompt_requests_weathering(text_prompt: str | None) -> bool:
    """Return true for weathering intent, including ambiguous fail-closed text."""

    try:
        return infer_weathering_effect(text_prompt) is not None
    except ValueError:
        return True
