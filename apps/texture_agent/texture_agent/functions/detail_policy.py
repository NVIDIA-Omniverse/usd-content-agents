# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture detail policy helpers."""

from __future__ import annotations

import re
from typing import Any

DETAIL_POLICY_DEFAULT = "default"
DETAIL_POLICY_SURFACE_ONLY = "surface_only"
VALID_DETAIL_POLICIES = frozenset(
    {
        DETAIL_POLICY_DEFAULT,
        DETAIL_POLICY_SURFACE_ONLY,
    }
)

SURFACE_ONLY_FORBIDDEN_DETAILS = (
    "copper traces",
    "traces",
    "vias",
    "pads",
    "silkscreen",
    "labels",
    "text",
    "logos",
    "component outlines",
    "holes",
    "seams",
    "fasteners",
    "circuitry",
    "stickers",
    "markings",
    "semantic modeled geometry details",
)

SURFACE_ONLY_PROMPT_PREFIX = "Surface-only material texture:"
SURFACE_ONLY_PROMPT_TEMPLATE = (
    SURFACE_ONLY_PROMPT_PREFIX + " material swatch: "
    "{description}. Avoid traces, vias, pads, labels, text, logos, holes, "
    "seams, fasteners, components, decals, stickers, linework, symbols, and "
    "geometry markings. Plain roughness, gloss, subtle color, dust, scratches, "
    "and mild wear."
)

_SURFACE_ONLY_STRIP_PATTERNS = (
    r"\b(?:printed|visible|exposed)?\s*copper\s+traces?\b",
    r"\bsilkscreen\s+(?:labels?|text|markings?)\b",
    r"\bcomponent\s+pads?\b",
    r"\b(?:realistic\s+)?(?:electronics?\s+)?boards?\s+markings?\b",
    r"\bprinted\s+circuit\s+board\b",
    r"\bcircuit\s+board\b",
    r"\belectronic\s+board\b",
    r"\belectronics?\b",
    r"\bpcb\b",
    r"\bsilkscreen\b",
    r"\btraces?\b",
    r"\bvias?\b",
    r"\bpads?\b",
    r"\blabels?\b",
    r"\btext\b",
    r"\blogos?\b",
    r"\bcomponent\s+outlines?\b",
    r"\bcomponents?\b",
    r"\bholes?\b",
    r"\bseams?\b",
    r"\bfasteners?\b",
    r"\bcircuits?\b",
    r"\bcircuitry\b",
    r"\bstickers?\b",
    r"\bmarkings?\b",
    r"\bboards?\b",
)


def normalize_detail_policy(
    value: Any,
    *,
    config_key: str,
    default: str = DETAIL_POLICY_DEFAULT,
) -> str:
    """Return a validated detail policy string."""
    candidate = default if value is None else value
    if not isinstance(candidate, str):
        raise ValueError(
            f"{config_key} must be one of {sorted(VALID_DETAIL_POLICIES)}, "
            f"got {type(candidate).__name__}"
        )
    normalized = candidate.strip()
    if normalized not in VALID_DETAIL_POLICIES:
        raise ValueError(
            f"{config_key} must be one of {sorted(VALID_DETAIL_POLICIES)}, "
            f"got {candidate!r}"
        )
    return normalized


def apply_detail_policy_to_prompt(prompt: str, detail_policy: str) -> str:
    """Append prompt guidance for policies that need backend-visible steering."""
    normalized = normalize_detail_policy(
        detail_policy,
        config_key="detail_policy",
    )
    if normalized != DETAIL_POLICY_SURFACE_ONLY:
        return prompt
    if _has_surface_only_guardrails(prompt):
        return prompt
    description = _surface_only_description(_remove_surface_only_prompt_prefix(prompt))
    return SURFACE_ONLY_PROMPT_TEMPLATE.format(description=description)


def _has_surface_only_guardrails(prompt: str) -> bool:
    return (
        prompt.startswith(SURFACE_ONLY_PROMPT_PREFIX)
        and "Avoid traces, vias, pads" in prompt
        and "Plain roughness, gloss" in prompt
    )


def _remove_surface_only_prompt_prefix(prompt: str) -> str:
    if not prompt.startswith(SURFACE_ONLY_PROMPT_PREFIX):
        return prompt
    return prompt.removeprefix(SURFACE_ONLY_PROMPT_PREFIX).strip()


def _surface_only_description(prompt: str) -> str:
    """Remove object/geometry detail terms that can dominate image generation."""
    sanitized = prompt
    for pattern in _SURFACE_ONLY_STRIP_PATTERNS:
        sanitized = re.sub(pattern, " ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    sanitized = re.sub(r"([,.;:]){2,}", r"\1", sanitized)
    sanitized = re.sub(
        r"\b(?:with|and|or|featuring|including)\s*([,.;:])",
        r"\1",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"([,.;:])\s*(?:and|or|with)\b\s*",
        r"\1 ",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(
        r"\b(?:with|and|or|featuring|including)\s*$",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = sanitized.strip(" ,.;:")
    return sanitized or "plain continuous material surface"
