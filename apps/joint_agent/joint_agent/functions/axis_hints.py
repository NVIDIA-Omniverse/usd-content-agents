# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared coordinate-axis hint normalization for Joint Agent."""

from __future__ import annotations

import re
from typing import Any

SUPPORTED_AXIS_HINTS = frozenset(
    {
        "x",
        "y",
        "z",
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    }
)

_SIGNED_AXIS_HINT_RE = re.compile(r"^([+-])[\s_-]*([xyz])(?:[\s_-]*axis)?$")

# Keep this table deliberately small. Ambiguous words such as "horizontal" or
# "sideways" are not axis evidence without explicit x/y/z scene-frame context.
_AXIS_HINT_NORMALIZATION: dict[str, str] = {
    "axis_x": "x",
    "axis_y": "y",
    "axis_z": "z",
    "xaxis": "x",
    "yaxis": "y",
    "zaxis": "z",
    "along_x": "x",
    "along_y": "y",
    "along_z": "z",
    "around_x": "x",
    "around_y": "y",
    "around_z": "z",
    "positive_x": "+x",
    "positive_y": "+y",
    "positive_z": "+z",
    "plus_x": "+x",
    "plus_y": "+y",
    "plus_z": "+z",
    "negative_x": "-x",
    "negative_y": "-y",
    "negative_z": "-z",
    "minus_x": "-x",
    "minus_y": "-y",
    "minus_z": "-z",
}


def normalize_axis_hint_token(value: Any) -> str | None:
    """Return a canonical coordinate-axis token, or None when not explicit."""
    axis = _clean_axis_hint_text(value)
    # First accept compact canonical spellings; then handle signed phrases with
    # separators, such as "+ y axis", before falling back to the alias table.
    compact_axis = axis.replace(" ", "")
    compact_axis = compact_axis.replace("_axis", "").replace("-axis", "")
    if compact_axis in SUPPORTED_AXIS_HINTS:
        return compact_axis
    signed_axis_match = _SIGNED_AXIS_HINT_RE.match(axis)
    if signed_axis_match is not None:
        return f"{signed_axis_match.group(1)}{signed_axis_match.group(2)}"
    axis_label_key = _axis_label_key(axis)
    normalized_axis = _AXIS_HINT_NORMALIZATION.get(axis_label_key)
    if normalized_axis is not None:
        return normalized_axis
    axis_label_without_suffix = _strip_axis_suffix(axis_label_key)
    if axis_label_without_suffix in SUPPORTED_AXIS_HINTS:
        return axis_label_without_suffix
    return _AXIS_HINT_NORMALIZATION.get(axis_label_without_suffix)


def axis_hint_alias_items() -> tuple[tuple[str, str], ...]:
    """Return explicit axis phrase aliases for contract tests."""
    return tuple(sorted(_AXIS_HINT_NORMALIZATION.items()))


def _clean_axis_hint_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text == "":
        return ""
    return " ".join(text.split())


def _axis_label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _strip_axis_suffix(value: str) -> str:
    if value.endswith("_axis"):
        return value.removesuffix("_axis")
    return value
