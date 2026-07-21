# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for material-assignment workflow policy."""

from __future__ import annotations

import pytest

from content_agent_workflows.material_assignment import (
    MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP,
    PAINTED_OR_SATURATED_MATERIAL_TAGS,
    structured_finalizer_guardrail_prompt,
    structured_finalizer_rejection,
)


def test_material_policy_exports_guardrail_prompt() -> None:
    prompt = structured_finalizer_guardrail_prompt()

    assert MATERIAL_ASSIGNMENT_TARGET_PRIM_SOFT_CAP == 16
    assert "paint" in PAINTED_OR_SATURATED_MATERIAL_TAGS
    assert "slender_bar_metal_family" in prompt
    assert "split_large_mixed_groups" in prompt


def test_structured_finalizer_rejection_names_guardrail() -> None:
    rejection = structured_finalizer_rejection("split_broad_painted_mixed_groups")

    assert "mixed-shape group" in rejection
    assert "(guardrail: split_broad_painted_mixed_groups)" in rejection


def test_structured_finalizer_rejection_rejects_unknown_guardrail() -> None:
    with pytest.raises(ValueError, match="unknown_rule"):
        structured_finalizer_rejection("unknown_rule")
