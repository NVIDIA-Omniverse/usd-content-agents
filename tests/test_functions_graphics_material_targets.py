# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared render material target policy."""

import pytest

from world_understanding.functions.graphics.material_targets import (
    normalize_render_material_target,
    preview_fallbacks_enabled_for_material_target,
)


@pytest.mark.parametrize(
    ("material_target", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("preview", "preview_surface"),
        ("openpbr", "openpbr_materialx"),
        ("mdl", "omnipbr_mdl"),
        ("native", "auto"),
    ],
)
def test_normalize_render_material_target_aliases(
    material_target: str | None,
    expected: str,
) -> None:
    assert normalize_render_material_target(material_target) == expected


def test_native_alias_preserves_authored_materials_without_preview_fallbacks() -> None:
    assert normalize_render_material_target("native") == "auto"
    assert preview_fallbacks_enabled_for_material_target("native") is False


def test_preview_fallback_policy_branches() -> None:
    assert preview_fallbacks_enabled_for_material_target("preview_surface") is True
    assert (
        preview_fallbacks_enabled_for_material_target(
            "auto", legacy_add_preview_fallbacks=True
        )
        is True
    )
    assert (
        preview_fallbacks_enabled_for_material_target(
            "auto", legacy_add_preview_fallbacks=False
        )
        is False
    )


def test_invalid_material_target_lists_supported_values() -> None:
    with pytest.raises(ValueError, match="Unsupported material_target"):
        normalize_render_material_target("bad-target")
