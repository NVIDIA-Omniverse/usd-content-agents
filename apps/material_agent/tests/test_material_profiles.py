# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material authoring profile helpers."""

import pytest

from material_agent.material_profiles import normalize_material_profile


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("native", "auto"),
        ("displayColor", "display_color"),
        ("preview", "preview_surface"),
        ("Usd Preview Surface", "preview_surface"),
        ("openpbr", "openpbr_materialx"),
        ("MaterialX", "openpbr_materialx"),
        ("mdl", "omnipbr_mdl"),
        ("rtx", "omnipbr_mdl"),
    ],
)
def test_normalize_material_profile_aliases(value: str | None, expected: str) -> None:
    assert normalize_material_profile(value) == expected


def test_normalize_material_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported material_profile"):
        normalize_material_profile("cycles")
