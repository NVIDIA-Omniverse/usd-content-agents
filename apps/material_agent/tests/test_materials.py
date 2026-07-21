# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared material-name helpers."""

from __future__ import annotations

import pytest

from material_agent.materials import (
    FALLBACK_MATERIAL_BINDING,
    FALLBACK_MATERIAL_ENTRY,
    FALLBACK_MATERIAL_NAME,
    is_actionable_material_name,
    is_default_library_fallback_name,
    is_fallback_material_name,
    is_unknown_material_name,
    material_entries_with_fallback,
    material_mapping_with_fallback,
    normalize_material_name,
)


@pytest.mark.parametrize("name", ["__UNKNOWN__", "__unknown__", " __UNKNOWN__ "])
def test_is_unknown_material_name_accepts_exact_sentinel_variants(name: str) -> None:
    assert is_unknown_material_name(name)
    assert not is_actionable_material_name(name)


@pytest.mark.parametrize(
    "name", [None, "", "   ", "unknown", "Unknown", "unknown material", 123]
)
def test_is_unknown_material_name_rejects_non_sentinel_values(name: object) -> None:
    assert not is_unknown_material_name(name)


@pytest.mark.parametrize(
    "name",
    ["__USE_DEFAULT_LIBRARY__", "__use_default_library__", " __USE_DEFAULT_LIBRARY__ "],
)
def test_is_default_library_fallback_name_accepts_sentinel_variants(
    name: str,
) -> None:
    assert is_default_library_fallback_name(name)
    assert not is_actionable_material_name(name)


@pytest.mark.parametrize("name", [None, "", "default", "__UNKNOWN__", 123])
def test_is_default_library_fallback_name_rejects_other_values(name: object) -> None:
    assert not is_default_library_fallback_name(name)


def test_is_actionable_material_name_requires_non_unknown_text() -> None:
    assert is_actionable_material_name(" Steel ")
    assert is_actionable_material_name("Unknown")
    assert normalize_material_name(" Steel ") == "Steel"
    assert not is_actionable_material_name("")


def test_fallback_material_name_is_actionable() -> None:
    assert is_fallback_material_name(FALLBACK_MATERIAL_NAME)
    assert is_actionable_material_name(FALLBACK_MATERIAL_NAME)
    assert not is_unknown_material_name(FALLBACK_MATERIAL_NAME)
    assert not is_default_library_fallback_name(FALLBACK_MATERIAL_NAME)


def test_material_fallback_helpers_preserve_existing_fallbacks() -> None:
    entries = [FALLBACK_MATERIAL_ENTRY.copy()]
    assert material_entries_with_fallback(entries) is entries
    assert material_entries_with_fallback([{"name": "Steel"}]) == [
        {"name": "Steel"},
        FALLBACK_MATERIAL_ENTRY,
    ]

    mapping = {FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_BINDING}
    assert material_mapping_with_fallback(None) is None
    assert material_mapping_with_fallback(mapping) == mapping
    legacy = [{FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_BINDING}]
    assert material_mapping_with_fallback(legacy) is legacy
    assert material_mapping_with_fallback("not a mapping") == "not a mapping"
