# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared material-name helpers for material-agent tasks."""

from __future__ import annotations

from typing import Any

UNKNOWN_MATERIAL_SENTINEL = "__UNKNOWN__"
USE_DEFAULT_LIBRARY_SENTINEL = "__USE_DEFAULT_LIBRARY__"
USE_DEFAULT_LIBRARY_DESCRIPTION = (
    "Use only when none of the generated materials match the part. This triggers "
    "a separate default-library fallback path instead of selecting from default "
    "materials here."
)
FALLBACK_MATERIAL_NAME = "__FALLBACK_MATERIAL__"
FALLBACK_MATERIAL_BINDING = "/World/Looks/Fallback_Neutral_Gray_Matte_Plastic"
FALLBACK_MATERIAL_DESCRIPTION = (
    "Neutral mid-gray matte plastic fallback for parts whose material cannot be "
    "identified. Use this only when the correct material is unknown or missing."
)
FALLBACK_MATERIAL_ENTRY = {
    "name": FALLBACK_MATERIAL_NAME,
    "binding": FALLBACK_MATERIAL_BINDING,
    "description": FALLBACK_MATERIAL_DESCRIPTION,
}
TRUSTED_FALLBACK_GUIDANCE = {
    USE_DEFAULT_LIBRARY_SENTINEL: USE_DEFAULT_LIBRARY_DESCRIPTION,
    FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_DESCRIPTION,
}
DISALLOWED_UNKNOWN_VALIDATION_STATUS = "disallowed_unknown"
PREDICTION_CONTAINER_KEYS = ("predictions", "results", "items", "objects")
PREDICTION_ID_KEYS = ("id", "object_id", "prim_path", "path")
# Top-level prediction material fields. The singular "materials" container is
# handled separately because it may be either a string or a structured dict.
PREDICTION_MATERIAL_KEYS = ("material", "predicted_material")
PREDICTION_VALIDATION_STATUS_KEYS = ("validation_status", "material_validation_status")
_UNKNOWN_MATERIAL_SENTINEL_NORMALIZED = UNKNOWN_MATERIAL_SENTINEL.lower()
_USE_DEFAULT_LIBRARY_SENTINEL_NORMALIZED = USE_DEFAULT_LIBRARY_SENTINEL.lower()
_FALLBACK_MATERIAL_NAME_NORMALIZED = FALLBACK_MATERIAL_NAME.lower()
_DISALLOWED_UNKNOWN_VALIDATION_STATUS_NORMALIZED = (
    DISALLOWED_UNKNOWN_VALIDATION_STATUS.lower()
)


def normalize_material_name(name: str) -> str:
    """Normalize whitespace around a material name while preserving display case."""
    return name.strip()


def is_unknown_material_name(name: object) -> bool:
    """Return True when a material value is the supported unknown sentinel."""
    return (
        isinstance(name, str)
        and normalize_material_name(name).lower()
        == _UNKNOWN_MATERIAL_SENTINEL_NORMALIZED
    )


def is_default_library_fallback_name(name: object) -> bool:
    """Return True when a material value requests default-library fallback."""
    return (
        isinstance(name, str)
        and normalize_material_name(name).lower()
        == _USE_DEFAULT_LIBRARY_SENTINEL_NORMALIZED
    )


def is_fallback_material_name(name: object) -> bool:
    """Return True when a material value is the canonical fallback material."""
    return (
        isinstance(name, str)
        and normalize_material_name(name).lower() == _FALLBACK_MATERIAL_NAME_NORMALIZED
    )


def is_disallowed_unknown_validation_status(status: object) -> bool:
    """Return True when validation recorded a cleared unknown sentinel."""
    return (
        isinstance(status, str)
        and normalize_material_name(status).lower()
        == _DISALLOWED_UNKNOWN_VALIDATION_STATUS_NORMALIZED
    )


def is_actionable_material_name(name: object) -> bool:
    """Return True when a material should be resolved and applied."""
    return (
        isinstance(name, str)
        and bool(normalize_material_name(name))
        and not is_unknown_material_name(name)
        and not is_default_library_fallback_name(name)
    )


def material_entries_with_fallback(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return material entries plus the canonical fallback entry if absent."""
    if any(entry.get("name") == FALLBACK_MATERIAL_NAME for entry in entries):
        return entries
    return [*entries, FALLBACK_MATERIAL_ENTRY.copy()]


def material_mapping_with_fallback(materials_mapping: Any) -> Any:
    """Return a material mapping plus the canonical fallback binding if absent."""
    if materials_mapping is None:
        return materials_mapping
    if isinstance(materials_mapping, dict):
        mapping = dict(materials_mapping)
        mapping.setdefault(FALLBACK_MATERIAL_NAME, FALLBACK_MATERIAL_BINDING)
        return mapping
    if isinstance(materials_mapping, list):
        has_fallback = any(
            isinstance(item, dict) and FALLBACK_MATERIAL_NAME in item
            for item in materials_mapping
        )
        if has_fallback:
            return materials_mapping
        return [*materials_mapping, {FALLBACK_MATERIAL_NAME: FALLBACK_MATERIAL_BINDING}]
    return materials_mapping
