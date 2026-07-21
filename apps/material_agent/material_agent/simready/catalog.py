# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight SimReady material catalog loading and selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SIMREADY_RELEASE_TAG = "v0.2.0"
SIMREADY_LIGHT_ID = "simready-light"
SIMREADY_FULL_ID = "simready-full"
SIMREADY_CATEGORY_PREFIX = "simready-category:"
_DEFAULT_MANIFEST_NAME = "physicalai_simready_materials_v0_2_0.yaml"


class SimReadyCatalogError(ValueError):
    """Raised when a SimReady material library selection cannot be resolved."""


@dataclass(frozen=True)
class SimReadyLibrarySelection:
    """Parsed SimReady material-library request."""

    library_id: str
    mode: str
    category: str | None = None


def is_simready_library_id(library_id: str | None) -> bool:
    """Return True if *library_id* is in the SimReady namespace."""
    if not isinstance(library_id, str):
        return False
    value = library_id.strip()
    return value in {SIMREADY_LIGHT_ID, SIMREADY_FULL_ID} or value.startswith(
        SIMREADY_CATEGORY_PREFIX
    )


def parse_simready_library_id(library_id: str) -> SimReadyLibrarySelection:
    """Parse a documented SimReady material-library ID."""
    value = library_id.strip()
    if value == SIMREADY_LIGHT_ID:
        return SimReadyLibrarySelection(library_id=value, mode="light")
    if value == SIMREADY_FULL_ID:
        return SimReadyLibrarySelection(library_id=value, mode="full")
    if value.startswith(SIMREADY_CATEGORY_PREFIX):
        category = value[len(SIMREADY_CATEGORY_PREFIX) :].strip()
        if not category:
            raise SimReadyCatalogError(
                "SimReady category library is missing a category"
            )
        return SimReadyLibrarySelection(
            library_id=value,
            mode="category",
            category=category,
        )
    raise SimReadyCatalogError(f"Unknown SimReady material library: {library_id}")


def load_default_manifest() -> dict[str, Any]:
    """Load the packaged default SimReady manifest."""
    data_ref = resources.files(__package__).joinpath("data", _DEFAULT_MANIFEST_NAME)
    with data_ref.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise SimReadyCatalogError("Packaged SimReady manifest is not a mapping")
    return loaded


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load a SimReady manifest from *path* or the packaged default."""
    if path is None:
        return load_default_manifest()
    with Path(path).open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise SimReadyCatalogError(f"SimReady manifest is not a mapping: {path}")
    return loaded


def _manifest_categories(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    categories = manifest.get("categories")
    if not isinstance(categories, dict):
        raise SimReadyCatalogError("SimReady manifest is missing categories")
    normalized: dict[str, dict[str, Any]] = {}
    for category, data in categories.items():
        if not isinstance(data, dict):
            raise SimReadyCatalogError(
                f"Category metadata must be a mapping: {category!r}"
            )
        normalized[str(category)] = data
    return normalized


def _manifest_materials(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    materials = manifest.get("materials")
    if not isinstance(materials, list):
        raise SimReadyCatalogError("SimReady manifest is missing materials")
    normalized: list[dict[str, Any]] = []
    required = ("name", "binding", "id", "category", "source_path")
    for idx, item in enumerate(materials):
        if not isinstance(item, dict):
            raise SimReadyCatalogError(
                f"Material entry at index {idx} is not a mapping"
            )
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise SimReadyCatalogError(
                f"Material entry at index {idx} missing required keys: {missing}"
            )
        normalized.append(item)
    return normalized


def _canonical_category(
    requested: str,
    categories: dict[str, dict[str, Any]],
) -> str:
    for category in categories:
        if category.lower() == requested.lower():
            return category
    raise SimReadyCatalogError(f"Unsupported SimReady category: {requested}")


def _allowed_categories(
    categories: dict[str, dict[str, Any]],
    allowed_categories: set[str] | None,
) -> set[str]:
    allowed = set(categories)
    if allowed_categories is not None:
        normalized = {item.lower() for item in allowed_categories}
        allowed = {category for category in allowed if category.lower() in normalized}
    allowed = {
        category
        for category in allowed
        if _category_archive_layout_is_supported(categories[category])
    }
    return allowed


def _category_archive_layout_is_supported(metadata: dict[str, Any]) -> bool:
    if bool(metadata.get("requires_split_archive")):
        return False
    archive_files = metadata.get("archive_files")
    return not isinstance(archive_files, list) or len(archive_files) <= 1


def _material_entry(material: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(material["name"]),
        "description": str(material.get("description") or ""),
        "binding": str(material["binding"]),
        "simready_id": str(material["id"]),
        "simready_category": str(material["category"]),
        "simready_source_path": str(material["source_path"]),
    }


def build_material_entries(
    manifest: dict[str, Any],
    library_id: str,
    *,
    allowed_categories: set[str] | None = None,
    split_archives_enabled: bool = False,
) -> list[dict[str, str]]:
    """Build Material Agent entries for a SimReady library ID.

    This performs metadata-only selection. It does not download or extract any
    SimReady release archive.
    """
    selection = parse_simready_library_id(library_id)
    categories = _manifest_categories(manifest)
    materials = _manifest_materials(manifest)
    available_categories = _allowed_categories(
        categories,
        allowed_categories,
    )

    if selection.mode == "category":
        assert selection.category is not None
        category = _canonical_category(selection.category, categories)
        if not _category_archive_layout_is_supported(categories[category]):
            raise SimReadyCatalogError(
                f"Split SimReady archives are not supported yet: {category}"
            )
        if category not in available_categories:
            raise SimReadyCatalogError(
                f"SimReady category is not enabled in this deployment: {category}"
            )
        selected = [
            material
            for material in materials
            if str(material.get("category")) == category
        ]
    elif selection.mode == "full":
        selected = [
            material
            for material in materials
            if str(material.get("category")) in available_categories
        ]
    elif selection.mode == "light":
        libraries = manifest.get("libraries")
        if not isinstance(libraries, dict):
            raise SimReadyCatalogError("SimReady manifest is missing library views")
        light = libraries.get(SIMREADY_LIGHT_ID)
        if not isinstance(light, dict):
            raise SimReadyCatalogError("SimReady manifest is missing simready-light")
        material_ids = light.get("material_ids")
        if not isinstance(material_ids, list):
            raise SimReadyCatalogError("simready-light material_ids is not a list")
        requested_ids = {str(item) for item in material_ids}
        selected = [
            material
            for material in materials
            if str(material.get("id")) in requested_ids
            and str(material.get("category")) in available_categories
        ]
    else:
        raise SimReadyCatalogError(f"Unsupported SimReady mode: {selection.mode}")

    return [_material_entry(material) for material in selected]


def category_names(manifest: dict[str, Any]) -> list[str]:
    """Return sorted category names from a SimReady manifest."""
    return sorted(_manifest_categories(manifest))
