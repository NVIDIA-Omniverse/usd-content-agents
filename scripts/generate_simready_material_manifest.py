#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a lightweight SimReady material manifest from release archives."""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPOSITORY = "NVIDIA-Omniverse/PhysicalAI-SimReady-Materials"
RELEASE_URL_TEMPLATE = (
    "https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials/"
    "releases/tag/{tag}"
)
DOWNLOAD_URL_TEMPLATE = (
    "https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials/"
    "releases/download/{tag}/{asset}"
)
DEFAULT_RELEASE_TAG = "v0.2.0"
LIGHT_CATEGORY_BUDGETS = {
    "Metal": 65,
    "Plastic": 45,
    "Paint": 35,
    "Glass": 30,
    "Concrete": 25,
    "Ceramic": 20,
    "Stone": 15,
    "Ground": 10,
    "Fabric": 10,
    "Paper": 5,
    "Liquids": 5,
}
LIGHT_EXCLUDED_CATEGORIES = {"Leather"}
PRIORITY_TOKENS = (
    "aluminum",
    "steel",
    "iron",
    "copper",
    "brass",
    "nickel",
    "titanium",
    "chrome",
    "zinc",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "clear",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "matte",
    "glossy",
    "brushed",
    "polished",
    "rough",
    "smooth",
    "frosted",
    "painted",
    "concrete",
    "ceramic",
    "stone",
    "asphalt",
    "water",
    "paper",
    "cardboard",
    "fabric",
    "canvas",
)
DEPRIORITY_TOKENS = (
    "standardized_surface_finish_v",
    "optical",
    "starflowers",
    "hexacircles",
    "seaweed",
    "fingerprints",
    "heavy_dirt",
    "dirty_broth",
    "dense",
)


@dataclass(frozen=True)
class ArchiveAsset:
    name: str
    path: Path
    size: int
    digest: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_name(stem: str) -> str:
    return stem.replace("_", " ")


def _stable_id(release_tag: str, category: str, stem: str) -> str:
    raw = f"{release_tag}:{category}:{stem}".lower()
    return re.sub(r"[^a-z0-9]+", "-", raw).strip("-")


def _binding(stem: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    return f"/World/Looks/{safe}"


def _generated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _description(category: str, display_name: str) -> str:
    return f"{display_name} SimReady {category.lower()} material."


def _archive_assets(release_assets: Path) -> dict[str, list[ArchiveAsset]]:
    grouped: dict[str, list[ArchiveAsset]] = {}
    for path in sorted(release_assets.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".zip", ".z01"}:
            continue
        category = path.stem
        grouped.setdefault(category, []).append(
            ArchiveAsset(
                name=path.name,
                path=path,
                size=path.stat().st_size,
                digest=_sha256(path),
            )
        )
    return grouped


def _zip_materials(zip_path: Path) -> list[tuple[str, str, str]]:
    materials: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            match = re.fullmatch(r"Materials/([^/]+)/([^/]+)\.usda", name)
            if not match:
                continue
            category, stem = match.groups()
            materials.append((category, stem, name))
    return materials


def _light_score(stem: str) -> tuple[int, str]:
    normalized = stem.lower()
    priority = sum(1 for token in PRIORITY_TOKENS if token in normalized)
    depriority = sum(1 for token in DEPRIORITY_TOKENS if token in normalized)
    shortness = max(0, 8 - normalized.count("_"))
    return (depriority * 10 - priority * 3 - shortness, normalized)


def _select_light_ids(materials: list[dict[str, Any]]) -> list[str]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for material in materials:
        category = str(material["category"])
        if category in LIGHT_EXCLUDED_CATEGORIES:
            continue
        by_category.setdefault(category, []).append(material)

    selected: list[str] = []
    for category, budget in LIGHT_CATEGORY_BUDGETS.items():
        candidates = sorted(
            by_category.get(category, []),
            key=lambda item: _light_score(str(item["source_stem"])),
        )
        selected.extend(str(item["id"]) for item in candidates[:budget])
    return sorted(selected)


def generate_manifest(release_assets: Path, release_tag: str) -> dict[str, Any]:
    grouped_assets = _archive_assets(release_assets)
    categories: dict[str, dict[str, Any]] = {}
    materials: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}

    for category in sorted(grouped_assets):
        zip_asset = next(
            (
                asset
                for asset in grouped_assets[category]
                if asset.name.endswith(".zip")
            ),
            None,
        )
        if zip_asset is None:
            continue

        parts = sorted(grouped_assets[category], key=lambda item: item.name)
        material_entries = _zip_materials(zip_asset.path)
        split_archive = len(parts) > 1
        categories[category] = {
            "archive_files": [
                {
                    "name": asset.name,
                    "url": DOWNLOAD_URL_TEMPLATE.format(
                        tag=release_tag,
                        asset=asset.name,
                    ),
                    "sha256": asset.digest,
                    "size": asset.size,
                }
                for asset in parts
            ],
            "material_count": len(material_entries),
            "requires_split_archive": split_archive,
        }

        for item_category, stem, source_path in material_entries:
            if item_category != category:
                raise ValueError(
                    f"Archive {zip_asset.name} contains unexpected category "
                    f"{item_category}"
                )
            display = _display_name(stem)
            binding = _binding(stem)
            existing = bindings.get(binding)
            if existing is not None:
                raise ValueError(
                    f"Duplicate SimReady binding target {binding}: "
                    f"{existing} and {category}/{stem}"
                )
            bindings[binding] = f"{category}/{stem}"
            materials.append(
                {
                    "id": _stable_id(release_tag, category, stem),
                    "name": display,
                    "category": category,
                    "source_path": source_path,
                    "source_stem": stem,
                    "binding": binding,
                    "description": _description(category, display),
                }
            )

    materials.sort(key=lambda item: (item["category"], item["name"], item["id"]))
    light_ids = _select_light_ids(materials)

    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "release_tag": release_tag,
        "release_url": RELEASE_URL_TEMPLATE.format(tag=release_tag),
        "generated_at": _generated_at(),
        "categories": categories,
        "libraries": {
            "simready-light": {
                "description": (
                    "Curated common SimReady material subset for Material Agent"
                ),
                "target_count_min": 200,
                "target_count_max": 300,
                "selection": {
                    "type": "budgeted_keyword_score",
                    "category_budgets": LIGHT_CATEGORY_BUDGETS,
                    "excluded_categories": sorted(LIGHT_EXCLUDED_CATEGORIES),
                },
                "material_ids": light_ids,
            },
            "simready-full": {
                "description": "All indexed SimReady materials for this release",
                "material_count": len(materials),
            },
        },
        "materials": materials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-assets",
        type=Path,
        required=True,
        help="Directory containing downloaded release archives.",
    )
    parser.add_argument(
        "--release-tag",
        default=DEFAULT_RELEASE_TAG,
        help="Upstream release tag.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest YAML path.",
    )
    args = parser.parse_args()

    manifest = generate_manifest(args.release_assets, args.release_tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=False)

    light_count = len(manifest["libraries"]["simready-light"]["material_ids"])
    total_count = len(manifest["materials"])
    print(f"Wrote {args.output}")
    print(f"Indexed materials: {total_count}")
    print(f"simready-light materials: {light_count}")


if __name__ == "__main__":
    main()
