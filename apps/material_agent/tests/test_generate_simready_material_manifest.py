# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SimReady manifest generation helper."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_manifest_generator() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "generate_simready_material_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "generate_simready_material_manifest",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_manifest_generator()


def _write_material_zip(path: Path, material_paths: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for material_path in material_paths:
            zf.writestr(material_path, "#usda 1.0\n")


def test_generate_manifest_scores_light_subset_and_archive_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_assets = tmp_path / "release-assets"
    release_assets.mkdir()
    archive_path = release_assets / "Metal.zip"
    _write_material_zip(
        archive_path,
        [
            "Materials/Metal/Steel_Brushed.usda",
            "Materials/Metal/Aluminum_Smooth.usda",
            "Materials/Metal/standardized_surface_finish_v_noise.usda",
        ],
    )
    monkeypatch.setattr(GENERATOR, "LIGHT_CATEGORY_BUDGETS", {"Metal": 2})

    manifest = GENERATOR.generate_manifest(release_assets, "v-test")

    archive = manifest["categories"]["Metal"]["archive_files"][0]
    assert archive["name"] == "Metal.zip"
    assert archive["size"] == archive_path.stat().st_size
    assert archive["sha256"] == GENERATOR._sha256(archive_path)
    assert manifest["categories"]["Metal"]["material_count"] == 3

    selected_ids = set(manifest["libraries"]["simready-light"]["material_ids"])
    selected_names = {
        material["name"]
        for material in manifest["materials"]
        if material["id"] in selected_ids
    }
    assert selected_names == {"Aluminum Smooth", "Steel Brushed"}


def test_generate_manifest_rejects_duplicate_binding_targets(tmp_path: Path) -> None:
    release_assets = tmp_path / "release-assets"
    release_assets.mkdir()
    _write_material_zip(
        release_assets / "Metal.zip",
        [
            "Materials/Metal/Foo-Bar.usda",
            "Materials/Metal/Foo_Bar.usda",
        ],
    )

    with pytest.raises(ValueError, match="Duplicate SimReady binding target"):
        GENERATOR.generate_manifest(release_assets, "v-test")


def test_generate_manifest_rejects_unexpected_category(tmp_path: Path) -> None:
    release_assets = tmp_path / "release-assets"
    release_assets.mkdir()
    _write_material_zip(
        release_assets / "Metal.zip",
        ["Materials/Plastic/Plastic_Test.usda"],
    )

    with pytest.raises(ValueError, match="contains unexpected category Plastic"):
        GENERATOR.generate_manifest(release_assets, "v-test")
