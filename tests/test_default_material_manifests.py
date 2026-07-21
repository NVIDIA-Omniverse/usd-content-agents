# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for bundled default material manifests."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pytest
import yaml

Usd = pytest.importorskip("pxr.Usd")
UsdShade = pytest.importorskip("pxr.UsdShade")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest_data(manifest_path: Path) -> tuple[Path, list[dict[str, object]]]:
    raw = cast(
        dict[str, Any], yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    )
    materials = cast(dict[str, Any], raw.get("materials", raw))

    library_file = materials["library_path"]
    assert isinstance(library_file, str)
    entries = materials["entries"]
    assert isinstance(entries, list)
    assert all(isinstance(entry, dict) for entry in entries)

    library_path = manifest_path.parent / library_file
    return library_path, cast(list[dict[str, object]], entries)


def _material_paths(library_path: Path) -> set[str]:
    stage = Usd.Stage.Open(str(library_path))
    assert stage is not None
    return {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdShade.Material)
    }


def _archive_member_path(destination: Path, member_name: str) -> Path:
    member_path = PurePosixPath(member_name)
    assert not member_path.is_absolute()
    assert ".." not in member_path.parts
    return destination.joinpath(*member_path.parts)


def _extract_data_member(
    archive: zipfile.ZipFile,
    member_name: str,
    destination: Path,
) -> Path:
    target_path = _archive_member_path(destination, member_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member_name) as source, target_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return target_path


def _assert_manifest_bindings_resolve_to_materials(manifest_path: Path) -> None:
    library_path, entries = _manifest_data(manifest_path)
    assert library_path.exists()
    assert entries

    stage = Usd.Stage.Open(str(library_path))
    assert stage is not None

    missing: list[str] = []
    for entry in entries:
        name = entry["name"]
        binding = entry["binding"]
        assert isinstance(name, str)
        assert isinstance(binding, str)

        prim = stage.GetPrimAtPath(binding)
        if not prim.IsValid() or not prim.IsA(UsdShade.Material):
            missing.append(f"{name}: {binding}")

    assert missing == []


@pytest.mark.parametrize(
    "manifest_path",
    [
        REPO_ROOT
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml",
        REPO_ROOT / "apps/material_agent_service/materials/default/materials.yaml",
    ],
)
def test_default_material_manifest_bindings_resolve_to_materials(
    manifest_path: Path,
) -> None:
    _assert_manifest_bindings_resolve_to_materials(manifest_path)


def test_service_default_materials_zip_bindings_resolve_to_materials(
    tmp_path: Path,
) -> None:
    service_manifest_path = (
        REPO_ROOT / "apps/material_agent_service/materials/default/materials.yaml"
    )
    zip_path = (
        REPO_ROOT
        / "apps/material_agent_service/materials/default/default_materials.zip"
    )
    with zipfile.ZipFile(zip_path) as archive:
        manifest_members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and Path(name).name == "materials.yaml"
        ]
        assert len(manifest_members) == 1
        for member_name in archive.namelist():
            if not member_name.endswith("/"):
                _extract_data_member(archive, member_name, tmp_path)

    zip_manifest_path = _archive_member_path(tmp_path, manifest_members[0])
    _assert_manifest_bindings_resolve_to_materials(zip_manifest_path)

    service_library_path, service_entries = _manifest_data(service_manifest_path)
    zip_library_path, zip_entries = _manifest_data(zip_manifest_path)
    assert zip_entries == service_entries
    assert _material_paths(zip_library_path) == _material_paths(service_library_path)
