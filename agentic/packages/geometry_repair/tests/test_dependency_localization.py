# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for caller-approved USD dependency localization."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("pxr", reason="OpenUSD Python bindings are not installed")

from pxr import Sdf, Usd, UsdGeom, UsdUtils

from geometry_repair.dependency_localization import (
    DependencyRemap,
    DependencyRemapManifest,
    localize_usd_dependencies,
)


def _asset_stage(path: Path, asset_paths: list[str]) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    for index, asset_path in enumerate(asset_paths):
        root.CreateAttribute(f"inputs:file{index}", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(asset_path)
        )
    stage.GetRootLayer().Save()
    return path


def _assert_portable(path: str) -> None:
    parsed = PurePosixPath(path)
    assert not parsed.is_absolute()
    assert ".." not in parsed.parts
    assert "\\" not in path


def test_localizes_recursive_approved_and_exact_dependencies_portably(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    approved = tmp_path / "approved"
    source_dir.mkdir()
    approved.mkdir()
    texture = approved / "albedo.png"
    texture.write_bytes(b"approved texture")
    child = _asset_stage(approved / "provided_child.usda", ["albedo.png"])
    child_before = child.read_bytes()

    source = source_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference("/legacy/missing_child.usda", "/Asset")
    stage.GetRootLayer().Save()
    source_before = source.read_bytes()

    remaps = DependencyRemapManifest(
        remaps=[
            DependencyRemap(
                authored_path="/legacy/missing_child.usda",
                local_path=str(child),
                sha256=hashlib.sha256(child_before).hexdigest(),
            )
        ]
    )
    first = localize_usd_dependencies(
        source,
        tmp_path / "bundle_a",
        approved_roots=[approved],
        remap_manifest=remaps,
    )
    second = localize_usd_dependencies(
        source,
        tmp_path / "bundle_b",
        approved_roots=[approved],
        remap_manifest=remaps,
    )

    assert source.read_bytes() == source_before
    assert child.read_bytes() == child_before
    assert first.source_unchanged is True
    assert first.dependency_complete is True
    assert first.portable_package_complete is True
    assert first.material_fidelity_claimed is False
    assert first.material_fidelity_status == "not_evaluated_dependency_complete"
    assert first.unresolved == []
    assert [item.package_asset_path for item in first.mappings] == [
        item.package_asset_path for item in second.mappings
    ]
    assert [item.source_sha256 for item in first.mappings] == [
        item.source_sha256 for item in second.mappings
    ]
    assert all(item.rewrite_applied for item in first.mappings)
    assert {item.resolution_method for item in first.mappings} == {
        "approved_root",
        "exact_remap",
    }
    for mapping in first.mappings:
        _assert_portable(mapping.package_path)
        _assert_portable(mapping.package_asset_path)
        localized = Path(first.package_root) / mapping.package_path
        assert localized.is_file()
        assert hashlib.sha256(localized.read_bytes()).hexdigest() == mapping.localized_sha256

    localized_stage = Usd.Stage.Open(first.localized_source_path)
    assert localized_stage is not None
    assert localized_stage.GetPrimAtPath("/Asset").IsValid()
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(first.localized_source_path)
    assert list(unresolved) == []


def test_requires_explicit_approval_and_blocks_material_fidelity_when_unresolved(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "adjacent.png").write_bytes(b"not implicitly approved")
    source = _asset_stage(
        source_dir / "scene.usda",
        ["adjacent.png", "https://example.invalid/never-fetch.png"],
    )
    source_before = source.read_bytes()

    report = localize_usd_dependencies(source, tmp_path / "bundle")

    assert source.read_bytes() == source_before
    assert report.mappings == []
    assert report.dependency_complete is False
    assert report.portable_package_complete is False
    assert report.material_fidelity_claimed is False
    assert report.material_fidelity_status == "not_claimed_unresolved_dependencies"
    assert [(item.authored_path, item.reason_code) for item in report.unresolved] == [
        ("adjacent.png", "not_approved"),
        ("https://example.invalid/never-fetch.png", "remote_not_fetched"),
    ]
    evidence = Path(report.evidence_path).read_text(encoding="utf-8")
    assert '"material_fidelity_claimed": false' in evidence
    assert "never-fetch.png" in evidence

    limited = localize_usd_dependencies(
        source,
        tmp_path / "limited_bundle",
        approved_roots=[source_dir],
        max_dependencies=1,
    )
    assert limited.dependency_complete is False
    assert limited.material_fidelity_status == "not_claimed_unresolved_dependencies"
    assert any(item.reason_code == "inventory_incomplete" for item in limited.unresolved)


def test_exact_remote_remap_uses_only_local_file_and_enforces_digest(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _asset_stage(
        source_dir / "scene.usda",
        ["https://example.invalid/albedo.png"],
    )
    replacement = tmp_path / "caller-provided.png"
    replacement.write_bytes(b"local replacement")
    authored = "https://example.invalid/albedo.png"

    complete = localize_usd_dependencies(
        source,
        tmp_path / "complete",
        remap_manifest={
            authored: {
                "local_path": str(replacement),
                "sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
            }
        },
    )
    mismatch = localize_usd_dependencies(
        source,
        tmp_path / "mismatch",
        remap_manifest={
            authored: {
                "local_path": str(replacement),
                "sha256": "0" * 64,
            }
        },
    )

    assert complete.dependency_complete is True
    assert len(complete.mappings) == 1
    assert complete.mappings[0].resolution_method == "exact_remap"
    assert mismatch.dependency_complete is False
    assert mismatch.mappings == []
    assert [(item.authored_path, item.reason_code) for item in mismatch.unresolved] == [
        (authored, "remap_hash_mismatch")
    ]


def test_refuses_traversal_source_symlink_and_destination_symlink_escape(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (approved / "escape.png").symlink_to(outside)
    source = _asset_stage(
        approved / "scene.usda",
        ["../outside.png", "escape.png"],
    )

    report = localize_usd_dependencies(
        source,
        tmp_path / "bundle",
        approved_roots=[approved],
    )

    assert report.mappings == []
    assert [(item.authored_path, item.reason_code) for item in report.unresolved] == [
        ("../outside.png", "traversal_escape"),
        ("escape.png", "symlink_escape"),
    ]

    safe_texture = approved / "safe.png"
    safe_texture.write_bytes(b"safe")
    safe_source = _asset_stage(approved / "safe_scene.usda", ["safe.png"])
    escaped_destination = tmp_path / "escaped_destination"
    escaped_destination.mkdir()
    malicious_bundle = tmp_path / "malicious_bundle"
    malicious_bundle.mkdir()
    (malicious_bundle / "dependencies").symlink_to(escaped_destination, target_is_directory=True)
    with pytest.raises(ValueError, match="destination contains a symlink"):
        localize_usd_dependencies(
            safe_source,
            malicious_bundle,
            approved_roots=[approved],
        )
    assert list(escaped_destination.iterdir()) == []
