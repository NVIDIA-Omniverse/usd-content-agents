# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from apps.texture_gen_service_common import usd_package as service_usd_package
from world_understanding.utils.archive import ArchiveSizeLimitExceeded
from world_understanding.utils.usd import package as package_utils
from world_understanding.utils.usd.package import (
    UsdzPackageError,
    extract_usdz_member_to_dir,
    extract_usdz_member_to_path,
    extract_usdz_members_to_dir,
    extract_usdz_package_for_edit,
    find_usdz_root_layer,
    package_member_cache_name,
    parse_package_member_asset_path,
    resolve_local_package_path,
    safe_usdz_member_name,
    safe_usdz_member_parts,
    split_package_member_asset_path,
)


def test_parse_package_member_asset_path_rejects_parent_escape(tmp_path: Path) -> None:
    package = tmp_path / "asset.usdz"
    package.write_bytes(b"not important for path parsing")

    assert (
        parse_package_member_asset_path(
            f"{package}[../escape.png]",
            base_dir=tmp_path,
        )
        is None
    )


def test_parse_package_member_asset_path_supports_brackets(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset[variant].usdz"
    package.write_bytes(b"not important for path parsing")

    assert parse_package_member_asset_path(
        f"{package}[textures/[base].png]",
        base_dir=tmp_path,
    ) == (package.resolve(), "textures/[base].png")


def test_standalone_package_member_splitter_matches_core_parser() -> None:
    cases = [
        "asset.usdz[textures/base.png]",
        "asset[variant].usdz[textures/[base].png]",
        "file:///tmp/asset.usdz[/textures/base.png]",
        "asset.usd[textures/base.png]",
        "asset.usdz[]",
        "asset.usdz[textures/base.png",
    ]

    for asset_path in cases:
        assert service_usd_package.split_package_member_asset_path(
            asset_path
        ) == split_package_member_asset_path(asset_path)


def test_resolve_local_package_path_preserves_file_authority() -> None:
    path = resolve_local_package_path("file://server/share/asset.usdz")

    assert path.as_posix() == "//server/share/asset.usdz"


def test_package_path_and_member_name_edge_cases(tmp_path: Path) -> None:
    package = tmp_path / "my asset!.usdz"
    package.write_bytes(b"placeholder")

    assert resolve_local_package_path(package.as_uri()) == package.resolve()
    assert (
        resolve_local_package_path("relative.usdz", tmp_path)
        == (tmp_path / "relative.usdz").resolve()
    )
    assert parse_package_member_asset_path("relative.usdz", base_dir=tmp_path) is None
    assert safe_usdz_member_parts("/textures/albedo.png") is None
    assert safe_usdz_member_parts(
        "/textures/albedo.png",
        allow_leading_slash=True,
    ) == ("textures", "albedo.png")
    assert safe_usdz_member_name("/textures/albedo.png") is None
    assert package_member_cache_name(package) == "my_asset"
    assert package_member_cache_name(package, digest_len=8).startswith("my_asset-")


def test_find_usdz_root_layer_skips_dirs_and_unsafe_entries(tmp_path: Path) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/", b"")
        archive.writestr("../escape.usda", "#usda 1.0\n")
        archive.writestr("root.usdc", b"usd")

    assert find_usdz_root_layer(package) == Path("root.usdc")

    bad = tmp_path / "bad.usdz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(UsdzPackageError, match="Invalid USDZ"):
        find_usdz_root_layer(bad)

    empty = tmp_path / "empty.usdz"
    with zipfile.ZipFile(empty, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/albedo.png", b"png")
    with pytest.raises(UsdzPackageError, match="contains no root"):
        find_usdz_root_layer(empty)


def test_extract_usdz_member_to_path_filters_absent_and_unsafe_members(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/", b"")
        archive.writestr("textures/albedo.png", b"png")
        symlink = zipfile.ZipInfo("textures/link.png")
        symlink.external_attr = 0xA000 << 16
        archive.writestr(symlink, b"target")

    assert (
        extract_usdz_member_to_path(
            tmp_path / "missing.usdz",
            "textures/albedo.png",
            tmp_path / "out.png",
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package.with_suffix(".zip"),
            "textures/albedo.png",
            tmp_path / "out.png",
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package,
            "../escape.png",
            tmp_path / "out.png",
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package,
            "textures/albedo.png",
            tmp_path / "out.jpg",
            allowed_suffixes={".jpg"},
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package,
            "textures/missing.png",
            tmp_path / "out.png",
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package,
            "textures/",
            tmp_path / "dir.png",
        )
        is None
    )
    assert (
        extract_usdz_member_to_path(
            package,
            "textures/link.png",
            tmp_path / "link.png",
        )
        is None
    )

    bad_zip = tmp_path / "bad.usdz"
    bad_zip.write_bytes(b"not a zip")
    assert (
        extract_usdz_member_to_path(
            bad_zip,
            "textures/albedo.png",
            tmp_path / "bad.png",
        )
        is None
    )


def test_extract_usdz_member_to_dir_enforces_actual_stream_limit(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/albedo.png", b"abc")

    extract_root = tmp_path / "extract"

    with pytest.raises(ArchiveSizeLimitExceeded):
        extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
            max_bytes=2,
        )

    assert not (extract_root / "textures" / "albedo.png").exists()


def test_extract_usdz_member_to_dir_filters_cached_disallowed_suffix(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/albedo.png", b"png")
    extract_root = tmp_path / "extract"
    cached = extract_root / "textures" / "albedo.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    assert (
        extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
            allowed_suffixes={".jpg"},
        )
        is None
    )


def test_extract_usdz_member_to_dir_cache_and_missing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("textures/albedo.png", b"png")
    extract_root = tmp_path / "extract"
    cached = extract_root / "textures" / "albedo.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    assert extract_usdz_member_to_dir(package, "../escape.png", extract_root) is None
    assert (
        extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
            allowed_suffixes={".png"},
        )
        == cached
    )
    assert (
        extract_usdz_member_to_dir(
            package,
            "textures/missing.png",
            extract_root,
        )
        is None
    )

    cached.unlink()
    monkeypatch.setattr(package_utils, "extract_usdz_member_to_path", lambda *a, **k: 1)
    assert (
        package_utils.extract_usdz_member_to_dir(
            package,
            "textures/albedo.png",
            extract_root,
        )
        is None
    )


def test_extract_usdz_members_to_dir_limits_and_filtered_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("nested/", b"")
        archive.writestr("nested/root.usda", "#usda 1.0\n")
        archive.writestr("textures/albedo.png", b"png")
        archive.writestr("notes.txt", b"text")

    with pytest.raises(ValueError, match="max_members"):
        extract_usdz_members_to_dir(package, tmp_path / "bad", max_members=-1)
    with pytest.raises(ValueError, match="max_total_bytes"):
        extract_usdz_members_to_dir(package, tmp_path / "bad", max_total_bytes=-1)

    stats = extract_usdz_members_to_dir(
        package,
        tmp_path / "filtered",
        allowed_suffixes={".png"},
    )
    assert stats.extracted_members == 1
    assert stats.skipped_members == 2

    limited = extract_usdz_members_to_dir(
        package,
        tmp_path / "limited",
        max_members=1,
    )
    assert limited.member_limit_reached is True
    assert limited.extracted_members == 1

    too_small = extract_usdz_members_to_dir(
        package,
        tmp_path / "too-small",
        max_total_bytes=1,
    )
    assert too_small.skipped_members >= 1

    full = extract_usdz_members_to_dir(
        package,
        tmp_path / "full",
        fail_on_filtered_member=True,
    )
    assert full.extracted_members == 3
    assert (tmp_path / "full" / "nested").is_dir()

    with pytest.raises(UsdzPackageError, match="more than 1 members"):
        extract_usdz_members_to_dir(
            package,
            tmp_path / "strict-limit",
            max_members=1,
            fail_on_filtered_member=True,
        )
    with pytest.raises(UsdzPackageError, match="exceed"):
        extract_usdz_members_to_dir(
            package,
            tmp_path / "strict-bytes",
            max_total_bytes=1,
            fail_on_filtered_member=True,
        )

    def _raise_copy(*args: object, **kwargs: object) -> int:
        raise RuntimeError("copy failed")

    monkeypatch.setattr(package_utils, "copy_stream_limited", _raise_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        extract_usdz_members_to_dir(package, tmp_path / "copy-error")

    bad_zip = tmp_path / "bad.usdz"
    bad_zip.write_bytes(b"not a zip")
    with pytest.raises(UsdzPackageError, match="Invalid USDZ"):
        extract_usdz_members_to_dir(bad_zip, tmp_path / "bad-zip")


def test_extract_usdz_package_for_edit_returns_root_layer(tmp_path: Path) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
        archive.writestr("textures/albedo.png", b"png")

    root = extract_usdz_package_for_edit(package, tmp_path / "extract")

    assert root == tmp_path / "extract" / "root.usda"
    assert root.read_text(encoding="utf-8") == "#usda 1.0\n"
    assert (tmp_path / "extract" / "textures" / "albedo.png").read_bytes() == b"png"


def test_strict_extract_checks_member_limit_before_layout_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
        archive.writestr("texture.png", b"png")

    monkeypatch.setattr(
        package_utils,
        "_validate_strict_usdz_member_layout",
        lambda _infos: pytest.fail("strict layout validation ran after limit failure"),
    )

    with pytest.raises(UsdzPackageError, match="more than 1 members"):
        extract_usdz_members_to_dir(
            package,
            tmp_path / "extract",
            max_members=1,
            fail_on_filtered_member=True,
        )


def test_extract_usdz_package_for_edit_clears_existing_dir(tmp_path: Path) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")

    extract_dir = tmp_path / "extract"
    stale = extract_dir / "textures" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    extract_usdz_package_for_edit(package, extract_dir)

    assert not stale.exists()

    file_extract_dir = tmp_path / "file-extract"
    file_extract_dir.write_text("not a dir", encoding="utf-8")

    extract_usdz_package_for_edit(package, file_extract_dir)

    assert file_extract_dir.is_dir()
    assert (file_extract_dir / "root.usda").is_file()


def test_extract_usdz_package_for_edit_rejects_unsafe_member(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
        archive.writestr("../escape.usda", "#usda 1.0\n")

    extract_dir = tmp_path / "extract"
    with pytest.raises(UsdzPackageError, match="unsafe entry"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert not (tmp_path / "escape.usda").exists()
    assert not extract_dir.exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []


@pytest.mark.parametrize(
    "duplicate_name",
    ["root.usda", "./root.usda", r".\root.usda", "root%2Eusda"],
)
def test_extract_usdz_package_for_edit_rejects_duplicate_normalized_members(
    tmp_path: Path,
    duplicate_name: str,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", '#usda 1.0\ndef Xform "First" {}\n')
        archive.writestr(duplicate_name, '#usda 1.0\ndef Xform "Second" {}\n')

    extract_dir = tmp_path / "extract"
    with pytest.raises(UsdzPackageError, match="duplicate normalized member"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert not extract_dir.exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []


def test_extract_usdz_package_for_edit_rejects_file_ancestor_collision(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
        archive.writestr("textures", b"not a directory")
        archive.writestr("textures/albedo.png", b"png")

    extract_dir = tmp_path / "extract"
    with pytest.raises(UsdzPackageError, match="ancestor collision"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert not extract_dir.exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []


def test_extract_usdz_package_for_edit_preserves_destination_on_late_failure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")
        archive.writestr("../textures/albedo.png", b"unsafe")

    extract_dir = tmp_path / "extract"
    sentinel = extract_dir / "previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("previous extraction", encoding="utf-8")

    with pytest.raises(UsdzPackageError, match="unsafe entry"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert sentinel.read_text(encoding="utf-8") == "previous extraction"
    assert not (extract_dir / "root.usda").exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []
    assert list(tmp_path.glob(".extract.rollback-*")) == []


def test_extract_usdz_package_for_edit_restores_destination_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")

    extract_dir = tmp_path / "extract"
    sentinel = extract_dir / "previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("previous extraction", encoding="utf-8")
    original_replace = Path.replace

    def fail_staged_publish(path: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if path.name.startswith(".extract.stage-") and target_path == extract_dir:
            raise OSError("forced extraction publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_publish)

    with pytest.raises(OSError, match="forced extraction publish failure"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert sentinel.read_text(encoding="utf-8") == "previous extraction"
    assert not (extract_dir / "root.usda").exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []
    assert list(tmp_path.glob(".extract.rollback-*")) == []


def test_extract_usdz_package_for_edit_cleans_failed_backup_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")

    extract_dir = tmp_path / "extract"
    sentinel = extract_dir / "previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("previous extraction", encoding="utf-8")
    original_replace = Path.replace

    def fail_backup(path: Path, target: str | Path) -> Path:
        if path == extract_dir and Path(target).name == "artifact":
            raise OSError("forced extraction backup failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_backup)

    with pytest.raises(OSError, match="forced extraction backup failure"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert sentinel.read_text(encoding="utf-8") == "previous extraction"
    assert list(tmp_path.glob(".extract.stage-*")) == []
    assert list(tmp_path.glob(".extract.rollback-*")) == []


def test_extract_usdz_package_for_edit_rejects_missing_staged_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", "#usda 1.0\n")

    monkeypatch.setattr(
        package_utils,
        "extract_usdz_members_to_dir",
        lambda *args, **kwargs: package_utils.UsdzExtractionStats(),
    )
    extract_dir = tmp_path / "extract"

    with pytest.raises(UsdzPackageError, match="root layer was not extracted"):
        extract_usdz_package_for_edit(package, extract_dir)

    assert not extract_dir.exists()
    assert list(tmp_path.glob(".extract.stage-*")) == []


def test_remove_extract_artifact_handles_file_and_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("artifact", encoding="utf-8")
    directory_path = tmp_path / "artifact-dir"
    directory_path.mkdir()
    (directory_path / "member.txt").write_text("member", encoding="utf-8")

    package_utils._remove_extract_artifact(file_path)
    package_utils._remove_extract_artifact(directory_path)

    assert not file_path.exists()
    assert not directory_path.exists()
