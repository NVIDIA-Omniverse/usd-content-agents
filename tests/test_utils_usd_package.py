# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import shutil
import struct
import zipfile
from pathlib import Path
from typing import BinaryIO, Literal

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
    validate_usdz_package_layout,
    validate_usdz_package_tree,
    write_usdz_package_from_directory,
)

_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_CENTRAL_HEADER_SIZE = 46
_ZIP_EOCD_SIZE = 22


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


def test_write_usdz_package_from_directory_is_complete_and_canonical(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "textures").mkdir(parents=True)
    (source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source / "sublayer.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source / "textures" / "base.png").write_bytes(b"not really a png")
    output = tmp_path / "asset.usdz"
    expected_order = (
        "root.usda",
        "textures/base.png",
        "sublayer.usda",
    )

    write_usdz_package_from_directory(
        source,
        Path("root.usda"),
        output,
        member_order=("textures/base.png", "root.usda", "sublayer.usda"),
    )

    assert (
        validate_usdz_package_layout(
            output,
            expected_root_member="root.usda",
            expected_member_order=expected_order,
        )
        == expected_order
    )
    with zipfile.ZipFile(output) as archive:
        assert archive.read("root.usda") == b"#usda 1.0\n"
        assert archive.read("sublayer.usda") == b"#usda 1.0\n"
        assert archive.read("textures/base.png") == b"not really a png"


def test_write_usdz_package_cleans_destination_after_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    root = source / "root.usda"
    root.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "asset.usdz"

    def reject_written_package(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        assert output.is_file()
        raise UsdzPackageError("simulated post-write validation failure")

    monkeypatch.setattr(
        package_utils,
        "validate_usdz_package_layout",
        reject_written_package,
    )

    with pytest.raises(UsdzPackageError, match="post-write validation failure"):
        write_usdz_package_from_directory(source, Path(root.name), output)

    assert not output.exists()


@pytest.mark.parametrize(
    ("root_name", "expected_extra_size"),
    [
        (f"{'r' * 26}.usda", 67),
        (f"{'r' * 29}.usda", 0),
    ],
)
def test_write_usdz_package_handles_alignment_boundary_profiles(
    tmp_path: Path,
    root_name: str,
    expected_extra_size: int,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / root_name).write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "asset.usdz"

    write_usdz_package_from_directory(source, Path(root_name), output)

    assert validate_usdz_package_layout(output) == (root_name,)
    with zipfile.ZipFile(output) as archive:
        assert len(archive.getinfo(root_name).extra) == expected_extra_size


@pytest.mark.parametrize(
    ("member_order", "message"),
    [
        (("root.usda",), "complete file set"),
        (("root.usda", "root.usda", "sidecar.bin"), "complete file set"),
        (("../root.usda", "sidecar.bin"), "unsafe"),
    ],
)
def test_write_usdz_package_rejects_incomplete_or_unsafe_order(
    tmp_path: Path,
    member_order: tuple[str, ...],
    message: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source / "sidecar.bin").write_bytes(b"sidecar")
    output = tmp_path / "asset.usdz"

    with pytest.raises(UsdzPackageError, match=message):
        write_usdz_package_from_directory(
            source,
            Path("root.usda"),
            output,
            member_order=member_order,
        )

    assert not output.exists()


def test_write_usdz_package_rejects_symlink_and_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (source / "link.bin").symlink_to(outside)
    output = tmp_path / "asset.usdz"

    with pytest.raises(UsdzPackageError, match="symlink"):
        write_usdz_package_from_directory(
            source,
            Path("root.usda"),
            output,
        )
    assert not output.exists()

    (source / "link.bin").unlink()
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        write_usdz_package_from_directory(
            source,
            Path("root.usda"),
            output,
        )
    assert output.read_bytes() == b"keep"


def test_validate_usdz_package_layout_rejects_noncanonical_archive(
    tmp_path: Path,
) -> None:
    compressed = tmp_path / "compressed.usdz"
    with zipfile.ZipFile(
        compressed,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("root.usda", b"#usda 1.0\n")
    with pytest.raises(UsdzPackageError, match="compressed|64-byte aligned"):
        validate_usdz_package_layout(compressed)

    unaligned = tmp_path / "unaligned.usdz"
    with zipfile.ZipFile(
        unaligned,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("root.usda", b"#usda 1.0\n")
    with pytest.raises(UsdzPackageError, match="64-byte aligned"):
        validate_usdz_package_layout(unaligned)

    symlink = tmp_path / "symlink.usdz"
    with zipfile.ZipFile(
        symlink,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        info = zipfile.ZipInfo("root.usda")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"#usda 1.0\n")
    with pytest.raises(UsdzPackageError, match="regular files"):
        validate_usdz_package_layout(symlink)


def _write_canonical_validation_package(tmp_path: Path) -> Path:
    source = tmp_path / "canonical-source"
    source.mkdir()
    (source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source / "payload.bin").write_bytes(b"payload bytes")
    package = tmp_path / "canonical.usdz"
    write_usdz_package_from_directory(
        source,
        Path("root.usda"),
        package,
        member_order=("root.usda", "payload.bin"),
    )
    return package


def test_package_root_final_recheck_preserves_usdz_errors(tmp_path: Path) -> None:
    package = _write_canonical_validation_package(tmp_path)
    manifest = validate_usdz_package_tree(package)
    package.write_bytes(package.read_bytes() + b"mutated")

    with pytest.raises(
        UsdzPackageError,
        match="root changed before its final recheck",
    ):
        package_utils._require_package_root_matches_manifest(package, manifest)


def _zip_record_offsets(
    archive_bytes: bytes | bytearray,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    eocd_offset = bytes(archive_bytes).rfind(b"PK\x05\x06")
    assert eocd_offset == len(archive_bytes) - _ZIP_EOCD_SIZE
    entry_count = struct.unpack_from("<H", archive_bytes, eocd_offset + 10)[0]
    central_offset = struct.unpack_from("<I", archive_bytes, eocd_offset + 16)[0]
    central_offsets: list[int] = []
    cursor = central_offset
    for _ in range(entry_count):
        assert archive_bytes[cursor : cursor + 4] == b"PK\x01\x02"
        filename_size, extra_size, comment_size = struct.unpack_from(
            "<HHH",
            archive_bytes,
            cursor + 28,
        )
        central_offsets.append(cursor)
        cursor += _ZIP_CENTRAL_HEADER_SIZE + filename_size + extra_size + comment_size
    assert cursor == eocd_offset
    local_offsets = tuple(
        struct.unpack_from("<I", archive_bytes, offset + 42)[0]
        for offset in central_offsets
    )
    return eocd_offset, tuple(central_offsets), local_offsets


def _write_raw_mutation(tmp_path: Path, name: str, archive_bytes: bytearray) -> Path:
    package = tmp_path / name
    package.write_bytes(archive_bytes)
    return package


def test_validate_usdz_package_layout_rejects_local_filename_drift(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, _, local_offsets = _zip_record_offsets(archive_bytes)
    filename_offset = local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE
    archive_bytes[filename_offset] ^= 0x01
    mutated = _write_raw_mutation(tmp_path, "local-name.usdz", archive_bytes)

    with pytest.raises(
        UsdzPackageError,
        match="local and central filename fields differ",
    ):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_wraps_invalid_utf8_metadata(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_name_offset = local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE
    central_name_offset = central_offsets[0] + _ZIP_CENTRAL_HEADER_SIZE
    archive_bytes[local_name_offset] = 0xFF
    archive_bytes[central_name_offset] = 0xFF
    local_flags = struct.unpack_from("<H", archive_bytes, local_offsets[0] + 6)[0]
    central_flags = struct.unpack_from(
        "<H",
        archive_bytes,
        central_offsets[0] + 8,
    )[0]
    struct.pack_into(
        "<H",
        archive_bytes,
        local_offsets[0] + 6,
        local_flags | (1 << 11),
    )
    struct.pack_into(
        "<H",
        archive_bytes,
        central_offsets[0] + 8,
        central_flags | (1 << 11),
    )
    mutated = _write_raw_mutation(tmp_path, "invalid-utf8.usdz", archive_bytes)

    with pytest.raises(
        UsdzPackageError,
        match="filename metadata",
    ) as caught:
        validate_usdz_package_layout(mutated)

    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


def test_validate_usdz_package_layout_rejects_ambiguous_cp437_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "non-ascii-source"
    source.mkdir()
    (source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (source / "é.usda").write_text("#usda 1.0\n", encoding="utf-8")
    canonical = tmp_path / "non-ascii.usdz"
    write_usdz_package_from_directory(
        source,
        Path("root.usda"),
        canonical,
        member_order=("root.usda", "é.usda"),
    )
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_flags = struct.unpack_from("<H", archive_bytes, local_offsets[1] + 6)[0]
    central_flags = struct.unpack_from(
        "<H",
        archive_bytes,
        central_offsets[1] + 8,
    )[0]
    assert local_flags & (1 << 11)
    assert central_flags & (1 << 11)
    struct.pack_into(
        "<H",
        archive_bytes,
        local_offsets[1] + 6,
        local_flags & ~(1 << 11),
    )
    struct.pack_into(
        "<H",
        archive_bytes,
        central_offsets[1] + 8,
        central_flags & ~(1 << 11),
    )
    mutated = _write_raw_mutation(tmp_path, "ambiguous-cp437.usdz", archive_bytes)

    with zipfile.ZipFile(mutated) as archive:
        assert archive.infolist()[1].filename != "é.usda"
    with pytest.raises(
        UsdzPackageError,
        match="must set the UTF-8 flag",
    ):
        validate_usdz_package_layout(mutated)


@pytest.mark.parametrize(
    ("field_name", "field_offset", "field_format"),
    [
        ("flags", 6, "<H"),
        ("compression method", 8, "<H"),
        ("CRC", 14, "<I"),
        ("compressed size", 18, "<I"),
        ("uncompressed size", 22, "<I"),
    ],
)
def test_validate_usdz_package_layout_rejects_local_header_field_drift(
    tmp_path: Path,
    field_name: str,
    field_offset: int,
    field_format: str,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, _, local_offsets = _zip_record_offsets(archive_bytes)
    value = struct.unpack_from(
        field_format,
        archive_bytes,
        local_offsets[0] + field_offset,
    )[0]
    struct.pack_into(
        field_format,
        archive_bytes,
        local_offsets[0] + field_offset,
        value ^ (0x0800 if field_name == "flags" else 0x01),
    )
    mutated = _write_raw_mutation(
        tmp_path,
        f"local-{field_name.replace(' ', '-')}.usdz",
        archive_bytes,
    )

    with pytest.raises(
        UsdzPackageError,
        match=rf"local and central {field_name} fields differ",
    ):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_requires_first_local_header_at_byte_zero(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    original = bytearray(canonical.read_bytes())
    eocd_offset, central_offsets, local_offsets = _zip_record_offsets(original)
    archive_bytes = bytearray(b"x") + original
    shifted_eocd_offset = eocd_offset + 1
    central_offset = struct.unpack_from(
        "<I",
        archive_bytes,
        shifted_eocd_offset + 16,
    )[0]
    struct.pack_into(
        "<I",
        archive_bytes,
        shifted_eocd_offset + 16,
        central_offset + 1,
    )
    for central_entry_offset, local_header_offset in zip(
        central_offsets,
        local_offsets,
        strict=True,
    ):
        struct.pack_into(
            "<I",
            archive_bytes,
            central_entry_offset + 1 + 42,
            local_header_offset + 1,
        )
    mutated = _write_raw_mutation(tmp_path, "prefix.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="local file header.*byte 0"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_requires_eocd_at_eof(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    archive_bytes.extend(b"x")
    mutated = _write_raw_mutation(tmp_path, "suffix.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="EOCD must end at EOF"):
        validate_usdz_package_layout(mutated)


def test_zip_end_record_uses_fixed_eof_offset_when_fields_contain_signature(
    tmp_path: Path,
) -> None:
    central_offset = 0x06054B50
    sparse_archive = tmp_path / "sparse-eocd.zip"
    end_record = struct.pack(
        "<4s4H2IH",
        b"PK\x05\x06",
        0,
        0,
        0,
        0,
        0,
        central_offset,
        0,
    )
    with sparse_archive.open("wb") as stream:
        stream.seek(central_offset)
        stream.write(end_record)

    with sparse_archive.open("rb") as stream:
        assert package_utils._read_zip_end_record(stream) == (
            central_offset,
            0,
            0,
            central_offset,
        )


def test_validate_usdz_package_layout_rejects_gap_before_central_directory(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    original = bytearray(canonical.read_bytes())
    eocd_offset, central_offsets, _ = _zip_record_offsets(original)
    central_offset = central_offsets[0]
    archive_bytes = original[:central_offset] + b"\0" + original[central_offset:]
    shifted_eocd_offset = eocd_offset + 1
    struct.pack_into(
        "<I",
        archive_bytes,
        shifted_eocd_offset + 16,
        central_offset + 1,
    )
    mutated = _write_raw_mutation(tmp_path, "central-gap.usdz", archive_bytes)

    with pytest.raises(
        UsdzPackageError,
        match="immediately follow the last member payload",
    ):
        validate_usdz_package_layout(mutated)


def _stored_size_mismatch_bytes(
    original: bytearray,
    payload_delta: int,
) -> tuple[bytearray, int, int]:
    """Resize the last stored member's payload and declared compressed size.

    The archive stays physically contiguous: bytes are inserted or removed
    immediately before the central directory, and the local header, central
    header, and EOCD offsets are all adjusted to match. Only the declared
    compressed size then disagrees with the uncompressed size.
    """

    eocd_offset, central_offsets, local_offsets = _zip_record_offsets(original)
    central_offset = central_offsets[0]
    last_local = local_offsets[-1]
    last_central = central_offsets[-1]
    file_size = struct.unpack_from("<I", original, last_central + 24)[0]
    declared_size = file_size + payload_delta

    if payload_delta > 0:
        archive_bytes = (
            original[:central_offset]
            + bytearray(payload_delta)
            + original[central_offset:]
        )
    else:
        archive_bytes = (
            original[: central_offset + payload_delta] + original[central_offset:]
        )
    struct.pack_into("<I", archive_bytes, last_local + 18, declared_size)
    struct.pack_into(
        "<I",
        archive_bytes,
        last_central + payload_delta + 20,
        declared_size,
    )
    struct.pack_into(
        "<I",
        archive_bytes,
        eocd_offset + payload_delta + 16,
        central_offset + payload_delta,
    )
    return archive_bytes, declared_size, file_size


@pytest.mark.parametrize(
    ("name", "payload_delta"),
    [
        ("stored-oversized.usdz", 4),
        ("stored-undersized.usdz", -4),
    ],
)
def test_validate_usdz_package_layout_rejects_stored_size_mismatch(
    tmp_path: Path,
    name: str,
    payload_delta: int,
) -> None:
    """Stored members must declare equal compressed and uncompressed sizes.

    An oversized declaration hides payload bytes that ``zipfile`` never exposes
    or CRC-checks and that escape the budgets computed from ``file_size``; an
    undersized declaration truncates the member. Both are rejected here as size
    violations rather than reaching the later payload/CRC gate.
    """

    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes, declared_size, file_size = _stored_size_mismatch_bytes(
        bytearray(canonical.read_bytes()),
        payload_delta,
    )
    mutated = _write_raw_mutation(tmp_path, name, archive_bytes)

    with zipfile.ZipFile(mutated) as archive:
        member = archive.infolist()[-1]
        assert member.compress_type == zipfile.ZIP_STORED
        assert member.compress_size == declared_size
        assert member.file_size == file_size

    with pytest.raises(UsdzPackageError, match="stored payload whose sizes differ"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_tree_rejects_nested_stored_size_mismatch(
    tmp_path: Path,
) -> None:
    """The package-tree seam reaches the same stored-size gate as the root."""

    inner_source = tmp_path / "mismatch-inner-source"
    inner_source.mkdir()
    (inner_source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    inner = tmp_path / "mismatch-inner.usdz"
    write_usdz_package_from_directory(inner_source, Path("root.usda"), inner)
    archive_bytes, _declared, _actual = _stored_size_mismatch_bytes(
        bytearray(inner.read_bytes()),
        4,
    )
    inner.write_bytes(archive_bytes)

    outer_source = tmp_path / "mismatch-outer-source"
    outer_source.mkdir()
    (outer_source / "root.usda").write_text(
        "#usda 1.0\n(\n    subLayers = [@inner.usdz@]\n)\n",
        encoding="utf-8",
    )
    shutil.copy2(inner, outer_source / "inner.usdz")
    outer = tmp_path / "mismatch-outer.usdz"
    write_usdz_package_from_directory(
        outer_source,
        Path("root.usda"),
        outer,
        member_order=("root.usda", "inner.usdz"),
    )

    with pytest.raises(UsdzPackageError, match="stored payload whose sizes differ"):
        validate_usdz_package_tree(outer)


def test_aligned_usdz_member_info_requires_an_open_writer(tmp_path: Path) -> None:
    """Alignment padding cannot be computed without the writer's byte offset."""

    archive = zipfile.ZipFile(tmp_path / "closed.usdz", "w")
    archive.close()
    assert archive.fp is None

    with pytest.raises(UsdzPackageError, match="writer is not open"):
        package_utils._aligned_usdz_member_info(archive, "root.usda", 10)


def test_physical_usdz_layout_rejects_gap_after_central_directory(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    original = bytearray(canonical.read_bytes())
    eocd_offset, _, _ = _zip_record_offsets(original)
    archive_bytes = original[:eocd_offset] + b"\0" + original[eocd_offset:]
    with zipfile.ZipFile(canonical) as archive:
        infos = archive.infolist()

    with pytest.raises(
        UsdzPackageError,
        match="central directory must end immediately before the EOCD",
    ):
        package_utils._validate_physical_usdz_layout(
            io.BytesIO(archive_bytes),
            infos,
        )


def test_validate_usdz_package_layout_rejects_overlapping_local_members(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    struct.pack_into(
        "<I",
        archive_bytes,
        central_offsets[1] + 42,
        local_offsets[1] - 1,
    )
    mutated = _write_raw_mutation(tmp_path, "local-overlap.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="gap, overlap, or reordered"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_reordered_central_members(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    original = bytearray(canonical.read_bytes())
    eocd_offset, central_offsets, _ = _zip_record_offsets(original)
    first_record = original[central_offsets[0] : central_offsets[1]]
    second_record = original[central_offsets[1] : eocd_offset]
    original[central_offsets[0] : eocd_offset] = second_record + first_record
    mutated = _write_raw_mutation(tmp_path, "central-reordered.usdz", original)

    with pytest.raises(UsdzPackageError, match="local file header.*byte 0"):
        validate_usdz_package_layout(mutated)


@pytest.mark.parametrize(
    ("name", "flag", "message"),
    [
        ("descriptor.usdz", 1 << 3, "data descriptors"),
        ("encrypted.usdz", 1 << 0, "encryption"),
    ],
)
def test_validate_usdz_package_layout_rejects_noncanonical_member_flags(
    tmp_path: Path,
    name: str,
    flag: int,
    message: str,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_flags = struct.unpack_from("<H", archive_bytes, local_offsets[0] + 6)[0]
    central_flags = struct.unpack_from("<H", archive_bytes, central_offsets[0] + 8)[0]
    struct.pack_into(
        "<H",
        archive_bytes,
        local_offsets[0] + 6,
        local_flags | flag,
    )
    struct.pack_into(
        "<H",
        archive_bytes,
        central_offsets[0] + 8,
        central_flags | flag,
    )
    mutated = _write_raw_mutation(tmp_path, name, archive_bytes)

    with pytest.raises(UsdzPackageError, match=message):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_zip64_extra_fields(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_filename_size, local_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        local_offsets[0] + 26,
    )
    central_filename_size, central_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        central_offsets[0] + 28,
    )
    assert local_extra_size >= 4
    assert central_extra_size >= 4
    struct.pack_into(
        "<H",
        archive_bytes,
        local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE + local_filename_size,
        0x0001,
    )
    struct.pack_into(
        "<H",
        archive_bytes,
        central_offsets[0] + _ZIP_CENTRAL_HEADER_SIZE + central_filename_size,
        0x0001,
    )
    mutated = _write_raw_mutation(tmp_path, "zip64-extra.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="ZIP64"):
        validate_usdz_package_layout(mutated)


@pytest.mark.parametrize(
    ("field_id", "name"),
    [
        (0x9999, "unknown-extra.usdz"),
        (0x7075, "unicode-path-extra.usdz"),
    ],
)
def test_validate_usdz_package_layout_rejects_noncanonical_extra_fields(
    tmp_path: Path,
    field_id: int,
    name: str,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_filename_size, local_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        local_offsets[0] + 26,
    )
    central_filename_size, central_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        central_offsets[0] + 28,
    )
    assert local_extra_size >= 4
    assert central_extra_size >= 4
    struct.pack_into(
        "<H",
        archive_bytes,
        local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE + local_filename_size,
        field_id,
    )
    struct.pack_into(
        "<H",
        archive_bytes,
        central_offsets[0] + _ZIP_CENTRAL_HEADER_SIZE + central_filename_size,
        field_id,
    )
    mutated = _write_raw_mutation(tmp_path, name, archive_bytes)

    with pytest.raises(UsdzPackageError, match="deterministic.*alignment"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_nonzero_alignment_padding(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_filename_size, local_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        local_offsets[0] + 26,
    )
    central_filename_size, central_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        central_offsets[0] + 28,
    )
    assert local_extra_size > 4
    assert central_extra_size == local_extra_size
    local_extra_offset = local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE + local_filename_size
    central_extra_offset = (
        central_offsets[0] + _ZIP_CENTRAL_HEADER_SIZE + central_filename_size
    )
    archive_bytes[local_extra_offset + local_extra_size - 1] = 1
    archive_bytes[central_extra_offset + central_extra_size - 1] = 1
    mutated = _write_raw_mutation(tmp_path, "nonzero-padding.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="zero-filled.*alignment"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_local_central_extra_drift(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, central_offsets, local_offsets = _zip_record_offsets(archive_bytes)
    local_filename_size, local_extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        local_offsets[0] + 26,
    )
    assert local_extra_size > 4
    local_extra_offset = local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE + local_filename_size
    archive_bytes[local_extra_offset + local_extra_size - 1] = 1
    mutated = _write_raw_mutation(tmp_path, "extra-drift.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="local and central extra field"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_archive_comment(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    eocd_offset, _, _ = _zip_record_offsets(archive_bytes)
    struct.pack_into("<H", archive_bytes, eocd_offset + 20, 1)
    archive_bytes.extend(b"x")
    mutated = _write_raw_mutation(tmp_path, "archive-comment.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="comments"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_rejects_member_comment(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    original = bytearray(canonical.read_bytes())
    eocd_offset, central_offsets, _ = _zip_record_offsets(original)
    first_record_end = central_offsets[1]
    archive_bytes = original[:first_record_end] + b"x" + original[first_record_end:]
    struct.pack_into("<H", archive_bytes, central_offsets[0] + 32, 1)
    shifted_eocd_offset = eocd_offset + 1
    central_size = struct.unpack_from(
        "<I",
        archive_bytes,
        shifted_eocd_offset + 12,
    )[0]
    struct.pack_into(
        "<I",
        archive_bytes,
        shifted_eocd_offset + 12,
        central_size + 1,
    )
    mutated = _write_raw_mutation(tmp_path, "member-comment.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="comments"):
        validate_usdz_package_layout(mutated)


def test_validate_usdz_package_layout_reads_payloads_to_verify_crc(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_validation_package(tmp_path)
    archive_bytes = bytearray(canonical.read_bytes())
    _, _, local_offsets = _zip_record_offsets(archive_bytes)
    filename_size, extra_size = struct.unpack_from(
        "<HH",
        archive_bytes,
        local_offsets[0] + 26,
    )
    payload_offset = (
        local_offsets[0] + _ZIP_LOCAL_HEADER_SIZE + filename_size + extra_size
    )
    archive_bytes[payload_offset] ^= 0x01
    mutated = _write_raw_mutation(tmp_path, "payload-crc.usdz", archive_bytes)

    with pytest.raises(UsdzPackageError, match="CRC verification"):
        validate_usdz_package_layout(mutated)


def _write_nested_validation_package(
    tmp_path: Path,
    *,
    nested_depth: int,
) -> Path:
    inner_package: Path | None = None
    for depth in range(nested_depth, -1, -1):
        source = tmp_path / f"nested-source-{depth}"
        source.mkdir()
        root = source / "root.usda"
        if inner_package is None:
            root.write_text("#usda 1.0\n", encoding="utf-8")
            order = ("root.usda",)
        else:
            root.write_text(
                "#usda 1.0\n(\n    subLayers = [@inner.usdz@]\n)\n",
                encoding="utf-8",
            )
            shutil.copy2(inner_package, source / "inner.usdz")
            order = ("root.usda", "inner.usdz")
        package = tmp_path / f"nested-{depth}.usdz"
        write_usdz_package_from_directory(
            source,
            Path("root.usda"),
            package,
            member_order=order,
        )
        inner_package = package
    assert inner_package is not None
    return inner_package


def test_snapshot_usdz_package_root_cleans_partial_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "asset.usdz"
    package.write_bytes(b"package bytes")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fail_copy(
        _source: BinaryIO,
        destination: BinaryIO,
        *,
        max_bytes: int,
    ) -> tuple[int, str]:
        assert max_bytes >= len(b"package bytes")
        destination.write(b"partial")
        raise OSError("simulated snapshot read failure")

    monkeypatch.setattr(
        package_utils,
        "_copy_package_stream_limited_with_sha256",
        fail_copy,
    )

    with pytest.raises(UsdzPackageError, match="Could not snapshot"):
        package_utils._snapshot_usdz_package_root(
            package,
            workspace=workspace,
            max_bytes=1024,
        )

    assert not (workspace / "root.usdz").exists()


def test_snapshot_usdz_package_root_preserves_contract_error(
    tmp_path: Path,
) -> None:
    package = tmp_path / "asset.usdz"
    package.write_bytes(b"package bytes")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(UsdzPackageError, match="validation byte budget"):
        package_utils._snapshot_usdz_package_root(
            package,
            workspace=workspace,
            max_bytes=0,
        )


def test_package_tree_payload_budget_excludes_root_archive_overhead(
    tmp_path: Path,
) -> None:
    package = _write_canonical_validation_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        infos = archive.infolist()
        payload_bytes = sum(info.file_size for info in infos)
        largest_member_bytes = max(info.file_size for info in infos)
    root_archive_bytes = package.stat().st_size
    assert root_archive_bytes > payload_bytes

    manifest = validate_usdz_package_tree(
        package,
        max_validated_bytes=payload_bytes,
        max_tree_scan_bytes=2 * root_archive_bytes,
    )

    assert manifest.validated_bytes == payload_bytes
    assert manifest.root_size_bytes == root_archive_bytes
    assert manifest.tree_scan_bytes == 2 * root_archive_bytes

    with pytest.raises(UsdzPackageError, match="payload validation budget"):
        validate_usdz_package_tree(
            package,
            max_validated_bytes=payload_bytes - 1,
            max_tree_scan_bytes=2 * root_archive_bytes,
        )
    with pytest.raises(UsdzPackageError, match="aggregate tree-scan budget"):
        validate_usdz_package_tree(
            package,
            max_validated_bytes=payload_bytes,
            max_tree_scan_bytes=(2 * root_archive_bytes) - 1,
        )
    with pytest.raises(UsdzPackageError, match="per-member validation budget"):
        validate_usdz_package_tree(
            package,
            max_validated_bytes=payload_bytes,
            max_tree_scan_bytes=2 * root_archive_bytes,
            max_member_bytes=largest_member_bytes - 1,
        )
    with pytest.raises(UsdzPackageError, match="member validation budget"):
        validate_usdz_package_tree(
            package,
            max_members=len(infos) - 1,
            max_validated_bytes=payload_bytes,
            max_tree_scan_bytes=2 * root_archive_bytes,
        )


def test_package_stream_copy_reports_actual_limit_overflow() -> None:
    destination = io.BytesIO()

    with pytest.raises(ArchiveSizeLimitExceeded) as caught:
        package_utils._copy_package_stream_limited_with_sha256(
            io.BytesIO(b"too large"),
            destination,
            max_bytes=3,
        )

    assert caught.value.max_bytes == 3
    assert caught.value.attempted_bytes == len(b"too large")
    assert destination.getvalue() == b""


@pytest.mark.parametrize(
    ("failure_mode", "message"),
    [
        ("copy_error", "Could not spool nested"),
        ("size_mismatch", "payload size differs"),
    ],
)
def test_validate_usdz_package_tree_cleans_failed_nested_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: Literal["copy_error", "size_mismatch"],
    message: str,
) -> None:
    outer = _write_nested_validation_package(tmp_path, nested_depth=1)
    original_copy = package_utils._copy_package_stream_limited_with_sha256
    original_unlink = Path.unlink
    copy_calls = 0
    spool_cleanup_calls: list[Path] = []

    def adversarial_nested_copy(
        source: BinaryIO,
        destination: BinaryIO,
        *,
        max_bytes: int,
    ) -> tuple[int, str]:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2 and failure_mode == "copy_error":
            destination.write(b"partial")
            raise OSError("simulated nested spool failure")
        copied, digest = original_copy(source, destination, max_bytes=max_bytes)
        if copy_calls == 2 and failure_mode == "size_mismatch":
            return copied - 1, digest
        return copied, digest

    def track_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith("package-"):
            assert path.exists()
            spool_cleanup_calls.append(path)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        package_utils,
        "_copy_package_stream_limited_with_sha256",
        adversarial_nested_copy,
    )
    monkeypatch.setattr(Path, "unlink", track_unlink)

    with pytest.raises(UsdzPackageError, match=message):
        validate_usdz_package_tree(outer)

    assert copy_calls == 2
    assert len(spool_cleanup_calls) == 1
    assert not spool_cleanup_calls[0].exists()


def test_validate_usdz_package_tree_deduplicates_nested_schedule_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _write_nested_validation_package(tmp_path, nested_depth=1)
    original_validation = package_utils._validate_usdz_package_layout_and_infos
    validation_calls = 0

    def duplicate_nested_record(
        path: Path,
        **kwargs: int | None,
    ) -> tuple[tuple[str, ...], tuple[zipfile.ZipInfo, ...], int]:
        nonlocal validation_calls
        validation_calls += 1
        member_names, infos, total_bytes = original_validation(path, **kwargs)
        if validation_calls == 1:
            nested = next(info for info in infos if info.filename == "inner.usdz")
            infos = (*infos, nested)
        return member_names, infos, total_bytes

    monkeypatch.setattr(
        package_utils,
        "_validate_usdz_package_layout_and_infos",
        duplicate_nested_record,
    )

    manifest = validate_usdz_package_tree(outer)

    assert manifest.package_count == 2
    assert manifest.package_chains == frozenset({(), ("inner.usdz",)})


def test_validate_usdz_package_tree_rejects_corrupt_nested_package(
    tmp_path: Path,
) -> None:
    inner_source = tmp_path / "corrupt-inner-source"
    inner_source.mkdir()
    (inner_source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    inner = tmp_path / "corrupt-inner.usdz"
    write_usdz_package_from_directory(
        inner_source,
        Path("root.usda"),
        inner,
    )
    with inner.open("ab") as stream:
        stream.write(b"POLYGLOT-TRAILER")

    outer_source = tmp_path / "corrupt-outer-source"
    outer_source.mkdir()
    (outer_source / "root.usda").write_text(
        "#usda 1.0\n(\n    subLayers = [@inner.usdz@]\n)\n",
        encoding="utf-8",
    )
    shutil.copy2(inner, outer_source / "inner.usdz")
    outer = tmp_path / "corrupt-outer.usdz"
    write_usdz_package_from_directory(
        outer_source,
        Path("root.usda"),
        outer,
        member_order=("root.usda", "inner.usdz"),
    )

    with pytest.raises(UsdzPackageError, match="EOCD must end at EOF"):
        validate_usdz_package_tree(outer)


def test_validate_usdz_package_tree_caches_identical_nested_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _write_nested_validation_package(tmp_path, nested_depth=0)
    outer_source = tmp_path / "cached-outer-source"
    outer_source.mkdir()
    (outer_source / "root.usda").write_text("#usda 1.0\n", encoding="utf-8")
    shutil.copy2(inner, outer_source / "left.usdz")
    shutil.copy2(inner, outer_source / "right.usdz")
    outer = tmp_path / "cached-outer.usdz"
    write_usdz_package_from_directory(
        outer_source,
        Path("root.usda"),
        outer,
        member_order=("root.usda", "left.usdz", "right.usdz"),
    )

    original = package_utils._validate_usdz_package_layout_and_infos
    validated_paths: list[Path] = []

    def counted_validation(
        path: Path,
        **kwargs: int | None,
    ) -> tuple[tuple[str, ...], tuple[zipfile.ZipInfo, ...], int]:
        validated_paths.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(
        package_utils,
        "_validate_usdz_package_layout_and_infos",
        counted_validation,
    )

    manifest = validate_usdz_package_tree(outer, max_packages=3)

    assert len(validated_paths) == 2
    assert manifest.package_count == 3
    assert manifest.root_member_order == ("root.usda", "left.usdz", "right.usdz")
    assert ("left.usdz",) in manifest.package_chains
    assert ("right.usdz",) in manifest.package_chains
    assert ("left.usdz", "root.usda") in manifest.member_chains
    assert ("right.usdz", "root.usda") in manifest.member_chains
    assert manifest.extracted_bytes == 2 * inner.stat().st_size
    assert manifest.tree_scan_bytes == (
        (2 * outer.stat().st_size) + (3 * inner.stat().st_size)
    )
    with pytest.raises(UsdzPackageError, match="package-count budget"):
        validate_usdz_package_tree(outer, max_packages=2)


def test_validate_usdz_package_tree_rejects_depth_before_full_recursion(
    tmp_path: Path,
) -> None:
    outer = _write_nested_validation_package(tmp_path, nested_depth=128)

    with pytest.raises(UsdzPackageError, match="nested-depth budget"):
        validate_usdz_package_tree(outer)
