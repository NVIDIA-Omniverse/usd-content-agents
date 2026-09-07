# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared USD package helpers.

This module owns USDZ/package-member semantics that are needed by multiple
agents. Low-level byte-counting lives in :mod:`world_understanding.utils.archive`;
callers here work in terms of USD package roots, safe member paths, and package
asset extraction.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from world_understanding.utils.archive import (
    ArchiveSizeLimitExceeded,
    copy_stream_limited,
)

USD_LAYER_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})
USD_PACKAGE_EXTENSIONS = USD_LAYER_EXTENSIONS | {".usdz"}
USD_TEXTURE_EXTENSIONS = frozenset(
    {".bmp", ".exr", ".hdr", ".jpg", ".jpeg", ".png", ".tga", ".tif", ".tiff"}
)
DEFAULT_MAX_USDZ_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_USDZ_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_USDZ_MEMBERS = 10_000
DEFAULT_MAX_USDZ_PACKAGE_TREE_DEPTH = 8
DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES = DEFAULT_MAX_USDZ_MEMBER_BYTES
DEFAULT_MAX_USDZ_PACKAGE_TREE_MEMBERS = 128
DEFAULT_MAX_USDZ_PACKAGE_TREE_PACKAGES = 64

_S_IFMT_MASK = 0xF000
_S_IFLNK = 0xA000
_S_IFREG = 0x8000

_ZIP_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_ZIP_CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3I5H2I")
_ZIP_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2IH")
_ZIP_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_ZIP_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x05\x06"
_ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x06\x06"
_ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP64_EXTRA_FIELD_ID = 0x0001
_ZIP64_VERSION = 45
_ZIP64_UINT16_SENTINEL = 0xFFFF
_ZIP64_UINT32_SENTINEL = 0xFFFFFFFF
_ZIP_DATA_DESCRIPTOR_FLAG = 1 << 3
_ZIP_ENCRYPTED_FLAG = 1 << 0
_ZIP_STRONG_ENCRYPTION_FLAG = 1 << 6
_ZIP_UTF8_FILENAME_FLAG = 1 << 11
_ZIP_CRC_READ_CHUNK_BYTES = 1024 * 1024


class UsdzPackageError(ValueError):
    """Raised when a USDZ package cannot be handled safely."""


@dataclass(frozen=True)
class UsdzExtractionStats:
    """Summary returned by bounded USDZ member extraction."""

    extracted_members: int = 0
    extracted_bytes: int = 0
    skipped_members: int = 0
    member_limit_reached: bool = False


@dataclass(frozen=True)
class UsdzPackageTreeManifest:
    """Immutable proof of one bounded, physically canonical USDZ package tree.

    Package chains are relative to ``root_path``. The empty tuple names the
    outer package; each non-empty package chain names nested ``.usdz`` members.
    ``member_chains`` names every regular member at every validated depth.
    """

    root_path: Path
    root_size_bytes: int
    root_sha256: str
    root_device: int
    root_inode: int
    root_mtime_ns: int
    root_ctime_ns: int
    root_member_order: tuple[str, ...]
    package_chains: frozenset[tuple[str, ...]]
    member_chains: frozenset[tuple[str, ...]]
    package_count: int
    member_count: int
    validated_bytes: int
    extracted_bytes: int
    tree_scan_bytes: int


@dataclass(frozen=True)
class RetainedUsdzPackageTree:
    """A package-tree proof plus its private snapshot, valid in one context."""

    manifest: UsdzPackageTreeManifest
    snapshot_path: Path
    _snapshot_state: tuple[int, int, int, int, int, int, int]

    def require_snapshot_unchanged(self) -> None:
        """Require the private snapshot's inode metadata to remain immutable."""

        try:
            observed = os.stat(self.snapshot_path, follow_symlinks=False)
        except OSError as exc:
            raise UsdzPackageError(
                "Retained USDZ package-tree snapshot is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or _package_snapshot_state(observed) != self._snapshot_state
        ):
            raise UsdzPackageError(
                "Retained USDZ package-tree snapshot changed after validation"
            )


@dataclass(frozen=True)
class _ZipCentralDirectoryEntry:
    """Physical fields that must agree with one local ZIP member header."""

    raw_filename: bytes
    filename: str
    flag_bits: int
    compression_method: int
    crc: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    extra: bytes


@dataclass(frozen=True)
class _UsdzNestedPackageContent:
    """One spooled nested package bound to immutable content."""

    member_name: str
    content_key: tuple[int, str]
    spool_path: Path


@dataclass(frozen=True)
class _UsdzPackageContentProof:
    """Cached physical proof for one immutable package payload."""

    member_names: tuple[str, ...]
    member_count: int
    validated_bytes: int
    nested_packages: tuple[_UsdzNestedPackageContent, ...]


def resolve_local_package_path(
    package_ref: str,
    base_dir: Path | None = None,
) -> Path:
    """Resolve a local package path or ``file://`` URI."""
    parsed = urlparse(package_ref)
    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
            return Path(url2pathname(path_text)).expanduser()
        package_path = Path(url2pathname(path_text))
    else:
        package_path = Path(package_ref)
    if not package_path.is_absolute() and base_dir is not None:
        package_path = base_dir / package_path
    return package_path.expanduser().resolve()


def parse_package_member_asset_path(
    asset_path: str,
    *,
    base_dir: Path | None = None,
) -> tuple[Path, str] | None:
    """Parse ``asset.usdz[path/in/package.png]`` into local package/member refs."""
    package_member = split_package_member_asset_path(asset_path)
    if package_member is None:
        return None

    package_ref, member_ref = package_member
    package_path = resolve_local_package_path(package_ref, base_dir)
    member = safe_usdz_member_name(member_ref, allow_leading_slash=True)
    if not package_path.is_file() or member is None:
        return None
    return package_path, member


def split_package_member_asset_path(asset_path: str) -> tuple[str, str] | None:
    """Split ``asset.usdz[member]`` into package and member text."""
    if not asset_path.endswith("]"):
        return None
    for separator_index, char in enumerate(asset_path[:-1]):
        if char != "[":
            continue
        package_ref = asset_path[:separator_index]
        member_ref = asset_path[separator_index + 1 : -1]
        if not package_ref or not member_ref:
            continue
        if Path(urlparse(package_ref).path).suffix.lower() == ".usdz":
            return package_ref, member_ref
    return None


def safe_usdz_member_parts(
    member_name: str,
    *,
    allow_leading_slash: bool = False,
) -> tuple[str, ...] | None:
    """Return normalized package-member path parts, or ``None`` if unsafe."""
    normalized = unquote(member_name).replace("\\", "/")
    if normalized.startswith("/"):
        if not allow_leading_slash:
            return None
        normalized = normalized.lstrip("/")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        return None
    return parts


def safe_usdz_member_name(
    member_name: str,
    *,
    allow_leading_slash: bool = False,
) -> str | None:
    """Return a normalized POSIX package-member path, or ``None`` if unsafe."""
    parts = safe_usdz_member_parts(
        member_name,
        allow_leading_slash=allow_leading_slash,
    )
    if parts is None:
        return None
    return "/".join(parts)


def package_member_cache_name(package_path: Path, *, digest_len: int = 0) -> str:
    """Return a stable filesystem-safe cache directory name for a USDZ package."""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", package_path.stem).strip("._-")
    safe_stem = safe_stem or "package"
    if digest_len <= 0:
        return safe_stem
    digest = hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
    return f"{safe_stem}-{digest[:digest_len]}"


def find_usdz_root_layer(usdz_path: Path) -> Path:
    """Return the first USD layer in package order, which is the USDZ root."""
    try:
        with zipfile.ZipFile(usdz_path) as package:
            for info in package.infolist():
                if info.is_dir():
                    continue
                parts = safe_usdz_member_parts(info.filename)
                if parts is None:
                    continue
                candidate = Path(*parts)
                if candidate.suffix.lower() in USD_LAYER_EXTENSIONS:
                    return candidate
    except zipfile.BadZipFile as exc:
        raise UsdzPackageError(f"Invalid USDZ package: {usdz_path}") from exc
    raise UsdzPackageError(f"USDZ package contains no root USD layer: {usdz_path}")


def extract_usdz_member_to_path(
    package_path: Path,
    member_name: str,
    dest: Path,
    *,
    allowed_suffixes: frozenset[str] | set[str] | None = None,
    max_bytes: int = DEFAULT_MAX_USDZ_MEMBER_BYTES,
    allow_leading_slash: bool = True,
) -> int | None:
    """Extract one safe USDZ member to ``dest``.

    Returns bytes written, ``None`` when the package/member is absent or filtered
    out, and raises on I/O or actual-byte limit failures.
    """
    if not package_path.is_file() or package_path.suffix.lower() != ".usdz":
        return None

    member_path = safe_usdz_member_name(
        member_name,
        allow_leading_slash=allow_leading_slash,
    )
    if member_path is None:
        return None
    if allowed_suffixes is not None:
        suffix = Path(member_path).suffix.lower()
        if suffix not in {ext.lower() for ext in allowed_suffixes}:
            return None

    try:
        with zipfile.ZipFile(package_path) as package:
            try:
                info = package.getinfo(member_path)
            except KeyError:
                return None
            if info.is_dir() or _zip_info_is_symlink(info):
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with package.open(info) as src, dest.open("wb") as dst:
                    return copy_stream_limited(
                        cast(BinaryIO, src),
                        dst,
                        max_bytes=max_bytes,
                    )
            except Exception:
                dest.unlink(missing_ok=True)
                raise
    except zipfile.BadZipFile:
        return None


def extract_usdz_member_to_dir(
    package_path: Path,
    member_name: str,
    extract_root: Path,
    *,
    allowed_suffixes: frozenset[str] | set[str] | None = None,
    max_bytes: int = DEFAULT_MAX_USDZ_MEMBER_BYTES,
    allow_leading_slash: bool = True,
) -> Path | None:
    """Extract one USDZ member under ``extract_root`` and return the local path."""
    parts = safe_usdz_member_parts(
        member_name,
        allow_leading_slash=allow_leading_slash,
    )
    if parts is None:
        return None
    if allowed_suffixes is not None:
        suffix = Path(parts[-1]).suffix.lower()
        if suffix not in {ext.lower() for ext in allowed_suffixes}:
            return None
    dest = extract_root.joinpath(*parts)
    if dest.is_file():
        return dest
    written = extract_usdz_member_to_path(
        package_path,
        "/".join(parts),
        dest,
        allowed_suffixes=allowed_suffixes,
        max_bytes=max_bytes,
        allow_leading_slash=False,
    )
    if written is None:
        return None
    return dest if dest.is_file() else None


def extract_usdz_members_to_dir(
    package_path: Path,
    extract_root: Path,
    *,
    allowed_suffixes: frozenset[str] | set[str] | None = None,
    max_members: int = DEFAULT_MAX_USDZ_MEMBERS,
    max_total_bytes: int = DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
    fail_on_filtered_member: bool = False,
) -> UsdzExtractionStats:
    """Boundedly extract USDZ members under ``extract_root``.

    ``fail_on_filtered_member=True`` is intended for edit flows where the full
    package must remain intact. Optional localization flows can keep the default
    and skip unsafe, symlink, unsupported, or oversized members.
    """
    if max_members < 0:
        raise ValueError("max_members must be non-negative")
    if max_total_bytes < 0:
        raise ValueError("max_total_bytes must be non-negative")

    extracted_members = 0
    extracted_bytes = 0
    skipped_members = 0

    try:
        with zipfile.ZipFile(package_path) as package:
            member_infos = package.infolist()
            if fail_on_filtered_member:
                if len(member_infos) > max_members:
                    raise UsdzPackageError(
                        f"USDZ package contains more than {max_members} members."
                    )
                _validate_strict_usdz_member_layout(member_infos)
            for info in member_infos:
                if info.is_dir():
                    if fail_on_filtered_member:
                        parts = safe_usdz_member_parts(info.filename)
                        if parts is None:
                            raise UsdzPackageError(
                                f"USDZ package contains unsafe entry path: {info.filename}"
                            )
                        extract_root.joinpath(*parts).mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                    continue

                parts = safe_usdz_member_parts(info.filename)
                suffix = Path(parts[-1]).suffix.lower() if parts else ""
                filtered = (
                    parts is None
                    or _zip_info_is_symlink(info)
                    or (
                        allowed_suffixes is not None
                        and suffix not in {ext.lower() for ext in allowed_suffixes}
                    )
                )
                if filtered:
                    if fail_on_filtered_member:
                        raise UsdzPackageError(
                            f"USDZ package contains unsupported or unsafe entry: "
                            f"{info.filename}"
                        )
                    skipped_members += 1
                    continue

                if extracted_members >= max_members:
                    if fail_on_filtered_member:
                        raise UsdzPackageError(
                            f"USDZ package contains more than {max_members} members."
                        )
                    return UsdzExtractionStats(
                        extracted_members=extracted_members,
                        extracted_bytes=extracted_bytes,
                        skipped_members=skipped_members,
                        member_limit_reached=True,
                    )

                remaining_bytes = max_total_bytes - extracted_bytes
                if info.file_size > remaining_bytes:
                    if fail_on_filtered_member:
                        raise UsdzPackageError(
                            "USDZ package extracted contents exceed "
                            f"{max_total_bytes} bytes."
                        )
                    skipped_members += 1
                    continue

                assert parts is not None
                dest = extract_root.joinpath(*parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with package.open(info) as src, dest.open("wb") as dst:
                        written = copy_stream_limited(
                            cast(BinaryIO, src),
                            dst,
                            max_bytes=remaining_bytes,
                        )
                except ArchiveSizeLimitExceeded as exc:
                    dest.unlink(missing_ok=True)
                    if fail_on_filtered_member:
                        raise UsdzPackageError(
                            "USDZ package extracted contents exceed "
                            f"{max_total_bytes} bytes."
                        ) from exc
                    skipped_members += 1
                    continue
                except Exception:
                    dest.unlink(missing_ok=True)
                    raise

                extracted_members += 1
                extracted_bytes += written
    except zipfile.BadZipFile as exc:
        raise UsdzPackageError(f"Invalid USDZ package: {package_path}") from exc

    return UsdzExtractionStats(
        extracted_members=extracted_members,
        extracted_bytes=extracted_bytes,
        skipped_members=skipped_members,
    )


def extract_usdz_package_for_edit(
    usdz_path: Path,
    extract_dir: Path,
    *,
    max_members: int = DEFAULT_MAX_USDZ_MEMBERS,
    max_total_bytes: int = DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
) -> Path:
    """Transactionally extract a complete USDZ package for mutation."""
    root_asset = find_usdz_root_layer(usdz_path)
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=extract_dir.parent,
            prefix=f".{extract_dir.name}.stage-",
        )
    )
    try:
        extract_usdz_members_to_dir(
            usdz_path,
            staging_dir,
            allowed_suffixes=None,
            max_members=max_members,
            max_total_bytes=max_total_bytes,
            fail_on_filtered_member=True,
        )

        staged_root = staging_dir / root_asset
        if not staged_root.exists():
            raise UsdzPackageError(
                f"USDZ root layer was not extracted: {root_asset} from {usdz_path}"
            )
        _replace_extract_dir_transactionally(staging_dir, extract_dir)
    finally:
        _remove_extract_artifact(staging_dir)
    return extract_dir / root_asset


def write_usdz_package_from_directory(
    source_dir: Path,
    root_member: Path,
    output_path: Path,
    *,
    member_order: tuple[str, ...] | None = None,
) -> None:
    """Write a complete, uncompressed, 64-byte-aligned USDZ package.

    This is the inverse of :func:`extract_usdz_package_for_edit` for callers
    that must mutate one owned USD layer while retaining every other package
    member. ``source_dir`` must contain only regular files and directories;
    symlinks and file/member aliases fail closed. ``root_member`` is always
    written first, as required by USDZ root-layer semantics. When
    ``member_order`` is supplied it must name the complete file set exactly.

    The destination is created exclusively and removed on any failure. Callers
    should build it in a private directory and use a descriptor-confined atomic
    publication helper for the final destination.
    """

    source_root = source_dir.resolve(strict=True)
    if not source_root.is_dir() or source_dir.is_symlink():
        raise UsdzPackageError(
            f"USDZ package source must be a regular directory: {source_dir}"
        )
    root_parts = safe_usdz_member_parts(root_member.as_posix())
    if root_parts is None:
        raise UsdzPackageError(f"USDZ root member is unsafe: {root_member}")
    root_name = "/".join(root_parts)
    if Path(root_name).suffix.lower() not in USD_LAYER_EXTENSIONS:
        raise UsdzPackageError(f"USDZ root member is not a USD layer: {root_member}")

    discovered: dict[str, Path] = {}
    for candidate in sorted(source_root.rglob("*")):
        if candidate.is_symlink():
            raise UsdzPackageError(
                f"USDZ package source contains a symlink: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise UsdzPackageError(
                f"USDZ package source contains a special entry: {candidate}"
            )
        relative = candidate.relative_to(source_root).as_posix()
        parts = safe_usdz_member_parts(relative)
        if parts is None or "/".join(parts) != relative:
            raise UsdzPackageError(
                f"USDZ package source contains an unsafe member: {relative}"
            )
        discovered[relative] = candidate
    if root_name not in discovered:
        raise UsdzPackageError(
            f"USDZ root member is absent from package source: {root_name}"
        )

    if member_order is None:
        ordered = (root_name, *sorted(set(discovered) - {root_name}))
    else:
        normalized_order: list[str] = []
        for member in member_order:
            parts = safe_usdz_member_parts(member)
            if parts is None:
                raise UsdzPackageError(f"USDZ member order is unsafe: {member}")
            normalized_order.append("/".join(parts))
        if len(normalized_order) != len(set(normalized_order)) or set(
            normalized_order
        ) != set(discovered):
            raise UsdzPackageError(
                "USDZ member order must name the complete file set exactly"
            )
        ordered = (
            root_name,
            *(member for member in normalized_order if member != root_name),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    try:
        with zipfile.ZipFile(
            output_path,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=False,
        ) as archive:
            for member in ordered:
                source = discovered[member]
                info = _aligned_usdz_member_info(
                    archive,
                    member,
                    source.stat(follow_symlinks=False).st_size,
                )
                with (
                    source.open("rb") as input_stream,
                    archive.open(
                        info,
                        mode="w",
                        force_zip64=False,
                    ) as output_stream,
                ):
                    shutil.copyfileobj(input_stream, output_stream)
        validate_usdz_package_layout(
            output_path,
            expected_root_member=root_name,
            expected_member_order=ordered,
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise


def validate_usdz_package_layout(
    usdz_path: Path,
    *,
    expected_root_member: str | None = None,
    expected_member_order: tuple[str, ...] | None = None,
    max_members: int | None = None,
    max_total_bytes: int | None = None,
    max_member_bytes: int | None = None,
) -> tuple[str, ...]:
    """Validate the project-canonical archive layout required for raw authoring.

    Canonical packages contain only safe regular-file members, put the root
    USD layer first, store every payload without compression or encryption and
    with equal compressed and uncompressed sizes so no member can hide
    unaccounted bytes, align every payload with the writer's exact
    zero-filled ``0x1986`` padding,
    encode non-ASCII names as declared UTF-8, and use one contiguous, non-ZIP64
    physical record sequence with no descriptors, comments, prefixes, gaps, or
    trailing bytes. Optional expectations make this suitable as an independent
    post-write completeness check.

    This deliberately strict profile is narrower than every USDZ that OpenUSD
    can read. A standard USDZ carrying ZIP comments or other non-canonical
    physical records must be normalized/repacked before entering this
    raw-authoring path.
    """

    observed_order, _infos, _total_bytes = _validate_usdz_package_layout_and_infos(
        usdz_path,
        max_members=max_members,
        max_total_bytes=max_total_bytes,
        max_member_bytes=max_member_bytes,
    )
    root = observed_order[0]
    if expected_root_member is not None and root != expected_root_member:
        raise UsdzPackageError(
            "USDZ root member differs from the expected root: "
            f"observed={root}, expected={expected_root_member}"
        )
    if expected_member_order is not None and observed_order != expected_member_order:
        raise UsdzPackageError(
            "USDZ member order differs from the complete expected order: "
            f"observed={observed_order}, expected={expected_member_order}"
        )
    return observed_order


def _validate_usdz_package_layout_and_infos(
    usdz_path: Path,
    *,
    max_members: int | None = None,
    max_total_bytes: int | None = None,
    max_member_bytes: int | None = None,
) -> tuple[tuple[str, ...], tuple[zipfile.ZipInfo, ...], int]:
    """Validate one package once and retain detached logical metadata."""

    _validate_optional_package_limit(max_members, name="max_members")
    _validate_optional_package_limit(max_total_bytes, name="max_total_bytes")
    _validate_optional_package_limit(max_member_bytes, name="max_member_bytes")
    try:
        with usdz_path.open("rb") as stream, zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if not infos:
                raise UsdzPackageError(f"USDZ package contains no members: {usdz_path}")
            total_bytes = _validate_usdz_package_resource_limits(
                infos,
                max_members=max_members,
                max_total_bytes=max_total_bytes,
                max_member_bytes=max_member_bytes,
            )
            _validate_strict_usdz_member_layout(infos)
            observed: list[str] = []
            for info in infos:
                if info.is_dir():
                    raise UsdzPackageError(
                        "Canonical USDZ packages must not contain directory "
                        f"entries: {info.filename}"
                    )
                file_type = (info.external_attr >> 16) & _S_IFMT_MASK
                if file_type not in {0, _S_IFREG}:
                    raise UsdzPackageError(
                        "Canonical USDZ packages must contain only regular "
                        f"files: {info.filename}"
                    )
                parts = safe_usdz_member_parts(info.filename)
                if parts is None or "/".join(parts) != info.filename:
                    raise UsdzPackageError(
                        f"USDZ package member is not canonical: {info.filename}"
                    )
                if info.compress_type != zipfile.ZIP_STORED:
                    raise UsdzPackageError(
                        f"USDZ package member is compressed: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise UsdzPackageError(
                        f"USDZ package member uses encryption: {info.filename}"
                    )
                if info.compress_size != info.file_size:
                    raise UsdzPackageError(
                        "USDZ package member declares a stored payload whose "
                        f"sizes differ: {info.filename} has "
                        f"compressed={info.compress_size} bytes and "
                        f"uncompressed={info.file_size} bytes"
                    )
                observed.append(info.filename)
            payload_offsets = _validate_physical_usdz_layout(stream, infos)
            for info, payload_offset in zip(infos, payload_offsets, strict=True):
                if payload_offset % 64:
                    raise UsdzPackageError(
                        "USDZ package member payload is not 64-byte aligned: "
                        f"{info.filename} at offset {payload_offset}"
                    )
            _verify_usdz_member_crcs(archive, infos)
    except zipfile.BadZipFile as exc:
        raise UsdzPackageError(f"Invalid USDZ package: {usdz_path}") from exc
    except UnicodeDecodeError as exc:
        raise UsdzPackageError(
            f"Invalid USDZ package filename metadata: {usdz_path}"
        ) from exc
    except OSError as exc:
        raise UsdzPackageError(f"Could not read USDZ package: {usdz_path}") from exc

    observed_order = tuple(observed)
    root = observed_order[0]
    if Path(root).suffix.lower() not in USD_LAYER_EXTENSIONS:
        raise UsdzPackageError(f"First USDZ member must be the root USD layer: {root}")
    return observed_order, tuple(infos), total_bytes


def _validate_optional_package_limit(value: int | None, *, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_usdz_package_resource_limits(
    infos: list[zipfile.ZipInfo],
    *,
    max_members: int | None,
    max_total_bytes: int | None,
    max_member_bytes: int | None,
) -> int:
    """Fail before payload reads when one package exceeds caller budgets."""

    if max_members is not None and len(infos) > max_members:
        raise UsdzPackageError(
            "USDZ package exceeds the member validation budget: "
            f"observed={len(infos)}, maximum={max_members}"
        )
    total_bytes = 0
    for info in infos:
        if max_member_bytes is not None and info.file_size > max_member_bytes:
            raise UsdzPackageError(
                "USDZ member exceeds the per-member validation budget: "
                f"{info.filename} has {info.file_size} bytes, "
                f"maximum={max_member_bytes}"
            )
        total_bytes += info.file_size
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            raise UsdzPackageError(
                "USDZ package exceeds the payload validation budget: "
                f"observed>{max_total_bytes} bytes"
            )
    return total_bytes


@contextmanager
def retain_usdz_package_tree(
    usdz_path: Path,
    *,
    max_depth: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_DEPTH,
    max_packages: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_PACKAGES,
    max_members: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_MEMBERS,
    max_validated_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_spooled_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_tree_scan_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_member_bytes: int = DEFAULT_MAX_USDZ_MEMBER_BYTES,
) -> Iterator[RetainedUsdzPackageTree]:
    """Prove a bounded tree of physically canonical nested USDZ packages.

    The proof is built before an OpenUSD resolver sees the package. Every
    nested ``.usdz`` member is streamed into a private mode-0700 workspace and
    validated under aggregate depth, package, member, payload, and spool
    budgets. Physical validation is cached by ``(size, SHA-256)`` content
    identity, while all canonical package and member chains are projected for
    each distinct prefix.

    ``tree_scan_bytes`` is the charged content volume for the root snapshot,
    each physically validated unique package, and each nested payload spool.
    It intentionally excludes small ZIP metadata rereads, downstream OpenUSD
    inspection of the retained snapshot, and the one final full source digest
    recheck performed when this context exits.

    The retained object keeps the mode-0400 root snapshot alive for downstream
    resolver work. Its manifest contains no temporary paths and remains bound
    to the caller's source by exact inode metadata plus a final descriptor
    digest recheck.
    """

    _validate_package_tree_limit(max_depth, name="max_depth", minimum=0)
    _validate_package_tree_limit(max_packages, name="max_packages", minimum=1)
    _validate_package_tree_limit(max_members, name="max_members", minimum=1)
    _validate_package_tree_limit(
        max_validated_bytes,
        name="max_validated_bytes",
        minimum=0,
    )
    _validate_package_tree_limit(
        max_spooled_bytes,
        name="max_spooled_bytes",
        minimum=0,
    )
    _validate_package_tree_limit(
        max_tree_scan_bytes,
        name="max_tree_scan_bytes",
        minimum=0,
    )
    _validate_package_tree_limit(
        max_member_bytes,
        name="max_member_bytes",
        minimum=0,
    )

    content_cache: dict[tuple[int, str], _UsdzPackageContentProof] = {}
    spool_paths: dict[tuple[int, str], Path] = {}
    package_chains: set[tuple[str, ...]] = set()
    member_chains: set[tuple[str, ...]] = set()
    scheduled_chains: set[tuple[str, ...]] = {()}
    member_count = 0
    validated_bytes = 0
    extracted_bytes = 0
    tree_scan_bytes = 0
    root_member_order: tuple[str, ...] | None = None

    with tempfile.TemporaryDirectory(prefix=".wu-usdz-tree-") as workspace_text:
        workspace = Path(workspace_text)
        workspace.chmod(0o700)
        if workspace.is_symlink():  # pragma: no cover - tempfile invariant
            raise UsdzPackageError("USDZ package-tree workspace is a symlink")
        (
            root_path,
            root_snapshot,
            root_size_bytes,
            root_sha256,
            root_state,
        ) = _snapshot_usdz_package_root(
            usdz_path,
            workspace=workspace,
            max_bytes=max_tree_scan_bytes,
        )
        tree_scan_bytes += root_size_bytes
        root_content_key = (root_size_bytes, root_sha256)
        pending: deque[tuple[tuple[str, ...], tuple[int, str], Path]] = deque(
            [((), root_content_key, root_snapshot)]
        )

        while pending:
            package_chain, content_key, package_path = pending.popleft()
            content_proof = content_cache.get(content_key)
            remaining_members = max_members - member_count
            remaining_validated_bytes = max_validated_bytes - validated_bytes
            if content_proof is None:
                package_physical_bytes = package_path.stat().st_size
                if package_physical_bytes > max_tree_scan_bytes - tree_scan_bytes:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the aggregate tree-scan "
                        f"budget: maximum={max_tree_scan_bytes}"
                    )
                member_names, infos, package_bytes = (
                    _validate_usdz_package_layout_and_infos(
                        package_path,
                        max_members=remaining_members,
                        max_total_bytes=remaining_validated_bytes,
                        max_member_bytes=max_member_bytes,
                    )
                )
                tree_scan_bytes += package_physical_bytes
                nested_infos = tuple(
                    info
                    for info in infos
                    if Path(info.filename).suffix.lower() == ".usdz"
                )
                if nested_infos and len(package_chain) >= max_depth:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the nested-depth budget: "
                        f"maximum={max_depth}, package_chain={package_chain}"
                    )
                if len(scheduled_chains) + len(nested_infos) > max_packages:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the package-count budget: "
                        f"maximum={max_packages}"
                    )
                nested_size = sum(info.file_size for info in nested_infos)
                if nested_size > max_spooled_bytes - extracted_bytes:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the aggregate spool-byte "
                        f"budget: maximum={max_spooled_bytes}"
                    )
                if nested_size > max_tree_scan_bytes - tree_scan_bytes:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the aggregate tree-scan "
                        f"budget: maximum={max_tree_scan_bytes}"
                    )

                nested_packages: list[_UsdzNestedPackageContent] = []
                try:
                    with zipfile.ZipFile(package_path) as archive:
                        for info in nested_infos:
                            spool_path = workspace / (
                                f"package-{len(spool_paths):08d}-"
                                f"{len(nested_packages):08d}.usdz"
                            )
                            try:
                                with (
                                    archive.open(info) as source,
                                    spool_path.open("xb") as destination,
                                ):
                                    copied, digest = (
                                        _copy_package_stream_limited_with_sha256(
                                            cast(BinaryIO, source),
                                            destination,
                                            max_bytes=min(
                                                max_member_bytes,
                                                max_spooled_bytes - extracted_bytes,
                                                max_tree_scan_bytes - tree_scan_bytes,
                                            ),
                                        )
                                    )
                            except BaseException:
                                spool_path.unlink(missing_ok=True)
                                raise
                            if copied != info.file_size:
                                spool_path.unlink(missing_ok=True)
                                raise UsdzPackageError(
                                    "Nested USDZ payload size differs from its "
                                    f"validated metadata: {info.filename}"
                                )
                            extracted_bytes += copied
                            tree_scan_bytes += copied
                            nested_key = (copied, digest)
                            retained_path = spool_paths.get(nested_key)
                            if retained_path is None:
                                spool_paths[nested_key] = spool_path
                                retained_path = spool_path
                            else:
                                spool_path.unlink()
                            nested_packages.append(
                                _UsdzNestedPackageContent(
                                    member_name=info.filename,
                                    content_key=nested_key,
                                    spool_path=retained_path,
                                )
                            )
                except UsdzPackageError:
                    raise
                except ArchiveSizeLimitExceeded as exc:
                    raise UsdzPackageError(
                        "Nested USDZ member exceeds the aggregate spool-byte budget"
                    ) from exc
                except (
                    EOFError,
                    NotImplementedError,
                    OSError,
                    RuntimeError,
                    UnicodeDecodeError,
                    zipfile.BadZipFile,
                ) as exc:
                    raise UsdzPackageError(
                        f"Could not spool nested USDZ package from {package_path}"
                    ) from exc

                content_proof = _UsdzPackageContentProof(
                    member_names=member_names,
                    member_count=len(member_names),
                    validated_bytes=package_bytes,
                    nested_packages=tuple(nested_packages),
                )
                content_cache[content_key] = content_proof
            else:
                if content_proof.member_count > remaining_members:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the aggregate member budget: "
                        f"maximum={max_members}"
                    )
                if content_proof.validated_bytes > remaining_validated_bytes:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the aggregate validated-byte "
                        f"budget: maximum={max_validated_bytes}"
                    )
                if content_proof.nested_packages and len(package_chain) >= max_depth:
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the nested-depth budget: "
                        f"maximum={max_depth}, package_chain={package_chain}"
                    )
                if (
                    len(scheduled_chains) + len(content_proof.nested_packages)
                    > max_packages
                ):
                    raise UsdzPackageError(
                        "USDZ package tree exceeds the package-count budget: "
                        f"maximum={max_packages}"
                    )

            package_chains.add(package_chain)
            if not package_chain:
                root_member_order = content_proof.member_names
            member_count += content_proof.member_count
            validated_bytes += content_proof.validated_bytes
            member_chains.update(
                package_chain + (member_name,)
                for member_name in content_proof.member_names
            )
            for nested in content_proof.nested_packages:
                nested_chain = package_chain + (nested.member_name,)
                if nested_chain in scheduled_chains:
                    continue
                scheduled_chains.add(nested_chain)
                pending.append(
                    (
                        nested_chain,
                        nested.content_key,
                        nested.spool_path,
                    )
                )

        try:
            final_state = os.stat(root_path, follow_symlinks=False)
        except OSError as exc:
            raise UsdzPackageError(
                f"Could not recheck USDZ package tree root: {root_path}"
            ) from exc
        if _package_root_state(final_state) != root_state:
            raise UsdzPackageError(
                "USDZ package tree root changed while its manifest was built"
            )

        assert root_member_order is not None
        manifest = UsdzPackageTreeManifest(
            root_path=root_path,
            root_size_bytes=root_size_bytes,
            root_sha256=root_sha256,
            root_device=root_state[0],
            root_inode=root_state[1],
            root_mtime_ns=root_state[3],
            root_ctime_ns=root_state[4],
            root_member_order=root_member_order,
            package_chains=frozenset(package_chains),
            member_chains=frozenset(member_chains),
            package_count=len(package_chains),
            member_count=member_count,
            validated_bytes=validated_bytes,
            extracted_bytes=extracted_bytes,
            tree_scan_bytes=tree_scan_bytes,
        )
        root_snapshot.chmod(0o400)
        snapshot_state = _package_snapshot_state(
            os.stat(root_snapshot, follow_symlinks=False)
        )
        primary_error: BaseException | None = None
        try:
            yield RetainedUsdzPackageTree(
                manifest=manifest,
                snapshot_path=root_snapshot,
                _snapshot_state=snapshot_state,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                retained_snapshot = RetainedUsdzPackageTree(
                    manifest=manifest,
                    snapshot_path=root_snapshot,
                    _snapshot_state=snapshot_state,
                )
                retained_snapshot.require_snapshot_unchanged()
                _require_package_root_matches_manifest(root_path, manifest)
            except BaseException as recheck_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Final retained USDZ integrity recheck also failed: "
                    f"{type(recheck_error).__name__}: {recheck_error}"
                )


def validate_usdz_package_tree(
    usdz_path: Path,
    *,
    max_depth: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_DEPTH,
    max_packages: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_PACKAGES,
    max_members: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_MEMBERS,
    max_validated_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_spooled_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_tree_scan_bytes: int = DEFAULT_MAX_USDZ_PACKAGE_TREE_BYTES,
    max_member_bytes: int = DEFAULT_MAX_USDZ_MEMBER_BYTES,
) -> UsdzPackageTreeManifest:
    """Build and close one retained package-tree proof."""

    with retain_usdz_package_tree(
        usdz_path,
        max_depth=max_depth,
        max_packages=max_packages,
        max_members=max_members,
        max_validated_bytes=max_validated_bytes,
        max_spooled_bytes=max_spooled_bytes,
        max_tree_scan_bytes=max_tree_scan_bytes,
        max_member_bytes=max_member_bytes,
    ) as retained:
        return retained.manifest


def _snapshot_usdz_package_root(
    usdz_path: Path,
    *,
    workspace: Path,
    max_bytes: int,
) -> tuple[Path, Path, int, str, tuple[int, int, int, int, int]]:
    """Copy one descriptor-stable regular root into the private workspace."""

    requested = usdz_path.expanduser()
    if requested.suffix.lower() != ".usdz":
        raise UsdzPackageError(
            f"USDZ package tree root must be a .usdz file: {usdz_path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise UsdzPackageError(
            f"Could not open regular USDZ package tree root: {usdz_path}"
        ) from exc

    try:
        with os.fdopen(descriptor, "rb") as source:
            initial_state = os.fstat(source.fileno())
            path_state = os.stat(requested, follow_symlinks=False)
            if (
                not stat.S_ISREG(initial_state.st_mode)
                or not stat.S_ISREG(path_state.st_mode)
                or (initial_state.st_dev, initial_state.st_ino)
                != (path_state.st_dev, path_state.st_ino)
            ):
                raise UsdzPackageError(
                    f"USDZ package tree root must be a regular file: {usdz_path}"
                )
            if initial_state.st_size > max_bytes:
                raise UsdzPackageError(
                    "USDZ package tree root exceeds the validation byte budget: "
                    f"observed={initial_state.st_size}, maximum={max_bytes}"
                )
            root_path = requested.resolve(strict=True)
            snapshot = workspace / "root.usdz"
            try:
                with snapshot.open("xb") as destination:
                    copied, digest = _copy_package_stream_limited_with_sha256(
                        cast(BinaryIO, source),
                        destination,
                        max_bytes=max_bytes,
                    )
            except BaseException:
                snapshot.unlink(missing_ok=True)
                raise
            final_state = os.fstat(source.fileno())
            final_path_state = os.stat(requested, follow_symlinks=False)
            root_state = _package_root_state(initial_state)
            if (
                _package_root_state(final_state) != root_state
                or _package_root_state(final_path_state) != root_state
            ):
                raise UsdzPackageError(
                    "USDZ package tree root changed while its descriptor was copied"
                )
            if copied != initial_state.st_size:
                raise UsdzPackageError(
                    "USDZ package tree root size changed while it was copied"
                )
    except ArchiveSizeLimitExceeded as exc:
        raise UsdzPackageError(
            "USDZ package tree root exceeds the validation byte budget"
        ) from exc
    except UsdzPackageError:
        raise
    except OSError as exc:
        raise UsdzPackageError(
            f"Could not snapshot USDZ package tree root: {usdz_path}"
        ) from exc
    return root_path, snapshot, copied, digest, root_state


def _package_root_state(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _package_snapshot_state(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_package_root_matches_manifest(
    path: Path,
    manifest: UsdzPackageTreeManifest,
) -> None:
    """Perform the one final full source recheck for a retained proof."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            initial_state = os.fstat(source.fileno())
            path_state = os.stat(path, follow_symlinks=False)
            expected_state = (
                manifest.root_device,
                manifest.root_inode,
                manifest.root_size_bytes,
                manifest.root_mtime_ns,
                manifest.root_ctime_ns,
            )
            if (
                _package_root_state(initial_state) != expected_state
                or _package_root_state(path_state) != expected_state
            ):
                raise UsdzPackageError(
                    "USDZ package tree root changed before its final recheck"
                )
            digest = hashlib.sha256()
            while chunk := source.read(_ZIP_CRC_READ_CHUNK_BYTES):
                digest.update(chunk)
            final_state = os.fstat(source.fileno())
            final_path_state = os.stat(path, follow_symlinks=False)
            if (
                _package_root_state(final_state) != expected_state
                or _package_root_state(final_path_state) != expected_state
                or digest.hexdigest() != manifest.root_sha256
            ):
                raise UsdzPackageError(
                    "USDZ package tree root changed during its final recheck"
                )
    except UsdzPackageError:
        raise
    except OSError as exc:
        raise UsdzPackageError(
            f"Could not complete final USDZ package-tree recheck: {path}"
        ) from exc


def _validate_package_tree_limit(value: int, *, name: str, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _copy_package_stream_limited_with_sha256(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    max_bytes: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    copied = 0
    while chunk := source.read(_ZIP_CRC_READ_CHUNK_BYTES):
        attempted = copied + len(chunk)
        if attempted > max_bytes:
            raise ArchiveSizeLimitExceeded(
                max_bytes=max_bytes,
                attempted_bytes=attempted,
            )
        destination.write(chunk)
        digest.update(chunk)
        copied = attempted
    return copied, digest.hexdigest()


def _validate_physical_usdz_layout(
    stream: BinaryIO,
    infos: list[zipfile.ZipInfo],
) -> tuple[int, ...]:
    """Prove one canonical, contiguous physical ZIP record graph."""

    (
        central_directory_offset,
        central_directory_size,
        entry_count,
        end_record_offset,
    ) = _read_zip_end_record(stream)
    if entry_count != len(infos):
        raise UsdzPackageError(
            "USDZ central-directory entry count differs from the logical "
            f"member count: physical={entry_count}, logical={len(infos)}"
        )

    entries = _read_zip_central_directory(
        stream,
        offset=central_directory_offset,
        size=central_directory_size,
        entry_count=entry_count,
    )
    if central_directory_offset + central_directory_size != end_record_offset:
        next_signature = _read_at(
            stream,
            central_directory_offset + central_directory_size,
            4,
            context="bytes after the ZIP central directory",
        )
        if next_signature in {
            _ZIP64_END_OF_CENTRAL_DIRECTORY_SIGNATURE,
            _ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR_SIGNATURE,
        }:
            raise UsdzPackageError("Canonical USDZ packages must not use ZIP64")
        raise UsdzPackageError(
            "USDZ central directory must end immediately before the EOCD record"
        )

    payload_offsets: list[int] = []
    next_local_header_offset = 0
    for index, (entry, info) in enumerate(zip(entries, infos, strict=True)):
        _validate_central_entry_against_zipinfo(entry, info)
        if entry.local_header_offset != next_local_header_offset:
            if index == 0:
                raise UsdzPackageError(
                    "First USDZ local file header must begin at byte 0"
                )
            raise UsdzPackageError(
                "USDZ local members contain a gap, overlap, or reordered entry "
                f"before {entry.filename}: expected offset "
                f"{next_local_header_offset}, observed {entry.local_header_offset}"
            )
        _validate_usdz_alignment_extra(
            entry.extra,
            local_header_offset=entry.local_header_offset,
            filename_size=len(entry.raw_filename),
        )
        payload_offset, next_local_header_offset = _read_zip_local_member(
            stream,
            entry,
        )
        if next_local_header_offset > central_directory_offset:
            raise UsdzPackageError(
                f"USDZ member payload overlaps the central directory: {entry.filename}"
            )
        payload_offsets.append(payload_offset)

    if next_local_header_offset != central_directory_offset:
        raise UsdzPackageError(
            "USDZ central directory must immediately follow the last member "
            f"payload: payload_end={next_local_header_offset}, "
            f"central_directory={central_directory_offset}"
        )
    return tuple(payload_offsets)


def _read_zip_end_record(stream: BinaryIO) -> tuple[int, int, int, int]:
    """Read the standard EOCD and reject comments, trailing bytes, and ZIP64."""

    stream.seek(0, 2)
    archive_size = stream.tell()
    if archive_size < _ZIP_END_OF_CENTRAL_DIRECTORY.size:
        raise UsdzPackageError("USDZ archive has no standard EOCD record")
    end_record_offset = archive_size - _ZIP_END_OF_CENTRAL_DIRECTORY.size
    end_record = _read_at(
        stream,
        end_record_offset,
        _ZIP_END_OF_CENTRAL_DIRECTORY.size,
        context="EOCD record",
    )
    (
        signature,
        disk_number,
        central_directory_disk,
        entries_on_disk,
        entry_count,
        central_directory_size,
        central_directory_offset,
        comment_size,
    ) = _ZIP_END_OF_CENTRAL_DIRECTORY.unpack(end_record)
    if signature != _ZIP_END_OF_CENTRAL_DIRECTORY_SIGNATURE:
        raise UsdzPackageError(
            "Canonical raw-authoring USDZ packages must not have comments or "
            "trailing bytes; the standard EOCD must end at EOF"
        )
    if end_record_offset + _ZIP_END_OF_CENTRAL_DIRECTORY.size + comment_size != (
        archive_size
    ):
        raise UsdzPackageError(
            "USDZ EOCD must end at EOF without undeclared trailing bytes"
        )
    if comment_size:
        raise UsdzPackageError(
            "Canonical raw-authoring USDZ packages must not have comments; "
            "normalize/repack this otherwise readable USDZ first"
        )
    if disk_number != 0 or central_directory_disk != 0:
        raise UsdzPackageError("Canonical USDZ packages must not span ZIP disks")
    if entries_on_disk != entry_count:
        raise UsdzPackageError("Canonical USDZ packages must not span ZIP disks")
    if (
        entries_on_disk == _ZIP64_UINT16_SENTINEL
        or entry_count == _ZIP64_UINT16_SENTINEL
        or central_directory_size == _ZIP64_UINT32_SENTINEL
        or central_directory_offset == _ZIP64_UINT32_SENTINEL
    ):
        raise UsdzPackageError("Canonical USDZ packages must not use ZIP64")
    if central_directory_offset + central_directory_size > end_record_offset:
        raise UsdzPackageError("USDZ central directory overlaps the EOCD record")
    return (
        central_directory_offset,
        central_directory_size,
        entry_count,
        end_record_offset,
    )


def _read_zip_central_directory(
    stream: BinaryIO,
    *,
    offset: int,
    size: int,
    entry_count: int,
) -> tuple[_ZipCentralDirectoryEntry, ...]:
    """Parse each standard central-directory entry without trusting ``zipfile``."""

    entries: list[_ZipCentralDirectoryEntry] = []
    cursor = offset
    limit = offset + size
    for index in range(entry_count):
        header = _read_at(
            stream,
            cursor,
            _ZIP_CENTRAL_DIRECTORY_HEADER.size,
            context=f"central-directory header {index}",
        )
        (
            signature,
            _version_made_by,
            version_needed,
            flag_bits,
            compression_method,
            _modified_time,
            _modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_size,
            extra_size,
            comment_size,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = _ZIP_CENTRAL_DIRECTORY_HEADER.unpack(header)
        if signature != _ZIP_CENTRAL_DIRECTORY_SIGNATURE:
            raise UsdzPackageError(
                f"USDZ central-directory entry {index} has an invalid signature"
            )
        if comment_size:
            raise UsdzPackageError(
                "Canonical raw-authoring USDZ package members must not have "
                "comments; normalize/repack this otherwise readable USDZ first"
            )
        if disk_start != 0:
            raise UsdzPackageError("Canonical USDZ packages must not span ZIP disks")
        _reject_zip64_header_values(
            version_needed=version_needed,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
            disk_start=disk_start,
        )
        _validate_zip_member_flags(flag_bits)

        variable_offset = cursor + _ZIP_CENTRAL_DIRECTORY_HEADER.size
        raw_filename = _read_at(
            stream,
            variable_offset,
            filename_size,
            context=f"central-directory filename {index}",
        )
        extra = _read_at(
            stream,
            variable_offset + filename_size,
            extra_size,
            context=f"central-directory extra field {index}",
        )
        filename = _decode_zip_filename(raw_filename, flag_bits)
        entries.append(
            _ZipCentralDirectoryEntry(
                raw_filename=raw_filename,
                filename=filename,
                flag_bits=flag_bits,
                compression_method=compression_method,
                crc=crc,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
                extra=extra,
            )
        )
        cursor = variable_offset + filename_size + extra_size + comment_size
        if cursor > limit:
            raise UsdzPackageError("USDZ central directory exceeds its declared size")

    if cursor != limit:
        raise UsdzPackageError(
            "USDZ central-directory size does not match its member records"
        )
    return tuple(entries)


def _read_zip_local_member(
    stream: BinaryIO,
    entry: _ZipCentralDirectoryEntry,
) -> tuple[int, int]:
    """Validate one local header and return its payload bounds."""

    header = _read_at(
        stream,
        entry.local_header_offset,
        _ZIP_LOCAL_FILE_HEADER.size,
        context=f"local header for {entry.filename}",
    )
    (
        signature,
        version_needed,
        flag_bits,
        compression_method,
        _modified_time,
        _modified_date,
        crc,
        compressed_size,
        uncompressed_size,
        filename_size,
        extra_size,
    ) = _ZIP_LOCAL_FILE_HEADER.unpack(header)
    if signature != _ZIP_LOCAL_FILE_SIGNATURE:
        raise UsdzPackageError(
            f"USDZ member has an invalid local header: {entry.filename}"
        )
    _reject_zip64_header_values(
        version_needed=version_needed,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
    )
    _validate_zip_member_flags(flag_bits)

    variable_offset = entry.local_header_offset + _ZIP_LOCAL_FILE_HEADER.size
    raw_filename = _read_at(
        stream,
        variable_offset,
        filename_size,
        context=f"local filename for {entry.filename}",
    )
    extra = _read_at(
        stream,
        variable_offset + filename_size,
        extra_size,
        context=f"local extra field for {entry.filename}",
    )
    _validate_zip_extra_fields(extra)

    field_pairs = (
        ("filename", raw_filename, entry.raw_filename),
        ("extra field", extra, entry.extra),
        ("flags", flag_bits, entry.flag_bits),
        ("compression method", compression_method, entry.compression_method),
        ("CRC", crc, entry.crc),
        ("compressed size", compressed_size, entry.compressed_size),
        ("uncompressed size", uncompressed_size, entry.uncompressed_size),
    )
    for field_name, local_value, central_value in field_pairs:
        if local_value != central_value:
            raise UsdzPackageError(
                f"USDZ local and central {field_name} fields differ for "
                f"{entry.filename}"
            )

    payload_offset = variable_offset + filename_size + extra_size
    payload_end = payload_offset + compressed_size
    return payload_offset, payload_end


def _validate_central_entry_against_zipinfo(
    entry: _ZipCentralDirectoryEntry,
    info: zipfile.ZipInfo,
) -> None:
    """Require the independent parser and ``zipfile`` view to agree."""

    field_pairs = (
        ("filename", entry.filename, info.filename),
        ("flags", entry.flag_bits, info.flag_bits),
        ("compression method", entry.compression_method, info.compress_type),
        ("CRC", entry.crc, info.CRC),
        ("compressed size", entry.compressed_size, info.compress_size),
        ("uncompressed size", entry.uncompressed_size, info.file_size),
        ("local-header offset", entry.local_header_offset, info.header_offset),
    )
    for field_name, physical_value, logical_value in field_pairs:
        if physical_value != logical_value:
            raise UsdzPackageError(
                f"USDZ physical and logical {field_name} fields differ for "
                f"{entry.filename}"
            )


def _validate_zip_member_flags(flag_bits: int) -> None:
    if flag_bits & _ZIP_DATA_DESCRIPTOR_FLAG:
        raise UsdzPackageError("Canonical USDZ packages must not use data descriptors")
    if flag_bits & (_ZIP_ENCRYPTED_FLAG | _ZIP_STRONG_ENCRYPTION_FLAG):
        raise UsdzPackageError("Canonical USDZ packages must not use encryption")
    unsupported_flags = flag_bits & ~_ZIP_UTF8_FILENAME_FLAG
    if unsupported_flags:
        raise UsdzPackageError(
            "Canonical USDZ member has unsupported ZIP flags: "
            f"0x{unsupported_flags:04x}"
        )


def _reject_zip64_header_values(
    *,
    version_needed: int,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int | None = None,
    disk_start: int | None = None,
) -> None:
    if (
        version_needed >= _ZIP64_VERSION
        or compressed_size == _ZIP64_UINT32_SENTINEL
        or uncompressed_size == _ZIP64_UINT32_SENTINEL
        or local_header_offset == _ZIP64_UINT32_SENTINEL
        or disk_start == _ZIP64_UINT16_SENTINEL
    ):
        raise UsdzPackageError("Canonical USDZ packages must not use ZIP64")


def _validate_zip_extra_fields(extra: bytes) -> None:
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise UsdzPackageError("USDZ member has a malformed ZIP extra field")
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if field_size > len(extra) - cursor:
            raise UsdzPackageError("USDZ member has a truncated ZIP extra field")
        if field_id == _ZIP64_EXTRA_FIELD_ID:
            raise UsdzPackageError("Canonical USDZ packages must not use ZIP64")
        cursor += field_size


def _validate_usdz_alignment_extra(
    extra: bytes,
    *,
    local_header_offset: int,
    filename_size: int,
) -> None:
    """Require the one deterministic alignment TLV emitted by this project."""

    _validate_zip_extra_fields(extra)
    data_start = local_header_offset + _ZIP_LOCAL_FILE_HEADER.size + filename_size
    padding_size = (-data_start) % 64
    if 0 < padding_size < 4:
        padding_size += 64
    if padding_size:
        payload_size = padding_size - 4
        expected = struct.pack("<HH", 0x1986, payload_size) + (b"\0" * payload_size)
    else:
        expected = b""
    if extra != expected:
        raise UsdzPackageError(
            "Canonical raw-authoring USDZ members must use only deterministic "
            "zero-filled 0x1986 alignment padding so payloads are 64-byte aligned"
        )


def _decode_zip_filename(raw_filename: bytes, flag_bits: int) -> str:
    if not flag_bits & _ZIP_UTF8_FILENAME_FLAG and not raw_filename.isascii():
        raise UsdzPackageError(
            "Canonical raw-authoring USDZ members must set the UTF-8 flag for "
            "non-ASCII filenames"
        )
    encoding = "utf-8" if flag_bits & _ZIP_UTF8_FILENAME_FLAG else "ascii"
    try:
        return raw_filename.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UsdzPackageError(f"USDZ member filename is not valid {encoding}") from exc


def _read_at(
    stream: BinaryIO,
    offset: int,
    size: int,
    *,
    context: str,
) -> bytes:
    stream.seek(offset)
    value = stream.read(size)
    if len(value) != size:
        raise UsdzPackageError(f"USDZ {context} is truncated")
    return value


def _verify_usdz_member_crcs(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
) -> None:
    """Stream every member through ``zipfile`` so CRC checks cannot be skipped."""

    for info in infos:
        bytes_read = 0
        try:
            with archive.open(info) as member:
                while chunk := member.read(_ZIP_CRC_READ_CHUNK_BYTES):
                    bytes_read += len(chunk)
        except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as exc:
            raise UsdzPackageError(
                f"USDZ member failed payload/CRC verification: {info.filename}"
            ) from exc
        if bytes_read != info.file_size:
            raise UsdzPackageError(
                "USDZ member payload size differs from its central-directory "
                f"size: {info.filename}"
            )


def _aligned_usdz_member_info(
    archive: zipfile.ZipFile,
    member: str,
    size_bytes: int,
) -> zipfile.ZipInfo:
    """Return one stored member whose payload begins on a 64-byte boundary."""

    info = zipfile.ZipInfo(member)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = size_bytes
    if archive.fp is None:
        raise UsdzPackageError("USDZ archive writer is not open")
    data_start = (
        archive.fp.tell() + _ZIP_LOCAL_FILE_HEADER.size + len(member.encode("utf-8"))
    )
    padding_size = (-data_start) % 64
    if 0 < padding_size < 4:
        padding_size += 64
    if padding_size:
        payload_size = padding_size - 4
        info.extra = struct.pack("<HH", 0x1986, payload_size) + (b"\0" * payload_size)
    return info


def _validate_strict_usdz_member_layout(
    member_infos: list[zipfile.ZipInfo],
) -> None:
    """Reject archive layouts whose extracted bytes can be ambiguous."""

    normalized_entries: dict[tuple[str, ...], zipfile.ZipInfo] = {}
    for info in member_infos:
        parts = safe_usdz_member_parts(info.filename)
        if parts is None:
            raise UsdzPackageError(
                f"USDZ package contains unsafe entry path: {info.filename}"
            )
        if parts in normalized_entries:
            previous = normalized_entries[parts]
            raise UsdzPackageError(
                "USDZ package contains duplicate normalized member paths: "
                f"{previous.filename} and {info.filename}"
            )
        normalized_entries[parts] = info

    file_paths = {
        parts for parts, info in normalized_entries.items() if not info.is_dir()
    }
    for parts in normalized_entries:
        for depth in range(1, len(parts)):
            ancestor = parts[:depth]
            if ancestor in file_paths:
                raise UsdzPackageError(
                    "USDZ package contains a file/member ancestor collision: "
                    f"{'/'.join(ancestor)} and {'/'.join(parts)}"
                )


def _replace_extract_dir_transactionally(staging_dir: Path, extract_dir: Path) -> None:
    backup_dir: Path | None = None
    backup_path: Path | None = None
    if extract_dir.exists() or extract_dir.is_symlink():
        backup_dir = Path(
            tempfile.mkdtemp(
                dir=extract_dir.parent,
                prefix=f".{extract_dir.name}.rollback-",
            )
        )
        backup_path = backup_dir / "artifact"
        try:
            extract_dir.replace(backup_path)
        except BaseException:
            shutil.rmtree(backup_dir, ignore_errors=True)
            raise

    try:
        staging_dir.replace(extract_dir)
    except BaseException as promotion_error:
        try:
            _remove_extract_artifact(extract_dir)
            if backup_path is not None:
                backup_path.replace(extract_dir)
        except BaseException as rollback_error:  # pragma: no cover - filesystem loss
            backup_location = str(backup_dir) if backup_dir is not None else "none"
            raise RuntimeError(
                "USDZ extraction promotion failed and rollback was incomplete; "
                f"backup remains under {backup_location}: {rollback_error}"
            ) from promotion_error
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _remove_extract_artifact(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & _S_IFMT_MASK) == _S_IFLNK
