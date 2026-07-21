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
import re
import shutil
import tempfile
import zipfile
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

_S_IFMT_MASK = 0xF000
_S_IFLNK = 0xA000


class UsdzPackageError(ValueError):
    """Raised when a USDZ package cannot be handled safely."""


@dataclass(frozen=True)
class UsdzExtractionStats:
    """Summary returned by bounded USDZ member extraction."""

    extracted_members: int = 0
    extracted_bytes: int = 0
    skipped_members: int = 0
    member_limit_reached: bool = False


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
