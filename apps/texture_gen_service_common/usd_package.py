# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD package helpers for standalone texture-generation services.

The Step1X public service image intentionally does not copy or install the
``world_understanding`` package. Keep the small USDZ extraction surface it needs
in this service-common package so public image health checks can exercise USDZ
paths without widening the dependency graph.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

DEFAULT_ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_USDZ_MEMBER_BYTES = 512 * 1024 * 1024
USD_LAYER_EXTENSIONS = frozenset({".usd", ".usda", ".usdc"})
USD_PACKAGE_EXTENSIONS = USD_LAYER_EXTENSIONS | {".usdz"}
USD_TEXTURE_EXTENSIONS = frozenset(
    {".bmp", ".exr", ".hdr", ".jpg", ".jpeg", ".png", ".tga", ".tif", ".tiff"}
)

_S_IFMT_MASK = 0xF000
_S_IFLNK = 0xA000


class ArchiveSizeLimitExceeded(ValueError):  # noqa: N818
    """Raised when an archive member exceeds an actual streamed-byte limit."""

    def __init__(self, *, max_bytes: int, attempted_bytes: int) -> None:
        super().__init__(f"Archive member exceeded {max_bytes} bytes while extracting.")
        self.max_bytes = max_bytes
        self.attempted_bytes = attempted_bytes


def copy_stream_limited(
    src: BinaryIO,
    dst: BinaryIO,
    *,
    max_bytes: int,
    chunk_size: int = DEFAULT_ARCHIVE_COPY_CHUNK_BYTES,
) -> int:
    """Copy a binary stream while enforcing an actual-byte limit."""
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    copied = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            return copied
        next_copied = copied + len(chunk)
        if next_copied > max_bytes:
            raise ArchiveSizeLimitExceeded(
                max_bytes=max_bytes,
                attempted_bytes=next_copied,
            )
        dst.write(chunk)
        copied = next_copied


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


# Keep this parser intentionally equivalent to
# world_understanding.utils.usd.package.split_package_member_asset_path. The
# public Step1X image copies only service-common code, so importing the shared
# package here would widen the scanned dependency surface.
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
                    return copy_stream_limited(src, dst, max_bytes=max_bytes)
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


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & _S_IFMT_MASK) == _S_IFLNK
