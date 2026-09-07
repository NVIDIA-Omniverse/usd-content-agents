#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that usd-cli component archives contain only reviewed package files."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
LICENSE_EXPRESSION_HEADER = "License-Expression: Apache-2.0"
LICENSE_FILE_HEADER = "License-File: LICENSE"
MAX_ARCHIVE_FILE_BYTES = 6 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1_000
MAX_ROOT_ARCHIVE_BYTES = 5 * 1024 * 1024
MAX_SERVICE_ARCHIVE_BYTES = 5 * 1024 * 1024

ROOT_SDIST_TOP_LEVEL = frozenset(
    {
        ".gitignore",
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "requirements",
        "src",
    }
)
ROOT_SDIST_REQUIRED = frozenset(
    {
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "requirements/constraints.txt",
        "src/usd_cli/main.py",
        "src/usd_core/session.py",
        "src/usd_server/app.py",
        "src/usd_telemetry/main.py",
    }
)
ROOT_WHEEL_PACKAGES = frozenset({"usd_cli", "usd_core", "usd_server", "usd_telemetry"})
ROOT_WHEEL_REQUIRED = frozenset(
    {
        "usd_cli/main.py",
        "usd_core/session.py",
        "usd_server/app.py",
        "usd_telemetry/main.py",
    }
)

SERVICE_SDIST_TOP_LEVEL = frozenset(
    {".gitignore", "LICENSE", "PKG-INFO", "README.md", "pyproject.toml", "service"}
)
SERVICE_SDIST_REQUIRED = frozenset(
    {
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "service/main.py",
        "service/models.py",
    }
)
SERVICE_WHEEL_PACKAGES = frozenset({"service"})
SERVICE_WHEEL_REQUIRED = frozenset({"service/main.py", "service/models.py"})


class ArtifactValidationError(RuntimeError):
    """A built archive violated the bounded component-package contract."""


@dataclass(frozen=True)
class ArchiveSummary:
    """Small deterministic evidence record for one validated archive."""

    files: int
    uncompressed_bytes: int
    sha256: str


def _safe_parts(name: str) -> tuple[str, ...]:
    if not name or "\0" in name or "\\" in name:
        raise ArtifactValidationError(f"unsafe archive member name: {name!r}")
    normalized_name = name[:-1] if name.endswith("/") else name
    raw_parts = normalized_name.split("/")
    if name.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        raise ArtifactValidationError(f"unsafe archive member path: {name!r}")
    return tuple(raw_parts)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_archive_file_size(path: Path, label: str) -> None:
    archive_bytes = path.stat().st_size
    if archive_bytes > MAX_ARCHIVE_FILE_BYTES:
        raise ArtifactValidationError(
            f"{label} archive file exceeds {MAX_ARCHIVE_FILE_BYTES} bytes"
        )


def _reject_lfs_pointer(label: str, name: str, prefix: bytes) -> None:
    if prefix == LFS_POINTER_PREFIX:
        raise ArtifactValidationError(
            f"{label} contains a Git LFS pointer instead of package bytes: {name}"
        )


def _validate_license_metadata(label: str, payload: bytes) -> None:
    try:
        headers = payload.decode("utf-8").split("\n\n", maxsplit=1)[0].splitlines()
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError(f"{label} package metadata is not UTF-8") from exc
    for required_header in (LICENSE_EXPRESSION_HEADER, LICENSE_FILE_HEADER):
        if required_header not in headers:
            raise ArtifactValidationError(
                f"{label} package metadata is missing {required_header!r}"
            )


def validate_sdist(
    path: Path,
    *,
    label: str,
    allowed_top_level: frozenset[str],
    required_files: frozenset[str],
    max_bytes: int,
    extract_to: Path | None = None,
) -> ArchiveSummary:
    """Validate one Hatch source archive without trusting archive extraction."""

    if not path.is_file():
        raise ArtifactValidationError(f"{label} does not exist: {path}")
    _validate_archive_file_size(path, label)
    if extract_to is not None:
        extract_to.mkdir(parents=True, exist_ok=False)

    seen: set[str] = set()
    archive_roots: set[str] = set()
    file_count = 0
    total_bytes = 0
    with tarfile.open(path, mode="r:gz") as archive:
        for member_count, member in enumerate(archive, start=1):
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise ArtifactValidationError(
                    f"{label} exceeds {MAX_ARCHIVE_MEMBERS} archive members"
                )
            parts = _safe_parts(member.name)
            archive_roots.add(parts[0])
            if len(parts) == 1:
                if not member.isdir():
                    raise ArtifactValidationError(
                        f"{label} has a file outside its archive root: {member.name}"
                    )
                continue
            relative_parts = parts[1:]
            relative_name = PurePosixPath(*relative_parts).as_posix()
            if relative_name in seen:
                raise ArtifactValidationError(
                    f"{label} contains duplicate member: {relative_name}"
                )
            seen.add(relative_name)
            if relative_parts[0] not in allowed_top_level:
                raise ArtifactValidationError(
                    f"{label} contains unreviewed top-level path: {relative_name}"
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise ArtifactValidationError(
                    f"{label} contains a non-regular member: {relative_name}"
                )

            file_count += 1
            total_bytes += member.size
            if total_bytes > max_bytes:
                raise ArtifactValidationError(
                    f"{label} exceeds {max_bytes} uncompressed bytes"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise ArtifactValidationError(
                    f"{label} could not read regular member: {relative_name}"
                )
            with stream:
                prefix = stream.read(len(LFS_POINTER_PREFIX))
                _reject_lfs_pointer(label, relative_name, prefix)
                if extract_to is not None:
                    target = extract_to.joinpath(*relative_parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as destination:
                        destination.write(prefix)
                        shutil.copyfileobj(stream, destination)

    if len(archive_roots) != 1:
        raise ArtifactValidationError(
            f"{label} must have one archive root, found {sorted(archive_roots)}"
        )
    archive_root = next(iter(archive_roots))
    missing = sorted(required_files - seen)
    if missing:
        raise ArtifactValidationError(f"{label} is missing required files: {missing}")
    actual_top_level = {PurePosixPath(name).parts[0] for name in seen}
    if actual_top_level != allowed_top_level:
        raise ArtifactValidationError(
            f"{label} top-level inventory differs: "
            f"expected={sorted(allowed_top_level)}, actual={sorted(actual_top_level)}"
        )
    with tarfile.open(path, mode="r:gz") as archive:
        metadata = archive.extractfile(f"{archive_root}/PKG-INFO")
        if metadata is None:
            raise ArtifactValidationError(f"{label} could not read PKG-INFO")
        with metadata:
            _validate_license_metadata(label, metadata.read())
    return ArchiveSummary(file_count, total_bytes, _sha256(path))


def validate_wheel(
    path: Path,
    *,
    label: str,
    package_roots: frozenset[str],
    required_files: frozenset[str],
    dist_info_prefix: str,
    max_bytes: int,
) -> ArchiveSummary:
    """Validate one wheel's paths, types, size, and payload boundaries."""

    if not path.is_file():
        raise ArtifactValidationError(f"{label} does not exist: {path}")
    _validate_archive_file_size(path, label)

    seen: set[str] = set()
    dist_info_roots: set[str] = set()
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ArtifactValidationError(
                f"{label} exceeds {MAX_ARCHIVE_MEMBERS} archive members"
            )
        for member in members:
            parts = _safe_parts(member.filename)
            member_name = PurePosixPath(*parts).as_posix()
            if member_name in seen:
                raise ArtifactValidationError(
                    f"{label} contains duplicate member: {member_name}"
                )
            seen.add(member_name)
            top_level = parts[0]
            if top_level.endswith(".dist-info"):
                dist_info_roots.add(top_level)
            elif top_level not in package_roots:
                raise ArtifactValidationError(
                    f"{label} contains unreviewed package root: {member_name}"
                )
            if member.is_dir():
                continue
            if member.flag_bits & 0x1:
                raise ArtifactValidationError(
                    f"{label} contains encrypted member: {member_name}"
                )
            mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if file_type not in {0, stat.S_IFREG}:
                raise ArtifactValidationError(
                    f"{label} contains a non-regular member: {member_name}"
                )
            file_count += 1
            total_bytes += member.file_size
            if total_bytes > max_bytes:
                raise ArtifactValidationError(
                    f"{label} exceeds {max_bytes} uncompressed bytes"
                )
            with archive.open(member) as stream:
                _reject_lfs_pointer(
                    label,
                    member_name,
                    stream.read(len(LFS_POINTER_PREFIX)),
                )

    if len(dist_info_roots) != 1:
        raise ArtifactValidationError(
            f"{label} must contain one dist-info root, found {sorted(dist_info_roots)}"
        )
    dist_info_root = next(iter(dist_info_roots))
    if not dist_info_root.startswith(dist_info_prefix) or not dist_info_root.endswith(
        ".dist-info"
    ):
        raise ArtifactValidationError(
            f"{label} has unexpected dist-info root: {dist_info_root}"
        )
    expected_top_level = package_roots | {dist_info_root}
    actual_top_level = {PurePosixPath(name).parts[0] for name in seen}
    if actual_top_level != expected_top_level:
        raise ArtifactValidationError(
            f"{label} package inventory differs: "
            f"expected={sorted(expected_top_level)}, actual={sorted(actual_top_level)}"
        )
    required = required_files | {
        f"{dist_info_root}/METADATA",
        f"{dist_info_root}/RECORD",
        f"{dist_info_root}/WHEEL",
        f"{dist_info_root}/licenses/LICENSE",
    }
    missing = sorted(required - seen)
    if missing:
        raise ArtifactValidationError(f"{label} is missing required files: {missing}")
    with zipfile.ZipFile(path) as archive:
        _validate_license_metadata(
            label,
            archive.read(f"{dist_info_root}/METADATA"),
        )
    return ArchiveSummary(file_count, total_bytes, _sha256(path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-sdist", type=Path, required=True)
    parser.add_argument("--root-wheel", type=Path, required=True)
    parser.add_argument("--service-sdist", type=Path, required=True)
    parser.add_argument("--service-wheel", type=Path, required=True)
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="Safely extract the two validated source archives for content scanning.",
    )
    return parser


def _report(label: str, summary: ArchiveSummary) -> None:
    print(
        f"{label}: files={summary.files}, "
        f"uncompressed_bytes={summary.uncompressed_bytes}, "
        f"sha256={summary.sha256}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.extract_dir is not None:
            args.extract_dir.mkdir(parents=True, exist_ok=False)
        root_sdist = validate_sdist(
            args.root_sdist,
            label="usd-cli sdist",
            allowed_top_level=ROOT_SDIST_TOP_LEVEL,
            required_files=ROOT_SDIST_REQUIRED,
            max_bytes=MAX_ROOT_ARCHIVE_BYTES,
            extract_to=args.extract_dir / "usd-cli" if args.extract_dir else None,
        )
        root_wheel = validate_wheel(
            args.root_wheel,
            label="usd-cli wheel",
            package_roots=ROOT_WHEEL_PACKAGES,
            required_files=ROOT_WHEEL_REQUIRED,
            dist_info_prefix="usd_cli-",
            max_bytes=MAX_ROOT_ARCHIVE_BYTES,
        )
        service_sdist = validate_sdist(
            args.service_sdist,
            label="OVRTX service sdist",
            allowed_top_level=SERVICE_SDIST_TOP_LEVEL,
            required_files=SERVICE_SDIST_REQUIRED,
            max_bytes=MAX_SERVICE_ARCHIVE_BYTES,
            extract_to=args.extract_dir / "ovrtx-service" if args.extract_dir else None,
        )
        service_wheel = validate_wheel(
            args.service_wheel,
            label="OVRTX service wheel",
            package_roots=SERVICE_WHEEL_PACKAGES,
            required_files=SERVICE_WHEEL_REQUIRED,
            dist_info_prefix="ovrtx_rendering_api-",
            max_bytes=MAX_SERVICE_ARCHIVE_BYTES,
        )
    except (
        ArtifactValidationError,
        OSError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"component artifact validation failed: {exc}", file=sys.stderr)
        return 1

    for label, summary in (
        ("usd-cli sdist", root_sdist),
        ("usd-cli wheel", root_wheel),
        ("OVRTX service sdist", service_sdist),
        ("OVRTX service wheel", service_wheel),
    ):
        _report(label, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
