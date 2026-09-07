#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove excluded Git LFS fixture bytes cannot change component artifacts."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LFS_FIXTURE_RELATIVE = Path(
    "sample_assets/A5E52945329_P5Z00052019_no_material_bindings.usd"
)
LFS_FIXTURE = ROOT / LFS_FIXTURE_RELATIVE
SOURCE_FILES = (
    Path(".gitignore"),
    Path("LICENSE"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements/constraints.txt"),
    Path("scripts/check_component_artifacts.py"),
    LFS_FIXTURE_RELATIVE,
)
SOURCE_DIRECTORIES = (Path("src"), Path("apps/ovrtx_rendering_api"))


def _run(*command: str, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _copy_source(destination: Path) -> None:
    destination.mkdir(parents=True)
    for relative_path in SOURCE_FILES:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
    ignored = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    )
    for relative_path in SOURCE_DIRECTORIES:
        shutil.copytree(
            ROOT / relative_path,
            destination / relative_path,
            ignore=ignored,
        )


def _build(source_root: Path, output: Path) -> dict[str, Path]:
    service_root = source_root / "apps/ovrtx_rendering_api"
    root_output = output / "root"
    service_output = output / "service"
    root_output.mkdir(parents=True)
    service_output.mkdir(parents=True)
    _run(
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        "--outdir",
        str(root_output),
        cwd=source_root,
    )
    _run(
        sys.executable,
        "-m",
        "build",
        "--no-isolation",
        str(service_root),
        "--outdir",
        str(service_output),
        cwd=source_root,
    )
    return {
        "root-sdist": _single(root_output, "usd_cli-*.tar.gz"),
        "root-wheel": _single(root_output, "usd_cli-*.whl"),
        "service-sdist": _single(service_output, "ovrtx_rendering_api-*.tar.gz"),
        "service-wheel": _single(service_output, "ovrtx_rendering_api-*.whl"),
    }


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tar_inventory(path: Path) -> tuple[tuple[object, ...], ...]:
    inventory: list[tuple[object, ...]] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            relative_name = (
                PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""
            )
            payload_digest = ""
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"could not read {member.name} from {path}")
                with stream:
                    payload_digest = _digest(stream.read())
            inventory.append(
                (
                    relative_name,
                    member.type,
                    member.mode,
                    member.size,
                    member.linkname,
                    payload_digest,
                )
            )
    return tuple(sorted(inventory, key=lambda entry: str(entry[0])))


def _wheel_inventory(path: Path) -> tuple[tuple[object, ...], ...]:
    inventory: list[tuple[object, ...]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            inventory.append(
                (
                    member.filename,
                    member.external_attr,
                    member.compress_type,
                    member.flag_bits,
                    member.file_size,
                    _digest(archive.read(member)),
                )
            )
    return tuple(sorted(inventory, key=lambda entry: str(entry[0])))


def _inventory(path: Path) -> tuple[tuple[object, ...], ...]:
    if path.suffix == ".whl":
        return _wheel_inventory(path)
    return _tar_inventory(path)


def _validate_artifacts(
    source_root: Path, artifacts: dict[str, Path], extract_dir: Path
) -> None:
    _run(
        sys.executable,
        "scripts/check_component_artifacts.py",
        "--root-sdist",
        str(artifacts["root-sdist"]),
        "--root-wheel",
        str(artifacts["root-wheel"]),
        "--service-sdist",
        str(artifacts["service-sdist"]),
        "--service-wheel",
        str(artifacts["service-wheel"]),
        "--extract-dir",
        str(extract_dir),
        cwd=source_root,
    )


def _git_status() -> bytes:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    attribute = subprocess.run(
        ["git", "check-attr", "filter", "--", LFS_FIXTURE_RELATIVE.as_posix()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not attribute.rstrip().endswith(": lfs"):
        raise RuntimeError(
            f"representative fixture is no longer Git LFS-owned: {attribute}"
        )

    original = LFS_FIXTURE.read_bytes()
    original_digest = _digest(original)
    original_stat = LFS_FIXTURE.stat()
    original_status = _git_status()
    if original.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        materialized = b"excluded-materialized-lfs-payload\n" * 256
    else:
        materialized = original
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{_digest(materialized)}\nsize {len(materialized)}\n".encode()
    )

    with tempfile.TemporaryDirectory(prefix="usd-cli-component-determinism-") as tmp:
        temporary_root = Path(tmp)
        pointer_source = temporary_root / "pointer-source"
        hydrated_source = temporary_root / "hydrated-source"
        _copy_source(pointer_source)
        _copy_source(hydrated_source)
        (pointer_source / LFS_FIXTURE_RELATIVE).write_bytes(pointer)
        (hydrated_source / LFS_FIXTURE_RELATIVE).write_bytes(materialized)

        pointer_artifacts = _build(pointer_source, temporary_root / "pointer")
        _validate_artifacts(
            pointer_source,
            pointer_artifacts,
            temporary_root / "pointer-extracted",
        )
        hydrated_artifacts = _build(hydrated_source, temporary_root / "hydrated")
        _validate_artifacts(
            hydrated_source,
            hydrated_artifacts,
            temporary_root / "hydrated-extracted",
        )

        for label in sorted(pointer_artifacts):
            pointer_inventory = _inventory(pointer_artifacts[label])
            hydrated_inventory = _inventory(hydrated_artifacts[label])
            if pointer_inventory != hydrated_inventory:
                raise RuntimeError(
                    f"{label} inventory/content changed when excluded LFS bytes changed"
                )
            print(f"{label}: deterministic members={len(pointer_inventory)}")

    final_stat = LFS_FIXTURE.stat()
    assert _digest(LFS_FIXTURE.read_bytes()) == original_digest
    assert (
        final_stat.st_mode,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        final_stat.st_ino,
    ) == (
        original_stat.st_mode,
        original_stat.st_size,
        original_stat.st_mtime_ns,
        original_stat.st_ino,
    )
    assert _git_status() == original_status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
