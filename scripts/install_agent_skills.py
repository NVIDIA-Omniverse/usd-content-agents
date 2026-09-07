# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize this repository's agent skills into a personal skill directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

IGNORED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache"}
IGNORED_FILE_NAMES = {".DS_Store"}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


def _default_target() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills"
    return Path.home() / ".codex" / "skills"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the repository's canonical skills into a personal skill "
            "directory while materializing repository-relative symlinks."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=_default_target(),
        help="Personal skill directory (default: $CODEX_HOME/skills or ~/.codex/skills).",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Replace only destination entries whose names collide with skills "
            "from this repository. Other personal skills remain untouched."
        ),
    )
    return parser.parse_args()


def _ignored_path(path: Path) -> bool:
    return (
        path.name in IGNORED_DIRECTORY_NAMES
        or path.name in IGNORED_FILE_NAMES
        or path.suffix.lower() in IGNORED_FILE_SUFFIXES
    )


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if _ignored_path(Path(name))}


def _skill_sources(source_root: Path, repo_root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for entry in sorted(source_root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            raise RuntimeError(f"skill entry is not a directory: {entry}")

        resolved = entry.resolve(strict=True)
        if not resolved.is_relative_to(repo_root):
            raise RuntimeError(f"skill resolves outside the repository: {entry}")
        if not (resolved / "SKILL.md").is_file():
            raise RuntimeError(f"skill has no SKILL.md: {entry}")

        for current_root, directory_names, file_names in os.walk(resolved):
            current = Path(current_root)
            directory_names[:] = [
                name for name in directory_names if not _ignored_path(current / name)
            ]
            for name in (*directory_names, *file_names):
                candidate = current / name
                if _ignored_path(candidate):
                    continue
                if candidate.is_symlink():
                    raise RuntimeError(
                        f"nested skill symlinks are not supported: {candidate}"
                    )

        sources[entry.name] = resolved
    if not sources:
        raise RuntimeError(f"no skills found under {source_root}")
    return sources


def _file_inventory(root: Path) -> dict[Path, str]:
    inventory: dict[Path, str] = {}
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        directory_names[:] = [
            name for name in directory_names if not _ignored_path(current / name)
        ]
        for name in file_names:
            path = current / name
            if _ignored_path(path):
                continue
            if path.is_symlink():
                raise RuntimeError(f"installed skill contains a symlink: {path}")
            inventory[path.relative_to(root)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return inventory


def _validate_copy(source: Path, destination: Path) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise RuntimeError(
            f"installed skill is not a materialized directory: {destination}"
        )
    if not (destination / "SKILL.md").is_file():
        raise RuntimeError(f"installed skill has no SKILL.md: {destination}")
    if _file_inventory(source) != _file_inventory(destination):
        raise RuntimeError(f"installed skill does not match its source: {destination}")


def _remove_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _validate_target(target: Path, source_root: Path, sources: dict[str, Path]) -> None:
    canonical_root = source_root.resolve()
    if target.is_relative_to(canonical_root):
        raise RuntimeError("the install target cannot be inside .agents/skills")

    for name, source in sources.items():
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise RuntimeError(
                f"the install target cannot overlap a canonical skill source: {name}"
            )


def _replace_skills_transactionally(
    sources: dict[str, Path],
    staged_root: Path,
    target: Path,
    backup_root: Path,
) -> None:
    """Install staged skills and restore every prior destination on failure."""
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    backup_root.mkdir(parents=True, exist_ok=True)

    try:
        for name, source in sources.items():
            destination = target / name
            if destination.exists() or destination.is_symlink():
                backup = backup_root / name
                shutil.move(str(destination), backup)
                backups.append((destination, backup))

            shutil.move(str(staged_root / name), destination)
            installed.append(destination)
            _validate_copy(source, destination)
    except (OSError, RuntimeError) as install_error:
        rollback_errors: list[str] = []
        for destination in reversed(installed):
            try:
                if destination.exists() or destination.is_symlink():
                    _remove_entry(destination)
            except OSError as error:
                rollback_errors.append(f"remove {destination}: {error}")

        for destination, backup in reversed(backups):
            try:
                if destination.exists() or destination.is_symlink():
                    _remove_entry(destination)
                shutil.move(str(backup), destination)
            except OSError as error:
                rollback_errors.append(f"restore {destination}: {error}")

        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                "skill installation failed and rollback was incomplete: " + details
            ) from install_error
        raise


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    source_root = repo_root / ".agents" / "skills"
    target = args.target.expanduser().resolve()

    try:
        sources = _skill_sources(source_root, repo_root)
        _validate_target(target, source_root, sources)
        collisions = sorted(
            name
            for name in sources
            if (target / name).exists() or (target / name).is_symlink()
        )
        if collisions and not args.replace_existing:
            names = ", ".join(collisions)
            raise RuntimeError(
                "destination entries already exist; no skills were installed: "
                f"{names}. Review them, then rerun with --replace-existing to "
                "replace only these colliding names."
            )

        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".world-understanding-skills-", dir=target.parent
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            staged_root = temporary_root / "staged"
            staged_root.mkdir()
            for name, source in sources.items():
                staged = staged_root / name
                shutil.copytree(
                    source,
                    staged,
                    symlinks=False,
                    copy_function=shutil.copy2,
                    ignore=_copy_ignore,
                )
                _validate_copy(source, staged)

            _replace_skills_transactionally(
                sources,
                staged_root,
                target,
                temporary_root / "backups",
            )
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Installed and validated {len(sources)} skills in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
