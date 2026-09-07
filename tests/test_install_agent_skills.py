# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safety regression tests for the personal skill installer."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = REPO_ROOT / "scripts/install_agent_skills.py"


def _load_installer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "install_agent_skills", INSTALLER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_skill(path: Path, body: str = "skill\n") -> None:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(body, encoding="utf-8")


def test_installer_rejects_empty_source_root(tmp_path: Path) -> None:
    installer = _load_installer()
    source_root = tmp_path / "repo/.agents/skills"
    source_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="no skills found"):
        installer._skill_sources(source_root, tmp_path / "repo")


def test_installer_rejects_broken_root_skill_link(tmp_path: Path) -> None:
    installer = _load_installer()
    repo_root = tmp_path / "repo"
    source_root = repo_root / ".agents/skills"
    source_root.mkdir(parents=True)
    (source_root / "broken").symlink_to(repo_root / "missing", target_is_directory=True)

    with pytest.raises(RuntimeError, match="skill entry is not a directory"):
        installer._skill_sources(source_root, repo_root)


def test_installer_rejects_cross_repo_root_skill_link(tmp_path: Path) -> None:
    installer = _load_installer()
    repo_root = tmp_path / "repo"
    source_root = repo_root / ".agents/skills"
    source_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    _write_skill(outside)
    (source_root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="resolves outside the repository"):
        installer._skill_sources(source_root, repo_root)


def test_installer_rejects_nested_skill_symlink(tmp_path: Path) -> None:
    installer = _load_installer()
    repo_root = tmp_path / "repo"
    source_root = repo_root / ".agents/skills"
    skill_root = source_root / "nested"
    _write_skill(skill_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (skill_root / "escape.txt").symlink_to(outside)

    with pytest.raises(RuntimeError, match="nested skill symlinks are not supported"):
        installer._skill_sources(source_root, repo_root)


def test_installer_detects_copy_digest_mismatch(tmp_path: Path) -> None:
    installer = _load_installer()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_skill(source, "source\n")
    _write_skill(destination, "modified\n")

    with pytest.raises(RuntimeError, match="does not match its source"):
        installer._validate_copy(source, destination)


def test_installer_rolls_back_every_replacement_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = _load_installer()
    sources = {
        "alpha": tmp_path / "sources/alpha",
        "bravo": tmp_path / "sources/bravo",
    }
    staged_root = tmp_path / "staged"
    target = tmp_path / "target"
    backup_root = tmp_path / "backups"
    for name, source in sources.items():
        _write_skill(source, f"new {name}\n")
        _write_skill(staged_root / name, f"new {name}\n")
        _write_skill(target / name, f"old {name}\n")

    validate_copy = installer._validate_copy

    def fail_second_install(source: Path, destination: Path) -> None:
        validate_copy(source, destination)
        if destination == target / "bravo":
            raise RuntimeError("simulated validation failure")

    monkeypatch.setattr(installer, "_validate_copy", fail_second_install)

    with pytest.raises(RuntimeError, match="simulated validation failure"):
        installer._replace_skills_transactionally(
            sources, staged_root, target, backup_root
        )

    assert (target / "alpha/SKILL.md").read_text(encoding="utf-8") == "old alpha\n"
    assert (target / "bravo/SKILL.md").read_text(encoding="utf-8") == "old bravo\n"
    assert list(backup_root.iterdir()) == []


def test_installer_rejects_target_inside_canonical_skill_tree() -> None:
    target = REPO_ROOT / ".agents/skills/installer-test-target"
    assert not target.exists()

    completed = subprocess.run(
        [sys.executable, str(INSTALLER_PATH), "--target", str(target)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "install target cannot be inside .agents/skills" in completed.stderr
    assert not target.exists()


def test_installer_rejects_target_resolved_inside_linked_skill_source(
    tmp_path: Path,
) -> None:
    installer = _load_installer()
    repo_root = tmp_path / "repo"
    source_root = repo_root / ".agents/skills"
    linked_source = repo_root / "agentic/.agents/skills/example"
    source_root.mkdir(parents=True)
    _write_skill(linked_source)
    (source_root / "example").symlink_to(linked_source, target_is_directory=True)
    sources = installer._skill_sources(source_root, repo_root)
    target = (source_root / "example/install").resolve()

    with pytest.raises(RuntimeError, match="overlap a canonical skill source"):
        installer._validate_target(target, source_root, sources)


def test_installer_rejects_target_containing_linked_skill_sources(
    tmp_path: Path,
) -> None:
    installer = _load_installer()
    repo_root = tmp_path / "repo"
    source_root = repo_root / ".agents/skills"
    linked_root = repo_root / "agentic/.agents/skills"
    linked_source = linked_root / "example"
    source_root.mkdir(parents=True)
    _write_skill(linked_source)
    (source_root / "example").symlink_to(linked_source, target_is_directory=True)
    sources = installer._skill_sources(source_root, repo_root)

    with pytest.raises(RuntimeError, match="overlap a canonical skill source"):
        installer._validate_target(linked_root.resolve(), source_root, sources)
