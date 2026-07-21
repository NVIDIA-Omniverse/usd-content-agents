# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight checks for preview agentic skills."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_PUBLIC_SKILLS = {
    "content-workbench",
    "content-workflow-asset-task-processing",
    "content-workflow-cli",
    "content-workflow-convert-to-usd",
    "content-workflow-large-scene",
    "content-workflow-material",
    "content-workflow-physics",
    "content-workflow-scene-collection",
    "content-workflow-scene-decomposition",
    "content-workflow-simready",
}

REQUIRED_INTERNAL_SKILLS = {
    "content-workflow-articulation",
    "content-workflow-geometry",
    "content-workflow-runtime-validation",
    "content-workflow-texture",
}

ROOT_WORKFLOW_SKILL_REFERENCE = re.compile(
    r"(?<!agentic/)\.(?:agents|codex|claude)/skills/content-workflow-cli"
    r"(?:/SKILL\.md)?(?![-A-Za-z0-9_])"
)


def test_workflow_cli_skill_is_isolated_to_preview_workspace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_skill_dirs = [
        repo_root / root / "skills" / "content-workflow-cli"
        for root in (".agents", ".codex", ".claude")
    ]
    preview_skill_file = (
        repo_root
        / "agentic"
        / ".agents"
        / "skills"
        / "content-workflow-cli"
        / "SKILL.md"
    )

    assert all(not path.exists() for path in root_skill_dirs)
    assert preview_skill_file.is_file()


def test_docs_do_not_reference_repo_root_workflow_cli_skill() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    excluded_dirs = {
        ".benchmark",
        ".build-resources",
        ".data",
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    documents: list[Path] = []

    for directory, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [name for name in dirnames if name not in excluded_dirs]
        for filename in filenames:
            if not (
                filename.endswith(".md")
                and filename.startswith(("README", "AGENTS", "CLAUDE"))
            ):
                continue
            documents.append(Path(directory) / filename)

    documents.extend(repo_root / filename for filename in ("llms.txt", "llms-full.txt"))
    violations = [
        str(path.relative_to(repo_root))
        for path in documents
        if ROOT_WORKFLOW_SKILL_REFERENCE.search(path.read_text(encoding="utf-8"))
    ]

    assert not violations, "repo-root workflow skill referenced by: " + ", ".join(
        violations
    )


def test_preview_skill_sandbox_points_to_agent_skills() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skills_dir = repo_root / "agentic" / ".agents" / "skills"
    codex_skill_link = repo_root / "agentic" / ".codex" / "skills"
    claude_skill_link = repo_root / "agentic" / ".claude" / "skills"

    assert codex_skill_link.is_symlink()
    assert codex_skill_link.resolve() == skills_dir
    assert claude_skill_link.is_symlink()
    assert claude_skill_link.resolve() == skills_dir


def test_preview_workspace_links_build_resources() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    build_resources_link = repo_root / "agentic" / ".build-resources"

    assert build_resources_link.is_symlink()
    assert build_resources_link.resolve() == repo_root / ".build-resources"


def test_preview_skills_have_discoverable_frontmatter() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skills_dir = repo_root / "agentic" / ".agents" / "skills"
    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    names = {path.parent.name for path in skill_files}

    required_skills = set(REQUIRED_PUBLIC_SKILLS)
    if (repo_root / "agentic" / "pyproject.toml").exists():
        required_skills.update(REQUIRED_INTERNAL_SKILLS)

    assert required_skills <= names
    for path in skill_files:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        frontmatter = text.split("---\n", 2)[1]
        assert f"name: {path.parent.name}" in frontmatter
        assert "description: " in frontmatter


def test_asset_task_skill_uses_frozen_workbench_url_for_material_commands() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skill_path = (
        repo_root
        / "agentic/.agents/skills/content-workflow-asset-task-processing/SKILL.md"
    )
    skill = skill_path.read_text(encoding="utf-8")

    assert "resolve_workbench_url.py" in skill
    assert skill.count('--workbench-url "$WORKBENCH_URL"') == 2
    assert not re.search(r"--workbench-url\s+https?://", skill)


def test_asset_task_workbench_url_resolver_preserves_frozen_value(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolver = (
        repo_root
        / "agentic/.agents/skills/content-workflow-asset-task-processing/scripts/"
        "resolve_workbench_url.py"
    )
    request_path = tmp_path / "request.json"

    for workbench_url in (
        "http://127.0.0.1:8088",
        "https://workbench.example.test:8443",
    ):
        request_path.write_text(
            json.dumps({"runtime": {"workbench_url": workbench_url}}),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(resolver), str(request_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == workbench_url


def test_asset_task_workbench_url_resolver_guides_invalid_requests(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolver = (
        repo_root
        / "agentic/.agents/skills/content-workflow-asset-task-processing/scripts/"
        "resolve_workbench_url.py"
    )
    missing_request = tmp_path / "missing" / "request.json"
    malformed_request = tmp_path / "malformed.json"
    malformed_request.write_text('{"runtime":', encoding="utf-8")

    for request_path, guidance in (
        (missing_request, "prepared scene workflow directory"),
        (malformed_request, "not readable valid JSON"),
    ):
        completed = subprocess.run(
            [sys.executable, str(resolver), str(request_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode != 0
        assert guidance in completed.stderr
