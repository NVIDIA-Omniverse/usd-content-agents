# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for direct ``requests`` dependency declarations.

Service `client.py` and four `world_understanding/` modules import `requests`
at top-level, but only `ovrtx_rendering_api` had it declared as a direct
dependency. The chain `langchain-* -> google-genai -> requests` masked the
gap during normal installs; `--no-deps` or any upstream dropping `requests`
would surface the bug.

Test scans every `pyproject.toml` in the repo (root + all apps), and for any
package whose source tree imports `requests` at top-level, asserts the
package declares `requests` in `[project] dependencies` (the runtime
contract — `optional-dependencies` would not satisfy a default install).
Narrow on purpose: just `requests`. A broader "every external import is
declared" check would need a module-name-to-distribution-name alias map and
would be flakier than the specific bug class it would replace.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# pyproject.toml + the directory whose source files we scan for imports.
PROJECT_ROOTS: list[tuple[Path, list[Path]]] = []
PROJECT_ROOTS.append(
    (REPO_ROOT / "pyproject.toml", [REPO_ROOT / "world_understanding"])
)
for app_pyproject in sorted((REPO_ROOT / "apps").glob("*/pyproject.toml")):
    PROJECT_ROOTS.append((app_pyproject, [app_pyproject.parent]))


def _imports_requests(source_dirs: list[Path]) -> Path | None:
    """Return the first .py file in source_dirs that imports `requests` at top
    level, or None if no source file imports it."""
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        for py in source_dir.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if isinstance(node, ast.Import) and any(
                    a.name.split(".")[0] == "requests" for a in node.names
                ):
                    return py
                if (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == "requests"
                ):
                    return py
    return None


def _declares_requests_runtime(pyproject_path: Path) -> bool:
    """Return True iff `requests` is declared in `[project].dependencies`.

    Optional-dependencies / dev-dependencies do *not* count — a default
    `pip install` of the package must pull in `requests`.
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    runtime_deps = list(data.get("project", {}).get("dependencies") or [])
    pattern = re.compile(r"^\s*requests(?:\[|\s|=|<|>|!|~|;|$)")
    return any(pattern.match(dep) for dep in runtime_deps)


@pytest.mark.parametrize(
    "pyproject_path,source_dirs",
    PROJECT_ROOTS,
    ids=[str(p[0].relative_to(REPO_ROOT)) for p in PROJECT_ROOTS],
)
def test_requests_declared_when_imported(
    pyproject_path: Path, source_dirs: list[Path]
) -> None:
    importer = _imports_requests(source_dirs)
    if importer is None:
        pytest.skip(
            f"{pyproject_path.relative_to(REPO_ROOT)} package does not import `requests`"
        )
    assert _declares_requests_runtime(pyproject_path), (
        f"{pyproject_path.relative_to(REPO_ROOT)} must declare `requests` in "
        f"[project].dependencies (NOT optional-dependencies) — "
        f"{importer.relative_to(REPO_ROOT)} imports it at top-level. "
        "Relying on transitive resolution from langchain / google-genai is "
        "fragile; `--no-deps` installs would break."
    )
