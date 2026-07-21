# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dynamic-version fallback behavior.

When users `git clone` the public repository without tags, `uv-dynamic-versioning`
reports a synthetic `0.0.0.post...` version. That is below the app dependency
floors such as `world-understanding>=0.2.0`, so chained editable installs fail
unless each dynamic package reads the repository version from `VERSION.md`.

When users `unzip` a public source archive (no `.git`), `uv-dynamic-versioning`
also needs a `fallback-version`. Test pins both:

1. Every dynamic `pyproject.toml` reads `[tool.uv-dynamic-versioning].from-file`
   from `VERSION.md`.
2. Every dynamic `pyproject.toml` declares `[tool.uv-dynamic-versioning].fallback-version`.
3. The fallback equals `VERSION.md` — so a chained `uv pip install -e apps/<svc>`
   still satisfies `world-understanding>=X.Y.Z` floors. The Codex round-2 review
   on !394 caught a `0.0.0` regression that would have broken those chained
   installs silently.
"""

import os
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PYPROJECT_PATHS = sorted(
    path
    for path in REPO_ROOT.rglob("pyproject.toml")
    if ".git" not in path.parts and ".venv" not in path.parts
)


def _expected_version() -> str:
    return (REPO_ROOT / "VERSION.md").read_text(encoding="utf-8").strip()


def _expected_version_source(pyproject_path: Path) -> str:
    return os.path.relpath(REPO_ROOT / "VERSION.md", pyproject_path.parent).replace(
        os.sep, "/"
    )


@pytest.mark.parametrize(
    "pyproject_path", PYPROJECT_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_dynamic_versioning_reads_version_md(pyproject_path: Path) -> None:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    udv = data.get("tool", {}).get("uv-dynamic-versioning")
    if udv is None:
        pytest.skip(
            f"{pyproject_path.relative_to(REPO_ROOT)} does not use uv-dynamic-versioning"
        )

    expected_source = _expected_version_source(pyproject_path)
    from_file = udv.get("from-file")
    assert from_file == {"source": expected_source}, (
        f"{pyproject_path.relative_to(REPO_ROOT)} must read its dynamic version "
        f"from VERSION.md using source={expected_source!r}. Tagless Git clones "
        "otherwise resolve to 0.0.0.post... and can fail dependency resolution."
    )

    fallback = udv.get("fallback-version")
    assert fallback, (
        f"{pyproject_path.relative_to(REPO_ROOT)} is missing "
        "[tool.uv-dynamic-versioning].fallback-version. ZIP-download installs "
        "(no .git) will fail without it."
    )
    expected = _expected_version()
    assert fallback == expected, (
        f"{pyproject_path.relative_to(REPO_ROOT)} fallback-version={fallback!r} "
        f"does not match VERSION.md={expected!r}. Bump these in "
        "lockstep — see CLAUDE.md 'Version Bumping' step."
    )
