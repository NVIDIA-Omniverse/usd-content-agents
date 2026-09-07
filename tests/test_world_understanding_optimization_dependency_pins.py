# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-contract tests for the shared optimization runtime."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependencies(pyproject: Path, extra: str) -> list[str]:
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return metadata["project"]["optional-dependencies"][extra]


def _requirements_named(requirements: list[str], name: str) -> list[str]:
    canonical_name = canonicalize_name(name)
    return [
        requirement
        for requirement in requirements
        if canonicalize_name(Requirement(requirement).name) == canonical_name
    ]


def _requirement_named(requirements: list[str], name: str) -> Requirement:
    matches = _requirements_named(requirements, name)
    assert len(matches) == 1, f"expected exactly one {name} requirement: {matches}"
    return Requirement(matches[0])


def test_physics_and_shared_optimization_extras_pin_same_botorch() -> None:
    shared = _optional_dependencies(REPO_ROOT / "pyproject.toml", "optimization")
    physics = _optional_dependencies(
        REPO_ROOT / "apps" / "physics_agent" / "pyproject.toml",
        "tuning",
    )

    assert _requirements_named(shared, "botorch") == _requirements_named(
        physics, "botorch"
    )


def test_physics_and_root_extras_pin_same_newton_contract() -> None:
    root = _optional_dependencies(REPO_ROOT / "pyproject.toml", "warp")
    physics = _optional_dependencies(
        REPO_ROOT / "apps" / "physics_agent" / "pyproject.toml",
        "newton",
    )

    for package_name in ("newton", "newton-usd-schemas"):
        root_requirement = _requirement_named(root, package_name)
        physics_requirement = _requirement_named(physics, package_name)
        assert str(physics_requirement.specifier) == str(root_requirement.specifier)

    assert _requirement_named(physics, "newton").extras == {"importers", "sim"}
