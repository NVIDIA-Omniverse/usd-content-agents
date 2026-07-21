# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package contracts for consumers of the shared rendering backend factory."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_API_FLOOR = Version("0.5.0")
PRE_FACTORY_RELEASE = Version("0.4.11")
SERVICE_DOCKERFILES = (
    REPO_ROOT / "apps/material_agent_service/Dockerfile",
    REPO_ROOT / "apps/physics_agent_service/Dockerfile",
    REPO_ROOT / "apps/joint_agent_service/Dockerfile",
    REPO_ROOT / "apps/texture_agent_service/Dockerfile",
)


def _requirement(dependencies: list[str], name: str) -> Requirement:
    matches = [
        Requirement(dependency)
        for dependency in dependencies
        if Requirement(dependency).name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _requires_explicit_api_floor(requirement: Requirement) -> bool:
    """Return whether the requirement declares the 0.5 factory API floor."""
    return any(
        specifier.operator == ">=" and Version(specifier.version) == PACKAGE_API_FLOOR
        for specifier in requirement.specifier
    )


@pytest.mark.parametrize(
    ("specifier", "expected"),
    (
        (">=0.5.0", True),
        (">=0.5.0,<0.6", True),
        (">=0.4.12", False),
        (">0.4.11", False),
        ("!=0.4.11", False),
    ),
)
def test_explicit_api_floor_predicate(specifier: str, expected: bool) -> None:
    assert _requires_explicit_api_floor(Requirement(f"example{specifier}")) is expected


@pytest.mark.parametrize(
    "agent_name",
    (
        "material_agent",
        "physics_agent",
        "joint_agent",
        "texture_agent",
        "validation_agent",
    ),
)
def test_agents_require_rendering_backend_factory_release(agent_name: str) -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "apps" / agent_name / "pyproject.toml").read_text(encoding="utf-8")
    )
    runtime = _requirement(pyproject["project"]["dependencies"], "world-understanding")
    development = _requirement(
        pyproject["project"]["optional-dependencies"]["dev"],
        "world-understanding",
    )

    assert _requires_explicit_api_floor(runtime)
    assert PRE_FACTORY_RELEASE not in runtime.specifier
    assert _requires_explicit_api_floor(development)
    assert PRE_FACTORY_RELEASE not in development.specifier
    assert development.extras == {"dev"}


@pytest.mark.parametrize(
    ("service_name", "agent_requirement"),
    (
        ("material_agent_service", "material-agent"),
        ("physics_agent_service", "physics-agent"),
        ("joint_agent_service", "joint-agent"),
        ("texture_agent_service", "texture-agent"),
    ),
)
def test_services_require_compatible_agent_and_core_releases(
    service_name: str,
    agent_requirement: str,
) -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "apps" / service_name / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    dependencies = pyproject["project"]["dependencies"]

    for requirement_name in (agent_requirement, "world-understanding"):
        requirement = _requirement(dependencies, requirement_name)
        assert _requires_explicit_api_floor(requirement)
        assert PRE_FACTORY_RELEASE not in requirement.specifier


@pytest.mark.parametrize(
    "dockerfile",
    SERVICE_DOCKERFILES,
    ids=lambda path: str(path.relative_to(REPO_ROOT)),
)
def test_service_images_use_release_version_for_editable_packages(
    dockerfile: Path,
) -> None:
    text = dockerfile.read_text(encoding="utf-8")

    assert "UV_DYNAMIC_VERSIONING_BYPASS=0.0.0" not in text
    editable_install_count = text.count("uv pip install -e")
    assert editable_install_count > 0
    assert (
        text.count('UV_DYNAMIC_VERSIONING_BYPASS="$(cat /app/VERSION.md)"')
        == editable_install_count
    )
    assert 'SETUPTOOLS_SCM_PRETEND_VERSION="$(cat /app/VERSION.md)"' in text
