# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
STEP1X_RUNTIME_PXR_FILES = (
    REPO_ROOT / "apps/texture_gen_step1x_service/runtime/edit_texture.py",
    REPO_ROOT / "apps/texture_gen_step1x_service/runtime/src/texture_edit/usd_utils.py",
)
STEP1X_SERVICE_DOCKERFILE = REPO_ROOT / "apps/texture_gen_step1x_service/Dockerfile"


def _active_usd_dependencies(
    dependencies: list[str],
    *,
    platform_system: str,
    platform_machine: str,
    python_version: str,
) -> list[str]:
    env = {
        "platform_system": platform_system,
        "platform_machine": platform_machine,
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "sys_platform": "linux" if platform_system == "Linux" else sys.platform,
    }
    active = []
    for dependency in dependencies:
        requirement = Requirement(dependency)
        if requirement.name not in {"usd-core", "usd-exchange"}:
            continue
        if requirement.marker is None or requirement.marker.evaluate(env):
            active.append(requirement.name)
    return active


def test_root_pyproject_selects_usd_binding_for_supported_platforms() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    for dependencies in (
        data["project"]["dependencies"],
        data["project"]["optional-dependencies"]["usd"],
    ):
        assert _active_usd_dependencies(
            dependencies,
            platform_system="Linux",
            platform_machine="aarch64",
            python_version="3.12",
        ) == ["usd-exchange"]
        assert _active_usd_dependencies(
            dependencies,
            platform_system="Linux",
            platform_machine="aarch64",
            python_version="3.13",
        ) == ["usd-exchange"]
        assert _active_usd_dependencies(
            dependencies,
            platform_system="Linux",
            platform_machine="x86_64",
            python_version="3.13",
        ) == ["usd-exchange"]


def test_step1x_runtime_uses_the_attributed_usd_exchange_constraint() -> None:
    runtime_files = [path for path in STEP1X_RUNTIME_PXR_FILES if path.exists()]
    if not runtime_files:
        pytest.skip("managed Step1X runtime files are not shipped in this artifact")

    for runtime_file in runtime_files:
        source = runtime_file.read_text(encoding="utf-8")
        assert '"usd-core"' not in source
        assert source.count('"usd-exchange>=2.3,<3"') == 1

    dockerfile = STEP1X_SERVICE_DOCKERFILE.read_text(encoding="utf-8")
    assert "usd-core" not in dockerfile
    assert dockerfile.count('"usd-exchange>=2.3,<3"') == 1
