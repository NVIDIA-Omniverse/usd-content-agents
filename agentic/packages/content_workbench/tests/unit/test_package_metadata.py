# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _dependency_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
        if match:
            names.add(match.group(1).replace("_", "-").lower())
    return names


def test_workbench_declares_world_understanding_dependency() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    dependencies = _dependency_names(pyproject["project"]["dependencies"])
    assert "content-agent-workflows" not in dependencies
    assert "physics-agent" in dependencies
    assert "world-understanding" in dependencies

    physics_source = pyproject["tool"]["uv"]["sources"]["physics-agent"]
    assert physics_source == {
        "path": "../../../apps/physics_agent",
        "editable": True,
    }

    source = pyproject["tool"]["uv"]["sources"]["world-understanding"]
    assert source == {"path": "../../..", "editable": True}
