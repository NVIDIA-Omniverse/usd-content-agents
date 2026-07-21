# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional material assignment authoring adapter."""

from __future__ import annotations

import builtins
import tomllib
from pathlib import Path
from typing import Any

import pytest

from content_workbench.material_apply_adapter import run_material_apply_task


def test_material_apply_adapter_reports_missing_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("material_agent"):
            raise ImportError("hidden for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="optional material-agent package"):
        run_material_apply_task({})


def test_content_workbench_metadata_does_not_require_material_agent():
    project_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))

    dependencies = metadata["project"]["dependencies"]
    source_dependencies = metadata.get("tool", {}).get("uv", {}).get("sources", {})

    assert "material-agent" not in dependencies
    assert "material-agent" not in source_dependencies
