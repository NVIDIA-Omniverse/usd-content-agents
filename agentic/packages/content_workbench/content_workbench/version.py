# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content Workbench package version helpers."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _service_version() -> str:
    try:
        return version("content-workbench")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            project_version = data.get("project", {}).get("version")
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            project_version = None
        if isinstance(project_version, str) and project_version:
            return project_version
        return "0.1.0"


SERVICE_VERSION = _service_version()
