# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the agentic preview Python package boundaries."""

from __future__ import annotations

import importlib


def test_preview_packages_import():
    package_names = [
        "content_agent_workflows",
        "content_agent_workflows.common",
        "content_agent_workflows.material_assignment",
        "content_agent_workflows.texture",
        "content_workbench",
        "content_workbench_agent_client",
        "content_workflow_cli",
    ]

    for package_name in package_names:
        assert importlib.import_module(package_name).__name__ == package_name
