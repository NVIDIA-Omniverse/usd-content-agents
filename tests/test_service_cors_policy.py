# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for wildcard service CORS configuration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_MAIN_FILES = (
    "apps/material_agent_service/service/main.py",
    "apps/joint_agent_service/service/main.py",
    "apps/physics_agent_service/service/main.py",
    "apps/texture_agent_service/service/main.py",
)


@pytest.mark.parametrize("relative_path", SERVICE_MAIN_FILES)
def test_wildcard_cors_does_not_allow_credentials(relative_path: str) -> None:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    cors_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_middleware"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "CORSMiddleware"
    ]

    assert len(cors_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in cors_calls[0].keywords}
    assert ast.literal_eval(keywords["allow_origins"]) == ["*"]
    assert ast.literal_eval(keywords["allow_credentials"]) is False
