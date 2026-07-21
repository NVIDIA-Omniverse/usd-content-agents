# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for module entry points and compatibility re-exports."""

from __future__ import annotations

import runpy

import material_agent.cli
from material_agent.tasks.config_optimize_usd import OptimizeUSDConfigTask
from material_agent.tasks.config_restore_usd import RestoreUSDConfigTask
from material_agent.tasks.optimize_usd import OptimizeUSDTask


def test_reexported_usd_tasks_are_available() -> None:
    assert OptimizeUSDConfigTask is not None
    assert RestoreUSDConfigTask is not None
    assert OptimizeUSDTask is not None


def test_module_entrypoint_invokes_cli_app(monkeypatch) -> None:
    called: list[bool] = []

    def fake_app() -> None:
        called.append(True)

    monkeypatch.setattr(material_agent.cli, "app", fake_app)

    runpy.run_module("material_agent.__main__", run_name="__main__")

    assert called == [True]
