# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-safe host capability gates for the console entry point."""

from __future__ import annotations

import sys

import pytest

from content_workflow_cli import cli, entrypoint


@pytest.mark.parametrize(
    "command",
    [
        ["materials", "assign"],
        ["physics", "apply"],
        ["mesh-segmentation", "run"],
        ["scene", "run"],
        ["texture", "run"],
        ["geometry", "run"],
        ["asset", "run"],
    ],
)
def test_windows_child_commands_reach_portable_cli(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli, "main", lambda arguments: observed.append(arguments) or 0)

    assert entrypoint.main(command) == 0
    assert observed == [command]


def test_windows_cad_to_simready_reaches_portable_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli, "main", lambda arguments: observed.append(arguments) or 0)

    arguments = ["cad-to-simready", "run", "asset.glb"]
    assert entrypoint.main(arguments) == 0
    assert observed == [arguments]


def test_macos_cad_to_simready_remains_host_gated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    assert entrypoint.main(["cad-to-simready", "run"]) == 2

    assert "supports Linux, WSL2, and native Windows" in capsys.readouterr().err


def test_windows_cad_to_simready_dry_run_reaches_portable_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli, "main", lambda arguments: observed.append(arguments) or 0)

    arguments = ["cad-to-simready", "run", "asset.glb", "--dry-run"]
    assert entrypoint.main(arguments) == 0
    assert observed == [arguments]


def test_windows_child_dry_run_reaches_portable_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli, "main", lambda arguments: observed.append(arguments) or 0)

    arguments = ["materials", "assign", "--dry-run"]
    assert entrypoint.main(arguments) == 0
    assert observed == [arguments]


def test_windows_child_help_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cli, "main", lambda arguments: observed.append(arguments) or 0)

    arguments = ["materials", "assign", "--help"]
    assert entrypoint.main(arguments) == 0
    assert observed == [arguments]
