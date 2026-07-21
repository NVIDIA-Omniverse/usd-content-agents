# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for credential-safe CLI command boundaries."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import pytest
import typer

from world_understanding.agentic.cli import sever_cli_exception_graph


@pytest.mark.parametrize("unexpected", [False, True])
def test_sever_cli_exception_graph_replaces_runtime_frames_and_arguments(
    caplog: pytest.LogCaptureFixture,
    unexpected: bool,
) -> None:
    sentinel = "shared-cli-boundary-credential-713"

    @sever_cli_exception_graph
    def command(config: Path, *, payload: dict[str, Any]) -> None:
        if payload["unexpected"]:
            raise RuntimeError(
                f"command failed with api_key={payload['api_key']} at {config}"
            )
        raise typer.Exit(7)

    original_signature = inspect.signature(command.__wrapped__)
    assert inspect.signature(command) == original_signature

    with caplog.at_level(logging.ERROR), pytest.raises(typer.Exit) as exc_info:
        command(
            Path(f"config.yaml?X-Amz-Signature={sentinel}"),
            payload={"api_key": sentinel, "unexpected": unexpected},
        )

    assert exc_info.value.exit_code == (1 if unexpected else 7)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    boundary_locals: list[Any] = []
    current = exc_info.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "world_understanding.agentic.cli.safety"
        ):
            boundary_locals.extend(current.tb_frame.f_locals.values())
        current = current.tb_next

    observable = repr((boundary_locals, caplog.records))
    assert sentinel not in observable
    assert all(record.exc_info is None for record in caplog.records)
    if unexpected:
        assert "CLI command failed" in caplog.text


def test_sever_cli_exception_graph_survives_a_failing_log_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "shared-cli-logging-failure-credential-713"

    @sever_cli_exception_graph
    def command(payload: dict[str, str]) -> None:
        raise RuntimeError(f"backend reflected {payload['api_key']}")

    logger = logging.getLogger(command.__module__)

    def fail_logging(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("logging backend failed")

    monkeypatch.setattr(logger, "error", fail_logging)

    with pytest.raises(typer.Exit) as exc_info:
        command({"api_key": sentinel})

    assert exc_info.value.exit_code == 1
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    boundary_locals: list[Any] = []
    current = exc_info.value.__traceback__
    while current is not None:
        if current.tb_frame.f_globals.get("__name__") == (
            "world_understanding.agentic.cli.safety"
        ):
            boundary_locals.extend(current.tb_frame.f_locals.values())
        current = current.tb_next
    assert sentinel not in repr(boundary_locals)
