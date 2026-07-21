# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline, table-driven coverage for the four public agent CLI boundaries."""

from __future__ import annotations

import importlib
import logging
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_APP_ROOTS = {
    "material": REPO_ROOT / "apps" / "material_agent",
    "physics": REPO_ROOT / "apps" / "physics_agent",
    "joint": REPO_ROOT / "apps" / "joint_agent",
    "texture": REPO_ROOT / "apps" / "texture_agent",
}


@lru_cache
def _agent_app(agent: str) -> typer.Typer:
    app_root = str(AGENT_APP_ROOTS[agent])
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    return importlib.import_module(f"{agent}_agent.cli").app


@pytest.fixture(autouse=True)
def _isolate_agent_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI logging setup from leaking across root test modules."""
    for agent in AGENT_APP_ROOTS:
        cli_module = importlib.import_module(f"{agent}_agent.cli")
        if agent == "texture":
            monkeypatch.setattr(
                cli_module,
                "_setup_logging",
                lambda *_args, **_kwargs: None,
            )
        else:
            monkeypatch.setattr(
                cli_module,
                "setup_logging",
                lambda *_args, _agent=agent, **_kwargs: logging.getLogger(
                    f"{_agent}_agent"
                ),
            )


@pytest.mark.parametrize("agent", tuple(AGENT_APP_ROOTS))
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("api_key: never-render-this-value\nsteps: [\n", "Unable to parse"),
        ("", "Pipeline configuration is empty"),
        ("- never-render-this-value\n", "must be a mapping"),
    ],
)
def test_agent_run_rejects_invalid_yaml_without_side_effects_or_tracebacks(
    agent: str,
    payload: str,
    message: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(payload, encoding="utf-8")

    result = CliRunner().invoke(
        _agent_app(agent),
        ["run", str(config_path), "--verbose"],
    )

    assert result.exit_code == 1
    assert message in result.output
    assert "never-render-this-value" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == [config_path]


@pytest.mark.parametrize("agent", tuple(AGENT_APP_ROOTS))
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--only", "predcit"), "Invalid --only step name(s): 'predcit'"),
        (
            ("--skip", "predict", "--only", "apply"),
            "--skip and --only cannot be used together",
        ),
        (("--only", "predict,,apply"), "--only contains an empty step name"),
    ],
)
def test_agent_run_rejects_invalid_step_filters_before_pipeline_work(
    agent: str,
    options: tuple[str, ...],
    message: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("project:\n  name: demo\n", encoding="utf-8")

    result = CliRunner().invoke(
        _agent_app(agent),
        ["run", str(config_path), *options],
    )

    assert result.exit_code == 1
    assert message in result.output
    if "cannot be used together" in message:
        assert "cannot be used together" in result.output
    else:
        assert "Valid steps:" in result.output
    assert list(tmp_path.iterdir()) == [config_path]


@pytest.mark.parametrize("agent", ("material", "physics", "joint"))
def test_agent_run_accepts_trimmed_deduplicated_valid_filters(
    agent: str,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "project:\n"
        "  name: demo\n"
        "input:\n"
        "  usd_path: input.usda\n"
        "steps:\n"
        "  predict:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        _agent_app(agent),
        ["run", str(config_path), "--only", " predict,predict ", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run complete" in result.output
