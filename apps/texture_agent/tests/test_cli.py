# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from pytest import LogCaptureFixture
from typer.main import get_command
from typer.testing import CliRunner

from texture_agent.cli import app
from texture_agent.config import unified_config
from texture_agent.workflows import factory as workflow_factory

runner = CliRunner()


def test_run_help_documents_resume_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0

    run_command = get_command(app).commands["run"]
    options_by_name = {param.name: param for param in run_command.params}

    resume_option = options_by_name["resume"]
    session_id_option = options_by_name["session_id"]

    assert "--resume" in resume_option.opts
    assert resume_option.help == "Reuse existing artifacts from the working directory"
    assert "--session-id" in session_id_option.opts
    assert session_id_option.help == "Reuse or override the config session ID"


@pytest.mark.parametrize("option", ["--only", "--skip"])
def test_run_rejects_empty_step_filter_from_cli(
    option: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: LogCaptureFixture,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")

    def _unexpected_load_config(
        path: Path,
        session_id: str | None = None,
        *,
        config_data: dict,
    ) -> dict:
        raise AssertionError("invalid filters must fail before config loading")

    monkeypatch.setattr(
        unified_config,
        "load_config",
        _unexpected_load_config,
    )
    monkeypatch.setattr(unified_config, "config_to_context", lambda config: {})

    result = runner.invoke(
        app,
        ["run", str(config_path), option, "", "--dry-run"],
    )

    assert result.exit_code == 1
    assert "empty step name" in caplog.text


def test_run_cli_applies_detail_policy_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    captured: dict[str, dict] = {}

    def _load_config(
        path: Path,
        session_id: str | None = None,
        *,
        config_data: dict,
    ) -> dict:
        captured["config_data"] = config_data
        return {"texture": {}}

    monkeypatch.setattr(
        unified_config,
        "load_config",
        _load_config,
    )

    def _capture_config(config: dict) -> dict:
        captured["config"] = config
        return {}

    monkeypatch.setattr(unified_config, "config_to_context", _capture_config)
    monkeypatch.setattr(
        workflow_factory,
        "run_pipeline",
        lambda context, **kwargs: context,
    )

    result = runner.invoke(
        app,
        ["run", str(config_path), "--detail-policy", "surface_only", "--dry-run"],
    )

    assert result.exit_code == 0
    assert captured["config_data"] == {"input": {}}
    assert captured["config"]["texture"]["detail_policy"] == "surface_only"


def test_run_cli_normalizes_step_filters_before_pipeline_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _load_config(
        path: Path,
        session_id: str | None = None,
        *,
        config_data: dict,
    ) -> dict:
        captured["config_data"] = config_data
        return {}

    def _run_pipeline(
        context: dict,
        *,
        skip: list[str],
        only: list[str],
        dry_run: bool,
    ) -> dict:
        captured.update(skip=skip, only=only, dry_run=dry_run)
        return context

    monkeypatch.setattr(unified_config, "load_config", _load_config)
    monkeypatch.setattr(unified_config, "config_to_context", lambda config: {})
    monkeypatch.setattr(workflow_factory, "run_pipeline", _run_pipeline)

    result = runner.invoke(
        app,
        [
            "run",
            str(config_path),
            "--only",
            " prepare_uvs,prepare_uvs ",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "config_data": {"input": {}},
        "skip": [],
        "only": ["prepare_uvs"],
        "dry_run": True,
    }


def test_generate_cli_applies_detail_policy_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    captured: dict[str, dict] = {}

    monkeypatch.setattr(unified_config, "load_config", lambda path: {"texture": {}})

    def _capture_config(config: dict) -> dict:
        captured["config"] = config
        return {}

    monkeypatch.setattr(unified_config, "config_to_context", _capture_config)
    monkeypatch.setattr(
        workflow_factory,
        "run_pipeline",
        lambda context, **kwargs: context,
    )

    result = runner.invoke(
        app,
        ["generate", str(config_path), "--detail-policy", "surface_only"],
    )

    assert result.exit_code == 0
    assert captured["config"]["texture"]["detail_policy"] == "surface_only"
