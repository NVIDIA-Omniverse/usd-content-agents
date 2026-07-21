# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone validation-agent CLI package."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner
from world_understanding.validation.cli import (
    PASS_EXIT_CODE,
    VALIDATION_CLI_ERROR_EXIT_CODE,
)

import validation_agent.cli as cli

app = cli.app
runner = CliRunner()


def test_validation_agent_validate_help_lists_render_backend_subset() -> None:
    result = runner.invoke(app, ["validate", "--help"])

    assert result.exit_code == 0, result.output
    assert "remote, ovrtx" in result.output


def test_load_cli_dotenv_searches_up_from_cwd(monkeypatch) -> None:
    dotenv_path = "/repo/.env"
    find_dotenv = Mock(return_value=dotenv_path)
    load_dotenv = Mock()
    monkeypatch.setattr(cli, "find_dotenv", find_dotenv)
    monkeypatch.setattr(cli, "load_dotenv", load_dotenv)

    cli._load_cli_dotenv()

    find_dotenv.assert_called_once_with(usecwd=True)
    load_dotenv.assert_called_once_with(dotenv_path=dotenv_path)


def test_validation_agent_run_dry_run_writes_stable_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate render evidence.",
                "inputs:",
                "  - render.png",
                "project:",
                "  name: local-render",
                "  working_dir: run-from-config",
                "requested_templates:",
                "  - render_valid",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["run", str(config_path), "--dry-run", "--format", "json"],
    )

    assert result.exit_code == PASS_EXIT_CODE, result.output
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["schema_version"] == "1.0"
    assert stdout_payload["verdict"] == "planned"
    assert stdout_payload["plan"]["steps"][0]["template_name"] == "render_valid"

    output_dir = tmp_path / "run-from-config"
    assert (output_dir / "validation_request.json").is_file()
    assert (output_dir / "validation_plan.json").is_file()
    assert (output_dir / "validation_result.json").is_file()


def test_validation_agent_validate_dry_run_writes_stable_artifacts(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"not decoded during dry-run")
    output_dir = tmp_path / "direct-run"

    result = runner.invoke(
        app,
        [
            "validate",
            "--task",
            "Validate render evidence.",
            "--dry-run",
            "--template",
            "render_valid",
            "--focus-prim",
            "/World/Handle",
            "--reference-image",
            str(reference_path),
            "--render-backend",
            "remote",
            "--render-view",
            "front",
            "--image-width",
            "128",
            "--image-height",
            "96",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
            str(image_path),
        ],
    )

    assert result.exit_code == PASS_EXIT_CODE, result.output
    stdout_payload = json.loads(result.stdout)
    assert stdout_payload["verdict"] == "planned"
    assert stdout_payload["request"]["task_description"] == "Validate render evidence."
    assert stdout_payload["request"]["inputs"] == [str(image_path)]
    assert stdout_payload["request"]["requested_templates"] == ["render_valid"]
    assert stdout_payload["request"]["focus"]["prim_paths"] == ["/World/Handle"]
    assert stdout_payload["request"]["policy"] == {
        "reference_image_paths": [str(reference_path)]
    }
    assert stdout_payload["request"]["render"] == {
        "backend": "remote",
        "image_width": 128,
        "image_height": 96,
        "views": ["front"],
        "animation_frames": None,
        "metadata": {},
    }
    assert (output_dir / "validation_request.json").is_file()
    assert (output_dir / "validation_plan.json").is_file()
    assert (output_dir / "validation_result.json").is_file()


def test_validation_agent_validate_rejects_unknown_format(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded")

    result = runner.invoke(
        app,
        [
            "validate",
            "--task",
            "Validate render evidence.",
            "--format",
            "xml",
            str(image_path),
        ],
    )

    assert result.exit_code == VALIDATION_CLI_ERROR_EXIT_CODE
    assert "xml" in result.output
    assert "text" in result.output
    assert "json" in result.output


def test_validation_agent_run_rejects_unknown_format(tmp_path: Path) -> None:
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate render evidence.",
                "inputs:",
                "  - render.png",
                "requested_templates:",
                "  - render_valid",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(config_path), "--format", "xml"])

    assert result.exit_code == VALIDATION_CLI_ERROR_EXIT_CODE
    assert "xml" in result.output
    assert "text" in result.output
    assert "json" in result.output


def test_validation_agent_accepts_global_logging_options(tmp_path: Path) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate render evidence.",
                "inputs:",
                "  - render.png",
                "requested_templates:",
                "  - render_valid",
                "",
            )
        ),
        encoding="utf-8",
    )
    log_file = tmp_path / "validation-agent.log"

    result = runner.invoke(
        app,
        [
            "--verbose",
            "--log-level",
            "DEBUG",
            "--log-file",
            str(log_file),
            "run",
            str(config_path),
            "--dry-run",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == PASS_EXIT_CODE, result.output
    assert json.loads(result.stdout)["verdict"] == "planned"
    assert log_file.is_file()
    log_content = log_file.read_text(encoding="utf-8")
    assert "Verbose mode enabled" in log_content
    assert "Log level: DEBUG" in log_content


def test_validation_agent_run_default_text_format_prints_summary(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "render.png"
    image_path.write_bytes(b"not decoded during dry-run")
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "task_description: Validate render evidence.",
                "inputs:",
                "  - render.png",
                "requested_templates:",
                "  - render_valid",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(config_path), "--dry-run"])

    assert result.exit_code == PASS_EXIT_CODE, result.output
    assert "Validation Agent" in result.output
    assert "planned" in result.output


def test_validation_agent_missing_config_exits_without_traceback() -> None:
    result = runner.invoke(app, ["run"])

    assert result.exit_code == VALIDATION_CLI_ERROR_EXIT_CODE
    assert "config" in result.output.lower()
    assert "Traceback" not in result.output


def test_validation_agent_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == PASS_EXIT_CODE, result.output
    assert "Validation Agent" in result.stdout
    assert "version" in result.stdout
