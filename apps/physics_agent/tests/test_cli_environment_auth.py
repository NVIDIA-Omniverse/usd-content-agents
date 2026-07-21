# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics CLI release tests for credential preflight and auth failures."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
import yaml
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
)

import physics_agent.api as api
import physics_agent.cli as cli


class AuthenticationError(RuntimeError):
    status_code = 401


def _patch_cli(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, list[str]]:
    logger = Mock()
    printed: list[str] = []
    monkeypatch.setattr(cli, "setup_logging", lambda **_kwargs: logger)
    monkeypatch.setattr(
        cli, "get_listener", lambda *_args, **_kwargs: SimpleNamespace(event=Mock())
    )
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **_kwargs: printed.append(" ".join(map(str, args))),
    )
    monkeypatch.setattr(api, "CLIEventListener", lambda **_kwargs: object())
    return logger, printed


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_run_fails_before_pipeline_and_output_for_missing_model_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _logger, printed = _patch_cli(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    working_dir = tmp_path / "must-not-exist"
    config = _write_config(
        tmp_path / "config.yaml",
        {
            "project": {"working_dir": str(working_dir)},
            "steps": {
                "predict": {
                    "enabled": True,
                    "vlm": {"backend": "nim", "model": "example"},
                }
            },
        },
    )
    run_pipeline = Mock()
    monkeypatch.setattr(api, "run_pipeline", run_pipeline)

    with pytest.raises(typer.Exit) as exc_info:
        cli.run(config)

    assert exc_info.value.exit_code == 1
    run_pipeline.assert_not_called()
    assert not working_dir.exists()
    assert "NVIDIA_API_KEY" in "\n".join(printed)


def test_run_dry_run_skips_model_credential_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _logger, printed = _patch_cli(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path / "config.yaml",
        {
            "project": {"name": "demo"},
            "steps": {
                "predict": {
                    "enabled": True,
                    "vlm": {"backend": "nim", "model": "example"},
                }
            },
        },
    )
    run_pipeline = Mock()
    monkeypatch.setattr(api, "run_pipeline", run_pipeline)

    cli.run(config, dry_run=True)

    run_pipeline.assert_not_called()
    assert "Dry run complete" in "\n".join(printed)


def test_run_maps_fake_provider_401_without_body_or_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger, printed = _patch_cli(monkeypatch)
    config = _write_config(
        tmp_path / "config.yaml",
        {"project": {"name": "demo"}, "steps": {"apply_physics": {"enabled": True}}},
    )
    monkeypatch.setattr(
        api,
        "run_pipeline",
        Mock(
            side_effect=AuthenticationError(
                "401 provider body contained bearer-secret and SDK internals"
            )
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.run(config)

    observable = repr((printed, logger.method_calls))
    assert exc_info.value.exit_code == 1
    assert MODEL_AUTHENTICATION_FAILURE_MESSAGE in observable
    assert "bearer-secret" not in observable
    assert "Traceback" not in observable
