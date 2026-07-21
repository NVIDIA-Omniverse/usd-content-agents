# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material CLI release test for sanitized provider authentication failures."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
)

import material_agent.api as api
import material_agent.cli as cli


class AuthenticationError(RuntimeError):
    status_code = 401


def test_run_maps_fake_provider_401_without_body_or_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setattr(
        api,
        "run_pipeline",
        Mock(
            side_effect=AuthenticationError(
                "401 provider body contained bearer-secret and SDK internals"
            )
        ),
    )
    config = tmp_path / "config.yaml"
    config.write_text("project:\n  name: demo\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc_info:
        cli.run(config)

    observable = repr((printed, logger.method_calls))
    assert exc_info.value.exit_code == 1
    assert MODEL_AUTHENTICATION_FAILURE_MESSAGE in observable
    assert "bearer-secret" not in observable
    assert "Traceback" not in observable


@pytest.mark.parametrize(
    ("command_name", "api_name"),
    (("benchmark", "run_benchmark"), ("evaluate", "run_evaluate")),
)
def test_benchmark_and_evaluate_map_fake_provider_401_without_body(
    command_name: str,
    api_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    printed: list[str] = []
    monkeypatch.setattr(cli, "setup_logging", lambda **_kwargs: logger)
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **_kwargs: printed.append(" ".join(map(str, args))),
    )
    monkeypatch.setattr(
        api,
        api_name,
        Mock(
            side_effect=AuthenticationError(
                "401 provider body contained bearer-secret and SDK internals"
            )
        ),
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc_info:
        getattr(cli, command_name)(config)

    observable = repr((printed, logger.method_calls))
    assert exc_info.value.exit_code == 1
    assert MODEL_AUTHENTICATION_FAILURE_MESSAGE in observable
    assert "bearer-secret" not in observable
    assert "Traceback" not in observable
