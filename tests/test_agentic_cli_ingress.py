# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared agent CLI ingress boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from world_understanding.agentic.cli import (
    load_cli_config_mapping,
    normalize_cli_step_filters,
)
from world_understanding.agentic.config import (
    ConfigEmptyError,
    ConfigParseError,
    ConfigStructureError,
)


@pytest.mark.parametrize(
    ("payload", "error_type", "message"),
    [
        (
            "sentinel: predict\nsteps: [\n",
            ConfigParseError,
            "Unable to parse pipeline configuration",
        ),
        ("", ConfigEmptyError, "Pipeline configuration is empty"),
        ("- predict\n", ConfigStructureError, "must be a mapping"),
    ],
)
def test_load_cli_config_mapping_rejects_invalid_yaml_roots_without_values(
    tmp_path: Path,
    payload: str,
    error_type: type[ValueError],
    message: str,
) -> None:
    sentinel = "never-render-this-cli-config-value"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(payload.replace("predict", sentinel), encoding="utf-8")

    with pytest.raises(error_type) as exc_info:
        load_cli_config_mapping(config_path)

    assert message in str(exc_info.value)
    assert sentinel not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_load_cli_config_mapping_preserves_valid_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: demo\n", encoding="utf-8")

    assert load_cli_config_mapping(config_path) == {"project": {"name": "demo"}}


@pytest.mark.parametrize(
    ("skip", "only", "message"),
    [
        (None, "predcit", "Invalid --only step name(s): 'predcit'"),
        ("predict,,apply", None, "--skip contains an empty step name"),
        ("predict", "apply", "--skip and --only cannot be used together"),
    ],
)
def test_normalize_cli_step_filters_rejects_ambiguous_filters(
    skip: str | None,
    only: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_cli_step_filters(
            skip=skip,
            only=only,
            valid_steps=("predict", "apply"),
        )

    assert message in str(exc_info.value)
    assert "Valid steps: predict, apply" in str(exc_info.value) or skip == "predict"


def test_normalize_cli_step_filters_trims_and_deduplicates() -> None:
    skip, only = normalize_cli_step_filters(
        skip=None,
        only=" predict,apply,predict ",
        valid_steps=("predict", "apply"),
    )

    assert skip == []
    assert only == ["predict", "apply"]
