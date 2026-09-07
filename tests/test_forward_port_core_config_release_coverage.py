# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from world_understanding.agentic.config import (
    isolation,
    model_credentials,
    unknown_keys,
)


class _ConfigMode(Enum):
    ACTIVE = "active"


def test_yaml_config_normalization_covers_supported_container_leaves(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    normalized = isolation.normalize_yaml_config_value(
        {
            "mode": _ConfigMode.ACTIVE,
            "path": config_path,
            "sequence": ("second", 1),
            "ordered": frozenset({"z", "a", 3}),
        }
    )

    assert normalized == {
        "mode": "active",
        "path": str(config_path),
        "sequence": ["second", 1],
        "ordered": [3, "a", "z"],
    }


def test_clone_config_containers_clones_frozenset_and_preserves_leaves() -> None:
    opaque_leaf = object()
    source = frozenset({opaque_leaf})

    cloned = isolation.clone_config_containers(source)

    assert cloned == source
    assert cloned is not source
    assert next(iter(cloned)) is opaque_leaf


def test_unknown_strict_keys_report_plain_and_suggested_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("forward-port-core-config-unknown-keys")
    schema = {"section": {"enabled": True}}
    config = {"section": {"enabld": False, "unrelated_option": True}}

    with caplog.at_level(logging.WARNING, logger=logger.name):
        unknown_keys.warn_unknown_nested_config_keys(
            config,
            schema,
            logger,
            strict_paths=[("section",)],
        )

    assert "section.enabld" in caplog.text
    assert "did you mean 'section.enabled'" in caplog.text
    assert "section.unrelated_option" in caplog.text


def test_unknown_key_suggestion_and_runtime_wiring_boundaries() -> None:
    assert unknown_keys._unambiguous_suggestion("enabld", ("enabled",)) == "enabled"
    assert unknown_keys._unambiguous_suggestion("unrelated", ("enabled",)) is None
    assert not unknown_keys._is_runtime_wiring_key(("section",), "source_path")


def test_model_config_deep_merge_preserves_nested_defaults() -> None:
    defaults = {"model": {"backend": "nim", "temperature": 0.0}}

    merged = model_credentials._deep_merge(
        defaults,
        {"model": {"temperature": 0.5}},
    )

    assert merged == {"model": {"backend": "nim", "temperature": 0.5}}
    assert defaults == {"model": {"backend": "nim", "temperature": 0.0}}


def test_model_requirement_handles_unknown_and_authenticated_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert model_credentials._missing_requirement("custom-backend", {}) is None

    monkeypatch.setattr(
        model_credentials,
        "_resolved_endpoint_key",
        lambda _config: ("endpoint-key", False),
    )
    monkeypatch.setattr(
        model_credentials,
        "get_nim_api_key_for_base_url",
        lambda _base_url, _endpoint_key: "resolved-key",
    )

    assert model_credentials._missing_requirement("nim", {}) is None


def test_model_credential_validation_skips_invalid_and_unregulated_backends(
    tmp_path: Path,
) -> None:
    observed: list[tuple[str, dict[str, Any], str]] = []

    def iter_configs(
        step_name: str,
        step_config: dict[str, Any],
        path: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        observed.append((step_name, step_config, path))
        return [
            (f"{path}.vlm", {"backend": 7}),
            (f"{path}.llm", {"backend": "custom-backend"}),
        ]

    model_credentials.validate_selected_model_credentials(
        {"project": {}, "steps": {"predict": {"enabled": True}}},
        tmp_path / "config.yaml",
        (),
        (),
        model_config_iterator=iter_configs,
    )

    assert observed == [
        ("predict", {"enabled": True}, "steps.predict"),
    ]
