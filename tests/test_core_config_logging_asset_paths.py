# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for small shared config, logging, and asset-path helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel
from rich.logging import RichHandler

from world_understanding.agentic.cli.logging import setup_logging as setup_agent_logging
from world_understanding.agentic.config.loader import (
    ConfigError,
    ConfigLoader,
    load_config,
)
from world_understanding.utils import logging as wu_logging
from world_understanding.utils.durable_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    FailurePhase,
    durable_diagnostic,
    log_durable_failure,
)
from world_understanding.utils.usd import asset_paths


class DemoConfig(BaseModel):
    name: str = "default"
    usd_path: str | None = None
    output_dir: str | None = None
    materials_library_path: str | None = None
    renderer: dict[str, Any] = {}

    @classmethod
    def _resolve_paths(cls) -> None:
        return None


_LoggerState = tuple[
    list[logging.Handler],
    int,
    bool,
    bool,
    list[logging.Filter],
]


def _snapshot_logging_state(*prefixes: str) -> dict[logging.Logger, _LoggerState]:
    """Capture every logger that setup_logging can mutate for these prefixes."""
    tracked = {logging.getLogger(), *(logging.getLogger(name) for name in prefixes)}
    for name, candidate in logging.root.manager.loggerDict.items():
        if isinstance(candidate, logging.Logger) and any(
            name.startswith(f"{prefix}.") for prefix in prefixes
        ):
            tracked.add(candidate)
    return {
        logger: (
            list(logger.handlers),
            logger.level,
            logger.propagate,
            logger.disabled,
            list(logger.filters),
        )
        for logger in tracked
    }


def _restore_logging_state(
    states: dict[logging.Logger, _LoggerState], *prefixes: str
) -> None:
    """Restore complete logger state and close test-owned handlers."""
    original_handlers = {
        handler
        for handlers, _level, _propagate, _disabled, _filters in states.values()
        for handler in handlers
    }
    newly_created = {
        candidate
        for name, candidate in logging.root.manager.loggerDict.items()
        if isinstance(candidate, logging.Logger)
        and candidate not in states
        and any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    new_handlers = {
        handler
        for logger in {*states, *newly_created}
        for handler in logger.handlers
        if handler not in original_handlers
    }
    for logger, (handlers, level, propagate, disabled, filters) in states.items():
        logger.handlers.clear()
        logger.handlers.extend(handlers)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = disabled
        logger.filters.clear()
        logger.filters.extend(filters)
    for logger in newly_created:
        logger.handlers.clear()
        logger.filters.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
        logger.disabled = False
    for handler in new_handlers:
        handler.close()


def test_config_loader_success_overrides_paths_and_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "scene",
                "usd_path": "asset.usd",
                "output_dir": "out",
                "renderer": {"backend": "mock"},
            }
        ),
        encoding="utf-8",
    )

    loader = ConfigLoader(DemoConfig)
    config = loader.load(
        config_path,
        overrides={
            "renderer.backend": "ovrtx",
            "renderer.quality": "high",
            "new_section.enabled": True,
            "materials_library_path": "materials.json",
        },
        context={"ignored": True},
    )
    assert config.name == "scene"
    assert config.renderer == {"backend": "ovrtx", "quality": "high"}
    assert config.model_extra is None or "new_section" not in config.model_extra
    assert config.usd_path == str((tmp_path / "asset.usd").resolve())
    assert config.output_dir == str((tmp_path / "out").resolve())
    assert config.materials_library_path == str((tmp_path / "materials.json").resolve())

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    assert load_config(DemoConfig, empty_path).name == "default"

    with pytest.raises(FileNotFoundError):
        loader.load(tmp_path / "missing.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="Failed to parse YAML"):
        loader.load(bad_yaml)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name:\n  nested: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        loader.load(invalid)


def test_agentic_cli_setup_logging_configures_handlers(tmp_path: Path) -> None:
    child = logging.getLogger("demo_agent.child")
    wu_child = logging.getLogger("world_understanding.child")
    logger_states = _snapshot_logging_state("demo_agent", "world_understanding")
    root = logging.getLogger()
    rich_handler = RichHandler()
    configured_handlers: set[logging.Handler] = set()
    root.addHandler(rich_handler)

    try:
        log_file = tmp_path / "agent.log"
        logger = setup_agent_logging("demo_agent", verbose=True, log_file=log_file)
        configured_handlers.update(logger.handlers)
        configured_handlers.update(logging.getLogger("world_understanding").handlers)

        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert any(isinstance(handler, RichHandler) for handler in logger.handlers)
        assert any(
            isinstance(handler, logging.FileHandler) for handler in logger.handlers
        )
        assert logging.getLogger("world_understanding").level == logging.DEBUG
        assert child.level == logging.DEBUG
        assert wu_child.level == logging.DEBUG
        assert rich_handler not in root.handlers
        assert log_file.exists()

        quiet_logger = setup_agent_logging("demo_agent", log_level="WARNING")
        assert quiet_logger.level == logging.WARNING
    finally:
        _restore_logging_state(logger_states, "demo_agent", "world_understanding")
        for handler in configured_handlers:
            handler.close()
        rich_handler.close()


def test_agentic_cli_logging_projects_log_path_and_replaces_open_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger_states = _snapshot_logging_state("safe_log_agent", "world_understanding")
    sentinel = "never-render-this-log-path-credential"
    sensitive_dir = tmp_path / f"client_secret={sentinel}"
    sensitive_dir.mkdir()
    log_file = sensitive_dir / "agent.log"

    try:
        logger = setup_agent_logging("safe_log_agent", log_file=log_file)

        try:
            for handler in logger.handlers:
                handler.flush()
            observable = capsys.readouterr().err + log_file.read_text(encoding="utf-8")
            assert sentinel not in observable
            assert "<redacted>" in observable
        finally:
            for handler in list(logger.handlers):
                handler.close()
            logger.handlers.clear()

        missing_parent = sensitive_dir / "missing" / "agent.log"
        with pytest.raises(RuntimeError, match="^Unable to open log file$") as exc_info:
            setup_agent_logging("safe_log_agent", log_file=missing_parent)

        assert sentinel not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
    finally:
        _restore_logging_state(logger_states, "safe_log_agent", "world_understanding")


def test_world_understanding_logging_config_and_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named = logging.getLogger("wu.logging.test")
    logger_states = _snapshot_logging_state("world_understanding", "wu")
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {"null": {"class": "logging.NullHandler"}},
                "root": {"level": "INFO", "handlers": ["null"]},
            }
        ),
        encoding="utf-8",
    )
    try:
        monkeypatch.setenv("LOGGING_CONFIG", str(config_path))
        assert wu_logging.get_logging_config()["version"] == 1

        named.propagate = False
        named.level = logging.ERROR
        wu_logging.setup_logging()
        assert named.propagate is True
        assert named.level == 0

        monkeypatch.setenv("LOGGING_CONFIG", str(tmp_path / "missing.yaml"))
        warnings: list[str] = []
        monkeypatch.setattr(
            wu_logging.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )
        wu_logging.setup_logging()
        assert "logging configuration file not found" in warnings[0]
    finally:
        _restore_logging_state(logger_states, "world_understanding", "wu")


def test_durable_diagnostic_contract_and_value_free_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    diagnostic = durable_diagnostic(
        "artifact_sync_failed",
        phase=FailurePhase.SYNC_UPLOAD,
        retryable=True,
    )

    assert diagnostic.to_dict() == {
        "schema": DIAGNOSTIC_SCHEMA,
        "code": "artifact_sync_failed",
        "phase": "sync_upload",
        "retryable": True,
    }
    for invalid_code in ("", "invalid-code", "x" * 97):
        with pytest.raises(ValueError, match="^Diagnostic code must be"):
            durable_diagnostic(
                invalid_code,
                phase=FailurePhase.PIPELINE_EXECUTION,
                retryable=False,
            )

    logger = logging.getLogger("durable-diagnostic-contract")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        log_durable_failure(
            logger,
            diagnostic.code,
            phase=FailurePhase.SYNC_UPLOAD,
            retryable=True,
        )

    assert DIAGNOSTIC_SCHEMA in caplog.text
    assert "code=artifact_sync_failed" in caplog.text
    assert "phase=sync_upload" in caplog.text
    assert "retryable=True" in caplog.text


def test_usd_asset_path_helpers(tmp_path: Path) -> None:
    assert asset_paths.is_windows_drive_path("C:/assets/tex.png")
    assert asset_paths.is_windows_drive_path("z:\\assets\\tex.png")
    assert not asset_paths.is_windows_drive_path("cache:C:/asset")

    assert asset_paths.usd_asset_uri_scheme("") == ""
    assert asset_paths.usd_asset_uri_scheme("C:/asset.png") == ""
    assert asset_paths.usd_asset_uri_scheme("folder/usd:asset.png") == ""
    assert asset_paths.usd_asset_uri_scheme("1bad:path") == ""
    assert asset_paths.usd_asset_uri_scheme("bad_scheme:path") == ""
    assert (
        asset_paths.usd_asset_uri_scheme("omniverse://server/asset.usd") == "omniverse"
    )
    assert asset_paths.is_uri_asset_path("s3://bucket/key")
    assert asset_paths.is_absolute_asset_path("/tmp/asset.png")
    assert asset_paths.is_absolute_asset_path("C:/asset.png")
    assert asset_paths.is_unsafe_resolver_asset_path("s3://bucket/key")
    assert asset_paths.is_unsafe_resolver_asset_path(str(tmp_path / "asset.png"))

    base = tmp_path / "base"
    base.mkdir()
    resolved = asset_paths.resolve_relative_asset_path_under_base(
        "textures/a.png", base
    )
    assert resolved == (base / "textures/a.png").resolve()
    assert asset_paths.is_relative_to(resolved, base.resolve())
    assert not asset_paths.is_relative_to(tmp_path, base.resolve())

    with pytest.raises(ValueError, match="empty asset path"):
        asset_paths.resolve_relative_asset_path_under_base("", base)
    with pytest.raises(ValueError, match="resolver URI"):
        asset_paths.resolve_relative_asset_path_under_base("omniverse://asset", base)
    with pytest.raises(ValueError, match="absolute asset"):
        asset_paths.resolve_relative_asset_path_under_base("/tmp/asset.png", base)
    with pytest.raises(ValueError, match="escapes"):
        asset_paths.resolve_relative_asset_path_under_base("../outside.png", base)
