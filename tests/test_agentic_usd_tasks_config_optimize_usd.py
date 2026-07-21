# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OptimizeUSDConfigTask backend validation logic."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from world_understanding.agentic.usd_tasks.config_optimize_usd import (
    OptimizeUSDConfigTask,
)


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "optimize_config.yaml"
    config_path.write_text(textwrap.dedent(content))
    return config_path


def _minimal_config(extra: str = "") -> str:
    return textwrap.dedent(
        f"""\
        input_usd_path: /tmp/input.usd
        output_usd_path: /tmp/output.usd
        optimization_config:
          backend: remote
        {extra}
        """
    )


class TestOptimizeUSDConfigTaskBackendValidation:
    """Tests for backend validation in OptimizeUSDConfigTask (lines 132-152)."""

    def test_invalid_backend_raises_value_error(self, tmp_path):
        """An unrecognised backend value must raise ValueError."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: cloud
            """,
        )
        task = OptimizeUSDConfigTask()
        with pytest.raises(
            ValueError,
            match="^Invalid optimization backend; expected 'local' or 'remote'$",
        ):
            task.run({"config_path": str(config_path)}, None)

    @pytest.mark.parametrize(
        ("optimization_config", "expected_message"),
        [
            (
                {
                    "backend": "remote",
                    "scene_optimizer_settings": {
                        "generate_report": (
                            "https://optimizer.example.test/run?sig="
                            "never-disclose-settings"
                        )
                    },
                },
                "Invalid scene_optimizer_settings in config",
            ),
            (
                {
                    "backend": (
                        "https://optimizer.example.test/run?token="
                        "never-disclose-backend"
                    )
                },
                "Invalid optimization backend; expected 'local' or 'remote'",
            ),
            (
                {
                    "backend": "remote",
                    "flatten_prototypes": (
                        "https://optimizer.example.test/run?key=never-disclose-flatten"
                    ),
                },
                "flatten_prototypes must be a boolean",
            ),
        ],
    )
    def test_invalid_config_diagnostics_are_value_free(
        self,
        optimization_config: dict[str, object],
        expected_message: str,
    ) -> None:
        listener = MagicMock()
        context = {
            "event_listener": listener,
            "config_dict": {
                "input_usd_path": "/tmp/input.usd",
                "output_usd_path": "/tmp/output.usd",
                "optimization_config": optimization_config,
            },
        }

        with pytest.raises(ValueError) as exc_info:
            OptimizeUSDConfigTask().run(context)

        assert str(exc_info.value) == expected_message
        assert exc_info.value.__cause__ is None
        observable = f"{exc_info.value}\n{listener.mock_calls!r}"
        for fragment in (
            "never-disclose-settings",
            "never-disclose-backend",
            "never-disclose-flatten",
            "optimizer.example.test",
        ):
            assert fragment not in observable

    def test_config_path_values_are_redacted_in_diagnostics(self) -> None:
        sentinel = "never-disclose-path-signature"
        listener = MagicMock()
        context = {
            "event_listener": listener,
            "config_dict": {
                "input_usd_path": (
                    f"https://storage.example.test/input.usd?sig={sentinel}"
                ),
                "output_usd_path": (
                    f"https://storage.example.test/output.usd?token={sentinel}"
                ),
                "optimization_config": {"backend": "remote"},
            },
        }

        OptimizeUSDConfigTask().run(context)

        observable = repr(listener.mock_calls)
        assert sentinel not in observable
        assert "storage.example.test" not in observable
        assert observable.count("<redacted>") >= 2

    def test_config_file_path_and_read_errors_are_value_free(
        self,
        tmp_path: Path,
    ) -> None:
        sentinel = "ZQX713ConfigPathCredentialMNP9"
        config_path = tmp_path / f"Authorization: Bearer {sentinel}"
        listener = MagicMock()
        context = {
            "event_listener": listener,
            "config_path": str(config_path),
        }

        with pytest.raises(FileNotFoundError) as missing_exc:
            OptimizeUSDConfigTask().run(context)
        assert str(missing_exc.value) == "Configuration file not found: <redacted>"

        config_path.write_text("input_usd_path: /tmp/input.usd\n", encoding="utf-8")
        with (
            patch(
                "pathlib.Path.open",
                side_effect=PermissionError(f"read denied for {sentinel}"),
            ),
            pytest.raises(OSError) as read_exc,
        ):
            OptimizeUSDConfigTask().run(context)

        assert str(read_exc.value) == (
            "Unable to read optimize_usd configuration file: <redacted>"
        )
        assert read_exc.value.__cause__ is None
        observable = f"{missing_exc.value}\n{read_exc.value}\n{listener.mock_calls!r}"
        assert sentinel not in observable
        assert "<redacted>" in observable

    def test_explicit_none_config_dict_falls_back_to_file(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, _minimal_config())

        result = OptimizeUSDConfigTask().run(
            {"config_dict": None, "config_path": config_path}
        )

        assert result["input_usd_path"] == "/tmp/input.usd"
        assert result["output_usd_path"] == "/tmp/output.usd"

    def test_malformed_yaml_uses_value_free_parse_error(self, tmp_path):
        sentinel = "never-disclose-optimize-yaml"
        config_path = _write_config(tmp_path, f"api_key: [{sentinel}\n")

        with pytest.raises(ValueError) as exc_info:
            OptimizeUSDConfigTask().run({"config_path": str(config_path)}, None)

        message = str(exc_info.value)
        assert message == (
            f"Unable to parse optimize_usd configuration file: {config_path}"
        )
        assert sentinel not in message

    def test_nvcf_backend_sets_poll_seconds_default(self, tmp_path):
        """backend='remote' without poll_seconds should default to 300."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: remote
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert result["optimization_config"]["poll_seconds"] == 300

    def test_nvcf_backend_respects_explicit_poll_seconds(self, tmp_path):
        """backend='remote' with an explicit poll_seconds should keep the user value."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: remote
              poll_seconds: 60
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert result["optimization_config"]["poll_seconds"] == 60

    def test_local_backend_skips_poll_seconds_default(self, tmp_path):
        """backend='local' must NOT inject poll_seconds into optimization_config."""
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
              scene_optimizer_settings:
                enable_deinstance: false
            """,
        )
        task = OptimizeUSDConfigTask()
        result = task.run({"config_path": str(config_path)}, None)
        assert "poll_seconds" not in result["optimization_config"]

    def test_local_backend_with_deinstance_no_warning(self, tmp_path, caplog):
        """backend='local' + enable_deinstance=True must not emit a warning (now supported)."""
        import logging

        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
              scene_optimizer_settings:
                enable_deinstance: true
            """,
        )
        task = OptimizeUSDConfigTask()
        with caplog.at_level(logging.WARNING):
            task.run({"config_path": str(config_path)}, None)

        deinstance_warnings = [
            r.message
            for r in caplog.records
            if r.levelno == logging.WARNING and "deinstance" in r.message
        ]
        assert deinstance_warnings == []

    def test_scene_optimizer_settings_require_one_enabled_operation(self, tmp_path):
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
              scene_optimizer_settings:
                enable_deinstance: false
                enable_split_meshes: false
                enable_deduplicate: false
            """,
        )

        with pytest.raises(ValueError, match="At least one operation must be enabled"):
            OptimizeUSDConfigTask().run({"config_path": str(config_path)}, None)

    def test_default_scene_optimizer_settings_require_enabled_operations(
        self, tmp_path, monkeypatch
    ):
        config_path = _write_config(
            tmp_path,
            """\
            input_usd_path: /tmp/input.usd
            output_usd_path: /tmp/output.usd
            optimization_config:
              backend: local
            """,
        )
        task = OptimizeUSDConfigTask()
        monkeypatch.setattr(task, "_build_enabled_operations", lambda _settings: [])

        with pytest.raises(ValueError, match="At least one operation must be enabled"):
            task.run({"config_path": str(config_path)}, None)

    def test_config_dict_is_preferred_and_isolated(self, tmp_path: Path) -> None:
        source = {
            "input_usd_path": "/tmp/input.usd",
            "output_usd_path": "/tmp/output.usd",
            "optimization_config": {
                "backend": "remote",
                "api_key": "xy",
                "nested": [{"token": "${OPTIMIZER_TOKEN}"}],
            },
        }
        result = OptimizeUSDConfigTask().run(
            {
                "config_dict": source,
                "config_path": tmp_path / "does-not-need-to-exist.yaml",
            }
        )

        assert result["optimization_config"]["api_key"] == "xy"
        assert result["optimization_config"]["nested"] == [
            {"token": "${OPTIMIZER_TOKEN}"}
        ]
        assert result["optimization_config"]["poll_seconds"] == 300
        assert "poll_seconds" not in source["optimization_config"]
        result["optimization_config"]["nested"][0]["token"] = "mutated"
        assert source["optimization_config"]["nested"][0]["token"] == (
            "${OPTIMIZER_TOKEN}"
        )
