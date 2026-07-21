# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RestoreUSDConfigTask YAML loading with Python-specific tags."""

import errno
import textwrap
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from world_understanding.agentic.usd_tasks.config_restore_usd import (
    RestoreUSDConfigTask,
)
from world_understanding.agentic.usd_tasks.optimizer_models import UsdFormat


def _assert_production_traceback_locals_exclude(
    error: BaseException, sentinel: str
) -> None:
    traceback_frame = error.__traceback__
    production_frames = 0
    while traceback_frame is not None:
        frame = traceback_frame.tb_frame
        if Path(frame.f_code.co_filename).resolve() != Path(__file__).resolve():
            production_frames += 1
            assert sentinel not in repr(frame.f_locals)
        traceback_frame = traceback_frame.tb_next
    assert production_frames > 0


class TestRestoreUSDConfigTaskYAMLFallback:
    """Tests for safe loading of legacy RestoreUSDConfigTask artifacts."""

    def _write_config(self, tmp_path: Path, content: str) -> Path:
        config_path = tmp_path / "restore_config.yaml"
        config_path.write_text(textwrap.dedent(content))
        return config_path

    def test_loads_standard_yaml(self, tmp_path):
        """Standard YAML (no Python tags) loads via safe_load."""
        config_path = self._write_config(
            tmp_path,
            """\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata:
              optimized: true
            """,
        )
        task = RestoreUSDConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)

        assert result["original_usd_path"] == "/tmp/original.usd"
        assert result["predictions_path"] == "/tmp/predictions.jsonl"
        assert result["optimization_metadata"] == {"optimized": True}

    def test_loads_yaml_with_python_none_tag(self, tmp_path: Path) -> None:
        """YAML with the legacy ``!!python/none`` scalar stays compatible."""
        config_path = tmp_path / "restore_config.yaml"
        config_path.write_text(
            "original_usd_path: /tmp/original.usd\n"
            "predictions_path: /tmp/predictions.jsonl\n"
            "output_predictions_path: /tmp/output.jsonl\n"
            "optimization_metadata:\n"
            "  none_val: !!python/none ''\n"
        )
        task = RestoreUSDConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)

        assert result["original_usd_path"] == "/tmp/original.usd"
        assert result["optimization_metadata"]["none_val"] is None

    def test_loads_allowlisted_legacy_usd_format_tag(self, tmp_path: Path) -> None:
        """Historical optimizer enum artifacts load without FullLoader."""
        config_path = self._write_config(
            tmp_path,
            """\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata:
              optimization_config:
                output_format: !!python/object/apply:world_understanding.agentic.usd_tasks.optimizer_models.UsdFormat
                - usdc
            """,
        )

        result = RestoreUSDConfigTask().run({"config_path": config_path}, None)

        assert (
            result["optimization_metadata"]["optimization_config"]["output_format"]
            is UsdFormat.USDC
        )

    @pytest.mark.parametrize("value", ["usdz", "", "USDC"])
    def test_rejects_invalid_legacy_usd_format_value(
        self, tmp_path: Path, value: str
    ) -> None:
        config_path = self._write_config(
            tmp_path,
            f"""\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata:
              output_format: !!python/object/apply:world_understanding.agentic.usd_tasks.optimizer_models.UsdFormat
              - {value!r}
            """,
        )

        with pytest.raises(ValueError, match="Unable to parse restore_usd"):
            RestoreUSDConfigTask().run({"config_path": config_path}, None)

    def test_rejects_unrelated_python_object_tags(self, tmp_path: Path) -> None:
        """Legacy compatibility must not enable arbitrary Python tags."""
        sentinel = "LEAK42"
        config_path = self._write_config(
            tmp_path,
            f"""\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata:
              unsafe: !!python/name:os.system '{sentinel}'
            """,
        )

        with pytest.raises(ValueError) as exc_info:
            RestoreUSDConfigTask().run({"config_path": str(config_path)}, None)

        message = str(exc_info.value)
        assert message == (
            f"Unable to parse restore_usd configuration file: {config_path}"
        )
        assert sentinel not in message
        assert "os.system" not in message
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    def test_missing_config_path_raises(self) -> None:
        """Missing config_path raises ValueError."""
        task = RestoreUSDConfigTask()
        with pytest.raises(
            ValueError,
            match="config_dict or config_path is required",
        ):
            task.run({}, None)

    def test_empty_config_raises(self, tmp_path):
        """Empty config file raises ValueError."""
        config_path = self._write_config(tmp_path, "")
        task = RestoreUSDConfigTask()
        with pytest.raises(ValueError, match="Empty configuration"):
            task.run({"config_path": str(config_path)}, None)

    def test_config_dict_is_preferred_and_isolated(self, tmp_path: Path) -> None:
        source: dict[str, Any] = {
            "original_usd_path": "/tmp/original.usd",
            "predictions_path": "/tmp/predictions.jsonl",
            "output_predictions_path": "/tmp/output.jsonl",
            "optimization_metadata": {
                "api_key": "xy",
                "nested": [{"secret": "${RESTORE_SECRET}"}],
            },
        }
        result = RestoreUSDConfigTask().run(
            {
                "config_dict": source,
                "config_path": tmp_path / "does-not-need-to-exist.yaml",
            }
        )

        assert result["optimization_metadata"] == source["optimization_metadata"]
        assert result["optimization_metadata"] is not source["optimization_metadata"]
        assert (
            result["optimization_metadata"]["nested"]
            is not source["optimization_metadata"]["nested"]
        )
        result["optimization_metadata"]["nested"][0]["secret"] = "changed"
        assert (
            source["optimization_metadata"]["nested"][0]["secret"]
            == "${RESTORE_SECRET}"
        )

    def test_none_config_dict_falls_back_to_config_path(self, tmp_path: Path) -> None:
        config_path = self._write_config(
            tmp_path,
            """\
            original_usd_path: /tmp/original.usd
            predictions_path: /tmp/predictions.jsonl
            output_predictions_path: /tmp/output.jsonl
            optimization_metadata: {}
            """,
        )

        result = RestoreUSDConfigTask().run(
            {"config_dict": None, "config_path": str(config_path)}, None
        )

        assert result["original_usd_path"] == "/tmp/original.usd"

    def test_normalizes_malformed_yaml_without_rendering_source(
        self, tmp_path: Path
    ) -> None:
        sentinel = "never-render-this-restore-credential"
        config_path = self._write_config(
            tmp_path,
            f"api_key: [{sentinel}\n",
        )

        with pytest.raises(ValueError) as exc_info:
            RestoreUSDConfigTask().run({"config_path": str(config_path)}, None)

        message = str(exc_info.value)
        assert message == (
            f"Unable to parse restore_usd configuration file: {config_path}"
        )
        assert sentinel not in message
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert sentinel not in repr(
            (exc_info.value.__cause__, exc_info.value.__context__)
        )
        _assert_production_traceback_locals_exclude(exc_info.value, sentinel)

    def test_private_loader_boundary_drops_raw_context_from_traceback(
        self, tmp_path: Path
    ) -> None:
        sentinel = "api_key=restore-loader-frame-secret-713"
        config_path = self._write_config(tmp_path, f"api_key: [{sentinel}\n")
        listener = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            RestoreUSDConfigTask()._load_config(
                {"config_path": config_path, "runtime_secret": sentinel},
                listener,
            )

        assert str(exc_info.value) == (
            f"Unable to parse restore_usd configuration file: {config_path}"
        )
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        _assert_production_traceback_locals_exclude(exc_info.value, sentinel)

    def test_missing_path_is_value_free(self, tmp_path: Path) -> None:
        sentinel = "ZQX713RestorePathCredentialMNP9"
        config_path = tmp_path / f"Authorization: Bearer {sentinel}"
        listener = MagicMock()
        context = {"event_listener": listener, "config_path": config_path}

        with pytest.raises(FileNotFoundError) as missing_exc:
            RestoreUSDConfigTask().run(context)
        assert str(missing_exc.value) == "Configuration file not found: <redacted>"
        assert missing_exc.value.__cause__ is None
        assert missing_exc.value.__context__ is None
        assert sentinel not in f"{missing_exc.value}\n{listener.mock_calls!r}"

    @pytest.mark.parametrize(
        ("method_name", "error_type"),
        [("exists", PermissionError), ("open", FileNotFoundError)],
    )
    def test_read_errors_are_value_free_and_sever_rejected_exception_graph(
        self,
        method_name: str,
        error_type: type[OSError],
        tmp_path: Path,
    ) -> None:
        sentinel = "ZQX713RestoreReadContextCredentialMNP9"
        config_path = tmp_path / f"Authorization: Bearer {sentinel}"
        if method_name == "open":
            config_path.write_text("value: present\n", encoding="utf-8")
        listener = MagicMock()
        context = {"event_listener": listener, "config_path": config_path}
        rejected_error = error_type(
            errno.EACCES,
            f"read provider echoed {sentinel}",
            str(config_path),
        )

        with (
            patch.object(Path, method_name, side_effect=rejected_error),
            pytest.raises(error_type) as exc_info,
        ):
            RestoreUSDConfigTask().run(context)

        assert exc_info.value.errno == errno.EACCES
        assert exc_info.value.filename == "<redacted>"
        assert "Unable to read restore_usd configuration file" in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        observable = "\n".join(
            (
                "".join(traceback.format_exception(exc_info.value)),
                repr(
                    (
                        exc_info.value,
                        exc_info.value.__cause__,
                        exc_info.value.__context__,
                    )
                ),
                repr(listener.mock_calls),
            )
        )
        assert sentinel not in observable
        _assert_production_traceback_locals_exclude(exc_info.value, sentinel)

    def test_config_value_paths_are_redacted_in_logs(self) -> None:
        sentinel = "restore-path-secret"
        listener = MagicMock()
        RestoreUSDConfigTask().run(
            {
                "event_listener": listener,
                "config_dict": {
                    "original_usd_path": (
                        f"https://user:{sentinel}@example.test/original.usd"
                    ),
                    "predictions_path": (
                        f"https://example.test/predictions?sig={sentinel}"
                    ),
                    "output_predictions_path": (
                        f"https://example.test/output?token={sentinel}"
                    ),
                    "optimization_metadata": {},
                },
            }
        )

        observable = repr(listener.mock_calls)
        assert sentinel not in observable
        assert observable.count("<redacted>") >= 3
