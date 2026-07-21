# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from world_understanding.agentic.usd_tasks import config_validate_usd as cvu


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)


def _write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_validate_usd_config_requires_path_and_file(tmp_path: Path) -> None:
    task = cvu.ValidateUSDConfigTask()

    with pytest.raises(ValueError, match="config_path is required"):
        task.run({})

    with pytest.raises(FileNotFoundError):
        task.run({"config_path": tmp_path / "missing.yaml"})


def test_validate_usd_malformed_yaml_uses_value_free_parse_error(
    tmp_path: Path,
) -> None:
    sentinel = "never-disclose-validate-yaml"
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text(f"api_key: [{sentinel}\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        cvu.ValidateUSDConfigTask().run({"config_path": config_path})

    message = str(exc_info.value)
    assert message == f"Unable to parse validate_usd configuration file: {config_path}"
    assert sentinel not in message


def test_validate_usd_explicit_none_config_dict_falls_back_to_file(
    tmp_path: Path,
) -> None:
    config_path = _write_yaml(
        tmp_path / "validate.yaml",
        {"input_usd_path": "scene.usd"},
    )

    context = cvu.ValidateUSDConfigTask().run(
        {"config_dict": None, "config_path": config_path}
    )

    assert context["input_usd_path"] == "scene.usd"


def test_validate_usd_redacts_config_and_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "ZQX713ValidatePathCredentialMNP9"
    config_path = tmp_path / f"Authorization: Bearer {sentinel}"
    listener = _Listener()
    monkeypatch.setattr(cvu, "get_listener", lambda context: listener)

    with pytest.raises(FileNotFoundError) as exc_info:
        cvu.ValidateUSDConfigTask().run({"config_path": config_path})
    assert str(exc_info.value) == "Configuration file not found: <redacted>"

    config_path.write_text(
        "input_usd_path: https://user:path-secret@example.test/scene.usd\n",
        encoding="utf-8",
    )
    cvu.ValidateUSDConfigTask().run({"config_path": config_path})

    observable = "\n".join(listener.infos)
    assert sentinel not in observable
    assert "path-secret" not in observable
    assert "<redacted>" in observable


def test_validate_usd_config_rejects_empty_and_missing_input(tmp_path: Path) -> None:
    task = cvu.ValidateUSDConfigTask()

    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty configuration file"):
        task.run({"config_path": empty_path})

    missing_input = _write_yaml(tmp_path / "missing-input.yaml", {"output_dir": "out"})
    with pytest.raises(ValueError, match="input_usd_path is required"):
        task.run({"config_path": missing_input})


def test_validate_usd_config_rejects_invalid_mode_and_categories(
    tmp_path: Path,
) -> None:
    task = cvu.ValidateUSDConfigTask()
    mode_sentinel = "api_key=never-echo-on-failure-713"

    bad_mode = _write_yaml(
        tmp_path / "bad-mode.yaml",
        {"input_usd_path": "scene.usd", "on_failure": mode_sentinel},
    )
    with pytest.raises(ValueError, match="Invalid on_failure mode") as mode_exc:
        task.run({"config_path": bad_mode})
    assert mode_sentinel not in str(mode_exc.value)

    category_sentinel = "api_key=never-echo-category-713"
    bad_category = _write_yaml(
        tmp_path / "bad-category.yaml",
        {
            "input_usd_path": "scene.usd",
            "validation_config": {"categories": [category_sentinel]},
        },
    )
    with pytest.raises(
        ValueError,
        match="Unknown validation categories",
    ) as category_exc:
        task.run({"config_path": bad_category})
    assert category_sentinel not in str(category_exc.value)


def test_validate_usd_config_loads_defaults_and_passthroughs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(cvu, "get_listener", lambda context: listener)
    config_path = _write_yaml(
        tmp_path / "validate.yaml",
        {
            "input_usd_path": "scene.usd",
            "output_dir": "reports",
            "original_usd_path": "original.usd",
            "baseline_validation": {"issues": []},
        },
    )

    context = cvu.ValidateUSDConfigTask().run({"config_path": config_path})

    assert context["input_usd_path"] == "scene.usd"
    assert context["output_dir"] == "reports"
    assert context["original_usd_path"] == "original.usd"
    assert context["baseline_validation"] == {"issues": []}
    assert context["on_failure"] == "warn"
    assert context["validation_config"] == {
        "categories": list(cvu.DEFAULT_VALIDATION_CATEGORIES),
        "poll_seconds": 300,
    }
    assert any("Categories:" in message for message in listener.infos)


def test_validate_usd_config_normalizes_categories_and_preserves_poll_seconds(
    tmp_path: Path,
) -> None:
    config_path = _write_yaml(
        tmp_path / "validate.yaml",
        {
            "input_usd_path": "scene.usd",
            "on_failure": "block",
            "validation_config": {"categories": ["Basic"], "poll_seconds": 5},
        },
    )

    context = cvu.ValidateUSDConfigTask().run({"config_path": config_path})

    assert context["on_failure"] == "block"
    assert context["validation_config"] == {"categories": ["Basic"], "poll_seconds": 5}


def test_validate_usd_config_prefers_isolated_config_dict(tmp_path: Path) -> None:
    source = {
        "input_usd_path": "scene.usd",
        "validation_config": {
            "api_key": "xy",
            "nested": [{"token": "${VALIDATION_TOKEN}"}],
        },
    }

    context = cvu.ValidateUSDConfigTask().run(
        {
            "config_dict": source,
            "config_path": tmp_path / "does-not-need-to-exist.yaml",
        }
    )

    assert context["validation_config"]["api_key"] == "xy"
    assert context["validation_config"]["nested"] == [{"token": "${VALIDATION_TOKEN}"}]
    assert "categories" not in source["validation_config"]
    assert "poll_seconds" not in source["validation_config"]
