# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small branch coverage for physics-agent helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from physics_agent.config.path_resolver import ProjectPathResolver
from physics_agent.config.usd_suffixes import default_apply_physics_output_suffix
from physics_agent.config.validator import ConfigValidator
from physics_agent.tasks.predictions import SavePredictionsTask, _load_dataset_from_path
from physics_agent.utils import display_results, format_prediction_output, get_version


class _Store:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def exists(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> Any:
        return self.data[key]


def test_utils_version_fallback_and_formatting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from importlib.metadata import PackageNotFoundError

    import physics_agent.utils as utils

    monkeypatch.setattr(
        utils,
        "version",
        lambda _name: (_ for _ in ()).throw(PackageNotFoundError("missing")),
    )

    assert get_version() == "0.0.1-dev"
    assert format_prediction_output(
        {"id": "p1", "vlm_response": "box", "confidence": 0.8},
        include_confidence=False,
    ) == {"id": "p1", "image_path": "", "classification": "box"}
    display_results(
        {
            "total_entries": 2,
            "successful": 1,
            "failed": 1,
            "output_path": "/tmp/out.jsonl",
        },
        title="Demo",
    )
    captured = capsys.readouterr()
    assert "Demo" in captured.out
    assert "Total Entries" in captured.out
    assert "/tmp/out.jsonl" in captured.out


def test_default_apply_physics_output_suffix_unknown() -> None:
    assert default_apply_physics_output_suffix(".abc") == ".usd"


def test_config_validator_none_section_and_unknown_step(
    caplog: pytest.LogCaptureFixture,
) -> None:
    validator = ConfigValidator()

    with pytest.raises(ValueError, match="project.name"):
        validator.validate(
            {
                "project": None,
                "input": {"usd_path": "asset.usd"},
                "steps": {"mystery": {}},
            }
        )


def test_project_path_resolver_warns_for_missing_reference(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    resolver = ProjectPathResolver(
        {
            "project": {"name": "demo", "working_dir": str(tmp_path / "work")},
            "input": {
                "usd_path": str(usd),
                "reference_images": [str(tmp_path / "missing.png")],
            },
        },
        tmp_path / "config.yaml",
    )

    resolver.validate_input_paths()


def test_load_dataset_from_path_missing_blank_bad_and_valid(tmp_path: Path) -> None:
    assert _load_dataset_from_path(None) == []
    assert _load_dataset_from_path(tmp_path / "missing.jsonl") == []

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('\nnot-json\n{"id":"a"}\n[]\n', encoding="utf-8")
    assert _load_dataset_from_path(dataset) == [{"id": "a"}]


def test_save_predictions_prefers_object_store_dataset(tmp_path: Path) -> None:
    out = tmp_path / "predictions.jsonl"
    context = {
        "predictions_path": str(out),
        "output_key": "classification",
        "dataset_path": str(tmp_path / "unused.jsonl"),
    }
    store = _Store(
        {
            "predictions": [
                {
                    "id": "p1",
                    "vlm_response": {
                        "classification": {
                            "density": 100.0,
                            "mass": 0.1,
                            "mass_scale": 1.0,
                        }
                    },
                    "quality_warnings": [{"code": "existing"}],
                }
            ],
            "dataset": [{"id": "p1", "bbox": [0, 0, 0, 1, 1, 1]}],
        }
    )

    result = SavePredictionsTask().run(context, object_store=store)

    assert result["predictions_saved"] is True
    assert out.exists()
