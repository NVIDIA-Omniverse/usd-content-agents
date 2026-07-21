# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge coverage for quality, dataset loading, and reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physics_agent.functions.mass_scale_quality import (
    build_mass_scale_quality_warnings,
    extract_bbox_metrics_meters,
    get_physical_properties,
    has_mass_scale_suspicious_warning,
    merge_quality_warnings,
)
from physics_agent.tasks.dataset_loading import DatasetLoadingTask
from physics_agent.tasks.reporting import GeneratePredictionReportTask


class _Listener:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_mass_scale_quality_edge_inputs() -> None:
    assert extract_bbox_metrics_meters(None) == {}
    assert extract_bbox_metrics_meters({"metadata": {"world_bbox_meters": "bad"}}) == {}
    assert (
        extract_bbox_metrics_meters(
            {"metadata": {"world_bbox_meters": {"size": [1, "bad", 3]}}}
        )
        == {}
    )
    assert (
        extract_bbox_metrics_meters(
            {"text": "Dimensions (meters): width=-1m, height=2m, depth=3m"}
        )
        == {}
    )
    assert extract_bbox_metrics_meters({"text": "Bounding box volume: -2 m^3"}) == {}
    assert get_physical_properties({"classification": "not-a-dict"}) == {}
    assert (
        build_mass_scale_quality_warnings(
            {"classification": {"physical_properties": {"estimated_mass_kg": 1}}},
            {},
        )
        == []
    )

    large_only = build_mass_scale_quality_warnings(
        {"classification": {"physical_properties": {"estimated_mass_kg": 10}}},
        {"metadata": {"world_bbox_meters": {"size": [6, 1, 1]}}},
    )
    assert [warning["code"] for warning in large_only] == ["large_component_scale"]
    assert has_mass_scale_suspicious_warning({"quality_warnings": "bad"}) is False
    assert merge_quality_warnings(
        [{"code": "a"}, "skip"], [{"code": "a"}, {"code": "b"}]
    ) == [
        {"code": "a"},
        {"code": "b"},
    ]


def test_dataset_loading_validate_entry_edges(tmp_path: Path) -> None:
    task = DatasetLoadingTask()
    listener = _Listener()
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    assert task._validate_entry({}, dataset_path, listener) is False
    assert task._validate_entry({"id": "a"}, dataset_path, listener) is False
    assert (
        task._validate_entry(
            {"id": "a", "images": ["missing.png"]}, dataset_path, listener
        )
        is False
    )
    assert (
        task._validate_entry(
            {"id": "a", "image_path": "missing.png"}, dataset_path, listener
        )
        is False
    )
    assert (
        task._validate_entry(
            {"id": "a", "media": {"images": [{"path": "missing.png"}]}},
            dataset_path,
            listener,
        )
        is False
    )
    assert (
        task._validate_entry(
            {"id": "a", "media": {"images": ["bad"]}},
            dataset_path,
            listener,
        )
        is False
    )


def test_reporting_skip_and_string_classification_edges(tmp_path: Path) -> None:
    task = GeneratePredictionReportTask()
    context: dict[str, Any] = {}
    assert task.run(context) is context
    missing = {"predictions_path": str(tmp_path / "missing.jsonl")}
    assert task.run(missing) is missing
    assert task._format_token_stats_html({}) == ""
    assert (
        task._format_token_stats_html({"total_tokens": 0, "invocation_count": 0}) == ""
    )

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "asset", "classification": "raw text"}) + "\n",
        encoding="utf-8",
    )
    result = task.run({"predictions_path": str(predictions_path)})
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "raw text" in report
