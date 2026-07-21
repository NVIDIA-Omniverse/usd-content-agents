# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for identifying unique material names from predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from material_agent.materials import FALLBACK_MATERIAL_NAME
from material_agent.tasks import identify_materials as identify_materials_module
from material_agent.tasks.identify_materials import IdentifyUniqueMaterialsTask
from material_agent.tasks.material_retrieval import MaterialRetrievalTask


def _write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


def test_unknown_material_is_counted_and_uses_fallback(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "id": "/World/HiddenPart",
                "materials": {
                    "material": "__UNKNOWN__",
                    "reason": "no visible geometry",
                },
            },
            {"id": "/World/VisiblePart", "materials": {"material": "Steel"}},
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Steel", FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["/World/HiddenPart"]


def test_unknown_prediction_uses_shared_prediction_id_keys(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "prim_path": "/World/HiddenPart",
                "materials": {
                    "material": "__UNKNOWN__",
                    "reason": "no visible geometry",
                },
            }
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == [FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["/World/HiddenPart"]


def test_nested_unknown_predictions_are_not_double_counted(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "id": "batch-1",
                "predictions": [
                    {
                        "id": "/World/HiddenPart",
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    {
                        "id": "/World/VisiblePart",
                        "materials": {"material": "Steel"},
                    },
                ],
            }
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Steel", FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["/World/HiddenPart"]


def test_nested_material_extraction_traverses_each_prediction_once(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "predictions": [
                    {
                        "predictions": [
                            {
                                "id": "/World/VisiblePart",
                                "materials": {"material": "Steel"},
                            }
                        ]
                    }
                ]
            }
        ],
    )
    task = IdentifyUniqueMaterialsTask()
    original = task._extract_materials_from_prediction
    call_count = 0

    def wrapped_extract_materials(prediction):
        nonlocal call_count
        call_count += 1
        return original(prediction)

    task._extract_materials_from_prediction = wrapped_extract_materials

    result = task.run({"predictions_path": str(predictions_path)})

    assert result["unique_materials"] == ["Steel"]
    assert call_count == 3


def test_string_predictions_keep_working_with_unknown_filter(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(["__UNKNOWN__", "Steel"]), encoding="utf-8")

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Steel", FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["index:0"]


def test_empty_predictions_debug_logging_and_non_int_existing_count(
    monkeypatch, tmp_path: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        identify_materials_module.logger,
        "isEnabledFor",
        lambda level: level == logging.DEBUG,
    )

    result = IdentifyUniqueMaterialsTask().run(
        {
            "predictions_path": str(predictions_path),
            "unknown_material_predictions": "stale",
        }
    )

    assert result["unique_materials"] == []
    assert result["total_predictions"] == 0
    assert result["unknown_material_predictions"] == 0


def test_jsonl_loader_skips_blank_lines(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        '{"material": "Steel"}\n\n{"material": "Plastic"}',
        encoding="utf-8",
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Plastic", "Steel"]


def test_preserves_existing_unknown_material_count(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [{"id": "/World/VisiblePart", "materials": {"material": "Steel"}}],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {
            "predictions_path": str(predictions_path),
            "unknown_material_predictions": 1,
        }
    )

    assert result["unique_materials"] == ["Steel"]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == []


def test_alternate_prediction_containers_are_counted(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "results": [
                    {
                        "id": "/World/HiddenPart",
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    {
                        "id": "/World/VisiblePart",
                        "materials": {"material": "Steel"},
                    },
                ],
                "objects": {
                    "/World/OtherVisible": {
                        "materials": {"material": "Plastic"},
                    }
                },
                "items": [
                    {
                        "objects": {
                            "/World/DeepVisible": {
                                "materials": {"material": "Copper Brushed"},
                            }
                        }
                    }
                ],
            }
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == [
        "Copper Brushed",
        "Plastic",
        "Steel",
        FALLBACK_MATERIAL_NAME,
    ]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["/World/HiddenPart"]


def test_path_keyed_prediction_mapping_is_counted(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "/World/HiddenPart": "__UNKNOWN__",
                "/World/VisiblePart": "Steel",
                "/World/OtherVisible": {
                    "materials": {"material": "Plastic"},
                },
            }
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Plastic", "Steel", FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["unknown_material_prediction_ids"] == ["/World/HiddenPart"]


def test_material_retrieval_uses_fallback_when_called_directly() -> None:
    listener = MagicMock()
    context = {
        "unique_materials": ["__UNKNOWN__", "Steel", "Unknown"],
        "materials_mapping": {
            "Steel": "s3://bucket/materials/steel.mdl",
            "Unknown": "s3://bucket/materials/unknown.mdl",
        },
    }

    with patch(
        "material_agent.tasks.material_retrieval.get_listener", return_value=listener
    ):
        result = MaterialRetrievalTask().run(context)

    assert result["unique_materials"] == ["Steel", "Unknown", FALLBACK_MATERIAL_NAME]
    assert result["unknown_material_predictions"] == 1
    assert result["search_stats"]["total_queries"] == 3
    assert result["matched_materials"]["Steel"][0]["s3_path"].endswith("steel.mdl")
    assert result["matched_materials"]["Unknown"][0]["s3_path"].endswith("unknown.mdl")
    assert result["matched_materials"][FALLBACK_MATERIAL_NAME][0]["metadata"] == {
        "source": "materials_mapping"
    }


def test_unknown_in_auxiliary_fields_does_not_override_selected_material(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {
                "id": "/World/VisiblePart",
                "materials": {"material": "Steel"},
                "material_predictions": ["__UNKNOWN__"],
            }
        ],
    )

    result = IdentifyUniqueMaterialsTask().run(
        {"predictions_path": str(predictions_path)}
    )

    assert result["unique_materials"] == ["Steel"]
    assert result["unknown_material_predictions"] == 0
    assert result["unknown_material_prediction_ids"] == []


def test_selected_material_normalizes_unknown_sentinel_before_counting() -> None:
    task = IdentifyUniqueMaterialsTask()

    assert (
        task._selected_material_from_prediction(
            {"materials": {"material": "  __unknown__  "}}
        )
        == "__UNKNOWN__"
    )


def test_material_extraction_helper_edge_shapes() -> None:
    task = IdentifyUniqueMaterialsTask()

    assert task._extract_unknown_prediction_ids(
        {"materials": {"material": "__UNKNOWN__"}},
        fallback_id="fallback",
    ) == ["fallback"]
    assert task._prediction_contains_unknown_material("__UNKNOWN__") is True
    assert task._extract_materials_from_prediction({"materials": "Wood"}) == ["Wood"]
    assert task._extract_materials_from_prediction({"material": "Copper"}) == ["Copper"]
    assert task._extract_all_materials_from_prediction(42) == []
    assert task._extract_materials_from_field("steel") == ["steel"]
    assert task._extract_materials_from_field(
        ["steel", {"type": "plastic"}, {"ignored": 1}]
    ) == ["steel", "plastic"]
    assert task._extract_materials_from_field({"nested": ["copper", 5]}) == ["copper"]
