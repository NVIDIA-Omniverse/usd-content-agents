# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for validate_predictions LLM repair paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import material_agent.tasks.validate_predictions as validate_predictions
from material_agent.materials import FALLBACK_MATERIAL_NAME
from material_agent.tasks.validate_predictions import ValidatePredictionsTask

VALID_NAMES = [
    "Stainless Steel Polished",
    "Gold Polished",
    "Plastic White",
]


def _write_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


def _read_predictions(path: Path) -> list[dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_with_llm_config_records_repairs_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/a", "materials": {"material": "first-invalid"}},
            {"id": "/b", "materials": {"material": "second-invalid"}},
        ],
    )

    matches = iter(
        [
            ("Stainless Steel Polished", 0.6),
            ("Plastic White", 0.2),
        ]
    )
    monkeypatch.setattr(
        validate_predictions,
        "_best_fuzzy_match",
        lambda name, valid_names: next(matches),
    )
    monkeypatch.setattr(
        ValidatePredictionsTask,
        "_llm_repair",
        lambda self, items, valid_names, llm_config, listener: [
            (0, "first-invalid", "Stainless Steel Polished"),
            (1, "second-invalid", None),
        ],
    )

    result = ValidatePredictionsTask().run(
        {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
            "llm_config": {"backend": "mock"},
        }
    )

    stats = result["validation_stats"]
    assert stats["llm_repaired"] == 1
    assert stats["failed"] == 1

    reloaded = _read_predictions(predictions_path)
    assert reloaded[0]["materials"]["material"] == "Stainless Steel Polished"
    assert reloaded[1]["materials"]["material"] == FALLBACK_MATERIAL_NAME

    report = json.loads((tmp_path / "validate_report.json").read_text())
    assert report["llm_repaired"] == [
        {"index": 0, "old": "first-invalid", "new": "Stainless Steel Polished"}
    ]
    assert report["failed"] == [
        {
            "index": 1,
            "name": "second-invalid",
            "best_fuzzy": "Plastic White",
            "score": 0.2,
        }
    ]


def test_run_with_llm_config_uses_best_fuzzy_fallback_on_repair_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [{"id": "/a", "materials": {"material": "repairable"}}],
    )

    monkeypatch.setattr(
        validate_predictions,
        "_best_fuzzy_match",
        lambda name, valid_names: ("Plastic White", 0.5),
    )
    monkeypatch.setattr(
        ValidatePredictionsTask,
        "_llm_repair",
        lambda self, items, valid_names, llm_config, listener: [
            (0, "repairable", None),
        ],
    )

    result = ValidatePredictionsTask().run(
        {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
            "llm_config": {"backend": "mock"},
        }
    )

    assert result["validation_stats"]["llm_repaired"] == 1
    assert result["validation_stats"]["failed"] == 0
    assert _read_predictions(predictions_path)[0]["materials"]["material"] == (
        "Plastic White"
    )


def test_llm_repair_returning_fallback_counts_as_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [{"id": "/a", "materials": {"material": "invalid-material"}}],
    )

    monkeypatch.setattr(
        validate_predictions,
        "_best_fuzzy_match",
        lambda name, valid_names: ("", 0.1),
    )
    monkeypatch.setattr(
        ValidatePredictionsTask,
        "_llm_repair",
        lambda self, items, valid_names, llm_config, listener: [
            (0, "invalid-material", FALLBACK_MATERIAL_NAME),
        ],
    )

    result = ValidatePredictionsTask().run(
        {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
            "llm_config": {"backend": "mock"},
        }
    )

    stats = result["validation_stats"]
    assert stats["llm_repaired"] == 0
    assert stats["failed"] == 1
    assert stats["fallback_applied"] == 1

    report = json.loads((tmp_path / "validate_report.json").read_text())
    assert report["llm_repaired"] == []
    assert report["fallback_applied"] == [
        {
            "index": 0,
            "id": "/a",
            "old": "invalid-material",
            "new": FALLBACK_MATERIAL_NAME,
            "reason": "unrepaired_invalid_material",
        }
    ]


def test_run_without_llm_config_uses_fuzzy_fallback_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        predictions_path,
        [
            {"id": "/a", "materials": {"material": "repairable"}},
            {"id": "/b", "materials": {"material": "unrepairable"}},
        ],
    )

    matches = iter(
        [
            ("Plastic White", 0.5),
            ("", 0.1),
        ]
    )
    monkeypatch.setattr(
        validate_predictions,
        "_best_fuzzy_match",
        lambda name, valid_names: next(matches),
    )

    result = ValidatePredictionsTask().run(
        {
            "predictions_path": str(predictions_path),
            "material_names": VALID_NAMES,
        }
    )

    stats = result["validation_stats"]
    assert stats["llm_repaired"] == 1
    assert stats["failed"] == 1

    reloaded = _read_predictions(predictions_path)
    assert reloaded[0]["materials"]["material"] == "Plastic White"
    assert reloaded[1]["materials"]["material"] == FALLBACK_MATERIAL_NAME


def test_llm_repair_returns_none_when_chat_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.chat_models",
        SimpleNamespace(create_chat_model_from_config=lambda config: None),
    )
    listener = SimpleNamespace(info=Mock(), warning=Mock())

    result = ValidatePredictionsTask()._llm_repair(
        [(3, "bad-name", "best-guess")],
        VALID_NAMES,
        {"backend": "mock"},
        listener,
    )

    assert result == [(3, "bad-name", None)]
    listener.warning.assert_called_once()


def test_llm_repair_splits_batches_and_recovers_from_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.chat_models",
        SimpleNamespace(create_chat_model_from_config=lambda config: object()),
    )
    listener = SimpleNamespace(info=Mock(), warning=Mock())
    task = ValidatePredictionsTask()
    monkeypatch.setattr(task, "_LLM_BATCH_SIZE", 1)

    calls = {"count": 0}

    def fake_batch(
        items: list[tuple[int, str, str]],
        valid_names: list[str],
        llm: object,
        listener: object,
    ) -> list[tuple[int, str, str | None]]:
        calls["count"] += 1
        if calls["count"] == 1:
            return [(items[0][0], items[0][1], "Plastic White")]
        raise RuntimeError("batch failed")

    monkeypatch.setattr(task, "_llm_repair_batch", fake_batch)

    result = task._llm_repair(
        [(0, "first", "Plastic White"), (1, "second", "Gold Polished")],
        VALID_NAMES,
        {"backend": "mock"},
        listener,
    )

    assert sorted(result) == [
        (0, "first", "Plastic White"),
        (1, "second", None),
    ]
    listener.info.assert_called_once()


def test_llm_repair_single_batch_returns_batch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.functions.models.chat_models",
        SimpleNamespace(create_chat_model_from_config=lambda config: object()),
    )
    listener = SimpleNamespace(info=Mock(), warning=Mock())
    task = ValidatePredictionsTask()

    monkeypatch.setattr(
        task,
        "_llm_repair_batch",
        lambda batch, valid_names, llm, listener: [
            (batch[0][0], batch[0][1], "Plastic White")
        ],
    )

    result = task._llm_repair(
        [(0, "bad-name", "Plastic White")],
        VALID_NAMES,
        {"backend": "mock"},
        listener,
    )

    assert result == [(0, "bad-name", "Plastic White")]
    listener.info.assert_not_called()


def test_llm_repair_batch_success_and_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(
            HumanMessage=lambda content: SimpleNamespace(content=content),
            SystemMessage=lambda content: SimpleNamespace(content=content),
        ),
    )

    parsed = {"bad-name": "Plastic White"}
    monkeypatch.setitem(
        sys.modules,
        "world_understanding.utils.llm_parsing",
        SimpleNamespace(extract_json_from_llm_response=lambda content: parsed),
    )

    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="{}"))
    listener = SimpleNamespace(warning=Mock())
    task = ValidatePredictionsTask()

    result = task._llm_repair_batch(
        [(0, "bad-name", "Plastic White")],
        VALID_NAMES,
        llm,
        listener,
    )

    assert result == [(0, "bad-name", "Plastic White")]

    monkeypatch.setitem(
        sys.modules,
        "world_understanding.utils.llm_parsing",
        SimpleNamespace(extract_json_from_llm_response=lambda content: []),
    )
    result = task._llm_repair_batch(
        [(1, "still-bad", "Gold Polished")],
        VALID_NAMES,
        llm,
        listener,
    )

    assert result == [(1, "still-bad", None)]
    listener.warning.assert_called_once()


def test_llm_repair_batch_returns_none_results_on_invoke_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(
            HumanMessage=lambda content: SimpleNamespace(content=content),
            SystemMessage=lambda content: SimpleNamespace(content=content),
        ),
    )

    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("LLM unavailable")

    result = ValidatePredictionsTask()._llm_repair_batch(
        [(0, "bad-name", "Plastic White")],
        VALID_NAMES,
        FailingLLM(),
        SimpleNamespace(warning=Mock()),
    )

    assert result == [(0, "bad-name", None)]
